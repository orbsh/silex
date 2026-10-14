# 预训练编码器路线（重点方案：ModernBERT）

**状态：规格定案，脚本未实现。** `train.py` 仍只走 TF-IDF→SVD + LightGBM/树模型；本文件是流程文档，不是代码先例。共享脚本的改动按 `cases/README.md` 的规则，由首个编码器案例的 NOTES.md 驱动。

现行 L2/L3 链（TF-IDF→SVD 稠密向量 + stats 算子 + LightGBM）在同一条业务上更小、更快、更便宜。编码器只在下面几条同时成立时才划得来：

- 判定依赖长文或跨段语义（词频统计近似不出边界）
- 在线延迟预算相对现行 L2 放宽一个数量级（现行 L2 毫秒级 → 编码器十毫秒级）
- 资产预算相对现行资产放宽一个数量级（现行 50MB 级 → int8 编码器百 MB 级）
- 铸塔侧有 GPU，或接受纯 CPU 的「冻结编码器 + 线性头」形态

判据是量级关系，不是绝对阈值：现行基线变了，跟着变的是两边各自的数字，不变量是「高一个数量级」。本节及后文的毫秒 / MB 数字都是当前基线下的一次测量参照（见 §4），不作验收线。

只满足「长文」一条时，走多通道编译（同一业务多次编译，长文单独一条编码器通道），不整条替换快路径。

## 1. backbone 选择

判据是语料语言构成，不是榜单排名。

| 语料 | 候选 | 参数量 / 上下文 | 核实状态 |
|---|---|---|---|
| 英文 + 代码 | `answerdotai/ModernBERT-base` / `ModernBERT-large` | 149M/22 层、395M/28 层；8192 | 模型卡与论文核实（Apache 2.0） |
| 中文 | `feynmanzhao/chinese-modernbert-large-wwm`（论文 2510.12285 同作者） | 0.4B；8192 | 模型卡核实；CLUE 上除 WSC 明显偏低外接近 RoBERTa-wwm-large |
| 中文 | `qihoo360/Zhinao-ChineseModernBert` | Base 级（除 Embedding 外约 100M）；8192 | 模型卡声明 CLUE 综合超 RoBERTa-wwm-large，未逐项核实 |
| 中文兜底 | 中文 RoBERTa-wwm | — | 现成方案，无长上下文优势 |
| 多语 | `jhu-clsp/mmBERT`（small/base/large） | 1800+ 语言、3T+ tokens，ModernBERT 架构 | 发布博客核实；各尺寸参数量未逐项核实 |

原版 ModernBERT 只吃英文 + 代码，中文会碎成字节级 token，不要直接用在中文任务上。

## 2. 链条定义

离线（阶段 2-4 变体）：

```
合成标注（LLM，含难负样本）→ 微调 或 冻结嵌入 + 线性头 → 导出 ONNX → int8 动态量化 → 双层门禁 → manifest.json 扩展字段
```

在线（与 `runtime/DESIGN.md`「编码器路线」一节对应）：

```
tokenizer（Rust tokenizers crate，tokenizer.json 随资产交付）→ encoder.onnx → 池化（mean/cls）→（可选）拼接 stats_ops → 头/阈值
```

两种装载形态：

- **形 A · 分类头在图内**：单 session，输出 `probabilities[:,1]`，`manifest.score` / `threshold` 语义不变。
- **形 B · 冻结编码器当 L3**：`encoder.onnx` 占现在 `embed.onnx` 的位置，输出池化后的嵌入；L2 头与门禁完全不动，`[dense | stats]` 拼接保留。迁移成本最低，优先。

## 3. 训练与微调

### 3.1 数据量级

编码器吃的数据比 TF-IDF + GBDT 高一个量级：现行阶段 2 的 ≥300 条合成集对 LightGBM 够用，对 149M 编码器偏少——少样本下种子方差大，且容量足够把标注口径整段记死。量级参考：几千～几万条。合成集与人工真值集必须分开（同 `gate.py` 的判据）。

### 3.2 多阶段迁移

论文里 RTE / MRPC / STS-B 都从 MNLI checkpoint 起训。AnswerDotAI issue #225 里照官方 notebook 单阶段从 base 起训 RTE，5 次 run 只到 57%~66%（论文 87.4%）。**目标任务小众或数据少时，先领域续训（MLM）或从相近任务 checkpoint 起步，别指望单阶段一把过。** 对外报指标前先确认数字的来源前提。

### 3.3 超参（三组有出处的配方）

| 来源 | batch | lr | epochs | 调度 / 其他 |
|---|---|---|---|---|
| Answer.AI 官方 GLUE notebook | 32 / 32 | 8e-5 | 2 | linear、adamw_torch、betas (0.9, 0.98)、eps 1e-6、wd 8e-6、bf16、DataCollatorWithPadding、装 flash-attn |
| philschmid 分类指南（2025） | 32 | 5e-5 | 5 | bf16、max_length 512（短文本不必上 8192） |
| Chinese ModernBERT CLUE 协议 | — | peak 3e-5 | 3 | linear decay |

共同区间：**lr 5e-5~8e-5、2~3 epochs、linear 衰减、bf16**（不用 fp16，社区有 NaN / 不稳定报告）。官方 notebook 的原话是「ModernBERT trains better with FlashAttention」——训练侧装上 flash-attn。

预训练 / 领域续训侧（模型卡记录的预训练配方）：StableAdamW + trapezoidal LR schedule + 1-sqrt decay。

### 3.4 池化与已知坑

- HF 实现默认 **mean pooling**，config 可切 `classifier_pooling='cls'`。旧版 transformers 上 cls + 分类头会把训练打成随机猜（transformers #38720，2025-07 修复；当时规避办法是切 mean）→ 钉死 transformers 版本，并把这条记入案例问题账。
- `torch.compile` 与 ModernBERT 冲突，需 `compile=False`（AnswerDotAI #190）。
- 原生 8192 不是从零训的：先在 1024 上下文预训练，再把 global 层 RoPE theta 从 1e4 提到 1.6e5、追加约 250B tokens 扩出来（论文）。短文本分类用不到 8192，长文才值这个价。

## 4. 导出与量化

- 官方仓库自带 ONNX，不用自己从头导：`onnx/{model, model_fp16, model_int8, model_uint8, model_q4, model_q4f16, model_bnb4}.onnx`（HF 文件清单核实）。
- 导出必须关 flash attention（回 eager/sdpa），代价是丢 unpadding。但 unpadding 是「批次内不等长」的批处理优化，现行在线形态是 batch=1 单请求，不受影响；吃它好处的是训练吞吐。**这条不是阻断项。**
- 量级锚点（第三方模型卡，132M 日语模型，CPU-only ONNX Runtime，batch=1，序列长度未标明）：PyTorch fp32 58.4ms → ONNX fp32 24.4ms（2.4x，质量无损）→ ONNX int8 动态量化 11.0ms（5.3x）；文件 505MB → 127MB；cos 0.990 / 排序 Spearman 0.933。**只作数量级参考，不是同构测量。**
- 量化漂移（约 1~2%）要在业务真值集上判，不是看 embedding 余弦。

## 5. 与现有资产的组合形态

推荐形 B：**冻结编码器 + `[dense | stats]` 拼接 + 线性头**。

- 编码器只当 L3 特征提取器（离线跑一遍取嵌入，或在线跑 `encoder.onnx`）。
- `stats_ops`（`char_len` / `char_count:<c>` / `keyword_hit:<kw>`）保留为纯 Rust 旁路，确定性特征不被吞进参数里。
- L2 头仍是现在这套模型与门禁，迁移成本最低；对「规则覆盖一半、变体覆盖另一半」的混合任务，精度与可解释性都有保障。
- 全量微调（形 A）在这类任务上未必赢，且失去可重算、可回滚的性质。

## 6. 门禁与飞轮适配

- **层 1（合成集一致性）在编码器上会被顶满**：容量足够背下标注口径，判别力比在 LightGBM 上更弱。层 2（人工真值）是唯一有信息量的度量——`cases/daily-score` 的 1.0 / 0.83 轨迹在编码器上只会更极端。
- **多 seed**：编码器微调种子方差大（#225 的 5 次 run 是 57%~66%），门禁应报均值 ± 方差；checkpoint / epoch 的选择只用合成 val，真值集只判不选。
- **低置信带（±0.15）**语义不变，但编码器概率普遍过自信 → band 更适合当「标定 / 重采样」的度量（温度标定 vs 直接重训另议）。
- **热替换**：int8 编码器的资产比现行链高一个数量级，飞轮触发与整包替换的频率要相应收敛。

## 7. 未决问题（待首个编码器案例驱动）

- 中文 backbone 钉哪个（Chinese ModernBERT large / qihoo360 base 级 / mmBERT），是否按语料语言在多通道里分开编译。
- 冻结嵌入形态要不要预计算缓存（离线取嵌入 → 线性头），缓存失效与重编译的触发条件。
- 池化选择（mean vs cls）是否固化成策略字段。
- int8 漂移在业务真值集上的可接受阈值。
- 微调在哪台机器跑、要不要 GPU：零 GPU 约束下只有形 B 可行，全量微调要 GPU 或数小时 CPU。
- 首个案例目录（`cases/<name>/`）什么时候开：按 `cases/README.md`，脚本改动必须由案例 NOTES 的缺陷驱动，编码器路线同样等这个案例。

## 参考来源

- 论文 *Smarter, Better, Faster, Longer*（arXiv 2412.13663）：架构、8192 扩展方式、预训练配方、GLUE 成绩。
- `answerdotai/ModernBERT-base` 模型卡（含 ONNX 文件清单、许可、评测量表）。
- Answer.AI 发布博客（2024-12-19）与官方 GLUE 微调 notebook（超参、flash-attn 建议）。
- philschmid《Fine-tune classifier with ModernBERT in 2025》（通用分类超参）。
- `feynmanzhao/chinese-modernbert-large-wwm` 模型卡与 arXiv 2510.12285（中文 backbone、CLUE 协议）。
- HuggingFace blog《mmBERT: ModernBERT goes Multilingual》（多语 backbone）。
- transformers #38720 / #35545、AnswerDotAI #190 / #225（池化 bug、compile、ONNX 导出、复现前提）。
- `monju-lab/modernbert-ja-130m-int8-onnx` 模型卡（CPU 延迟与量化锚点，第三方单点测量）。
- ACL/IJCNLP 2025《ModernBERT or DeBERTaV3?》（控制变量对照：ModernBERT 的确定优势是长上下文与速度，样本效率上上一代 DeBERTaV3 系仍强）。