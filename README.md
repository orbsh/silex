# Silex Compiler（燧石编译器）

## 命名

Silex，拉丁语"燧石/硅石"——沙子的核心成分，半导体与光导纤维的原料。命名承载两条隐喻，正好框定本系统的边界：

**燧石**是第一种被人类有意识加工的算法介质：打制石器是"编译"——从一块随机石料（需求）中打出标准化刃口（确定性资产），刃口一旦成形，切割不再依赖打制工匠在场。大模型就是那位工匠：只在打制期出现，不在使用期出现。

**硅石**是提纯链的隐喻：粗砂（石英砂，~98%）→ 冶金级硅（99%）→ 多晶硅（99.9999%）→ 电子级硅（99.9999999%）。每一级提纯去除的杂质，对应 L1 蒸馏中剔除的口语噪声、同义改写、情绪化表达；最终单晶（ONNX 资产）在纯净晶格中稳定导电——线上毫秒级确定性推理。沙子不值钱，电子级硅单晶按克计价：**价值不在原料，在提纯工艺**。LLM 按 Token 付费的"原料"极廉价，蒸馏后沉淀的资产才是复利所在。

## 理论位置：智力金字塔的编译执行器

本项目是 wiki [智力金字塔模型](https://github.com/orbsh/wiki/blob/main/intelligence-pyramid.md)（[EN](https://github.com/orbsh/wiki/blob/main/intelligence-pyramid-en.md)）中「编译流水线：L1 蒸馏至 L2」一节的可执行实现。引用关系：

| 金字塔论点 | Silex 落点 |
|---|---|
| L1 输出必须是编译后的制品，不作在线依赖 | `dist/` ONNX 资产包 = 制品；在线 Rust 链路零 LLM |
| 编译五级：路由→数据→特征→训练→导出 | `matrix.md`（ISA 查表）→ SKILL.md 阶段 1-5 → `train.py` |
| L2 输出须过 L3 硬约束闸门 | `manifest.json` 的 `threshold`：置信度经阈值/低置信带复核 |
| 飞轮：运行反馈回流 L1 | `low_confidence_band` 采样率 → 触发离线重编译 |
| 金字塔法则：能用底层绝不用上层 | L1 是熔炉不是工人；路由判据强制下沉 |

一句话：智力金字塔定义了 L1→L2 卸载的方向性法则，Silex 提供其工程实现——一个可执行的编译器，含 ISA（`matrix.md`）、编译流水线（SKILL.md 五阶段）、质量门禁（`gate.py`）、运行时契约（`manifest.json`）。

## 为什么现在成立：两条腿

单一任务（固定输入输出规范 + 可预测判定逻辑）在线本不需要 LLM——没有 LLM 的时代，工业界就是用传统 ML 和小型神经网络解决这类问题的。它一直没被批量重做，卡的不是算法，是两个成本：

- **架构腿（为什么 L2 就够）**：大模型的规模是为"一副参数同时装下尽可能多的任务"堆出来的。参数训练完成后就冻结不动，它吃下所有任务靠的是两件事：注意力让每个 token 的计算随上下文变（变的只是信息配比，不是参数本身）；专家路由（MoE）让不同输入只激活网络的一角。KPI 打分只有一项判定，那副多任务的容量全程闲置——没有 LLM 的时代，这类任务就是用小型神经网络/传统 ML 做的，容量刚好匹配。
- **经济腿（为什么以前贵、现在便宜）**：传统 ML 生命周期的真实门槛是算法工程师人力——清洗数据、手工特征工程、雇人肉眼打标。LLM 恰好同时具备两样东西：该领域的常识（知道什么是高质量日报，能设计特征）和接管繁琐劳动的能力（离线合成标注、生成训练代码）。门槛被抹平后，"向下编译"才第一次便宜到值得批量做。

所以本项目的 SKILL 只干一件事：教通晓一切的 LLM 在离线期替工程师跑通 L2 编译的完整生命周期（特征设计→合成打标→训练导出→双层门禁→飞轮回边），最终在线上彻底抹除自己的存在。

## 定位

接收业务需求 + 判定指标 → 端到端"降维编译"为本地微型 ML 资产（50MB 级 ONNX；算法选型偏好见 `matrix.md`「选型偏好」——总计算量差距不大时优先神经网络路线）。LLM 只在两处的离线时刻出现：

1. **阶段 1**：查 ISA 矩阵做路由判定（决定任务归哪层）
2. **阶段 2**：充当合成数据标注员（打标 + 同义改写扩增）

交付后，在线链路只有 Rust tokenizer → ONNX Runtime → 阈值闸门，零 Token 成本。

## 与 AI 原蓝图的三处修正（核实后落地）

1. **无 langgraph 状态机**：五阶段顺序管线 + 一条带错误日志的回边，普通脚本可表达；声明重型图框架违背 skillforge 引擎独立性（ADR-0006 已删外部引擎）。
2. **导出工具改正**：skl2onnx 只转 scikit-learn 管道；LightGBM/XGBoost 走 **onnxmltools**（`convert_lightgbm`，zipmap=False 出纯概率矩阵）。hummingbird-ml 已归档，不作首选。
3. **门禁拆双层**（核心修正）：合成集一致性（模型 ↔ 标注 LLM）+ 人工真值集（模型 ↔ 业务真值）。单层 accuracy 是循环论证——标注员和判卷人不能是同一个模型。`gate.py` 强制真值集 ≥20 条，缺失拒绝放行。

## 仓库结构

```
silex/
├── matrix.md              # ISA：场景-算法对照表 + 路由判据（L1 拆解前必读）
├── skill/
│   ├── SKILL.md           # Agent 技能：五阶段编译工作流
│   └── scripts/
│       ├── lib.py         # predict_dist 链接推理（Python 参照实现，Rust 重放规格）
│       ├── train.py       # 铸塔：拟合 + 导出 embed.onnx / model.onnx / manifest.json
│       └── gate.py        # 磐石门禁：双层校验
├── cases/                # 案例集：一案例一目录，持续追加（布局约定见 cases/README.md）
│   └── daily-score/      # 端到端实例（含门禁暴露偏差→飞轮回边的完整演示）
└── runtime/DESIGN.md      # Rust 在线塔基设计（代码下一里程碑）
```

## 快速验证

```bash
uv venv .venv && uv pip install -p .venv/bin/python scikit-learn lightgbm onnxmltools skl2onnx onnx onnxruntime
.venv/bin/python cases/daily-score/gen_data.py
.venv/bin/python skill/scripts/train.py --data cases/daily-score/data.csv \
    --strategy cases/daily-score/strategy.json --out cases/daily-score/dist/
.venv/bin/python skill/scripts/gate.py --dist cases/daily-score/dist \
    --synthetic cases/daily-score/data.csv --truth cases/daily-score/truth.csv --min 0.85
```

实例真实运行轨迹：宽松合成标注 → 一致性 1.0 / 真值 0.83 → 门禁拒绝（exit 1）；收紧标注口径重训 → 真值 1.0 → 放行（exit 0）。**门禁按设计暴露了"模型↔LLM 一致 ≠ 模型↔业务真值一致"**——这正是双层校验存在的理由。
