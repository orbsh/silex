# Silex ISA：场景-算法对照矩阵

编译器不许让 LLM 凭空猜算法。L1 拆解需求时必须查这张表，遵循刚性约束：**能用底层绝不用上层**。

## 路由判据（两个轴）

- **输入规范性**：文本能否用规则穷举？改写/变体多 → 排除 L3/L4
- **判定边界清晰度 + 输出尺度**：边界清晰、尺度固定 → 适合 L2；开放常识推理/生成 → 退回 L1

## 选型偏好：优先选择矩阵化的计算形态

**根原则**：计算形态越接近稠密、规整、无分支的矩阵乘，现代硬件（CPU SIMD/AVX-512/AMX、GPU Tensor Core）利用率越高。依据见 wiki [树模型 vs 小型神经网络](https://github.com/orbsh/wiki/blob/main/lambda-to-silicon.md)：随机分支触发分支预测失败、指针追逐打断硬件预取——树模型的硬件劣势源于计算形态，不是精度或实现。

**硬件友好度梯度**（单核在线推理，只到序不构成倍数承诺）：

- 稠密矩阵乘：微型 Transformer / CNN / MLP——无分支流式指令，权重常驻 L1/L2 时吃满向量化
- 向量化线性：逻辑回归、固定维余弦相似度——单层乘加 + FMA
- 树遍历：XGBoost / LightGBM——逐树逐节点比较 + 条件跳转，仅靠运行时 SIMD 遍历内核缓解
- 字节扫描（L4）：正则/统计——非数值计算，纳秒~微秒级，不适用本梯度

**推论**：总计算量差距不大时，优先选择小型神经网络，不默认树模型。树模型的推理优势只来自一类场景：高基数离散类别特征免维度膨胀、且 batch=1 实时推理时向量收益塌缩——两个条件同时成立才选树。本流水线的默认特征形态是 L3 已把输入稠密化（TF-IDF→SVD），第一个条件通常不成立。

**ONNX 化 ≠ 矩阵化**：把模型导出为 ONNX 不改变其计算形态。onnxmltools 导出的决策树是 ONNX `TreeEnsembleClassifier` 节点，运行时（ort/tract）执行逐树遍历内核——树还是树。真正把树编译成张量矩阵乘（tree→GEMM）的是 Hummingbird，已归档不作首选；若将来需要矩阵化树，路径是自研 GEMM 遍历内核进 Rust 运行时，属提级项非现状。ONNX 化的真实目的：资产可移植 + 单运行时装载（一个引擎吃 L3 图 + L2 图），硬件友好度由运行时内核决定，不由导出格式自动获得。

落地状态：v1 资产路线（`train.py` 的 `algo` 默认 `lightgbm`）是导出工具链先行验证的产物，不构成选型先例；MLP 经 skl2onnx 直转 ONNX 即可切换，微型 BERT 路线未实现。

## 矩阵

| 业务场景（流沙输入） | L1 语义拆解定位 | 编译目标资产 | 卸载效果 |
|---|---|---|---|
| 垃圾/作弊文本过滤 | 口语变体、暗号、变体作弊；边界清晰，二分类 | L4 统计算子 + L3 TF-IDF→SVD 稠密向量 + L2 LightGBM | 全本地毫秒级，零 Token |
| 口语化日报质量评分 | 尺度固定（0-5），变体无法穷举 | L4 统计特征 + L2 模型（v1 单通道；多通道=多次编译）；长难语义可换微型 BERT 路线（未实现） | 在线 LLM 全卸载 |
| 动态定价 | 宏观策略意图（清仓/提价） | L3：分段函数+惩罚因子，LLM 离线调参 | 财务链路零随机性 |
| 客服意图路由 | 提问口吻千奇百怪，目标 20 个固定处理组 | L4 实体提取 + L3 TF-IDF + 余弦相似度 | 微秒级分流 |

## 层级-产物-运行时映射（Rust 在线链路）

| 层级 | 离线编译产物 | 在线装载 | 单次耗时 |
|---|---|---|---|
| L4 机械过滤 | 正则/黑名单/统计算子（manifest 的 `stats_ops`） | Rust `regex`（DFA，O(n) 无回溯） | < 10 µs |
| L3 特征变换 | `embed.onnx`（TF-IDF→SVD 图）或公式参数 | `tokenizers` crate / 纯 Rust 算子 | < 50 µs |
| L2 模型 | `model.onnx`（onnxmltools 导出，非 hummingbird） | `ort`（=2.0.0-rc.13）/ `tract` | 1~3 ms |

## 运行时采样回流（飞轮的触发端）

在线推理落在低置信带（距阈值 ±0.15）的样本 → 异步采样入队 → 唤醒离线重编译（LLM 对新样本打标 → train → gate → 热替换 `model.onnx`）。门禁的 `low_confidence_band` 字段度量该采样率。

## 工具事实（已核实，2026-10）

- skl2onnx 只转 scikit-learn 管道；LightGBM/XGBoost 走 onnxmltools（`convert_lightgbm`，`zipmap=False` 出纯概率矩阵）。hummingbird-ml 已归档，不作首选。
- StringTensor 入口形状必须是 `[batch, 1]`（一列文本），`[1, batch]` 会报维度错。
- 链接推理（embed.onnx → 拼接 → model.onnx）的 Python 参照在 `skill/scripts/lib.py::predict_dist`，Rust 侧按 manifest 重放同一链条。
