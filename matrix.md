# Silex ISA：场景-算法对照矩阵

编译器不许让 LLM 凭空猜算法。L1 拆解需求时必须查这张表，遵循刚性约束：**能用底层绝不用上层**。

## 路由判据（两个轴）

- **输入规范性**：文本能否用规则穷举？改写/变体多 → 排除 L3/L4
- **判定边界清晰度 + 输出尺度**：边界清晰、尺度固定 → 适合 L2；开放常识推理/生成 → 退回 L1

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
