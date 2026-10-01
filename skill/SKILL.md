---
name: silex-compiler
description: 将在线 LLM 任务离线蒸馏为轻量本地 ML 资产（L2/L3/L4）的端到端编译器。当业务需求判定边界清晰、在线并发高且要求零 Token 成本时调用。
---

# Silex 编译器（聚沙成塔）

L1（本 agent）是熔炼工匠：直觉拆解需求 → 查 ISA 矩阵 → 合成标注数据 → 调用脚本训练/导出/门禁 → 交付 ONNX 蒸馏资产。全程离线，产物在线运行不再依赖任何 LLM。

ISA 矩阵与路由判据见仓库根目录 `matrix.md`，**拆解前必读**。

## 适用场景

用户诉求中出现以下信号组合时启用本技能（理论依据：wiki `intelligence-pyramid.md`「编译流水线」）：

- **高频在线**：调用量以千/万计（评论过滤、日报打分、意图分流、风控初筛），每次调 LLM 的 Token 费和排队延迟不可接受
- **边界清晰、尺度固定**：判定结果是二分类或有限分值（0/2/5、通过/拒绝、20 个固定处理组之一），而非开放式推理或生成
- **变体杀规则**：口语化改写、同义词、加空格/同音字绕过——确定性正则穷举不完备，这正是 L2 稠密向量的主场

**不适用（阶段 1 必须拒绝并说明理由）**：

- 开放常识推理/生成类（"帮我写诗""总结这篇文章"）→ 任务本就属于 L1，无可卸载
- 规则可穷举且追求可解释（"校验邮箱格式""提取身份证号"）→ 直接交付 L4 正则，触发训练就是浪费
- 判定边界本身就模糊多变、每周都在改口径 → 蒸馏资产维护成本高于收益，留在 L1
- 无真实业务日志且 LLM 合成数据分布无法覆盖线上分布 → 冷启动不成立

## 任务记忆（多步任务必读）

本技能是多步编译任务，必须用任务记忆工具显式跟踪，禁止摊在会话上下文里。

**Step 0 · 建任务**：收到编译请求先调用：

```
memento_create(
    skill_name  = "silex-compiler",
    content     = "阶段1：拆解需求并查表定策略",
    description = "Silex 编译: <业务一句话>"
)
```

每完成一个阶段 `memento_update(task_id, item_id, done=true, result=<产物路径或关键指标>)`；新阶段用 `memento_create(task_id, ...)` 追加。所有任务项完成前，gate 会硬拦截脚本执行。

## 五阶段工作流

**阶段 1 · 查表路由（L1 直觉，agent 完成）**
读 `matrix.md`，对需求沿两轴判定：输入规范性、判定边界+输出尺度。不满足 L2 条件（开放式推理/生成）→ 直接回复"本任务应留在 L1，不编译"。满足 → 按 `matrix.md`「选型偏好」定 `algo`（总计算量差距不大时优先神经网络路线），写出策略 JSON 存为 `strategy.json`：

```json
{
  "task": "classification",            // 或 regression
  "algo": "lightgbm",
  "n_estimators": 100,
  "threshold": 0.5,                     // classification 用
  "features": {
    "tfidf": {"max_features": 200, "svd": 16},
    "stats": ["char_len", "char_count:!", "keyword_hit:退款"]
  }
}
```

**阶段 2 · 流沙熔炼（L1 打标，agent 完成）**
按策略合成 ≥300 条 `text,label` 样本（CSV，含难负样本变体），存 `data.csv`。合成时明确提示词中的标签定义，保证边界一致。

**阶段 3 · 铸塔（确定性脚本）**
```bash
uv run python skill/scripts/train.py --data data.csv --strategy strategy.json --out dist/
```
产出 `dist/model.onnx`（+可选 `dist/embed.onnx`、`manifest.json`）。

**阶段 4 · 磐石门禁（双层，脚本强制真值集）**
准备 ≥20 条**人工核验**样本 `truth.csv`（向用户索要或从其真实业务日志中人工标——合成集只度量模型与标注 LLM 的一致性，度量不了业务真值）。
```bash
uv run python skill/scripts/gate.py --dist dist/ --synthetic data.csv --truth truth.csv --min 0.9
```
exit 0 放行；exit 1/2 未过 → 阶段 5。

**阶段 5 · 飞轮回边（最多 3 轮）**
依据门禁输出修正：合成一致性低 → 特征不足，回阶段 1 加 `stats` 算子或调 svd；真值低但一致性高 → 标注 prompt 边界与人工真值有偏差，重写阶段 2；报错 → 看 traceback 修策略。≥3 轮未过 → 熔断上报，建议该任务留在 L1。

## 交付物

`dist/` 包：`model.onnx` + `embed.onnx`（若有）+ `manifest.json`（Rust 在线装载契约：张量名/shape/统计算子/阈值）。向用户报告门禁两层指标与 `low_confidence_band`（线上采样回流率预估）。
