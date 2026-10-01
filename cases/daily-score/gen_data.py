# 生成 cases/daily-score 的合成数据集与人工核验真值集
# 真值用独立规则定义（比 LLM 标注更"挑剔"），用于演示门禁能暴露合成标注与业务真值的偏差
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

def make(pool, n):
    out = []
    for _ in range(n):
        s = rng.choice(pool)
        out.append(s)
    return out

# 合成标注（阶段2 LLM 打标产物）：较宽松，只看长度+关键词
good = ["今天完成了订单模块联调，测试通过", "修复了登录超时的问题并补充了单测",
        "整理需求文档，与产品对齐了字段定义", "排查线上内存泄漏，定位到缓存未失效",
        "配合前端完成接口对接，联调顺利"]
bad = ["随便看看", "摸鱼一天", "无", "还行吧", "今天啥也没干"]
syn = make(good, 200) + make(bad, 100) + make(good + bad, 50)  # 含难样本
syn_lbl = [1 if any(g in t for g in good) else 0 for t in syn]
pd.DataFrame({"text": syn, "label": syn_lbl}).to_csv("cases/daily-score/data.csv", index=False)

# 真值（人工规则更严：要求含"具体动作动词"，把部分合成正样本判为负）
strict_neg = ["配合前端完成接口对接，联调顺利", "还行吧"]
truth = make(good + bad, 15) + strict_neg * 5 + make(bad, 10)
truth_lbl = []
for t in truth:
    # 人工口径：必须是"完成/修复/排查/定位/整理"这类实质动作，且非水话
    ok = any(k in t for k in ["完成", "修复", "排查", "定位", "整理"]) and t not in strict_neg
    truth_lbl.append(1 if ok else 0)
pd.DataFrame({"text": truth, "label": truth_lbl}).to_csv("cases/daily-score/truth.csv", index=False)
print("data.csv", len(syn), "truth.csv", len(truth))
