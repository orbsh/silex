# Silex 共享库：蒸馏资产的链接推理（训练期参照与门禁共用同一实现，保证与 Rust 侧语义一致）
import json
import numpy as np

# 支持的 L4/L3 统计算子（简单确定性操作，Rust 侧按同名实现直接执行，不进入 ONNX 图）
#   char_len            -> 字符数
#   char_count:<c>      -> 字符 c 出现次数 / 字符总数（密度，避免长度共线）
#   keyword_hit:<kw>    -> 关键词是否出现（0/1）

def apply_stats(ops, texts):
    cols = []
    for op in ops:
        if op == "char_len":
            cols.append([float(len(t)) for t in texts])
        elif op.startswith("char_count:"):
            c = op.split(":", 1)[1]
            cols.append([t.count(c) / max(len(t), 1) for t in texts])
        elif op.startswith("keyword_hit:"):
            k = op.split(":", 1)[1]
            cols.append([float(k in t) for t in texts])
        else:
            raise ValueError(f"未知统计算子: {op}")
    return np.column_stack(cols).astype(np.float32) if cols else np.zeros((len(texts), 0), np.float32)


def build_features(tfidf, svd, stats_ops, texts):
    """离线训练与在线链接共用的特征装配：[dense | stats]"""
    parts = []
    if tfidf is not None:
        parts.append(svd.transform(tfidf.transform(texts)).astype(np.float32))
    parts.append(apply_stats(stats_ops, texts))
    return np.hstack(parts).astype(np.float32)


def load_manifest(dist_dir):
    with open(f"{dist_dir}/manifest.json", encoding="utf-8") as f:
        return json.load(f)


def predict_dist(dist_dir, texts):
    """链接推理：embed.onnx（可选）→ 拼接统计特征 → model.onnx → 正类概率。

    这是 Rust 在线运行时的 Python 参照实现：Rust 侧按 manifest 重放同一链条。
    """
    import onnxruntime as rt

    manifest = load_manifest(dist_dir)
    ops = manifest["stats_ops"]
    x = np.asarray(texts, dtype=object).reshape(-1, 1)  # StringTensor 要求 [batch, 1]

    if manifest.get("embed_onnx"):
        sess = rt.InferenceSession(f"{dist_dir}/{manifest['embed_onnx']}",
                                   providers=["CPUExecutionProvider"])
        dense = np.array(sess.run(None, {sess.get_inputs()[0].name: x})[0], dtype=np.float32)
        feats = np.hstack([dense, apply_stats(ops, texts)]).astype(np.float32)
    else:
        feats = apply_stats(ops, texts)

    sess = rt.InferenceSession(f"{dist_dir}/model.onnx", providers=["CPUExecutionProvider"])
    # 只请求概率输出（zipmap=False 的 label 声明 batch=1，全量请求会触发 ort 维度警告）
    # 回归模型输出名为 label，需回退到请求全部
    want = [o.name for o in sess.get_outputs() if "label" not in o.name.lower()]
    outs = sess.run(want or None, {sess.get_inputs()[0].name: feats})
    prob = None
    for o in outs:
        arr = np.array(o)
        if arr.ndim == 2 and arr.shape[1] == 2:
            prob = arr[:, 1]
    if prob is None:  # 回归路径：单一数值输出
        prob = np.array(outs[-1]).ravel().astype(np.float32)
    return prob


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
