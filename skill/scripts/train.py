# Silex 铸塔：按策略 JSON 在合成数据集上拟合，导出链接 ONNX 蒸馏资产
# 用法: uv run python train.py --data data.csv --strategy strategy.json --out dist/
# data.csv 需含列 text,label（二分类 0/1；回归任务 label 为连续值）
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import onnx

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib import build_features, apply_stats, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    strategy = json.load(open(args.strategy, encoding="utf-8"))
    df = pd.read_csv(args.data)
    texts, y = df["text"].astype(str).tolist(), df["label"].astype(float).tolist()

    feat_cfg = strategy.get("features", {})
    tfidf_cfg, stats_ops = feat_cfg.get("tfidf"), feat_cfg.get("stats", [])
    task = strategy.get("task", "classification")

    tfidf = svd = dense = None
    if tfidf_cfg:
        tfidf = TfidfVectorizer(max_features=tfidf_cfg.get("max_features", 200))
        tfidf.fit(texts)
        n_comp = min(tfidf_cfg.get("svd", 16), max(len(tfidf.vocabulary_) - 1, 1),
                     max(len(set(texts)) - 1, 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        dense = svd.fit_transform(tfidf.transform(texts))

    feats = build_features(tfidf, svd, stats_ops, texts)

    algo = strategy.get("algo", "lightgbm")
    n_est = strategy.get("n_estimators", 100)
    if task == "classification":
        yl = np.array(y).astype(int)
        model = _lgb("LGBMClassifier")(n_estimators=n_est, verbose=-1, random_state=0).fit(feats, yl)
    else:
        model = _lgb("LGBMRegressor")(n_estimators=n_est, verbose=-1, random_state=0).fit(feats, y)

    # ---- 导出段 1：text -> dense（skl2onnx，StringTensor 入口）----
    embed_file = None
    if tfidf is not None:
        from sklearn.pipeline import Pipeline
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import StringTensorType
        onx = convert_sklearn(Pipeline([("v", tfidf), ("s", svd)]),
                              initial_types=[("text", StringTensorType([None, 1]))])
        embed_file = "embed.onnx"
        onnx.save(onx, f"{args.out}/{embed_file}")

    # ---- 导出段 2：feat -> proba（onnxmltools，非 hummingbird）----
    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    onx = convert_lightgbm(model, name="silex_l2",
                           initial_types=[("feat", FloatTensorType([None, feats.shape[1]]))],
                           zipmap=False)
    onnx.save(onx, f"{args.out}/model.onnx")
    model.booster_.save_model(f"{args.out}/lgbm_ref.txt")  # 参照件，供门禁比对数值一致性

    # ---- manifest：Rust 在线链路的装载契约 ----
    manifest = {
        "version": 1,
        "task": task,
        "algo": algo,
        "input": {"tensor": "text", "type": "string", "shape": ["batch", 1]},
        "embed_onnx": embed_file,
        "dense_dim": int(dense.shape[1]) if dense is not None else 0,
        "stats_ops": stats_ops,
        "feature_dim": int(feats.shape[1]),
        "model_onnx": "model.onnx",
        "score": "probabilities[:,1]",
        "threshold": strategy.get("threshold", 0.5),
        "channels": strategy.get("channels", []),  # 多任务通道由多次编译产出（v1 单通道）
    }
    write_json(f"{args.out}/manifest.json", manifest)
    print(json.dumps({"status": "compiled", "feature_dim": manifest["feature_dim"],
                      "embed": bool(embed_file), "out": args.out}))


def _lgb(name):
    import lightgbm
    return getattr(lightgbm, name)


if __name__ == "__main__":
    main()
