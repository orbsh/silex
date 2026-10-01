# Silex 磐石门禁：双层校验蒸馏资产
# 用法: uv run python gate.py --dist dist/ --synthetic data.csv --truth truth.csv [--metric accuracy] [--min 0.9]
# truth.csv 必须是人工核验样本（数十条即可）——合成集只度量模型与标注 LLM 的一致性，
# 真值差距只能由人工核验集度量。缺 truth 文件时门禁拒绝放行。
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib import load_manifest, predict_dist  # noqa: E402


def score_of(df, dist_dir, manifest):
    p = predict_dist(dist_dir, df["text"].astype(str).tolist())
    if manifest["task"] == "classification":
        thr = manifest.get("threshold", 0.5)
        return (p >= thr).astype(int), p
    return p, p  # 回归：直接输出分值


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True)
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--min", type=float, default=0.9)
    args = ap.parse_args()

    manifest = load_manifest(args.dist)
    task = manifest["task"]
    result = {"task": task, "layers": {}}

    # ---- 层 1：合成集一致性（模型 ↔ 标注 LLM）----
    syn = pd.read_csv(args.synthetic)
    pred, _ = score_of(syn, args.dist, manifest)
    ysyn = syn["label"].to_numpy()
    if task == "classification":
        consist = float((pred == ysyn).mean())
    else:
        consist = float(1.0 - np.abs(pred - ysyn).mean() / max(np.ptp(ysyn), 1e-6))
    result["layers"]["synthetic_consistency"] = {"value": consist}

    # ---- 层 2：人工核验真值集（模型 ↔ 业务真值）----
    truth = pd.read_csv(args.truth)
    if len(truth) < 20:
        result["pass"] = False
        result["error"] = f"真值集仅 {len(truth)} 条（<20），门禁拒绝：需人工核验样本"
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(2)
    predt, probt = score_of(truth, args.dist, manifest)
    ytruth = truth["label"].to_numpy()
    if task == "classification":
        truth_acc = float((predt == ytruth).mean())
        # 低置信带：距阈值 0.15 以内的样本占比 = 线上应采样回 L1 的比例
        band = float((np.abs(probt - manifest.get("threshold", 0.5)) <= 0.15).mean())
        result["layers"]["truth"] = {"accuracy": truth_acc, "low_confidence_band": band}
        passed = truth_acc >= args.min
    else:
        mae = float(np.abs(predt - ytruth).mean())
        rng = max(float(np.ptp(ytruth)), 1e-6)
        result["layers"]["truth"] = {"mae": mae, "normalized": 1.0 - mae / rng}
        passed = mae / rng <= (1.0 - args.min)

    result["pass"] = bool(passed)
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
