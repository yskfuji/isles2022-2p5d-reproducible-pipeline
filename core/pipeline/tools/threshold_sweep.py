from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _grid(step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    xs = [round(i * step, 10) for i in range(1, int(round(1.0 / step)))]
    xs = [float(x) for x in xs if 0.0 < x < 1.0]
    if 0.5 not in xs:
        xs.append(0.5)
    return sorted(set(xs))


def main() -> int:
    ap = argparse.ArgumentParser(description="Threshold sweep wrapper for evaluate_isles_25d")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--probs-dir", default="")
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--min-size", type=int, default=64)
    ap.add_argument("--prob-filter", type=float, default=0.96)
    ap.add_argument("--cc-score", default="none")
    ap.add_argument("--cc-score-thr", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--temperature", default="1.0")
    ap.add_argument("--normalize", default="fixed_nonzero_zscore")
    ap.add_argument("--stage1-probs-dir", default="")
    ap.add_argument("--tta", action="store_true", default=False)
    args = ap.parse_args()

    if bool(str(args.model_path).strip()) == bool(str(args.probs_dir).strip()):
        raise ValueError("Provide exactly one of --model-path or --probs-dir")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = ",".join(str(x) for x in _grid(float(args.step)))

    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.evaluate_isles_25d",
        "--csv-path",
        args.csv_path,
        "--root",
        args.root,
        "--split",
        args.split,
        "--out-dir",
        str(out_dir),
        "--normalize",
        args.normalize,
        "--thresholds",
        thresholds,
        "--min-size",
        str(int(args.min_size)),
        "--prob-filter",
        str(float(args.prob_filter)),
        "--cc-score",
        str(args.cc_score),
        "--cc-score-thr",
        str(float(args.cc_score_thr)),
        "--top-k",
        str(int(args.top_k)),
        "--temperature",
        str(args.temperature),
    ]
    if str(args.model_path).strip():
        cmd += ["--model-path", args.model_path]
    else:
        cmd += ["--probs-dir", args.probs_dir]
    if str(args.stage1_probs_dir).strip():
        cmd += ["--stage1-probs-dir", args.stage1_probs_dir]
    if bool(args.tta):
        cmd.append("--tta")

    subprocess.run(cmd, check=True)

    summary = json.loads((out_dir / "summary.json").read_text())
    best = max(summary.get("per_threshold") or [], key=lambda r: float(r.get("mean_dice") or -1.0))
    payload = {
        "best_threshold": float(best["threshold"]),
        "best_mean_dice": float(best.get("mean_dice") or 0.0),
        "summary_path": str(out_dir / "summary.json"),
    }
    (out_dir / "best_threshold.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
