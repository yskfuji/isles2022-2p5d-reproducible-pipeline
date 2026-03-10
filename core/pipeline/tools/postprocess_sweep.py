from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class SweepRow:
    out_dir: str
    min_size: int
    prob_filter: float
    top_k: int
    cc_score: str
    cc_score_thr: float
    best_threshold: float
    mean_dice: float | None
    median_dice: float | None
    voxel_precision: float | None
    voxel_recall: float | None
    detection_rate_case: float | None
    false_alarm_rate_case: float | None


def _csv_ints(s: str) -> list[int]:
    return [int(float(x)) for x in str(s).split(",") if x.strip()]


def _csv_floats(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if x.strip()]


def _slug(v: object) -> str:
    return str(v).replace(".", "p").replace("-", "m")


def main() -> int:
    ap = argparse.ArgumentParser(description="Grid-search postprocessing for evaluate_isles_25d")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--probs-dir", default="")
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--thresholds", default="0.70,0.75,0.80,0.85,0.90")
    ap.add_argument("--min-sizes", default="0,32,64")
    ap.add_argument("--prob-filters", default="0.0,0.90,0.96")
    ap.add_argument("--top-ks", default="0,1")
    ap.add_argument("--cc-scores", default="none,max_prob")
    ap.add_argument("--cc-score-thrs", default="0.5,0.7")
    ap.add_argument("--temperature", default="1.0")
    ap.add_argument("--normalize", default="fixed_nonzero_zscore")
    ap.add_argument("--stage1-probs-dir", default="")
    ap.add_argument("--tta", action="store_true", default=False)
    args = ap.parse_args()

    if bool(str(args.model_path).strip()) == bool(str(args.probs_dir).strip()):
        raise ValueError("Provide exactly one of --model-path or --probs-dir")

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[SweepRow] = []
    for ms in _csv_ints(args.min_sizes):
        for pf in _csv_floats(args.prob_filters):
            for tk in _csv_ints(args.top_ks):
                for cc in [x.strip() for x in str(args.cc_scores).split(",") if x.strip()]:
                    cc_thrs = [float(_csv_floats(args.cc_score_thrs)[0])] if cc == "none" else _csv_floats(args.cc_score_thrs)
                    for cct in cc_thrs:
                        name = f"ms{_slug(ms)}_pf{_slug(pf)}_tk{_slug(tk)}_cc{_slug(cc)}_cct{_slug(cct)}"
                        out_dir = out_root / name
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
                            args.thresholds,
                            "--min-size",
                            str(ms),
                            "--prob-filter",
                            str(pf),
                            "--cc-score",
                            cc,
                            "--cc-score-thr",
                            str(cct),
                            "--top-k",
                            str(tk),
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
                        rows.append(
                            SweepRow(
                                out_dir=str(out_dir),
                                min_size=int(ms),
                                prob_filter=float(pf),
                                top_k=int(tk),
                                cc_score=str(cc),
                                cc_score_thr=float(cct),
                                best_threshold=float(best["threshold"]),
                                mean_dice=best.get("mean_dice"),
                                median_dice=best.get("median_dice"),
                                voxel_precision=best.get("voxel_precision"),
                                voxel_recall=best.get("voxel_recall"),
                                detection_rate_case=best.get("detection_rate_case"),
                                false_alarm_rate_case=best.get("false_alarm_rate_case"),
                            )
                        )

    rows_sorted = sorted(rows, key=lambda r: float(r.mean_dice or -1.0), reverse=True)
    payload = [asdict(r) for r in rows_sorted]
    (out_root / "postprocess_sweep_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload[:10], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
