#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text())


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _best_threshold(summary: dict[str, Any]) -> float:
    best = max(summary["per_threshold"], key=lambda r: float(r.get("mean_dice") or 0.0))
    return float(best["threshold"])


def _extract_metric(row: dict[str, Any], thr: float, key: str) -> Any:
    specific = row.get(f"{key}@{thr:g}")
    if specific is not None:
        return specific
    return row.get(key)


def _load_run(run_dir: Path) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    summary = _read_json(run_dir / "summary.json")
    rows_raw = _read_json(run_dir / "metrics.json")
    thr = _best_threshold(summary)
    rows = []
    for row in rows_raw:
        rows.append(
            {
                "case_id": row.get("case_id"),
                "dice": _safe_float(_extract_metric(row, thr, "dice")),
                "detected": _extract_metric(row, thr, "detected"),
                "gt_vox": _safe_float(row.get("gt_vox")),
                "pred_vox": _safe_float(_extract_metric(row, thr, "pred_vox")),
                "tp_vox": _safe_float(_extract_metric(row, thr, "tp_vox")),
                "fp_vox": _safe_float(_extract_metric(row, thr, "fp_vox")),
                "fn_vox": _safe_float(_extract_metric(row, thr, "fn_vox")),
                "fp_cc": _safe_float(_extract_metric(row, thr, "fp_cc")),
                "slice_spacing_bucket": row.get("slice_spacing_bucket"),
                "gt_size_bucket": row.get("gt_size_bucket"),
            }
        )
    return thr, summary, rows


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def _p(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = int(round((len(xs) - 1) * q))
    return float(xs[i])


@dataclass
class DiffRow:
    case_id: str
    dice_a: float
    dice_b: float
    delta: float
    gt_vox: float | None
    group: str
    size: str


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two eval runs and summarize failure modes.")
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", default="")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    run_a = Path(args.run_a).expanduser().resolve()
    run_b = Path(args.run_b).expanduser().resolve()
    thr_a, summary_a, rows_a = _load_run(run_a)
    thr_b, summary_b, rows_b = _load_run(run_b)

    a_by = {r["case_id"]: r for r in rows_a if r.get("case_id")}
    b_by = {r["case_id"]: r for r in rows_b if r.get("case_id")}
    common = sorted(set(a_by).intersection(set(b_by)))

    diffs: list[DiffRow] = []
    for cid in common:
        ra, rb = a_by[cid], b_by[cid]
        da = float(ra.get("dice") or 0.0)
        db = float(rb.get("dice") or 0.0)
        diffs.append(
            DiffRow(
                case_id=cid,
                dice_a=da,
                dice_b=db,
                delta=db - da,
                gt_vox=_safe_float(ra.get("gt_vox")),
                group=str(ra.get("slice_spacing_bucket") or rb.get("slice_spacing_bucket") or "unknown"),
                size=str(ra.get("gt_size_bucket") or rb.get("gt_size_bucket") or "unknown"),
            )
        )

    diffs_sorted = sorted(diffs, key=lambda r: r.delta)
    worst = diffs_sorted[: max(0, int(args.top_k))]
    best = list(reversed(diffs_sorted[-max(0, int(args.top_k)) :]))

    by_group: dict[str, list[DiffRow]] = {}
    by_size: dict[str, list[DiffRow]] = {}
    for r in diffs:
        by_group.setdefault(r.group, []).append(r)
        by_size.setdefault(r.size, []).append(r)

    report = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "name_a": args.name_a,
        "name_b": args.name_b,
        "best_threshold_a": thr_a,
        "best_threshold_b": thr_b,
        "split_a": summary_a.get("split"),
        "split_b": summary_b.get("split"),
        "n_common": len(diffs),
        "delta_summary": {
            "mean": _mean([r.delta for r in diffs]) if diffs else 0.0,
            "p10": _p([r.delta for r in diffs], 0.10),
            "p50": _p([r.delta for r in diffs], 0.50),
            "p90": _p([r.delta for r in diffs], 0.90),
        },
        "by_slice_spacing_bucket": {
            k: {"n": len(v), "mean_delta": _mean([x.delta for x in v])} for k, v in sorted(by_group.items())
        },
        "by_gt_size_bucket": {
            k: {"n": len(v), "mean_delta": _mean([x.delta for x in v])} for k, v in sorted(by_size.items())
        },
        "worst_cases": [r.__dict__ for r in worst],
        "best_cases": [r.__dict__ for r in best],
    }

    out_dir = Path(args.out).expanduser().resolve() if args.out else run_b
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "failure_mode_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
