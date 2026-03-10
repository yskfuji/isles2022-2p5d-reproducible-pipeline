"""Ensemble evaluation for 2.5D ISLES models.

Supports either:
- direct checkpoint ensembling (`--model-paths`), or
- saved probability-map ensembling (`--probs-dirs`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..datasets.isles_dataset import IslesVolumeDataset
from ..training.utils_train import prepare_device
from .common_25d import (
    dice_score,
    fp_component_stats,
    gt_size_bucket,
    infer_volume,
    infer_volume_logits,
    lesionwise_f1,
    load_25d_model,
    parse_float_bins,
    parse_int_bins,
    parse_thresholds,
    postprocess,
    prob_to_logit,
    slice_spacing_bucket,
)
from .evaluate_isles_25d import _load_probs_npz, _stats_for_subset


def main() -> None:
    p = argparse.ArgumentParser(description="Ensemble 2.5D ConvNeXt models on ISLES.")
    p.add_argument("--model-paths", nargs="*", default=None, help="Paths to best.pt checkpoints.")
    p.add_argument("--probs-dirs", nargs="*", default=None, help="Directories containing <case_id>.npz probability maps.")
    p.add_argument("--csv-path", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--normalize", default="fixed_nonzero_zscore")
    p.add_argument("--thresholds", default="0.85")
    p.add_argument("--temperature", default="1.0")
    p.add_argument("--min-size", type=int, default=64)
    p.add_argument("--prob-filter", type=float, default=0.96)
    p.add_argument("--cc-score", default="none")
    p.add_argument("--cc-score-thr", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--closing-mm", type=int, default=0)
    p.add_argument("--tta", action="store_true", default=False)
    p.add_argument("--stage1-probs-dirs", nargs="*", default=None,
                   help="Per-model Stage1 prob dirs. Use 'none' for models without hints.")
    p.add_argument("--save-probs-dir", default=None)
    p.add_argument("--save-probs-dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--gt-size-bins", default="250,1000")
    p.add_argument("--slice-spacing-bins-mm", default="3.0")
    args = p.parse_args()

    model_paths = [Path(x).expanduser().resolve() for x in (args.model_paths or []) if str(x).strip()]
    probs_dirs = [Path(x).expanduser().resolve() for x in (args.probs_dirs or []) if str(x).strip()]
    if not model_paths and not probs_dirs:
        raise ValueError("Provide --model-paths or --probs-dirs")
    if model_paths and probs_dirs:
        raise ValueError("Use either --model-paths or --probs-dirs, not both")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_probs_dir: Optional[Path] = Path(args.save_probs_dir).expanduser().resolve() if args.save_probs_dir else None
    if save_probs_dir is not None:
        save_probs_dir.mkdir(parents=True, exist_ok=True)
    out_dtype = np.float16 if str(args.save_probs_dtype).lower() == "float16" else np.float32

    thresholds = parse_thresholds(model_paths[0] if model_paths else None, str(args.thresholds))
    temperature = float(args.temperature)
    gt_bins = parse_int_bins(args.gt_size_bins)
    spacing_bins = parse_float_bins(args.slice_spacing_bins_mm)

    ds = IslesVolumeDataset(csv_path=args.csv_path, split=args.split, root=args.root, normalize=args.normalize)
    print(f"Split='{args.split}': {len(ds)} cases")

    device = prepare_device() if model_paths else None
    models_info: list[dict[str, Any]] = []
    s1_dirs: list[Optional[Path]] = []
    if model_paths:
        raw_s1 = list(args.stage1_probs_dirs or [])
        while len(raw_s1) < len(model_paths):
            raw_s1.append(None)
        for d in raw_s1[: len(model_paths)]:
            if d is None or str(d).strip().lower() == "none":
                s1_dirs.append(None)
            else:
                s1_dirs.append(Path(str(d)).expanduser().resolve())
        for mp, s1 in zip(model_paths, s1_dirs):
            info = load_25d_model(mp, device, stage1_probs_dir=str(s1) if s1 is not None else None)
            models_info.append(info)
            print(f"Loaded: {mp.name} offsets={info['offsets']} img_size={info['img_size']}")
    else:
        print(f"Using saved probabilities from {len(probs_dirs)} directories")

    records: list[dict[str, Any]] = []

    for idx in range(len(ds)):
        sample = ds[idx]
        case_id = str(sample["case_id"])
        vol = sample["image"].astype(np.float32)
        gt = (sample["mask"] > 0.5).astype(np.uint8)
        meta = sample.get("meta") or {}
        slice_spacing_mm = meta.get("slice_spacing_mm") if isinstance(meta, dict) else None
        size_bucket = gt_size_bucket(int(gt.sum()), gt_bins)
        spacing_bucket = slice_spacing_bucket(float(slice_spacing_mm) if slice_spacing_mm is not None else None, spacing_bins)

        probs_list: list[np.ndarray] = []
        if models_info:
            assert device is not None
            for info in models_info:
                extra_vol = None
                s1_dir = info.get("stage1_probs_dir")
                if s1_dir:
                    npz_path = Path(str(s1_dir)) / f"{case_id}.npz"
                    if npz_path.exists():
                        extra_vol = _load_probs_npz(npz_path)
                    else:
                        extra_vol = np.zeros((vol.shape[1], vol.shape[2], vol.shape[3]), dtype=np.float32)
                if float(temperature) == 1.0:
                    prob = infer_volume(
                        vol,
                        info["model"],
                        offsets=info["offsets"],
                        img_size=info["img_size"],
                        device=device,
                        extra_vol=extra_vol,
                        tta=bool(args.tta),
                        temperature=1.0,
                    )
                else:
                    logits = infer_volume_logits(
                        vol,
                        info["model"],
                        offsets=info["offsets"],
                        img_size=info["img_size"],
                        device=device,
                        extra_vol=extra_vol,
                        tta=bool(args.tta),
                    )
                    prob = (1.0 / (1.0 + np.exp(-(logits / float(temperature))))).astype(np.float32, copy=False)
                probs_list.append(prob)
        else:
            for d in probs_dirs:
                probs_list.append(_load_probs_npz(d / f"{case_id}.npz"))
            if float(temperature) != 1.0:
                probs_list = [
                    (1.0 / (1.0 + np.exp(-(prob_to_logit(p) / float(temperature))))).astype(np.float32, copy=False)
                    for p in probs_list
                ]

        prob_ens = np.mean(np.stack(probs_list, axis=0), axis=0).astype(np.float32, copy=False)
        if save_probs_dir is not None:
            np.savez_compressed(str(save_probs_dir / f"{case_id}.npz"), probs=prob_ens.astype(out_dtype, copy=False))

        rec: dict[str, Any] = {
            "case_id": case_id,
            "gt_vox": int(gt.sum()),
            "gt_pos": bool(int(gt.sum()) > 0),
            "gt_size_bucket": size_bucket,
            "slice_spacing_bucket": spacing_bucket,
            "slice_spacing_mm": None if slice_spacing_mm is None else float(slice_spacing_mm),
        }
        for thr in thresholds:
            pred = postprocess(
                prob_ens,
                thr=float(thr),
                min_size=int(args.min_size),
                prob_filter=float(args.prob_filter),
                cc_score=str(args.cc_score),
                cc_score_thr=float(args.cc_score_thr),
                top_k=int(args.top_k),
                closing_mm=int(args.closing_mm),
            )
            pred_vox = int(pred.sum())
            tp_vox = int((pred * gt).sum())
            fp_vox = pred_vox - tp_vox
            fn_vox = int(gt.sum()) - tp_vox
            fp_cc, fp_cc_vox, fp_cc_size_p90 = fp_component_stats(pred, gt)
            rec[f"dice@{thr:g}"] = float(dice_score(pred, gt))
            rec[f"pred_vox@{thr:g}"] = int(pred_vox)
            rec[f"tp_vox@{thr:g}"] = int(tp_vox)
            rec[f"fp_vox@{thr:g}"] = int(fp_vox)
            rec[f"fn_vox@{thr:g}"] = int(fn_vox)
            rec[f"detected@{thr:g}"] = bool(tp_vox > 0)
            rec[f"fp_cc@{thr:g}"] = int(fp_cc)
            rec[f"fp_cc_vox@{thr:g}"] = int(fp_cc_vox)
            rec[f"fp_cc_size_p90@{thr:g}"] = fp_cc_size_p90

        best_case_thr = max(thresholds, key=lambda t: float(rec[f"dice@{t:g}"]))
        print(
            f"[{idx+1:2d}/{len(ds)}] {case_id}: gt={int(gt.sum()):7d} best_case_thr={best_case_thr:g} "
            f"dice={float(rec[f'dice@{best_case_thr:g}']):.4f}",
            flush=True,
        )
        records.append(rec)

    per_threshold: list[dict[str, Any]] = []
    for thr in thresholds:
        row = _stats_for_subset(records, float(thr))
        row["threshold"] = float(thr)
        row["by_gt_size_bucket"] = {
            bucket: _stats_for_subset([r for r in records if str(r.get("gt_size_bucket")) == bucket], float(thr))
            for bucket in sorted(set(str(r.get("gt_size_bucket", "all")) for r in records))
        }
        row["by_slice_spacing"] = {
            bucket: _stats_for_subset([r for r in records if str(r.get("slice_spacing_bucket")) == bucket], float(thr))
            for bucket in sorted(set(str(r.get("slice_spacing_bucket", "all")) for r in records))
        }
        per_threshold.append(row)

    primary = max(per_threshold, key=lambda r: float(r.get("mean_dice") or -1.0))
    best_thr = float(primary["threshold"])
    for rec in records:
        rec.update(
            {
                "dice": rec[f"dice@{best_thr:g}"],
                "pred_vox": rec[f"pred_vox@{best_thr:g}"],
                "tp_vox": rec[f"tp_vox@{best_thr:g}"],
                "fp_vox": rec[f"fp_vox@{best_thr:g}"],
                "fn_vox": rec[f"fn_vox@{best_thr:g}"],
                "detected": rec[f"detected@{best_thr:g}"],
                "fp_cc": rec[f"fp_cc@{best_thr:g}"],
                "fp_cc_vox": rec[f"fp_cc_vox@{best_thr:g}"],
                "fp_cc_size_p90": rec[f"fp_cc_size_p90@{best_thr:g}"],
            }
        )

    summary = {
        "model_paths": [str(x) for x in model_paths] if model_paths else None,
        "probs_dirs": [str(x) for x in probs_dirs] if probs_dirs else None,
        "n_models": int(len(model_paths) or len(probs_dirs)),
        "split": args.split,
        "csv_path": args.csv_path,
        "root": args.root,
        "normalize": args.normalize,
        "temperature": float(temperature),
        "thresholds": [float(t) for t in thresholds],
        "best_threshold": float(best_thr),
        "min_size": int(args.min_size),
        "prob_filter": float(args.prob_filter),
        "cc_score": str(args.cc_score),
        "cc_score_thr": float(args.cc_score_thr),
        "top_k": int(args.top_k),
        "closing_mm": int(args.closing_mm),
        "gt_size_bins": [int(x) for x in gt_bins],
        "slice_spacing_bins_mm": [float(x) for x in spacing_bins],
        "n": int(len(records)),
        "mean_dice": primary.get("mean_dice"),
        "median_dice": primary.get("median_dice"),
        "mean_dice_pos": primary.get("mean_dice_pos"),
        "detection_rate_case": primary.get("detection_rate_case"),
        "false_alarm_rate_case": primary.get("false_alarm_rate_case"),
        "voxel_precision": primary.get("voxel_precision"),
        "voxel_recall": primary.get("voxel_recall"),
        "mean_fp_vox": primary.get("mean_fp_vox"),
        "mean_fp_cc": primary.get("mean_fp_cc"),
        "mean_pred_vox": primary.get("mean_pred_vox"),
        "per_threshold": per_threshold,
    }

    (out_dir / "metrics.json").write_text(json.dumps(records, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    print(f"  best_threshold   : {best_thr:g}")
    print(f"  mean_dice       : {summary['mean_dice']:.4f}" if summary["mean_dice"] is not None else "")
    print(f"  median_dice     : {summary['median_dice']:.4f}" if summary["median_dice"] is not None else "")
    print(f"\n  Results saved → {out_dir}/")


if __name__ == "__main__":
    main()
