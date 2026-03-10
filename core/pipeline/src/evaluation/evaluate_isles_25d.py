"""Evaluation script for 2.5D ConvNeXt segmentation on ISLES.

Adds 3D-style evaluation features to the public 2.5D repo:
  - multi-threshold evaluation (`per_threshold`)
  - `--probs-dir` reuse of saved probability maps
  - `--save-probs` / `--save-probs-dir`
  - temperature scaling
  - component score filtering and top-k connected components
  - GT-size / slice-spacing stratified summaries
  - optional ISLES-style extra metrics at the best threshold
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
    f1_score,
    fp_component_stats,
    gt_size_bucket,
    infer_volume,
    infer_volume_logits,
    lesionwise_f1,
    lesionwise_stats,
    load_25d_model,
    load_temperature,
    parse_float_bins,
    parse_int_bins,
    parse_thresholds,
    postprocess,
    prob_to_logit,
    slice_spacing_bucket,
    surface_distance_metrics_mm,
    voxel_volume_mm3,
)


def _safe_mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def _load_probs_npz(path: Path) -> np.ndarray:
    with np.load(str(path)) as z:
        return z["probs"].astype(np.float32, copy=False)


def _stats_for_subset(rows: list[dict[str, Any]], thr: float) -> dict[str, float | int | None]:
    if not rows:
        return {"n": 0, "mean_dice": None, "median_dice": None}
    dices = [float(r[f"dice@{thr:g}"]) for r in rows]
    gt_pos_rows = [r for r in rows if bool(r.get("gt_pos"))]
    gt_neg_rows = [r for r in rows if not bool(r.get("gt_pos"))]

    total_tp = sum(int(r[f"tp_vox@{thr:g}"]) for r in rows)
    total_fp = sum(int(r[f"fp_vox@{thr:g}"]) for r in rows)
    total_fn = sum(int(r[f"fn_vox@{thr:g}"]) for r in rows)
    total_pred_vox = sum(int(r[f"pred_vox@{thr:g}"]) for r in rows)
    fp_cc_vals = [int(r.get(f"fp_cc@{thr:g}", 0)) for r in rows]
    fp_vox_vals = [int(r[f"fp_vox@{thr:g}"]) for r in rows]

    det_rate = None
    if gt_pos_rows:
        det_rate = float(np.mean([bool(r[f"detected@{thr:g}"]) for r in gt_pos_rows]))
    far = None
    if gt_neg_rows:
        far = float(np.mean([int(r[f"pred_vox@{thr:g}"]) > 0 for r in gt_neg_rows]))

    precision = float(total_tp / (total_tp + total_fp + 1e-6))
    recall = float(total_tp / (total_tp + total_fn + 1e-6))

    return {
        "n": int(len(rows)),
        "mean_dice": float(np.mean(dices)),
        "median_dice": float(np.median(dices)),
        "mean_dice_pos": float(np.mean([float(r[f"dice@{thr:g}"]) for r in gt_pos_rows])) if gt_pos_rows else None,
        "detection_rate_case": det_rate,
        "false_alarm_rate_case": far,
        "voxel_precision": precision,
        "voxel_recall": recall,
        "mean_fp_vox": float(np.mean(fp_vox_vals)) if fp_vox_vals else None,
        "mean_fp_cc": float(np.mean(fp_cc_vals)) if fp_cc_vals else None,
        "mean_pred_vox": float(total_pred_vox / max(1, len(rows))),
    }


def _compute_extra_metrics(
    *,
    ds: IslesVolumeDataset,
    probs_src_dir: Path,
    thr: float,
    min_size: int,
    prob_filter: float,
    cc_score: str,
    cc_score_thr: float,
    top_k: int,
    closing_mm: int,
) -> dict[str, Any]:
    vol_diff_ml: list[float] = []
    abs_vol_diff_ml: list[float] = []
    lesion_count_diff: list[int] = []
    abs_lesion_count_diff: list[int] = []
    assd_mm: list[float] = []
    hd_mm: list[float] = []
    hd95_mm: list[float] = []

    total_gt_lesions = 0
    total_pred_lesions = 0
    total_tp_gt = 0
    total_tp_pred = 0
    n_dist_valid = 0

    for i in range(len(ds)):
        sample = ds[i]
        case_id = str(sample["case_id"])
        gt = (sample["mask"] > 0.5).astype(np.uint8)
        meta = sample.get("meta") or {}
        zooms = meta.get("zooms_mm") if isinstance(meta, dict) else None
        probs = _load_probs_npz(probs_src_dir / f"{case_id}.npz")
        pred = postprocess(
            probs,
            thr=float(thr),
            min_size=int(min_size),
            prob_filter=float(prob_filter),
            cc_score=str(cc_score),
            cc_score_thr=float(cc_score_thr),
            top_k=int(top_k),
            closing_mm=int(closing_mm),
        )

        vv = voxel_volume_mm3(zooms)
        if vv is not None:
            gt_ml = float(int(gt.sum()) * vv / 1000.0)
            pred_ml = float(int(pred.sum()) * vv / 1000.0)
            d = float(pred_ml - gt_ml)
            vol_diff_ml.append(d)
            abs_vol_diff_ml.append(abs(d))

        lw = lesionwise_stats(pred, gt)
        n_gt = int(lw["n_gt"])
        n_pred = int(lw["n_pred"])
        tp_gt = int(lw["tp_gt"])
        tp_pred = int(lw["tp_pred"])
        lesion_count_diff.append(n_pred - n_gt)
        abs_lesion_count_diff.append(abs(n_pred - n_gt))
        total_gt_lesions += n_gt
        total_pred_lesions += n_pred
        total_tp_gt += tp_gt
        total_tp_pred += tp_pred

        dm = surface_distance_metrics_mm(pred, gt, zooms)
        if dm.get("assd_mm") is not None:
            n_dist_valid += 1
            assd_mm.append(float(dm["assd_mm"]))
            hd_mm.append(float(dm["hd_mm"]))
            hd95_mm.append(float(dm["hd95_mm"]))

    if total_pred_lesions > 0:
        lesion_prec: float | None = float(total_tp_pred / float(total_pred_lesions))
    elif total_gt_lesions == 0:
        lesion_prec = float("nan")
    else:
        lesion_prec = 0.0

    if total_gt_lesions > 0:
        lesion_rec: float | None = float(total_tp_gt / float(total_gt_lesions))
    elif total_pred_lesions == 0:
        lesion_rec = float("nan")
    else:
        lesion_rec = 0.0

    return {
        "threshold": float(thr),
        "volume_diff_ml": {
            "mean": _safe_mean(vol_diff_ml),
            "mean_abs": _safe_mean(abs_vol_diff_ml),
        },
        "lesion_count_diff": {
            "mean": _safe_mean([float(x) for x in lesion_count_diff]),
            "mean_abs": _safe_mean([float(x) for x in abs_lesion_count_diff]),
        },
        "lesionwise": {
            "precision_micro": lesion_prec,
            "recall_micro": lesion_rec,
            "f1_micro": f1_score(lesion_prec, lesion_rec),
            "total_gt_lesions": int(total_gt_lesions),
            "total_pred_lesions": int(total_pred_lesions),
        },
        "boundary_distance_mm": {
            "n_valid": int(n_dist_valid),
            "assd_mean": _safe_mean(assd_mm),
            "hd_mean": _safe_mean(hd_mm),
            "hd95_mean": _safe_mean(hd95_mm),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate 2.5D ConvNeXt segmentation model on ISLES.")
    p.add_argument("--model-path", default=None, help="Path to best.pt checkpoint (omit when using --probs-dir).")
    p.add_argument("--probs-dir", default=None, help="Directory with <case_id>.npz probability maps.")
    p.add_argument("--csv-path", required=True, help="Split CSV (case_id, split).")
    p.add_argument("--root", required=True, help="Preprocessed data root (images/, labels/).")
    p.add_argument("--split", default="val", help="Which split to evaluate (train/val/test).")
    p.add_argument("--out-dir", required=True, help="Directory to write metrics.json + summary.json.")
    p.add_argument("--normalize", default="fixed_nonzero_zscore")
    p.add_argument("--allow-missing-label", action="store_true", default=False)
    p.add_argument("--k-slices", type=int, default=None)
    p.add_argument("--img-size", default=None, help="'H,W'. Auto from config.")
    p.add_argument("--backbone", default=None)
    p.add_argument("--dec-ch", type=int, default=None)
    p.add_argument("--deep-sup", action="store_true", default=None)
    p.add_argument("--thresholds", default="0.85", help="Comma-separated thresholds or from_run_best/from_run_last.")
    p.add_argument("--temperature", default="1.0", help="Float or from_run_best/from_run_last.")
    p.add_argument("--min-size", type=int, default=64, help="Min CC component voxels.")
    p.add_argument("--prob-filter", type=float, default=0.96, help="Min mean prob per CC component.")
    p.add_argument("--cc-score", default="none", help="none|max_prob|p95_prob|mean_prob")
    p.add_argument("--cc-score-thr", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--closing-mm", type=int, default=0)
    p.add_argument("--stage1-probs-dir", default=None)
    p.add_argument("--save-probs", action="store_true", default=False, help="Save probability maps to out_dir/probs.")
    p.add_argument("--save-probs-dir", default=None, help="If set, save probability maps as <case_id>.npz to this directory.")
    p.add_argument("--save-probs-dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--tta", action="store_true", default=False, help="Average original + LR/UD/LR+UD flips.")
    p.add_argument("--gt-size-bins", default="250,1000")
    p.add_argument("--slice-spacing-bins-mm", default="3.0")
    p.add_argument("--extra-metrics", action="store_true", default=False)
    args = p.parse_args()

    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else None
    probs_dir = Path(args.probs_dir).expanduser().resolve() if args.probs_dir else None
    if (model_path is None) == (probs_dir is None):
        raise ValueError("Provide exactly one of --model-path or --probs-dir")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_probs_dir: Optional[Path] = None
    if args.save_probs_dir:
        save_probs_dir = Path(args.save_probs_dir).expanduser().resolve()
    elif args.save_probs or args.extra_metrics:
        save_probs_dir = out_dir / "probs"
    if save_probs_dir is not None:
        save_probs_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_thresholds(model_path, str(args.thresholds))
    temperature = load_temperature(model_path, str(args.temperature)) if model_path is not None else float(args.temperature)
    gt_bins = parse_int_bins(args.gt_size_bins)
    spacing_bins = parse_float_bins(args.slice_spacing_bins_mm)
    out_dtype = np.float16 if str(args.save_probs_dtype).lower() == "float16" else np.float32

    ds = IslesVolumeDataset(
        csv_path=args.csv_path,
        split=args.split,
        root=args.root,
        normalize=args.normalize,
        allow_missing_label=bool(args.allow_missing_label),
    )
    print(f"Split='{args.split}': {len(ds)} cases")

    model_info: dict[str, Any] | None = None
    device = None
    if model_path is not None:
        device = prepare_device()
        img_size = None
        if args.img_size:
            h, w = [int(x) for x in str(args.img_size).split(",")]
            img_size = (h, w)
        model_info = load_25d_model(
            model_path,
            device,
            stage1_probs_dir=args.stage1_probs_dir,
            k_slices=args.k_slices,
            img_size=img_size,
            backbone=args.backbone,
            dec_ch=args.dec_ch,
            deep_sup=args.deep_sup,
        )
        print(
            f"Model: {model_info['backbone']}, offsets={model_info['offsets']}, "
            f"dec_ch={model_info['dec_ch']}, deep_sup={model_info['deep_sup']}, img_size={model_info['img_size']}"
        )
    else:
        print(f"Using precomputed probabilities from: {probs_dir}")
    print(f"Temperature: {temperature:.4f}")

    case_to_gt: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []

    for idx in range(len(ds)):
        sample = ds[idx]
        case_id = str(sample["case_id"])
        vol = sample["image"].astype(np.float32)
        gt = (sample["mask"] > 0.5).astype(np.uint8)
        case_to_gt[case_id] = gt
        meta = sample.get("meta") or {}
        slice_spacing_mm = None
        zooms = None
        if isinstance(meta, dict):
            slice_spacing_mm = meta.get("slice_spacing_mm")
            zooms = meta.get("zooms_mm")
        gt_vox = int(gt.sum())
        size_bucket = gt_size_bucket(gt_vox, gt_bins)
        spacing_bucket = slice_spacing_bucket(float(slice_spacing_mm) if slice_spacing_mm is not None else None, spacing_bins)

        if probs_dir is not None:
            prob = _load_probs_npz(probs_dir / f"{case_id}.npz")
            if float(temperature) != 1.0:
                prob = (1.0 / (1.0 + np.exp(-(prob_to_logit(prob) / float(temperature))))).astype(np.float32, copy=False)
        else:
            assert model_info is not None and device is not None
            extra_vol = None
            s1_dir = model_info.get("stage1_probs_dir")
            if s1_dir:
                npz_path = Path(str(s1_dir)) / f"{case_id}.npz"
                if npz_path.exists():
                    extra_vol = _load_probs_npz(npz_path)
                else:
                    extra_vol = np.zeros((vol.shape[1], vol.shape[2], vol.shape[3]), dtype=np.float32)

            if float(temperature) == 1.0:
                prob = infer_volume(
                    vol,
                    model_info["model"],
                    offsets=model_info["offsets"],
                    img_size=model_info["img_size"],
                    device=device,
                    extra_vol=extra_vol,
                    tta=bool(args.tta),
                    temperature=1.0,
                )
            else:
                logits = infer_volume_logits(
                    vol,
                    model_info["model"],
                    offsets=model_info["offsets"],
                    img_size=model_info["img_size"],
                    device=device,
                    extra_vol=extra_vol,
                    tta=bool(args.tta),
                )
                prob = (1.0 / (1.0 + np.exp(-(logits / float(temperature))))).astype(np.float32, copy=False)

        if save_probs_dir is not None:
            np.savez_compressed(str(save_probs_dir / f"{case_id}.npz"), probs=prob.astype(out_dtype, copy=False))

        rec: dict[str, Any] = {
            "case_id": case_id,
            "gt_vox": gt_vox,
            "gt_pos": bool(gt_vox > 0),
            "gt_size_bucket": size_bucket,
            "slice_spacing_bucket": spacing_bucket,
            "slice_spacing_mm": None if slice_spacing_mm is None else float(slice_spacing_mm),
            "zooms_mm": zooms,
        }

        for thr in thresholds:
            pred = postprocess(
                prob,
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
            fn_vox = gt_vox - tp_vox
            fp_cc, fp_cc_vox, fp_cc_size_p90 = fp_component_stats(pred, gt)
            dice = dice_score(pred, gt)

            rec[f"dice@{thr:g}"] = float(dice)
            rec[f"pred_vox@{thr:g}"] = int(pred_vox)
            rec[f"tp_vox@{thr:g}"] = int(tp_vox)
            rec[f"fp_vox@{thr:g}"] = int(fp_vox)
            rec[f"fn_vox@{thr:g}"] = int(fn_vox)
            rec[f"detected@{thr:g}"] = bool(tp_vox > 0)
            rec[f"fp_cc@{thr:g}"] = int(fp_cc)
            rec[f"fp_cc_vox@{thr:g}"] = int(fp_cc_vox)
            rec[f"fp_cc_size_p90@{thr:g}"] = fp_cc_size_p90

        if len(thresholds) == 1:
            thr0 = thresholds[0]
            pred0 = postprocess(
                prob,
                thr=float(thr0),
                min_size=int(args.min_size),
                prob_filter=float(args.prob_filter),
                cc_score=str(args.cc_score),
                cc_score_thr=float(args.cc_score_thr),
                top_k=int(args.top_k),
                closing_mm=int(args.closing_mm),
            )
            rec.update(
                {
                    "dice": rec[f"dice@{thr0:g}"],
                    "pred_vox": rec[f"pred_vox@{thr0:g}"],
                    "tp_vox": rec[f"tp_vox@{thr0:g}"],
                    "fp_vox": rec[f"fp_vox@{thr0:g}"],
                    "fn_vox": rec[f"fn_vox@{thr0:g}"],
                    "detected": rec[f"detected@{thr0:g}"],
                    "fp_cc": rec[f"fp_cc@{thr0:g}"],
                    "fp_cc_vox": rec[f"fp_cc_vox@{thr0:g}"],
                    "fp_cc_size_p90": rec[f"fp_cc_size_p90@{thr0:g}"],
                    **lesionwise_f1(pred0, gt),
                }
            )
            print(
                f"[{idx+1:2d}/{len(ds)}] {case_id}: gt={gt_vox:7d} pred={int(rec['pred_vox']):7d} "
                f"dice={float(rec['dice']):.4f} lesion_f1={float(rec['lesion_f1']):.3f}",
                flush=True,
            )
        else:
            best_case_thr = max(thresholds, key=lambda t: float(rec[f"dice@{t:g}"]))
            print(
                f"[{idx+1:2d}/{len(ds)}] {case_id}: gt={gt_vox:7d} best_case_thr={best_case_thr:g} "
                f"dice={float(rec[f'dice@{best_case_thr:g}']):.4f}",
                flush=True,
            )

        records.append(rec)

    per_threshold: list[dict[str, Any]] = []
    for thr in thresholds:
        row = _stats_for_subset(records, float(thr))
        by_size = {}
        for bucket in sorted(set(str(r.get("gt_size_bucket", "all")) for r in records)):
            sub = [r for r in records if str(r.get("gt_size_bucket", "all")) == bucket]
            by_size[bucket] = _stats_for_subset(sub, float(thr))
        by_spacing = {}
        for bucket in sorted(set(str(r.get("slice_spacing_bucket", "all")) for r in records)):
            sub = [r for r in records if str(r.get("slice_spacing_bucket", "all")) == bucket]
            by_spacing[bucket] = _stats_for_subset(sub, float(thr))
        row.update({
            "threshold": float(thr),
            "by_gt_size_bucket": by_size,
            "by_slice_spacing": by_spacing,
        })
        per_threshold.append(row)

    primary = max(per_threshold, key=lambda r: float(r.get("mean_dice") or -1.0)) if per_threshold else None
    best_thr = float(primary["threshold"]) if primary is not None else float(thresholds[0])

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

    summary: dict[str, Any] = {
        "model_path": None if model_path is None else str(model_path),
        "probs_dir": None if probs_dir is None else str(probs_dir),
        "csv_path": args.csv_path,
        "root": args.root,
        "split": args.split,
        "normalize": args.normalize,
        "temperature": float(temperature),
        "thresholds": [float(t) for t in thresholds],
        "gt_size_bins": [int(x) for x in gt_bins],
        "slice_spacing_bins_mm": [float(x) for x in spacing_bins],
        "min_size": int(args.min_size),
        "prob_filter": float(args.prob_filter),
        "cc_score": str(args.cc_score),
        "cc_score_thr": float(args.cc_score_thr),
        "top_k": int(args.top_k),
        "closing_mm": int(args.closing_mm),
        "n": int(len(records)),
        "n_gt_pos": int(sum(1 for r in records if bool(r.get("gt_pos")))),
        "n_gt_neg": int(sum(1 for r in records if not bool(r.get("gt_pos")))),
        "best_threshold": float(best_thr),
        "per_threshold": per_threshold,
        "mean_dice": None if primary is None else primary.get("mean_dice"),
        "median_dice": None if primary is None else primary.get("median_dice"),
        "mean_dice_pos": None if primary is None else primary.get("mean_dice_pos"),
        "detection_rate_case": None if primary is None else primary.get("detection_rate_case"),
        "false_alarm_rate_case": None if primary is None else primary.get("false_alarm_rate_case"),
        "voxel_precision": None if primary is None else primary.get("voxel_precision"),
        "voxel_recall": None if primary is None else primary.get("voxel_recall"),
        "mean_fp_vox": None if primary is None else primary.get("mean_fp_vox"),
        "mean_fp_cc": None if primary is None else primary.get("mean_fp_cc"),
        "mean_pred_vox": None if primary is None else primary.get("mean_pred_vox"),
    }
    if model_info is not None:
        summary.update(
            {
                "k_slices": int(args.k_slices) if args.k_slices is not None else int(model_info["cfg_data"].get("k_slices", 2)),
                "img_size": list(model_info["img_size"]),
                "backbone": model_info["backbone"],
                "dec_ch": int(model_info["dec_ch"]),
                "deep_sup": bool(model_info["deep_sup"]),
            }
        )

    if args.extra_metrics:
        probs_src_dir = save_probs_dir if save_probs_dir is not None else probs_dir
        if probs_src_dir is None:
            raise RuntimeError("--extra-metrics requires --probs-dir or --save-probs")
        summary["extra_metrics_best"] = _compute_extra_metrics(
            ds=ds,
            probs_src_dir=probs_src_dir,
            thr=float(best_thr),
            min_size=int(args.min_size),
            prob_filter=float(args.prob_filter),
            cc_score=str(args.cc_score),
            cc_score_thr=float(args.cc_score_thr),
            top_k=int(args.top_k),
            closing_mm=int(args.closing_mm),
        )

    (out_dir / "metrics.json").write_text(json.dumps(records, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    print(f"  best_threshold   : {best_thr:g}")
    print(f"  mean_dice       : {summary['mean_dice']:.4f}" if summary["mean_dice"] is not None else "")
    print(f"  median_dice     : {summary['median_dice']:.4f}" if summary["median_dice"] is not None else "")
    print(f"  mean_dice (pos) : {summary['mean_dice_pos']:.4f}" if summary["mean_dice_pos"] is not None else "")
    print(f"  detection_rate  : {summary['detection_rate_case']:.3f}" if summary["detection_rate_case"] is not None else "")
    print(f"  false_alarm     : {summary['false_alarm_rate_case']:.3f}" if summary["false_alarm_rate_case"] is not None else "")
    print(f"  precision       : {summary['voxel_precision']:.4f}" if summary["voxel_precision"] is not None else "")
    print(f"  recall          : {summary['voxel_recall']:.4f}" if summary["voxel_recall"] is not None else "")
    print(f"\n  Results saved → {out_dir}/")


if __name__ == "__main__":
    main()
