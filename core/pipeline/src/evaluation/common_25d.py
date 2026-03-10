from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from numpy.typing import NDArray
from scipy.ndimage import binary_closing, binary_erosion, distance_transform_edt, label as cc_label

from ..models.convnext_nnunet_seg import ConvNeXtNnUNetSeg

_STRUCT_26 = np.ones((3, 3, 3), dtype=np.uint8)


def center_pad_crop_np(arr: NDArray, out_h: int, out_w: int) -> NDArray:
    h, w = int(arr.shape[-2]), int(arr.shape[-1])
    pad_h = max(0, out_h - h)
    pad_w = max(0, out_w - w)
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    pl, pr = pad_w // 2, pad_w - pad_w // 2

    if arr.ndim == 3:
        arr = np.pad(arr, ((0, 0), (pt, pb), (pl, pr)), mode="constant", constant_values=0.0)
    else:
        arr = np.pad(arr, ((pt, pb), (pl, pr)), mode="constant", constant_values=0.0)

    h2, w2 = int(arr.shape[-2]), int(arr.shape[-1])
    if h2 > out_h:
        top = (h2 - out_h) // 2
        arr = arr[..., top : top + out_h, :]
    if w2 > out_w:
        left = (w2 - out_w) // 2
        arr = arr[..., :, left : left + out_w]
    return arr


def restore_pad_crop_np(pred: NDArray, orig_h: int, orig_w: int, out_h: int = 256, out_w: int = 256) -> NDArray:
    if orig_h > out_h:
        t = (orig_h - out_h) // 2
        pred = np.pad(pred, ((0, 0), (t, orig_h - out_h - t), (0, 0)), constant_values=0.0)
    if orig_w > out_w:
        l = (orig_w - out_w) // 2
        pred = np.pad(pred, ((0, 0), (0, 0), (l, orig_w - out_w - l)), constant_values=0.0)

    h_cur, w_cur = pred.shape[-2], pred.shape[-1]
    if h_cur > orig_h:
        t = (h_cur - orig_h) // 2
        pred = pred[..., t : t + orig_h, :]
    if w_cur > orig_w:
        l = (w_cur - orig_w) // 2
        pred = pred[..., :, l : l + orig_w]
    return pred


def _sigmoid_np(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def prob_to_logit(prob: NDArray[np.float32], eps: float = 1e-6) -> NDArray[np.float32]:
    p = np.clip(prob.astype(np.float32, copy=False), eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)


def parse_thresholds(model_path: Optional[Path], thresholds: str) -> list[float]:
    t = str(thresholds).strip().lower()
    if t in {"from_run_best", "from_run_last"}:
        if model_path is None:
            raise ValueError("--thresholds from_run_* requires --model-path")
        meta_name = "val_threshold_best.json" if t == "from_run_best" else "val_threshold_last.json"
        meta = json.loads((model_path.parent / meta_name).read_text())
        return [float(meta["val_threshold"])]
    out = [float(x) for x in str(thresholds).replace(";", ",").split(",") if x.strip()]
    if not out:
        raise ValueError("At least one threshold is required")
    return out


def load_temperature(model_path: Optional[Path], temperature: str) -> float:
    t = str(temperature).strip().lower()
    if t in {"from_run_best", "from_run_last"}:
        if model_path is None:
            raise ValueError("--temperature from_run_* requires --model-path")
        meta_name = "temperature_best.json" if t == "from_run_best" else "temperature_last.json"
        meta = json.loads((model_path.parent / meta_name).read_text())
        return float(meta.get("temperature", 1.0))
    return float(temperature)


def parse_float_bins(text: str | None) -> list[float]:
    if text is None:
        return []
    t = str(text).strip().lower()
    if t in {"", "none", "off", "false"}:
        return []
    return [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]


def parse_int_bins(text: str | None) -> list[int]:
    return [int(float(x)) for x in parse_float_bins(text)]


def gt_size_bucket(gt_vox: int, bins: list[int]) -> str:
    if not bins:
        return "all"
    xs = sorted(int(x) for x in bins)
    if gt_vox <= xs[0]:
        return f"le_{xs[0]}"
    for lo, hi in zip(xs[:-1], xs[1:]):
        if gt_vox <= hi:
            return f"{lo+1}_{hi}"
    return f"gt_{xs[-1]}"


def slice_spacing_bucket(mm: float | None, bins: list[float]) -> str:
    if mm is None:
        return "unknown"
    if not bins:
        return "all"
    xs = sorted(float(x) for x in bins)
    if mm <= xs[0]:
        return f"le_{xs[0]:g}mm"
    for lo, hi in zip(xs[:-1], xs[1:]):
        if mm <= hi:
            return f"{lo:g}_{hi:g}mm"
    return f"gt_{xs[-1]:g}mm"


def _component_score(vals: NDArray[np.float32], mode: str) -> float:
    if mode == "max_prob":
        return float(vals.max())
    if mode == "mean_prob":
        return float(vals.mean())
    if mode == "p95_prob":
        return float(np.percentile(vals, 95.0))
    raise ValueError(f"Unknown cc_score: {mode!r}")


def keep_top_k_components(pred: NDArray[np.uint8], k: int) -> NDArray[np.uint8]:
    if int(k) <= 0:
        return pred
    lbl, n_cc = cc_label(pred.astype(np.uint8), structure=_STRUCT_26)
    if int(n_cc) <= int(k):
        return pred
    sizes = []
    for comp_id in range(1, int(n_cc) + 1):
        sizes.append((int((lbl == comp_id).sum()), comp_id))
    keep = {comp_id for _, comp_id in sorted(sizes, reverse=True)[: int(k)]}
    out = np.zeros_like(pred, dtype=np.uint8)
    for comp_id in keep:
        out[lbl == comp_id] = 1
    return out


def filter_cc_score(
    pred: NDArray[np.uint8],
    probs: NDArray[np.float32],
    score_mode: str,
    score_thr: float,
) -> NDArray[np.uint8]:
    mode = str(score_mode).strip().lower()
    if mode in {"none", "off", "false"}:
        return pred
    lbl, n_cc = cc_label(pred.astype(np.uint8), structure=_STRUCT_26)
    if int(n_cc) == 0:
        return pred
    out = pred.copy()
    thr = float(score_thr)
    for comp_id in range(1, int(n_cc) + 1):
        m = lbl == comp_id
        vals = probs[m]
        if vals.size == 0 or _component_score(vals.astype(np.float32, copy=False), mode) < thr:
            out[m] = 0
    return out


def postprocess(
    prob: NDArray[np.float32],
    *,
    thr: float,
    min_size: int,
    prob_filter: float,
    cc_score: str = "none",
    cc_score_thr: float = 0.5,
    top_k: int = 0,
    closing_mm: int = 0,
) -> NDArray[np.uint8]:
    pred = (prob > float(thr)).astype(np.uint8)
    if int(min_size) > 0 or float(prob_filter) > 0.0:
        lbl, n_cc = cc_label(pred.astype(np.uint8), structure=_STRUCT_26)
        filtered = np.zeros_like(pred, dtype=np.uint8)
        for comp_id in range(1, int(n_cc) + 1):
            comp = lbl == comp_id
            if int(comp.sum()) < int(min_size):
                continue
            if float(prob_filter) > 0.0 and float(prob[comp].mean()) < float(prob_filter):
                continue
            filtered[comp] = 1
        pred = filtered
    pred = filter_cc_score(pred, prob, cc_score, float(cc_score_thr))
    pred = keep_top_k_components(pred, int(top_k))
    if int(closing_mm) > 0:
        r = int(closing_mm)
        coords = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
        sphere = (coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2) <= r ** 2
        pred = binary_closing(pred, structure=sphere).astype(np.uint8)
    return pred


def dice_score(pred: NDArray[np.uint8], gt: NDArray[np.uint8], eps: float = 1e-6) -> float:
    tp = int((pred * gt).sum())
    return float((2 * tp + eps) / (int(pred.sum()) + int(gt.sum()) + eps))


def lesionwise_f1(pred: NDArray[np.uint8], gt: NDArray[np.uint8]) -> dict[str, float | int]:
    lbl_p, n_p = cc_label(pred.astype(np.uint8), structure=_STRUCT_26)
    lbl_g, n_g = cc_label(gt.astype(np.uint8), structure=_STRUCT_26)
    tp_l = fn_l = fp_l = 0
    for gi in range(1, int(n_g) + 1):
        g_comp = lbl_g == gi
        if bool((pred[g_comp] > 0).any()):
            tp_l += 1
        else:
            fn_l += 1
    for pi in range(1, int(n_p) + 1):
        p_comp = lbl_p == pi
        if not bool((gt[p_comp] > 0).any()):
            fp_l += 1
    prec = tp_l / (tp_l + fp_l + 1e-6)
    rec = tp_l / (tp_l + fn_l + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return {
        "lesion_tp": int(tp_l),
        "lesion_fp": int(fp_l),
        "lesion_fn": int(fn_l),
        "lesion_precision": float(prec),
        "lesion_recall": float(rec),
        "lesion_f1": float(f1),
    }


def lesionwise_stats(pred: NDArray[np.uint8], gt: NDArray[np.uint8]) -> dict[str, int]:
    g = (gt > 0).astype(np.uint8, copy=False)
    p = (pred > 0).astype(np.uint8, copy=False)
    lbl_g, n_gt = cc_label(g, structure=_STRUCT_26)
    lbl_p, n_pred = cc_label(p, structure=_STRUCT_26)
    if int(n_gt) == 0 and int(n_pred) == 0:
        return {"n_gt": 0, "n_pred": 0, "tp_gt": 0, "tp_pred": 0}
    tp_pred = 0
    tp_gt = 0
    if int(n_pred) > 0 and int(g.sum()) > 0:
        ids = np.unique(lbl_p[g > 0])
        tp_pred = int(np.sum(ids > 0))
    if int(n_gt) > 0 and int(p.sum()) > 0:
        ids = np.unique(lbl_g[p > 0])
        tp_gt = int(np.sum(ids > 0))
    return {"n_gt": int(n_gt), "n_pred": int(n_pred), "tp_gt": int(tp_gt), "tp_pred": int(tp_pred)}


def f1_score(prec: float | None, rec: float | None) -> float | None:
    if prec is None or rec is None:
        return None
    if (prec + rec) <= 0:
        return 0.0
    return float((2.0 * prec * rec) / (prec + rec))


def fp_component_sizes(pred: NDArray[np.uint8], gt: NDArray[np.uint8]) -> list[int]:
    lbl = cc_label(pred.astype(np.uint8), structure=_STRUCT_26)[0].astype(np.int64, copy=False)
    if int(lbl.max()) == 0:
        return []
    sizes = np.bincount(lbl.ravel())
    out: list[int] = []
    for comp_id in range(1, int(lbl.max()) + 1):
        comp_sz = int(sizes[comp_id]) if comp_id < len(sizes) else int((lbl == comp_id).sum())
        if comp_sz > 0 and int((gt[lbl == comp_id] > 0).sum()) == 0:
            out.append(comp_sz)
    return out


def fp_component_stats(pred: NDArray[np.uint8], gt: NDArray[np.uint8]) -> tuple[int, int, float | None]:
    sizes = fp_component_sizes(pred, gt)
    if not sizes:
        return 0, 0, None
    fp_cc = int(len(sizes))
    fp_cc_vox = int(np.sum(np.asarray(sizes, dtype=np.int64)))
    p90 = float(np.percentile(np.asarray(sizes, dtype=np.float32), 90.0))
    return fp_cc, fp_cc_vox, p90


def safe_zooms_xyz(zooms_xyz: list[float] | tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if zooms_xyz is None:
        return None
    try:
        x, y, z = float(zooms_xyz[0]), float(zooms_xyz[1]), float(zooms_xyz[2])
    except Exception:
        return None
    if (not np.isfinite(x)) or (not np.isfinite(y)) or (not np.isfinite(z)) or x <= 0 or y <= 0 or z <= 0:
        return None
    return (x, y, z)


def voxel_volume_mm3(zooms_xyz: list[float] | tuple[float, float, float] | None) -> float | None:
    z = safe_zooms_xyz(zooms_xyz)
    if z is None:
        return None
    return float(z[0] * z[1] * z[2])


def surface_mask(mask: NDArray[np.uint8]) -> NDArray[np.bool_]:
    m = (mask > 0).astype(bool, copy=False)
    if not bool(m.any()):
        return np.zeros_like(m, dtype=bool)
    er = binary_erosion(m, structure=_STRUCT_26, border_value=0)
    return (m & (~er)).astype(bool, copy=False)


def surface_distance_metrics_mm(
    pred: NDArray[np.uint8],
    gt: NDArray[np.uint8],
    zooms_xyz: list[float] | tuple[float, float, float] | None,
) -> dict[str, float | None]:
    pred_any = bool((pred > 0).any())
    gt_any = bool((gt > 0).any())
    if (not pred_any) and (not gt_any):
        return {"assd_mm": 0.0, "hd_mm": 0.0, "hd95_mm": 0.0}
    if pred_any != gt_any:
        return {"assd_mm": None, "hd_mm": None, "hd95_mm": None}
    z = safe_zooms_xyz(zooms_xyz)
    sampling_zyx = (1.0, 1.0, 1.0) if z is None else (float(z[2]), float(z[1]), float(z[0]))
    sp = surface_mask(pred)
    sg = surface_mask(gt)
    if (not bool(sp.any())) or (not bool(sg.any())):
        return {"assd_mm": None, "hd_mm": None, "hd95_mm": None}
    dt_to_gt = distance_transform_edt(~sg, sampling=sampling_zyx)
    dt_to_pred = distance_transform_edt(~sp, sampling=sampling_zyx)
    d1 = dt_to_gt[sp]
    d2 = dt_to_pred[sg]
    if d1.size == 0 or d2.size == 0:
        return {"assd_mm": None, "hd_mm": None, "hd95_mm": None}
    d = np.concatenate([d1.astype(np.float32, copy=False), d2.astype(np.float32, copy=False)], axis=0)
    return {
        "assd_mm": float(np.mean(d)),
        "hd_mm": float(np.max(d)),
        "hd95_mm": float(np.percentile(d, 95.0)),
    }


def _infer_volume_logits_single(
    vol: NDArray[np.float32],
    model: torch.nn.Module,
    offsets: list[int],
    img_size: tuple[int, int],
    device: torch.device,
    extra_vol: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    _, zdim, out_h_orig, out_w_orig = vol.shape
    out_h, out_w = img_size
    logits_256 = np.zeros((zdim, out_h, out_w), dtype=np.float32)
    with torch.no_grad():
        for z in range(zdim):
            slices = [
                center_pad_crop_np(vol[:, int(np.clip(z + off, 0, zdim - 1)), :, :], out_h, out_w)
                for off in offsets
            ]
            inp_arr = np.concatenate(slices, axis=0)
            if extra_vol is not None:
                extra_slice = center_pad_crop_np(extra_vol[z : z + 1], out_h, out_w)
                inp_arr = np.concatenate([inp_arr, extra_slice], axis=0)
            inp = torch.from_numpy(inp_arr).float().unsqueeze(0).to(device)
            logits = model(inp)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            logits_256[z] = logits.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
    return logits_256


def infer_volume_logits(
    vol: NDArray[np.float32],
    model: torch.nn.Module,
    offsets: list[int],
    img_size: tuple[int, int],
    device: torch.device,
    extra_vol: Optional[NDArray[np.float32]] = None,
    tta: bool = False,
) -> NDArray[np.float32]:
    _, _, h_orig, w_orig = vol.shape
    out_h, out_w = img_size
    logits_256 = _infer_volume_logits_single(vol, model, offsets, img_size, device, extra_vol)
    if tta:
        vol_lr = vol[:, :, :, ::-1].copy()
        ext_lr = extra_vol[:, :, ::-1].copy() if extra_vol is not None else None
        l_lr = _infer_volume_logits_single(vol_lr, model, offsets, img_size, device, ext_lr)[:, :, ::-1]

        vol_ud = vol[:, :, ::-1, :].copy()
        ext_ud = extra_vol[:, ::-1, :].copy() if extra_vol is not None else None
        l_ud = _infer_volume_logits_single(vol_ud, model, offsets, img_size, device, ext_ud)[:, ::-1, :]

        vol_lrud = vol[:, :, ::-1, ::-1].copy()
        ext_lrud = extra_vol[:, ::-1, ::-1].copy() if extra_vol is not None else None
        l_lrud = _infer_volume_logits_single(vol_lrud, model, offsets, img_size, device, ext_lrud)[:, ::-1, ::-1]

        logits_256 = (logits_256 + l_lr + l_ud + l_lrud) / 4.0
    return restore_pad_crop_np(logits_256, h_orig, w_orig, out_h, out_w).astype(np.float32, copy=False)


def infer_volume(
    vol: NDArray[np.float32],
    model: torch.nn.Module,
    offsets: list[int],
    img_size: tuple[int, int],
    device: torch.device,
    extra_vol: Optional[NDArray[np.float32]] = None,
    tta: bool = False,
    temperature: float = 1.0,
) -> NDArray[np.float32]:
    logits = infer_volume_logits(vol, model, offsets, img_size, device, extra_vol=extra_vol, tta=tta)
    return _sigmoid_np(logits / float(temperature))


def resolve_stage1_dir(cfg_data: dict[str, Any], cli_stage1_probs_dir: Optional[str]) -> str | None:
    cands = [cli_stage1_probs_dir, cfg_data.get("stage1_probs_dir_val"), cfg_data.get("stage1_probs_dir")]
    for c in cands:
        if c is None:
            continue
        s = str(c).strip()
        if s:
            return s
    return None


def load_25d_config(model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = model_path.parent / "config.yaml"
    cfg_data: dict[str, Any] = {}
    cfg_train: dict[str, Any] = {}
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
        cfg_data = dict(cfg.get("data", {}) or {})
        cfg_train = dict(cfg.get("train", {}) or {})
    return cfg_data, cfg_train


def load_25d_model(
    model_path: Path,
    device: torch.device,
    *,
    stage1_probs_dir: Optional[str] = None,
    k_slices: Optional[int] = None,
    img_size: Optional[tuple[int, int]] = None,
    backbone: Optional[str] = None,
    dec_ch: Optional[int] = None,
    deep_sup: Optional[bool] = None,
) -> dict[str, Any]:
    cfg_data, cfg_train = load_25d_config(model_path)
    k = int(k_slices if k_slices is not None else cfg_data.get("k_slices", 2))
    raw_offsets = cfg_data.get("slice_offsets")
    offsets = [int(x) for x in raw_offsets] if raw_offsets is not None else list(range(-k, k + 1))
    if img_size is None:
        raw_img_size = cfg_data.get("img_size")
        img_size = tuple(int(x) for x in raw_img_size) if raw_img_size else (256, 256)
    bb = str(backbone if backbone is not None else cfg_train.get("backbone", "convnext_tiny"))
    dch = int(dec_ch if dec_ch is not None else cfg_train.get("dec_ch", 256))
    dsup = bool(cfg_train.get("deep_sup", False)) if deep_sup is None else bool(deep_sup)
    hint_attn = bool(cfg_train.get("hint_attn", False))
    resolved_stage1 = resolve_stage1_dir(cfg_data, stage1_probs_dir)
    has_stage1 = bool(resolved_stage1 and Path(resolved_stage1).exists())
    in_channels = 3 * len(offsets) + (1 if has_stage1 else 0)

    model = ConvNeXtNnUNetSeg(
        in_channels=in_channels,
        backbone=bb,
        pretrained=False,
        first_conv_init=str(cfg_train.get("first_conv_init", "repeat")),
        dec_ch=dch,
        out_channels=1,
        deep_sup=dsup,
        hint_attn=hint_attn,
        stage_dropout_p=float(cfg_train.get("stage_dropout_p", 0.0)),
        decoder_dropout_p=float(cfg_train.get("decoder_dropout_p", 0.0)),
    )
    state = torch.load(str(model_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state and not any(k.startswith("encoder.") for k in state):
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    return {
        "model": model,
        "offsets": offsets,
        "img_size": img_size,
        "hint_attn": hint_attn,
        "stage1_probs_dir": resolved_stage1,
        "cfg_data": cfg_data,
        "cfg_train": cfg_train,
        "backbone": bb,
        "dec_ch": dch,
        "deep_sup": dsup,
    }
