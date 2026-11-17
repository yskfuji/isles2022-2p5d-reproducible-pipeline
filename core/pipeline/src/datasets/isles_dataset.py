"""Dataset definitions for medical volumes and 2.5D slices (ISLES-compatible).

Replace paths/csv/modalities to adapt for other datasets.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
from ..preprocess.utils_io import load_nifti


def _safe_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _compute_fixed_nonzero_zscore_stats(
    *,
    csv_path: str,
    root: Path,
    percentiles: tuple[float, float] = (0.5, 99.5),
    max_vox_per_case: int = 50_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute dataset-level (train-split) stats for fixed_nonzero_zscore.

    We sample up to max_vox_per_case nonzero voxels per case per channel to keep
    this reasonably fast/memory-bounded.
    """
    df = pd.read_csv(csv_path)
    df_tr = df[df["split"] == "train"].reset_index(drop=True)
    rng = np.random.default_rng(seed)

    # Gather sampled voxels per channel.
    ch_samples: list[list[Any]] = []

    for _, row in df_tr.iterrows():
        case_id = str(row["case_id"])
        img_path_raw = None
        if "image_path" in df_tr.columns:
            v = row.get("image_path")
            if isinstance(v, str) and v.strip():
                img_path_raw = v.strip()
        if img_path_raw is None:
            img_path = root / "images" / f"{case_id}.nii.gz"
        else:
            p = Path(img_path_raw)
            img_path = p if p.is_absolute() else (root / p)

        img_data, _ = load_nifti(str(img_path))
        if img_data.ndim == 3:
            img_data = img_data[None, ...]
        img_data = img_data.astype(np.float32)

        while len(ch_samples) < img_data.shape[0]:
            ch_samples.append([])

        for c in range(img_data.shape[0]):
            ch = img_data[c]
            nonzero = ch != 0
            if not np.any(nonzero):
                continue
            vals = ch[nonzero]
            if vals.size > max_vox_per_case:
                idx = rng.choice(vals.size, size=max_vox_per_case, replace=False)
                vals = vals[idx]
            ch_samples[c].append(vals)

    stats: dict[str, Any] = {
        "mode": "fixed_nonzero_zscore",
        "percentiles": [float(percentiles[0]), float(percentiles[1])],
        "max_vox_per_case": int(max_vox_per_case),
        "seed": int(seed),
        "channels": [],
    }

    for c, parts in enumerate(ch_samples):
        if not parts:
            stats["channels"].append({"lo": 0.0, "hi": 1.0, "mean": 0.0, "std": 1.0, "n": 0})
            continue
        vals = np.concatenate(parts, axis=0)
        lo, hi = np.percentile(vals, [percentiles[0], percentiles[1]]).astype(np.float64)
        vals_clip = np.clip(vals.astype(np.float64), float(lo), float(hi))
        mean = float(vals_clip.mean())
        std = float(vals_clip.std())
        stats["channels"].append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "mean": mean,
                "std": std,
                "n": int(vals.size),
            }
        )

    return stats


class IslesVolumeDataset(Dataset):
    """Return full 3D volumes (C, Z, Y, X) and mask (Z, Y, X)."""

    def __init__(
        self,
        csv_path: str,
        split: str,
        root: str,
        transform: Optional[Any] = None,
        normalize: str = "legacy_zscore",
        allow_missing_label: bool = False,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.normalize = normalize
        self.allow_missing_label = bool(allow_missing_label)

        self._fixed_norm_stats: dict[str, Any] | None = None
        mode = (self.normalize or "none").lower()
        if mode in {"fixed_nonzero_zscore", "trainset_nonzero_zscore"}:
            stats_path = self.root / "norm_stats_fixed_nonzero_zscore.json"
            if stats_path.exists():
                self._fixed_norm_stats = json.loads(stats_path.read_text())
            else:
                stats = _compute_fixed_nonzero_zscore_stats(csv_path=csv_path, root=self.root)
                _safe_write_json(stats_path, stats)
                self._fixed_norm_stats = stats

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]
        case_id = str(row["case_id"])

        img_path_raw = None
        if "image_path" in self.df.columns:
            v = row.get("image_path")
            if isinstance(v, str) and v.strip():
                img_path_raw = v.strip()
        if img_path_raw is None:
            img_path = self.root / "images" / f"{case_id}.nii.gz"
        else:
            p = Path(img_path_raw)
            img_path = p if p.is_absolute() else (self.root / p)

        lbl_path_raw = None
        if "label_path" in self.df.columns:
            v = row.get("label_path")
            if isinstance(v, str) and v.strip():
                lbl_path_raw = v.strip()

        if lbl_path_raw is None:
            lbl_path = self.root / "labels" / f"{case_id}.nii.gz"
        else:
            p = Path(lbl_path_raw)
            lbl_path = p if p.is_absolute() else (self.root / p)

        img_data, img_nii = load_nifti(str(img_path))
        if lbl_path.exists():
            lbl_data, _ = load_nifti(str(lbl_path))
        else:
            if not self.allow_missing_label:
                raise FileNotFoundError(f"Label not found for case_id={case_id!r}: {lbl_path}")
            # Treat missing label as negative case (all zeros).
            if img_data.ndim == 4:
                zyx = img_data.shape[1:]
            else:
                zyx = img_data.shape
            lbl_data = np.zeros(zyx, dtype=np.float32)

        if img_data.ndim == 3:
            img_data = img_data[None, ...]

        img_norm = img_data.astype(np.float32)
        mode = (self.normalize or "none").lower()
        if mode in {"none", "off", "false"}:
            pass
        elif mode in {"legacy", "legacy_zscore", "zscore"}:
            # legacy: 背景0を含めた全ボリュームでmean/std
            for c in range(img_norm.shape[0]):
                ch = img_norm[c]
                mean = float(ch.mean())
                std = float(ch.std())
                img_norm[c] = (ch - mean) / (std + 1e-8)
            img_data = img_norm
        elif mode in {"nonzero", "nonzero_zscore", "nnunet"}:
            # best-practice: 非ゼロ領域のみで統計、背景は0維持 + 軽いクリップ
            for c in range(img_norm.shape[0]):
                ch = img_norm[c]
                nonzero = ch != 0
                if np.any(nonzero):
                    vals = ch[nonzero]
                    lo, hi = np.percentile(vals, [0.5, 99.5])
                    vals_clip = np.clip(vals, lo, hi)
                    mean = float(vals_clip.mean())
                    std = float(vals_clip.std())
                    out = np.zeros_like(ch, dtype=np.float32)
                    out[nonzero] = (np.clip(ch[nonzero], lo, hi) - mean) / (std + 1e-8)
                    img_norm[c] = out
                else:
                    mean = float(ch.mean())
                    std = float(ch.std())
                    img_norm[c] = (ch - mean) / (std + 1e-8)
            img_data = img_norm
        elif mode in {"fixed_nonzero_zscore", "trainset_nonzero_zscore"}:
            stats = self._fixed_norm_stats
            if not stats or not stats.get("channels"):
                raise RuntimeError("fixed_nonzero_zscore stats missing; expected norm_stats_fixed_nonzero_zscore.json")
            ch_stats = list(stats["channels"])
            for c in range(img_norm.shape[0]):
                ch = img_norm[c]
                nonzero = ch != 0
                if np.any(nonzero) and c < len(ch_stats):
                    st = ch_stats[c] or {}
                    lo = float(st.get("lo", 0.0))
                    hi = float(st.get("hi", 1.0))
                    mean = float(st.get("mean", 0.0))
                    std = float(st.get("std", 1.0))
                    out = np.zeros_like(ch, dtype=np.float32)
                    out[nonzero] = (np.clip(ch[nonzero], lo, hi) - mean) / (std + 1e-8)
                    img_norm[c] = out
                else:
                    mean = float(ch.mean())
                    std = float(ch.std())
                    img_norm[c] = (ch - mean) / (std + 1e-8)
            img_data = img_norm
        elif mode in {"robust_nonzero_zscore", "nonzero_robust_zscore", "nonzero_mad", "mad"}:
            # robust: 非ゼロ領域のみ + percentile clip の後、median/MAD で標準化
            for c in range(img_norm.shape[0]):
                ch = img_norm[c]
                nonzero = ch != 0
                if np.any(nonzero):
                    vals = ch[nonzero]
                    lo, hi = np.percentile(vals, [0.5, 99.5])
                    vals_clip = np.clip(vals, lo, hi)
                    med = float(np.median(vals_clip))
                    mad = float(np.median(np.abs(vals_clip - med)))
                    # Consistent estimator for std under normality.
                    denom = (1.4826 * mad) + 1e-8
                    out = np.zeros_like(ch, dtype=np.float32)
                    out[nonzero] = (np.clip(ch[nonzero], lo, hi) - med) / denom
                    img_norm[c] = out
                else:
                    mean = float(ch.mean())
                    std = float(ch.std())
                    img_norm[c] = (ch - mean) / (std + 1e-8)
            img_data = img_norm
        elif mode in {"nonzero_minmax", "nonzero_minmax01", "minmax_nonzero"}:
            # min-max: 非ゼロ領域を percentile clip して [0,1] にスケール（背景は0維持）
            for c in range(img_norm.shape[0]):
                ch = img_norm[c]
                nonzero = ch != 0
                if np.any(nonzero):
                    vals = ch[nonzero]
                    lo, hi = np.percentile(vals, [0.5, 99.5])
                    denom = float(hi - lo) + 1e-8
                    out = np.zeros_like(ch, dtype=np.float32)
                    out[nonzero] = (np.clip(ch[nonzero], lo, hi) - float(lo)) / denom
                    img_norm[c] = out
                else:
                    lo = float(ch.min())
                    hi = float(ch.max())
                    denom = float(hi - lo) + 1e-8
                    img_norm[c] = (ch - lo) / denom
            img_data = img_norm
        else:
            raise ValueError(f"Unknown normalize mode: {self.normalize!r}")

        zooms = tuple(float(z) for z in img_nii.header.get_zooms()[:3])
        slice_spacing_mm = float(max(zooms)) if zooms else None

        sample = {
            "image": img_data.astype(np.float32),
            "mask": lbl_data.astype(np.float32),
            "case_id": case_id,
            "meta": {
                "image_path": str(img_path),
                "label_path": str(lbl_path),
                "zooms_mm": [float(z) for z in zooms],
                "slice_spacing_mm": slice_spacing_mm,
            },
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class IslesSliceDataset(Dataset):
    """2.5D slices for training 2D U-Net."""

    def __init__(self, volume_dataset: IslesVolumeDataset, k: int = 2, transform: Optional[Any] = None,
                 slice_offsets: Optional[List[int]] = None,
                 stage1_probs_dir: Optional[str] = None) -> None:
        self.volume_dataset = volume_dataset
        self.k = k
        self.transform = transform
        self._offsets: List[int] = list(slice_offsets) if slice_offsets is not None else list(range(-k, k + 1))
        self._stage1_probs_dir: Optional[Path] = Path(stage1_probs_dir) if stage1_probs_dir is not None else None
        self.index_map: list[tuple[int, int]] = []

        # Build slice index without loading full volumes into memory.
        df = self.volume_dataset.df
        root = self.volume_dataset.root
        for vidx in range(len(self.volume_dataset)):
            row = df.iloc[int(vidx)]
            case_id = str(row["case_id"])

            img_path_raw = None
            if "image_path" in df.columns:
                v = row.get("image_path")
                if isinstance(v, str) and v.strip():
                    img_path_raw = v.strip()
            if img_path_raw is None:
                img_path = root / "images" / f"{case_id}.nii.gz"
            else:
                p = Path(img_path_raw)
                img_path = p if p.is_absolute() else (root / p)

            img = nib.load(str(img_path))
            shape = tuple(int(s) for s in img.header.get_data_shape())
            # NIfTI can be (Z,Y,X) or (C,Z,Y,X). Match IslesVolumeDataset behavior.
            if len(shape) == 4:
                zdim = int(shape[1])
            elif len(shape) == 3:
                zdim = int(shape[0])
            else:
                raise ValueError(f"Unexpected NIfTI shape for case_id={case_id!r}: {shape}")

            for z in range(zdim):
                self.index_map.append((int(vidx), int(z)))

        # Small cache: helps when a worker happens to request nearby slices.
        self._cache_vidx: int | None = None
        self._cache_case: Dict[str, Any] | None = None
        self._cache_probs_vidx: Optional[int] = None
        self._cache_probs_zyx: Optional[np.ndarray] = None  # (Z, Y, X), float32

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vidx, z = self.index_map[idx]
        if self._cache_vidx == int(vidx) and self._cache_case is not None:
            case = self._cache_case
        else:
            case = self.volume_dataset[int(vidx)]
            self._cache_vidx = int(vidx)
            self._cache_case = case
        img3d = case["image"]
        mask3d = case["mask"]
        case_id = case["case_id"]

        slices = []
        for offset in self._offsets:
            zi = int(np.clip(z + offset, 0, img3d.shape[1] - 1))
            slices.append(img3d[:, zi])
        img2_5d = np.concatenate(slices, axis=0)
        mask2d = mask3d[z]

        # --- Stage1 prob channel (center slice only → +1ch) ---
        if self._stage1_probs_dir is not None:
            if self._cache_probs_vidx == int(vidx) and self._cache_probs_zyx is not None:
                probs_zyx = self._cache_probs_zyx
            else:
                npz_path = self._stage1_probs_dir / f"{case_id}.npz"
                if npz_path.exists():
                    data_npz = np.load(str(npz_path))
                    probs_zyx = np.clip(data_npz["probs"].astype(np.float32), 0.0, 1.0)
                else:
                    probs_zyx = np.zeros(
                        (img3d.shape[1], img3d.shape[2], img3d.shape[3]), dtype=np.float32
                    )
                self._cache_probs_vidx = int(vidx)
                self._cache_probs_zyx = probs_zyx
            prob_slice = probs_zyx[int(np.clip(z, 0, probs_zyx.shape[0] - 1))]  # (Y, X)
            img2_5d = np.concatenate([img2_5d, prob_slice[np.newaxis]], axis=0)
        # --- END Stage1 prob ---

        sample = {"image": img2_5d, "mask": mask2d, "case_id": case_id, "z": z}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
