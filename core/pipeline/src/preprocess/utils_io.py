"""IO utilities for medical volume preprocessing.

Centralizes NIfTI I/O for the public 2.5D pipeline.
"""
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np


def load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data, img


def save_nifti(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: str) -> None:
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header), str(out_path_p))


def save_nifti_with_affine(
    data: np.ndarray,
    affine: np.ndarray,
    out_path: str,
    *,
    header: Optional[nib.Nifti1Header] = None,
) -> None:
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    hdr = header.copy() if header is not None else None
    nib.save(nib.Nifti1Image(data, affine=np.asarray(affine, dtype=np.float64), header=hdr), str(out_path_p))
