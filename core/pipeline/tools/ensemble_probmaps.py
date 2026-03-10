"""Average multiple per-case probability-map directories into one ensemble directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-dirs", nargs="+", required=True, help="Input dirs containing <case_id>.npz")
    ap.add_argument("--out-probs-dir", required=True, help="Output directory for averaged <case_id>.npz")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    in_dirs = [Path(p).expanduser().resolve() for p in args.probs_dirs]
    for d in in_dirs:
        if not d.exists():
            raise FileNotFoundError(str(d))

    out_dir = Path(args.out_probs_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case_paths = sorted(in_dirs[0].glob("*.npz"))
    if not case_paths:
        raise RuntimeError(f"No npz files found in {in_dirs[0]}")

    out_dtype = np.float16 if args.dtype == "float16" else np.float32
    n = 0
    for p0 in case_paths:
        case_id = p0.stem
        arrays = []
        shape = None
        for d in in_dirs:
            p = d / f"{case_id}.npz"
            if not p.exists():
                raise FileNotFoundError(f"Missing {case_id}.npz in {d}")
            with np.load(str(p)) as z:
                arr = z["probs"].astype(np.float32, copy=False)
            if shape is None:
                shape = arr.shape
            elif arr.shape != shape:
                raise ValueError(f"Shape mismatch for case {case_id}: {arr.shape} vs {shape}")
            arrays.append(arr)
        avg = np.mean(np.stack(arrays, axis=0), axis=0).astype(out_dtype, copy=False)
        np.savez_compressed(str(out_dir / f"{case_id}.npz"), probs=avg)
        n += 1

    print(f"[done] wrote {n} cases to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
