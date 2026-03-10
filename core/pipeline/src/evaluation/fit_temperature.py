"""Fit scalar temperature for post-hoc calibration of the 2.5D ISLES model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer

from ..datasets.isles_dataset import IslesVolumeDataset
from ..training.utils_train import prepare_device
from .common_25d import infer_volume_logits, load_25d_model

app = typer.Typer(add_completion=False)


def _sample_indices(rng: np.random.Generator, idxs: np.ndarray, k: int) -> np.ndarray:
    if idxs.size <= k:
        return idxs
    return rng.choice(idxs, size=int(k), replace=False)


@app.command()
def main(
    model_path: str = typer.Option(..., help="model checkpoint"),
    csv_path: str = typer.Option(..., help="split csv"),
    root: str = typer.Option(..., help="processed root"),
    split: str = typer.Option("val", help="calibration split"),
    out_path: str = typer.Option("", help="output json path (default: next to checkpoint)"),
    normalize: str = typer.Option("fixed_nonzero_zscore"),
    stage1_probs_dir: str = typer.Option("", help="optional Stage1 probs dir"),
    allow_missing_label: bool = typer.Option(False),
    max_cases: int = typer.Option(0),
    max_pos_vox: int = typer.Option(20000),
    max_neg_vox: int = typer.Option(50000),
    seed: int = typer.Option(42),
    steps: int = typer.Option(200),
    lr: float = typer.Option(0.05),
    min_temp: float = typer.Option(0.05),
    max_temp: float = typer.Option(20.0),
    tta: bool = typer.Option(False),
):
    mp = Path(model_path).expanduser().resolve()
    device = prepare_device()
    info = load_25d_model(mp, device, stage1_probs_dir=(stage1_probs_dir or None))

    ds = IslesVolumeDataset(
        csv_path=csv_path,
        split=split,
        root=root,
        normalize=normalize,
        allow_missing_label=bool(allow_missing_label),
    )
    if len(ds) == 0:
        raise ValueError(f"No cases found for split={split!r}")

    rng = np.random.default_rng(int(seed))
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    n_use = len(ds) if int(max_cases) <= 0 else min(len(ds), int(max_cases))

    for i in range(n_use):
        sample = ds[i]
        vol = sample["image"].astype(np.float32)
        gt = (sample["mask"] > 0.5).astype(np.uint8)
        case_id = str(sample["case_id"])

        extra_vol = None
        s1_dir = info.get("stage1_probs_dir")
        if s1_dir:
            npz_path = Path(str(s1_dir)) / f"{case_id}.npz"
            if npz_path.exists():
                with np.load(str(npz_path)) as z:
                    extra_vol = z["probs"].astype(np.float32, copy=False)
            else:
                extra_vol = np.zeros((vol.shape[1], vol.shape[2], vol.shape[3]), dtype=np.float32)

        logits = infer_volume_logits(
            vol,
            info["model"],
            offsets=info["offsets"],
            img_size=info["img_size"],
            device=device,
            extra_vol=extra_vol,
            tta=bool(tta),
        )

        flat_logits = logits.reshape(-1)
        flat_gt = gt.reshape(-1)
        pos_idxs = np.flatnonzero(flat_gt > 0)
        neg_idxs = np.flatnonzero(flat_gt == 0)
        pos_sel = _sample_indices(rng, pos_idxs, int(max_pos_vox))
        neg_sel = _sample_indices(rng, neg_idxs, int(max_neg_vox))
        sel = np.concatenate([pos_sel, neg_sel], axis=0)
        if sel.size == 0:
            continue
        rng.shuffle(sel)
        logits_list.append(flat_logits[sel].astype(np.float32))
        labels_list.append(flat_gt[sel].astype(np.float32))
        print(f"[{i+1}/{n_use}] {case_id} pos={int(pos_sel.size)} neg={int(neg_sel.size)} sel={int(sel.size)}", flush=True)

    if not logits_list:
        raise RuntimeError("No voxels sampled; check data/labels")

    x = torch.from_numpy(np.concatenate(logits_list, axis=0)).float().cpu()
    y = torch.from_numpy(np.concatenate(labels_list, axis=0)).float().cpu()

    def nll(temp: torch.Tensor) -> torch.Tensor:
        t = temp.clamp(min=float(min_temp), max=float(max_temp))
        return torch.nn.functional.binary_cross_entropy_with_logits(x / t, y)

    log_t = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=float(lr))

    with torch.no_grad():
        loss_before = float(nll(torch.tensor(1.0)).item())

    best_loss = None
    best_t = None
    for step_idx in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        temp = torch.exp(log_t)
        loss = nll(temp)
        loss.backward()
        opt.step()
        with torch.no_grad():
            t_val = float(torch.exp(log_t).clamp(min=float(min_temp), max=float(max_temp)).item())
            l_val = float(loss.item())
        if best_loss is None or l_val < best_loss:
            best_loss = l_val
            best_t = t_val
        if (step_idx + 1) % 20 == 0 or step_idx == 0:
            print(f"step {step_idx+1}/{steps} T={t_val:.4f} nll={l_val:.6f}", flush=True)

    payload = {
        "model_path": str(mp),
        "csv_path": str(Path(csv_path)),
        "root": str(Path(root)),
        "split": str(split),
        "normalize": str(normalize),
        "max_cases": int(max_cases),
        "max_pos_vox": int(max_pos_vox),
        "max_neg_vox": int(max_neg_vox),
        "seed": int(seed),
        "n_samples": int(x.numel()),
        "pos_frac": float(y.mean().item()),
        "temperature": float(best_t),
        "nll_before": float(loss_before),
        "nll_after": float(best_loss),
    }

    out_p = Path(out_path).expanduser().resolve() if str(out_path).strip() else (mp.parent / "temperature_best.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {out_p} (T={float(best_t):.4f})", flush=True)


if __name__ == "__main__":
    app()
