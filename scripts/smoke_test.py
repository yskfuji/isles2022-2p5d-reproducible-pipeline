#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

REPO_NAME = "isles2022-2p5d-reproducible-pipeline"
REQUIRED_PATHS = [
    "README.md",
    "sample_manifest.json",
    "isles2022_25d/README.md",
    "isles2022_25d/README_en.md",
    "core/pipeline/README.md",
    "core/pipeline/README_en.md",
    "core/pipeline/tools/make_manifest.py",
    "core/pipeline/src/models/convnext_nnunet_seg.py",
    "core/pipeline/src/training/train_isles_25d_convnext_fpn.py",
    "core/pipeline/src/evaluation/evaluate_isles_25d.py",
]
ENTRYPOINTS = [
    "python -m src.training.train_isles_25d_convnext_fpn",
    "python -m src.evaluation.evaluate_isles_25d",
    "python -m src.evaluation.evaluate_isles_25d_ensemble",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _check_paths(root: Path) -> list[str]:
    missing: list[str] = []
    for rel in REQUIRED_PATHS:
        if not (root / rel).exists():
            missing.append(rel)
    return missing


def _load_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "sample_manifest.json").read_text(encoding="utf-8"))


def _run_manifest(root: Path, out_dir: Path) -> Path:
    out_path = out_dir / "MANIFEST.sha256.txt"
    cmd = [
        sys.executable,
        str(root / "core/pipeline/tools/make_manifest.py"),
        "--root",
        str(root),
        "--out",
        str(out_path.relative_to(root)),
        "--exclude",
        "artifacts/smoke_dummy/**",
    ]
    subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a no-data smoke test for the public ISLES 2.5D repository.")
    parser.add_argument("--use_dummy_data", action="store_true", help="Load the bundled dummy manifest and generate a smoke-test summary.")
    args = parser.parse_args()

    root = _repo_root()
    missing = _check_paths(root)
    if missing:
        print("Missing required files:", *missing, sep="\n- ")
        return 1

    out_dir = root / "artifacts" / "smoke_dummy"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = _load_manifest(root) if args.use_dummy_data else {"sample_cases": []}
    manifest_out = _run_manifest(root, out_dir)

    summary: dict[str, Any] = {
        "repo": REPO_NAME,
        "status": "passed",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "used_dummy_data": bool(args.use_dummy_data),
        "checked_paths": REQUIRED_PATHS,
        "entrypoints": ENTRYPOINTS,
        "sample_cases": manifest.get("sample_cases", []),
        "generated_files": [
            str(manifest_out.relative_to(root)),
            "artifacts/smoke_dummy/summary.json",
        ],
        "notes": [
            "This smoke test validates repository wiring without bundling protected medical data.",
            "Use isles2022_25d/README_en.md for full training and evaluation commands.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[{REPO_NAME}] smoke test passed")
    print(f"- dummy data mode: {args.use_dummy_data}")
    print(f"- manifest: {manifest_out.relative_to(root)}")
    print("- next: open README.md for quickstart or isles2022_25d/README_en.md for full reproduction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
