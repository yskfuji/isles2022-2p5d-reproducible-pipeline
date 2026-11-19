#!/usr/bin/env python3
"""Fail if a relative import points at a missing module.

Publication dropped source packages when a case-insensitive ``Datasets/``
ignore rule matched them. This check does not import third-party packages.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def missing_relative_imports(root: Path) -> list[tuple[str, str]]:
    bad: list[tuple[str, str]] = []
    for path in root.rglob("*.py"):
        if ".git" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            bad.append((str(path.relative_to(root)), f"syntax: {exc}"))
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level or not node.module:
                continue
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            target = base.joinpath(*node.module.split("."))
            if not (target.with_suffix(".py").exists() or target.is_dir()):
                bad.append((str(path.relative_to(root)), node.module))
    return bad


def main() -> int:
    bad = missing_relative_imports(ROOT)
    print(f"unresolved relative imports: {len(bad)}")
    for path, module in bad[:20]:
        print(f"FAILED {path}: {module}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
