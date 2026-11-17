---
name: Add benchmark summary table
about: Publish a compact summary of core ISLES 2.5D segmentation results near the top-level docs.
title: "add benchmark summary table"
labels: ["documentation", "enhancement"]
---

## Goal
Add a concise benchmark table that helps external reviewers understand segmentation quality quickly.

## Suggested rows
- mean Dice (val / test)
- lesion-wise detection rate / FP rate
- per-size Dice breakdown (small / medium / large lesions)

## Definition of done
The root page and detailed docs show a consistent metrics summary.
