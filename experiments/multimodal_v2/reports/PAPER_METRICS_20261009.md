# V2 paper evaluation implementation — 2026-10-09

Base: refactor-v2-standalone, 505abdcb81bf993630ca422a32257ddf5133b53e.

## Changes

- paper_metrics.py: conditional Top1 XYZ/3D RMSE, coverage and full-denominator
  Success; optional strict day/night grouping; official image-level COCO metrics;
  CSV suitable for a paper table.
- evaluate.py: retain existing summary and per-query fields, enrich records with
  GT/predictions and image identity, export paper JSON/CSV and COCO audit files.
- training.py: log validation XYZ/3D RMSE and its output count. Loss, architecture,
  initialization and checkpoint monitoring remain unchanged.
- export_paper_metrics.py: regenerate tables offline from enriched records.
- Dependencies: add pycocotools in pyproject.toml and requirements.txt.
- Tests and README: definitions, commands, unsupported/undefined metric policy.

## Verification

CPU environment: Python 3.12, torch 2.14.1+cpu, torchvision 0.29.1+cpu,
Lightning 2.6.6, pycocotools 2.0.11. This is not the server CUDA environment.

V2 tests: 110 passed, 1 skipped (existing optional test), 4 existing attention
mask deprecation warnings. Tests exercise axis/3D RMSE identity, missing-output
counts, pooled vs macro RMSE, strict labels, deterministic deduplication,
official COCO perfect/empty/false-positive cases and B1 null 3D metrics.

Offline CLI smoke: two synthetic query records selecting one image produced
one unique COCO image, AP=1, B1 localization=null, JSON/CSV and COCO exports.
Evaluation --help, Python compilation and git diff --check passed.
Patch package is checked against the base tree, with before/after file copies.

## Unverified or deliberately unsupported

- No real MMAUD model inference or CUDA evaluation was run in this environment.
  Re-evaluate the existing best checkpoints on the server for actual metrics.
- Axis physical directions remain to be verified against the GT coordinate frame.
- No reliable day/night metadata was supplied; these columns are blank by default.
- No bandwidth measurement was supplied; this column is blank by default.
- COCO AP evaluates only images with valid boxes, using final V/RV candidates
  (usually at most 10/image), not all raw detector queries. Empty annotation files
  are not asserted to be verified negative images. This scope must appear in the
  paper alongside image deduplication and split definitions.
- Conditional RMSE excludes missing outputs; coverage and Success expose them.
  Mean Euclidean error and pooled RMSE are different statistics. Day/night macro
  average is explicitly separate from overall pooled RMSE.
- New metrics do not establish comparability with papers using different splits,
  axis conventions, missing-output handling or candidate/image evaluation scopes.
