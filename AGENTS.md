# AGENTS.md

This file is the canonical agent-guidance entry point for this repo. `CLAUDE.md` references it — edit here, not there.

## Project

Research codebase for low-altitude tiny-UAV multimodal detection and localization on the MMAUD dataset. Python, PyTorch. Current active branch: `lidar-uav-v2`.

Key constraints:

- `src/rdq_uav/lidar_v2/` is audited and must be reused read-only; the multimodal work lives in a new `src/rdq_uav/multimodal_v1/` package (YAML + Registry driven).
- The frozen experiment plan `小目标雷达多模态_冻结版实验方案_V1.docx` (see `docs/agents/domain.md`) is the authoritative V1 design baseline; deviations must be recorded explicitly.
- Environment: WSL2, RTX 3070 8GB, 12GB RAM cap — batch size, fp16, and memory-aware processing are real constraints, not optimizations.

## Agent skills

### Issue tracker

Issues and specs live as local markdown under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
