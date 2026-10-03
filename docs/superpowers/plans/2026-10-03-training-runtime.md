# Training runtime implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline.

**Goal:** Execute released training on portable datasets/batches and emit resumable and inference state.
**Architecture:** Keep pinned training math behind validated batch/config/state boundaries. Isolate kernels, optimizer sessions, numeric state serialization and public workflow nodes.
**Tech Stack:** Torch, NumPy, ComfyUI model management, safetensors, JSON.
**Spec:** docs/superpowers/specs/2026-10-03-training-runtime.md.

## Global constraints

- No pickle, downloads, telemetry, automatic providers or Motion runtime imports.
- Preserve pinned math; retain UniMate, SiT and guided-diffusion notices.
- Never select hardware independently of ComfyUI or put live state on sockets.
- Validate identities/shapes/budgets before allocation or model mutation.
- Full training remains open until public nodes and local/headless/worker evidence pass.

## Review focus

Zero valid lengths; auxiliary-loss path incompatibility; RNG leakage; malformed resume state; partial optimizer updates during cancellation.

### Task 1: Flow and EMA kernel

Files: `_vendor/flow/{transport,path,integrators}.py`, `_vendor/training_math.py`, `_vendor/ema.py`, `training_loss.py`, `tests/test_training_loss.py`, `_vendor/SOURCES.json` and license notices.
Interfaces: `create_flow_schedule(options)` validates explicit options and returns pinned transport; `flow_training_loss(model, portable_batch, *, options=None, seed=0, cancel=None, max_workspace_bytes=512*1024*1024)` returns differentiable scalar and float metrics. Model mode/device are caller-owned; seed scope restores process RNG state.

- [x] Observe failing source/reference/gradient/EMA tests.
- [x] Vendor licensed math with recorded hashes; add validation and batch/device adapter.
- [x] Compare all paths, parameterizations and weightings, auxiliary losses and gradients; test invalid input and cancellation/RNG restoration.
- [x] Run suite, review and record exact scope.
- [ ] Commit/push.

### Subsequent required tasks

- [ ] Diffusion schedule/loss kernel and reference comparisons.
- [ ] AdamW, LR schedule, EMA, accumulation/precision/distributed sessions and exact resume comparisons.
- [ ] Bounded portable model/optimizer/scheduler/EMA/RNG/data-position checkpoint contracts.
- [ ] Model architecture/job configuration and public training/progress/checkpoint nodes.
- [ ] Installed-model batch/backprop/update, exported checkpoint inference and independent playback.
- [ ] Windows/headless stadia/actual-worker resume, cancellation, cleanup and artifact retrieval.

Keep subsequent tasks detailed before implementing them. Their unchecked status
does not exempt them from GOAL.md.

Ruling: add explicit stable/released geodesic policies. Unchanged source conversion
creates NaNs for masked/degenerate rotations; stable substitutes identity before
source geodesic math. Valid rotation math remains exact. The released policy keeps
the source failure observable; nonfinite outputs are rejected before updates.

Evidence: full suite 961 passed, 50 skipped, six subtests. Four small backbone
updates and caller-selected CUDA passed; no public or installed-checkpoint
training claim. Review's reduced-mean overflow finding fixed through RED/GREEN.
