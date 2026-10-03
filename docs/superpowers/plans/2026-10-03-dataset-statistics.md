# Dataset Statistics Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement inline.

**Goal:** Match every released statistics mode on safe numeric training clips.

**Architecture:** A NumPy-only module computes root/local statistics from clip
records. Pinned source comparisons independently verify all three option axes.
Later dataset nodes consume this module; no training imports enter ComfyUI.

**Tech Stack:** Python, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-dataset-statistics.md`.

## Global constraints

- No pickle, automatic downloads or model/device changes.
- Reference revision `2c5b384715aa63d8639b1ed7eb74bfe614570c7a`.
- Float64 output vectors of width 12; standard-deviation floor 1e-8.
- Preserve global object-name grouping and all source arrays.
- Direct default-branch work is authorized by the user.

## Review focus

- Identical object names in different datasets retain released grouping.
- Root/local samples use different counts when joint counts differ.
- Constant channels have positive standard deviations.
- Invalid clips/options fail before expensive accumulation.
- Cancellation propagates without input mutation.

### Task 1: Numeric statistics and pinned comparisons

Files: create `unimate_pack/dataset_stats.py`, `tests/test_dataset_stats.py`.

Interface: `compute_statistics(clips, *, per_dataset=True, balanced=False,
tie_std=False, cancel=None) -> dict`. Each clip has exactly `dataset_type`,
`object_type`, `features`; the features are finite floating `(T,J,12)` arrays.
Output fields are specified in the spec. Total input arrays are bounded by
`contracts.MAX_ARRAY_BYTES`; working-array estimates must reject oversized
inputs before conversion. Cancellation receives no arguments and may raise.

- [x] Write tests for eight source comparisons, constant-channel floors,
  unequal joint counts, duplicate object labels, input mutation, bad records,
  overflow and cancellation. Extract original method ASTs only from the pinned
  source file with SHA-256 `413a539e4ad4606e599c502758f9d1a199e5750feb44eebccb28f9845ca6961e`.
- [x] Run tests and observe missing implementation.
- [x] Implement validation and bounded float64 moment accumulation, source
  variance formulas, grouping, tying and copied global results.
- [x] Run the complete new test file with the pinned checkout configured;
  require no skipped reference checks. Run changed-file Ruff and diff checks.
- [x] Obtain independent review; fix substantive findings with regressions.
- [ ] Record evidence and remaining public integration gaps; commit and push.

Public dataset/statistics nodes and training consumption require subsequent
plans under the full training/dataset audit. This plan closes only the numeric
component when its evidence passes.

Execution record: initial tests failed at the missing module. Float32 reference
cases then exposed premature upcasting; retaining source reduction precision
passed all modes. A retained-group workspace regression failed, then passed
after expanding the estimate. Independent review found masked-array validation
bypass; its regression failed, then passed after requiring ordinary arrays.
The reference fixture now checks both checkout revision and source digest.
No public dataset/statistics node, training job or cloud support is claimed.
