# Dataset Nodes Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline.

**Goal:** Public portable dataset/statistics nodes with verified file transport.

**Architecture:** Pure adapters own pair collection and statistics contracts;
nodes own V3 schemas/managed files. Reuse shard, numeric and cloud codecs.

**Tech Stack:** Python, NumPy, ComfyUI V3, existing Cloud Offload siblings.

**Spec:** `docs/superpowers/specs/2026-10-03-dataset-nodes.md`.

## Global constraints

- No pickle, downloads, host paths in values, provider calls or mesh-limit changes.
- Keep train/evaluation separation and exact source rig/motion validation.
- Source statistics revision `2c5b384715aa63d8639b1ed7eb74bfe614570c7a`.
- Direct default-branch work/push is authorized.

## Review focus

- Reused rig conditioning can contain non-rest offsets; training must derive its offsets.
- List-mode scalar configuration must be unambiguous, never silently take the first.
- Shared numeric payloads preserve distinct labels and membership.
- Statistics artifacts must roundtrip exact payload bytes, identity and precision.
- Cancellation precedes file publication and cleans staged output.

### Task 1: Pure public adapters

Create `dataset_builder.py`, `statistics.py`, `statistics_io.py`; tests for each.
Interfaces: `build_dataset(rigs,motions,labels,default_dataset,cancel=None)`;
`dataset_statistics(dataset,per_dataset=True,balanced=False,tie_std=False,cancel=None)`;
`validate_statistics(value)`; `dump_statistics(value)` / `load_statistics(bytes)`.

- [x] RED tests for paired identities/list lengths/labels, source bytes/origins,
  deduplication and cancellation; all statistics modes, source provenance,
  malformed contracts and lossless archives.
- [x] Implement pure adapters against existing shard/statistics contracts.
- [x] Run focused/reference checks and Ruff; independently review.

### Task 2: Six public nodes and managed files

Modify `nodes.py`, `__init__.py`, `tests/test_nodes.py`. Add V3 classes
UniMateBuildDataset, UniMateLoadDataset, UniMateSaveDataset,
UniMateDatasetStatistics, UniMateLoadStatistics, UniMateSaveStatistics.

- [x] RED schema/registration, real adapter plumbing, list-mode validation,
  filename containment, staging declarations, fingerprint and cancellation tests.
- [x] Implement schema/file adapters without training dependencies.
- [ ] Run node and foundation suites; review; document and commit.

### Task 3: Transport and workflows

Add `tools/dataset_workflow.py` and workflow checks using existing verifier.

- [x] Verify actual client/runner dictionary/bytes codec for both new types.
- [ ] Run local/headless stadia actual partition capture/restore, artifact save/
  reload and statistical comparison, recording all repo/runtime identities.
- [ ] Update README/DESIGN/COVERAGE/VALIDATION and push verified completed work.

Keep goal active while any task or broader training/dataset gate remains open.

Execution record: Tasks 1/2 implemented; 141 tests and 6 subtests pass; independent
node review found no blockers. Task 2 commit and Task 3 headless workflow gates
remain tracked separately. Expansion preflight regression tests failed before
the guard and passed afterward. No claim of training runtime coverage.
