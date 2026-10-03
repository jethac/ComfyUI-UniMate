# Dataset Contracts Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline.

**Goal:** Safe content-addressed numeric dataset shards for public training nodes.

**Architecture:** `dataset_contracts.py` owns validation and train-record
extraction; `dataset_io.py` owns archive serialization. Reuse safe numeric
archive primitives. Node and Cloud Offload adapters follow after these contracts
are verified; keep their coverage gates open.

**Tech Stack:** Python, NumPy, ZIP/JSON, pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-dataset-contracts.md`.

## Global constraints

- No pickle, filesystem extraction, automatic downloads or mesh-limit changes.
- Shard payload/expanded budget 256 MiB; manifest budget 4 MiB.
- Maximum 4096 clips/topologies, 8192 payload files; 2–4096 joints.
- Features `(T>=1,J,12)` float16/32/64; FPS 30; exact field sets.
- Direct default-branch work is authorized.

## Review focus

- Evaluation clips never enter normalization.
- Digest-valid payloads with mismatched topology fail.
- Unicode captions and original numeric precision survive archives/transport.
- Repeated payloads deduplicate without conflating clip labels/splits.
- Aggregate archive budgets hold across many individually valid members.

### Task 1: Shard validation and statistics extraction

Create `unimate_pack/dataset_contracts.py`, `tests/test_dataset_contracts.py`.
Interfaces: `make_dataset(manifest: dict, files: dict[str, bytes]) -> dict`,
`validate_dataset(value: dict) -> None`, `training_records(value: dict) -> list`.
The spec fixes fields; output records contain dataset_type/object_type/features.

- [x] Write roundtrip/dedup/Unicode, >70-joint and train/eval isolation tests.
  Add digest tampering, unknown fields, bad topology, mismatched feature width,
  empty training selection and manifest mutation tests. Observe missing feature.
- [x] Implement strict validation with declared and expanded aggregate bounds;
  preserve exact payload bytes, copied manifest and ordinary arrays.
- [x] Run the whole contract file and numeric-statistics suite with no missing
  reference comparisons. Run Ruff. Independently review and fix regressions.
- [x] Record evidence and commit.

### Task 2: Safe deterministic shard archives

Create `unimate_pack/dataset_io.py`, `tests/test_dataset_io.py`.
Interfaces: `dump_dataset(value: dict) -> bytes`, `load_dataset(payload: bytes) -> dict`.

- [x] Write deterministic, Unicode and feature-byte roundtrip tests, malformed
  manifests, duplicate/traversal ZIP members, digest tampering and aggregate
  expansion limits. Observe missing feature.
- [x] Implement canonical ZIP/JSON using existing archive checks; validate both
  directions and avoid filesystem extraction.
- [x] Run both dataset test files, reference statistics and Ruff; review.
- [ ] Record evidence, update coverage and push direct.

Public nodes/transport and collection-based training are the next integration
plans. No completion claim for them follows from these foundation tasks.

Execution: both new modules were absent in their initial test runs. Contract
checks then passed. Load cancellation and early outer aggregate-budget sentinel
tests failed, then passed after narrowing exception handling and adding declared
payload summation. Independent review found no blocking defects. Additional
precision, byte-order, joint-limit, nested object-array and later-stage
cancellation checks bring the combined suite to 85 tests, all passing with no
skipped reference comparisons. Ruff passed. No public/cloud node is claimed.
