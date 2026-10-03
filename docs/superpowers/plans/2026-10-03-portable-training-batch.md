# Portable training batch implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task.

**Goal:** Carry released collated motion/conditioning through ComfyUI and Cloud Offload.
**Architecture:** Wrap the existing pinned collator. Store numeric arrays and ordered sample provenance; validate by reconstructing numeric samples and checking source collation invariants. Restore CPU tensors at the training boundary.
**Tech Stack:** NumPy, Torch CPU, numeric NPZ, ComfyUI V3.
**Spec:** GOAL.md and DESIGN.md training dataset foundations.

## Global constraints

- No pickle, downloads, model weights or live tensors in portable values.
- Preserve source layouts, masks, ordered topology and standard-deviation padding.
- Bound archives and aggregate workspace before decoding/allocation; check cancellation.
- Push reviewed completed work directly to the default branch.

## Review focus

Mixed joint counts; ragged captions; forged padding/masks; excessive aggregate inputs; repeated sample identities must preserve ordering.

### Task 1: Portable contract

Files: create `unimate_pack/training_batch_contracts.py`, `tests/test_training_batch_contracts.py`.
Interfaces: `collate_training_samples(values, *, cancel=None, max_workspace_bytes=512*1024*1024)` returns portable `unimate.training_batch.v1`; `validate_training_batch(value, *, cancel=None, max_workspace_bytes=512*1024*1024)` returns source-format `(motion_tensor, cond_dict)` on CPU.

- [x] Write tests comparing restored fields/dtypes with pinned source, ownership, ordering, malformed digests/shapes/masks/padding, predecode budget and cancellation.
- [x] Observe failure, implement contract and run focused regression tests.

### Task 2: Public node and transport

Files: modify `training_nodes.py`, `nodes.py`, `__init__.py`, `tests/test_training_nodes.py`, README.md, DESIGN.md, COVERAGE.md, VALIDATION.md.
Interface: `UniMateCollateTrainingSamples`, list input `UNIMATE_TRAINING_SAMPLE`, scalar workspace budget, output `UNIMATE_TRAINING_BATCH` plus provenance JSON.

- [x] Observe missing-node/schema test failure; implement registration and list collection.
- [x] Check actual client/runner codec round trips and source tensor restoration.
- [x] Run regression checks, review, document precise evidence.
- [x] Commit and push (`7cc37f4`).
- [x] Track server/headless batch workflows and actual training consumption as open until executed.

Ruling: pack topology into two numeric arrays plus per-sample parent dtype metadata,
rather than one pair of archive members per sample. This preserves original parent
dtypes and supports the existing 4096-sample bound without exceeding the numeric
archive's 128-member limit. Validation rechecks representability before casting.

Evidence: 16 contract tests; full suite 903 passed, 50 skipped, six subtests.
Review found no actionable findings. Full-suite failures in the initial run were
the stale expected node inventory and a temporary inventory cache leaked by the
test fixture; both corrected before the successful run.

Follow-up: Windows and headless stadia batch workflows now pass direct collection
and three actual partition-handler jobs using 18 model-encoded samples. Restored
batch tensors and reversed order match; training consumption remains open.
