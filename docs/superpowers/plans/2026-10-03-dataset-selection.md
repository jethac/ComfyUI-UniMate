# Dataset Selection Implementation Plan

Use superpowers:executing-plans inline. Design:
`docs/superpowers/specs/2026-10-03-dataset-selection.md`.

## Task 1: Reference-verified selection adapters

Create `unimate_pack/dataset_selection.py`, `tests/test_dataset_selection.py`.
- [x] RED split membership/order, override/missing/draining, immutable bytes,
  invalid options and cancellation tests against unchanged pinned AST methods.
- [x] Implement splitting over portable dataset metadata.
- [x] RED float64 weights and exact epoch indices for single/two-level shared
  object labels, alpha values/epochs; malformed portable plan/identity tests.
- [x] Implement sampling plan and validation; reuse statistics source identity.
- [x] Run foundation/reference suites and Ruff; independent final review.

## Task 2: Public nodes and cloud workflow

Modify dataset_nodes.py, nodes.py, __init__.py, node tests and dataset_workflow.py.
- [x] RED real V3 schemas, adapter plumbing and registration/portable transport.
- [x] Implement Split/Plan nodes with portable Dataset/Sampling sockets.
- [x] Extend real workflow harness to exercise nodes and plan capture/restore.
- [x] Run local and headless stadia; confirm split IDs and weights/indices.
- [x] Review; update README/DESIGN/COVERAGE/VALIDATION; commit/push main.

Dependencies: Task2 consumes Task1 dataset/report and sampling-plan schema.
Dataset archives preserve original bytes; selection changes only membership.
Training, augmentation and batch consumption stay open after this plan.

Execution record: 265 tests, six subtests, no skips; Ruff passed. Windows and
stadia direct/three-job workflows each restored six archives and four sampling
plans. Review found one unmatched-dataset report gap; RED/GREEN regression fixed
it, and follow-up review confirmed resolution. Reference uses unchanged AST
bodies with legacy no-op Sampler base constructor for current Torch.
