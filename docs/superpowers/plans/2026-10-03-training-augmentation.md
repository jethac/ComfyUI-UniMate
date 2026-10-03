# Training Augmentation Implementation Plan

Use superpowers:executing-plans inline. Spec:
`docs/superpowers/specs/2026-10-03-training-augmentation.md`.

## Task 1: Reference fixture and matrix recomputation

- [x] Create tests/test_training_augmentation.py with pinned source/file hashes,
  reference-only Motion checkout and nontrivial branching/chain samples.
- [x] RED FK/RIFKE/rest comparisons, precision, immutability and RNG isolation.
- [x] Implement training_augmentation.py matrix recomputation/validation layer.

## Task 2: Released operations and wrapper

- [x] RED all five operations plus no-op and released randomized wrapper,
  embedding/stat propagation, weighted choices, rates and budgets/cancellation.
- [x] Implement source-compatible operations using local random generators.
- [x] Run reference/foundation checks and Ruff; review complete changes.
- [x] Document exact evidence and remaining public/sample/text-cache gaps;
  commit/push directly to main. Keep full goal active.

Interfaces: task2 consumes task1 validated source-compatible aug dictionaries.
No public UNIMATE_TRAINING_SAMPLE socket is invented before its sample/text-cache
producer and full contract exist. Existing source lineage/license notices remain.

