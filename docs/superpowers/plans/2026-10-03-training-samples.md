# Training samples Implementation Plan

Use superpowers:executing-plans inline.
Spec: ../specs/2026-10-03-training-samples.md.
Goal: source-compatible encoded samples and usable portable batch workflows.
Architecture: numeric transforms, encoder/cache values, sample production,
collation and thin public nodes. Python/NumPy/Torch and existing V3 runtime.

## Task 1: Numeric transforms
- [x] Write pinned unchanged-source comparisons for crop modes, explicit short
  tails, normalization, padding dtype and parent-copy features; observe RED.
- [x] Implement validated numeric transforms with local RNG/cancellation.
- [x] Verify exact source parity and negative cases; retain MIT lineage.

## Task 2: Facing realignment and complete sample assembly
- [ ] Compare nonidentity quaternion realignment with reference-only Motion.
- [ ] Assemble augmented samples in source order, preserving identities/stats.

## Task 3: Encoder/cache and portable values
- [ ] Produce actual source-compatible embedding views with installed runtime.
- [ ] Add bounded numeric cache/sample archives and identity validation.

## Task 4: Batch collation and public workflows
- [ ] Compare unchanged source collation across variable joints/spectral/captions.
- [ ] Add V3 cache/sample/augmentation/batch nodes and transport contracts.
- [ ] Run installed encoder and Windows/stadia direct plus worker workflows.

All tasks require malformed input, budgets and cancellation checks, evidence in
VALIDATION.md and accurate COVERAGE.md. No fake embeddings or pickle. Preserve
full GOAL.md, including training runtime and all released encoder/model families.

