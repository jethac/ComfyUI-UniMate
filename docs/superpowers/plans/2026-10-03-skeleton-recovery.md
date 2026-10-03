# Skeleton Recovery Implementation Plan

Execute inline with the executing-plans skill.

**Goal:** expose UniMate FK and RIC recovery and skeleton image previews.

**Architecture:** dependency-light numeric recovery consumes the existing validated rig/motion contracts. A versioned portable skeleton value carries canonical positions, parent indices, joint names, rig identity and recovery mode. A separate renderer produces ComfyUI IMAGE batches from that value. Recovery uses the same root integration as mesh animation; preview framing is fixed across the clip.

**Tech stack:** NumPy, Pillow, PyTorch at the node image boundary, current ComfyUI extension API.

**Spec:** GOAL.md skeleton previews and FK/RIC recovery requirement.

- [x] Add numerical reference tests for both modes on rotating branching motion, plus rig identity, invalid mode, cancellation and origin checks; observe failures.
- [x] Implement `unimate_pack/skeleton.py`: recovery and validated portable numeric value. Compare FK against upstream `recover_unimate_joint_pos_from_rot` using canonical `tpos_offsets`, and RIC against `recover_unimate_joint_pos_from_ric`.
- [x] Add `UniMateRecoverSkeleton` and `UniMatePreviewSkeleton` schemas, node wiring, renderer tests and extension registration. IMAGE batches include every frame and use a fixed projection and clip-wide bounds; enforce allocation limits before rendering.
- [ ] Verify node schemas, numerical references, concrete Cloud Offload serialization and server execution; then exercise headless stadia execution and retrieval.
- [ ] Update current coverage/docs with exact evidence and remaining gaps; commit and push tested changes.

Preserve offline behavior, canonical root origin, 30 fps, variable clip length and exact rig identity. Positions must remain finite. Do not vendor the unlicensed Motion reference used in tests. No new model or asset downloads.

Windows and stadia server execution, reference comparisons and codec round trips passed;
actual cloud runner execution remains pending. Implementation and evidence are pushed;
completion of the remaining cloud gate is still required.
