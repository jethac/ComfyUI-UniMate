# Full UniMate coverage implementation plan

> Execute inline using the executing-plans and test-driven-development skills. Track evidence in COVERAGE.md. This plan implements GOAL.md; it does not reduce its scope.

**Goal:** Complete the node and verification coverage listed in GOAL.md and COVERAGE.md.

**Architecture:** Keep portable rig/model/motion dictionaries and Comfy-managed inference. Add separate motion IO, constrained sampling, collection, visualization, training and preprocessing adapters. Preserve original assets while extending export paths.

**Tech stack:** Python, NumPy, PyTorch, pinned UniMate, ComfyUI extension API, external Blender.

**Constraints:** Offline runtime; no forced torch replacement; safe numeric archives; preserved appearance and coordinate mappings; portable Cloud Offload values; cancellation and cleanup; direct default-branch pushes; factual documentation.

## Stages

- [ ] Motion contracts: update `unimate_pack/contracts.py` and export/playback paths to accept bounded variable-length sequences. Test short clips, long expansion clips, empty/oversized sequences and legacy 60-frame compatibility in `tests/test_contracts.py`.
- [ ] Motion IO: add `unimate_pack/motion_io.py` and node schemas in `nodes.py` for numeric feature save/load and source-asset motion extraction. Preserve rig identity and canonical basis; test rotated/scaled rest poses, incompatible rigs and finite data before exposing reference inputs.
- [ ] Constrained sampling: add `unimate_pack/constrained.py`, adapt `inference.py`, and add Edit/In-between/Expand nodes. Compare fixed-noise replacement/Euler behavior against the released reference, including signed indices, joint names, variable valid lengths and overlap seams. Test cancellation during each segment.
- [ ] Model families: replace exact single-config reconstruction in `upstream.py` with validated config-driven architecture selection; update `bundle.py` and `tools/build_bundle.py`. Load actual released checkpoints for every family, with matching text/statistics paths and numerical reference comparisons.
- [ ] Collections: expose repetitions, batching and shared-prompt target-rig workflows with typed portable collections. Verify seeds and bounded memory behavior independently of chunk size.
- [ ] Preprocessing and export: expose conditioning/canonical asset outputs, GLB/FBX export, FK/RIC previews. Verify skinning, textures, frame counts and coordinates through Blender and independent evaluators.
- [ ] Training and data utilities: expose isolated configuration/dataset/job/checkpoint contracts for released training and preprocessing tooling. Verify execution, resume, progress, cancellation and output artifacts; require explicit setup for external annotation services.
- [ ] Paper-described methods: locate or implement foot locking and deformation options. Compare contact detection and IK behavior with the paper's specified method; distinguish renderer limitations from motion capability.
- [ ] Cloud and integration: extend portable-value inventories and staging as needed. Run every inference mode headlessly on stadia-testbed, compare kept constraints and playback, and verify actual transport/output retrieval.
- [ ] Completion audit: inspect implementation and validation evidence for every GOAL.md/COVERAGE.md row. Update documentation, commit and push completed stages. Do not mark the goal achieved while any required capability remains missing or unverified.

Each code stage follows: write a meaningful failing test, observe the expected failure, implement, run relevant checks, review the resulting diff and record evidence. Existing tests remain regression coverage; they cannot substitute for the expanded end-to-end checks.
