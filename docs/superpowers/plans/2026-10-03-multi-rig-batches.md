# Multi-rig batch workflow

Execute inline under the active full-coverage goal.

Design: Combine UniMate Rigs collects two execution lists in order, validates
portable rig values and emits one typed rig execution list. Collections can be
chained. Generate UniMate Batch consumes the entire rig list once; model, prompts,
repetitions, seed, guidance and normalization each require one control value.
Case order stays rig → prompt → repetition, with incrementing uint64 seeds.
Motion remains output slot 0; matching rigs occupy output slot 1. Both are list
outputs, so exports receive the correct rig for each case. Existing single-rig
workflows retain motion slot 0 and their export behavior.

Files: unimate_pack/batch.py owns paired case expansion; nodes.py translates V3
execution lists and collection validation; __init__.py registers the collection.
Tests use actual V3 schemas and ordered sampling fixtures. The verifier builds
five- and seven-joint synthetic rigs and checks every exported case's identity,
seed, prompt and independent playback. Worker coverage must exercise both paired
lists and reference the exact client/runtime envelope revisions.

- [x] Add failing paired-case, collection/schema and control validation tests.
- [x] Implement collection and paired batch outputs; check current regressions.
- [x] Add an explicit API workflow example and update docs without overclaiming.
- [x] Run local multi-topology model workflow with independent playback checks.
- [ ] Run headless stadia and worker boundary workflows; verify every case.
- [ ] Review and push completed changes directly to main.

All other full-goal capabilities and gates remain in scope.

