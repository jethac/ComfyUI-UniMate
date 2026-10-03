# Training runtime implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline.

**Goal:** Execute released training on portable datasets/batches and emit resumable and inference state.
**Architecture:** Keep pinned training math behind validated batch/config/state boundaries. Isolate kernels, optimizer sessions, numeric state serialization and public workflow nodes.
**Tech Stack:** Torch, NumPy, ComfyUI model management, safetensors, JSON.
**Spec:** docs/superpowers/specs/2026-10-03-training-runtime.md.

## Global constraints

- No pickle, downloads, telemetry, automatic providers or Motion runtime imports.
- Preserve pinned math; retain UniMate, SiT and guided-diffusion notices.
- Never select hardware independently of ComfyUI or put live state on sockets.
- Validate identities/shapes/budgets before allocation or model mutation.
- Full training remains open until public nodes and local/headless/worker evidence pass.

## Review focus

Zero valid lengths; auxiliary-loss path incompatibility; RNG leakage; malformed resume state; partial optimizer updates during cancellation.

### Task 1: Flow and EMA kernel

Files: `_vendor/flow/{transport,path,integrators}.py`, `_vendor/training_math.py`, `_vendor/ema.py`, `training_loss.py`, `tests/test_training_loss.py`, `_vendor/SOURCES.json` and license notices.
Interfaces: `create_flow_schedule(options)` validates explicit options and returns pinned transport; `flow_training_loss(model, portable_batch, *, options=None, seed=0, cancel=None, max_workspace_bytes=512*1024*1024)` returns differentiable scalar and float metrics. Model mode/device are caller-owned; seed scope restores process RNG state.

- [x] Observe failing source/reference/gradient/EMA tests.
- [x] Vendor licensed math with recorded hashes; add validation and batch/device adapter.
- [x] Compare all paths, parameterizations and weightings, auxiliary losses and gradients; test invalid input and cancellation/RNG restoration.
- [x] Run suite, review and record exact scope.
- [x] Commit/push (`a5b0ba0`).

### Task 2: Diffusion kernel

Vendor pinned Gaussian diffusion, respacing and probability losses with retained
MIT notices. `create_diffusion_schedule(options)` reconstructs the released
factory. `diffusion_training_loss` follows the flow kernel's portable batch,
caller-owned device/mode, scoped RNG, cancellation and finite-loss boundaries.
Compare masked losses and gradients against independently loaded pinned source.
Test timestep mapping, rescaling, variance training, stable geodesic handling,
invalid options and RNG restoration before documenting or committing.

Ruling: default `timestep_policy=mapped` wraps respaced training inputs; explicit
`released` reproduces the omitted upstream training wrapper. Default
`variance_policy=trained` adds the computed variational bound to learned-variance
MSE; explicit `released` reproduces the omitted upstream objective term. These
corrections affect training trajectories and must remain recorded in job state.

### Task 3: Optimizer session

`TrainingSession(model, *, paradigm, loss_options, options, seed)` owns AdamW,
source warmup/cosine-min LR and EMA. The caller selects model/device; float32
parameters support none/bf16 autocast and CUDA fp16 scaling. `step(batches)`
executes an explicit accumulation group, divides each loss by configured
accumulation, clips after backward/unscale, updates AdamW → scheduler → EMA.
Partial final groups require an explicit flag and keep the source divisor.

Serialize execution with inference's model/RNG lock. Keep an advancing private
CPU/selected-CUDA RNG stream while preserving process streams. Snapshot model,
optimizer, LR, EMA, scaler, RNG and batch/update position at group entry; rollback
on cancellation, invalid loss/gradient or update failure. State-copy budget
preflight precedes cloning. Internal snapshots support exact uninterrupted vs
restore tests; portable untrusted checkpoint validation is Task 4 and must not
be claimed here. Compare source optimizer/scheduler/EMA update order and values;
exercise partial groups, rollback, RNG, clipping and CPU/CUDA precision.

Task 3 foundation is single-process. Distributed reduction/skip/accumulation,
public session ownership and ComfyUI memory-management integration remain
mandatory before full training coverage.

### Task 4: Portable checkpoint contract

`make_training_checkpoint(session, binding, position)` emits
`unimate.training_checkpoint.v1`: bounded JSON tensor-reference tree and
safetensors bytes, source/runtime identities and content digests. Binding records
model class/config/initial-weight identity, ordered dataset/statistics/text-cache
identities and sampling options. Position records epoch, batch offset, epoch-plan
digest and consumed count; count must match session state.

`validate_training_checkpoint(value, session, binding, expected_position=None)`
checks exact binding/runtime, option/state schemas, parameter shapes/trainability,
AdamW moments/counters, LR state, EMA, scaler and RNG. Header/tensor-reference
preflight and workspace checks precede safetensors allocation. Numeric finite
checks and RNG-generator validation precede live session mutation.
`restore_training_checkpoint` validates then uses transactional internal restore.
No dynamic import, pickle or local paths. Exact resume currently requires the
same runtime/device identity; compatible cross-runtime resume needs separate
explicit policy and evidence. Dataset/plan digests are caller-supplied bindings;
public jobs must derive them from validated actual dataset/epoch contracts.

Tests: round-trip and uninterrupted/resumed flow/diffusion, four backbones and
precision state; malformed tree/header/budget rejection before decode; corrupt
optimizer/LR/EMA/scaler/RNG/config/dataset position rejection without mutation;
binary integrity, cancellation and transport dictionary/bytes compatibility.

### Task 5: Public training ownership and job execution

Reconstruct all four source factory axes through a separate validated training
factory; the inference factory's restricted configuration matrix is not suitable.
Compare source initialization, ablation switches, CPU RNG preservation, cancellation
and parameter/buffer allocation budgets. Text dimension must come from validated
job/cache binding rather than an unverified user label. Keep live objects private.

Build portable job configuration binding actual dataset/statistics/cache identities,
model configuration, initialization selection, sampling, augmentation, optimizer
and loss policies. Generate deterministic epoch plans and accumulation groups from
validated source data; record epoch position and reject incompatible resume before
mutation. Use ComfyUI-selected device and managed full model residency. Run chunks
return portable checkpoint and progress, and release only owned model resources.
Add atomic checkpoint save/load nodes, explicit raw/EMA inference export and actual
installed-model inference verification. Exercise public schemas and real headless
Windows/stadia/worker training, resume, cancellation and output retrieval.

Current intermediate work: `training_model.py` provides scratch reconstruction,
meta-device allocation preflight and isolated CPU initialization. Backbone audit
against training revision 2c5b384 found identical numerical bodies; differences
are package imports and a package-name docstring. Public job/device ownership,
mixed-dataset execution, distributed training, selected-weight initialization and
learned-variance model output integration remain required open work.

Task 5 configuration increment: missing factory/job/public node RED → GREEN.
Source factory initialization matches eight attention/conditioning/positional
combinations; automated AST comparison checks every vendored backbone body against
the pinned training source. Mixed-label epoch plans match the validated sampler.
Actual V3 node execution and client/runner job codec round trips pass.
Review found two Important accepted-but-unusable configurations. Both fixed in
one pass: direct/linear/random insertion capacity (three RED→GREEN cases), and
missing/malformed training topology (ten RED→GREEN cases). Job and sample producers
share the existing augmentation-layer topology validation. Initial topology test
fixtures failed before reaching job creation; corrected content-addressed topology
IDs, observed all ten missing-rejection failures, then verified integration GREEN.
Whole suite: 1,152 passed, 50 skipped, six subtests. Headless stadia configuration
checks: 58 passed, one CUDA skip. Evidence: docs/2026-10-03-training-job-validation.md.
Final minor (deferred): independent numerical forward/gradient comparisons for
every ablation combination; source bodies and factory wiring are checked, but
these do not establish every ablation's training trajectory. Public execution
and the remaining Task 5 requirements are not complete.

Task 5 execution increment: missing execution module/public node RED → GREEN.
Actual prepared datasets execute source session updates and resume with matching
checkpoint bytes. Mid-epoch three-batch/accumulation-two tests cross a partial
group and next epoch. Public node API uses Comfy-selected residency, releases
owned models, and resumes through a different installation package namespace.
Stable known-backbone source identities supplement legacy qualified identities.
Comfy outer inference-mode failure reproduced → explicit gradient context GREEN.
Independent review's two Important findings fixed in one RED→GREEN pass: null
loss options canonicalized, and full-attention activation preflight before model
allocation. Full suite: 1,168 passed, 50 skipped, six subtests; headless focused
execution/job/node/checkpoint checks: 97 passed, three CUDA skips.
Evidence: docs/2026-10-03-training-execution-validation.md.
Ruling: use conservative math-SDP activation estimates even on CUDA, rather than
assuming fused kernels; cost is rejection under a budget that a fused kernel
might fit, resolved by an explicit larger workspace. Independent sample RNG
streams are an adapter policy, already required by sample contracts, rather than
a claim of source DataLoader global-random trajectory equality.
Remaining Task 5: selected installed raw/EMA initialization; checkpoint file IO;
trained inference export; actual headless server/partition handler execution,
artifact retrieval, injected cancellation and cleanup. Distributed/unbalanced
loaders and learned-variance model output remain required full-goal work.

### Subsequent required tasks

- [x] Diffusion schedule/loss kernel and reference comparisons.
- [x] Single-process optimizer session, internal resume, accumulation and precision foundation.
- [ ] AdamW, LR schedule, EMA, accumulation/precision/distributed sessions and exact resume comparisons.
- [x] Bounded portable model/optimizer/scheduler/EMA/RNG/data-position checkpoint value contract.
- [x] Portable checkpoint value contract, supplied artifact/position binding and exact fixture resume.
- [ ] Model architecture/job configuration and public training/progress/checkpoint nodes.
- [ ] Installed-model batch/backprop/update, exported checkpoint inference and independent playback.
- [ ] Windows/headless stadia/actual-worker resume, cancellation, cleanup and artifact retrieval.

Keep subsequent tasks detailed before implementing them. Their unchecked status
does not exempt them from GOAL.md.

Ruling: add explicit stable/released geodesic policies. Unchanged source conversion
creates NaNs for masked/degenerate rotations; stable substitutes identity before
source geodesic math. Valid rotation math remains exact. The released policy keeps
the source failure observable; nonfinite outputs are rejected before updates.

Evidence: full suite 961 passed, 50 skipped, six subtests. Four small backbone
updates and caller-selected CUDA passed; no public or installed-checkpoint
training claim. Review's reduced-mean overflow finding fixed through RED/GREEN.

Task 2: source/loss/gradient tests RED (missing module) → GREEN, 51 focused
checks. Cumulative schedule underflow RED → guarded ValueError GREEN. Four
small backbone updates and selected-CUDA source equality passed. Full suite:
1,012 passed, 50 skipped, six subtests. Final review: no actionable correctness
findings. Final: minor (deferred): direct diffusion valid-rotation geodesic
reference comparison; shared helpers already independently compared in flow.

Task 3 foundation: missing module RED → session/reference tests GREEN. Internal
OrderedDict finite/copy traversal failed corruption/update tests before fix.
Final: fixed EMA overflow — `test_ema_overflow_rolls_back_complete_state`
RED→GREEN; full suite 1,047 passed, 50 skipped, six subtests. Four backbone
dropout resumes and actual Linear autocast output dtype/restore checks pass.
Ruling: reject nonfinite accumulation groups atomically rather than applying
source microbatch skipping — avoids invalid partial updates; cost is different
data consumption on failures, which public/distributed jobs must make explicit.
Final: minor (deferred): compare partial final-group divisor numerically.
Final: minor (deferred): inject scheduler/EMA exceptions and pre-update cancellation.

Task 4: missing contract RED → round-trip/corruption checks GREEN. Strict
boolean/integer comparisons, missing moments, pre-copy workspace and post-load
cancellation were reproduced before fixes. Final review found three Important
issues, fixed in one pass with RED→GREEN: aggregate/escaped JSON preflight;
numeric runtime settings; partial Adam history deletion/lowered counters.
Sessions now record independent per-parameter committed update counts, including
zero-count unused/frozen parameters. Full suite after fixes: 1,098 passed,
50 skipped, six subtests; headless checkpoint checks 48 passed, three CUDA skips.
Actual client/runner codec round trips pass. No public or real-dataset training
claim. Final: minor (deferred): validate expected-position schema/types as strictly
as stored position; expected position is currently a caller-owned comparison.
