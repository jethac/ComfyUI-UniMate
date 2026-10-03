# Trained bundles implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline.

**Goal:** Make trained raw/EMA checkpoints usable through public model loading,
generation, constraints, expansion and animated exports.
**Architecture:** Assemble a versioned numeric bundle with the selected inference
weights and exact artifact identities. Dispatch a private Comfy-managed runtime
from the bundle version; reuse configured conditioning and recorded sampling math.
**Tech Stack:** Torch, safetensors, NumPy, JSON/ZIP, ComfyUI, Blender.
**Spec:** GOAL.md and DESIGN.md, selected inference weights and model management.

## Global constraints

- Keep released `unimate.bundle.v1` acceptance and behavior intact.
- No pickle, automatic downloads, provider calls or live objects on sockets.
- Preserve explicit raw/EMA selection, source revision and checkpoint identity.
- Match full statistics/cache identities recorded in `job.datasets`; matching
  array bytes alone do not establish matching metadata or encoder identity.
- Use ComfyUI device management; support interruption and private cache cleanup.
- Internal helpers are dependencies, not evidence of a public workflow.
- Full GOAL.md remains active after this plan; other source APIs remain required.

## Files and responsibilities

- `unimate_pack/trained_bundle.py`: assembly, manifest and cross-artifact binding.
- `unimate_pack/bundle.py`: version dispatch, bounded inventory/hash inspection
  and safe extraction shared by released and trained bundles.
- `unimate_pack/contracts.py`: portable model acceptance for the trained schema.
- `unimate_pack/training_text_model.py`: shared installed encoder identity formula.
- `unimate_pack/inference.py`: Comfy-managed trained model/text runtime and public
  inference dispatch; factor trained ownership if needed to keep this readable.
- `unimate_pack/trained_sampling.py`: recorded normalized sampling; extend source
  constrained behavior only with independent numerical comparisons.
- `unimate_pack/expansion.py`, `nodes.py`: configured windows/statistics selection,
  public assembly node, managed outputs and Cloud Offload asset declarations.
- `tests/test_trained_bundle.py`, `tests/test_trained_runtime.py`: numeric bundle
  safety and public inference behavior. Actual server harness under `tools/`.

## Task 1: complete portable bundle assembly

- [ ] Write missing-interface RED tests for
  `assemble_trained_bundle(weights, statistics, text_cache, encoder_model,
  sampling=None, *, cancel=None, max_workspace_bytes=...) -> UNIMATE_MODEL`.
- [ ] Specify strict `unimate.bundle.v2` inventory: selected weights archive,
  portable statistics metadata/arrays, bound cache metadata, encoder assets when
  consumed, manifest with recorded job/source/checkpoint/weight identities,
  sampling options and output fps. Persist no optimizer or text-cache tensor data
  unnecessary for inference.
- [ ] Reuse existing validated job/weights/statistics/cache identities. Require one
  recorded dataset binding with both exact statistics and cache identities;
  reject swapped labels/encoder/source or recomputed forged metadata.
- [ ] Factor the installed encoder identity without changing its existing digest.
  Match bundled inventory and encoder identity to the cache and trained text width.
  Encoder omission is valid only when the recorded model consumes neither caption
  nor joint-name embeddings; prove that gating against all actual backbones.
- [ ] Reject duplicates, paths, links, encryption, unknown members, oversized JSON,
  inconsistent digests and corrupt numeric archives before model construction.
  Test budget and cancellation before expensive decode/copy and atomic output.
- [ ] Add public Assemble UniMate Model node, returned model plus managed `.unimate`
  file, and actual declared-input Cloud Offload staging/retrieval/reload tests.
- [ ] Review, run full suite and commit/push verified assembly.

## Task 2: Comfy-managed trained generation

- [ ] RED tests load complete bundles through existing Model Loader; reconstruct
  exact selected frozen backbones and portable statistics with no global RNG change.
- [ ] Load bound encoder offline, using the same recorded pooling/token policy as
  training. Reject incompatible encoder architectures rather than substituting T5.
- [ ] Use `build_training_condition` and `sample_trained_model` with recorded
  capacities/loss/sampling settings; transfer tensors to the Comfy-selected device.
  Preserve raw/EMA and source checkpoint provenance in output motion metadata.
- [ ] Replace fixed normalization/window assumptions in public generation and
  batches with runtime-derived choices while preserving released defaults.
- [ ] Compare actual trained raw/EMA output to independent pinned inference math;
  test configured windows, spectral width, statistics labels and all backbone axes.
- [ ] Exercise private RoPE/text caches, memory unload, cancellation and failed
  construction cleanup. Verify CPU and Comfy-selected CUDA independently.
- [ ] Review, run full suite and commit/push verified runtime.

## Task 3: constraints, expansion and playback

- [ ] RED tests preserve exact selected reference frames/joints at configured
  windows; independently compare pinned source constrained sampling.
- [ ] Implement flow and applicable diffusion constrained workflows, documenting
  paper/source semantics for paths without a released constrained implementation.
  Preserve reference length, skeleton identity and normalization roundoff handling.
- [ ] Adapt prompt expansion overlap/window controls to trained capacities; reject
  invalid overlap before model allocation. Verify prompt transitions numerically.
- [ ] Actual ComfyUI graph: trained checkpoint -> selected weights -> complete bundle
  -> load -> free/inbetween/edit/expand -> GLB/FBX export and numeric archives.
- [ ] Execute Windows and headless stadia-testbed graphs plus actual Cloud Offload
  partition jobs, declared input staging and client retrieval. Compare direct and
  worker outputs within the recorded runtime/device.
- [ ] Independently evaluate every exported frame's transforms/skinned vertices,
  appearance and rig identity; record asset licenses and exact runtime identities.
- [ ] Review, run full suite, update coverage/README evidence, commit/push.

## Required follow-through

Learned-variance backbone output must match the source kernel's doubled joint
axis; fixture predictor coverage does not satisfy this. Other released encoders,
precision paths, SDE/reverse/likelihood and distributed/data-processing capabilities
remain explicit inventory gaps until implemented and validated. Complete registry
publication and installation only with accurate implemented-scope descriptions.
