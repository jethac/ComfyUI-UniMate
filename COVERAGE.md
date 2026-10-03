# UniMate coverage audit

Audit date: 2026-10-03. This is a capability inventory, not a claim of complete support.

## Evidence baseline

- [Project page](https://linzhanmou.com/unimate/) and [paper, arXiv v1](https://arxiv.org/html/2609.05415v1).
- [Released source](https://github.com/Friedrich-M/UniMate/tree/2c5b384715aa63d8639b1ed7eb74bfe614570c7a), inspected locally at that commit. Current pack remains pinned to `5d6aabedd947297b5ba6706d8e9113e68c0c3e4f`.
- [Model repository](https://huggingface.co/Linzhan/UniMate/tree/971da7cfc1c8d99c2af6c00be9d2ed5700f99073), file inventory checked through the Hub API. Model weights other than the existing v2 bundle were not downloaded or tested during this audit.
- Local implementation: `nodes.py`, `unimate_pack/upstream.py`, `inference.py`, `contracts.py`, and `bundle.py`. Existing validation is recorded in `VALIDATION.md`.

## User-facing capability mapping

| Capability | Released evidence | Current pack | Required node surface |
| --- | --- | --- | --- |
| Text-conditioned motion on a target skeleton | `unimate/inference/sample.py`, `generate.py` | Implemented for one v2 graph/AdaLN configuration | Generate Motion |
| Unconditional generation | `sample.py`: text-list cases and guidance 1 | Implemented through guidance 1; needs explicit UI explanation | Generation mode |
| Multiple seeds, repetitions, cases and inference chunks | `sample.py` | One prompt/sample per invocation | Batch/repetition inputs and motion collection outputs |
| Cross-topology behavior transfer through text | Paper §5.5; same behavior prompt drives each target skeleton | Possible by repeating generation on different rigs; no dedicated workflow | Shared prompt → target rig collection example |
| In-betweening | `motion_inbetweening.py`, `sample.py --inbetween --keep_frames` | Node and constrained inference implemented; offline synthetic-rig model/export check passed; server/cloud checks pending | In-between Motion, reference motion and frame selection |
| Text-guided editing with selected joints held fixed | `motion_editing.py`, `sample.py --motion_edit --keep_joints` | Node and constrained inference implemented; offline synthetic-rig model/export check passed; server/cloud checks pending | Edit Motion, reference motion and joint selection |
| Multi-prompt motion expansion | `motion_expansion.py`, `sample.py --motion_expand --expand_overlap` | Node and overlap stitching implemented; real-model 110-frame synthetic-rig export/playback check passed; server/cloud checks pending | Expand Motion, ordered prompts and overlap |
| Known-motion input needed by editing/in-betweening | `sample.py` reads a selected dataset clip | Numeric archive loader implemented; source GLB animation extraction missing | Load Motion Features and Extract Asset Motion |
| Rest-pose rig conditioning | `data_process/mesh_animation/preprocess_char.py` | Implemented for restricted GLB assets, explicit facing | Prepare Rig; expose inspected joint names and mapping |
| Canonical asset and conditioning export | `preprocess_char.py`: GLB/FBX, conditioning and optional visualization | Missing as outputs; internal conditioning exists | Export Conditioning / Canonical Asset |
| Generated numeric motion export/reload | `sample.py`: `(T,J,12)` motion files and caption ledger | Safe numeric archive load/save nodes implemented; upstream plain-feature interchange pending | Save/Load Motion Features with rig identity |
| Animated mesh export | `data_process/mesh_animation/animate_motion.py`: GLB and FBX | GLB only | Export Animation with supported format selection |
| Skeleton visualization, FK and RIC recovery | `sample.py --save_ric`, visualization utilities | Both recovery modes and fixed-frame IMAGE rendering implemented; upstream comparisons, Windows and headless stadia ComfyUI retrieval passed; cloud runner execution pending | Recover Skeleton (`fk`/`ric`) → Render Skeleton → Preview Image / Save Image |
| Released model selection | Hub inventory: Mixamo, UniML3D preview, v2 graph/AdaLN, v2 full/cross-attention | Only v2 graph/AdaLN accepted | Config-driven Model Loader and converter |
| Checkpoint selection | `sample.py --model_path`; Hub includes multiple training steps | Bundle selects one EMA checkpoint during conversion | Explicit installed checkpoint/bundle selection and metadata |
| Model-dependent dimensions and normalization | Config/schema, dataset statistics, model factory | Hardcoded 60 frames, 71 padded joints, narrow architecture | Derive supported dimensions from validated model config |

## Sampling details that affect implementation

The released constrained sampler uses fixed noise and replacement along the flow interpolant, then a fixed-step Euler update. It accepts a broadcastable mask for frames or joints. The implementation must match this path; clamping the result of the existing unconstrained `dopri5` sampler is not equivalent.

Editing matches original or cleaned joint names without case sensitivity and handles actual valid clip lengths. In-betweening supports signed frame indices, resolved against valid lengths. Both require reference motion encoded in the same canonical skeleton basis and normalized with the selected model statistics.

Expansion generates the first segment freely, pins each later segment's prefix to the preceding segment's tail, and removes the duplicate overlap when concatenating. For window length T, N segments and overlap O, output length is T + (T − O)(N − 1). Export, transport validation, provenance and playback checks must accept the resulting length.

## Model and research coverage

Released config inspection at Hugging Face revision `971da7cfc1c8d99c2af6c00be9d2ed5700f99073`:

| Family | Attention / text conditioning | Padded joint slots | Depth capacity | Layers |
| --- | --- | --- | --- | --- |
| `unimate_mixamo_f60` | graph / AdaLN | 22 | 7 | 6 |
| `unimate_uniml3d_f60_preview` | graph / AdaLN | 61 | 19 | 10 |
| `unimate_uniml3d_f60_v2` | graph / AdaLN | 71 | 19 | 10 |
| `unimate_uniml3d_f60_v2_full_cross_attn` | full / cross-attention | 71 | 19 | 10 |

All four configs use 60-frame, 12-feature windows and FLAN-T5-base. Their capacity differences must drive both denoiser reconstruction and conditioning padding. The current constrained path hardcodes 71 slots; Mixamo statistics are not yet validated against the current three-family statistics contract. Exact config acceptance alone would not establish support. Each family requires strict checkpoint loading, numeric conditioning comparison and actual inference/export evidence.

The released factory implements graph/full attention and AdaLN/cross-attention combinations. The current adapter accepts one exact graph/AdaLN configuration. Supporting the released full/cross-attention model requires token text conditioning and config-driven reconstruction, not relaxing validation alone. Preview and Mixamo bundles must each pass strict loading and numerical checks against their own configs and statistics.

Training is also released: flow/diffusion schedules, EMA, resume, mixed datasets, balanced sampling, topology augmentation, cached text embeddings and architecture ablations. These are not currently represented by nodes. A literal claim of 100% technology coverage must account for them explicitly; an inference-only pack cannot make that claim. Training/job nodes need isolated configuration, dataset input, progress, cancellation and exported checkpoint contracts. Dataset curation includes filtering, canonicalization, rendering and language-assisted annotation; those utilities likewise need a documented interface or an explicit coverage gap.

The paper additionally describes foot-locking with contact detection and inverse kinematics (§E.5), and discusses linear-blend or dual-quaternion mesh deformation (§B.7). Their public node coverage is missing. This audit found no named foot-locking implementation in the inspected inference/mesh-animation paths; paper-level coverage requires locating further source or implementing and independently validating the described method. Do not label these capabilities released-and-integrated on the strength of the paper alone.

## Corrected scope

UniMate consumes skeletons and existing skinning. Automatic rigging is an adjacent technology, not a UniMate model capability. The abandoned SkinTokens investigation does not establish UniMate coverage and must not drive this pack's scope.

Cross-topology text-mediated transfer is supported technology. Conventional correspondence-based transfer of an existing clip is a different operation. Keep that distinction visible without dropping UniMate's demonstrated transfer workflow.

## Implementation order and proof requirements

1. Add reference-motion import/extraction, frame/joint selectors and variable-length motion contracts. Verify canonical round trips and rig identity before sampling.
2. Add in-betweening, editing and expansion through the released constrained sampler. Compare each numerically to the pinned reference; assert kept values and seam behavior independently.
3. Generalize model reconstruction and conversion for every released family. Record model/config/statistics identities; test actual checkpoints, not only schemas.
4. Add collections, conditioning outputs, numeric file round trips, FBX export and skeleton previews. Verify each through ComfyUI execution.
5. Account for training, preprocessing utilities and paper-only postprocessing in separate explicit coverage rows before claiming complete technology coverage.
6. Extend Cloud Offload staging and retrieval for every new portable value; run each inference mode headlessly on stadia-testbed and evaluate exported playback independently.

The existing five-node tests prove the narrow generation/export path. They do not prove complete UniMate coverage.

## Execution evidence

2026-10-03: headless stadia-testbed at pack `2e17283` accepted the pre-fix Windows motion archive against its freshly prepared Linux rig. Load Motion, both skeleton recovery modes, both rendered image batches and GLB/provenance export passed; all 34 PNGs were retrieved. Evidence: remote `run-legacy-windows-20261003/report.json`. This completes the recorded Windows-to-Linux and Linux-to-Windows archive compatibility checks for the identical synthetic conditioning payload.

2026-10-03: traced the Windows/Linux rig ID difference to NPZ ZIP creator-OS metadata. Source asset, mapping and every numeric array were byte-identical; only directory OS markers differed. Numeric encoding and new rig preparation now canonicalize that marker to 3, preserving array contents and the existing Linux rig ID. Consumers recognize exact legacy Windows/Linux IDs recomputed from a validated rig, including public GLB/FBX exports. Fresh Windows Blender preparation matched the recorded Linux ID `bc2317bc8cc514dbf28979e78d0b9f432d2290ccff87c11b2340f4a5e2cd6e51`. The previously rejected 60-frame stadia archive passed the Windows ComfyUI workflow and all 120 PNGs plus GLB/provenance were retrieved. Focused regression tests passed (102 plus 6 subtests); code review found no remaining blockers. Evidence: `.runtime/identity-windows-fixed`, `.runtime/identity-linux`, `.runtime/skeleton-cross-platform-fixed/report.json`. Numerical differences are not aliased; this fix addresses the observed container-metadata difference.

2026-10-03: skeleton workflow passed headlessly on stadia-testbed at pack commit `a0d841a`, ComfyUI `84ba85773925f071c516f0208184773802b4d44a`, Python 3.11.15, PyTorch 2.14.1+cpu and Blender 5.1.1. The workflow loaded the earlier 60-frame generated in-between archive, prepared the matching source rig, recovered FK and RIC, rendered and retrieved all 120 PNGs plus GLB/provenance. Remote evidence: `run-skeleton-20261003/report.json`. This verifies Linux archive/recovery/render execution; it does not resolve cross-environment re-preparation identity stability or cloud runner execution.

2026-10-03: added Recover Skeleton and Render Skeleton. Both FK and RIC match pinned upstream on a rotating 17-frame branching case within 1e-9. Origin, identity, topology, archive digest, cancellation, fixed framing and bounded image allocation checks pass; actual Cloud Offload codec round trips preserve the skeleton value. Focused tests passed (109 plus 6 subtests). Windows ComfyUI execution loaded a locally prepared 17-frame synthetic archive, recovered both modes, retrieved all 34 PNGs through `/view`, and exported GLB/provenance. The first rendered frame was inspected. Evidence is local at `.runtime/skeleton-server-fixture-check/report.json`. Cloud runner and stadia execution remain pending. A stadia archive was rejected when paired with a separately prepared Windows rig; cross-environment re-preparation identity stability requires investigation, with no identity bypass introduced.

2026-10-03: the corrected expanded inference graph passed headlessly on stadia-testbed using CPU inference, offline v2 EMA/T5 and Blender 5.1.1. ComfyUI retrieved four GLBs, four provenance JSON files and three motion archives. Independent skinning evaluation checked all 290 frames (60 generation, 60 in-betweening, 60 editing, 110 expansion), with finite changing vertices and unchanged source meshes, skins, materials, images and textures. Numeric archives decoded with the expected frame counts. This run does not cover the newer extraction, canonical, conditioning, batch, FBX or Cloud Offload paths.

2026-10-03: reproduced Blender FBX dropping base-color textures behind glTF's constant multiply node. Conversion now folds that expression into a 16-bit PNG, bounds effective linear-color error to 5e-5, embeds the texture, removes private sidecars and checks reimported pixels within 2e-6. Seven- and 110-frame skinning/texture checks and node regressions passed (23 tests plus 6 subtests). Other PBR expressions and server/cloud FBX retrieval remain open.

2026-10-03: registered Export FBX with the same rig/motion identity, path confinement, cancellation and atomic provenance publication as Export GLB. Binary FBX and its format-specific SHA-256 metadata are returned as retrievable files. Seven- and 110-frame branching clips passed the worker's per-frame Blender skinning round trip; node/export regression checks passed (23 tests plus 6 subtests), and lint passed. Material/texture fidelity, actual server retrieval and cloud execution remain pending.

2026-10-03: added external Blender binary-FBX conversion after the preserving GLB animation path. The worker bakes at 30 fps without leaf bones, embeds exported images, reimports with zero animation offset, verifies bone identities and compares evaluated skinning at every frame using bidirectional nearest-vertex checks. The seven-frame transformed branching fixture passed (4.79 seconds) after identifying Blender's default FBX import offset. This is adapter evidence only: the public FBX output node, long clips, material/texture fidelity and cloud/server transport remain pending.

2026-10-03: explicit Rig Conditioning output retains the exact validated topology NPZ, SHA-256, canonical rig identity and coordinate mapping. Conditioning and Canonical Asset values both passed concrete boundary-type validation and exact dump/load round trips through the installed `ComfyUI-Cloud-Offload/partition_protocol.py` codec (two tests, 3.50 seconds; lint clean). This proves local serialization only. Actual ComfyUI bridge execution, worker execution, staging/retrieval and expanded cloud workflows remain pending.

2026-10-03: registered Canonical Asset using the existing portable `UNIMATE_ASSET` socket. The adapter wraps the active scene under the prepared similarity transform while preserving original binary/accessors/mesh/skin/material/texture data. Independent vertex evaluation and joint-world comparisons verify canonical coordinates on the transformed branching fixture. Node and canonical adapter checks passed (21 tests plus 6 subtests), with lint clean. ComfyUI server, Blender reimport and Cloud Offload execution remain pending; explicit conditioning output is still missing.

2026-10-03: Prepare Rig now exposes shoulder pairs and body-axis selection, stores their mode in rig metadata and carries it into source extraction. External Blender prepared both four-joint modes and exported seven-frame rest-preserving clips checked with the independent skinning evaluator. Facing options on explicit-axis preparation fail rather than being ignored; rig contracts validate boolean body-axis metadata and restrict unavailable-name sentinels to two indices. Focused Blender/source/contract checks passed (68 tests, one optional reference skip), and lint passed. Expanded server/cloud facing checks remain pending.

2026-10-03: added a shared numeric facing adapter for two-joint and four-joint lateral pairs, body-axis correction and unavailable-name sentinels. All four modes match pinned `get_root_facing_quat` matrices within 1e−10 on deterministic motion; degenerate pairs fail explicitly. Source extraction now uses this adapter for its existing two-joint conditioning. Nine facing/extraction tests passed with local references enabled and lint passed. Preparation UI and conditioning contracts still need four-joint/body-axis inputs before those modes are user-accessible.

2026-10-03: Generate Batch executed through the actual ComfyUI server with two prompts × two repetitions. Four list elements produced four GLBs and four provenance files; all eight outputs were retrieved through `/view`. Saved metadata matched seeds 0–3, prompt/repetition ordering and 60-frame lengths. Independent skinning evaluation verified all 240 playback frames were finite and each clip changed over time. Report/artifacts remain local under `.runtime/batch-server-check`, using CUDA RTX 5060 Ti and ComfyUI `84ba85773925f071c516f0208184773802b4d44a`. This does not prove Cloud Offload list boundaries or multi-rig transfer.

2026-10-03: source extraction's joint-pair facing was compared numerically with pinned `data_process/utils/skeleton.py:get_root_facing_quat` over a 17-pose turning branching clip. Extracted root 6D rotations matched within 2e−6; all four source-motion tests passed with references enabled. The Motion dependency remains local and test-only. Four-joint/body-axis facing and source extraction through ComfyUI/Cloud Offload remain unverified.

2026-10-03: preview's portable EMA/T5 bundle passed conditioned and unconditional free generation, finite 60-frame features, Blender export and independent changing skinning playback (two tests, 205.76 seconds). Outputs are local under `.runtime/preview-free-model-check`. All four released families now have direct free/constrained inference and export evidence. Expanded server/cloud workflows, model-family capacity boundaries and headless coverage remain separate gates. The headless verifier now supports an expanded inference graph and validates retrieved numeric archives and GLB structure; two focused artifact-validation tests and lint passed. Its stadia execution remains active and has not completed artifact retrieval.

2026-10-03: Mixamo's installed EMA/T5 bundle passed conditioned and unconditional free generation, finite 60-frame feature checks, Blender export and independent skinning playback (two tests, 115.98 seconds). Outputs are local under `.runtime/mixamo-free-model-check`. Preview free-generation remains running. The expanded stadia ComfyUI graph registered successfully and is executing on CPU; completion and artifact checks remain unproven.

2026-10-03: the full-cross-attention bundle passed offline free sampling at guidance 1 and 3, local T5 conditioning, finite 60-frame features, Blender export and independent changing skinning playback across all frames (two tests, 169.61 seconds). Outputs are local under `.runtime/full-cross-free-model-check`. Equivalent Mixamo and preview checks are running; server/cloud/stadia coverage remains separate.

2026-10-03: official EMA conditional and unconditional forward outputs for Mixamo, preview and full-cross-attention matched the pinned upstream backbones bit for bit on deterministic padded branching inputs, with finite outputs. The three opt-in checkpoint tests passed in 498.50 seconds and lint passed. These comparisons use fixture text embeddings and statistics to isolate denoiser reconstruction; they do not substitute for end-to-end T5, sampling or cloud checks.

2026-10-03: preview and full-cross-attention portable EMA bundles built offline and passed 60-frame joint-preserving editing and in-betweening, exact reference-feature checks, Blender export and independent finite/changing skinning checks across every frame (104.28 and 105.55 seconds respectively). Their artifacts are local under `.runtime/preview-model-check` and `.runtime/full-cross-model-check`. Full-cross-attention also passed strict checkpoint reconstruction (36.40 seconds). Conditional/unconditional free-generation and official-EMA forward-output comparisons remain in progress; these results do not prove expanded ComfyUI or Cloud Offload execution.

2026-10-03: Mixamo's portable EMA bundle passed offline 60-frame in-betweening and joint-preserving editing with its own normalization statistics. Exact constrained features, finite changed output, Blender export and independent skinning evaluation across all frames passed (one integration test, 107.62 seconds); outputs are local under `.runtime/mixamo-model-check`. Preview's official step-100000 checkpoint passed strict raw state loading and upstream state/EMA inventory checks (61.02 seconds). Full-cross-attention has downloaded and its reconstruction test is running. Free generation, numeric forward-output comparisons, expanded ComfyUI/Cloud Offload and stadia-testbed checks remain pending for these added families.

2026-10-03: the official Mixamo step-100000 checkpoint at model revision `971da7cfc1c8d99c2af6c00be9d2ed5700f99073` passed restricted `weights_only=True` loading, strict raw state reconstruction, upstream state-key/parameter-order/shape comparisons, and finite EMA shadow checks (one integration test, 30.91 seconds). Portable bundle conversion and actual Mixamo inference remain pending. Preview and full-cross-attention checkpoints are downloading; they have not passed this gate.

2026-10-03: short constrained references now set their actual motion length and preserve the upstream `(B,1,1,T)` frame-mask shape. The corrected adapter passed offline released v2 EMA/T5 inference for seven-frame in-betweening and editing, exact feature constraints, Blender export and independent finite/changing skinned-vertex evaluation at every frame (one parameterized integration test, 79.07 seconds). Artifacts are local under `.runtime/short-reference-mask-check`. This is direct runtime evidence; expanded ComfyUI server and stadia-testbed execution remain pending.

2026-10-03: registered Extract Motion and connected glTF clip sampling to canonical feature encoding. The adapter checks animated bone lengths, retains rig identity and initial root XZ position, and emits F−1 frames. Tests cover +Z/+X preparation and independent skinned-vertex round trips with a nonzero initial root position. Focused node, source extraction, clip sampling, contract and constrained-inference checks passed (88 tests plus 6 subtests); lint passed. Joint-pair facing and extraction through ComfyUI server/Cloud Offload remain unverified. This does not establish full reference-motion coverage.

2026-10-03: added glTF clip evaluation at 30 fps with LINEAR/STEP/CUBICSPLINE channels, shortest-arc quaternion interpolation, duration-scaled cubic tangents, endpoint clamping and a bounded pose allocation. A 110-frame generated clip preserves expected root/world transforms through evaluation. Clip/export/contract checks passed (78 tests before the additional STEP hold test); lint passed. This adapter follows the linked Khronos specification. Canonical facing, source-to-prepared identity checks and the public extraction node remain pending; sampling poses alone is not complete source-motion extraction.

2026-10-03: implemented dependency-light forward feature encoding and rest-pose rotation rebasing for reference-motion extraction. The encoder constructs RIFKE positions, parent-shifted 6D rotations and destination-facing local velocities, emitting F−1 features from F poses as upstream does. Tests compare rotating branching motion and nonidentity rest orientations numerically with pinned `unimate/utils/motion_utils.py`; all 4 tests passed with local references enabled, and lint passed. The unlicensed Motion dependency remains test-only and is not shipped. Asset clip evaluation/canonicalization and the public Extract Motion node remain unimplemented; this is numerical adapter evidence only.

2026-10-03: added Expand Motion with ordered JSON prompts, 1–59-frame overlaps, per-segment uint64 seeds and retained provenance. Unit checks verify tail replacement, duplicate-overlap removal, total length and invalid requests; node checks passed (23 tests plus 6 subtests). Numeric/export regressions passed (90 tests), and lint passed. Actual offline v2 EMA/T5 generation for two prompts produced a 110-frame seven-joint motion, exported through Blender, and passed independent skinning evaluation across all 110 frames with finite changing vertices. Original meshes/skins/materials/images/textures and the source binary prefix remained unchanged. The opt-in model expansion test passed in 151.49 seconds; artifacts remain under `.runtime/expansion-model-check`. This is direct inference/export evidence, not ComfyUI server, cloud scheduling or stadia-testbed validation.

2026-10-03: Load Motion and Save Motion nodes now round-trip safe numeric archives, confine files to managed paths, fingerprint input contents, stage selected archives through `__input__`, and atomically publish outputs after cancellation checks. Focused regression checks passed (88 tests plus 6 subtests), with lint clean. The actual offline v2 EMA/T5 model generated both constrained modes on the original seven-joint synthetic branching rig; selected endpoint frames/joint features remained exactly equal to reference features, other values changed and stayed finite, and Blender exported all 60 frames in each GLB. `tests/test_constrained_model.py` passed in 103.04 seconds using the installed licensed bundle and Blender 5.1.1. This test invokes inference directly, not the ComfyUI server or cloud job scheduling. The resulting GLBs remain local under `.runtime/constrained-model-check`. Source-clip extraction, expansion, independent constrained playback evaluation and stadia-testbed/Cloud Offload checks remain pending.

2026-10-03: registered In-between Motion and Edit Motion nodes and connected reference motion to the Comfy-managed inference path. Added canonical normalization, same-rig/joint/window validation, original/clean joint aliases, constrained solver provenance, and exact reference-feature preservation across normalization roundoff. Existing node tests and new deterministic inference plumbing tests passed (17 tests plus 6 subtests); lint passed. These checks use a fixture denoiser, not the released model. Real-model constrained generation, source-clip extraction, archive loader/saver nodes, expansion and headless integration remain pending.

2026-10-03: added signed frame selection against valid reference length and case-insensitive joint selection with original/clean aliases. Added locally seeded classifier-free guidance around constrained flow; tests verify global RNG preservation and the explicit guided velocity result. With the pinned upstream checkout enabled, the constrained and selection tests passed (15 tests). Public node wiring and real-model checks are still pending.

2026-10-03: added a constrained Euler kernel with fixed-noise replacement, broadcast frame/joint masks and cancellation checks. Both mask modes match the pinned upstream sampler bit for bit on the deterministic velocity fixture (7 tests passed including cancellation and invalid-step checks). Added safe numeric reference-motion archive save/load with canonical rig identity and Unicode metadata round trips (3 tests passed). These are subsystem interfaces; public IO/edit/in-between/expansion nodes and real-model constrained validation remain pending.

2026-10-03: motion validation and GLB export now accept nonempty variable-length clips within the existing numeric archive limits. Blender playback verification iterates over the actual clip length. Contract and independent GLB skinning tests cover 1-, 59-, 60-, 110- and 600-frame values where applicable: 74 focused tests passed. Generation remains fixed to its existing model window; expansion and reference-motion nodes are not implemented by this change. External Blender and Cloud Offload validation for expanded clips remain pending.

2026-10-03: actual Cloud Offload partition-handler workflows passed on Windows
(17-frame fixture) and headless stadia (60-frame released-model archive). Six types
crossed eight boundaries, then were restored and consumed. Canonical preparation,
conditioning, motion extraction, FK/RIC rendering and GLB/FBX export passed. Stadia
retrieved 368 files; independent playback and all 360 preview images matched after
restoration. See VALIDATION.md for revisions and evidence. Provider scheduling,
deployed containers, inference/batch paths and remaining capability gaps are open.

2026-10-03: corrected partition execution-list loss (repeated captures overwrote
earlier cases). New client/runner envelopes preserve ordered mapped values and
nested list-valued scalar data. Windows and headless stadia three-job tests passed;
stadia restored two distinct 60-frame cases, recovered both skeleton modes and
retrieved all 600 preview frames. Independent position and pixel checks confirmed
case preservation. New model batches and multi-rig public workflows remain open.
