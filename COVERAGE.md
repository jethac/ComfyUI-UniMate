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
| Skeleton visualization, FK and RIC recovery | `sample.py --save_ric`, visualization utilities | No preview/render nodes | Preview Skeleton / Render Motion; distinguish recovery paths |
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
