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
| In-betweening | `motion_inbetweening.py`, `sample.py --inbetween --keep_frames` | Missing | In-between Motion, reference motion and frame selection |
| Text-guided editing with selected joints held fixed | `motion_editing.py`, `sample.py --motion_edit --keep_joints` | Missing | Edit Motion, reference motion and joint selection |
| Multi-prompt motion expansion | `motion_expansion.py`, `sample.py --motion_expand --expand_overlap` | Missing; motion validator requires exactly 60 frames | Expand Motion, ordered prompts and overlap |
| Known-motion input needed by editing/in-betweening | `sample.py` reads a selected dataset clip | Missing; source GLB animation is discarded | Load Motion Features and Extract Asset Motion |
| Rest-pose rig conditioning | `data_process/mesh_animation/preprocess_char.py` | Implemented for restricted GLB assets, explicit facing | Prepare Rig; expose inspected joint names and mapping |
| Canonical asset and conditioning export | `preprocess_char.py`: GLB/FBX, conditioning and optional visualization | Missing as outputs; internal conditioning exists | Export Conditioning / Canonical Asset |
| Generated numeric motion export/reload | `sample.py`: `(T,J,12)` motion files and caption ledger | Portable internal NPZ exists; no user loader/saver | Save/Load Motion Features with rig identity |
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
