# Released training and dataset audit

Source: UniMate `2c5b384715aa63d8639b1ed7eb74bfe614570c7a`, inspected
2026-10-03. The inference adapter remains pinned to `5d6aabe`. This inventory
records requirements for the next integration; it does not claim node support.

## Training

| Source | Behavior to preserve | Required evidence |
| --- | --- | --- |
| `unimate/configs/schema.py` | Full configuration blocks; legacy model-type conversion with conflict rejection; graph/full attention and AdaLN/cross-attention axes; optional architecture embeddings, spectral encoding and gradient checkpointing | Configuration reconstruction and factory matrix |
| `training/trainer.py::_setup_diffusion`, `compute_loss` | Flow transport or diffusion schedule; diffusion samples integer timesteps; both reduce loss components over the batch | Pinned loss and gradient comparisons for both paradigms |
| `training/train.py::train_diffusion` | AdamW, configured betas/weight decay, cosine warmup and minimum LR; effective optimizer steps drive scheduler, EMA, progress and checkpoint intervals | Accumulation, clipping and scheduler trajectories |
| Same loop | Accelerate device-specific seed, autocast, distributed batch sharding; nonfinite-loss decision reduced across ranks; discard accumulated gradients when the final microbatch is invalid | Single-device checks and actual distributed checks, including invalid loss |
| `training/ema.py` | Trainable-parameter shadows, decay schedule, optional warmup, stored step and parameters | Independent updates and save/restore comparisons |
| `train.py::_save_checkpoint_and_visualize` | Model, optimizer, scheduler, step, epoch and optional EMA; debug sampling precedes saving | Real checkpoint retrieval and conversion into an inference bundle |
| `train.py::_resume_from_checkpoint` | Restore available optimizer/scheduler; absent scheduler fast-forwards; absent EMA reinitializes from loaded weights | Full resume plus each legacy fallback |
| `training/tracker.py` | Exactly one duration mode: epochs or effective steps | Both stopping modes with accumulation |

The released checkpoint does not persist RNG state, sampler position within an
epoch or a mixed-precision scaler. Its resume path restores recorded state and
begins dataloader iteration again. Do not claim exact continuation of the original
random sequence. If the adapter adds stronger continuation, identify it as an
adapter extension and retain a released-semantics mode.

The scheduler is deliberately not wrapped by Accelerate: its target duration is
already counted in optimizer steps. Wrapping it can advance the schedule once
per process and change multi-device behavior.

## Training data

| Source | Behavior to preserve | Required evidence |
| --- | --- | --- |
| `dataset/factory.py`, `mixture/dataset.py::Mixture` | Truebones, Mixamo and Objaverse layouts; caption-free input; category/object filters; joint-count limits; explicit held-out objects | Each layout, missing optional metadata and filtering decisions |
| `MotionDataset::_split_names` | Seeded clip- or object-level holdout per dataset; explicit object lists override ratio; retain at least one training clip/type | Exact pinned split membership and order |
| `MotionDataset::_compute_max_joints_and_depth` | Loaded topology determines padded width/depth; addition gets one extra joint; configured depth can cap it | Capacity checks with enabled addition and model reconstruction |
| `_calculate_dataset_stats`, `_stats_frame_weighted`, `_stats_balanced` | Per-dataset or global statistics; separate root/local channels; frame-weighted or equal object-type moments; standard-deviation floor 1e-8; optional channel-group tying | All eight scope/balance/tie combinations against pinned functions |
| `MixtureSampler` | One-level object balancing or two-level dataset/object balancing; weights proportional to powers of clip counts; replacement sampling seeded by epoch | Exact weights and sampled indices for multiple epochs |
| `_apply_augmentations`, `mixture/augmentations.py` | Choose at most one enabled augmentation or no-op; ellipsoid joint addition, leaf removal, pass-through pooling, per-bone perturbation | Fixed-seed arrays, topology, normalization and FK/RIC agreement for every operation |
| `dataset/transforms.py`, `MotionDataset::__getitem__` | T-pose random cropping or first-frame conditioning; optional facing realignment; identity rest rotations; normalization before zero-padding; parent features | Exact samples for both conditioning modes, short clips and moved crop origin |
| `mixture/collate.py` | `(B,J,12,T)` motion; joint/frame masks; padded standard deviations of one; spectral/text padding; one valid caption-token position on caption-free rows | Mixed topology/length/token batches and masked loss behavior |
| Text-cache methods in `MotionDataset` | Joint-name and caption embeddings plus caption token sequences | Encoder/cache identity, offline cache reuse and cancellation; deeper cache audit remains open |

Global balanced statistics group by `object_type` alone. The sampler groups by
`(dataset_type, object_type)`. Preserve this distinction in reference mode.
Executing the two unmodified statistics method bodies against three synthetic
clips confirmed root means 60/7 (frame-weighted) and 40/3 (balanced), with a shared
object name in two datasets. Source file SHA-256:
`413a539e4ad4606e599c502758f9d1a199e5750feb44eebccb28f9845ca6961e`.
Evidence: `.runtime/training-stats-reference.json`. This verifies a source
behavior, not a pack implementation.

## Dataset preparation and annotation

`data_process/feature_extraction/extract_features.py` exposes the following
inputs; each must be represented in nodes or a documented node configuration:

- Dataset layout, exported motions and annotation metadata.
- Clip length and overlap derived from the longest training crop; truncation or
  overlapping-window mode.
- Topology path length and spectral frequencies; target skeleton diameter.
- Per-motion or rest-pose ground height.
- Activity, static-frame trimming, discontinuity and minimum-frame filters.
- Mixamo core-joint selection and minimum/maximum skeleton size.
- Explicit filtered clips/objects, reviewed head trims, activity exemptions and
  category groups.
- Worker count and previews with or without the ground/follow-camera view.

Metadata outputs include conditioning, captions, filtered-clip reports and
dataset summaries. The per-object cache has parameter hashing, stale-clip pruning
and atomic outputs. The processing implementation, cache invalidation and exact
filter order still need detailed reference comparisons.

Additional released stages identified by source inventory are motion export
(`motion_export/export_general.py`, `export_truebones.py`, `export_objaverse.py`),
joint annotation (`joint_annotation`), rendering, motion captioning, caption
rewriting, body-plan classification and reviewed annotation patches. These are
separate coverage requirements; feature extraction alone does not cover them.
Their full function/option audit remains open.

`vlm_caption/backends.py` supports local Qwen and OpenAI-compatible/Gemini APIs.
The pack must support local annotation offline and explicit provider selection
for equivalent remote workflows. No provider request is authorized by this
audit. Credentials must remain runtime-local and outside portable values.

## Integration constraints

User inputs stay JSON plus numeric arrays with no pickle. Released conditioning
and saved statistics use pickled NumPy dictionaries; adapting them requires a
trusted boundary, not enabling pickle in archive loaders. Training dependencies
must remain outside ComfyUI's inference environment. Dataset, configuration,
resume state and result values must stage through Cloud Offload without host
paths or live model objects. Cancellation must terminate any training/annotation
process tree and leave no partially published checkpoint.

Next design units: safe dataset/statistics contracts and deterministic processing;
isolated training runtime and checkpoint conversion; annotation/rendering stages.
All remain implementation and validation gaps under `GOAL.md`.
