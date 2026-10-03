# ComfyUI-UniMate

UniMate skeletal animation nodes for ComfyUI. Input: a rigged GLB and a motion prompt. Output: an animated GLB and generation metadata.

| Node | Function |
| --- | --- |
| Load Rigged GLB | Read a self-contained skinned asset from ComfyUI input |
| Prepare UniMate Rig | Build topology conditioning from the bind pose |
| Load UniMate Model | Read an installed `.unimate` bundle |
| Generate UniMate Motion | Sample 60 frames at 30 fps |
| Export UniMate GLB | Add animation to the original GLB; save provenance JSON |
| In-between UniMate Motion | Preserve selected reference frames and generate the transition |
| Edit UniMate Motion | Preserve selected joints and regenerate the remaining motion |
| Load UniMate Motion | Load a numeric motion archive with canonical rig identity |
| Save UniMate Motion | Save a numeric motion archive for later reuse |
| Expand UniMate Motion | Generate an ordered prompt chain with constrained segment overlaps |
| Extract UniMate Motion | Resample a source GLB clip into the prepared rig's motion features |
| Generate UniMate Batch | Generate rig/prompt/repetition cases with parallel motion and rig lists |
| Combine UniMate Rigs | Collect prepared rigs in order for batch generation |
| Canonical UniMate Asset | Apply prepared coordinates to the original asset without rewriting skin or mesh data |
| UniMate Rig Conditioning | Expose numeric topology conditioning, its hash and canonical rig identity |
| Export UniMate FBX | Bake animation to binary FBX and verify skinning after Blender reimport |
| Recover UniMate Skeleton | Recover canonical joint positions through FK or RIC |
| Render UniMate Skeleton | Render every recovered frame as a ComfyUI IMAGE batch |
| Foot Lock UniMate Motion | Detect ground contacts, anchor limb chains and return corrected motion/report |
| Build UniMate Dataset | Collect paired rig/motion lists into a labelled numeric shard |
| Load UniMate Dataset | Read a `.unimatedata` shard from ComfyUI input |
| Save UniMate Dataset | Save a portable numeric shard |
| UniMate Dataset Statistics | Compute training-only normalization and source provenance |
| Load UniMate Statistics | Read a `.unimatestats` archive from ComfyUI input |
| Save UniMate Statistics | Save normalization arrays and provenance |
| Split UniMate Dataset | Assign seeded clip/object holdouts; report unmatched explicit objects |
| Plan UniMate Sampling | Build one/two-level weights and replacement indices for an epoch |
| Build UniMate Text Cache | Encode dataset joint names and captions with the installed bundle's encoder |
| Prepare UniMate Training Sample | Bind a training clip, statistics and text cache; augment, crop and normalize |
| Collate UniMate Training Samples | Collect a sample list into a portable batch with source masks and padding |
| Configure UniMate Training | Bind architecture, losses, optimizer and epoch sampling to prepared artifacts |
| Train UniMate | Run optimizer groups from scratch or selected installed weights; return checkpoint and progress |
| Load UniMate Training Checkpoint | Load a numeric `.unimatetrain` checkpoint from managed inputs |
| Save UniMate Training Checkpoint | Save a numeric checkpoint to managed outputs |
| Export UniMate Inference Weights | Select raw or EMA denoiser weights from a training checkpoint and save `.unimateweights` |
| Load UniMate Inference Weights | Load `.unimateweights` from managed inputs |

The target is complete UniMate capability coverage. Generation, in-betweening, editing and expansion are implemented. All four released model families passed checkpoint reconstruction, free and constrained inference, and Blender export on synthetic rigs. Expanded inference passed headlessly through ComfyUI on stadia-testbed; expanded Cloud Offload checks remain pending. Other gaps are recorded in [COVERAGE.md](COVERAGE.md). Input currently requires a rigged GLB. Mesh data, skin weights, inverse binds, materials, and textures remain in the original asset. Export replaces source clips with the selected motion.

## Install

Use ComfyUI's Python environment. Python 3.11+ and ComfyUI's current extension API are required. Tested Blender version: 5.1.1.

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/jethac/ComfyUI-UniMate.git comfy-unimate
cd comfy-unimate
python -m pip install -r requirements.txt
```

Set `UNIMATE_BLENDER` to Blender's executable before starting ComfyUI. Windows example:

```powershell
$env:UNIMATE_BLENDER = 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe'
```

The pack does not install Blender or download models during execution.

## Model bundle

Conversion accepts the official Mixamo, uniml3d preview, v2 and v2 full-cross-attention configs, all with FLAN-T5-base. The example below uses `unimate_uniml3d_f60_v2`, graph attention/AdaLN. Download the pinned files explicitly:

```sh
hf download Linzhan/UniMate --revision 387a344c3031299bc25fcbef35d36bd186d5afe7 --include 'unimate_uniml3d_f60_v2/config.json' 'unimate_uniml3d_f60_v2/dataset_stats.npy' 'unimate_uniml3d_f60_v2/checkpoints/checkpoint_step_100000.pt' --local-dir models/source/unimate
hf download google/flan-t5-base --revision 7bcac572ce56db69c1ea7c8af255c5d7c9672fc2 --include 'config.json' 'model.safetensors' 'spiece.model' 'tokenizer.json' 'tokenizer_config.json' 'special_tokens_map.json' --local-dir models/source/flan-t5-base
python tools/build_bundle.py --checkpoint models/source/unimate/unimate_uniml3d_f60_v2/checkpoints/checkpoint_step_100000.pt --config models/source/unimate/unimate_uniml3d_f60_v2/config.json --stats models/source/unimate/unimate_uniml3d_f60_v2/dataset_stats.npy --text-encoder models/source/flan-t5-base --output models/unimate-v2.unimate --model-revision 387a344c3031299bc25fcbef35d36bd186d5afe7 --text-revision 7bcac572ce56db69c1ea7c8af255c5d7c9672fc2 --trust-legacy-stats
```

Copy the resulting bundle to `ComfyUI/models/unimate/`. Conversion selects EMA weights by default; `--weights raw` selects raw weights explicitly. It validates the known legacy statistics and writes safetensors plus numeric statistics. Runtime loading does not use pickle. The bundle includes the local tokenizer and encoder; inference works offline.

For the other families, select their matching config, statistics and checkpoint from model revision `971da7cfc1c8d99c2af6c00be9d2ed5700f99073`. Mixamo has 22 joint slots and depth capacity 7; preview has 61 slots; v2 variants have 71. Skeletons must fit the checkpoint capacity and the current 70-joint rig contract. Mixamo requires `mixamo` normalization. Conversion rejects unrecognized legacy statistics and incompatible checkpoint inventories.

## Use

Put a rigged GLB under `ComfyUI/input/`, then connect Load Rigged GLB → Prepare UniMate Rig → Generate UniMate Motion → Export UniMate GLB. Connect Load UniMate Model to Generate. [API workflow](examples/unimate_api.json).

Choose the source facing direction explicitly. Joint-pair facing requires raw left/right joint names. Generate exposes prompt, seed, guidance, and normalization family (`objaverse`, `mixamo`, `truebones`). Guidance 1 is unconditional, matching upstream. Solver settings follow the pinned model. A clip has 60 keys at `i/30` seconds and is not automatically looped.

With `joint_pair`, optional left/right shoulder names add a second lateral pair. Set `body_axis` for a head-to-tail pair instead of a lateral pair. The selected mode is retained for source-motion extraction. Four-joint and body-axis preparation passed external Blender rest-preservation checks; server/cloud checks for these options remain pending.

Canonical UniMate Asset returns a portable asset in the prepared coordinate system. A scene-parent transform preserves the source binary, skin, materials and textures. It has a new asset identity; prepare it again before generating motion for it. Independent coordinate and skinning checks passed. Server/cloud execution remains pending.

UniMate Rig Conditioning returns `UNIMATE_CONDITIONING`: a versioned dictionary with numeric archive bytes, their SHA-256, rig identity and coordinate mapping. It exposes the prepared topology data without the source GLB or live model objects. Server/cloud execution remains pending.

## Supported assets

Save UniMate Motion writes `.npz` archives. Copy an archive into ComfyUI input and select it in Load UniMate Motion to reuse it as a reference. In-betweening accepts comma-separated frame indices (`0,-1` preserves the first and last frame); editing accepts original or cleaned joint names. Reference clips must belong to the same prepared rig and fit the current 60-frame model window. Both modes require guidance greater than 1.

Extract UniMate Motion selects a zero-based animation clip from the prepared rig's source GLB. It resamples at 30 fps and produces F−1 feature frames from F poses, following upstream velocity encoding. LINEAR, STEP and CUBICSPLINE channels are supported. Animated bone lengths must match the prepared skeleton. The motion retains its initial canonical XZ position for export.

Expand UniMate Motion accepts a JSON array of prompts in segment order. Each segment has 60 frames; overlap must be 1–59 frames. With N prompts and overlap O, the result has `60 + (60 - O) * (N - 1)` frames. Seeds increment per segment modulo uint64. Later segments preserve the preceding tail; duplicated overlap frames are omitted from the output. Expansion requires guidance greater than 1.

Generate UniMate Batch accepts prepared rigs, 1–32 JSON prompts and 1–64 repetitions, capped at 256 cases. Cases run in rig, prompt, then repetition order; seeds increment modulo uint64. Sampling runs one case at a time. Model and sampling settings must each contain one value. Combine UniMate Rigs collects two rig lists and can be chained. Connect the batch's matching rigs output and motion output to export; each motion keeps its own skeleton. See [multi-rig API workflow](examples/unimate_multi_rig_api.json).

Eight cases across five- and seven-joint synthetic rigs passed Windows and headless stadia server execution. The partition handler staged the model/assets, generated the cases and restored both paired lists; all 16 original/restored exports matched across 960 frames on each platform. Provider dispatch and deployed containers remain unverified.

Foot Lock UniMate Motion applies the contact filter and damped limb IK described in the paper's Appendix E.5. Empty `joint_names` selects distal canonical foot names; explicit names are comma-separated. Overlapping correction chains fail. Root channels and unrelated chains are preserved. Corrections blend over five boundary frames; unreachable anchors are reported. The numeric workspace estimate is capped at 512 MiB. Numeric, Blender playback, Windows server and Windows/headless stadia partition-handler checks passed on a 60-frame synthetic legged clip. Both worker exports passed independent playback and exact restored-skinning checks.

One skin, one connected skeleton with 5–70 joints, triangle primitives, dense accessors, up to four skin influences, embedded PNG/JPEG textures, and positive uniform scales. Unsupported content fails validation. No FBX, sparse/compressed geometry, morph targets, unskinned scene meshes, glTF extensions, shear, negative scale, or nonuniform scale. Asset limit: 256 MiB.

Export UniMate FBX writes binary FBX and provenance JSON. The external Blender job bakes at 30 fps without leaf bones, embeds images, reimports at frame zero and checks bone identity and evaluated skinning for every frame. Constant glTF base-color multipliers are folded into 16-bit PNG textures; embedded pixels are checked after reimport. Windows and headless stadia partition-handler export passed on synthetic assets. Other material expressions remain unverified. FBX input is not supported.

## Skeleton recovery

Skeleton recovery exposes canonical joint positions, ordered parents and joint names in a portable `UNIMATE_SKELETON` value with the motion's rig identity. `fk` uses rest offsets and recovered rotations; `ric` uses facing-relative position channels. Render UniMate Skeleton returns all frames as an IMAGE batch, with front, side or top projection and fixed bounds across the clip. Connect it to ComfyUI Preview Image or Save Image. Frames remain at 30 fps; the IMAGE socket does not carry timing. Output allocation is capped at 256 MiB; reduce resolution for long clips.

## Datasets

Build Dataset collects equally sized rig/motion lists with exact rig identities.
Its `labels` accepts `[]` for defaults or one JSON object per pair, with optional
`id`, `dataset_type`, `object_type`, `caption` and `split` fields. Statistics uses
only `train` clips; `per_dataset`, `balanced` and `tie_std` expose the released
normalization options. Saved shards preserve original numeric payloads and labels.

Load Dataset accepts numeric topologies with 2–4096 joints; Build Dataset uses
the existing mesh adapter's limits. These nodes provide data and normalization;
direct and Windows/headless stadia partition-handler workflows passed, including
archive staging, capture/restore and file retrieval.

Split Dataset recomputes membership across all input clips. Defaults hold out
object types for Truebones/Objaverse and clips for Mixamo. Its `options` accepts
`modes` and `explicit_eval_objects` mappings; explicit object lists override the
ratio and cannot empty a dataset's training set. Plan Sampling returns portable
weights and indices for one epoch, using only training clips. `two_level` adds
dataset balancing to object balancing.

Build Text Cache encodes captions and joint names using the selected installed
model. Prepare Training Sample applies augmentation, cropping and normalization
to a training clip with matching statistics and text-cache provenance. All seven
augmentation choices and both conditioning modes passed Windows and headless
stadia partition-handler capture/restore, model staging and file retrieval.
Collate Training Samples collects execution-list values into a portable batch.
Numeric restoration matches the pinned source collator, including variable joint
counts, caption lengths and spectral widths. Client/runner codec round trips
passed. Windows and headless stadia server/partition-handler workflows collected
18 samples into one batch, restored it and preserved reversed sample ordering.
Configure Training takes a dataset, statistics and text cache plus JSON options.
It derives text dimensions from the cache and records artifact identities. Options
include `model`, `optimizer`, `paradigm`, `loss`, `sample`, `sampling`, `batch_size`,
`drop_last` and `seed`. Train runs a selected number of optimizer updates on
ComfyUI's CPU/CUDA device. Connect its checkpoint to a subsequent Train node to
resume with the same job and runtime. Save/Load Training Checkpoint persists this
value as `.unimatetrain`, using JSON and safetensors. Windows and headless stadia
CPU server/partition-handler checks verified socket resume, file staging and
resume, and client checkpoint retrieval against uninterrupted training in each
runtime. See [checkpoint validation](docs/2026-10-03-training-checkpoint-io-validation.md).
Connect an installed Model Loader value to `initialization` on both Configure
Training and Train to start from that bundle's explicit raw/EMA weights. Configure
derives the architecture and rejects conflicting options; resume requires the
same initialization bundle. Use a sufficient workspace on both nodes; the v2
installed-model checks use 32,768 MiB. Optimizer and EMA state start fresh unless
a training checkpoint is connected. The converter's `--weights raw` selects raw
weights; its default is `--weights ema`, with no fallback.
Installed v2 raw/EMA training, socket/file resume and client retrieval passed
Windows and stadia CPU server/handler checks; see [initialization validation](docs/2026-10-03-training-initialization-validation.md).
Current execution is single-process, using balanced epoch sampling.
Export Inference Weights takes a training checkpoint and an explicit `raw` or
`ema` selection. It saves denoiser tensors and training-job metadata without
optimizer, scaler or RNG state. Load Inference Weights reads this file from
managed inputs. Both nodes require sufficient workspace (32,768 MiB for the
installed v2 checks). `.unimateweights` is an intermediate artifact: Model Loader
does not accept it. Assembly with normalization statistics and a text encoder,
and generation from these artifacts remain open. Selected-weight export,
declared-input reload and client retrieval passed headless stadia server/worker
checks; see [selected-weight validation](docs/2026-10-03-inference-weights-validation.md).
Distributed/unbalanced loaders, learned-variance backbone output and the full
server/worker training matrix remain open.
Raw-data curation remains open.

## Cloud Offload

The original five-node generation workflow passed execution in a [ComfyUI-Cloud-Offload](https://github.com/jethac/ComfyUI-Cloud-Offload) partition, using the [cloud-offload coordinator and worker service](https://github.com/jethac/cloud-offload). Expanded node workflows still require verification. Custom socket values are portable dictionaries containing bytes; the model bundle crosses in full when its loader is outside the box. Transfers include roughly 706 MiB of model data.

The runner needs Blender, this pack, its Python dependencies, and the Cloud Offload input-staging/output-retrieval changes described in [deploy/README.md](deploy/README.md). The default runner without those changes is insufficient. [DESIGN.md](DESIGN.md) defines the contracts; [VALIDATION.md](VALIDATION.md) records verification and limits.

## License

[MIT](LICENSE). Independent integration of [UniMate](https://github.com/Friedrich-M/UniMate), Linzhan Mou et al., SIGGRAPH Asia 2026. Vendored code retains UniMate, [SiT](unimate_pack/_vendor/LICENSE-SiT) and [guided-diffusion](unimate_pack/_vendor/LICENSE-guided-diffusion) notices. UniMate weights are MIT; FLAN-T5 is Apache-2.0. Imported assets retain their source licenses.
