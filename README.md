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
| Generate UniMate Batch | Generate prompt/repetition cases as a typed motion list |
| Canonical UniMate Asset | Apply prepared coordinates to the original asset without rewriting skin or mesh data |

The target is complete UniMate capability coverage. Generation, in-betweening, editing and expansion are implemented. All four released model families passed checkpoint reconstruction, free and constrained inference, and Blender export on synthetic rigs. Expanded ComfyUI server and Cloud Offload checks remain pending. Other gaps are recorded in [COVERAGE.md](COVERAGE.md). Input currently requires a rigged GLB. Mesh data, skin weights, inverse binds, materials, and textures remain in the original asset. Export replaces source clips with the selected motion.

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

Copy the resulting bundle to `ComfyUI/models/unimate/`. Conversion selects EMA weights, validates the known legacy statistics, and writes safetensors plus numeric statistics. Runtime loading does not use pickle. The bundle includes the local tokenizer and encoder; inference works offline.

For the other families, select their matching config, statistics and checkpoint from model revision `971da7cfc1c8d99c2af6c00be9d2ed5700f99073`. Mixamo has 22 joint slots and depth capacity 7; preview has 61 slots; v2 variants have 71. Skeletons must fit the checkpoint capacity and the current 70-joint rig contract. Mixamo requires `mixamo` normalization. Conversion rejects unrecognized legacy statistics and incompatible checkpoint inventories.

## Use

Put a rigged GLB under `ComfyUI/input/`, then connect Load Rigged GLB → Prepare UniMate Rig → Generate UniMate Motion → Export UniMate GLB. Connect Load UniMate Model to Generate. [API workflow](examples/unimate_api.json).

Choose the source facing direction explicitly. Joint-pair facing requires raw left/right joint names. Generate exposes prompt, seed, guidance, and normalization family (`objaverse`, `mixamo`, `truebones`). Guidance 1 is unconditional, matching upstream. Solver settings follow the pinned model. A clip has 60 keys at `i/30` seconds and is not automatically looped.

With `joint_pair`, optional left/right shoulder names add a second lateral pair. Set `body_axis` for a head-to-tail pair instead of a lateral pair. The selected mode is retained for source-motion extraction. Four-joint and body-axis preparation passed external Blender rest-preservation checks; server/cloud checks for these options remain pending.

Canonical UniMate Asset returns a portable asset in the prepared coordinate system. A scene-parent transform preserves the source binary, skin, materials and textures. It has a new asset identity; prepare it again before generating motion for it. Independent coordinate and skinning checks passed. Server/cloud execution remains pending.

## Supported assets

Save UniMate Motion writes `.npz` archives. Copy an archive into ComfyUI input and select it in Load UniMate Motion to reuse it as a reference. In-betweening accepts comma-separated frame indices (`0,-1` preserves the first and last frame); editing accepts original or cleaned joint names. Reference clips must belong to the same prepared rig and fit the current 60-frame model window. Both modes require guidance greater than 1.

Extract UniMate Motion selects a zero-based animation clip from the prepared rig's source GLB. It resamples at 30 fps and produces F−1 feature frames from F poses, following upstream velocity encoding. LINEAR, STEP and CUBICSPLINE channels are supported. Animated bone lengths must match the prepared skeleton. The motion retains its initial canonical XZ position for export.

Expand UniMate Motion accepts a JSON array of prompts in segment order. Each segment has 60 frames; overlap must be 1–59 frames. With N prompts and overlap O, the result has `60 + (60 - O) * (N - 1)` frames. Seeds increment per segment modulo uint64. Later segments preserve the preceding tail; duplicated overlap frames are omitted from the output. Expansion requires guidance greater than 1.

Generate UniMate Batch accepts 1–32 JSON prompts and 1–64 repetitions, capped at 256 cases. Cases run in prompt order, then repetition order; seeds increment modulo uint64. Its typed motion list feeds ComfyUI's normal list execution, including export and numeric saving. Sampling runs one case at a time. A four-case batch passed ComfyUI server execution, export, artifact retrieval, provenance and independent playback checks. Cloud Offload list transport still requires validation.

One skin, one connected skeleton with 5–70 joints, triangle primitives, dense accessors, up to four skin influences, embedded PNG/JPEG textures, and positive uniform scales. Unsupported content fails validation. No FBX, sparse/compressed geometry, morph targets, unskinned scene meshes, glTF extensions, shear, negative scale, or nonuniform scale. Asset limit: 256 MiB.

## Cloud Offload

The original five-node generation workflow passed execution in a [ComfyUI-Cloud-Offload](https://github.com/jethac/ComfyUI-Cloud-Offload) partition, using the [cloud-offload coordinator and worker service](https://github.com/jethac/cloud-offload). Expanded node workflows still require verification. All four custom socket values are portable dictionaries containing bytes; the model bundle crosses in full when its loader is outside the box. Transfers include roughly 706 MiB of model data.

The runner needs Blender, this pack, its Python dependencies, and the Cloud Offload input-staging/output-retrieval changes described in [deploy/README.md](deploy/README.md). The default runner without those changes is insufficient. [DESIGN.md](DESIGN.md) defines the contracts; [VALIDATION.md](VALIDATION.md) records verification and limits.

## License

[MIT](LICENSE). Independent integration of [UniMate](https://github.com/Friedrich-M/UniMate), Linzhan Mou et al., SIGGRAPH Asia 2026. Vendored code retains its notices. UniMate weights are MIT; FLAN-T5 is Apache-2.0. Imported assets retain their source licenses.
