# ComfyUI-UniMate

UniMate skeletal animation nodes for ComfyUI. Input: a rigged GLB and a motion prompt. Output: an animated GLB and generation metadata.

| Node | Function |
| --- | --- |
| Load Rigged GLB | Read a self-contained skinned asset from ComfyUI input |
| Prepare UniMate Rig | Build topology conditioning from the bind pose |
| Load UniMate Model | Read an installed `.unimate` bundle |
| Generate UniMate Motion | Sample 60 frames at 30 fps |
| Export UniMate GLB | Add animation to the original GLB; save provenance JSON |

Automatic rigging and retargeting are outside this pack. Mesh data, skin weights, inverse binds, materials, and textures remain in the original asset. Existing source clips are ignored and replaced by one generated clip.

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

The supported model is official `unimate_uniml3d_f60_v2`, graph attention/AdaLN, with FLAN-T5-base. Download the pinned files explicitly:

```sh
hf download Linzhan/UniMate --revision 387a344c3031299bc25fcbef35d36bd186d5afe7 --include 'unimate_uniml3d_f60_v2/config.json' 'unimate_uniml3d_f60_v2/dataset_stats.npy' 'unimate_uniml3d_f60_v2/checkpoints/checkpoint_step_100000.pt' --local-dir models/source/unimate
hf download google/flan-t5-base --revision 7bcac572ce56db69c1ea7c8af255c5d7c9672fc2 --include 'config.json' 'model.safetensors' 'spiece.model' 'tokenizer.json' 'tokenizer_config.json' 'special_tokens_map.json' --local-dir models/source/flan-t5-base
python tools/build_bundle.py --checkpoint models/source/unimate/unimate_uniml3d_f60_v2/checkpoints/checkpoint_step_100000.pt --config models/source/unimate/unimate_uniml3d_f60_v2/config.json --stats models/source/unimate/unimate_uniml3d_f60_v2/dataset_stats.npy --text-encoder models/source/flan-t5-base --output models/unimate-v2.unimate --model-revision 387a344c3031299bc25fcbef35d36bd186d5afe7 --text-revision 7bcac572ce56db69c1ea7c8af255c5d7c9672fc2 --trust-legacy-stats
```

Copy the resulting bundle to `ComfyUI/models/unimate/`. Conversion selects EMA weights, validates the known legacy statistics, and writes safetensors plus numeric statistics. Runtime loading does not use pickle. The bundle includes the local tokenizer and encoder; inference works offline.

## Use

Put a rigged GLB under `ComfyUI/input/`, then connect Load Rigged GLB → Prepare UniMate Rig → Generate UniMate Motion → Export UniMate GLB. Connect Load UniMate Model to Generate. [API workflow](examples/unimate_api.json).

Choose the source facing direction explicitly. Joint-pair facing requires raw left/right joint names. Generate exposes prompt, seed, guidance, and normalization family (`objaverse`, `mixamo`, `truebones`). Guidance 1 is unconditional, matching upstream. Solver settings follow the pinned model. A clip has 60 keys at `i/30` seconds and is not automatically looped.

## Supported assets

One skin, one connected skeleton with 5–70 joints, triangle primitives, dense accessors, up to four skin influences, embedded PNG/JPEG textures, and positive uniform scales. Unsupported content fails validation. No FBX, sparse/compressed geometry, morph targets, unskinned scene meshes, glTF extensions, shear, negative scale, or nonuniform scale. Asset limit: 256 MiB.

## Cloud Offload

All five nodes can execute in a [ComfyUI-Cloud-Offload](https://github.com/jethac/ComfyUI-Cloud-Offload) partition, using the [cloud-offload coordinator and worker service](https://github.com/jethac/cloud-offload). All four custom socket values are portable dictionaries containing bytes; the model bundle crosses in full when its loader is outside the box. Transfers include roughly 706 MiB of model data.

The runner needs Blender, this pack, its Python dependencies, and the Cloud Offload input-staging/output-retrieval changes described in [deploy/README.md](deploy/README.md). The default runner without those changes is insufficient. [DESIGN.md](DESIGN.md) defines the contracts; [VALIDATION.md](VALIDATION.md) records verification and limits.

## License

[MIT](LICENSE). Independent integration of [UniMate](https://github.com/Friedrich-M/UniMate), Linzhan Mou et al., SIGGRAPH Asia 2026. Vendored code retains its notices. UniMate weights are MIT; FLAN-T5 is Apache-2.0. Imported assets retain their source licenses.
