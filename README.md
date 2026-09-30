# ComfyUI-UniMate

Text-guided skeletal animation for rigged 3D characters in ComfyUI, based on [UniMate](https://github.com/Friedrich-M/UniMate).

**Status: design only.** This repository contains the proposed node interfaces and implementation requirements. There are no working nodes, installation instructions, or downloadable workflows yet.

The first release will take a rigged GLB, prepare its skeleton, generate motion from a prompt, and export an animated GLB. It targets artists building reusable 3D assets and game animations, including humanoids, animals, and articulated objects.

## Planned workflow

```text
Load Rigged GLB → Prepare UniMate Rig ───────────────┐
                          │                        │
Load UniMate Model ────────┴→ Generate UniMate Motion│
                                      │            │
                                      └→ Export UniMate GLB
```

Generation exposes a prompt, seed, and guidance scale. Export writes a playable animated asset. The first version uses external Blender for rig preparation and export, with inference inside ComfyUI.

A generated mesh needs a skeleton and skin weights before it can enter this workflow. UniMate does not supply automatic rigging. Existing ComfyUI mesh sockets are insufficient unless they preserve the complete rig.

## Scope

- First release: one rigged GLB, one skin, one generated animation, local inference, animated GLB export.
- Later: motion preview, imported motion constraints, in-betweening, joint editing, and motion extension.
- Outside the first release: FBX, automatic rigging, retargeting between different rigs, training, and cloud execution.

## Upstream release

The [official model release](https://huggingface.co/Linzhan/UniMate) currently recommends `unimate_uniml3d_f60_v2`: 60 frames at 30 fps, a 74.1M-parameter denoiser, and a separate FLAN-T5-base text encoder. Quality and practical hardware requirements must be measured before release.

Upstream documentation and code differ on importing a rig without animation. At the revision examined for this design, `preprocess_char.py` includes a rest-only conditioning path. This project will test that path rather than require users to supply a dummy clip. See [DESIGN.md](DESIGN.md) for the source revision and acceptance criteria.

Models and text encoders will be installed explicitly. Queueing a workflow must not trigger network downloads.

## Development

Read [DESIGN.md](DESIGN.md) for the architecture, node contracts, release gates, and implementation sequence. Read [AGENTS.md](AGENTS.md) before making changes.

## License and attribution

This node pack is licensed under the [MIT License](LICENSE). It is an independent integration, not an official UniMate release. UniMate's code and checkpoints have their own license files; source datasets and imported assets retain their respective terms. Do not bundle training datasets or third-party characters with this pack.

UniMate: One Unified Model to Animate Diverse Skeletons, Linzhan Mou et al., SIGGRAPH Asia 2026. [Project](https://linzhanmou.com/unimate/) · [Paper](https://arxiv.org/abs/2609.05415) · [Code](https://github.com/Friedrich-M/UniMate) · [Models](https://huggingface.co/Linzhan/UniMate).
