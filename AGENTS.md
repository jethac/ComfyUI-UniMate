# Agent instructions

## Project scope

Build a ComfyUI node pack covering UniMate's technology and released capabilities. COVERAGE.md records the source audit and omissions; DESIGN.md defines the existing implementation and release gates. Do not silently narrow coverage to basic text generation or substitute an unrelated rigging backend project. Distinguish verified behavior, released-but-unintegrated code, and paper-described capabilities without located implementation.

Implementation is now authorized. All five nodes must work inside ComfyUI Cloud Offload partitions, including input/model staging and animated output retrieval. Keep portable values free of local paths and live model objects.

## Engineering rules

- Keep changes small and at the layer that owns the behavior. Nodes translate ComfyUI inputs; rig code owns coordinate conversion; inference code owns model conditioning and sampling; export code owns animated GLB output.
- Preserve skin weights, inverse bind matrices, joint identity, topology, materials, and textures. Never silently flatten a rig into a static mesh.
- Use explicit versioned rig and motion contracts. A motion must identify the exact prepared skeleton it was generated for. Reject mismatches before export.
- Reuse pinned UniMate inference and canonicalization math. Do not implement a second interpretation of root motion, quaternion order, or relative rotations without reference comparisons.
- Use ComfyUI's current extension API and model/device management. Do not force a torch, CUDA, or torchvision version, choose GPUs independently, or install the full upstream training environment into ComfyUI.
- Run Blender as an explicit external process. Never import bpy into the ComfyUI server. Invoke argument arrays without a shell; support Windows paths and terminate children on cancellation.
- Do not add automatic downloads, telemetry, remote captioning, or provider calls. Setup downloads require explicit user action. Inference must work offline with installed artifacts.
- Treat assets and archives as untrusted input. Resolve inputs and outputs through ComfyUI's permitted paths; validate extensions, containment, sizes, indices, and finite numeric values. Do not enable execution of embedded Blender scripts.
- Use JSON and numeric arrays without pickle for this pack's persisted data. Never load user-supplied NumPy object arrays with allow_pickle=True. Convert legacy upstream conditioning only at the trusted adapter boundary.
- Load only explicitly selected model artifacts. Prefer restricted checkpoint loading, document any trusted conversion, and never silently fall back from EMA to raw weights.
- Do not vendor code or dependencies without checking their license and retaining required notices. The MIT license on this repository does not relicense third-party assets.

## Verification

- Test skeleton ordering, transform reversal, rest-pose preservation, skinning, motion/rig identity, and export playback. Compare adapters against the pinned upstream reference, not just against themselves.
- Use small synthetic rigs in automated tests. Use redistributable real assets for integration checks; record their provenance and licenses.
- Keep dependency-light contract tests separate from GPU and Blender checks. Record exact model, upstream, ComfyUI, Blender, device, and precision versions for integration evidence.
- Exercise cancellation, invalid rigs, missing models, paths with spaces, and memory unload behavior before release.
- Do not claim support for hardware or operating systems that have not passed the documented checks. Do not invent latency or VRAM figures.
- Update README.md when implemented scope or prerequisites change. Keep design decisions and deviations in DESIGN.md; avoid duplicating the specification elsewhere.
- All READMEs must be terse and precise. No self-congratulatory language, marketing filler, or claims beyond verified behavior.

## Git and collaboration

Use short direct commit subjects. Do not commit model weights, generated assets, caches, local settings, credentials, or dependency environments. Preserve unrelated user changes. Publishing, provider spending, or messages to others require authorization in the current conversation.

Push authorized completed work directly to the default branch. Do not create pull requests unless the user explicitly asks for one.

ComfyUI-related repositories belong under the `jethac` GitHub account. `distiller` remains under `splatterfacegames`.
