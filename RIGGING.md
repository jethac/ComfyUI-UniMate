# Automatic rigging integration

Status: superseded scope investigation. The user clarified that the target is full UniMate technology coverage; see [COVERAGE.md](COVERAGE.md). Automatic rigging belongs to adjacent technology. The requirements below record the earlier proposal, not the current implementation mandate. No rigging implementation or end-to-end rigging validation exists.

## Required result

A connected ComfyUI graph accepts an unrigged generated mesh, predicts a skeleton and skin weights, generates UniMate motion for that skeleton, and exports an animated GLB. No manual file round trip or manual skeleton construction is required. Existing rigged assets remain valid inputs.

Preserve positions, topology, UVs, materials, embedded textures, vertex colours, and source coordinates through rigging. Add joint attributes, inverse binds, and a skeleton to the source asset rather than relying on a lossy whole-asset re-export. Prediction may use a normalized geometry copy; its transforms must be reversed explicitly.

Do not substitute a fixed skeleton, nearest-bone weights, or hand-authored fixture weights for learned automatic rigging in the end-to-end test. Reject invalid predicted hierarchies or unsupported joint counts with actionable diagnostics.

## Interfaces to inspect

- ComfyUI core emits `MESH` and typed 3D files. Inspect the pinned core schemas and file payloads before specifying an adapter.
- The installed Trellis2 wrapper emits `MESHWITHVOXEL` and has a `TRIMESH` conversion node. Connect its textured output without requiring users to save and reload a file.
- Rigging output must feed the existing `UNIMATE_ASSET` → `UNIMATE_RIG` path. Socket values crossing Cloud Offload boundaries contain data and bytes, never host paths or live backend objects.
- Backend binaries and model artifacts are explicitly installed. Runtime does not download weights or provision providers.

## Backend candidates

### UniRig

[Upstream](https://github.com/VAST-AI-Research/UniRig) provides skeleton and skinning prediction; its repository license is MIT. Upstream documents CUDA dependencies including sparse convolution and PyG extensions. Validate checkpoint and transitive dependency licenses separately.

[Existing ComfyUI wrapper](https://github.com/PozzettiAndrea/ComfyUI-UniRig) accepts `TRIMESH`. Its `UniRigAutoRig` node returns an FBX path and receives a dictionary of live models. Reusing its inference requires an adapter for portable values, asset preservation, cancellation, and backend/model installation. Do not assume its export preserves source topology or materials.

### SkinTokens C++

[skin-tokens.cpp](https://github.com/localai-org/skin-tokens.cpp), inspected at `37b28284d0015c4e61e2657072f3b0f5166af207`, provides learned skeleton and skinning inference through CPU/Vulkan and GGUF artifacts. Its CPU path is a candidate for headless stadia-testbed verification. Its README states atlas textures are not round-tripped, and unconstrained skeleton generation is experimental. Inspect prediction interfaces and preserve appearance in our own source-asset adapter before adopting it. Verify code, converted model, and transitive licenses separately.

## Verification required

1. Real backend inference on an unrigged redistributable character with useful geometry, not only the existing tiny rig fixtures.
2. A ComfyUI graph from generated-mesh socket data through rigging, preparation, motion, and export, with no hand-authored skeleton supplied.
3. Source appearance and geometry comparisons; rest-pose reconstruction and normalized finite skin weights.
4. Independent animation playback evaluation across all frames, including visible mesh deformation and valid joint transforms.
5. Headless execution on stadia-testbed using an actually supported backend. Record exact binary, model, precision, device, and source identities.
6. Cloud Offload input/model staging and animated-output retrieval for the expanded graph; exercise boundaries around rigging as well as animation.
7. Cancellation, missing artifacts, malformed output, cache invalidation, and cleanup checks.

Backend selection and the implementation plan remain pending inspection of the actual inference and export code. The existing rigged-GLB tests do not prove this workflow complete.
