# UniMate implementation plan

The user requested parallel implementation and mandatory Cloud Offload execution on 2026-09-30. This supersedes the earlier documentation-only instruction and DESIGN.md's deferred cloud scope.

## Shared contracts

All socket values are plain dictionaries. No dataclasses, NumPy arrays, local paths, or live model objects may cross a boundary. Binary payloads use bytes; arrays use a numeric NPZ payload loaded with `allow_pickle=False`.

- `UNIMATE_ASSET`: `{"schema":"unimate.asset.v1", "glb": bytes, "sha256": str, "name": str}`.
- `UNIMATE_RIG`: `{"schema":"unimate.rig.v1", "asset": asset, "rig_id": str, "conditioning": bytes, "mapping": dict}`. Conditioning contains numeric/string arrays only. Mapping carries original skeleton identity and reversible canonicalization.
- `UNIMATE_MODEL`: `{"schema":"unimate.model.v1", "bundle": bytes, "sha256": str, "name": str}`. A single safe `.unimate` ZIP contains model config, numeric normalization stats, EMA safetensors, local text encoder/tokenizer, and a manifest. Large boundary transfers are accepted for correctness; no reference to a local-only model path.
- `UNIMATE_MOTION`: `{"schema":"unimate.motion.v1", "rig_id": str, "features": bytes, "fps":30, "metadata": dict}`. Features NPZ contains float32 `features` of shape `(60,J,12)`.

Shared functions in `unimate_pack/contracts.py`: `make_asset(glb: bytes, name: str) -> dict`, `validate_asset(value: dict) -> None`, `make_rig(asset: dict, conditioning: bytes, mapping: dict) -> dict`, `validate_rig(value: dict) -> None`, `make_model(bundle: bytes, name: str) -> dict`, `validate_model(value: dict) -> None`, `make_motion(rig_id: str, features: bytes, metadata: dict) -> dict`, `validate_motion(value: dict, rig_id: str | None = None) -> None`, `encode_arrays(**arrays) -> bytes`, `decode_arrays(payload: bytes) -> dict[str,np.ndarray]`.

Subsystem APIs:

- `unimate_pack/assets.py`: `validate_glb(glb: bytes) -> dict` returns parsed metadata after checked supported-rig validation.
- `unimate_pack/blender.py`: `prepare_rig(asset: dict, facing: str, left_joint: str = "", right_joint: str = "") -> dict`; `export_glb(rig: dict, motion: dict) -> bytes`. Blender executable uses `UNIMATE_BLENDER`; task temp roots must be managed and cancellable. Worker script imports bpy only inside Blender.
- `unimate_pack/inference.py`: `load_model_bundle(path: str | Path) -> dict`; `generate_motion(model: dict, rig: dict, prompt: str, seed: int, guidance: float, normalization: str = "objaverse") -> dict`.
- `nodes.py`: five IDs exactly as DESIGN.md. `UniMateLoadRig.cloud_offload_assets(inputs)` declares the selected `asset` under category `__input__`; loader `bundle` is a registered `unimate` model asset. Export reports `3d` GLB plus `files` provenance.

## Ownership and verification

Agents write disjoint files and do not commit, dispatch helpers, or change another owner's files without coordination. Tests belong to the owning subsystem. Parent integrates and commits. Use test-first; retain failed-test evidence in task reports. Each task report names what was actually verified and outstanding integration concerns.

1. Contracts owner: contracts, GLB validator, dependency-light safety/round-trip tests. Numeric archives reject pickle, malicious entries, excess sizes, nonfinite values, and malformed structures. GLB checks one skin, connected valid joints, supported transforms, skin indices, buffers/accessors, and embedded data.
2. Rig owner: Blender process and worker, upstream canonicalization and motion recovery adapters, synthetic redistributable GLB fixture generator, Blender rest/deformation/animation tests. Preserve original asset topology/appearance and source coordinates. Prove rest-only preparation and both topology cases.
3. Inference owner: local safe bundle packer and runtime inference, pinned compatible upstream source, license notices, tests against the reference. ComfyUI selects devices; models unload; no runtime network. Exercise real official weights if available.
4. Nodes owner: five node schemas, managed filenames/outputs, API-format workflow, ComfyUI registration and node tests. All socket values conform to shared contracts; node import is dependency-light.
5. Cloud owner: selected input asset declarations/staging, bundle staging, output retrieval, generic protocol compatibility, runner prerequisites and end-to-end partition tests. No paid GPU provisioning required for local runner verification.
6. Parent: package metadata, precise setup docs, integration and review. Verify the installed nodes through a real headless ComfyUI API workflow, crossing custom values through actual Cloud Offload partition bundles, with Blender and official model inference. A protocol encode-only check cannot establish runner support.

## Rulings

- Work on branch `feat/unimate-cloud-nodes` in the fresh project checkout. Parallel owners have disjoint paths; no extra worktrees required.
- Plain dictionaries replace design dataclasses to use Cloud Offload's existing safe bytes/dict transport without arbitrary plugin deserialization.
- A self-contained model bundle replaces multiple local references. This preserves offline inference and arbitrary box boundaries while using existing model staging. It costs boundary bandwidth; future optimization must preserve portability.
- Cloud Offload is required in the initial implementation. Missing runner dependencies fail preflight/runner preparation with explicit remedies; never silently execute locally.
- README language is terse and factual, as requested.
- Official normalization has three families. Expose `normalization` on Generate with `objaverse`, `mixamo`, and `truebones`; record the choice in provenance.
- Append animation to the original GLB instead of exporting rebuilt geometry. Use Blender to validate preparation and evaluate the result; compare canonicalization and recovery math against upstream.
