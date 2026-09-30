# Validation

Recorded 2026-09-30. Implementation is present; release gates remain open. Evidence covers Windows GPU execution, actual ComfyUI partition bridges, authenticated localhost transport, and Linux CPU/Blender CI. No paid provider was provisioned.

## Runtime

| Component | Recorded version |
| --- | --- |
| Host | Windows build 26200 |
| ComfyUI | `84ba85773925f071c516f0208184773802b4d44a` |
| Server/reference Python | 3.11.9 |
| Blender | 5.1.1, build `b70da489d7f4`; bundled Python 3.13.9 |
| Device / precision | NVIDIA GeForce RTX 5060 Ti selected by ComfyUI; float32 inference |
| torch | 2.11.0+cu128 |
| NumPy / SciPy | 2.4.3 / 1.17.1 |
| transformers / sentencepiece | 5.8.0 / 0.2.1 |
| torch-geometric / torchdiffeq | 2.8.0.post1 / 0.2.5 |
| safetensors / einops | 0.8.0 / 0.8.2 |
| UniMate source | `5d6aabedd947297b5ba6706d8e9113e68c0c3e4f` |
| UniMate model Hub revision | `387a344c3031299bc25fcbef35d36bd186d5afe7` |
| FLAN-T5-base Hub revision | `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` |

The final licensed bundle is 739,698,499 bytes, SHA-256 `3d4420752e64b873f98c8aec2d6f01edf7be861c704920dfc095bd1d700664b8`. Real model and workflow checks were repeated with this archive after adding its required license file.

## Recorded checks

Suites overlap; counts are not additive. Skips are not passed checks. The repository run includes the final required-mapping and process-cleanup regressions.

| Check | Result | Scope |
| --- | --- | --- |
| Contract/GLB owner suite | 116 passed | Safe archives, numeric/graph/transform/skin/material/image checks |
| Repository Blender/reference run | 184 passed, 6 skipped, 6 subtests passed | External model/workflow cases remain opt-in |
| Node suite | 14 passed | Real latest API; isolated backend plumbing/file regressions |
| Rig suite | 31 passed, 1 skipped | Blender, pinned references, two topologies; actual POSIX lifecycle skipped on Windows |
| Inference dependency/reference suite | 24 passed | Conversion, normalization, masks/topology, RNG/cancellation/cache |
| Current licensed model integration | 1 passed | Real EMA/local T5, CUDA management, network blocked, reference comparison, unload/reload |
| Cloud extension suite | 115 passed, 1 skipped | Selected assets, transport, restoration |
| Cloud partition compiler | 33 passed | Partition compilation |
| Cloud coordinator suite | 1237 passed, 6 skipped | Coordinator regressions after review fixes |
| Authenticated localhost transport | Passed separately | Real HTTP transfer, source resolution, Worker staging/return |
| Current licensed real workflow | 1 passed | All five nodes, four boundary types, six output downloads |
| Linux CI repository suite | 170 passed, 20 skipped | Ubuntu 24.04, Python 3.11.16, pinned ComfyUI; no model weights |
| Linux Blender integration | 10 passed | Blender 5.1.1, both topologies, actual POSIX process-tree cleanup |

All five nodes also registered through real ComfyUI `load_custom_node`. A model-loader filename collision checks exact `unimate` category selection against a different checkpoint. Export cancellation signals the real ComfyUI interrupt flag during fsync and verifies no output pair is published.

## Rig and reference evidence

Fixtures are original MIT procedural GLBs: a five-joint chain and seven-joint branching skeleton, with transformed ancestors, nonidentity bind rotations, mixed skin weights, PBR material, UVs, and an embedded 1×1 PNG. No third-party characters or textures are included.

Checks cover rest-only input, facing choices, joint ordering/identity, source-animation ignoring, transformed ancestors/intermediaries, rest skinning, original binary preservation, root displacement, timestamps, and quaternion continuity. Blender and an independent glTF skinning evaluator agree at every sample within `2e-5` world units. Canonical/FK reference tolerances range from `2e-7` to `3e-6` by stage.

Reference tests execute pinned rest-extraction/preprocessing/recovery functions independently. Recovery comparisons use local-only Motion checkout `ac236251f90e5ca37c444c53ad383fc85de6d833`; it is not redistributed. The full upstream preprocessing CLI was not run. Windows child cancellation/reaping passed. The actual POSIX parent/grandchild lifecycle test passed in Linux CI.

Official inference tests verify original EMA tensors/T5 embeddings, global RNG isolation, cancellation, and ComfyUI unload/reload including unregistered rotary GPU cache cleanup. Adapter and pinned original sampling on the same upstream model instance are bit-identical. Independently allocated original/managed instances have normalized full-solver maximum difference `1.44004822e-4` overall, `3.88622284e-5` on the actual five joints; comparison tolerance is `2e-4`. Allocation/kernel effects are plausible, but the precise cause is unproven. Cross-device bitwise reproducibility is unclaimed.

## Real workflows and transport

`tools/verify_workflow.py` starts a separate isolated ComfyUI API process with actual runner `CloudPartitionInput`/`CloudPartitionOutput` classes. One graph runs all five nodes and captures four custom values. Another restores the archives and runs preparation, generation, and export on restored inputs. Boundary member inventories retain exact SHA-256 hashes.

Both five-joint and seven-joint fixtures passed with the same installed EMA bundle, offline T5, ComfyUI GPU management, and external Blender. Each run retrieves three animated GLBs and three provenance JSON files through `/view`. The chain initially used the helper directly; branching passed the opt-in pytest wrapper. The current licensed archive repeat also passed. Its local/restored/regenerated GLBs share SHA-256 `e536ddf6a45f4c6205e9d6475d6815ed5cfbde3bc4f3025ced8740589648dd2d`. Disposable servers stopped; the harness uses an in-memory database to isolate the shared ComfyUI database.

Separate localhost tests use a real bearer-authenticated FastAPI/uvicorn server, multipart upload, artifact HEAD, digest-checked download, LocalStorage resolution, and actual Worker staging. Inputs land under ComfyUI/input; bundles under models/unimate; GLB/provenance descriptors restore locally. Their small bundle fixture checks transport, not inference. These transport and real inference bridge tests cover different local execution stages.

Cloud evidence references coordinator baseline `46de766fcbd0f557adefeac6f740973b8dd0aa3b` plus the integration commit recorded in deploy/README.md. No published corrected image is identified. The Linux [runner recipe](deploy/README.md) is prepared, unbuilt, and untested. [Linux CI](https://github.com/splatterfacegames/ComfyUI-UniMate/actions/runs/36657862434) passed on node-pack revision `718a12a`, using checksum-verified Blender 5.1.1 and the pinned ComfyUI fork. This verifies CPU node plumbing, Blender, and process cleanup; it does not run model inference or the worker container.

## Repeat checks

The integration changes are now on the sibling repositories' default branches at the revisions in deploy/README.md. After applying them to those branches, the focused coordinator suite passed 77 tests and the extension suite passed 103 tests. The earlier full-suite counts above describe the original integration baseline.

Use ComfyUI's Python environment and installed test dependencies:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m unittest discover -s tests -p test_nodes.py -v
```

An unrelated globally installed xonsh plugin failed automatic pytest plugin loading; recorded runs disabled it. Blender/reference tests require `UNIMATE_BLENDER`, `UNIMATE_REFERENCE`, and `UNIMATE_MOTION_REFERENCE`. Model tests select local artifacts explicitly; no test downloads them.

For a real workflow, select local paths and a fresh disposable work directory:

```sh
python tools/verify_workflow.py --comfy-root /path/to/ComfyUI --bundle /path/to/model.unimate --blender /path/to/blender --workdir /path/to/fresh-run --cloud-root /path/to/cloud-offload
```

`tests/test_workflow_integration.py` uses `UNIMATE_INTEGRATION_BUNDLE`, `COMFYUI_ROOT`, `CLOUD_OFFLOAD_ROOT`, and `UNIMATE_BLENDER`. Exact model/reference environment variables are in their integration test files. Test elapsed times include setup/reference work and are not generation benchmarks.

## Open gates

- Linux GPU inference, worker-container execution, and live provider provisioning.
- Redistributable real characters, arbitrary-rig motion quality, and the complete upstream preprocessing CLI.
- Independent graphical glTF viewer playback/appearance review; current independent verification is numeric.
- Complete CPU generation, other GPUs/operating systems, and lower precision.
- Latency/peak VRAM measurements and a real deployed-image workflow.

These limits prevent a claim that all [release gates](DESIGN.md#verification-and-release-gates) passed.
