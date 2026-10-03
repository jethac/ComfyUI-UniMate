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

Cloud evidence references coordinator baseline `46de766fcbd0f557adefeac6f740973b8dd0aa3b` plus the integration commit recorded in deploy/README.md. No published corrected image is identified. The Linux [runner recipe](deploy/README.md) is prepared, unbuilt, and untested. [Linux CI](https://github.com/jethac/ComfyUI-UniMate/actions/runs/36657862434) passed on node-pack revision `718a12a`, using checksum-verified Blender 5.1.1 and the pinned ComfyUI fork. This verifies CPU node plumbing, Blender, and process cleanup; it does not run model inference or the worker container.

## Headless stadia-testbed run — 2026-10-01

Passed on `stadia-testbed`: Ubuntu Linux kernel 5.15.0-191-generic, Intel Xeon W-2135, Python 3.11.15, Blender 5.1.1, torch 2.14.1+cpu, transformers 5.18.0, NumPy 2.4.6, SciPy 1.17.1. The process used six CPU threads; its AMD Vega GPU was not used. Sources: this pack `317f466` plus the harness fix below, ComfyUI `84ba8577`, coordinator `ab8b2d8`, extension `220273f`. The model bundle hash matches the licensed archive recorded above.

The seven-joint branching fixture ran all five nodes with prompt `A character walks forward.`, seed 0, guidance 3, and `objaverse` normalization. A second workflow restored and consumed all four actual partition boundary types, including the full model bundle, then regenerated motion. Both workflows succeeded without cached nodes. Six outputs were retrieved: three GLBs and three provenance JSONs. Total workflow time was 1,182.99 seconds, including startup and transfer work; this is a recorded CPU run, not a generation benchmark.

All three GLBs were byte-identical: 13,164 bytes, SHA-256 `00be100482edd5bc7e32fa1dd51ce7605359f6d24420f87475d825d7ebd69d91`. Provenance values were identical after JSON parsing. Source nodes, skins, meshes, materials, textures, images, and original binary data were preserved. Headless Blender playback matched an independent glTF skinning evaluator over all 60 frames at 30 fps, with maximum vertex error `1.708e-6` world units against a `2e-5` limit. The generated geometry moved and remained finite.

A separate authenticated coordinator HTTP test transferred the real 739,698,499-byte model bundle and input GLB, staged them through production worker methods, and retrieved/restored the generated GLB/provenance pair with exact byte comparisons. This covered artifact transfer and staging; it did not exercise provider provisioning, rental preflight, or job scheduling. The remote repository suite passed 170 tests with 20 skips and 6 subtests; focused coordinator and extension suites passed 77 and 103 tests. All task servers and processes stopped.

The first attempt exposed a Unix harness bug: resolving a virtualenv Python symlink launched the system interpreter and lost installed dependencies. The harness now preserves the executable path, accepts `--cpu`, and records the interpreter and ComfyUI system statistics. The fixed run above verified this in a real virtualenv.

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

For a CPU-only host, add `--cpu` to the workflow command or set `UNIMATE_INTEGRATION_CPU=1` for the pytest integration wrapper.

## Open gates

### Expanded stadia inference, 2026-10-03

`tools/verify_workflow.py --branching --cpu --extended` passed in workspace
`/home/jethac/workspaces/comfy-unimate-e2e-20261001/run-expanded-archive-fixed-20261003`.
The server ran ComfyUI `84ba85773925f071c516f0208184773802b4d44a`, Python 3.11.15,
PyTorch 2.14.1+cpu and Blender 5.1.1. Inference was offline, float32, with v2 EMA bundle
SHA-256 `3d4420752e64b873f98c8aec2d6f01edf7be861c704920dfc095bd1d700664b8`,
upstream `5d6aabedd947297b5ba6706d8e9113e68c0c3e4f` and FLAN-T5-base revision
`7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`.

Four GLBs, four provenance JSON files and three numeric archives were retrieved through
ComfyUI. Independent glTF skinning evaluation passed all 290 frames: generation 60,
in-betweening 60, editing 60 and expansion 110. Vertices were finite and changed over time;
source meshes, skins, materials, images and textures were unchanged. Archives decoded to
`(60,7,12)`, `(60,7,12)` and `(110,7,12)`. Local evidence is under
`.runtime/stadia-expanded-archive-fixed`, including `report.json` and `independent-check.json`.
Newer nodes and Cloud Offload were not exercised by this graph.

### FBX base-color textures, 2026-10-03

External Blender tests passed seven- and 110-frame branching clips with a textured,
tinted material. The adapter checks effective base-color baking within 5e-5 linear color,
embedded pixels within 2e-6 after deleting private sidecars, skeleton names and skinning
at every frame. `tests/test_fbx.py tests/test_nodes.py`: 23 passed, 6 subtests passed.
This does not establish fidelity for other PBR material expressions or cloud execution.

### Remaining gates

Skeleton archive/recovery/rendering also passed headlessly on stadia-testbed at pack
`a0d841a`: 60 generated frames per mode, 120 PNGs retrieved, plus GLB/provenance.
Remote report: `/home/jethac/workspaces/comfy-unimate-e2e-20261001/run-skeleton-20261003/report.json`.
Runtime: ComfyUI `84ba85773925f071c516f0208184773802b4d44a`, Python 3.11.15,
PyTorch 2.14.1+cpu, Blender 5.1.1. Cloud runner execution remains open.

- Skeleton recovery/rendering: pinned FK/RIC numerical comparisons and Windows server PNG retrieval passed. Focused tests: 109 passed, 6 subtests passed. Windows evidence: `.runtime/skeleton-server-fixture-check/report.json`, with 17 frames per mode and 34 retrieved PNGs; stadia evidence is recorded above. Actual cloud runner execution remains open.

### Rig identity portability correction, 2026-10-03

Headless stadia at pack `2e17283` also accepted the old Windows archive against the
freshly prepared Linux rig: 17 frames per recovery mode, all 34 PNGs retrieved plus
GLB/provenance. Remote evidence: `run-legacy-windows-20261003/report.json`.

The observed Windows/Linux ID mismatch came from NPZ ZIP creator-OS markers, not numeric
conditioning differences. The asset, mapping and every array's dtype, shape and payload
matched exactly. Canonical marker 3 preserves the existing Linux identity on fresh Windows
preparation. Legacy marker-0/3 motion IDs are accepted only when their exact digests can be
recomputed from the validated rig; different payloads remain different identities.

Independent skinning evaluation compared the Linux export with its Windows re-export
at all 60 frames and 21 vertices: maximum coordinate difference was zero. Evidence:
`.runtime/skeleton-cross-platform-fixed/independent-playback.json`.

The earlier stadia archive passed the Windows ComfyUI workflow: 60 frames per recovery
mode, 120 retrieved PNGs plus GLB/provenance. An old Windows archive also passed against
the new canonical rig: 17 frames per mode, 34 retrieved PNGs plus GLB/provenance.
Evidence: `.runtime/skeleton-cross-platform-fixed/report.json` and
`.runtime/skeleton-legacy-windows-check/report.json`. Focused regression checks passed:
102 tests plus 6 subtests, including public legacy GLB/FBX export acceptance. This proves
the observed container metadata correction; it does not treat different numeric results
from other environments as equivalent rigs.

External Blender GLB/FBX regression checks passed: 13 tests, one reference-dependent
test skipped in that run. Contract/node/reference checks are recorded separately above.

- Linux GPU inference, worker-container execution, and live provider provisioning.
- Redistributable real characters, arbitrary-rig motion quality, and the complete upstream preprocessing CLI.
- Independent graphical glTF viewer playback/appearance review; current independent verification is numeric.
- Other GPUs/operating systems and lower precision. CPU generation passed the recorded synthetic fixture; broader asset quality remains open.
- Latency/peak VRAM measurements and a real deployed-image workflow.

These limits prevent a claim that all [release gates](DESIGN.md#verification-and-release-gates) passed.

### Actual partition handler verification, 2026-10-03

Windows worker-mode verification passed two real ComfyUI partition jobs, using
Cloud Offload's partition handler, artifact storage, declared input staging and
client file restoration. Eight boundaries covered six value types: asset, rig,
motion, conditioning, skeleton and IMAGE. Restored values were consumed by the
second job; canonical preparation and animation extraction retained their checked
asset/rig identities. Every round-trip bundle member matched its original digest.

The 17-frame fixture produced 102 PNGs, two GLBs, two FBXs and four provenance files.
FBX export performed its Blender re-import checks. Report:
`.runtime/preprocessing-worker-fbx-check/report.json`. Runtime: ComfyUI
`84ba85773925f071c516f0208184773802b4d44a`, Blender 5.1.1, Windows Python 3.11.
Focused checks: six harness tests, 36 worker tests and 48 client tests passed.

This exercises the actual handler and executor against a local ComfyUI server.
Provider scheduling, deployed worker containers and stadia execution of this graph
remain separate gates. The fixture does not perform model inference.

### Headless partition handler verification, 2026-10-03

Stadia-testbed passed the same two-job worker graph at pack `7790f88`, worker
`3f7d613` and client `176e24c`. The input was the 60-frame archive from the earlier
released v2 in-betweening run. All six boundary types restored and were consumed;
canonical preparation and 59-frame animation extraction passed their identity
checks. Retrieved outputs: 360 PNGs, two GLBs, two FBXs and four provenance files.
Remote evidence: `run-preprocessing-worker-20261003/report.json` under the recorded
stadia workspace. Runtime: Python 3.11.15, PyTorch 2.14.1+cpu, Blender 5.1.1 and
ComfyUI `84ba85773925f071c516f0208184773802b4d44a`.

Independent downloaded-output checks evaluated all 60 frames and 21 vertices in
both GLBs: finite, changing motion with zero difference after restoration.
Mesh, skin, material, texture and image structures matched. The original,
restored IMAGE and re-rendered skeleton PNGs matched pixel for pixel for both
FK and RIC (360 frames checked). Local evidence:
`.runtime/stadia-preprocessing-worker/independent-check.json`.
This run performs no new inference and does not test provider scheduling or a
deployed worker container.

### Partition execution-list correction, 2026-10-03

The previous scalar capture schema let ComfyUI map a batch into repeated writes
at one boundary path. A two-case reproduction retained only the last case:
`.runtime/batch-bridge-loss.json`. Bridges now capture the whole execution list
once and restore list outputs using `comfy.partition.execution.v1` envelopes.
Legacy unwrapped values restore as one scalar, including Python list-valued data.
Client and runner updates must deploy together; old readers cannot read the new
envelope. The bounded, pickle-free bundle format remains unchanged.

Client/protocol checks: 70 passed. Worker checks: 36 passed. Harness checks: seven
passed; changed harness lint passed. The Windows scalar worker regression passed
both jobs and retrieved 110 files. A third job with two distinct, ordered motions
also passed, preserving motion payloads and recovering different FK/RIC skeletons.
All 170 PNGs and eight GLB/FBX/provenance files were retrieved across three jobs.
Evidence: `.runtime/preprocessing-mapped-list-check/report.json`. This verifies
same-rig mapped values, not multiple rigs, new model inference or deployment.

Headless stadia passed the three-job mapped-list workflow at pack `3ce0237`,
worker `43bd1a0` and client `4a7a937`: 600 PNGs, two GLBs, two FBXs and four
provenance files retrieved. Runtime matches the preceding stadia record. Remote
report: `run-mapped-execution-lists-20261003/report.json`. Independent checks
compared both recovered cases at all 60 frames: first-case positions and pixels
matched the original capture; the second case differed by up to 0.5846864 position
units and had different preview pixels at every frame, for both FK and RIC.
Remote evidence: `run-mapped-execution-lists-20261003/independent-list-check.json`.
The second case is a controlled velocity change to the earlier model archive;
this run does not generate a new batch or establish multiple-rig support.

### Multi-rig model batching, 2026-10-03

Combine Rigs and paired Batch outputs passed the Windows ComfyUI model workflow
on original five-joint chain and seven-joint branching fixtures: two prompts,
two repetitions per rig, seeds 0–7, eight 60-frame GLBs and eight provenance files.
The verifier checked every seed/prompt/source assignment and two distinct,
stable rig IDs. Independent skinning evaluated all 480 frames: finite, changing
vertices; original mesh, skin, material, image and texture structures and binary
prefixes preserved. Evidence: `.runtime/multi-rig-model-check/report.json` and
`independent-playback.json`.

The two-job batch worker workflow also passed. It staged both declared GLBs and
the selected model bundle from digest-keyed storage, then generated and published
eight paired rig/motion cases. A second handler job restored both lists and
exported all eight cases. Complete artifact inventories matched across transport;
client-restored files matched their published bytes. Independent skinning checked
all 16 GLBs / 960 frames, with zero difference between each original and restored
case. Evidence: `.runtime/multi-rig-worker-model-check/report.json` and
`independent-playback.json`.

Runtime: Windows, Python 3.11, PyTorch 2.11.0+cu128, RTX 5060 Ti, Blender 5.1.1,
ComfyUI `84ba85773925f071c516f0208184773802b4d44a`. Model bundle SHA256:
`3d4420752e64b873f98c8aec2d6f01edf7be861c704920dfc095bd1d700664b8`.
Source/model/T5 pins match the earlier v2 records. Focused checks: 39 tests plus
six subtests passed; changed code lint passed. Review found no blocker.

Staging is explicitly invoked before the actual partition handler. Provider
dispatch, gateway submission/extraction and deployed containers are not exercised
by this harness.

The same two-job workflow passed headlessly on stadia-testbed using pack
`0a2d77b`, cloud runner `43bd1a0` and client `4a7a937`. Both declared rigs and
the model bundle were digest-staged; eight paired cases used seeds 0–7. The
independent playback checker evaluated all 16 GLBs / 960 frames and found exact
original/restored skinning equality, with preserved appearance and finite,
changing vertices. Runtime: Linux, Python 3.11.15, PyTorch 2.14.1+cpu, Blender
5.1.1, CPU device, float32. ComfyUI and model bundle identities match the Windows
record above. Evidence: `.runtime/stadia-multi-rig-worker/report.json` and
`independent-playback.json`. Provider dispatch, gateway submission/extraction and
deployed containers remain outside this harness.

### Foot-lock numeric and Blender checks, 2026-10-03

Foot Lock Motion implements Appendix E.5 contact selection/filtering, anchoring,
bounded damped IK and boundary quaternion blending independently. Focused checks:
52 tests plus six subtests passed; the Blender-dependent test was skipped in that
run and passed separately with Blender 5.1.1 (60-frame legged clip, 92.18 seconds).
Independent glTF skinning and Blender evaluated vertices agree at every frame
within 2e-5 using bidirectional nearest-vertex checks. Original materials, textures,
skins and source binary are preserved in numeric export checks.

Portable correction tests cover moving root/facing, nonzero origin, exact root
channels, FK/RIC agreement, velocity re-encoding, cancellation during correction,
invalid/overlapping contact overrides and positive rest-height requirements. A
512 MiB estimated workspace budget rejects oversized work before FK allocation.
Review found and closed overlapping manual-chain overwrite behavior.

Windows server verification timed out during core ComfyUI startup before the node
loaded. A startup trace locates the delay in Transformers 5.8.0 package metadata
scanning (`importlib.metadata.packages_distributions`). Server, headless stadia and
actual partition-handler execution were unverified at that point. A later run
reached the node and correctly rejected a fixture label mismatch in rig identity.
Reference preparation now uses the verifier's exact asset filename and Blender
joint ordering; the identity check was retained.

Windows server verification subsequently passed at pack `d3333c0`, retrieving
123 files: corrected NPZ, GLB/provenance and 120 FK/RIC PNGs. The saved archive
authenticates its source features, contains active contact segments and changed
motion features, and preserves root channels/origin. Independent channel and
skinning checks evaluated all 60 frames; maximum canonical joint error was
1.758e-7 and maximum interior anchor error 6.624e-7. Appearance/binary preservation
checks pass. Evidence: `.runtime/foot-lock-server-check-3/report.json` and
`independent-playback.json`.

Headless stadia's actual two-job partition-handler run also passed at `d3333c0`:
declared GLB/archive staging, portable corrected motion, FK/RIC skeletons/images,
GLB/FBX exports, source extraction, storage and client restoration. It retrieved
369 files (two GLBs, two FBXs, four provenance JSONs, one corrected NPZ, 360 PNGs).
Independent checks evaluated both GLBs / 120 frames and found exact
original/restored skinning equality and the same joint/anchor bounds as Windows.
Runtime: Linux, Python 3.11.15, PyTorch 2.14.1+cpu, Blender 5.1.1, CPU device;
ComfyUI `84ba85773925f071c516f0208184773802b4d44a`, cloud runner `43bd1a0`, client
`4a7a937`. Evidence: `.runtime/stadia-foot-lock-worker/report.json` and
`independent-playback.json`. Provider dispatch and deployed containers are outside
this harness. Windows worker verification stopped before submission at a core
runner-identity Git wait timeout that escaped through `/system_stats`; it remains
open pending that fix and a successful run.
