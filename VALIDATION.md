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
runner-identity Git wait timeout that escaped through `/system_stats`; at that
point it was open pending a fix and a successful run.


Windows worker verification subsequently passed at pack `9f0437c` with ComfyUI
`e2f44d7fe65e270ac111237366b03e396b94dcea`, cloud runner `43bd1a0` and client
`4a7a937`. Two actual partition jobs retrieved 369 files (360 PNGs, two GLBs,
two FBXs, four provenance JSONs and one corrected NPZ). Independent playback
checked both GLBs across 120 frames: restored skinning matched exactly, source
appearance was preserved, maximum joint error was 1.758e-7 and maximum interior
anchor error was 6.624e-7. Runtime: Python 3.11.9, PyTorch 2.11.0+cu128,
RTX 5060 Ti, Blender 5.1.1; the numeric correction uses NumPy on CPU. Evidence:
`.runtime/foot-lock-worker-check-2/report.json` and `independent-playback.json`.

The ComfyUI prerequisite fix catches the bounded Git wait timeout and closes
the Windows process job, output stream and reader during cleanup. Core runner
identity and endpoint tests passed (23 tests); changed-file Ruff passed. The
core worktree retains an unrelated untracked launcher, so no clean-worktree
identity claim is made. Provider dispatch and deployed containers remain open.

## Dataset statistics numeric component — 2026-10-03

`unimate_pack/dataset_stats.py` implements all combinations of per-dataset/global
pooling, frame-weighted/object-balanced moments and channel-group standard-
deviation tying. Root/local counts, 1e-8 floors, global object-name grouping and
per-clip input reduction precision match the inspected release. Input arrays
remain unchanged; ordinary finite floating arrays are required. Overflow,
malformed records/options, estimated workspace limits and cancellation fail
explicitly. The allocation estimate includes balanced group and output arrays;
it is not a measured peak.

Verification: 41 tests passed, no skips; changed-file Ruff passed. The 24
reference cases cover eight modes across float16, float32 and float64. Original
statistics method bodies are executed independently from checkout
`2c5b384715aa63d8639b1ed7eb74bfe614570c7a`, with both revision and source file
digest asserted. Review found a masked-NaN validation bypass; the regression
failed before the fix and passed after rejecting array subclasses.

Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
`UNIMATE_DATASET_REFERENCE=.runtime/upstream-audit`, then
`.runtime/testenv/Scripts/python.exe -m pytest tests/test_dataset_stats.py -q`.
Runtime: Windows, Python 3.11.9, NumPy 2.4.3, CPU. This validates the numeric
component only. At that point public dataset/statistics nodes, persisted/portable
contracts, training consumption and local/stadia/Cloud Offload execution remained open.

## Numeric dataset shards and archives — 2026-10-03

`dataset_contracts.py` and `dataset_io.py` implement bounded content-addressed
numeric shards and deterministic ZIP/JSON archives. Mesh assets are omitted;
conditioning has its own identity and an optional source-rig provenance digest.
Training topologies support 2–4096 joints independently of the mesh adapter.
Feature precision, byte order, origin, captions and train/evaluation membership
survive persistence. Statistics extraction excludes evaluation clips and shares
read-only arrays for duplicate payloads.

The combined dataset contract/archive/statistics suite passed 85 tests, no skips;
changed-file Ruff passed. Cases include 2/151/4096-joint inputs, rejection at
1/4097 joints, original float16/32/64 and non-native-byte-order payloads, a
float64-only distinguishable value, malformed/digest-mismatched inputs, duplicate
and traversal ZIP members, nested object arrays, cancellation and mutation.
Sentinel checks prove outer aggregate payload rejection precedes array reads and
inner aggregate expanded-member rejection precedes numeric decoding. Initial
cancellation regressions exposed exception relabeling during load; callbacks now
propagate unchanged. Independent review found no blocking defects.

Command: the prior statistics reference environment, then
`.runtime/testenv/Scripts/python.exe -m pytest tests/test_dataset_contracts.py tests/test_dataset_io.py tests/test_dataset_stats.py -q`.
Runtime: Windows, Python 3.11.9, NumPy 2.4.3, CPU. No ComfyUI node execution is
claimed by this check. Public dataset/statistics nodes, shard collections for
larger training data, Cloud Offload registration/staging, training consumption
and local/stadia integration remain open.


## Public dataset/statistics nodes (2026-10-03)

Six V3 nodes collect paired rig/motion lists, compute training-only statistics,
and load/save numeric dataset and statistics archives. Shared conditioning keeps
per-clip source rig identity. Archive saves check cancellation before publication
and clean temporary files. Statistics ZIP preflight rejects expanded payloads
above 8 MiB (arrays) or 16 MiB (persisted archive) before numeric decoding;
compressed over-budget fixtures demonstrated RED then GREEN.

Focused foundations/adapters/transport/installed V3 suite: 141 passed, 6 subtests,
no skips, in 7.92s. Changed-file Ruff passed. Command:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UNIMATE_DATASET_REFERENCE=<pinned upstream-audit> python -m pytest tests/test_dataset_adapters.py tests/test_dataset_contracts.py tests/test_dataset_io.py tests/test_dataset_stats.py tests/test_dataset_transport.py tests/test_nodes.py -q`.
Windows Python 3.11.9, NumPy 2.4.3; upstream statistics revision
`2c5b384715aa63d8639b1ed7eb74bfe614570c7a`. Plugin autoload is disabled because
an unrelated installed xonsh plugin requires missing prompt_toolkit.

Independent node review found no blockers and ran 53 adapter/transport/V3 tests.
Its review preceded the final expansion-budget guard; those regression checks
are included in the 141-test run. Eight-mode pooling/balancing correctness relies
on the retained pinned-upstream numeric reference suite; the public adapter's
single-dataset fixture alone cannot distinguish those modes.

Actual sibling client/runner codec tests preserve both portable types. This does
not prove partition-handler staging or saved archive retrieval. Local/headless
stadia dataset workflows and training runtime consumption remain open.


## Dataset headless worker workflows (2026-10-03)

`tools/dataset_workflow.py` passed on Windows and headless stadia-testbed. Each
run executed one direct ComfyUI graph and three real partition-handler jobs:
collection/statistics capture, portable value restoration, and saved archive
reload. All six public nodes were registered and executed. Each worker job
returned two archives; all six files restored through the actual client, and
restored bytes and loaded values matched their source exactly. Two archives
were resolved/uploaded/staged through real runner helpers before reload.

Fixture: one previously verified synthetic legged rig, three matching motions,
two training datasets and one evaluation clip with deliberately divergent
features. All eight statistics modes matched direct results exactly, including
original array payload bytes. Evaluation membership was excluded. The fixture
has one training object per dataset, so balanced-reduction correctness still
relies on the pinned-upstream numeric reference suite.

Pack revision: `1acbe982ad2a7e3cd02ec8de9ef29c5bd5b4f6d2` plus the subsequently
committed workflow harness. Runner `43bd1a0d998ffcba2568de2289cd3131f271aaa6`;
client `4a7a9376d8e20cc3decfba26cd1e626bd198ab7a`. Windows ComfyUI
`e2f44d7fe65e270ac111237366b03e396b94dcea`, Python 3.11.9, NumPy 2.4.3.
Stadia ComfyUI `84ba85773925f071c516f0208184773802b4d44a`, Python 3.11.15,
NumPy 2.4.6. Both servers ran CPU mode; no models or Blender processes were
needed. Reports retain system identities, source bundle hashes, graph/job
identities, boundary artifacts and staged asset digests.

Windows report: `.runtime/dataset-worker-check-1/report.json`. Stadia source:
`/home/jethac/workspaces/comfy-unimate-e2e-20261001/run-dataset-worker-20261003-1/report.json`;
local copy `.runtime/stadia-dataset-worker/report.json`. Command:
`python tools/dataset_workflow.py --comfy-root <ComfyUI> --cloud-root <cloud-offload> --client-root <ComfyUI-Cloud-Offload> --python <ComfyUI-python> --rig <captured-rig.part> --motion <captured-motion.part> --workdir <empty-directory>`.

Independent review found no blockers and ran 27 workflow/adapter tests. Asset
declarations are supplied directly to runner helpers; coordinator/client
declaration discovery is not established. Worker cancellation was not injected.
Provider scheduling and deployed-container execution remain unverified.


## Released splits and epoch sampling (2026-10-03)

Split Dataset and Plan Sampling now expose released train/evaluation splitting
and one/two-level replacement sampling. Pinned AST methods/classes retain
released computation bodies from revision
`2c5b384715aa63d8639b1ed7eb74bfe614570c7a`, dataset.py SHA-256
`413a539e4ad4606e599c502758f9d1a199e5750feb44eebccb28f9845ca6961e`.
Torch 2.11 removed the deprecated no-op Sampler(data_source) constructor; the
reference harness supplies that base constructor only. Weight and epoch bodies
are unchanged. Membership/order matches 36 ratio/seed/override combinations;
60 sampler comparisons cover five object-balancing exponents, four dataset
scopes/exponents and three epochs, including the unsigned seed maximum.
Multi-dataset shared object names remain separate sampling groups. Global RNG
state is preserved. Malformed values, stale identity, inconsistent source/epoch
arrays, invalid options and cancellation fail. Independent review caught missing
reports for explicit objects in absent datasets; the regression failed before
the fix and passed after it. Review has no remaining findings.

Combined reference/foundation/transport/actual V3 suite: 265 passed, six subtests,
no skips, 9.38s; changed-file Ruff passed. Command: prior pinned source environment,
`python -m pytest tests/test_dataset_selection.py tests/test_dataset_workflow.py tests/test_dataset_adapters.py tests/test_dataset_contracts.py tests/test_dataset_io.py tests/test_dataset_stats.py tests/test_dataset_transport.py tests/test_nodes.py -q --tb=short`.

Expanded Windows and headless stadia workflows each passed direct execution,
three real worker jobs and six restored archives. Five paired clips include
multiple objects, unequal group sizes, shared object names across datasets and
a divergent explicit evaluation object. All eight statistics modes and four
sampling configurations are checked against direct results; four portable plans
cross capture/restore and are rebuilt after dataset archive reload. Source
validation recomputes ordered training IDs, weights and epoch indices.
Reports: `.runtime/dataset-selection-worker-check-1/report.json`,
`.runtime/stadia-dataset-selection-worker/report.json`; remote source
`/home/jethac/workspaces/comfy-unimate-e2e-20261001/run-dataset-selection-worker-20261003-1/report.json`.
The runs used pack `f3ff61e` plus the subsequently committed selection changes.
The final report-only unmatched-dataset fix does not affect this fixture's
membership or sampling. ComfyUI/runner/client and device identities match the
preceding dataset workflow runs; Windows Python 3.11.9/NumPy 2.4.3 and stadia
Python 3.11.15/NumPy 2.4.6, CPU. No model or Blender process is required.

These are epoch plans and data workflows. Augmentation, collation, text-cache
consumption and training execution remain open. Explicit declarations use real
staging helpers; coordinator discovery, injected worker cancellation and
provider/container deployment remain separate unverified gates.

## Training augmentation numeric foundation (2026-10-03)

Pinned UniMate `2c5b384715aa63d8639b1ed7eb74bfe614570c7a` augmentation,
motion, rotation, topology and dataset wrapper bodies are compared unchanged.
The reference fixture verifies all five source-file SHA-256 identities and three
reference-only Motion file identities recorded in test_training_augmentation.py;
it checks the actual imported paths. Motion is not distributed or imported by
the production adapter. MIT adaptation lineage is recorded in SOURCES.json.

Comparisons cover ellipsoid/linear addition, removal, pooling, perturbation,
no-op and randomized wrapper, two topologies, multiple seeds and three dtypes.
Motion channels are bit-exact for float16/float32; the measured float64 maximum
motion difference was 1.3322676295501878e-15. Tests cover quaternion diagonal ties,
source-fixed wrapper parameters, parameter boundaries, immutable inputs, global
RNG isolation, graph consistency, FK/RIC agreement, pre-copy workspace rejection
and cancellation during rejection sampling.

Independent FK checks demonstrate released addition duplicates parent local
rotation and can alter original poses. Default released policy reproduces it;
explicit neutral_fk corrects insertion, with original-joint FK preservation
checked across both addition methods, both topologies and three seeds. Velocity
channels retain released behavior. Pooling is not claimed pose-preserving.
Review found inaccurate path-limit reporting for removal/pooling overrides.
Both regressions failed before the effective-limit report fix and passed after.

Combined suite: 431 passed, six subtests, no skips, 15.46s; two Torch JIT
DeprecationWarnings. Changed-file Ruff and git diff --check passed.
Windows Python 3.11.9, NumPy 2.4.3, Torch 2.11.0+cu128, SciPy 1.17.1.
Command, with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and explicit pinned
UNIMATE_DATASET_REFERENCE/UNIMATE_MOTION_REFERENCE paths:
`python -m pytest tests/test_training_augmentation.py tests/test_dataset_selection.py tests/test_dataset_workflow.py tests/test_dataset_adapters.py tests/test_dataset_contracts.py tests/test_dataset_io.py tests/test_dataset_stats.py tests/test_dataset_transport.py tests/test_nodes.py -q --tb=short`.

This is numeric foundation evidence. Public encoded-sample generation, text
caches, crop/normalization/collation, augmentation nodes, training execution and
their Linux/headless/cloud workflows remain open. The configured workspace
estimate is not measured process peak memory; bulk NumPy topology/eigen calls
are not internally interruptible.

## Training sample transform foundation (2026-10-03)

training_transforms.py matches unchanged pinned dataset/transforms.py (revision
2c5b384715aa63d8639b1ed7eb74bfe614570c7a, SHA-256
0c16cd112fb5df83778d245df06dbeaa040546fda56611421286340ae0fe231a).
38 tests compare both crop/condition modes, random and explicit starts, short
tails, float16/32/64 normalization, padding dtype and parent-copy features.
Single-frame first_frame extraction yields empty motion with valid length zero;
its regression failed before the validation correction and passes after it.
Invalid options/statistics/parents, pre-allocation padding budget and cancellation
are checked. RNG state remains unchanged. Source padding promotes to float64.

Combined dataset/augmentation/transforms/transport/V3 suite: 469 passed, six
subtests, no skips, 17.39s; two existing Torch JIT deprecation warnings. Command
is the preceding augmentation suite plus tests/test_training_transforms.py in
the same pinned environment. Changed-file Ruff and git diff --check passed.
Source MIT lineage is retained. This does not prove facing realignment, encoded
sample assembly, portable sample/cache values, collation or public sample nodes;
those remain open, as do their headless and Cloud Offload workflow checks.

## Cropped training facing realignment (2026-10-03)

realign_clip matches the unchanged pinned realign_unimate_clip body using the
reference-only Motion quaternion implementation. Six comparisons cover two
nontrivial topologies and float16/32/64, with nonidentity facing after cropping.
All output motion channels are bit-exact; source inputs, position and velocity
channels remain unchanged. Production uses independently expressed signed
quaternion conversion/composition equations, not Motion imports or vendored code.
Cancellation and a pre-copy workspace estimate guard execution. This completes
numeric realignment only; encoded sample assembly and public workflows remain
open. Combined suite and source lineage are recorded with this change.
Combined command: preceding 469-test suite with six added realignment comparisons; 475 passed, six subtests, no skips, 15.96s. Two existing Torch JIT warnings; changed-file Ruff passed.

## Encoded numeric sample assembly (2026-10-03)

training_samples.assemble_sample adapts the post-augmentation portion of pinned
MotionDataset.__getitem__. It accepts actual embedding views as explicit arrays;
tests use controlled numeric views to isolate assembly, not to claim encoder
execution. The fixture compiles the unchanged __getitem__ body, unchanged shared
transforms and original reference-only Motion code. Augmentation is supplied by
the separately source-compared operation fixture. Dataset/stats lookup is outside
this layer; no end-to-end training dataset claim follows from this harness.

60 exact field/dtype comparisons span tpos/first_frame, float16/32/64, no-op,
addition/removal/pooling/perturbation and cropped/padded lengths. Checks cover
rest feature expansion, facing realignment, normalization, valid length, copied
parent features and original topology/statistics/embedding fields. Outputs own
their arrays and inputs/global RNG remain unchanged. Seven invalid input cases
cover empty captions, capacity/options, embedding dimensions/nonfinite tokens
and workspace budget; cancellation propagates. The absent adapter failed before
implementation. Numeric caption views are preserved independently without pooling.

Combined suite: 543 passed, six subtests, no skips, 17.81s; two existing Torch JIT
warnings. Command: previous 475-test command plus tests/test_training_samples.py,
in the same explicitly pinned reference environment. Changed-file Ruff passed.
Encoder/cache production, persisted identity contracts, batch collation, public
sample nodes and headless/Cloud Offload execution remain required unverified work.
Review found dense topology validation preceded the workspace budget. A regression
forbidding entry into validation under a one-byte budget failed before the fix.
Shallow field/array/rank/capacity checks and the conservative input/dense/output
estimate now reject before topology validation. This estimate is not measured
process peak memory.
Final regression run: 544 passed, six subtests, no skips, 18.20s; two existing warnings. Changed-file Ruff and git diff --check passed.
Independent review rechecked the budget correction and has no remaining actionable findings.

## Numeric training batch collation (2026-10-03)

The released MIT mixture collator is retained unchanged at its verified source
SHA-256 7cb7672ba97572282ec7a3153abd21aeb4917ecf7f77bf7d3079e13a9f888eb2,
revision 2c5b384715aa63d8639b1ed7eb74bfe614570c7a. Source lineage/license remains
in SOURCES.json and LICENSE-UniMate. training_collation wraps its CPU tensor
boundary with shape/dtype/capacity/embedding validation, plain finite arrays,
float32 representability, positive std, estimated workspace preflight and owned
copies. Float32 default Torch dtype is required explicitly. This wrapper does
not introduce persisted identities or portable tensor serialization.

15 complete nested output comparisons cover float16/32/64 inputs and all,
mixed-caption, caption-free, partial-spectral and required-only layouts. Samples
have unequal joint counts, different spectral widths and token lengths, padded
time and zero valid length. Every source tensor, array, label, mask and optional
key matches. Padded std equals one; caption-less rows retain one valid zero token.
Returned parents/edges cannot mutate caller arrays. Five malformed cases plus
pre-source budget/cancellation and empty filtered batch checks bring the focused
suite to 22 tests. The missing adapter failed before implementation.

Cancellation is checked during validation and before/after the unchanged source
collator; its tensor allocation/copy loops are not internally interruptible.
Workspace is a conservative numeric estimate, not measured process peak memory.
Actual encoder/cache production, portable sample/batch contracts, public nodes
and headless/Cloud Offload sample workflows remain open.
Review found two source-conversion hazards. Positive float64 std=1e-100 became
zero in float32, and uint64 relation 2**64-1 became a negative int64 embedding
index. Both regressions failed before correction. Validation now checks std
positivity in float32 and bounds unsigned integer arrays plus converted scalar
fields to int64 before source invocation.
Final combined suite: 568 passed, six subtests, no skips, 16.80s; two existing warnings. Command: preceding 544-test suite plus tests/test_training_collation.py, same pinned environment. Changed-file Ruff and git diff --check passed.
Independent review rechecked both conversion fixes; no remaining actionable findings.

## Portable training text views and installed encoder (2026-10-03)

training_text builds unimate.text_cache.v1 numeric dictionary/bytes values:
sorted unique strings, explicit encoder type/version/artifact SHA-256, pinned
source revision, float32 concatenated ragged tokens, int64 lengths and separate
float32 encoder-pooled vectors. Missing strings are encoded in chunks <=256;
complete matching caches need no encoder call. Arrays and ZIP expansion are
bounded to 64 MiB, entries to 4096 and individual token sequences to 512 rows.
No pickle, local paths, model objects or device tensors are persisted. Cached
views retain the encoder vector; fresh views use source-order trimmed-row mean.
The estimate/bounds cover cache arrays, not encoder VRAM or total process peak.
Nine foundation tests exercise duplicates, chunking, identity/schema/digest and
missing-text errors, cache reuse, malformed output, budgets and cancellation.
Review found finite tokens whose float32 mean overflows; its regression failed
before the fix. Fresh pooling keeps source arithmetic and rejects nonfinite output.

Actual offline test command: with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1,
UNIMATE_TEST_BUNDLE=.runtime/models/unimate-v2-licensed.unimate,
UNIMATE_TEST_COMFY=B:/lab/ComfyUI and pinned UNIMATE_DATASET_REFERENCE,
`python -m pytest tests/test_training_text_model.py -q -s --tb=short`.
One passed, 40.30s, two existing Torch JIT warnings. Network connects were
forbidden. Four texts include anatomical names, a caption and empty text.
Every token/cached pooled/fresh pooled view matches unchanged pinned
sequences_from_hidden, pooled_from_hidden and pool exactly; complete-cache
rebuild also passes without encoder access. Source SHA-256:
f1433ec15449a77721d921ebc16ae76ef8a552eed79b6413839370e64725dd01.
Device cuda:0, NumPy 2.4.3, Torch 2.11.0+cu128; encoder artifact identity
f79264a13a940769912d95fa9521c1f6c4448e1fe129d1ad4e1cf6a7c4a4259d
hashes the pinned text-encoder manifest/file inventory. Model bundle SHA-256
3d4420752e64b873f98c8aec2d6f01edf7be861c704920dfc095bd1d700664b8;
cache arrays SHA-256 136c3b0cf764e1819a6dd8ff4091db709afee34404e373cc5d5592a34071edb9.
Models are unloaded and private runtime extraction is cleaned in finally.

This proves installed FLAN-T5 numeric cache production on Windows. It does not
prove other released encoders, external cache-file conversion, public nodes,
portable sample/batch identities or Linux/headless/Cloud Offload execution.
Combined foundation suite: 577 passed, six subtests, no skips, 17.85s; two existing warnings. Command: preceding 568-test suite plus tests/test_training_text.py. Changed-file Ruff and git diff --check passed. Independent review rechecked pooling overflow; no remaining actionable findings.

## Portable encoded training sample contract (2026-10-03)

unimate.training_sample.v1 carries fifteen numeric array fields and six source
sample metadata fields, preserving input precision and owning decoded arrays.
Provenance records dataset digest, portable clip ID, statistics digest and text
cache digest. Options retain crop mode/seed/start, realignment and augmentation
report. An identity digest covers schema/source, provenance, options, metadata
and the array digest. Optional expected provenance rejects stale source identity.
This verifies declared provenance; dataset/cache-bound producer recomputation is
still required and is not inferred from a caller-provided digest.

Array creation is budgeted before encoding, persisted NPZ members have 64 MiB
compressed/expanded limits with exact field membership, and metadata is bounded
canonical JSON. Validation reuses numeric sample validation without allocating
source collation tensors. All arrays serialize in sorted order. Eleven tests
cover float16/32/64 round trips, ownership, malformed/stale identities, metadata
and options, precopy budget rejection and absence of source tensor allocation.
The missing contract failed before implementation. Existing collation comparisons
remain green after separating validation from allocation.

Combined foundation suite: 588 passed, six subtests, no skips, 17.62s; two existing
Torch JIT warnings. Command: preceding 577-test suite plus
tests/test_training_sample_contracts.py in the same pinned environment. An unused
import caused initial Ruff failure; removed it and changed-file Ruff passed.
Public nodes, dataset/cache-bound sample producer, portable batch encoding and
Windows/stadia/Cloud Offload sample workflows remain required open work.
Review found workspace rejection followed numeric decoding and explicit requested
crop starts could disagree with recorded starts. Both new regressions failed
before correction. Declared compressed/expanded archive size is now bounded by
min(64 MiB, workspace/4) before decode; explicit starts must match metadata in
creation and validation. Numeric validation still checks the conservative batch
workspace estimate without allocating source tensors.
Final suite: 590 passed, six subtests, no skips, 17.40s; two existing warnings. Changed-file Ruff and git diff --check passed.
Independent review rechecked both fixes: thirteen contract tests pass, no remaining actionable findings.

## Dataset-bound encoded sample producer (2026-10-03)

produce_training_sample selects a captioned training clip from validated
portable dataset bytes, requires matching statistics dataset identity and text
coverage, and derives provenance itself. Statistics identity hashes its complete
metadata/options plus numeric digest; text identity hashes encoder/source/text
inventory plus arrays digest. Missing/evaluation clips, stale statistics and
missing/empty captions fail. Aggregate declared numeric archive expansion is
bounded before input decoding; subsequent augmentation/sample limits apply.

Conditioning uses cleaned names unless blanks require the released raw-name
fallback, retained in the report. Grounded or original rest positions determine
identity-rest offsets by parent subtraction; source topology and spectral signs
are preserved unless requested width requires source eigen recomputation.
Root/local normalization rows come from selected dataset statistics. Cached and
fresh pooling policies are explicit. Input feature clips are already prepared:
this producer never re-extracts or re-grounds motion features. Raw feature
extraction, collection preprocessing and their grounding semantics remain open.

Augmentation and cropping use independent explicit seeds. Four exact comparison
cases execute unchanged source model_joint_names, create_condition,
_precompute_object_type_meta, _get_normalization_stats and __getitem__ for both
condition modes and both rest-grounding policies. Reference-only Motion computes
rest offsets. The source no-op wrapper's random.choice([None]) is bypassed in
this comparison solely to compare the independent crop stream; this is not a
claim of identical joint RNG streams for the full source constructor/getitem.
Ten additional production paths span five augmentation operations and both
modes, plus five invalid cases and predecode budget/cancellation. Twenty focused
tests pass; actual installed encoder views were validated in the prior text-cache
check, while these producer fixtures use controlled numeric views.
Public nodes, portable batches, Windows/stadia/Cloud Offload producer workflows,
raw dataset processing and training runtime remain required open work.
Combined suite: 610 passed, six subtests, no skips, 22.56s; two existing warnings. Command: preceding 590-test command plus tests/test_training_dataset_samples.py. Changed-file Ruff and git diff --check passed; independent review has no actionable findings.

## Public text-cache and encoded-sample nodes (2026-10-03)

Build UniMate Text Cache and Prepare UniMate Training Sample register through
the actual V3 extension, bringing the node count to 29. Text production derives
encoder identity from the selected bundle's verified text-file inventory and
revision. It gathers cleaned joint vocabulary with source fallback and captions;
complete matching caches skip runtime creation. Missing strings use the existing
ComfyUI-managed offline encoder. Sample execution exposes crop/condition modes,
separate seeds, all augmentation choices and random flags, released/neutral_fk,
rest/text policies, spectral width and workspace budget. Individual numeric
augmentation parameters still use released defaults through this node.

Four node/codec tests verify schemas, actual sample execution, text delegation
and both directions of the installed client/runner codecs. Portable text/sample
values retain exact bytes and validate after restore. Generic socket support
already accepts concrete custom types; no sibling protocol mutation was needed.
A builder regression verifies complete-cache reuse without runtime creation.
These are codec/unit checks, not worker-handler or coordinator execution.

The installed offline source-math test was extended through actual bundle →
dataset text cache → dataset-bound portable sample. Seven joint embeddings and
caption tokens have dimension 768; complete matching dataset cache reuse needs
no runtime. Existing exact source token/pooled/fresh checks remain. One passed,
53.27s, CUDA:0; model/text source/runtime identities match the preceding installed
text check. Network connections forbidden; models unloaded/private runtime cleaned.
The harness reuses its already loaded actual runtime through _get_runtime to avoid
a second model instantiation; server node runtime instantiation remains a workflow
gate. No server/headless/cloud sample workflow or training run is claimed.

Review found duplicate registration introduced during extension editing.
Uniqueness regression failed (31 entries/29 IDs), then the extra pair was removed.
Combined suite: 615 passed, six subtests, no skips, 46.92s; two existing Torch JIT
warnings. Command: preceding 610-test command plus tests/test_training_nodes.py
and the extended tests/test_training_text.py. Changed-file Ruff and diff check
passed. Public schemas are unique after registration correction.
Independent review rechecked registration: 29 unique node IDs, no remaining actionable findings.

## Training sample server and worker workflows (2026-10-03)

`tools/training_workflow.py` passed on Windows and headless stadia-testbed.
Each run used an actual isolated CPU ComfyUI server, installed FLAN-T5 bundle,
runner partition handlers and client file restoration. Seven augmentation choices
crossed both conditioning modes; ellipsoid/linear addition also used `neutral_fk`:
18 samples per workflow. Direct execution and three worker jobs (build, cached
producer restore, encoded-value reload) preserved dataset/statistics/cache values
and byte-exact expected samples within each runtime. The runner staged the model
by SHA-256. Six returned dataset/statistics archives per platform matched their
original bytes and loaded values.

Reports: `.runtime/training-worker-check-2/report.json` and
`.runtime/stadia-training-worker/report.json`; remote original:
`/home/jethac/workspaces/comfy-unimate-e2e-20261001/run-training-worker-20261003-1/report.json`.
Reports record base pack `695b6f4a10cb324cd8aef3dceea619344590d819`
with this commit's harness changes present in the worktree. Windows ComfyUI:
`e2f44d7fe65e270ac111237366b03e396b94dcea`; stadia ComfyUI:
`84ba85773925f071c516f0208184773802b4d44a`. Both used runner
`43bd1a0d998ffcba2568de2289cd3131f271aaa6` and client
`4a7a9376d8e20cc3decfba26cd1e626bd198ab7a`.
Windows: Python 3.11.9, Torch 2.11.0+cu128 running CPU.
Stadia: Python 3.11.15, Torch 2.14.1+cpu running CPU.
Training reference remains `2c5b384715aa63d8639b1ed7eb74bfe614570c7a`.

Bundle SHA-256:
`3d4420752e64b873f98c8aec2d6f01edf7be861c704920dfc095bd1d700664b8`.
Text encoder artifact SHA-256:
`f79264a13a940769912d95fa9521c1f6c4448e1fe129d1ad4e1cf6a7c4a4259d`.
Cache digests differ between runtimes; no cross-platform byte parity is claimed.
The first Windows attempt failed because the harness omitted six required API
inputs and selected a server environment lacking torch-geometric. Explicit graph
inputs and the installed pack environment corrected those failures. The graph
regression failed before the fix and passed afterward.

Combined dataset/training/node checks: 616 passed, six subtests, no skips, 26.67s,
with two existing Torch JIT warnings. This evidence covers prepared-feature
samples, not portable batches, training consumption, other encoders, coordinator
asset discovery, deployed containers, provider dispatch or injected worker
cancellation. Independent review found no actionable harness or scope findings.

## Portable training batches (2026-10-03)

Collate UniMate Training Samples registers as the thirtieth V3 node and collects
an execution list. `unimate.training_batch.v1` holds numeric motion/conditioning,
packed parents/edges, original parent dtypes, labels and ordered sample references.
Restoration returns owned CPU tensors in the released collator's layout. Source
comparison uses the pinned unchanged collator at
`2c5b384715aa63d8639b1ed7eb74bfe614570c7a`; the reference file digest remains
`7cb7672ba97572282ec7a3153abd21aeb4917ecf7f77bf7d3079e13a9f888eb2`.
Variable joint counts, caption lengths, spectral widths and a zero valid-motion
length match exactly, including tensor dtypes, masks and standard-deviation padding.
Parent dtype and repeated sample order survive restoration; returned tensors own
their storage. Client/runner codecs carry the new value without sibling changes.

Tests reject forged digests, source pins, parent topology, caption-mask dtype,
joint masks, padding values and labels. Aggregate sample/archive limits precede
decoding; cancellation can abort before it. Sixteen contract tests passed;
contract/collator/node focused checks passed. Independent review found no
actionable findings. No batch server/headless workflow or training consumption
is established by these tests.

Full `pytest -q --tb=short` with training and Motion references enabled:
903 passed, 50 skipped, six subtests, 43.61s; two existing Torch JIT warnings.
Skipped checks remain outside this run's evidence. The first full run exposed
an outdated expected inventory and leaked temporary-path inventory cache in the
node test fixture (901 passed, two failed, 50 skipped). Updated inventory and
fixture-owned cache restoration corrected both before the successful run.
