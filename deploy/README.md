# Cloud Offload runner

The runner needs ComfyUI, this pack, its dependencies, external Blender, and the sibling Cloud Offload changes below. Default runner images without these changes are insufficient. Models are staged artifacts; the image contains no weights.

Required integration:

- ComfyUI-Cloud-Offload: trusted `cloud_offload_assets` declarations, `.unimate` discovery, unresolved-file upload, and `3d`/`files` restoration.
- cloud-offload: artifact HEAD/upload resolution, `__input__` staging under ComfyUI/input, provenance retrieval, and worker file-output return.
- Runner bridge: `CloudPartitionInput` / `CloudPartitionOutput` using `comfy.partition.bundle.v1`, with `COMFY_PARTITION_ROOT` set to a managed job directory.

Integration is on both default branches: [ComfyUI-Cloud-Offload `220273f`](https://github.com/jethac/ComfyUI-Cloud-Offload/commit/220273f4f7e2e1fabd9d32743376ff3c55626288) and [cloud-offload `ab8b2d8`](https://github.com/jethac/cloud-offload/commit/ab8b2d8db615e5ae4a9767eba2720b20b8cd2e51). These are source revisions; no corrected runner image has been published.

The bridge passed separate-process Windows ComfyUI execution with real inference and Blender. Authenticated localhost HTTP staging/retrieval also passed. Linux image execution and live provider provisioning are unverified; see [VALIDATION.md](../VALIDATION.md).

## Build recipe

[Dockerfile](Dockerfile) extends a corrected Cloud Offload ComfyUI runner. Supply its immutable digest after rebuilding the sibling integration. An older base alone lacks these changes. From the repository root:

```sh
export CLOUD_OFFLOAD_IMAGE='your-registry/worker-comfyui@sha256:YOUR_REBUILT_IMAGE_DIGEST'
docker build --build-arg CLOUD_OFFLOAD_IMAGE="$CLOUD_OFFLOAD_IMAGE" -f deploy/Dockerfile -t unimate-runner:local .
```

This explicit build downloads Blender 5.1.1 Linux x64, checks its SHA-256, installs requirements, and runs a Blender import/version check. It sets `UNIMATE_BLENDER`, `HF_HUB_OFFLINE`, and `TRANSFORMERS_OFFLINE`. The recipe is unbuilt/untested; no tested Linux image digest is available.

Preserve the base entry point and `/opt/cloud-offload/runtime-profile.json`. The existing profile identifies `comfyui`, `linux-x86_64`, `cp311`, `comfyui-partition-v1`, and bundle protocol v1. Coordinator configuration must use an image pinned with `@sha256:`, model capability `comfyui-partition-v1`, and matching `image_profile`. Pin this pack's source revision when building the runner. The coordinator checks pack identifiers and reports version differences; it does not enforce installed source-digest equality.

## Artifacts

Install the explicit `.unimate` bundle from the root [setup instructions](../README.md). A loader inside a partition declares its selected `unimate` file; a loader outside sends the full portable bundle. The tested bundle is approximately 706 MiB, so transfers and host memory include the complete payload.

The input loader declares its GLB under `__input__`. Preflight checks content digests, resolves configured sources, and uploads only missing declared identities. Queueing does not download models. Inference uses installed local encoder/tokenizer data.

Animated GLB and provenance JSON return through `3d` and `files`. The gateway restores them under `output/cloud_offload/<job_id>/<remote subfolder>/`, with containment checks and distinct pairs for repeated exports.

Before claiming a deployed runner, build the image, run both synthetic topology workflows, check cancellation/unload, and inspect playback. Local bridge tests do not establish provider deployment or GPU capacity requirements.
