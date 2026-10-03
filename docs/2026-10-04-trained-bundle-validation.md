# Trained bundle validation

Assemble Model returns `UNIMATE_MODEL` and atomically saves a managed `.unimate`.
The existing Model Loader accepts its numeric `unimate.bundle.v2`. This record
covers assembly and reload, not trained generation or playback.

## Contract evidence

The bundle stores selected denoiser safetensors and metadata, statistics arrays
and metadata, cache identity metadata, canonical sampling options and 30 fps.
Source checkpoint, job, selected weights and full statistics/cache identities
must match the recorded training job. Required encoder assets retain their
license/notice and match the cache's exact installed artifact identity and width.
Only FLAN-T5-base encoder-bearing assembly is currently implemented.

Raw/EMA encoder-free round trips cover all four actual backbone axes. Reloaded
state is exactly equal to selected state. Actual forwards with NaN caption/name
embeddings remain finite and unchanged when both conditioning gates are disabled.
Caption and joint-name modes independently require a matching encoder.

Tests reject changed statistics/cache metadata, mismatched encoder inventories,
unsafe/unknown/incomplete members and rehashed manifest mismatches. Existing
common archive tests cover duplicate files, links, encryption, corrupt digests
and size limits. No cached text tensors or optimizer state enter the bundle.

Regressions reproduced before fixes:

- Invalid numeric payloads raised TypeError; they now reject as ValueError.
- Final contract validation dropped the selected workspace/cancellation settings.
  Both are preserved through assembly, donor validation and reload. Loader budget
  rejects before file reading; entry and final cancellation are checked.
- ZIP timestamps changed bundle bytes; fixed timestamps, permissions and sorted
  members make repeated assembly deterministic.
- The added Model Loader workspace input was required and broke an existing reload
  graph with HTTP 400. It is now optional, with a default of 32,768 MiB. The same
  graph is used in the final server rerun.

Read-only review reports no remaining Important findings in assembly. Runtime,
generation and playback integration are separate required work.

## Runs

Windows: Python 3.11.9, Torch 2.11.0+cu128, NumPy 2.4.3. Before the final optional
loader correction, full suite: 1,368 passed, 50 skipped, six subtests passed in
352.05 seconds. Final full suite: **1,368 passed, 50 skipped, six subtests passed**,
174.57 seconds; local log `.runtime/trained-bundle-final-full.log`. Skipped opt-in
checks are not validated by this run. Dependency and intentional
overflow/optimizer-test warnings are recorded in the logs.

Headless stadia-testbed: Python 3.11.15, Torch 2.14.1+cpu, NumPy 2.4.6.
Final focused suite: **88 passed, six subtests passed**, 17.30 seconds; local log
`.runtime/stadia-trained-bundle-final.log`. Before the optional correction:
88 passed, six subtests passed in 19.81 seconds.
The source audit checkout is
`/home/jethac/workspaces/comfy-unimate-e2e-20261001/trained-bundle-audit-20261004`,
based on `f81a50a` plus tested changes. Source remains training revision
`2c5b384715aa63d8639b1ed7eb74bfe614570c7a`.

Actual CPU server/worker rerun passed two direct raw/EMA baselines, **three handler
jobs and two retrieved bundle files**, plus declared model staging and reload.
Weights/statistics reloaded from each complete bundle exactly match the inputs;
direct and worker bundle bytes/hashes match. Reload retrieves no new file and
returns the same raw bundle. No generation or playback ran in this harness.

Report:
`/home/jethac/workspaces/comfy-unimate-e2e-20261001/run-trained-bundle-worker-20261004-2/report.json`.
Local copies: `.runtime/stadia-trained-bundle-worker-report.json` and
`.runtime/stadia-trained-bundle-worker-final.log`.

| Selection | Job | Bundle SHA-256 |
| --- | --- | --- |
| Raw | `9190d6df-75a6-4527-81f8-af05e45a4e5a` | `1f8ed96249abf973d5dd6873c7830eb3621c16df3fc94eb1235c0ecfd9a1e163` |
| EMA | `77d09c6e-ac8a-4231-af5b-f67eccdbf5eb` | `825633593d17d915b4c41abcb5f5968b8f53f50120a19258985efd025436996a` |
| Raw reload | `f8eecd0d-2888-4c25-9c8f-fc22fd7b9d8b` | `1f8ed96249abf973d5dd6873c7830eb3621c16df3fc94eb1235c0ecfd9a1e163` |

Source checkpoint identity:
`9da295ef2ebd79a574eeb7dd7f0579ed1b09de00bb1dc7ee7a77d5e164fe3b33`.
ComfyUI: `84ba85773925f071c516f0208184773802b4d44a`; runner:
`43bd1a0d998ffcba2568de2289cd3131f271aaa6`; client:
`4a7a9376d8e20cc3decfba26cd1e626bd198ab7a`. Actual device: CPU.

The first audit retains two direct and two worker assembly/retrieval results and
the failed reload prompt in
`run-trained-bundle-worker-20261004-1`; no final report was written for that failed
run. Final evidence above comes from the completed second run.

Ruff and `git diff --check` pass. Model data and generated audit artifacts are
excluded from Git.

## Remaining work

Connect complete bundles to Comfy-managed trained runtime, configured conditioning,
recorded sampling, constraints and expansion. Validate actual generation,
GLB/FBX export, independent playback, Windows server execution, server cancellation
and unload. Other encoders, learned-variance backbone output and the full source
sampling/precision matrix remain open. Full UniMate coverage is not achieved.
