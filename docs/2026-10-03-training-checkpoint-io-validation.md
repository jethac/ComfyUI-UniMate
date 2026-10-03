# Training checkpoint files and worker resume

The public Load/Save Training Checkpoint nodes persist `.unimatetrain` files.
The envelope contains bounded JSON and safetensors, with no pickle or dynamic
imports. Digest, layout, references, counters and finite data are checked on
load. Full session compatibility is checked by Train against the selected job,
artifacts, model and runtime.

## Verification

Missing checkpoint IO, nodes and workflow harness first failed tests; their
implementation passed. Windows full suite: **1,182 passed, 50 skipped,
6 subtests**, 60.25 seconds. Six warnings were existing JIT/SWIG deprecations and
intentional optimizer-hook tests. Ruff and `git diff --check` passed. Headless
stadia focused IO/node/workflow tests: **21 passed**, 5.21 seconds.

`tools/training_execution_workflow.py` ran actual CPU ComfyUI servers on Windows
and stadia-testbed. Each runtime executed two direct baselines and three worker
partition-handler jobs: one update, two updates resumed over a socket, and two
updates resumed from a staged checkpoint file. Server logs recorded five model
loads. All three jobs restored one checkpoint file through the actual client.
Complete checkpoint values, including tensor bytes, matched uninterrupted
training within each runtime. Progress advanced from one to three updates.
The dataset had one batch per epoch, so these runs do not prove full-group
accumulation on multi-batch epochs; separate execution tests cover that case.

Training used a small scratch graph/AdaLN flow model and an actual prepared T5
cache (`google/flan-t5-base`, artifact SHA256
`f79264a13a940769912d95fa9521c1f6c4448e1fe129d1ad4e1cf6a7c4a4259d`).
No new model download or provider call was made.

Local reports:

- `.runtime/training-execution-worker-check-1/report.json`
- `.runtime/stadia-training-execution-worker-report.json`
- `.runtime/training-checkpoint-io-full-suite.log`

Remote report:
`/home/jethac/workspaces/comfy-unimate-e2e-20261001/run-training-execution-worker-20261003-1/report.json`.

Run the harness with `--comfy-root`, `--cloud-root`, `--client-root`, `--python`,
`--source-dir` and `--workdir`. Source directories contain the verified dataset,
statistics and text cache from `tools/training_workflow.py`:
`direct-dataset.part`, `direct-statistics.part`, `direct-cache.part`.
Windows source was `.runtime/training-worker-check-2/partition`; stadia source
was `run-training-worker-20261003-1/partition` under the workspace above.

## Identities and limits

Runs used baseline pack `338eb5749ea2a39bf65bf614bd83327e534396a1` plus this
increment. Windows: Python 3.11.9, Torch 2.11.0+cu128, CPU server; ComfyUI
`e2f44d7fe65e270ac111237366b03e396b94dcea` with existing local changes.
Stadia: Python 3.11.15, Torch 2.14.1+cpu; clean ComfyUI
`84ba85773925f071c516f0208184773802b4d44a`.
Runner: `43bd1a0d998ffcba2568de2289cd3131f271aaa6`.
Client: `4a7a9376d8e20cc3decfba26cd1e626bd198ab7a`.
Pinned upstream: `2c5b384715aa63d8639b1ed7eb74bfe614570c7a`.

Matching Windows/stadia source SHA256 values:

| File | SHA256 |
| --- | --- |
| `unimate_pack/training_checkpoint_io.py` | `f8a17a06bcb069f8ec8d079b0b9e0d3ac84623a6e7192497975a384c9e80892e` |
| `training_nodes.py` | `31699443797951ed577a9d7b145729c713f4fd0b92348aef6889474bf0a3971d` |
| `tools/training_execution_workflow.py` | `c0710b2b42ae37216b9a33616ca24a5ce1d0054d8e87fc27d26f0342ef934f6e` |

No cross-platform checkpoint equality, installed-weight training, inference
export, distributed/unbalanced loading, provider/container/live coordinator or
injected server cancellation is established. A node API test verifies
cancellation after file staging and before publication removes temporary output.

Independent review found no Critical or Important defects. Two Minor test gaps
are deferred: patch the imported `load_tensors` alias in the no-allocation test;
add BF16/BOOL and post-4-MiB-boundary corruption cases. These are not coverage
claims for unsupported training precision or server cancellation.
