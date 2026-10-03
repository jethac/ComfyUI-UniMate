# Selected inference weights

Export Inference Weights selects raw or EMA denoiser state from an artifact-bound
training checkpoint. Load Inference Weights restores the numeric intermediate
file. These nodes do not yet assemble a complete inference bundle or sample it.

The checkpoint/job binding, model class, layout, source trainability, aliases,
finite data and EMA configuration are checked before numeric decoding. The
output binds the source checkpoint identity and includes the exact training job.
Optimizer, scaler and RNG numeric state are omitted. Resume validation remains
separate. Workspace preflight reserves eight payload copies plus 64 MiB.

## Automated checks

All four backbone combinations passed raw and EMA selected-state and frozen
forward comparisons after two actual optimizer updates, for both flow and
diffusion (16 cases). EMA reference application uses the vendored source
`EMAModel`; full selected state and forward outputs compare exactly. Numeric
file round trips are deterministic. Tests cover managed paths, input staging
declarations, fingerprints, explicit selection, malformed job/layout/alias/EMA
data, workspace rejection and cancellation before decode.

Independent review reproduced forged trainability allowing raw tensors labeled
EMA. Four regression cases first failed, then passed after flags were bound to
the known factory's trainability before numeric decoding. The reviewer reran
all four successfully and reported no remaining findings within the fix scope.

Windows full suite: **1,223 passed, 50 skipped, six subtests**, 211.33 seconds,
with pinned dataset and motion references enabled. Focused Windows checks:
**59 passed, nine skipped**, 104.42 seconds. Headless stadia focused checks:
**58 passed, ten skipped**, 22.10 seconds. These skips include opt-in/source and
CUDA-only checks; the CPU headless results do not establish CUDA execution.
Ruff and whitespace checks passed.

## Headless server and worker

`tools/inference_weights_workflow.py` passed against the actual three-update
raw-initialized v2 checkpoint on stadia. Two direct server exports provided
baselines; three actual worker-handler jobs exported raw, exported EMA and
reloaded a declared input. Portable values matched baselines exactly. Both
retrieved files matched client-restored bytes and parsed to the expected value;
reload produced no new file. A separate check loaded the checkpoint into its
known model and applied source `EMAModel.copy_to`: all **443** named tensors in
each worker-exported selection matched exactly.

The first run failed because the harness assumed a `files` UI key for an empty
reload result. Client source confirmed it returns `{}`; the corrected fresh run
passed. No node implementation changed after the full suite.

Source checkpoint identity:
`9da295ef2ebd79a574eeb7dd7f0579ed1b09de00bb1dc7ee7a77d5e164fe3b33`.
Raw output identity:
`734d8718583263a1d824efa207f5a0cb058e3bcb66cedb97356e42efa118a19e`.
EMA output identity:
`b8ac4dc025f7e6ad10c807eba4563642d57df9071b3085c10b52843b9cba96fe`.

Tested source was this increment applied to baseline
`ec30523add13f627e5795816c00461a20604b823`. ComfyUI:
`84ba85773925f071c516f0208184773802b4d44a`; runner:
`43bd1a0d998ffcba2568de2289cd3131f271aaa6`; client:
`4a7a9376d8e20cc3decfba26cd1e626bd198ab7a`.
Stadia: Python 3.11.15, Torch 2.14.1+cpu, Transformers 5.18.0, CPU server with
four intra-op threads. Windows automated checks: Python 3.11.9, Torch
2.11.0+cu128. Training source remains pinned to
`2c5b384715aa63d8639b1ed7eb74bfe614570c7a`.

Reports remain under `.runtime/stadia-inference-weights-worker-report.json` and
`.runtime/stadia-inference-weights-selected-state.json`; remote artifacts remain
under `run-inference-weights-worker-20261003-2`. Windows server export, injected
server cancellation/unload and provider/container/live coordinator execution
are not established by these checks.

## Remaining workflow gates

Complete trained
bundle assembly with exact statistics and text encoding, generalized sampling,
constraints, export and independent playback remain open under GOAL.md.
