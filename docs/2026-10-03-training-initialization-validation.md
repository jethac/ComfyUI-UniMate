# Installed training initialization

Configure Training and Train accept the same optional installed model value.
Configure derives the released architecture and binds the bundle SHA256,
denoiser SHA256 and explicit raw/EMA selection. Train rejects missing/different
artifacts and conflicting architectures before allocating a real model.
The owned CPU model receives selected weights before optimizer and EMA creation.
Resume restores complete session state with that initialization binding.
Scratch job values retain their existing v1 schema and identities.

The bundle converter accepts `--weights raw` or `--weights ema` (default).
Raw requires the complete raw state; EMA requires compatible shadow parameters
without fallback. Both validate dtype, finite data and shared aliases. The bundle
manifest records the selected kind. Runtime archive loading never uses pickle.
The explicit conversion command retains restricted `weights_only=True` loading.

## Automated and reference checks

Tests first failed for the absent initialization module and absent initialized
job interface. Focused checks passed for explicit selection, architecture
derivation, identity mismatch, malformed state rejection without model mutation,
shared aliases, budget and cancellation. Independent review found one Important
issue: config/header compatibility was checked after real model allocation.
The regression test first failed, then passed after meta-device inventory,
shape/dtype and alias preflight was added. No Critical findings were reported.
The reviewer could not collect tests under system Python; results below use the
project test environment.

Final Windows full suite: **1,190 passed, 50 skipped, six subtests**, 71.11 seconds.
Final headless focused checks: **90 passed, 10 skipped**, 14.22 seconds.
Source-dependent factory checks and CUDA-only cases account for focused skips;
the Windows full suite enabled pinned source references. Ruff passed.

The explicit raw conversion of official v2 step-100000 passed offline. All
**443** named tensors in the resulting bundle match restricted-loaded raw source
state exactly, including shared aliases. Final meta-device preflight passed on
the actual artifact. Local record: `.runtime/raw-bundle-reference-validation.json`.
The artifact SHA256 is
`95164c867fc9ea4bb6719e97160c24f7ff3b09a10a933e2912515c3453f477a1`;
denoiser SHA256 is
`65f510f85aadffae571e13c207724590fa91fe45119638c471ab9b10f95e5090`.
Weights and runtime evidence remain outside git.

## Server/worker verification

`tools/training_execution_workflow.py --initialization-bundle <installed bundle>`
runs two direct baselines and three partition-handler jobs for socket/file resume
and actual client artifact retrieval. Both nodes use a 32,768-MiB workspace.
Windows and stadia EMA runs completed with exit zero. Each passed all three
handler jobs, three retrieved checkpoint files, exact complete-state comparison
with uninterrupted training and progress from one to three updates. Windows and
stadia raw runs also completed with exit zero and passed those comparisons.
Each of the four runs loaded the actual model five times: two direct baselines
and three handler jobs. In total, twelve jobs retrieved twelve checkpoint files.
No Windows/Linux byte-equality claim is made. The dataset has one batch per epoch,
so these runs exercise partial accumulation groups; separate execution tests
cover full multi-batch accumulation groups.

Local workspaces:

- `.runtime/initialized-training-worker-check-1` (EMA)
- `.runtime/initialized-raw-training-worker-check-1` (raw)

EMA report copies include `.runtime/stadia-initialized-training-worker-report.json`.
Raw report copies include `.runtime/stadia-initialized-raw-training-worker-report.json`.
The EMA bundle SHA256 is
`3d4420752e64b873f98c8aec2d6f01edf7be861c704920dfc095bd1d700664b8`.

Stadia workspaces under
`/home/jethac/workspaces/comfy-unimate-e2e-20261001`:

- `run-initialized-training-worker-20261003-1` (EMA)
- `run-initialized-raw-training-worker-20261003-1` (raw)

These runs use baseline pack `99e8ae76fc7cf62c51ae2fd0404a5ca18e970242` plus
this increment. EMA servers started before the final meta-device preflight fix;
raw servers include that fix. The fix adds validation before allocation and does
not change initialized numerical state. Do not label both runs as identical
source versions. Final EMA meta-device preflight separately passed on the actual
bundle (`.runtime/ema-final-preflight-validation.json`).

Windows: Python 3.11.9, Torch 2.11.0+cu128, CPU server; ComfyUI
`e2f44d7fe65e270ac111237366b03e396b94dcea` with existing local changes.
Stadia: Python 3.11.15, Torch 2.14.1+cpu; clean ComfyUI
`84ba85773925f071c516f0208184773802b4d44a`.
Runner: `43bd1a0d998ffcba2568de2289cd3131f271aaa6`.
Client: `4a7a9376d8e20cc3decfba26cd1e626bd198ab7a`.
Pinned training source: `2c5b384715aa63d8639b1ed7eb74bfe614570c7a`.
Source text cache artifact:
`f79264a13a940769912d95fa9521c1f6c4448e1fe129d1ad4e1cf6a7c4a4259d`.

Final raw-run sources matched on Windows and stadia:

| File | SHA256 |
| --- | --- |
| `unimate_pack/training_initialization.py` | `8e520e3312c233ffc062c4d681906375c7ae44ce74f2f715621420b627c4feda` |
| `unimate_pack/training_model.py` | `e141559a9f5e983445f4ddad1b851a341f9b790c507a96aef6f507841b0beffd` |
| `unimate_pack/training_execution.py` | `e7ec838398ce214843d80ff68c6ef10d9d0a50932c4a0b9cbdb9b6df7c7cd283` |
| `training_nodes.py` | `e0df0dfc4a9b651345a5ba41e84fc3c2aa6a22601d0a72738ed66b7d4d2b140a` |
| `tools/training_execution_workflow.py` | `1fb4417cd2f732c33bad861929d8965e7c4631049b21f946722447b0b7005f25` |

Inference export, other installed model families, the full training matrix,
distributed/unbalanced loading and injected server cancellation remain open.
