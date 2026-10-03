# Prepared-data training execution validation

Source: UniMate 2c5b384715aa63d8639b1ed7eb74bfe614570c7a.
Baseline: 34c0c98. This increment adds the Train node and private execution and
ComfyUI residency adapters; source backbone and optimization math are unchanged.

Windows: full suite `python -m pytest -q --tb=short`, plugin autoload disabled,
explicit source/Motion test references, `OMP_NUM_THREADS=2`. Result: 1,168 passed,
50 skipped, six subtests, six warnings. Log: `.runtime/training-execution-full-suite.log`.
Python 3.11.9, Torch 2.11.0+cu128, NumPy 2.4.3, RTX 5060 Ti.

Headless stadia-testbed: isolated `training-execution-audit-20261003` worktree
under `/home/jethac/workspaces/comfy-unimate-e2e-20261001`, based on 34c0c98 with
the increment's files copied in. `PYTHONPATH` points to that worktree;
plugin autoload disabled, `OMP_NUM_THREADS=2`. Tests: training execution, job,
nodes and checkpoint modules. Result: 97 passed, three CUDA skips, one JIT warning.
Python 3.11.15, Torch 2.14.1+cpu, NumPy 2.4.6.

Verified behavior:

- Actual prepared datasets/statistics/text caches produce batches and optimizer
  updates; complete checkpoints match uninterrupted vs chunked/resumed execution
  for flow and diffusion and accumulation counts one/two.
- A three-batch epoch resumes from batch two, crosses its partial final group and
  subsequent epoch, and matches uninterrupted state.
- The public ComfyUI V3 node runs through actual selected model residency and
  releases its patcher. Its checkpoint resumes through a different Python package
  namespace using a stable known-backbone identity.
- Explicit gradient contexts permit training under outer inference mode/no-grad.
- Wrong epoch-plan identities fail before optimizer updates. Progress callback
  failure exits residency; invalid chunks/workspace fail before model allocation.
- Null loss options normalize to source defaults; oversized full-attention
  activation estimates fail before constructing a model.
- Existing checkpoint corruption, precision, rollback and codec tests pass.

Matching Windows/stadia SHA256 values:

| File | SHA256 |
| --- | --- |
| training_execution.py | b4edb206703d68bb19fe05ff3c2749a38c1639b18103509d10b77f1a4f718f48 |
| training_residency.py | e4d6dcd1279c3309b78b27d706ea205dc6d23bab9c907f1e325b4684d983508b |
| training_checkpoint.py | b9d43b5b7e3e5bcc6f560461e82c9e45e1215c937f1d41d1825e2ade64764d91 |
| training_job.py | 249be8b9b39440e5d69165a094dcd88188ec4e2f7cd84769ba3609310c1258b9 |
| test_training_execution.py | 00360c2f4bef28290f6fe9bee55c227a02e475f95bda20f3d0fd29a66acb0b94 |

This is node API and prepared synthetic-data evidence. It does not establish
actual ComfyUI server/partition-handler training, checkpoint file retrieval,
installed-weight updates, exported inference checkpoints, distributed/unbalanced
loaders, learned-variance backbone output or full UniMate coverage.
