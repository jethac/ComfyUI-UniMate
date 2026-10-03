# Training configuration validation

Source: UniMate 2c5b384715aa63d8639b1ed7eb74bfe614570c7a.
Factory SHA256: 51fb3959d2efb3f80e93cd8270d07a65de9312605675b554228219fea0a00e3a.
Implementation baseline: 476b9bd; configuration increment follows this commit.

Windows full suite: `python -m pytest -q --tb=short`, plugin autoload disabled,
explicit upstream and Motion test references. Result: 1,152 passed, 50 skipped,
six subtests, six warnings. Log: local `.runtime/training-job-full-suite-final.log`.
Runtime: Python 3.11.9, Torch 2.11.0+cu128, NumPy 2.4.3, RTX 5060 Ti.

Headless stadia-testbed: isolated `training-job-audit-20261003` worktree under
`/home/jethac/workspaces/comfy-unimate-e2e-20261001`, based on 476b9bd with the
increment's files copied in. `PYTHONPATH` points to that worktree; source factory
and backbone test references are copied under its `source` directory. Plugin
autoload disabled; `OMP_NUM_THREADS=2`. Tests: `test_training_model.py`,
`test_training_job.py`, `test_training_nodes.py`. Result: 58 passed, one CUDA skip,
one JIT deprecation warning. Runtime: Python 3.11.15, Torch 2.14.1+cpu, NumPy 2.4.6.

Matching Windows/stadia source SHA256 values:

| File | SHA256 |
| --- | --- |
| training_model.py | 67a3bd57d23b8080113dd16ad54c993aaa538d414627623d8c4067848f6d33a9 |
| training_job.py | 65790583669b40ae34afc5717b1b16066e992bf20dfdb4d0902e650479451c27 |
| training_dataset_samples.py | 185812f209e2785a6ab92b22cbf93b61ce26afa168c1b8cb20a825168482e240 |
| test_training_job.py | 807546bcc3aa395d25e7ffffa1c27900d10d15971019f8b58fb9bae2aa14f7a3 |

Evidence covers factory initialization, source-body equivalence, artifact-bound
jobs, balanced epoch plans, actual ComfyUI V3 schemas/execution, codec round trips,
invalid input and cancellation boundaries. It does not establish training server
execution, model residency, installed-weight updates, exported inference state,
worker execution/retrieval, distributed loaders or full technology coverage.
