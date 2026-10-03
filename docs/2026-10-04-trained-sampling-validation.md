# Trained sampling validation

The internal sampler consumes a frozen private model, numeric conditioning and
the recorded training job. It returns normalized float32 features. It is not a
public trained-bundle generation workflow.

## Reference evidence

Source: `Friedrich-M/UniMate` at
`2c5b384715aa63d8639b1ed7eb74bfe614570c7a`. Independent tests load the pinned
transport, integrator and Gaussian diffusion source; integrator file SHA-256:
`0ed70897c18f346a1498c45b395888d3a1dae5f2ded3c408254d7a9e7c012c4b`.

- 35 exact Euler/dopri5 comparisons: Linear/GVP/VP, velocity/noise/score, CFG 1/3.
- One dopri5/VP/noise/CFG 3 fixture produces nonfinite values in the pinned
  source and is rejected by the pack. This is source-failure evidence, not a
  successful generation case or a claim that every VP model fails.
- 16 exact ancestral/DDIM comparisons: original/respaced steps, xstart/noise
  prediction, variance/rescaling options and DDIM eta 0.3.
- 16 actual two-update model checks: graph/full attention, AdaLN/cross-attention,
  flow/diffusion and raw/EMA selection. Repeated same-seed outputs are equal and
  finite. Learned variance comparisons use a fixture predictor, not a trained
  learned-variance backbone.
- Invalid options, capacities, budgets, nonfinite values and entry cancellation
  reject before model calls. Mid-sampling cancellation/nonfinite failures restore
  process RNG. Solver workspace is bounded before noise allocation; activation
  memory is outside that estimate.

Storage regression failed with 43,008 retained bytes for a 6,144-byte motion.
Copying the final result removes the retained trajectory; numerical comparisons
remain exact. Read-only review reports no remaining Important findings.

## Runs

Windows: Python 3.11.9, Torch 2.11.0+cu128, NumPy 2.4.3. CPU focused suite:
**89 passed**, 72.73 seconds. Local log:
`.runtime/trained-sampling-focused-final2.log`.

Headless stadia-testbed: Python 3.11.15, Torch 2.14.1+cpu, NumPy 2.4.6.
**89 passed**, 17.00 seconds. Isolated audit:
`/home/jethac/workspaces/comfy-unimate-e2e-20261001/trained-sampling-audit-20261004`.
Baseline `c08dbfb` plus the tested sampler/test files. Reference cloned from a
Git bundle of the pinned revision with CRLF checkout for existing raw-source
hash fixtures. Local log: `.runtime/stadia-trained-sampling-final.log`.

Both initial expanded runs recorded the VP/noise failure before the independent
source-failure check was added. Dependency deprecation warnings remain in logs.
Full Windows suite: **1,335 passed, 50 skipped, six subtests passed**, 228.04
seconds; seven dependency/intentional optimizer-test warnings. Local log:
`.runtime/trained-sampling-full.log`. Skipped opt-in checks are not validated
by this run. Ruff passes for both new Python files; `git diff --check` passes.

## Remaining work

Assemble complete trained model bundles with bound statistics and text encoder;
connect Comfy-managed inference, configured constraints and expansion; validate
actual server/worker generation, exports and independent playback. Cover remaining
source sampling APIs, learned-variance backbone output and required precision and
cancellation/unload matrices. These checks do not establish CUDA sampling, public
trained-generation support or full UniMate coverage.
