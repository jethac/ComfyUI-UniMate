# Trained conditioning dependency

`build_training_condition` uses validated training model options and portable
statistics. It selects an explicit normalization row and preserves pinned
normalization/topology/collation math. The released wrapper retains its existing
configuration restrictions and matches the trained wrapper for equivalent
inputs. No new public trained-generation workflow is established here.

## Checks

Missing-entry-point tests first failed, then passed. Eight actual frozen model
forwards cover all four backbone axes with 9-/17-frame windows, 11 padded joint
slots, 4-/12-column spectral features and 7-/11-dimensional text embeddings.
Two independent pinned-source numeric pipeline comparisons check every nested
condition field. Invalid normalization, empty tokens, nonfinite text/spectral
features, parent ordering and excess depth are rejected.

Finite but extreme normalization/text inputs first produced nonfinite float32
conditions. Two regression cases now reject normalization-cast and pooled-text
overflow. Independent review identified allocation through configured padding
capacity. Budget/cancellation tests first failed, then passed after preflight
was added before topology work. Review reported no remaining Important findings
within that fix's scope.

Raw bundle motion metadata was previously hardcoded to EMA. Two raw-selection
constrained tests reproduced the wrong label; raw/EMA metadata now comes from
the loaded manifest. Both selections and constraint modes pass plumbing checks.

Windows focused: **38 passed**, 32.12 seconds. Full suite: **1,246 passed,
50 skipped, six subtests**, 481.71 seconds, with pinned dataset and motion
references. Python 3.11.9, Torch 2.11.0+cu128. Ruff passed.

Source pin: `2c5b384715aa63d8639b1ed7eb74bfe614570c7a`. Reference file SHA256:

- Collator: `7cb7672ba97572282ec7a3153abd21aeb4917ecf7f77bf7d3079e13a9f888eb2`.
- Topology: `96ba6f69536de7bca48c1f2710754bd82b3c7dff831c977696bbd2e998a77dfd`.
- Transforms: `0c16cd112fb5df83778d245df06dbeaa040546fda56611421286340ae0fe231a`.

The collator's LF-normalized SHA256 is
`512600e6dfe9193256c2ee4087892c0fd658f9f803d14a0757ba5f502ba9dd34`.
Its reference fixture now pins that complete normalized source and compares
the vendored code after the same CRLF→LF normalization. A separate check proved
the tracked Git blob and normalized reference are byte-identical.
The staged three-file reference archive was checked on both systems with SHA256
`e158a4a524b23c67d68291ab4c7fbfa3042188336ed29a40171c1c61d1d30181`.

Windows logs remain under `.runtime/trained-conditioning-*.log`. The first
stadia run reported two setup errors because an earlier audit source directory
contained only a subset of the source. After staging, the reused fixture's raw
CRLF/LF byte comparison failed; the normalized-source check resolves this
platform difference without ignoring code changes. Final headless checks:
**62 passed**, 5.92 seconds, including the source pipeline and additional
collation tests. Python 3.11.15, Torch 2.14.1+cpu; no CUDA execution claim.
Log: `.runtime/stadia-trained-conditioning-final.log`.
Tested source is this increment applied to baseline
`57be3aace1f25563490a31810bb6cb20d8ce5ca8` in the isolated stadia
`trained-conditioning-audit-20261003` worktree.

## Open gates

Trained bundle assembly, source-bound encoder/statistics loading, flow/diffusion
samplers, configured-window constraints, actual server/worker generation and
independent exported playback remain required. Conditioning helper/fixture
tests do not establish those workflows or full training/UniMate coverage.
