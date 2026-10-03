# Foot-locking implementation

Execute inline under the active full-coverage goal. Spec:
../specs/2026-10-03-foot-locking.md

- [x] Add canonical contact selection and detection tests before implementation.
  File: tests/test_foot_contacts.py. Production: unimate_pack/foot_contacts.py.
- [ ] Add original legged rig/motion fixtures and finite-difference Jacobian tests.
  Implement bounded damped IK in unimate_pack/foot_lock.py; verify root, lengths,
  unaffected chains, boundary ramps, cancellation and unreachable reports.
  Numeric helpers in foot_ik.py now pass central finite-difference Jacobian,
  reachable-anchor, unchanged-root/other-chain, unreachable and cancellation tests.
  Chain selection and five-frame quaternion boundary blending pass numeric tests.
  lock_features now passes a synthetic legged-pose test for interior anchoring,
  unchanged root/other-chain features, bone lengths and FK/RIC foot agreement.
  Portable adapter and UniMateFootLockMotion are wired and tested; the original
  legged GLB fixture has validated binding/ground contacts. Raw animation-channel
  evaluation agrees with every corrected canonical pose across 20 frames.
  Review found overlapping manual contact chains could overwrite each other;
  the adapter now rejects them explicitly. Successful portable correction,
  moving root/facing, independent feature re-encoding, mid-operation cancellation,
  malformed overrides and selected zero-height behavior now pass. An early
  workspace guard rejects estimated allocations above 512 MiB before FK.
  Estimate: 20 times feature bytes plus 128 bytes per frame; not a measured peak.
- [x] Wire Foot Lock Motion using latest V3 schemas; keep contracts portable.
  Update __init__.py, nodes.py and schema/contract regression checks.
- [x] Compare all exported foot poses with Blender and independent glTF skinning.
  The original 60-frame legged clip passed Blender 5.1.1 import/playback against
  independent glTF skinning at every frame, using bidirectional nearest-vertex
  error below 2e-5. Windows test: 1 passed in 92.18 seconds. Source appearance and
  corrected pose channels also pass numeric checks. Server/partition checks pass.
- [x] Execute Windows and headless stadia workflows including save/load and previews.
- [x] Execute actual worker staging, motion restoration and export paths.
- [ ] Review, update README/DESIGN/COVERAGE/VALIDATION and push direct to main.

Foot metrics and solver configuration belong to the owning numeric modules.
No inference math is replaced. Training, dataset utilities, deformation options,
model-capacity/statistics matrices and deployed-worker gates remain in scope.

Current verification: tests/test_foot_contacts.py, tests/test_foot_ik.py and
tests/test_foot_lock.py cover numeric foundations, feature correction and public
node plumbing. Server/partition evidence is recorded below. The final velocity channel
has no next stored pose; the adapter retains that source terminal velocity.

Windows and stadia partition-handler checks now pass. Each retrieved 369 files;
independent playback checked two exports / 120 frames with exact restored
skinning. Windows evidence: `.runtime/foot-lock-worker-check-2`; stadia evidence:
`.runtime/stadia-foot-lock-worker`. Broader assets and deployed workers remain
full-coverage gates.
