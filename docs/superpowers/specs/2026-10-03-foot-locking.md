# Foot locking

Implement the postprocess described in UniMate Appendix E.5, independently of
inference. Input is a validated prepared rig and same-rig motion; output is a
portable motion of the same length, fps and rig identity, plus a JSON report.

## Source requirements

Source: https://arxiv.org/html/2609.05415v1#A5.SS5

Auto-select distal ground-contact joints using canonical Foot/Toe/Ball/Paw/Hoof/
Pastern names. Normalize distances by rest root height above the lowest rest
joint. Contact uses normalized height <0.05 and horizontal displacement <0.025.
Apply a five-frame binary median filter and discard runs shorter than three
frames. Anchor each run at its mean horizontal location on the ground plane.
Use damped least-squares IK on the contact chain, with at most three parents;
keep root motion and other chains unchanged. Blend corrections at both ends over
five frames. The inspected release has no located implementation of this method.

## Implementation decisions

- Use FK positions, since they drive exported skinning. Preserve the existing
  parent-slot 6D rotation convention; write each changed parent's rotation into
  all its child slots. Leaf orientation is not represented by this feature format.
- Match original and cleaned names with case-insensitive canonical-word tokens;
  explicit joint overrides permit authored nonstandard labels. Do not call a
  language service to infer names. Dataset vocabulary translation remains a
  separate coverage requirement.
- Exclude the root from rotational optimization. Stop a chain before a branching
  ancestor that would move an unrelated chain. Joint-local rotations outside
  the optimized chains remain byte-identical.
- Compute the IK Jacobian from global joint axes and target lever arms. Use a
  bounded angular update, damping and finite iteration budget; report residuals
  for unreachable targets rather than changing root motion or bone lengths.
- Blend solved local rotations with quaternion interpolation. The five-frame
  boundary ramp includes weights 0 and 1. Short segments may contain only ramps.
  This boundary convention and solver parameters are implementation choices,
  since the paper does not specify them numerically.
- Update position/velocity features only where the corrected limb requires it.
  Preserve all root feature channels, canonical root origin and unrelated chains.
- No detected contacts returns unchanged feature bytes with a skipped report.
  A selected contact rig with zero rest root height fails explicitly.
- Check cancellation during contact analysis, each frame and IK iteration. Use
  bounded numeric arrays and existing portable motion persistence.

## Verification

Selection tests cover distal joints, multiple limbs, canonical tokens and manual
aliases. Contact tests cover normalized thresholds, median boundaries and short
runs. Compare analytical Jacobians with finite differences. A new original MIT
legged fixture must exercise reachable anchoring, immutable root/unrelated chains,
bone-length preservation, ramps and unreachable residuals. Compare the modified
pose with Blender export evaluation independently. Exercise numeric save/load,
FK/RIC previews, Windows server execution, headless stadia and actual partition
handler restoration. Record exact source/runtime/model identities for model runs.
