# Goal: full UniMate coverage

Implement usable ComfyUI nodes covering UniMate's technology, released code and models, and paper-described capabilities. Use COVERAGE.md as the minimum checklist; audit further source changes without silently reducing scope.

Required coverage:

- Text-conditioned and unconditional generation, text-mediated cross-topology transfer, batches and repetitions.
- In-betweening, joint-preserving text-guided editing, and multi-prompt motion expansion.
- Reference-motion import/extraction, numeric save/load, variable-length motion contracts, and frame/joint selection.
- Rest-pose preprocessing, canonical assets and conditioning outputs.
- Every released model family, validated config reconstruction, checkpoint selection and normalization.
- Animated GLB/FBX export, skeleton previews, FK and RIC recovery paths.
- Released training and dataset-processing capabilities.
- Paper-described postprocessing, including foot locking and deformation options; locate upstream implementations or independently implement and validate missing methods.

Preserve appearance, skinning, coordinates, rig identity, offline runtime, cancellation and cleanup. All applicable nodes and values must support Cloud Offload staging, execution and output retrieval.

Compare numerical behavior against pinned upstream references. Run the expanded inference workflows headlessly on stadia-testbed and check exported playback independently. Record exact source, model, runtime and device identities.

Keep documentation and coverage evidence accurate. Push authorized completed work directly to jethac repository default branches without PRs.

Completion requires implementation and authoritative validation evidence for every capability. Missing upstream code is a gap to resolve, not completed coverage. Neither the existing five-node path nor an unrelated automatic-rigging project satisfies this goal.

This goal supersedes the earlier generated-mesh/automatic-rigging objective at the user's instruction.
