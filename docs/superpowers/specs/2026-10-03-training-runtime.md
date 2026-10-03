# UniMate training runtime

Implement released training inside ComfyUI, using the portable dataset, sampling,
text, sample and batch contracts already established. A usable workflow must train
from scratch or selected raw/EMA weights, consume mixed datasets and epoch plans,
run all released flow/diffusion schedules and architecture ablations, report
progress, support cancellation and emit portable resumable state and usable
inference checkpoints. Distributed/accumulation behavior, EMA, warmup/cosine LR,
gradient clipping, precision and resume must be compared to the pinned source.
No telemetry, downloads, Motion dependency or external captioning in the runtime.

Use ComfyUI's selected device and model management. Live model/optimizer objects
stay in the process; portable sockets carry validated numeric state and metadata.
Model, optimizer, scheduler, EMA, training/data position and RNG states must be
bound together; resumption must reproduce uninterrupted updates. Raw and EMA
selection is explicit. Persist tensors without pickle; reject invalid/mismatched
states before mutating a running model. Artifact publication is atomic after
cancellation checks. Headless Windows/stadia and actual worker runs must prove
batch consumption, training, cancellation, cleanup, resume and artifact retrieval.

The first dependency is the released differentiable flow loss and EMA kernel.
Keep all three flow paths and parameterizations/weightings, masked L2, geodesic
and smoothness options. Preserve numeric operations; validation may reject
nonfinite/zero-valid-length inputs before updating weights. These kernels alone
are not public training coverage. Diffusion, optimizer sessions, portable resume,
architecture/job nodes and integration evidence remain required.

Reference finding: the released geodesic converter produces NaNs for degenerate
6D rotations before padding masks are applied. Keep an explicit `released` policy;
default `stable` substitutes identity in masked/degenerate slots before unchanged
geodesic arithmetic. Test unchanged valid-rotation behavior and reproduce the
source failure. Reject nonfinite per-sample and reduced loss before updates.
