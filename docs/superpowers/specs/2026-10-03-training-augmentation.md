# Released training augmentation adapter

Full training coverage includes source-ordered text encoding, augmentation,
cropping, conditions, normalization, padding, collation and training execution.
Implement the numeric augmentation layer now; public sample/text-cache interfaces
remain required. Do not fabricate embeddings or augment raw mesh export rigs.

Reference: MIT UniMate revision `2c5b384715aa63d8639b1ed7eb74bfe614570c7a`,
`unimate/dataset/mixture/augmentations.py` and dataset.py wrappers. Existing
vendored topology code is compared to that revision before reuse. Motion's
Quaternions/Animation dependency is reference-only, never redistributed here.

Interface: `augment_sample(sample, operation, seed, *, enabled=(), removal_rate=None,
pool_rate=None, max_scale=.1, sigma=.5, lateral_ratio=.5, max_freqs=8,
max_path_len=5, max_workspace_bytes=512*1024*1024, cancel=None,
addition_policy='released')` returns copied
aug-dict and JSON report. Input contains exact source aug keys: motion (T,J,12),
ordered parents, edges, graph distance/relations, depths, spectral features,
tpos/rest offsets, joint embeddings, mean/std (J,12). It has no mesh or local
paths. Rest offsets derive from rest positions before this layer; do not confuse
source animation offsets with identity-rest training offsets.

Operations: none; ellipsoid addition; linear-addition ablation; leaf removal;
single-child pooling; independent bone perturbation; random source wrapper choice.
Random choice is [none]+enabled in released addition/removal/pooling/perturbation
order, choosing at most one. Removal default draws U(.05,.15), pooling U(.1,.3).
Preserve weighted sampling, three-leaf removal cap, at-least-one pooling rule,
inserted HML slots, name-embedding interpolation, copied statistics and source
recomputation of positions/rest/topology. Source does not recompute velocity
channels after offset changes; keep that behavior explicit in evidence.

Ruling from independent FK checks: both released addition variants duplicate the
parent's local BVH rotation into the inserted joint and can change original poses,
despite source comments claiming neutral insertion. Default `released` preserves
that behavior and numerical comparisons. Explicit `neutral_fk` gives the inserted
joint identity local rotation by correcting the shifted child's HML slot; preserve
all original joints' FK poses and report the selected policy. This correction is
independent pack behavior, not a claim about released source. Existing velocity
channels and zero inserted velocity still follow the released policy.

Use local Python Random and NumPy RandomState, seeded with uint32, matching
legacy reference RNG streams without altering global state or threading races.
Cancellation is checked before work, every joint/group edit and every rejection
sampling iteration. Reject malformed/nonfinite arrays and invalid probabilities,
rates, sigma, embedding dimensions or std values. Reject degenerate 6D rotations
when an operation needs FK. Sigma must be nonnegative, rates and perturbation
scale in [0,1], lateral ratio nonnegative. Zero sigma and unit perturbation scale
retain valid released boundary cases. Seed is [0,2**32-1]. The randomized wrapper
fixes sigma/lateral ratio/scale/path length to released defaults; direct-operation
overrides apply only outside that wrapper.

Validate topology-derived inputs consistently. Recompute derived arrays after
structural changes using pinned topology functions. Matrix FK/RIFKE adapters
replace unavailable runtime Quaternion dependencies; compare their output to
unchanged source Quaternion/Animation code on nontrivial rotations and root
facings. Do not claim pooling preserves the original moving pose without checking
that property separately. FK/RIC consistency checks concern the augmented value.

Estimate workspace conservatively before copies and dense eigendecomposition:
8 times input array bytes, 12 float64 J^2 arrays, and 16 float64 T*J*3 arrays,
using J+1 for addition headroom. Default 512 MiB, configurable positive integer.
This is a planned numeric workspace budget, not measured process peak memory.
Storage's broader topology contract does not imply every transform fits that
default budget. Output must be finite and source input immutable.

Validation: all operations/seeds/dtypes against pinned unchanged source bodies,
RNG isolation, selected wrapper operation/rate, interpolated embeddings, graph
changes, shape/dtype/finite checks, cancellation including rejection loops and
pre-allocation budget rejection. Public node/cloud/headless training sample
execution remains open until encoded sample generation is integrated.
