# Dataset splitting and epoch sampling

Full training coverage requires released split and MixtureSampler behavior,
alongside augmentation, conditioning, collation and training execution. This
subsystem integrates the first two without claiming the others are complete.
Reference: UniMate `2c5b384715aa63d8639b1ed7eb74bfe614570c7a`,
`unimate/dataset/mixture/dataset.py` (_split_names and MixtureSampler).

## Decisions

Use pure dataset adapters plus two V3 nodes. Embedding split logic in a training
runner would hide useful dataset curation and impede standalone verification.
Calling the whole upstream dataset constructor would require unsafe legacy
archives and unrelated training dependencies. Neither is needed for these rules.
Direct main implementation/push and autonomous full-goal execution are authorized.

`split_dataset(dataset, ratio, seed, options, cancel=None)` returns a dataset and
JSON-compatible report. Options is bounded duplicate-key-free JSON with optional
`modes` (dataset label -> clip/object_type) and `explicit_eval_objects` (dataset
label -> list of object labels). Recompute train/eval membership over all input
clips. Default modes match the released constructor: object_type for truebones
and objaverse, clip for mixamo and otherwise unlisted labels. Overrides merge.
Preserve clip order, IDs, provenance and exact payload bytes. Preserve the
released sorted traversal, Python Random(seed), round-to-even counts, at-least-one
train clip/type, explicit override, ignored missing held-out objects and refusal
to drain a dataset. Report missing/unmatched objects without logging private data.
Require finite ratio in [0,1), integer seed in [0,2**64-1]; Boolean is not integer.
Unknown option fields/modes and malformed labels fail before publication.

`sampling_plan(dataset, alpha=.5, dataset_alpha=None, epoch=0, cancel=None)` selects
train clips only. Group by (dataset_type,object_type) in original clip order.
Single level gives per-clip n_k**(-alpha); two levels gives dataset mass
N_d**(1-dataset_alpha), object mass n_k**(1-alpha), divided by object clip count.
Normalize in float64 using released operation order. Finite exponents accepted;
reject nonfinite/nonpositive resulting weights. Epoch is the CPU Torch generator
seed; draw exactly N indices with replacement, matching released multinomial.
Do not alter global RNG state or select a GPU.

Value UNIMATE_SAMPLING schema `unimate.sampling.v1`: dataset_sha256, clip_ids,
alpha, dataset_alpha, epoch, upstream_revision, arrays, sha256. Exact fields;
clip IDs are ordered unique training IDs. Numeric arrays: weights float64 (N,),
indices int64 (N,), positive finite normalized weights, indices [0,N). Identity
uses the same canonical source-dataset envelope as statistics. Strict validation
and source matching precede later training consumption. Values contain no paths,
model objects or Torch tensors. Limit payload and declared expansion to 1 MiB
(4096 weights/indices require 64 KiB). No new persistence format is necessary:
existing Cloud Offload dictionary/bytes transport carries this value.

Public nodes: Split UniMate Dataset (dataset,ratio,seed,options -> dataset/report)
and Plan UniMate Sampling (dataset,alpha,two_level,dataset_alpha,epoch -> plan/report).
Cancellation propagates before/within grouping and before returning results.
Keep source inputs immutable. Validation and generation compare to unchanged
pinned AST method/class bodies, including multi-dataset shared object names.
Headless Windows/stadia workflows must split, sample and capture/restore values.
Coordinator discovery, injected worker cancellation and full training remain
separate full-goal gates; do not relabel them complete from codec checks.
