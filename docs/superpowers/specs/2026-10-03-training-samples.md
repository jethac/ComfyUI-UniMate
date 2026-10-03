# Encoded training samples and batches

## Purpose and scope

Complete the released training data path after numeric dataset/statistics/sampling
and augmentation foundations. Full GOAL.md remains authoritative. Add usable
sample/cache/batch nodes, not mock embeddings or a substitute training format.
Implementation is authorized by the existing full-coverage instruction.

## Source baseline

UniMate revision: 2c5b384715aa63d8639b1ed7eb74bfe614570c7a.
SHA-256:
- dataset/transforms.py: 0c16cd112fb5df83778d245df06dbeaa040546fda56611421286340ae0fe231a
- dataset/mixture/collate.py: 7cb7672ba97572282ec7a3153abd21aeb4917ecf7f77bf7d3079e13a9f888eb2
- utils/text_emb_cache.py: f1433ec15449a77721d921ebc16ae76ef8a552eed79b6413839370e64725dd01
Dataset/motion hashes are pinned in test_training_augmentation.py.

## Ordering and exact behavior

1. Resolve clip identity, conditioning, source features and matching statistics.
   Joint names use released model_joint_names fallback rules. Derive identity-rest
   offsets from grounded canonical rest positions, not arbitrary raw rig offsets.
2. Obtain actual installed encoder outputs or a matching portable text cache.
   Fresh dataset joint names and captions pool trimmed token rows. Cache hits
   preserve their stored joint vector and separate caption pooled/token views.
   The existing inference runtime returns both views; do not assume its pooled
   vector is byte-identical to the fresh dataset trimmed mean.
3. Assemble the augmented dictionary and invoke the source-compatible adapter.
   Explicitly retain released versus neutral_fk policy and velocity semantics.
4. Crop: first_frame takes max_length+1 and default start zero; tpos takes
   max_length with a local seeded uniform start. Explicit start may yield a short
   tail. Reject invalid ranges before allocation. Realign when enabled and start
   is positive: root facing q_t*q0^-1, direct root children q0*q_child; leave
   RIFKE positions/velocities unchanged. Compare quaternion behavior to source.
5. Expand rest positions with identity 6D rotation and zero velocity. Extract
   first frame (remove it from motion) or rest condition. Normalize both with
   source arithmetic; pad motion after normalization with zeros. Preserve source
   promotion caused by default float64 padding. Record unpadded valid length.
6. Parent condition features copy normalized parent rows. Preserve augmented
   topology, offsets, embeddings, statistics, crop start and original clip ID.
7. Collate variable skeletons to configured joint/time limits. Output motion
   B,J,12,T float32; embedding-index matrices/depths int64. Padded std is one.
   Joint mask B,1,1,J and time mask B,1,1,T exclude padding. Spectral widths may
   vary. Caption tokens pad to batch maximum; caption-less rows retain one valid
   zero token to avoid all-masked attention. Match all optional-field rules.

## Portable boundaries

Use versioned dictionary/bytes values for text cache, encoded sample and batch;
no device tensors, live encoder, local path or pickle in persisted values.
Bind cache identity to encoder/tokenizer artifact identity and source text, and
sample identity to dataset/clip/statistics/cache/options. Validate finite plain
numeric arrays, shapes, topology, positive statistics and bounded decoded sizes
before allocation. Preserve bytes through Cloud Offload. Do not silently accept
legacy object metadata; any upstream cache conversion is an explicit trusted
setup boundary. Missing captions are rejected for source-compatible training;
caption-free inference/evaluation is a separate explicit mode.

## Components and verification

Prefer separate numeric sample transforms, portable contracts, encoder/cache
production, batch collation and thin V3 nodes. Keep source math in owning modules.
Comparison fixtures compile unchanged pinned bodies with reference-only Motion.
Test cropping modes/short tails, nonidentity facing, all dtypes, normalized rest,
mask/std padding, differing spectral widths, mixed captions and global RNG
isolation. Test malformed values, budgets and cancellation. Then actual installed
encoder comparisons and direct V3 plus runner/client portable round trips on
Windows and headless stadia. Record exact runtime identities. Training runtime,
all model families/text encoder types and provider/container gates remain required.
