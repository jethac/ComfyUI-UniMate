# Dataset and statistics nodes

Expose the numeric foundations through six V3 nodes: Build Dataset, Load Dataset,
Save Dataset, Dataset Statistics, Load Statistics and Save Statistics. Dataset
archives use `.unimatedata`; statistics archives use `.unimatestats`. Managed
input/output containment, fingerprinting, explicit cloud input declarations and
atomic publication follow the existing motion nodes. Cancellation is mandatory.

Build Dataset consumes paired Rig/Motion execution lists. It collects them into
one shard; lengths must match. Each pair passes exact existing rig/motion
validation, including legacy container identity equivalence. Original conditioning
and feature bytes are preserved. Labels input is a single JSON array: empty uses
default per-clip labels; otherwise its length matches pairs and entries may set
only id/dataset_type/object_type/caption/split. Defaults are `clip<N>`, selected
dataset type, rig asset basename without extension, motion prompt or empty text,
and train. Reject duplicate IDs and unknown metadata fields. Origin comes from
the source motion, defaulting to zero. Topologies and features deduplicate by
digest; clips retain canonical source rig IDs. Shared topologies retain a source
rig ID only when all their source rigs agree; otherwise that field is null.

Build only accepts existing mesh-adapter rigs. Load Dataset independently accepts
the broader numeric shard contract. This distinction must remain explicit.
Training rest offsets are derived from rest positions as released; preserving the
full conditioning here does not imply its raw offsets field is already suitable
for training.

Statistics selects only train clips, calls the verified numeric adapter with
per_dataset/balanced/tie_std Boolean options, and outputs `UNIMATE_STATISTICS`
plus a JSON provenance report. The value schema `unimate.statistics.v1` contains
dataset_sha256, clip_count, datasets, options, upstream_revision, arrays, sha256.
Dataset identity hashes the canonical shard archive-manifest envelope. Dataset
labels index rows in four float64 `(D,12)` arrays: mean_root/std_root/mean_local/
std_local. Standard deviations are positive; all arrays finite. Consolidated
arrays avoid a per-label NPZ-member limit. Validate exact fields, unique bounded
labels, options, digests, revision and row counts. Preserve array payload bytes.

Statistics archives are numeric NPZs with scalar Unicode schema
`unimate.statistics.file.v1`, UTF-8 manifest bytes and original array-payload bytes
in uint8 vectors. They never load pickle. Save nodes return their original values
and core `files` descriptors; files publish atomically after cancellation checks.
Statistics payloads are limited to 8 MiB and persisted archives to 16 MiB,
including declared ZIP expansion, before numeric decoding. These budgets cover
all 4096 supported labels with four float64 normalization arrays and metadata.

Cloud boundaries use concrete custom types and existing dictionary/bytes codec.
Verify actual codec round trips, staging declarations and output descriptors;
then execute local and stadia partition-handler workflows, capture/restore values,
reload saved artifacts and compare statistics against direct numeric results.
No paid provider is provisioned. Large collection training, splits/augmentations,
raw upstream dataset conversion and annotation remain separate required gates.
