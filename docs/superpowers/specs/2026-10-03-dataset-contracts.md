# Portable training dataset shards

The full training integration needs numeric datasets independent of mesh-import
restrictions. Implement a bounded shard value, then public build/load/save,
statistics and shard-combination workflows. Multiple shards must remain usable
by training; a per-shard bound is not a whole-dataset size limit.

`UNIMATE_DATASET` uses `unimate.dataset.v1` with exactly `schema`, `manifest` and
`files`. `files` maps SHA-256 digests to numeric NPZ bytes. The manifest contains
`topologies` and `clips`. Payloads are deduplicated and all files must be referenced.
No pickle, host paths, live arrays/models or embedded executable objects.

Topology entries contain `id`, `source_rig_id` and `conditioning`. The topology
ID equals its conditioning digest. `source_rig_id` is an optional prepared mesh
rig digest for provenance; it is not a substitute for checking the conditioning.
Required arrays: ordered connected `parents`, `offsets`, `tpos_first_frame`,
`joint_names`, `clean_joint_names`. Accept additional bounded numeric/Unicode
conditioning arrays unchanged for released topology/embedding fields. Require
2–4096 joints, integer parents with root -1 and parents before children, finite
floating `(J,3)` offsets/rest positions and Unicode `(J,)` names.

Clip entries contain `id`, `topology_id`, `dataset_type`, `object_type`, `caption`,
`split`, `features`, `origin`, `fps`. Features are an NPZ containing only
`features`, shaped `(T>=1,J,12)` with float16/32/64 dtype and finite values.
Feature joint count must match the selected topology. FPS is exactly 30. Origin
is a finite length-three JSON list. Split is `train` or `eval`. IDs are unique
portable labels within their respective lists; dataset/object labels are portable
names. Captions are bounded plain text, including punctuation and non-ASCII.
An optional `source_rig_id` records per-clip prepared-rig provenance. The builder
always includes it. Older shards without it remain valid. When multiple source
meshes share identical conditioning, their clips retain individual rig IDs and
the deduplicated topology's single-source provenance is null.

The archive uses `manifest.json` plus `arrays/<digest>.npz`, with a file schema
`unimate.dataset.file.v1`. Canonical JSON includes the socket manifest and file
digests; ZIP timestamps and creator OS are fixed. Digest failures, missing/extra
files, duplicate JSON keys/ZIP members, unsafe members, malformed NPZs and
unknown fields fail. No filesystem extraction is used.

Per shard: 256 MiB payload/expanded NPZ-member budget (including NPY headers),
4 MiB complete archive-manifest budget, at most
4096 clips/topologies and 8192 files. Check declarations before decoding and
aggregate expanded sizes while validating. Constructors copy manifest data;
consumers validate before using it. Bytes remain immutable. Caption text is
limited to 16 KiB UTF-8 per clip. Both topology and clip lists are nonempty.
Validation, construction, extraction and archive operations accept optional
cancellation callbacks, checked between payload/decode stages; exceptions
propagate unchanged. Extracted duplicate feature arrays share read-only storage.

Statistics extraction uses only `train` clips and rejects an empty training
split. It returns ordinary numeric records consumed by `dataset_stats`;
caption, origin and provenance remain in the shard. Evaluation data must never
enter normalization implicitly.

Public build nodes will accept existing rig/motion values and preserve their
strict identity check. Raw dataset imports will use the broader numeric topology
contract. Their adapters must prove canonical coordinates against source before
claiming full dataset preparation. Cloud Offload needs a registered portable
type, archive staging and original/restored output verification.

Alternatives: embedding mesh assets in every clip wastes transport space and
ties training to GLB restrictions; storing local dataset paths breaks worker
portability. Content-addressed numeric shards preserve independent training
topologies and allow staged collections.

Completion of the contract alone does not close dataset nodes, collection
training, augmentation, source-format preparation or annotation coverage.
