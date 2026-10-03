# Dataset statistics

Implement the released dataset normalization as a reusable numeric adapter,
then expose it through the forthcoming dataset nodes. This is one component of
the training/dataset audit, not a substitute for its remaining capabilities.

Use `2c5b384715aa63d8639b1ed7eb74bfe614570c7a` as the reference. Inputs are
unnormalized canonical `(T,J,12)` feature arrays, dataset labels and object-type
labels for training clips only. Reject empty collections, malformed labels,
nonfinite/nonfloating features, fewer than two joints, invalid options and
oversized allocations. Require ordinary NumPy arrays; masked arrays and other
subclasses are rejected. Floating element widths above 64 bits are unsupported.
Never alter inputs. Cancellation is checked while
validating and accumulating clips.

Support per-dataset/global pools, frame-weighted/object-balanced moments and
channel-group standard-deviation tying. These are three independent Boolean
options. Split root and local channels; floor standard deviations at 1e-8.
Global balanced pooling uses object names alone, as released. Global results
are copied to each dataset label. Tying averages standard deviations over
position, rotation and velocity channel groups, separately for root/local.
Retain input precision during per-clip reductions, then accumulate in float64,
as released. Upcasting float32 clips before reduction changes the statistics.

Total input arrays are capped at 256 MiB. The 512 MiB workspace estimate includes
input bytes, three times the largest clip at float64 width, 16 width-12 float64
vectors per balanced group and four per output dataset. This is a conservative
numeric allocation estimate, not a measured peak or a process-memory limit.

Return a dictionary keyed by dataset label with float64 `(12,)` arrays named
`mean_root`, `std_root`, `mean_local`, `std_local`. Archive/socket contracts
will encode these numeric arrays without pickle. Do not change model bundle
normalization or existing inference defaults in this component.

Reference tests execute the original statistics method bodies from the pinned
checkout, avoiding training imports and their dependencies. Cover every option
combination, heterogeneous lengths/joint counts, shared object names across
datasets, constant channels, root/local separation and mutation/cancellation.
Require the reference pin and source digest to match before comparisons.

Alternatives considered: importing the full training dataset brings encoder and
legacy pickle dependencies into ComfyUI; adapting only one normalization mode
leaves released configurations uncovered. A dependency-light numeric adapter
supports all modes while keeping training runtime isolation intact.

Remaining integration: safe dataset contracts and build/load/save nodes;
Statistics node and portable output; training runtime consumption; actual local
and Cloud Offload workflows. Keep those gates open until implemented and tested.
