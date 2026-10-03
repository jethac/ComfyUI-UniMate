# Preserve partition execution lists

Goal: preserve every batch case across local/remote boundaries, including mapped
nodes. Preserve nested list-valued data, scalar behavior and legacy readability.

Design: bundles carry a reserved `comfy.partition.execution.v1` envelope with the
whole ComfyUI execution list in `values`. Gateway/output capture receive all
inputs once (`is_input_list`); input/extraction expose list outputs. Control fields
must contain one value. Python lists remain individual entries. Legacy unwrapped
bundles restore as one value, even when that value is a list.

Files: both partition_protocol.py copies own envelope validation; client
partition_nodes.py and runner __init__.py own bridge semantics. The worker stores
opaque bundles. tools/cloud_workflow.py unwraps its one-value fixtures while
retaining identity checks.

- [x] Reproduce repeated writes discarding earlier cases.
- [x] Add failing envelope, legacy, nested-list and bridge schema tests.
- [x] Implement codec helpers and collect/restore schemas in both bridges.
- [x] Test gateway control validation and wrapping without provider calls.
- [x] Run scalar worker regressions and real mapped-list server workflows.
- [ ] Run multi-rig/repetition model workflows headlessly on stadia.
- [ ] Review, document evidence and push direct to default branches.

Deployment/scheduling, training and postprocessing remain separate goal gaps.
