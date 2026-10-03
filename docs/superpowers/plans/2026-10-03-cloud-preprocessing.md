# Cloud preprocessing verification

Execute inline with the executing-plans skill.

**Goal:** verify preprocessing, archive loading, FK/RIC recovery and rendering through
the actual Cloud Offload partition handler, artifact storage and client restoration.

**Design:** extend the isolated ComfyUI verifier with an explicit worker mode. The
worker stages declared GLB/NPZ inputs from digest-keyed storage, submits real workflows
through its executor and publishes typed boundary artifacts. A second job restores and
consumes those values, re-prepares a canonical asset, extracts an exported animation and
retrieves images and file outputs through the real client restoration function. No
provider provisioning or paid resources are required.

- [x] Reproduce lost image outputs at worker/client boundaries with failing tests.
- [x] Preserve image descriptors in worker partition results and restore them under
  ComfyUI's `images` UI key; existing path containment rules remain in force.
- [x] Add worker-mode harness and unit checks for its manifest/type coverage.
- [x] Run real Windows and headless stadia partition jobs, verify output values,
  restored files and independent animation playback.
- [x] Review the cross-repository changes, push direct to default branches and record
  exact revisions/runtime evidence. Keep deployment/scheduling gates distinct.
