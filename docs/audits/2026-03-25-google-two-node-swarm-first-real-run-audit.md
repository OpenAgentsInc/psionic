# Google Two-Node Swarm First Real Run Audit

> Status: written 2026-03-25 after executing GitHub issue `#508` against real
> `openagentsgemini` Google Compute Engine nodes and validating the retained
> evidence bundles in GCS.

## Question

Did the first real Google two-node configured-peer swarm lane reach
`bounded_success` from the repo-owned operator surface?

Yes.

On March 25, 2026 UTC, the lane reached `bounded_success` on:

- one clean baseline run
- one admitted impaired-network rerun with the `mild_wan` profile

That is enough to close `#508`. It is also enough to close the master issue
`#501`.

The claim stays narrow:

- two private `g2-standard-8` plus `L4` nodes
- one configured-peer cluster
- one bounded `open_adapter_backend.cuda.gpt_oss_lm_head` adapter-delta window
- one coordinator-validator-aggregator-contributor node plus one contributor
  node
- no trusted-cluster full-model Google training claim
- no elastic membership, public discovery, or cross-region claim

## Exact Successful Runs

### Clean Baseline

- run id: `gswarm508-clean3-20260325000951`
- launch receipt timestamp: `2026-03-25T05:10:42Z`
- final manifest timestamp: `2026-03-25T05:21:28Z`
- selected impairment profile: `clean_baseline`
- selected zone pair: `us-central1-a__us-central1-b`
- nodes:
  - coordinator: `gswarm508-clean3-20260325000951-coord` in `us-central1-a`
    at `10.42.10.5:34100`
  - contributor: `gswarm508-clean3-20260325000951-contrib` in `us-central1-b`
    at `10.42.11.5:34101`
- final result classification: `bounded_success`
- evidence bundle:
  `gs://openagentsgemini-psion-train-us-central1/runs/gswarm508-clean3-20260325000951/final/psion_google_two_node_swarm_evidence_bundle.json`
- evidence bundle sha256:
  `f63ce2a62aa15afc02f55a8e8e53a33c3d182ae179428c0bd5234ab037ee7de1`
- checker verdict:
  - `submission_receipt_count=2`
  - `retained_object_count=6`
- coordinator runtime report:
  - `observed_wallclock_ms=9352`
  - validator summary present
  - promotion receipt present

### Mild-WAN Rerun

- run id: `gswarm508-mildwan-20260325002345`
- launch receipt timestamp: `2026-03-25T05:24:39Z`
- final manifest timestamp: `2026-03-25T05:35:59Z`
- selected impairment profile: `mild_wan`
- selected zone pair: `us-central1-a__us-central1-b`
- nodes:
  - coordinator: `gswarm508-mildwan-20260325002345-coord` in `us-central1-a`
    at `10.42.10.6:34100`
  - contributor: `gswarm508-mildwan-20260325002345-contrib` in `us-central1-b`
    at `10.42.11.6:34101`
- final result classification: `bounded_success`
- evidence bundle:
  `gs://openagentsgemini-psion-train-us-central1/runs/gswarm508-mildwan-20260325002345/final/psion_google_two_node_swarm_evidence_bundle.json`
- evidence bundle sha256:
  `cfc1795c371d7f4da136ba27b1196313077638485bfd4a26cdc88a0c64a11cb9`
- admitted impairment receipts retained for both hosts
- impairment parameters:
  - `delay_ms=45`
  - `jitter_ms=10`
  - `loss_percent=0.2`
  - `rate_mbit=600`
- checker verdict:
  - `submission_receipt_count=2`
  - `retained_object_count=8`
- coordinator runtime report:
  - `observed_wallclock_ms=9489`
  - validator summary present
  - promotion receipt present

## Failed Attempts And Fixes

Two earlier real attempts failed before the lane reached a truthful closeout.
Those failures mattered because they forced the runtime policy to match real
Google execution timing instead of the shorter local assumptions.

### Failed Attempt 1

- run id: `gswarm508-clean-20260325041122`
- launch receipt timestamp: `2026-03-25T04:13:43Z`
- retained artifacts:
  - both bring-up reports
  - contributor runtime report
  - no coordinator runtime report

Live startup-log inspection on the coordinator showed the run died during local
window validation with:

`Validation(WindowContract(UploadRequired { ... }))`

That exposed a stale freshness policy for the bounded Google lane. The fix was
to widen the adapter worker protocol freshness budget on this lane to:

- `heartbeat_timeout_ms=60_000`
- `claim_ttl_ms=300_000`

That change kept the real Google coordinator path from invalidating its own
local submission before the validator or aggregator path sealed the window.

### Failed Attempt 2

- run id: `gswarm508-clean2-20260324234816`
- launch receipt timestamp: `2026-03-25T04:49:09Z`
- retained artifacts:
  - both bring-up reports
  - no runtime reports

Live startup-log inspection showed the contributor timed out after 90 seconds
trying to connect to `10.42.10.4:34100`, while the coordinator was still
compiling the swarm binary and only started listening later.

That exposed a second real-world timing mismatch. The fix was to widen the
contributor peer-connect retry budget to 600 seconds and document the expected
cold-build skew in `docs/PSION_GOOGLE_TWO_NODE_SWARM_RUNBOOK.md`.

## What The Evidence Proves

The successful runs prove all of the things the issue stack claimed and nothing
more:

- the repo-owned Google launch surface can boot two private nodes in distinct
  admitted zones and subnets under one configured-peer manifest
- both nodes can emit machine-legible bring-up reports before the bounded
  runtime begins
- the contributor can dial the coordinator over the private cluster port and
  return one adapter-delta contribution
- the coordinator can retain two submission receipts, one validator summary,
  and one promotion receipt for the bounded window
- the finalizer can bind launch artifacts, host reports, optional impairment
  receipts, and final disposition into one retained cluster-wide evidence bundle
- the `mild_wan` impairment profile can be applied and retained on both hosts
  without breaking the bounded window

The successful runs do not prove:

- trusted-cluster full-model training on Google
- wider-network discovery
- elastic membership
- public swarm compute
- cross-region execution

## Repo Readiness After This Run

The Google two-node configured-peer swarm path is now implemented and proven at
the bounded runbook level.

What is complete:

- issue stack `#502` through `#508`
- one clean baseline `bounded_success`
- one impaired rerun `bounded_success`
- one repo-owned runbook, checker, finalizer, and evidence path

What still remains outside this claim:

- wider Google multinode topologies
- mixed-backend or mixed-hardware Google swarm math
- trusted-cluster full-model Google training

## Conclusion

Issue `#508` is done.

The first real Google two-node configured-peer swarm rehearsal reached
`bounded_success` on March 25, 2026 UTC under both clean and mildly impaired
network conditions. The truthful repo claim is now:

`psionic` can execute one bounded two-node configured-peer Google swarm
adapter-delta window on two private `g2` plus `L4` nodes, retain the full
machine-legible evidence bundle in GCS, and survive the admitted `mild_wan`
impairment profile.
