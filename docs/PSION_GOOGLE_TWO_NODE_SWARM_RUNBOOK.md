# Psion Google Two-Node Swarm Runbook

> Status: canonical bounded runbook for the first Google two-node configured-peer
> swarm lane, added 2026-03-25 after landing the contract, preflight, launch,
> impairment, runtime, bring-up, evidence, and finalizer surfaces.

This is the exact operator guide for the first Google two-node configured-peer
swarm lane:

- project: `openagentsgemini`
- region family: `us-central1`
- two private `g2-standard-8` plus `L4` nodes in distinct admitted zones
- one configured-peer cluster namespace and no public discovery posture
- one coordinator-validator-aggregator-contributor node
- one contributor node
- one bounded `open_adapter_backend.cuda.gpt_oss_lm_head` adapter-delta lane

This runbook is intentionally narrower than the broader cluster validation
runbooks. Use it when the goal is the exact first Google configured-peer swarm
lane and not a broader cluster claim.

## What This Runbook Freezes

- contract:
  `fixtures/psion/google/psion_google_two_node_swarm_contract_v1.json`
- launch authority:
  `fixtures/psion/google/psion_google_two_node_swarm_launch_profiles_v1.json`
- network posture:
  `fixtures/psion/google/psion_google_two_node_swarm_network_posture_v1.json`
- identity profile:
  `fixtures/psion/google/psion_google_two_node_swarm_identity_profile_v1.json`
- operator preflight policy:
  `fixtures/psion/google/psion_google_two_node_swarm_operator_preflight_policy_v1.json`
- impairment policy:
  `fixtures/psion/google/psion_google_two_node_swarm_impairment_policy_v1.json`
- contract checker:
  `scripts/check-psion-google-two-node-swarm-contract.sh`
- operator preflight:
  `scripts/psion-google-operator-preflight-two-node-swarm.sh`
- launcher:
  `scripts/psion-google-launch-two-node-swarm.sh`
- node startup:
  `scripts/psion-google-two-node-swarm-startup.sh`
- impairment helper:
  `scripts/psion-google-two-node-swarm-impair.sh`
- finalizer:
  `scripts/psion-google-finalize-two-node-swarm-run.sh`
- evidence-bundle checker:
  `scripts/check-psion-google-two-node-swarm-evidence-bundle.sh`
- teardown:
  `scripts/psion-google-delete-two-node-swarm.sh`

## What This Runbook Does Not Claim

- no trusted-cluster full-model Google training claim
- no cross-region claim
- no public or wider-network discovery claim
- no elastic-membership claim
- no production Google swarm claim

The truthful claim remains one bounded configured-peer two-node adapter-delta
lane inside one operator-managed Google environment.

## Frozen Artifact Layout

Every run uses one prefix:

`gs://openagentsgemini-psion-train-us-central1/runs/<run_id>/`

Required launch objects:

- `launch/psion_google_two_node_swarm_cluster_manifest.json`
- `launch/psion_google_two_node_swarm_launch_receipt.json`
- `launch/psion-google-two-node-swarm-startup.sh`
- `launch/psion_google_two_node_swarm_quota_preflight.json`

Required node objects:

- coordinator bring-up:
  `host/coordinator/psion_google_two_node_swarm_bringup_report.json`
- contributor bring-up:
  `host/contributor/psion_google_two_node_swarm_bringup_report.json`
- coordinator runtime:
  `host/coordinator/psion_google_two_node_swarm_runtime_report.json`
- contributor runtime:
  `host/contributor/psion_google_two_node_swarm_runtime_report.json`
- optional coordinator impairment receipt:
  `host/coordinator/psion_google_two_node_swarm_impairment_receipt.json`
- optional contributor impairment receipt:
  `host/contributor/psion_google_two_node_swarm_impairment_receipt.json`

Required final objects:

- `final/psion_google_two_node_swarm_evidence_bundle.json`
- `final/psion_google_two_node_swarm_final_manifest.json`

## First Command

Validate the exact lane contract before describing the lane as frozen:

```bash
scripts/check-psion-google-two-node-swarm-contract.sh
```

If this fails, stop.

## Operator Preconditions

Run the dual-node operator preflight:

```bash
scripts/psion-google-operator-preflight-two-node-swarm.sh
```

This must return `result=ready`.

Before spending money, the broader configured-peer cluster validation gates
should also stay green:

- `cargo test -p psionic-cluster --test local_cluster_transport authenticated_configured_peers_discover_each_other_with_signed_control_plane_messages`
- `cargo test -p psionic-cluster --test local_cluster_transport authenticated_nodes_can_boot_from_operator_manifest`
- `cargo test -p psionic-cluster --test local_cluster_transport unreachable_configured_peer_surfaces_explicit_health_and_backoff`
- `cargo test -p psionic-cluster --test local_cluster_transport late_joining_configured_peer_recovers_health_after_degraded_attempts`

## Clean Baseline Launch

Pick one explicit run id:

```bash
export RUN_ID=psion-google-swarm-$(date -u +%Y%m%dT%H%M%SZ)
```

Launch the clean baseline:

```bash
scripts/psion-google-launch-two-node-swarm.sh \
  --run-id "${RUN_ID}" \
  --impairment-profile clean_baseline
```

What this does:

- selects the first admitted zone pair that passes quota preflight
- launches both private nodes
- assigns coordinator versus contributor role explicitly
- writes the cluster manifest and launch receipt to the training bucket
- starts the bounded runtime through the node startup path

## Monitoring During The Run

Watch the coordinator startup log:

```bash
gcloud compute ssh "$(jq -r '.nodes[] | select(.role_kind == "coordinator_validator_aggregator_contributor") | .instance_name' <(gcloud storage cat "gs://openagentsgemini-psion-train-us-central1/runs/${RUN_ID}/launch/psion_google_two_node_swarm_cluster_manifest.json"))" \
  --project=openagentsgemini \
  --zone="$(jq -r '.nodes[] | select(.role_kind == "coordinator_validator_aggregator_contributor") | .zone' <(gcloud storage cat "gs://openagentsgemini-psion-train-us-central1/runs/${RUN_ID}/launch/psion_google_two_node_swarm_cluster_manifest.json"))" \
  --tunnel-through-iap \
  --command='sudo tail -f /var/log/psion-google-two-node-swarm-startup.log'
```

Watch the contributor startup log:

```bash
gcloud compute ssh "$(jq -r '.nodes[] | select(.role_kind == "contributor") | .instance_name' <(gcloud storage cat "gs://openagentsgemini-psion-train-us-central1/runs/${RUN_ID}/launch/psion_google_two_node_swarm_cluster_manifest.json"))" \
  --project=openagentsgemini \
  --zone="$(jq -r '.nodes[] | select(.role_kind == "contributor") | .zone' <(gcloud storage cat "gs://openagentsgemini-psion-train-us-central1/runs/${RUN_ID}/launch/psion_google_two_node_swarm_cluster_manifest.json"))" \
  --tunnel-through-iap \
  --command='sudo tail -f /var/log/psion-google-two-node-swarm-startup.log'
```

Minimum healthy artifacts before finalization:

- both bring-up reports exist
- both runtime reports exist
- both runtime reports carry the expected runtime roles
- the coordinator runtime report retains two submission receipts

On a cold machine image, the two nodes can spend several minutes compiling the
bounded swarm binary before the coordinator starts listening. The contributor
runtime now retries the peer endpoint for up to ten minutes to absorb that
startup skew. Do not treat a missing runtime report as a hard failure until
that window expires or the startup log shows an explicit runtime error.

## Impaired Rerun

After the clean baseline, run at least one admitted impaired profile. Start with
`mild_wan`.

Launch the impaired rerun:

```bash
scripts/psion-google-launch-two-node-swarm.sh \
  --run-id "${RUN_ID}" \
  --impairment-profile mild_wan
```

Apply the impairment on the coordinator node and upload the receipt:

```bash
gcloud compute ssh "<coordinator-instance>" \
  --project=openagentsgemini \
  --zone="<coordinator-zone>" \
  --tunnel-through-iap \
  --command='cd /var/lib/psion-google-two-node-swarm/runs/'"${RUN_ID}"'/coordinator/repo && sudo scripts/psion-google-two-node-swarm-impair.sh --action apply --profile mild_wan --host-role coordinator --run-id "'"${RUN_ID}"'" --interface ens4 --receipt-out /tmp/psion_google_two_node_swarm_impairment_receipt.json && gcloud storage cp /tmp/psion_google_two_node_swarm_impairment_receipt.json gs://openagentsgemini-psion-train-us-central1/runs/'"${RUN_ID}"'/host/coordinator/psion_google_two_node_swarm_impairment_receipt.json'
```

Apply the impairment on the contributor node and upload the receipt:

```bash
gcloud compute ssh "<contributor-instance>" \
  --project=openagentsgemini \
  --zone="<contributor-zone>" \
  --tunnel-through-iap \
  --command='cd /var/lib/psion-google-two-node-swarm/runs/'"${RUN_ID}"'/contributor/repo && sudo scripts/psion-google-two-node-swarm-impair.sh --action apply --profile mild_wan --host-role contributor --run-id "'"${RUN_ID}"'" --interface ens4 --receipt-out /tmp/psion_google_two_node_swarm_impairment_receipt.json && gcloud storage cp /tmp/psion_google_two_node_swarm_impairment_receipt.json gs://openagentsgemini-psion-train-us-central1/runs/'"${RUN_ID}"'/host/contributor/psion_google_two_node_swarm_impairment_receipt.json'
```

Clear the impairment after the bounded drill:

```bash
gcloud compute ssh "<coordinator-instance>" \
  --project=openagentsgemini \
  --zone="<coordinator-zone>" \
  --tunnel-through-iap \
  --command='cd /var/lib/psion-google-two-node-swarm/runs/'"${RUN_ID}"'/coordinator/repo && sudo scripts/psion-google-two-node-swarm-impair.sh --action clear --profile mild_wan --host-role coordinator --run-id "'"${RUN_ID}"'" --interface ens4'

gcloud compute ssh "<contributor-instance>" \
  --project=openagentsgemini \
  --zone="<contributor-zone>" \
  --tunnel-through-iap \
  --command='cd /var/lib/psion-google-two-node-swarm/runs/'"${RUN_ID}"'/contributor/repo && sudo scripts/psion-google-two-node-swarm-impair.sh --action clear --profile mild_wan --host-role contributor --run-id "'"${RUN_ID}"'" --interface ens4'
```

Use `temporary_partition` only for a short explicit drill. Apply it briefly and
clear it. Do not leave the lane partitioned and then describe the resulting
receipts as a healthy bounded success.

## Finalization

Finalize the run from the operator machine:

```bash
scripts/psion-google-finalize-two-node-swarm-run.sh \
  --run-id "${RUN_ID}" \
  --coordinator-impairment-receipt "gs://openagentsgemini-psion-train-us-central1/runs/${RUN_ID}/host/coordinator/psion_google_two_node_swarm_impairment_receipt.json" \
  --contributor-impairment-receipt "gs://openagentsgemini-psion-train-us-central1/runs/${RUN_ID}/host/contributor/psion_google_two_node_swarm_impairment_receipt.json"
```

Validate the uploaded evidence bundle:

```bash
scripts/check-psion-google-two-node-swarm-evidence-bundle.sh --run-id "${RUN_ID}"
```

If this fails, do not describe the run as truthful.

## Result Classifications

The finalizer only admits these result classes:

- `configured_peer_launch_failure`
- `cluster_membership_failure`
- `network_impairment_gate_failure`
- `contributor_execution_failure`
- `validator_refusal`
- `aggregation_failure`
- `bounded_success`

Interpretation:

- `configured_peer_launch_failure`:
  one or both nodes never reached a truthful ready bring-up state
- `cluster_membership_failure`:
  the retained runtime reports did not prove the admitted coordinator and
  contributor role split
- `network_impairment_gate_failure`:
  the selected impairment profile and the retained impairment evidence drifted
- `contributor_execution_failure`:
  the contributor did not retain a full local contribution path
- `validator_refusal`:
  the coordinator did not retain validator posture
- `aggregation_failure`:
  the coordinator did not retain aggregation posture
- `bounded_success`:
  launch truth, bring-up truth, impairment truth when required, contribution
  flow, validator posture, and aggregation posture all stayed intact

## Teardown

Delete both VMs only after the final manifest exists:

```bash
scripts/psion-google-delete-two-node-swarm.sh --run-id "${RUN_ID}"
```

Use `--force` only when a failed run still needs cleanup and the final manifest
does not exist yet.
