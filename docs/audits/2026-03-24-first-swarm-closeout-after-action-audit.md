# 2026-03-24 First Swarm Closeout After-Action Audit

This audit closes the current first local swarm lane:

- one Mac MLX Metal coordinator, validator, and aggregator
- one Linux RTX 4080 CUDA contributor
- one shared `gpt_oss.decoder_lm_head_lora` open-adapter contract
- one trusted-LAN topology with local-snapshot-only promotion posture

It answers five concrete questions:

- did the current lane produce mergeable outputs
- did the current lane publish a local snapshot
- what did the current lane actually prove
- what blocked the lane from a truthful published result
- what must exist before the next attempt

## Final Outcome

The current first swarm attempt ends with:

- merge outcome: `no_merge`
- publish outcome: `refused`
- promotion outcome: `no_promotion`

No local snapshot was published.

The exact publish path remains frozen, but unused:

- surface:
  `psionic-mlx-workflows::MlxWorkflowWorkspace::publish_bundle`
- target:
  `hugging_face_snapshot`
- expected local directory:
  `local_publish/openagents_swarm_local_open_adapter/first-swarm-local-snapshot`

That is the right result for the current artifact set. Anything stronger would
be fabricated.

## Artifacts Reviewed

The closeout is bound to these retained artifacts:

- `fixtures/swarm/first_swarm_run_contract_v1.json`
- `fixtures/swarm/reports/swarm_mac_mlx_bringup_v1.json`
- `fixtures/swarm/reports/swarm_linux_rtx4080_bringup_v1.json`
- `fixtures/swarm/first_swarm_live_workflow_plan_v1.json`
- `fixtures/swarm/first_swarm_trusted_lan_topology_contract_v1.json`
- `fixtures/swarm/reports/first_swarm_trusted_lan_failure_drills_v1.json`
- `fixtures/swarm/reports/first_swarm_trusted_lan_rehearsal_v1.json`
- `fixtures/swarm/reports/first_swarm_trusted_lan_evidence_bundle_v1.json`
- `fixtures/swarm/reports/first_swarm_trusted_lan_closeout_v1.json`

The closeout report is useful because it makes the end state machine-legible
instead of leaving the repo with a bundle and no explicit merge or publish
verdict.

## What This Lane Proved

The first swarm lane now proves these narrower facts:

- the repo can freeze one exact mixed-hardware decentralized open-adapter lane
  across Mac MLX Metal and Linux CUDA
- the repo can retain comparable contributor receipts and one shared workflow
  plan for both nodes
- the repo can freeze one exact trusted-LAN topology, failure-drill bundle, and
  runbook for the two-node pair
- the repo can retain one truthful rehearsal report that stops at `no_go`
  instead of pretending mixed-hardware execution is already earned
- the repo can retain one refused live-attempt evidence bundle without faking
  contributor execution, validator, replay, aggregation, or publication
  receipts
- the repo can now also close out the lane with one explicit `no_merge` and
  `publish_refused` report tied to the existing MLX workflow publish surface

That is useful progress. It is not the same thing as a successful two-node
mixed-hardware training run.

## What This Lane Did Not Prove

The current first swarm lane still does not prove:

- one real two-node contributor execution receipt set for the exact trusted-LAN
  lane
- accepted validator outcomes for both contributors
- replay receipts for accepted contributions
- one completed aggregation result under `aggregation.open_adapter.mean_delta`
- one promoted local snapshot
- one published local snapshot directory emitted by
  `psionic-mlx-workflows::MlxWorkflowWorkspace::publish_bundle`

This means the lane is still short of the original `SWARM-0` completion bar.

## Main Blockers

The primary blockers are direct:

- the rehearsal report still ends `no_go`
- the live-attempt evidence bundle is still a refused bundle
- contributor rows remain `not_executed` for validator, aggregation, and replay
- promotion remains `no_promotion`

The largest surprise is that publication is not the hard part here.

The local publish surface already exists and is narrow enough. The missing work
is still the live contributor, validator, replay, and aggregation receipt set.

## What Must Exist Before The Next Attempt

The next truthful attempt needs:

1. one real two-node contributor execution receipt set for the exact trusted-LAN
   topology
2. validator acceptance receipts for both contributors under the frozen swarm
   contract
3. replay receipts for each accepted contribution
4. one completed aggregation result and one promoted local snapshot
5. only then one local snapshot publish through the existing MLX workflow
   surface

That means the next attempt does not need a new publish subsystem.

It needs live mixed-hardware execution truth.

## Bottom Line

The current first swarm lane is now operator-solid as a bounded mixed-hardware
planning, bring-up, rehearsal, refused-attempt, and closeout substrate.

It is not yet a successful mixed-hardware training result.

The repo should keep `SWARM-0` open until one later attempt earns real
two-node contributor execution, accepted mergeable outputs, promotion, and a
truthful local snapshot publication.
