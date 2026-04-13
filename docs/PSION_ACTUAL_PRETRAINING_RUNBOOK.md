# Psion Actual Pretraining Runbook

Status: canonical operator runbook for the explicit actual `Psion`
pretraining lane, written 2026-04-02 when the repo landed the first real
start, dry-run, resume, and status command for `psion_actual_pretraining_v1`.

## What This Runbook Is For

This runbook exists so the repo has one explicit operator path for the actual
broader-pretraining lane, and it now defines the default meaning of `./TRAIN`.

The commands are:

```bash
./TRAIN [start] [options]
./TRAIN record-checkpoint --run-root <path> --checkpoint-label <label> --optimizer-step <step> --checkpoint-ref <ref> [options]
./TRAIN backup --run-root <path> [options]
./TRAIN decide-continue-restart --run-root <path> [options]
./TRAIN rehearse-base-lane [options]
./TRAIN resume --run-root <path> [options]
./TRAIN status --run-root <path>
./TRAIN dashboard --run-root <path>
```

The older reference pilot is still available explicitly:

```bash
./TRAIN --lane reference_pilot [reference-lane options]
```

For machine supervision, the stable process boundary is now the repo-local
`psionic-train` binary rather than the shell wrapper:

```bash
cargo run -q -p psionic-train --bin psionic-train -- manifest --manifest <path-to-psionic.train.invocation_manifest.v1.json>
```

For assigned strong-node execution, the actual lane now also has one typed
packaging surface in
`crates/psionic-train/src/psion_actual_pretraining_launcher.rs`:

- `psion.actual_pretraining_automatic_execution_request.v1`
- `psion.actual_pretraining_automatic_execution_outputs.v1`

That request is the assignment-shaped wrapper around the same machine runtime.
It compiles one admitted strong-node turn into the generic
`psionic.train.invocation_manifest.v1` contract with fixed
`lane_id = psion_actual_pretraining_v1`,
`work_class = full_island_local_update_training`, and the fixed admitted
actual-lane `release_id` and `environment_ref`. The caller supplies the
admitted `build_digest`, `run_id`, coordination envelope, selected git ref,
launch or retained-state root, and any resume or checkpoint-specific fields
already declared by the generic runtime surface.

The paired output plan names the deterministic retained surfaces the node will
materialize from that same request: the machine run/window status packets,
membership revision receipt, checkpoint surface, actual-lane status and
summary files, the operation manifest path for `start` or `resume`, and the
window contribution bundle under `windows/<window_id>/...` when the admitted
coordination envelope already binds one `window_id`, `assignment_id`, and
`node_pubkey`. That keeps strong-node launch automation on the same manifest,
status, checkpoint, and sealed-window contracts already used by the weak-device
lane instead of creating a second runtime path.

That machine path consumes one explicit JSON manifest and emits one final
`psionic.train.status_packet.v1` packet with a stable exit code, retryability
bit, authority owner, refusal class when applicable, retained artifact paths,
the resolved runtime attestation, and the retained absolute paths for the
machine-readable run/window status packets. The invocation manifest now also
includes:

- one shared coordination envelope for `network_id`, `window_id`,
  `assignment_id`, `challenge_id`, `node_pubkey`, and `membership_revision`
- one required admitted `node_pubkey` for the local membership contract
- one admitted `release_id`
- one admitted `build_digest`
- one admitted `environment_ref`
- one optional `peer_node_pubkey` for recovery-source `serve-checkpoint`
- one optional `peer_checkpoint_handoff_receipt` artifact binding for joiner
  `resume`
- one optional `validator_target_contribution_receipt` artifact binding for
  validator `validate-contribution`
- one optional `validator_target_contribution_artifact_manifest` artifact
  binding for validator `validate-contribution`
- one optional `grouped_stage_input_transport` artifact binding for grouped
  stage execution

Each artifact binding now carries one stable `artifact_ref` tuple
(`artifact_id`, optional digest, optional byte count) plus one optional local
`materialized_path`. The logical artifact reference is the signed identity used
in invocation-manifest, contribution-receipt, and checkpoint-handoff digests.
The `materialized_path` is only one local execution binding for the current
machine. For resume handoff on the launch path, the runtime can now
re-materialize the outer handoff receipt plus its nested checkpoint pointer and
checkpoint manifest from the canonical local cache under
`<run-root>/artifacts/resolved/<sanitized-artifact-id>[.json]` when the caller
has already fetched those resolver-backed artifacts. Resume therefore no longer
requires SCP or hand placement of those inputs. Validator replay now follows
that same resolver-backed posture for the challenged contribution receipt,
contribution artifact manifest, and their nested replay inputs, so launch-path
replay no longer depends on manual artifact placement either.

The runtime refuses before launch when the executing release id, build digest,
or environment ref do not match the admitted identity in that manifest. When a
run root exists, it also persists:

- `status/psionic_train_run_status_packet.json`
- `status/psionic_train_window_status_packet.json`
- `status/membership_revision_receipt.json`
- `status/checkpoint_surface.json`

The retained membership receipt now freezes the first local cluster-session
contract too. It binds the admitted `node_pubkey`, release id, build digest,
environment ref, backend family, topology class, local membership revision,
heartbeat timestamps, stale and expiry thresholds, lease timers, and drain
deadline. It appends revision history under `status/membership_revisions/` and
automatically records same-node rejoin, different-node replacement, and failed
session posture from retained state instead of relying on manual metadata
fixups.

The retained checkpoint surface is the machine-readable checkpoint summary for
that same run root. It points at the latest accepted checkpoint pointer,
checkpoint manifest, backup receipt, backup copies, peer handoff receipt, and
auto-resume receipt when present. It also records the current checkpoint
phase, pointer state, checkpoint label and step, manifest digest, object
digest, byte count, backup state, upload outcome, and auto-resume recovery
result so supervisors can read the latest checkpoint posture without reopening
the full retained artifact family themselves.

For the strong actual-pretraining lane specifically, automatic `resume` can now
keep the request and output plan resolver-safe. The request may name one
`peer_checkpoint_handoff_receipt` by logical artifact id alone, while the
compiled machine manifest still carries the same artifact binding that the
runtime rematerializes from the local resolver cache before continuing. The
output plan then points at the retained `checkpoints/auto_resume_receipt.json`
surface under the same run root, which is the handoff from launch-time
materialization back into the normal checkpoint surface and status-packet path.

When the coordination envelope declares `window_id`, `assignment_id`, and the
admitted `node_pubkey`, the machine path also retains one window contribution
artifact family under `windows/<window_id>/`. That family now includes:

- `window_execution.json`
- `contributions/<contribution_id>/artifact_manifest.json`
- `contributions/<contribution_id>/contribution_receipt.json`
- `sealed_window_bundle.json`

`window_execution.json` binds the deterministic window execution id, current
assignment materialization, admitted role, runtime build digest, and capability
projection for that local runtime turn. The contribution artifact manifest then
hashes the concrete retained inputs that formed the local contribution surface:
the invocation manifest, launch manifest when it exists, membership receipt,
checkpoint surface and pointers, backup or peer handoff receipts, recovery
receipt, current status, retained summary, launcher log, and closeout bundle
when present. The contribution receipt binds that artifact-manifest digest to
the local outcome, refusal class when applicable, authority owner, retryability
bit, and stable contribution id.

`sealed_window_bundle.json` is the first local sealed-window rollup contract in
the machine runtime. It scans the retained contribution receipts already stored
for the same `window_id`, orders them deterministically by assignment and
contribution id, and emits one count-and-digest summary over the current local
receipt set. Contribution receipts and artifact manifests now canonicalize away
their local `materialized_path` fields before computing signed digests, so the
same retained artifact family can move between machines without changing its
logical identity. The machine-readable run/window status packets still repeat
the absolute paths for `window_execution_path`, `contribution_receipt_path`,
`contribution_artifact_manifest_path`, and `sealed_window_bundle_path` whenever
those retained window artifacts exist.

Validator replay is now admitted on that same machine surface too. One
validator manifest with `role = validator` and
`operation = validate_contribution` consumes one retained contribution receipt
plus one retained contribution artifact manifest, binds them to the declared
`window_id`, `assignment_id`, and `challenge_id`, and emits:

- `windows/<window_id>/validators/<challenge_id>/validator_score_artifact.json`
- `windows/<window_id>/validators/<challenge_id>/validator_score_receipt.json`
- `windows/<window_id>/validators/<challenge_id>/validator_quality_drift_signal.json`
- `windows/<window_id>/validators/<challenge_id>/validator_rollback_signal.json`
- `windows/<window_id>/validators/<challenge_id>/weak_device_validation_replay_proof.json`
  when an Apple / Metal weak-device `validation_replay` challenge returns an
  accepted `10_000` bps result with non-regressed quality drift and rollback
  posture `hold`

Those validator-target bindings may now carry logical artifact references
without one pre-staged `materialized_path`. Replay first checks any declared
local path, then falls back to the canonical resolver cache under
`<run-root>/artifacts/resolved/<sanitized-artifact-id>[.json]`. When the cache
contains the challenged receipt, artifact manifest, or nested checkpoint and
evidence artifacts, the validator rematerializes the retained contribution
family back into the local run root before bounded replay continues. That keeps
weak-device proof continuity and contributor-family path expectations intact
without reintroducing SCP or operator handoff steps. Missing cache entries stay
machine-boundary refusals and now report the expected resolver-cache location.

The current validator replay is still deliberately bounded. It does not claim a
full independent model rerun. It replays the retained contribution artifact
family, rechecks canonical contribution and artifact-manifest digests against
their logical artifact references, loads the retained checkpoint surface when
the challenged contribution succeeded, and then emits a typed `accepted`,
`quarantined`, `rejected`, or `replay_required` validator disposition. Missing
checkpoint state, stale assignment binding, or artifact drift stay
refusal-class failures at the machine process boundary. The run/window status
packets now repeat the retained `validator_score_receipt_path` when validator
replay completes successfully, and the validator artifact surface carries
`weak_device_validation_replay_proof_path` whenever that narrow accepted weak-
device replay proof is materialized. That proof only counts
validator-recognized weak-device participation. It does not by itself claim
checkpoint promotion, payout closeout, or network-wide finality.
That machine contract is now locked by focused validator-classification unit
tests plus subprocess CLI coverage for stale assignment, resolver-backed
replay-input rematerialization, missing replay-input, and artifact-digest
drift refusal paths.

The machine path now also covers late-join and recovery-source handoff
explicitly. One recovery-source manifest with `operation = serve_checkpoint`
and a target `peer_node_pubkey` can retain
`status/peer_checkpoint_handoff_receipt.json` from the latest accepted primary
checkpoint or from the durable backup family when the primary pointer is
missing. A joiner `resume` manifest can then point
`peer_checkpoint_handoff_receipt` at that retained receipt through one artifact
binding. The retained handoff receipt now carries logical artifact references
for the served checkpoint pointer and checkpoint manifest, while its local
source path remains only an operator diagnostic. When the caller stages the
resolver-backed handoff family into `<run-root>/artifacts/resolved/`, the
resume path copies the checkpoint pointer and manifest into the local run root
before the normal preflight validation emits the ordinary
`auto_resume_receipt.json`.

`./TRAIN` remains the operator convenience path above the same actual lane
logic.

## Current Claim Boundary

The actual-lane command now does these things for real:

- loads the frozen actual-lane spec, recipe bundle, scaling bundle, data
  bundle, baseline-tools bundle, systems bundle, topology/storage bundle,
  evidence contract, and status surface contract directly from committed repo
  artifacts
- writes `preflight/hardware_qualification.json` before launch or resume
- writes `preflight/run_shape_qualification.json` before launch or resume
- refuses non-dry-run start or resume when either retained preflight receipt is
  not admitted
- refuses dirty launches by default unless `--allow-dirty-tree` is supplied
- resolves and retains the selected git ref plus exact commit SHA
- writes the canonical launch or resume manifest under the retained evidence
  family
- writes the canonical current-status and retained-summary files
- writes the canonical latest-checkpoint pointer file
- writes accepted checkpoint manifests plus durable-backup receipts
- writes auto-resume receipts and retained stale/corrupt-pointer recovery drills
- writes automatic checkpoint-eval decisions on accepted checkpoints
- can retain checkpoint-eval retry receipts plus a redacted alert when the eval
  worker is unavailable
- writes one retained dashboard packet plus one retained aggregate active-alert
  feed
- can inject a failed-upload refusal drill without manual artifact editing
- writes one retained checkpoint-comparison receipt plus one retained
  continue-restart decision over the latest accepted checkpoint
- can run one end-to-end base-lane rehearsal that upgrades
  `closeout/closeout_bundle.json` from provisional provenance to a final
  closeout packet with proof gates, retained drill evidence, and explicit
  claim-boundary sections
- exposes the canonical status command
- exposes the canonical dashboard command

It does not yet claim:

- external alert delivery or paging
- a cluster-connected streaming dashboard
- completed broader actual-lane distributed cluster execution beyond one bounded retained rehearsal segment

Those come later in the roadmap. This launcher is the operator contract, not
the full hardening pass.

## Start Command

Dry-run materialization:

```bash
./TRAIN --dry-run
```

Dry-run with an admitted retained observation snapshot:

```bash
./TRAIN start \
  --dry-run \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json
```

Start with an explicit run id and output root:

```bash
./TRAIN start \
  --run-id run-psion-actual-20260402t120000z \
  --output-root ~/scratch/psion_actual_pretraining_runs/run-psion-actual-20260402t120000z
```

Default output root:

- `~/scratch/psion_actual_pretraining_runs/<run_id>`

The start path writes:

- `manifests/launch_manifest.json`
- `status/current_run_status.json`
- `status/retained_summary.json`
- `checkpoints/latest_accepted_checkpoint_pointer.json`
- `preflight/hardware_qualification.json`
- `preflight/run_shape_qualification.json`
- `closeout/closeout_bundle.json`
- `dashboard/current_dashboard.json`
- `alerts/active_alerts.json`
- `logs/launcher.log`

Non-dry-run `start` now refuses when either preflight receipt lands with
`admission_state = refused`.

Before the first accepted checkpoint exists, the retained state is explicit:

- phase: `dry_run_planned` or `launch_staged`
- latest checkpoint label: `pending_first_checkpoint`
- last completed step: `0`

## Record Checkpoint

Canonical accepted-checkpoint materialization:

```bash
./TRAIN record-checkpoint \
  --run-root <path> \
  --checkpoint-label broader-pretrain-final \
  --optimizer-step 16384 \
  --checkpoint-ref checkpoint://psion/broad/pretrain/final
```

This command promotes the run from `pending_first_checkpoint` into one accepted
checkpoint lineage. It writes:

- `checkpoints/step-<optimizer_step>/checkpoint_manifest.json`
- `checkpoints/latest_accepted_checkpoint_pointer.json`
- `checkpoints/latest_accepted_checkpoint_backup_receipt.json`
- `checkpoints/backups/latest_accepted_checkpoint_pointer.backup.json`
- `checkpoints/backups/step-<optimizer_step>/checkpoint_manifest.backup.json`
- `evals/checkpoint_eval_step-<optimizer_step>.json`
- `evals/latest_checkpoint_eval_decision.json`
- refreshed status, retained-summary, closeout, and launcher-log surfaces
- refreshed retained dashboard and active-alert feed

The default checkpoint byte count comes from the frozen systems bundle. The
default checkpoint object digest is a stable synthetic digest over the accepted
checkpoint identity unless the operator provides an explicit digest.

Unavailable-worker rehearsal:

```bash
./TRAIN record-checkpoint \
  --run-root <path> \
  --checkpoint-label broader-pretrain-final \
  --optimizer-step 16384 \
  --checkpoint-ref checkpoint://psion/broad/pretrain/final \
  --inject-eval-worker-unavailable
```

That path writes:

- `evals/checkpoint_eval_failure_step-<optimizer_step>.json`
- `evals/latest_checkpoint_eval_failure.json`
- `alerts/latest_redacted_alert.json`

It keeps the checkpoint itself admitted and backed up, but retains explicit
retry-required evidence instead of silently skipping automatic eval.

## Backup Command

Canonical durable-backup replay:

```bash
./TRAIN backup --run-root <path>
```

This command rereads the current accepted pointer and checkpoint manifest and
re-materializes the retained backup family plus
`checkpoints/latest_accepted_checkpoint_backup_receipt.json`.
It also refreshes the retained status, closeout, dashboard, and active-alert
surfaces so backup refusal or success becomes operator-visible without reading
raw receipts directly first.

Failure-injection rehearsal:

```bash
./TRAIN backup \
  --run-root <path> \
  --inject-failed-upload
```

That drill writes:

- a refused backup receipt
- `checkpoints/failures/failed_upload_drill.json`

It retains declared secret/config source names only. Raw SSH, bucket, or
service-account payloads are not copied into retained artifacts or logs.

## Decide Continue Or Restart

Canonical long-run decision:

```bash
./TRAIN decide-continue-restart --run-root <path>
```

This command consumes the latest accepted checkpoint pointer, backup receipt,
checkpoint-eval decision or failure, retained hardware qualification, retained
run-shape qualification, and the committed systems bundle. It writes:

- `decisions/checkpoint_comparison_step-<optimizer_step>.json`
- `decisions/latest_checkpoint_comparison.json`
- `decisions/continue_restart_decision_step-<optimizer_step>.json`
- `decisions/latest_continue_restart_decision.json`
- refreshed status, retained-summary, closeout, launcher-log, and dashboard
  surfaces

The bounded decision states are:

- `continue`
- `hold_and_investigate`
- `restart_from_last_accepted_checkpoint`

The continue threshold is intentionally stricter than raw admission:

- eval decision must stay `continue`
- durable backup must stay `backed_up`
- hardware and run-shape admission must stay `admitted`
- throughput must stay at or above `90%` of the trusted-cluster anchor
- step latency must stay at or below `115%` of the trusted-cluster anchor
- checkpoint write throughput must stay at or above `90%` of the trusted-cluster
  anchor
- dataloader stalls must stay at or below `1`

If those conditions fail, the actual lane retains `hold_and_investigate`
instead of guessing.

## Resume Command

Canonical resume:

```bash
./TRAIN resume --run-root <path>
```

For the launch path, that run root is also the canonical local staging root for
resolver-backed peer handoff artifacts:

- `<run-root>/artifacts/resolved/<sanitized-artifact-id>.json`
- `<run-root>/artifacts/resolved/<sanitized-artifact-id>`

Resume first reads:

- `<run-root>/checkpoints/latest_accepted_checkpoint_pointer.json`

If that primary pointer is stale or corrupt, resume falls back to:

- `<run-root>/checkpoints/latest_accepted_checkpoint_backup_receipt.json`
- `<run-root>/checkpoints/backups/latest_accepted_checkpoint_pointer.backup.json`
- `<run-root>/checkpoints/backups/step-<optimizer_step>/checkpoint_manifest.backup.json`

Resume refuses when neither the primary pointer nor the retained backup family
can produce an admitted accepted checkpoint. It also writes and consumes
`preflight/hardware_qualification.json` plus
`preflight/run_shape_qualification.json`, and non-dry-run resume refuses when
either receipt is not admitted. Every resume attempt now writes:

- `checkpoints/auto_resume_receipt.json`

Corrupt or stale primary-pointer recovery also writes:

- `checkpoints/failures/corrupt_pointer_drill.json`
- or `checkpoints/failures/stale_pointer_drill.json`

When resume succeeds, it writes:

- `manifests/resume_manifest.json`
- refreshed status and retained-summary files
- `continuation/accepted_checkpoint_handoff.json`
- refreshed closeout bundle
- appended `logs/launcher.log`

The continuation handoff binds the accepted checkpoint to the frozen
`general_sft -> agentic_sft` target and preserves the bounded plugin
benchmark-pack bindings already attached to that target. It does not claim that
the continuation stage has already run.

If auto-resume cannot select a valid checkpoint, the command still retains an
explicit `auto_resume_receipt.json` with `resolution_state = refused` and logs
`phase=resume_refused_auto_resume` instead of leaving the run root ambiguous.

## Base-Lane Rehearsal Command

Canonical base-lane proof gate:

```bash
./TRAIN rehearse-base-lane \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json
```

Tri-host bounded distributed rehearsal on the same actual-lane path:

```bash
./TRAIN rehearse-base-lane \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json \
  --remote-host archlinux \
  --secondary-remote-host macbook-pro-m2 \
  --cleanup-remote
```

Longer bounded tri-host rehearsal with the same actual workload and a larger
step budget:

```bash
PSION_REFERENCE_PILOT_MAX_STEPS=6 \
PSION_REFERENCE_PILOT_STEPS_PER_WINDOW=3 \
PSION_REFERENCE_PILOT_WINDOWS_PER_CADENCE=2 \
./TRAIN rehearse-base-lane \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json \
  --remote-host archlinux \
  --secondary-remote-host macbook-pro-m2 \
  --cleanup-remote
```

Production-candidate canary target on the same shipped path:

```bash
PSION_REFERENCE_PILOT_MAX_STEPS=12 \
PSION_REFERENCE_PILOT_STEPS_PER_WINDOW=3 \
PSION_REFERENCE_PILOT_WINDOWS_PER_CADENCE=2 \
./TRAIN rehearse-base-lane \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json \
  --remote-host archlinux \
  --secondary-remote-host macbook-pro-m2 \
  --cleanup-remote
```

Those environment variables only widen the bounded distributed bringup segment
that the operator delegates. They do not change the command surface or turn the
rehearsal into the full long-running production cluster lane. The current
production-candidate canary target is:

- `12` optimizer steps
- `3` steps per window
- `2` windows per cadence
- one shared tri-host optimizer path across the local Apple-silicon machine,
  the admitted M2 Mac, and the admitted Tailnet RTX `4080` host

The current clean source-of-truth run for that canary is:

- run id:
  `psion-actual-pretraining-tri-host-actual-prodcanary-optimized-20260413t101910Z`
- run root:
  `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-prodcanary-optimized-20260413t101910Z`
- topology: `multi_host_joint_gradient_average`
- contributor count: `3`
- optimizer steps: `12`
- contribution receipt count: `36`
- retained progress checkpoint count: `4`
- retained progress window count: `4`
- retained progress cadence count: `2`
- final cumulative train tokens processed: `2194`
- final cumulative mean tokens per second: `45`
- accepted checkpoint label: `bounded-actual-pretraining-bringup-step-12`
- full elapsed time from retained train-log timestamps: `1421s`
- distributed execution elapsed time from retained train-log timestamps:
  `1388s`

The previous clean `zstd` baseline run for the same `12`-step shape took
`2285s` full elapsed and `2243s` from distributed launch to remote cleanup.
The optimized canary cut both elapsed views by about `38%` on the same step
budget and admitted topology.

This command replays the actual operator path in one retained sequence:

- `start`
- one bounded distributed actual-pretraining bringup when `--remote-host` is supplied
- `record-checkpoint`
- `backup --inject-failed-upload`
- `backup`
- `decide-continue-restart`
- `resume`

It then upgrades `closeout/closeout_bundle.json` into the final base-lane
proof packet. That closeout bundle now carries:

- exact git/ref provenance
- explicit retained artifact refs for launch, checkpoint, backup, eval,
  continue-decision, resume, dashboard, and handoff truth
- one bounded distributed-bringup evidence family when a remote host is configured
- one retained failed-upload drill plus its recovered end state
- explicit closeout gates
- `can_now_claim` and `still_out_of_scope` sections

The final retained phase becomes `base_lane_rehearsal_complete`.

When `--remote-host` is present, the operator retains:

- `distributed_execution/distributed_actual_pretraining_bringup.json`
- the bounded actual-pretraining bringup cluster topology receipt
- the bounded actual-pretraining bringup contribution summary receipts
- the bounded actual-pretraining bringup contributor continuity receipt
- the bounded actual-pretraining bringup progress checkpoint receipts
- the bounded actual-pretraining bringup progress checkpoint directory
- the bounded actual-pretraining bringup checkpoint manifest and checkpoint weights

That is a real model-progress-bearing multi-host optimizer segment inside the
actual operator path using the larger actual-lane model id
`psion-compact-decoder-internal-v1` and the canonical dataset identity
`psion_corpus_tokenized@v1`. The claim boundary stays explicit: the runbook
proves one bounded rehearsal segment, not the full broader actual-pretraining
cluster lane.

The distributed bringup summary now also records:

- total retained progress checkpoint count
- total retained progress window count
- total retained progress cadence count
- final cumulative train tokens processed
- final cumulative mean tokens per second
- whether every configured contributor stayed in the optimizer path for every
  retained step

Two implementation details now matter for the clean production-candidate
canary:

- remote contributions are collected concurrently instead of waiting for the
  RTX `4080` host and the M2 host serially
- default batch-row sizing is backend-aware on the actual bringup path:
  local control plane `2`, primary CUDA remote `12`, secondary CPU remote `2`

Those shipped defaults can still be overridden through:

- `PSION_REFERENCE_PILOT_DUAL_HOST_CONTROL_BATCH_ROWS`
- `PSION_REFERENCE_PILOT_DUAL_HOST_REMOTE_BATCH_ROWS`
- `PSION_REFERENCE_PILOT_DUAL_HOST_SECONDARY_REMOTE_BATCH_ROWS`

The transport and staging path for this longer canary now stays on compressed
artifacts throughout the distributed segment:

- step exchange requests and responses are retained as `.json.zst`
- repo staging for remotes now uses `.tar.gz`
- SSH and SCP are run with compression enabled on the distributed bringup path

That change is operationally important. The earlier 12-step attempts completed
the optimizer work but stalled on oversized retained JSON receipts and large
plain transport payloads. The current clean canary is the first source-of-truth
run that completed end to end with compact retained receipts and compressed
transport on the shipped path.

The retained `final_cumulative_mean_tokens_per_second` field is still a bounded
internal training metric, not a wall-clock measurement over the whole operator
run. Use retained train-log timestamps or shell timing for honest elapsed-time
claims and treat the retained throughput field as an internal model-progress
signal only.

Residual risk for the next phase stays explicit:

- first-run remote compile time still dominates startup
- repo staging for a cold remote is still a noticeable part of run bringup
- this is still a bounded production-candidate canary, not the continuous live
  actual-lane execution program

The repo also now commits one clean example run root for this proof gate under:

- `fixtures/psion/pretrain/psion_actual_pretraining_base_lane_rehearsal_example/run-psion-actual-20260402t160000z/`

## Planned Interruption Recovery Drill

The same shipped `rehearse-base-lane` surface now also proves planned
checkpoint-boundary interruption and recovery without manual path surgery.

Source-of-truth recovery command:

```bash
PSION_REFERENCE_PILOT_MAX_STEPS=6 \
PSION_REFERENCE_PILOT_STEPS_PER_WINDOW=3 \
PSION_REFERENCE_PILOT_WINDOWS_PER_CADENCE=1 \
./TRAIN rehearse-base-lane \
  --run-id psion-actual-pretraining-tri-host-actual-recovery-20260413t154800Z \
  --hardware-observation fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json \
  --run-shape-observation fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json \
  --remote-host archlinux \
  --secondary-remote-host macbook-pro-m2 \
  --planned-interruption-step 3 \
  --cleanup-remote
```

This proof:

- runs the interrupted segment for `3` optimizer steps
- retains that interrupted checkpoint family under
  `distributed_actual_pretraining_bringup/interrupted_segment/...`
- resumes the recovered segment for the remaining `3` optimizer steps from the
  retained interrupted artifact family
- writes one retained recovery receipt at
  `distributed_actual_pretraining_bringup/actual_pretraining_bringup_recovery_drill.json`
- keeps one combined cluster topology, contribution, continuity, progress, and
  checkpoint family under the canonical
  `distributed_actual_pretraining_bringup/actual_pretraining_bringup_artifacts/`
  root

The current clean source-of-truth recovery run is:

- run id:
  `psion-actual-pretraining-tri-host-actual-recovery-20260413t154800Z`
- run root:
  `/Users/christopherdavid/scratch/psion_actual_pretraining_runs/psion-actual-pretraining-tri-host-actual-recovery-20260413t154800Z`
- topology: `multi_host_joint_gradient_average`
- contributor count: `3`
- interruption step: `3`
- total target optimizer steps: `6`
- recovered checkpoint ref: `psion-reference-pilot-step-6`
- interrupted checkpoint ref: `psion-reference-pilot-step-3`
- recovered checkpoint label: `bounded-actual-pretraining-bringup-step-6`
- lineage preserved: `true`
- contribution receipt count: `18`
- retained progress checkpoint count: `2`
- final cumulative train tokens processed: `394`
- final cumulative mean tokens per second: `16`

The retained recovery receipt is now the machine-readable answer to:

- what interruption was injected
- where the interrupted segment stopped
- whether the resumed run stayed on the same logical checkpoint lineage
- whether the continuation was honestly refused or completed

When recovery cannot proceed honestly, the same path writes a refused recovery
receipt with `recovery_state = refused` and one explicit `refusal_reason`
instead of silently forking the artifact family.

## Continuation-Handoff Rehearsal

The continuation proof gate intentionally stays separate from the base-lane
proof. The current repo-owned rehearsal is generated by:

```bash
cargo run -q -p psionic-train --example psion_actual_pretraining_continuation_handoff_rehearsal_fixtures
```

That rehearsal consumes the accepted checkpoint from the retained base-lane
closeout and writes:

- `fixtures/psion/pretrain/psion_actual_pretraining_continuation_handoff_rehearsal_bundle_v1.json`
- `fixtures/psion/pretrain/psion_actual_pretraining_continuation_handoff_refusal_packet_v1.json`
- `fixtures/psion/pretrain/psion_actual_pretraining_continuation_handoff_rehearsal_example/run-psion-actual-20260402t160000z/`

The rehearsal keeps exact run id, checkpoint lineage, plugin-conditioned stage
manifest identity, continuation-alignment evidence, and one mismatched
alignment refusal packet together without claiming cluster-scale continuation
execution or plugin-conditioned RL execution.

## Dashboard Command

```bash
./TRAIN dashboard --run-root <path>
```

This is a thin wrapper over
`scripts/psion-actual-pretraining-dashboard.sh`. It reads:

- `dashboard/current_dashboard.json`
- `alerts/active_alerts.json`

and prints:

- run id
- phase
- git provenance
- throughput posture
- loss and gradient visibility posture
- checkpoint backup and eval posture
- hardware health posture
- active alert count plus the highest severity
- one line per active alert when any alert is present

## Status Command

```bash
./TRAIN status --run-root <path>
```

This is a thin wrapper over `scripts/psion-actual-pretraining-status.sh`. It
prints:

- run id
- phase
- last completed step
- latest checkpoint label
- selected git ref
- git commit SHA
- dirty-tree admission posture
- status surface id
- latest checkpoint-eval decision and score when present
- latest checkpoint-eval failure and latest alert when present

## Dirty Trees And Provenance

Dirty working trees are refused by default.

If an operator deliberately overrides that rule:

```bash
./TRAIN start --allow-dirty-tree --dry-run
```

the launcher records:

- `dirty_tree_admission = allowed_by_operator_override`
- `workspace_status_sha256`

It still retains:

- `selected_git_ref`
- `git_commit_sha`

Those fields appear in the launch or resume manifest and repeat in the
closeout bundle.

## Related Docs

- `docs/PSION_ACTUAL_PRETRAINING_LANE.md`
- `docs/PSION_ACTUAL_PRETRAINING_RECIPE.md`
- `docs/PSION_ACTUAL_PRETRAINING_SCALING_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_BASELINE_TOOLS_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_DATA_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT.md`
- `docs/PSION_ACTUAL_PRETRAINING_CHECKPOINT_EVALS.md`
- `docs/PSION_ACTUAL_PRETRAINING_HARDWARE_QUALIFICATION.md`
- `docs/PSION_ACTUAL_PRETRAINING_RUN_SHAPE_QUALIFICATION.md`
- `docs/PSION_ACTUAL_PRETRAINING_STATUS_SURFACE.md`
- `docs/PSION_ACTUAL_PRETRAINING_SYSTEMS_BUNDLE.md`
- `docs/PSION_ACTUAL_PRETRAINING_CONTINUATION_HANDOFF.md`
- `docs/PSION_ACTUAL_PRETRAINING_CHECKPOINT_RECOVERY.md`
