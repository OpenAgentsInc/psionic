# 2026-03-27 First Swarm Trusted-LAN Real Run Audit

This audit records the first truthful retained mixed-hardware swarm run for the
bounded `SWARM-0` lane:

- one Mac Apple Silicon node acting as coordinator, validator, aggregator, and
  MLX contributor
- one Linux desktop node with an RTX 4080 acting as the CUDA contributor
- one shared `gpt_oss.decoder_lm_head_lora` open-adapter contract
- one trusted-LAN cluster with no internet discovery or elastic-membership
  claim

It answers five concrete questions:

- did the repo finally execute a real two-node mixed-hardware run
- what exact evidence was retained
- what was actually validated during the run
- what still remains outside the claim boundary
- is `SWARM-0` now complete

## Final Verdict

Yes.

The retained real run completed with:

- run id: `first-swarm-live-20260327-real-2`
- result classification: `bounded_success`
- merge disposition: `merged`
- publish disposition: `refused`
- promotion disposition: `held`

That is the correct truthful closeout for this lane.

The run earned the two-node mixed-hardware execution proof that the older
historical closeout explicitly said was still missing. It did not publish a
model snapshot, and it does not need to in order to close `SWARM-0`, because
the original lane was always a bounded decentralized open-adapter proof rather
than a full served-model promotion lane. A separate retained publication proof
now exists for the frozen first-swarm local snapshot target, but that proof is
not the same thing as the live run earning publication.

## Artifacts Reviewed

The retained run is bound to these artifacts:

- `fixtures/swarm/runs/first-swarm-live-20260327-real-2/operator_manifest.json`
- `fixtures/swarm/runs/first-swarm-live-20260327-real-2/coordinator_runtime_report.json`
- `fixtures/swarm/runs/first-swarm-live-20260327-real-2/contributor_runtime_report.json`
- `fixtures/swarm/runs/first-swarm-live-20260327-real-2/first_swarm_real_run_bundle.json`
- `fixtures/swarm/first_swarm_trusted_lan_topology_contract_v1.json`
- `fixtures/swarm/first_swarm_live_workflow_plan_v1.json`
- `fixtures/swarm/reports/swarm_mac_mlx_bringup_v1.json`
- `fixtures/swarm/reports/swarm_linux_rtx4080_bringup_v1.json`
- `scripts/run-first-swarm-trusted-lan-live.sh`
- `scripts/check-first-swarm-trusted-lan-real-run.sh`

For historical contrast, this audit also keeps the older refusal artifacts in
scope:

- `fixtures/swarm/reports/first_swarm_trusted_lan_evidence_bundle_v1.json`
- `fixtures/swarm/reports/first_swarm_trusted_lan_closeout_v1.json`
- `docs/audits/2026-03-24-first-swarm-closeout-after-action-audit.md`

## What Actually Ran

The operator manifest froze these exact endpoints:

- coordinator: `192.168.1.122:34100`
- contributor: `192.168.1.189:34101`

The run used the frozen trusted-LAN topology and workflow plan, then executed
one bounded shared window:

- window id: `first-swarm-live-20260327-real-2-window-1`
- cluster id:
  `5d8174121f64fde406275a3e8a477fb9c0647791481393547f090c2389d1bb60`
- adapter family: `gpt_oss.decoder_lm_head_lora`
- coordinator backend:
  `open_adapter_backend.mlx.metal.gpt_oss_lm_head`
- contributor backend:
  `open_adapter_backend.cuda.gpt_oss_lm_head`

The coordinator report proves:

- both nodes were admitted as ready contributors
- both assignments were claimed and acknowledged
- both submissions were accepted
- validator summary recorded `accepted_contributions=2`
- replay summary recorded `replay_checked_contributions=2`
- one aggregation receipt set was emitted
- the mixed-backend compatibility digest was retained

The contributor report proves:

- the Linux RTX 4080 node actually dialed the Mac coordinator over the trusted
  LAN
- it accepted the assignment, emitted heartbeats, and returned one CUDA
  contributor receipt plus one adapter payload

## Measured Outcome

The retained bundle and runtime reports show:

- total contributions: `2`
- accepted contributions: `2`
- replay-checked contributions: `2`
- submission receipts: `2`
- coordinator wallclock: `158558 ms`
- contributor wallclock after connect: `886 ms`
- coordinator payload bytes: `1472`
- contributor payload bytes: `1488`

The promotion receipt stayed `held` for one honest reason:

- `validator_window_not_promotion_ready`

That is consistent with the bounded lane. The run proved contributor,
validator, replay, and aggregation truth. It did not claim a later publish or
promoted served snapshot.

## What Was Tested

The closure path now has three layers of test evidence:

1. local binary integrity:
   `cargo check -q -p psionic-train --bin first_swarm_trusted_lan_live_runtime`
2. retained bundle validation:
   `scripts/check-first-swarm-trusted-lan-real-run.sh --bundle fixtures/swarm/runs/first-swarm-live-20260327-real-2/first_swarm_real_run_bundle.json`
3. real operator execution:
   `scripts/run-first-swarm-trusted-lan-live.sh --run-id first-swarm-live-20260327-real-2`

The most important result is the third one. This audit is not closing the issue
on a simulated rehearsal or a manifest-only bundle. It is closing on a real
cross-machine run whose retained reports match the checker expectations.

## Failure Encountered And Fixed

The first 2026-03-27 live attempt failed honestly before this retained run
landed.

The coordinator waited for the remote clean-worktree build to finish, and that
delay let the local Mac assignment age past the tighter heartbeat/claim window.
The immediate validator error was:

- `UploadRequired` on the coordinator contribution

That was the right failure. It meant the coordinator submission was no longer
being accepted as an active contribution by the time validation ran.

The fix was minimal and correct:

- extend the coordinator worker-protocol heartbeat timeout and claim TTL for
  the bounded live-run path
- emit a fresh coordinator heartbeat before sealing the local submission

That change landed in commit `e4730a44` and the next live run succeeded without
changing the claim boundary.

## What This Run Now Proves

This lane now proves these concrete facts:

- `psionic` can execute one real trusted-LAN mixed-hardware decentralized
  open-adapter run across Mac MLX and Linux CUDA
- both backends can produce retained contributor receipts under one shared run
  family
- both contributions can be accepted under the shared worker protocol
- validator, replay, and aggregation truth can be sealed from the real
  cross-machine receipt set
- the operator flow is now repeatable through one script plus one bundle
  checker

That is enough to mark the original bounded `SWARM-0` lane complete.

## What This Run Still Does Not Prove

The retained run still does not prove:

- full-model dense training across Apple and NVIDIA backends
- MLX distributed parity with CUDA collectives
- internet discovery, NAT traversal, or elastic swarm membership
- automatic served-model promotion
- a published snapshot directly earned by the retained live run

The repo now does separately retain one truthful local snapshot publication
proof at:

- `fixtures/swarm/publications/first_swarm_local_snapshot_publication_v1.json`
- `docs/audits/2026-03-27-first-swarm-local-snapshot-publication-proof.md`

That closes the old “can the publish surface write one real snapshot
directory” gap without changing the more important live-run claim boundary.

Those are valid future goals. They are not part of the truthful completion bar
for this exact first swarm issue.

## How To Operate The Lane Now

1. Verify both machines are still ready:
   `scripts/check-swarm-mac-mlx-bringup.sh`
   `scripts/check-swarm-linux-4080-bringup.sh`
2. Run the retained live operator flow:
   `scripts/run-first-swarm-trusted-lan-live.sh --run-id first-swarm-live-$(date -u +%Y%m%dT%H%M%SZ)`
3. Verify the emitted bundle:
   `scripts/check-first-swarm-trusted-lan-real-run.sh --bundle fixtures/swarm/runs/<run_id>/first_swarm_real_run_bundle.json`
4. Review the coordinator and contributor runtime reports before making any
   stronger claim than the current bounded open-adapter proof.

One practical note matters:

- the script intentionally uses a clean remote worktree
  `~/code/psionic-swarm-first-live` instead of mutating the user’s ordinary
  Linux checkout

That keeps the operator path honest and reproducible even when the normal
remote checkout is dirty.

## Bottom Line

`SWARM-0` is now complete.

The repo finally has one committed real two-node mixed-hardware run with:

- explicit contributor receipts
- explicit submission receipts
- explicit validator and replay truth
- explicit aggregation outcome
- one retained after-action audit

The lane remains intentionally bounded and still refuses to fake broader claims.
That is the correct result.
