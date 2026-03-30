# Psion Executor 4080 Interruption Recovery

> Status: canonical `PSION-0305` / `#729` record, updated 2026-03-30 after
> landing the first retained interruption-recovery packet for the admitted
> Mac -> 4080 Tailnet executor lane.

This document records the first retained 4080 interruption-recovery packet that
binds the live smoke-run checkpoint and heartbeat evidence to the frozen
failure-drill contract so recovery, replay, and lost-work policy stay explicit.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_4080_interruption_recovery_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_4080_interruption_recovery_fixtures
```

## What Landed

`psionic-train` now owns one typed interruption-recovery packet that binds:

- the prerequisite retained smoke-run packet
- the frozen trusted-LAN failure-drill bundle
- the retained admitted coordinator and contributor runtime reports
- the Linux worker progress heartbeats and submission receipt
- the live replay-policy identity for the admitted Linux worker slice
- the checkpoint pointer and replay receipt digests that keep restore evidence
  machine-legible

That means the admitted 4080 lane now has one explicit packet for:

- stale-worker timeout and replay posture
- contributor-loss and upload-disagreement refusal posture
- bounded lost-work policy for one admitted Linux contribution window
- explicit restore evidence tied to the live smoke-run checkpoint

## Current Retained Truth

- packet digest:
  `d07f14dd64ce0f66d8827a9de1c6353dd5f1d001a9c81bc74669bc12e2def2c6`
- smoke packet SHA256:
  `d32483674a6e1b35f1e1efaa8c88c3a3d2c3930ecefc423af4245b6aeded41ac`
- failure-drill bundle SHA256:
  `38a163a309134e3438b7ea437dacc24f217bd8f605642eff1c813104c9bd3dde`
- failure-drill bundle digest:
  `ad84b1d34b232ac2562a75b0afad1e60085f146bcbacbd9bf59780807ea2a08f`
- coordinator report SHA256:
  `dca06156d1c590d4343959f375d1572879540fe29071f0426c93eec7171b9c5c`
- contributor report SHA256:
  `f1ca4595a4b3c77f037b519d62048508a426c38ccf4393054218d1d4039c84bc`
- run id:
  `tailrun-home-admitted-20260328k`
- run family id:
  `swarm.local.mlx_metal_plus_rtx4080.open_adapter.v1`
- worker id:
  `swarm-linux-4080-a`
- worker session id:
  `swarm-linux-4080-a-session-tailrun-home-admitted-20260328k`
- worker role id:
  `swarm.linux.cuda.rtx4080.contributor`
- worker slice id:
  `swarm_linux_cuda_rtx4080_contributor-2`
- replay policy id:
  `replay.open_adapter.strict`
- shared replay identity digest:
  `05f957065eaa2d1cf2a8bc90b9726722cb1d223728113e09d492d024a028af2c`
- checkpoint family:
  `swarm.local.open_adapter.policy:tailrun-home-admitted-20260328k`
- checkpoint pointer digest:
  `dd1aa85c355eb43934a20afe7e4204b3ed82bb85f4fe392dfde45229f4e434f8`
- checkpoint ref:
  `checkpoint://swarm/first-swarm-live-plan/policy`
- checkpoint step:
  `12`
- stale-worker timeout:
  `5000 ms`
- stale-worker disposition:
  `replay_required`
- contributor-loss disposition:
  `replay_required`
- upload-disagreement disposition:
  `rejected`
- uneven-worker-speed disposition:
  `wait_then_replay`
- uneven-worker-speed observed skew:
  `18000 ms`
- live Linux upload manifest digest:
  `c251d7336b6401d02e310723466946a286497f614a03ec3a0c759ba914812593`
- max unreported progress:
  `4 steps / 4 samples`
- max replay loss:
  `12 steps / 12 samples`
- Linux progress heartbeat receipt digests:
  `4350af732b0fd8e650c5be097f0fbe9cef4839faa2182a76ed043bf25c122d8c`,
  `38f47ace2b9b73dc691026173e228e51475c441b41b39fd87443103f6b3ffdbc`
- Linux submission receipt digest:
  `2ac5b5f344d8dd640bbcd0ce50a5c6b35e603049f21472161dc19c27135e2267`
- Linux contributor receipt digest:
  `865ef2f86a2b50a6790996a07f53a345068eea2514cc7907c47bdece2c4c6305`
- validator summary digest:
  `a47d1c38e65537b1ac576bffb5fe66082a4cdaca20f14e074f14458777548534`
- replay receipt digests:
  `3382217759f7af9e85eae71b2462d4cb51561685d28c39e2e0e13087553db6a4`,
  `633590d3e581862b8214cf7158f0207932dc959cb7234990735c62182f47b23d`
- coordinator report digest:
  `eff0788c3f7f1d459b4eba3dffa1a118096d6577912fcf211e1707764f647ea5`
- contributor report digest:
  `6ea6d11a125487435a7a1e08ab9e15605c94768ee0ec86d17478b48f21a27fe2`

## Claim Boundary

This packet counts as the first admitted **interruption-recovery packet** for
the 4080 lane.

It does **not** claim:

- a separately retained crashed-then-resumed live run bundle
- that the lane has already closed decision-grade 4080 readiness
- that replay-required drills are now safe to ignore
- that external or broader cluster recovery is solved

Instead it binds the real smoke-run checkpoint and heartbeat evidence to the
frozen recovery drills so stale-worker timeout, replay posture, lost-work
bounds, and restore evidence are all reviewable in one retained packet.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_4080_interruption_recovery_fixtures`
- `cargo test -q -p psionic-train builtin_executor_4080_interruption_recovery_packet_is_valid -- --exact --nocapture`
- `cargo test -q -p psionic-train executor_4080_interruption_recovery_fixture_matches_committed_truth -- --exact --nocapture`
