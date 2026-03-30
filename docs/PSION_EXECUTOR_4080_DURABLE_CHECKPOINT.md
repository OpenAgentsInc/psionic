# Psion Executor 4080 Durable Checkpoint

> Status: canonical `PSION-0302` / `#726` record, updated 2026-03-30 after
> landing the first retained durable-checkpoint packet for the admitted Mac ->
> 4080 Tailnet executor lane.

This document records the first retained packet that makes the admitted 4080
checkpoint path reviewable instead of leaving it implicit in swarm rerun JSON.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_4080_durable_checkpoint_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_4080_durable_checkpoint_fixtures
```

## What Landed

`psionic-train` now owns one typed durable-checkpoint packet that binds:

- the prerequisite admitted Mac -> 4080 remote-launch packet
- the retained admitted rerun bundle
- the retained coordinator runtime report
- the retained Linux RTX 4080 contributor runtime report
- the retained merged-artifact report
- the retained merged portable bundle readable from the control plane

That means the admitted 4080 lane now has one explicit packet for:

- the checkpoint pointer digest retained by the control-plane window plan
- the fact that both worker submission receipts preserve that same pointer
  digest as both source and target
- the remote contributor receipt checkpoint family
- the merged portable bundle that the control plane can import back through the
  shipped model-IO surface

## Current Retained Truth

- packet digest:
  `f8b326e0eb2ae45a3b7bafc553dfa4119074fd9e7e4c0d318c5bf52f2d03f0b5`
- prerequisite remote-launch packet SHA256:
  `299ba27704f2f96fc172815ccb1b07116715479f73c18f94cac7f24f4dd99dfe`
- retained run bundle file SHA256:
  `35c4cdaba64e5b4b235e3d1f77cc506cc74ec6bcf9a8f8eb9309bf9441e0bc83`
- retained run bundle in-band digest:
  `b0471dc0d8173862dbe5230d917b1b8af33989bee21757f63de4d5cd8b95a452`
- coordinator report SHA256:
  `dca06156d1c590d4343959f375d1572879540fe29071f0426c93eec7171b9c5c`
- contributor report SHA256:
  `f1ca4595a4b3c77f037b519d62048508a426c38ccf4393054218d1d4039c84bc`
- merged-artifact report SHA256:
  `7103aac1c1c50d35c84c39bcdc8b9cd366101330506a1f5248242f2fef5929b1`
- portable-bundle artifact digest / file SHA256:
  `67ae6a55a8c0b271467cfc06552baa4f5ed496b227afd7dbce34be5c2e076e99`
- checkpoint family:
  `swarm.local.open_adapter.policy:tailrun-home-admitted-20260328k`
- imported checkpoint family:
  `swarm.local.open_adapter.policy:tailrun-home-admitted-20260328k:merged`
- checkpoint pointer digest:
  `dd1aa85c355eb43934a20afe7e4204b3ed82bb85f4fe392dfde45229f4e434f8`
- checkpoint manifest digest:
  `007b3709823cc31f85a79ec775095eb1929da16eee6bd7e82b125cf00dca4337`
- checkpoint ref:
  `checkpoint://swarm/first-swarm-live-plan/policy`
- Linux submission receipt digest:
  `2ac5b5f344d8dd640bbcd0ce50a5c6b35e603049f21472161dc19c27135e2267`
- Mac submission receipt digest:
  `cf872e996686032214f7e78bbc8febc63e3e27e364d2e23fd1397a2c0571aa6a`
- local contributor receipt digest:
  `33134218130ac038b4d8aac4ff5ea582ba596105be45d0bdc4d3f0b86872f3eb`
- remote contributor receipt digest:
  `865ef2f86a2b50a6790996a07f53a345068eea2514cc7907c47bdece2c4c6305`
- merged portable-bundle state-dict digest:
  `b452249b53bb2b9e0625f78294bbbd10825d88706e1bc42b8a6a0eacc438c62c`
- deferred import-plan digest:
  `990d2d123049d497758a0fb9f967cbc2e0a3b58f784ae907b93343ad9a098719`
- compatibility-contract digest:
  `1ef6f938dc279e803e4c7fc2a43b4fc0dfff3d15a759864b8b94f78c95c8ec2d`
- validator summary digest:
  `a47d1c38e65537b1ac576bffb5fe66082a4cdaca20f14e074f14458777548534`
- promotion receipt digest:
  `d9b26f554a79b9b6c88586d635d943e014873e72265a2b8a4006e15aa0e11a50`

## Retained Checklist Rows

- `pointer_receipt_green`
- `submission_resume_anchor_green`
- `control_plane_readback_green`

## Claim Boundary

This packet counts as the first admitted **4080 durable checkpoint path** for
the executor lane.

It does **not** claim:

- a live interruption or restart rehearsal by itself
- checkpoint-time frequent-pack eval by itself
- decision-grade 4080 training closure
- independent worker-side checkpoint authority outside the controller-owned
  bundle path

Those stay later EPIC 3 obligations.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_4080_durable_checkpoint_fixtures`
- `cargo test -q -p psionic-train psion_executor_4080_durable_checkpoint -- --nocapture`
