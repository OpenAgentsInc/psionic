# Psion Executor 4080 Remote Launch

> Status: canonical `PSION-0301` / `#725` record, updated 2026-03-30 after
> landing the first retained Mac -> 4080 Tailnet remote-launch packet for the
> admitted executor lane.

This document records the first retained packet that makes the admitted Mac
control-plane to Linux RTX 4080 launch path reviewable instead of leaving it in
shell history.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_4080_remote_launch_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_4080_remote_launch_fixtures
```

## What Landed

`psionic-train` now owns one typed remote-launch packet that binds:

- the admitted Tailnet operator entrypoint
  `scripts/run-first-swarm-tailnet-admitted-live.sh`
- the retained admitted operator manifest
- the retained admitted run bundle
- the retained coordinator runtime report
- the retained Linux RTX 4080 contributor runtime report
- the retained worker acknowledgement receipt from the Mac coordinator

That means the admitted 4080 lane now has one explicit packet for:

- the one-command Tailnet launch entrypoint
- repo and workflow transfer to the remote worktree over SSH
- the frozen topology and workflow digests for the admitted run
- the remote worker acknowledgement that proves the Linux RTX 4080 node
  actually accepted the launch

## Current Retained Truth

- packet digest:
  `19aef8ffcf62006272d40206793d22f031a064a63dbc1254ad69ccd1351f4158`
- launch entrypoint SHA256:
  `96d26eb0a701acea39767c5af0a2f5516010cb92befe143539bf066e89c31266`
- operator manifest SHA256:
  `a08631ebe784f591c3c6bc1a77a8c43f56a12685db2337d40aeba50009148f05`
- retained run bundle file SHA256:
  `2666bec1a4ddbd65f3910e20c65ec4ddeb98a49620c0a56e39431c108689ff36`
- retained run bundle in-band digest:
  `bbefac6f4ab99f6a0ee446bfd657de62a17c37caa9c9cdcf8551619ad0450817`
- contributor report file SHA256:
  `b09eb8417025e11adc11d79900828fc69b0723b56b741f1353f002840673b886`
- retained contributor report digest:
  `28c67b4653f4208f9732b6e4b660913c9a346592b6cd678ed01c411533ae5da5`
- remote contributor host:
  `archlinux`
- coordinator endpoint:
  `100.127.107.31:35200`
- contributor endpoint:
  `100.108.56.85:35201`
- worker acknowledgement digest:
  `0779a0a548e9c84701e279c42f4e525138f816619093748abaa22488f127e3de`
- topology digest:
  `144481e413d4892b375e7b5f2fd8d24b7d98094806506b0e00a34262a2a9aa07`
- workflow-plan digest:
  `573b6df00266981819b8cb08654ac6e6f727418cb5a46768255e8ea27b2f2821`

## Retained Checklist Rows

- `launch_command_documented_green`
- `config_transfer_green`
- `worker_acknowledgement_green`

## Claim Boundary

This packet counts as the first admitted **Mac -> 4080 Tailnet launch receipt**
for the executor lane.

It does **not** claim:

- durable checkpoint return by itself
- checkpoint-time eval by itself
- decision-grade 4080 training closure
- recovery or replay closure

Those stay later EPIC 3 obligations.

## Validation

- `cargo run -q -p psionic-train --example psion_executor_4080_remote_launch_fixtures`
- `cargo test -q -p psionic-train psion_executor_4080_remote_launch -- --nocapture`
