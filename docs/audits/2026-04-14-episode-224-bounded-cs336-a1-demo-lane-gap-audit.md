# Episode 224 Bounded CS336 A1 Demo Lane Gap Audit

> Status: implemented on 2026-04-14 to close `psionic#943` without widening
> the bounded lane claim boundary.

This audit answers the lane-owned portion of the Episode 224 demo question:

Can the packaged CS336 A1 demo lane be rerun now as an operator workflow, or
are we still only citing the retained dual-host proof from Episode 223?

## Deliverable A: Gap Table

| Layer | Already implemented | Missing / partial | Severity | Repo / owner | Exact recommended fix |
| --- | --- | --- | --- | --- | --- |
| Packaged lane identity | Stable lane id, release id, environment ref, backend family, and topology class already existed in `psion_cs336_a1_demo_*` and `train_runtime.rs` | No first-class lane-side command that decides whether one fresh run is actually demo-valid | `important` | `psionic` / `psionic-train` | Add `verify --run-root <path>` and make it fail closed unless the retained runtime packets, checkpoint surface, accepted pointer, closeout bundle, and loss descent all exist together |
| Operator verification path | `start`, `rehearse-base-lane`, and `status` already existed | Operators still had to eyeball multiple JSON files by hand | `important` | `psionic` / `scripts` | Add one dedicated checker script that can launch a fresh bounded rehearsal or validate an existing run root |
| Docs / honesty boundary | Focused lane doc already existed and correctly stated the bounded claim boundary | No repo-owned Episode 224 lane audit, no explicit success gate, no blunt still-not-working list | `important` | `psionic` / `docs` | Add this audit and extend the lane doc with the exact demo-valid gate and caveats |
| Fixture portability | Canonical fixture request/output pair and example run root already exist | Checked-in example JSON still carries one repo-local absolute path family, which is acceptable for local evidence but not ideal for portability | `nice-to-have` | `psionic` / fixtures | Normalize or project a portable fixture view in a later cleanup pass without pretending the runtime emits relative paths today |

## Deliverable B: Demo-Readiness Verdict

Verdict: `ready with operator-only path`

Why:

- the packaged bounded lane is real and already writes the retained outputs that
  `Pylon` and `Nexus` expect
- a fresh local bounded run can now be launched and verified through explicit
  repo-owned commands instead of a manual JSON inspection ritual
- the lane still does not prove multi-host distribution by itself; that proof
  remains a live `Pylon`/`Nexus` concern above this repo

## Specific Questions

- Can the bounded CS336 A1 lane be relaunched safely and repeatedly?
  Yes. The lane is still fixed to the same tiny corpus, fixed four-step budget,
  fixed release id, and fixed environment ref, and the new verifier now gives
  one explicit pass/fail answer for a fresh run root.
- Are the packaged lane identity, release id, and environment ref still correct?
  Yes. They remain `psion_cs336_a1_demo_v1`,
  `psionic-train.psion_cs336_a1_demo.release.v1`, and
  `psionic.environment.psion_cs336_a1_demo.host_cpu.operator@v1`.
- Do checkpoints and closeout outputs still land where `Pylon` and `Nexus` expect?
  Yes. The verifier now explicitly checks for the retained runtime packets, the
  generic checkpoint surface, the accepted checkpoint pointer, and the closeout
  bundle under the canonical run-root family.
- Is fresh multi-host proof generation currently realistic, or is the system still relying on a prior retained proof?
  Fresh multi-host proof is realistic only when live `Pylon` and `Nexus`
  assignment intake are healthy. This repo can now prove the bounded lane is
  launchable and locally valid again, but it does not own live distribution.
- What exact caveats must remain in the runbook to keep the demo honest?
  The runbook must keep three caveats visible: this verifier is single-host,
  live multi-host proof still depends on `Pylon` and `Nexus`, and treasury /
  payout reconciliation remain downstream concerns outside this lane.

## Deliverable C: Implemented Changes

Files changed:

- `crates/psionic-train/src/psion_cs336_a1_demo_operator.rs`
- `scripts/train-psion-cs336-a1-demo.sh`
- `scripts/check-psion-cs336-a1-demo-lane.sh`
- `docs/PSION_CS336_A1_DEMO_LANE.md`
- `docs/TRAIN_SYSTEM.md`
- `docs/audits/2026-04-14-episode-224-bounded-cs336-a1-demo-lane-gap-audit.md`

Contract change:

- added `psionic-train cs336-a1-demo verify --run-root <path>`
- the verifier loads the retained launch manifest, current status, retained
  summary, checkpoint surface, and closeout bundle
- it fails closed when a run is only a dry run, when retained runtime packets
  are missing, when the generic checkpoint surface is missing, when the
  closeout bundle is missing, or when loss does not descend

Verification coverage:

- `packaged_demo_verify_reports_success_for_real_run`
- `packaged_demo_verify_rejects_dry_run_as_demo_ready`

## Deliverable D: Operator Runbook Contribution

1. Launch a fresh bounded run:

```bash
./TRAIN --lane cs336_a1_demo rehearse-base-lane \
  --run-id psion-cs336-a1-demo-episode-224 \
  --output-root ~/scratch/psion_cs336_a1_demo_runs/psion-cs336-a1-demo-episode-224 \
  --git-ref HEAD
```

2. Inspect the retained state if needed:

```bash
./TRAIN --lane cs336_a1_demo status \
  --run-root ~/scratch/psion_cs336_a1_demo_runs/psion-cs336-a1-demo-episode-224
```

3. Ask the lane for the pass/fail answer:

```bash
./TRAIN --lane cs336_a1_demo verify \
  --run-root ~/scratch/psion_cs336_a1_demo_runs/psion-cs336-a1-demo-episode-224
```

4. Or run the one-shot checker:

```bash
bash scripts/check-psion-cs336-a1-demo-lane.sh
```

Expected success shape:

- lane id: `psion_cs336_a1_demo_v1`
- release id: `psionic-train.psion_cs336_a1_demo.release.v1`
- environment ref: `psionic.environment.psion_cs336_a1_demo.host_cpu.operator@v1`
- status phase: `rehearsed` or `completed`
- one accepted checkpoint pointer
- one generic checkpoint surface
- one closeout bundle
- `final_loss < initial_loss`

What this proves:

- the bounded lane can still be launched now
- the same retained artifact family still lands under one run root
- downstream repos can rely on the lane’s retained packet family again

What this does not prove:

- live multi-host assignment pickup by itself
- live contribution acceptance by `Nexus`
- treasury hydration, payout dispatch, or reconciliation

## Deliverable E: Still Not Working

- The repo-owned checker is intentionally single-host. It does not create fresh
  multi-host proof without live `Pylon` and `Nexus`.
- The checked-in example fixture family still carries repo-local absolute paths.
  That is acceptable for deterministic local evidence, but it is not yet a
  portable path projection.
- Treasury, payout, and contribution reconciliation are not lane-owned and must
  stay out of scope here.
