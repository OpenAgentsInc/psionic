# Psion Executor Acceptance Profile

> Status: canonical `PSION-0004` / `#703` executor-capable `Psion`
> acceptance profile, updated 2026-03-30.

## Why This Doc Exists

The executor-capable `Psion` lane needs one explicit acceptance profile for
promotion, export, and replacement work.

That profile cannot be inferred from the generic compact-decoder `Psion`
acceptance matrix, because the executor lane carries additional route,
replacement, fast-path, CPU-matrix, and consumer-seam obligations.

## Separation From Generic Psion Acceptance

- `docs/PSION_ACCEPTANCE_MATRIX.md` remains the generic compact-decoder
  learned-lane acceptance contract
- this doc owns executor-capable `Psion` acceptance only
- no promotion gate for the executor lane is allowed to flatten these two
  profiles into one contract

## Phase-One Executor Gates

Every executor-capable candidate must keep all of the following green:

### Exactness gate

- clear the exactness suite on the admitted executor workload family
- preserve saturated exactness rows already green on the baseline

### Held-out and adversarial gate

- clear the frozen held-out and adversarial suites for the executor lane
- preserve regression-free posture on those suites

### `reference_linear` anchor gate

- `reference_linear` remains green as the measured baseline truth anchor
- `reference_linear` remains the wider floor for bounded route validity

### `hull_cache` fast-route gate

- `hull_cache` remains the admitted fast-route target on the executor workload
  family
- admitted-workload `hull_cache` route replacement must stay green

### Throughput-floor gate

- throughput must stay above the frozen executor threshold bands
- throughput wins do not override exactness or held-out regressions

### CPU-matrix gate

- deterministic runtime validation must stay green on:
  - `host_cpu_aarch64`
  - `host_cpu_x86_64`

### Export and replacement gate

- candidate exports cleanly into the canonical executor artifact flow
- replacement packet carries descriptor, artifact, lineage, and replacement
  report truth

### Local-cluster gate

- at least one green Mac -> 4080 -> Mac local-cluster roundtrip exists for the
  candidate path where required by the phase

### Consumer-seam gate

- remote-training visibility is green where applicable
- promoted-artifact compatibility is green where applicable
- compiled-agent shadow and rollback safety stay green where applicable

## Promotion Use Rule

Use this profile for:

- `tassadar.eval.promotion.v0`
- executor promotion packets
- executor replacement verdicts
- first `trained-v1` candidate promotion

No executor-capable promotion issue should define its own gate set without
pointing back to this profile.

## Frozen Pack Binding

- `tassadar.eval.frequent.v0` is the checkpoint-time decision pack that keeps
  exactness, held-out exclusions, operator review cases, and throughput
  blockers frozen between serious runs
- `tassadar.eval.promotion.v0` is the promotion pack that binds the exactness,
  held-out, adversarial, runtime, serving, `reference_linear`, `hull_cache`,
  and throughput-threshold surfaces back to this profile
