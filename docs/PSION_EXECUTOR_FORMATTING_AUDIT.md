# Psion Executor Formatting Audit

> Status: canonical `PSION-0107` / `#712` formatting, normalization, and
> post-processing audit for the frozen executor eval families, updated
> 2026-03-30.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_formatting_audit_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_formatting_audit_fixtures
```

## What This Audit Checks

This audit does not create a second eval system.

It records, suite by suite, that the frozen executor eval packs already use one
explicit prompt surface, one explicit normalization posture, and one explicit
post-processing posture.

The current retained audit covers every frozen suite in:

- `tassadar.eval.frequent.v0`
- `tassadar.eval.promotion.v0`

No eval family is allowed to remain unchecked.

## Prompt-Surface Rules

The audit freezes four prompt-surface classes:

- `article_benchmark_package`
  - used by exactness, throughput, `reference_linear`, and `hull_cache`
    fast-route suites
- `generalization_gate_case_rows`
  - used by held-out and adversarial suites
- `exclusion_boundary_manifest`
  - used by held-out exclusion rows that are boundary truth, not benchmark
    prompts
- `manual_checklist`
  - used by operator, runtime, and serving blockers

This keeps the formatting claim narrow:

- benchmark suites stay evaluator-owned
- generalization suites stay classification-owned
- boundary rows stay manifest-owned
- operator/runtime/serving blockers stay checklist-owned

## Normalization And Post-Processing Rules

The retained audit freezes four normalization or post-processing postures:

- exactness and throughput metrics are evaluator-owned only
- held-out and adversarial suites normalize only through explicit
  `exact` / `mismatch` / `refusal` classification plus retained equality facts
- exclusion rows normalize only through manifest membership and lineage
- checklist suites normalize only through explicit green/red review state

The important consequence is simple:

- there is no hidden output cleanup layer on benchmark suites
- there is no silent mismatch suppression on held-out or adversarial suites
- there is no prompt execution claim on exclusion rows
- there is no benchmark shortcut that replaces operator/runtime/serving review

## Manual Review Slices

Automation is not enough for every suite. The retained audit therefore defines
these manual slices:

- `held_out_boundary_review_v0`
- `operator_readiness_review_v0`
- `generalization_red_case_review_v0`
- `promotion_runtime_blocker_review_v0`
- `promotion_serving_blocker_review_v0`

Those slices cover the families where human review still owns the final truth:

- exclusion-boundary integrity
- artifact packet and restore/export readiness
- held-out or adversarial mismatch/refusal review
- runtime blocker review
- serving and rollback seam review

## Current Honest Boundary

This audit makes the formatting contract explicit.

It does not claim that every operator, runtime, or serving blocker is already
fully machine-audited. Those lanes remain explicit manual slices, and that is
the correct current boundary for decision-grade admission.
