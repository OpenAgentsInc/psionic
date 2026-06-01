# Legal Benchmark Signature Routing

> Status: implemented_early.

Psionic now has a public/synthetic Harvey-compatible signature-routing fixture
lane for Probe+Codex. The lane does not claim private Harvey performance. It
tests whether structured legal failure families select the right Probe
signatures and whether the routed run preserves the evidence needed for human
review and promotion gates.

## Boundary

Psionic owns:

- public/synthetic legal task envelopes
- legal failure taxonomy
- answer path, source refs, answer integrity, score report, and judge sidecar
  evidence requirements
- deterministic fixture reports

Probe owns:

- runtime signature selection
- Codex session prompt/context injection
- tool policy enforcement at run time

Autopilot owns:

- business-facing review
- approval
- acceptance
- promotion and public/private projection

The fixture suite does not expose hidden Harvey labels, hidden rubrics, private
client data, or legal authority. It is a workflow and evidence gate.

## Implemented Fixture Lane

Code:

- `crates/psionic-eval/src/legal_benchmark_signature_routing.rs`
- `crates/psionic-eval/examples/legal_benchmark_signature_routing_report.rs`

Fixtures:

- `fixtures/legal_benchmark/signature_routing/harvey_public_synthetic_signature_routing_suite.json`
- `fixtures/legal_benchmark/signature_routing/harvey_public_synthetic_signature_routing_report.json`

The suite covers six public/synthetic failure families:

| Failure family | Expected signatures |
| --- | --- |
| `missing_deliverable` | `legal.deliverable_file_workflow`, `legal.output_path_contract`, `legal.answer_integrity_guard` |
| `wrong_output_path` | `legal.output_path_contract`, `legal.deliverable_file_workflow`, `legal.answer_integrity_guard` |
| `source_grounding_missing` | `legal.source_grounding_trace`, `legal.citation_provenance_check`, `legal.answer_integrity_guard` |
| `citation_provenance_missing` | `legal.citation_provenance_check`, `legal.source_grounding_trace`, `legal.answer_integrity_guard` |
| `answer_integrity_invalid` | `legal.answer_integrity_guard`, `legal.deliverable_file_workflow`, `legal.output_path_contract` |
| `judge_supervisor_needed` | `benchmark.legal_judge_supervisor`, `legal.answer_integrity_guard`, `legal.source_grounding_trace` |

The selector input is structured enum data, not keyword matching over a legal
prompt.

## Current Report

Generate the retained report:

```bash
cargo run -q -p psionic-eval --no-default-features --example legal_benchmark_signature_routing_report
```

Current retained result:

- fixtures: `6`
- selector pass rate: `10000` bps
- raw Codex deterministic fixture mean: `2222` bps
- Probe+Codex deterministic fixture mean: `10000` bps
- mean fixture delta: `7777` bps
- report hash:
  `c06b2076ee7db63afb0468741747f5b0f21ce5303e8d649c675d5607baf64f97`

Interpretation: this proves the typed legal signature-routing fixture and
evidence contract. It does not prove live Codex legal quality, private Harvey
score lift, or legal correctness.

Small deterministic eval-suite smoke:

```bash
cargo run -q -p psionic-eval --no-default-features --example legal_benchmark_eval_suite -- \
  --suite suites/harvey_public_001_single.json \
  --model raw-codex-fixture \
  --adapter probe-codex-signatures \
  --out target/legal/harvey_public_signature_routing_smoke
```

Recorded result: base score `0` bps, adapter score `10000` bps, delta
`10000` bps, report hash
`b141c7ecaada9f0b715aee9fb6755394434437979d607507a50ab8d02e5e1deb`.
This smoke uses deterministic replay outcomes; it is a wiring check.

## Required Evidence

Probe+Codex fixture rows must preserve:

- required answer file at the exact task path
- source reference sidecar
- answer integrity receipt tying final answer bytes to a model-authored write
- score report
- judge sidecar when the failure family is `judge_supervisor_needed`

Wrong-output-path and missing-deliverable baselines must fail with explicit
taxonomy labels instead of collapsing into an unclassified failure.

## Validation

Focused validation:

```bash
cargo test -p psionic-eval --no-default-features --lib legal_benchmark_signature_routing
python3 -m json.tool fixtures/legal_benchmark/signature_routing/harvey_public_synthetic_signature_routing_suite.json >/dev/null
python3 -m json.tool fixtures/legal_benchmark/signature_routing/harvey_public_synthetic_signature_routing_report.json >/dev/null
```

Full legal benchmark CI:

```bash
scripts/check-legal-benchmark-ci.sh
```

## Next Work

The next implementation step is a real retained legal run where Probe passes
the selected legal signatures into a Codex-backed session and Psionic imports
the resulting answer files, source refs, integrity report, score report, and
judge sidecar. That run should stay public/synthetic until the hidden/private
benchmark leakage rules are reviewed again.
