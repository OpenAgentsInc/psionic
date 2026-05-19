# Legal Benchmark Regression Guardrails

> Status: implemented_early.

Benchmark hill-climbing must not promote a legal benchmark candidate into
Autopilot4 production paths only because its Harvey-compatible score improved.
The product regression guardrail runs synthetic, non-Harvey tasks that represent
the Autopilot surfaces most likely to be harmed by benchmark overfitting.

## Fixture Suite

The checked suite lives at:

- `fixtures/legal_benchmark/product_regression_suite.json`

It covers these surfaces:

- chat
- Coder
- Work Orders
- GitHub provider
- CRM
- memory
- provider and tool routing

Every fixture task records:

- a synthetic instruction packet
- expected capabilities
- a minimum score
- a production baseline score
- a maximum allowed score drop
- a data policy that disallows live user data and Harvey hidden criteria

## Gate Evaluation

The Rust contract lives in
`crates/psionic-eval/src/legal_benchmark_regression.rs`.

`evaluate_product_regression_gate` consumes:

- a `ProductRegressionSuite`
- a `ProductRegressionGateConfig`
- a `ProductRegressionCandidateRun`

It emits a `ProductRegressionGateReport` with:

- benchmark target score
- benchmark candidate score
- aggregate product regression score
- per-surface product scores
- blocking failures
- generated Work Orders
- an Autopilot4 release-gate import object

The gate blocks promotion when any required task fails, a task drops below its
minimum score, a task regresses too far from its baseline, a product surface
falls below the configured surface minimum, the aggregate suite score falls
below the configured suite minimum, or the benchmark target score is missed.

## Failure Simulation

The checked failure candidate lives at:

- `fixtures/legal_benchmark/product_regression_candidate_failure.json`

It represents a candidate that improves the Harvey-compatible benchmark score
while regressing memory and provider/tool routing behavior. The deterministic
unit test proves this candidate creates Work Orders and exports a blocked
Autopilot4 release gate.

## Autopilot4 Import

Autopilot4 should import `ProductRegressionAutopilot4GateImport` and attach it
to the candidate Module Version release gate. A blocked import must prevent the
candidate from becoming the production default until the linked Work Orders are
resolved or a release owner records an explicit waiver in Autopilot4.

Psionic owns the score, fixture, and report contracts. Autopilot4 owns the
promotion decision, waiver policy, and operator UI.
