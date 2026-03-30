# Compiled Agent Phase Six

> Status: phase-six operational reliability retained on 2026-03-29 for the
> same bounded compiled-agent family, with repeatable Tailnet runs, stricter
> route evidence, external anomaly detection, and no validator-threshold
> relaxation.

## Why This Exists

Phase five proved the Tailnet-first loop could run once without lying about the
governance boundary.

Phase six keeps that same narrow lane and makes it operationally credible:

- the Tailnet run is compared against the previous retained run
- route evidence gets harder, not easier
- grounded answer stays stable instead of being over-tuned
- external evidence gets anomaly flags and contributor trust posture
- alerts stay fail-closed for regression, quarantine spikes, and validator
  boundary violations

This is still not a new architecture phase.

## Canonical Fixtures

- `fixtures/compiled_agent/compiled_agent_phase_six_operational_report_v1.json`
- `fixtures/compiled_agent/compiled_agent_xtrain_cycle_receipt_v1.json`
- `fixtures/compiled_agent/compiled_agent_promoted_artifact_contract_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_submission_staging_ledger_v1.json`
- `fixtures/compiled_agent/external/compiled_agent_external_quarantine_report_v1.json`
- `fixtures/compiled_agent/tailnet/compiled_agent_tailnet_governed_run_v1.json`

## Current Retained Truth

- phase-six operational report digest:
  `b0d4061de01e35a21b83ff6c2f57fb5737905420cfb110d031c0116c3fabad86`
- current Tailnet governed run digest:
  `a6f3caf208fb510793d048fa44e8bab8f2761282f0a1c6d42d0845fe8208f2dc`
- current Tailnet staging digest:
  `5f4ce9b6126629220a69083eaa62f029bfd4054af38161cee52092489d0dabf8`
- current Tailnet quarantine digest:
  `1b71d21d1f1f0ca2ea2b2420709c91b8f1d6012cdcd884ee91bfe1431ddf00a3`
- current XTRAIN receipt digest:
  `b432ca5f00bffae428592411712bcae980262038bd684aa7ad2f6f39b8d49073`
- current promoted-artifact contract digest:
  `5835f484b83deb27ac7a7a96ae909011dd02a0c74582109b11ae04b3dfbeada4`

## Route Truth

- decision: `promote`
- promoted route artifact:
  `compiled_agent.route.multinomial_nb_v1`
- promoted route artifact digest:
  `2b66abacb8647f719f9b9a46a8cef007a5026b18c27af998cf2351e7a7a4560c`
- replay moved from `25` in phase five to `28`
- held-out moved from `12` in phase five to `14`
- retained regression count: `0`

Phase six also freezes the current route failure surfaces as permanent
held-out traps, including new compare / exclusion / negation rows:

- `openagents_ignore_wallet_compare_provider_heldout_receipt_v1`
- `openagents_without_provider_compare_wallet_heldout_receipt_v1`
- `openagents_dont_compare_wallet_provider_poem_heldout_receipt_v1`

Confidence calibration is now retained instead of implied:

- learned correct mean confidence: `0.89200014`
- learned incorrect mean confidence: `0.8866666`
- low-confidence disagreement count: `1`

That is not a solved calibration problem. It is a retained surface that can now
force review without relaxing the no-regression rule.

## Grounded Answer Truth

- decision: `promote`
- promoted grounded-answer artifact:
  `compiled_agent.grounded_answer.multinomial_nb_v1`
- promoted grounded-answer artifact digest:
  `869217d751e61e52f32f1dfdd0f5dc18d3e9c0a1d15dce0f86066356075b2782`
- replay match count: `28`
- held-out match count: `16`
- retained regression count: `0`

Grounded answer stays the boring control lane in phase six:

- no new grounded failure class is promoted into authority
- disagreement rows stay reviewable
- missing or conflicting facts still refuse instead of being guessed

## External Evidence Boundary Truth

- staging ledger digest:
  `035a9a3b928df3a27fed1d7770f7a9805f5774354dcbd9c5e16acb4f2252e5c2`
- quarantine report digest:
  `c53147143de900e4c8675cc2688bc5b17fbe8ba56fd31c2b5827ec5c62ac2e4e`
- anomaly submission count: `2`
- watch contributors:
  - `contrib.external.alpha`

Retained anomaly submission ids:

- `submission.compiled_agent.external_benchmark_invalid.alpha.v1`
- `submission.compiled_agent.external_runtime_disagreement.alpha.v1`

Retained trust posture now exists without changing authority:

- one contributor history is `watch` because it produced a rejected submission
  plus repeated anomaly flags
- one contributor history remains `neutral` while it accumulates accepted and
  review-required bounded evidence on the admitted family

That is the intended evidence boundary:

- schema / contract / digest drift fail closed
- anomaly flags tighten review posture
- nothing bypasses staging or quarantine

## Tailnet Cadence Truth

The phase-six report compares the current run against the retained phase-five
baseline instead of treating the Tailnet loop as a one-off:

- previous governed run:
  `dc9ab99b00fa05ae990693b5e758cc728d7d06dcef36bb51b86bf769c7f18b37`
- current governed run:
  `a6f3caf208fb510793d048fa44e8bab8f2761282f0a1c6d42d0845fe8208f2dc`
- previous XTRAIN digest:
  `4f7655b1b65931c538c3fbea643452a8a16e1ad7738ae4a9e12896ef722cef45`
- current XTRAIN digest:
  `b432ca5f00bffae428592411712bcae980262038bd684aa7ad2f6f39b8d49073`

All retained operational alerts remain fail-closed and currently untriggered:

- held-out regression
- quarantine spike
- validator-boundary violation

## Commands

Refresh the phase-six retained loop:

```bash
cargo run -q -p psionic-train --bin compiled_agent_receipt_to_replay
cargo run -q -p psionic-train --bin compiled_agent_xtrain_loop
```

Refresh the Tailnet governed run:

```bash
cargo run -q -p psionic-train --bin compiled_agent_tailnet_node_bundle -- \
  --profile tailnet_m5_mlx

cargo run -q -p psionic-train --bin compiled_agent_tailnet_node_bundle -- \
  --profile tailnet_archlinux_rtx4080_cuda

cargo run -q -p psionic-train --bin compiled_agent_tailnet_governed_run -- \
  --local-bundle fixtures/compiled_agent/tailnet/compiled_agent_tailnet_m5_node_bundle_v1.json \
  --remote-bundle fixtures/compiled_agent/tailnet/compiled_agent_tailnet_archlinux_node_bundle_v1.json
```

Verify the retained phase-six surfaces:

```bash
cargo test -q -p psionic-train compiled_agent_external_intake -- --nocapture
cargo test -q -p psionic-train compiled_agent_phase_six -- --nocapture
cargo test -q -p psionic-eval \
  full::compiled_agent_module_eval::tests::compiled_agent_module_eval_report_matches_committed_truth \
  -- --exact --nocapture
```

## Boundary

Phase six makes the bounded loop more reliable. It does not:

- widen the compiled-agent family
- relax validator thresholds
- grant contributor promotion authority
- treat external evidence as authority
- make Tassadar a prerequisite
