# Qwen Tuned Adapter Serving

> Status: implemented local smoke metadata path on 2026-05-19.

This document describes the first Psionic legal benchmark path for comparing a
base Qwen candidate with a tuned Qwen adapter candidate through one
OpenAI-compatible provider abstraction.

The code lives in:

- `crates/psionic-eval/src/legal_benchmark_provider.rs`
- `crates/psionic-eval/src/legal_benchmark_agent.rs`

The training-side adapter smoke is documented in
`docs/QWEN_LEGAL_FINETUNE_LANE.md`.

## Candidate Identity

Every Qwen legal candidate must bind:

- base model id: `Qwen/Qwen3.5-4B`
- served model id: `qwen3.5-4b`
- family acceptance label: `qwen35`
- base artifact digest
- tokenizer digest
- tokenizer contract digest
- prompt-template digest
- dataset digest
- eval-pack digest
- serving revision

Tuned candidates additionally bind:

- adapter id
- adapter revision
- adapter artifact digest
- adapter identity digest
- last-known-good adapter revision
- rollback adapter revision

`ensure_qwen_candidate_pair_compatible` refuses wrong base, wrong template,
wrong tokenizer, wrong dataset, and wrong eval-pack pairings before a
base-versus-adapter score bundle can be emitted.

## Serving Path

`QwenLegalBasePlusAdapterCandidateIdentity::openai_compatible_route` builds the
route used by the legal benchmark provider. It keeps the normal
`/chat/completions` path, sets the route model id to the active serving alias,
and adds non-secret Psionic headers:

- `x-psionic-base-artifact-digest`
- `x-psionic-template-digest`
- `x-psionic-dataset-digest`
- `x-psionic-eval-pack-digest`
- `x-psionic-adapter-artifact-digest` for tuned candidates

The provider copies the same candidate metadata onto every `ModelResponse`.
The legal benchmark agent then copies route metadata onto `RunRecord` and
`LegalBenchmarkRunReceipt`, so response and run receipts retain the base,
adapter, template, dataset, and eval-pack digests.

## Rollback

The first sidecar path is metadata-only and intentionally conservative:

- `last_known_good_adapter_revision` records the active rollback target.
- `rollback_adapter_revision` records the fallback revision or `base`.
- The provider refuses incompatible base/template/tokenizer/dataset/eval-pack
  pairings instead of silently selecting a fallback adapter.

Operational serving should resolve the active alias before launch and only
point `serving_model_id` at a revision that passed local smoke. A failed tuned
adapter should be rolled back by changing the alias and preserving both old and
new candidate identities in the score bundle.

## Score Bundles

`qwen_legal_base_vs_adapter_score_bundle` separates:

- `base_model_score`
- `tuned_adapter_score`
- `mock_local_smoke_score`
- `retained_score_claim`

Mock/local smoke candidates are explicitly blocked from retained public score
claims. A retained score entry is emitted only when the tuned candidate identity
declares `RetainedScoreClaim`.

## Local Smoke

Run the focused provider tests:

```bash
cargo test -p psionic-eval qwen_
```

The smoke proves:

- base and tuned candidates share one OpenAI-compatible provider path
- tuned routes carry adapter/base/template/dataset/eval-pack digests
- response metadata records the same digests
- wrong-template candidate pairs fail closed
- mock/local smoke scores cannot become retained score claims

This is not a public Harvey score claim.
