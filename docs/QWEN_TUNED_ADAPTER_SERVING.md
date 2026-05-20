# Qwen Tuned Adapter Serving

> Status: implemented local smoke metadata path on 2026-05-19; real local
> Qwen/MLX tool-backed Rust agent smoke added on 2026-05-20; local RL-seed
> resumed adapter smoke added on 2026-05-20; public Harvey MFN training-slice
> adapter and task run added on 2026-05-20.

This document describes the first Psionic legal benchmark path for comparing a
base Qwen candidate with a tuned Qwen adapter candidate through one
OpenAI-compatible provider abstraction.

The code lives in:

- `crates/psionic-eval/src/legal_benchmark_provider.rs`
- `crates/psionic-eval/src/legal_benchmark_agent.rs`

`psionic-eval` now includes `ReqwestBlockingHttpTransport` for local
OpenAI-compatible smoke routes. It refuses unresolved `<secret_ref:...>`
headers, so hosted provider credentials still need a resolver-owned transport.

The training-side adapter smoke is documented in
`docs/QWEN_LEGAL_FINETUNE_LANE.md`.

## Current Local Real-Weight Adapter

The current material adapter that can be served locally is:

```text
fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002/adapters.safetensors
```

It was trained on 2026-05-20 with `mlx_lm.lora` against
`Qwen/Qwen3.5-0.8B`. The machine-readable report is:

```text
fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002/report.json
```

Report digest:

```text
b9c3c9dac55c469be1e946c9ea2e7be9255dfa2f02a097d31df97bf9d64592d5
```

Start the local adapter server:

```bash
scripts/run-qwen35-08b-legal-mlx-lora-server.sh
```

Then point a Psionic/OpenAI-compatible Harvey smoke route at:

```text
base_url: http://127.0.0.1:18088/v1
model: Qwen/Qwen3.5-0.8B
```

The MLX server verified `POST /v1/chat/completions` with that model id and the
adapter path. Do not use a synthetic alias as the request model id unless an
aliasing wrapper is added; MLX resolves unknown model ids through Hugging Face.

Claim boundary: this is a locally trained Qwen-family LoRA adapter suitable for
smoke benchmark routing. It is not the retained `Qwen/Qwen3.6-35B-A3B` target,
not RL, and not a retained Harvey score claim.

## Rust Harvey Agent Smoke

With the adapter server running, run the Rust benchmark-agent smoke:

```bash
scripts/run-qwen35-08b-legal-mlx-lora-harvey-smoke.sh
```

The 2026-05-20 fixture result is:

- run id:
  `run.legal.qwen35_08b_mlx_lora.harvey_tool_smoke.f2972e6fead2.qwen35-08b-mlx-lora-2026-05-20`
- terminal state: `submitted`
- adapter: `Qwen/Qwen3.5-0.8B` plus the committed MLX LoRA adapter
- output artifact count: `1`
- tool receipt count: `1`
- run record hash:
  `3463444d89f01b57a7d25304cce0a3033665fa01e8ed3c130613db456fd026db`
- transcript hash:
  `0b1262fe073f221de39854311216107836d18287612deb8f772332b3beaeaf60`
- smoke report digest:
  `13c28f8ff6f3e8fad7b81b537947dfa029295449aa913b8c0b57800be76d90c9`
- deterministic score report digest:
  `610eba2cc13ad7a16069d60eee9dbfa95f829ca1a4dfa20bb45108d5f004ac2d`
- training record bundle digest:
  `db28588382457abf216b31e00d6875c0525e026a706c3448e78e26c6497e74e3`

The smoke artifacts live at:

```text
fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002/harvey_agent_smoke
```

This run is useful as an RL trajectory seed because the adapter wrote a
deliverable through the Rust `write` tool, produced a receipt-backed output
artifact, submitted through the legal benchmark agent loop, and exported one
canonical legal benchmark training record. It is still a no-source smoke task,
not a retained Harvey benchmark score.

## RL-Seed Adapter Smoke

The current local policy-refresh candidate resumes from the first adapter and
trains on the accepted benchmark trajectory:

```text
fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_rl_seed_2026_05_20_003/adapters.safetensors
```

Report:

```text
fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_rl_seed_2026_05_20_003/report.json
```

Report SHA-256:

```text
1637bc930dbe06607899bf0ebc9c7f8c37bf15562728edd42fe1bfa175bf194c
```

Serve it:

```bash
MODEL_ID=Qwen/Qwen3.5-0.8B \
ADAPTER_PATH=fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_rl_seed_2026_05_20_003 \
PORT=18089 \
scripts/run-qwen35-08b-legal-mlx-lora-server.sh
```

The recorded Rust agent smoke against this adapter has:

- terminal state: `submitted`
- output artifact count: `1`
- tool receipt count: `1`
- run record hash:
  `0df6c6767ea204c70a16f1b513ec2517a638790852f219f26413929712d131cf`
- transcript hash:
  `74c3211b61e78b4424bc50d74db388afef6131535fbcc99da071e97a3bf80ab3`
- smoke report digest:
  `db934020917452de144e330d3767b3242590a8adb838eac1a9676429b691f206`
- score report digest:
  `1402538472788138e97f1055422a75bbbd3f3d2c208a2e4c1783645eb040d48f`
- training record bundle digest:
  `e8efbbedfaba5d4af1de3250c05ab912550e5ff83e1d9ffdfb0d6dcba8b52ede`

This is now the strongest local Qwen candidate for the Harvey smoke route. It
is a resumed LoRA policy refresh from accepted benchmark behavior, not a
retained Harvey score claim or full RL run.

## Harvey MFN Training-Slice Adapter

The strongest local Harvey-task candidate is now:

```text
fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004/adapters.safetensors
```

Report:

```text
fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004/report.json
```

Adapter SHA-256:

```text
59c4dede1354cd9d7166e37acfc097090e8c398e729feef5deb77a94fb25b119
```

Report SHA-256:

```text
138d73c329896906c5ce8dd9d2e2e71aa9a6cb7b107b262f5e44b289442ad363
```

Serve it:

```bash
MODEL_ID=Qwen/Qwen3.5-0.8B \
ADAPTER_PATH=fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004 \
PORT=18090 \
MAX_TOKENS=4096 \
scripts/run-qwen35-08b-legal-mlx-lora-server.sh
```

Run it against the public Harvey MFN training slice:

```bash
QWEN_LEGAL_MLX_BASE_URL=http://127.0.0.1:18090/v1 \
QWEN_LEGAL_ADAPTER_PATH=fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004/adapters.safetensors \
QWEN_LEGAL_ADAPTER_DIGEST=59c4dede1354cd9d7166e37acfc097090e8c398e729feef5deb77a94fb25b119 \
QWEN_LEGAL_ADAPTER_REPORT_DIGEST=138d73c329896906c5ce8dd9d2e2e71aa9a6cb7b107b262f5e44b289442ad363 \
QWEN_LEGAL_RUN_NONCE=qwen35-08b-mlx-lora-harvey-mfn-slice-2026-05-20-final \
cargo run -p psionic-eval --example qwen35_legal_mlx_lora_harvey_mfn_slice -- \
  fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004/harvey_mfn_slice_run
```

Recorded result:

- terminal state: `submitted`
- output artifact count: `1`
- tool receipt count: `2`
- public training-slice criterion-title/token pass count: `8 / 83`
- score report digest:
  `7da1e7559aea3f45466cec8d6c085772ba988b0a10b60b216b5ad1d6000177ad`
- training record bundle digest:
  `842430aae1c7a4f675c5b27eadde9f28197833c0ebe22fd6528b28980fc14888`

This is the first local Qwen LoRA candidate that both trains from prior
accepted benchmark behavior and runs a real Harvey task through the Rust agent
loop. It is still a public training-slice score, not a retained Harvey
leaderboard score.

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
- local MLX-served Qwen adapter can create and submit a tool-backed output
  artifact through the Rust agent loop

This is not a public Harvey score claim.
