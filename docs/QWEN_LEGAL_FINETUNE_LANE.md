# Qwen Legal Adapter Fine-Tune Lane

> Status: implemented smoke lane for `psionic-train` on 2026-05-19; real
> local Qwen/MLX LoRA plus Rust legal-agent smoke added on 2026-05-20.

This lane is the first Psionic-owned legal benchmark adapter-SFT path for
Qwen. It starts with `Qwen/Qwen3.5-4B` only to prove the wiring:

- legal benchmark training records from `docs/LEGAL_BENCHMARK_TRAINING_RECORDS.md`
- deterministic hidden-state supervision
- `psionic-train` adapter training through the shared open-adapter backend
- exact checkpoint save/restore
- `safetensors` LM-head LoRA export
- eval-pack and score-import metadata for Autopilot4

It does not claim a Harvey retained score lift. The serious retained attempt
should move to `Qwen3.6-35B-A3B` after this lane, the Qwen model-admission
update, and the Autopilot4 import loop are green.

The Qwen replacement model set and the current `qwen36_alias_qwen35`
conformance decision live in `docs/QWEN_REPLACEMENT_MODEL_CONFORMANCE.md`.

## Current Real-Weight Result

On 2026-05-20, the lane gained a material local Qwen-family LoRA result in
addition to the deterministic smoke trainer:

- run id: `qwen_legal_real_qwen35_08b_mlx_lora_2026_05_20_002`
- base model: `Qwen/Qwen3.5-0.8B`
- Hugging Face revision:
  `2fc06364715b967f1860aea9cf38778875588b17`
- backend: `mlx_lm.lora`
- logical Pylon worker: `pylon.local.macos.mlx.01`
- data:
  `fixtures/qwen_legal/real_finetune/mlx_lora_seed`
- adapter:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002/adapters.safetensors`
- report:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002/report.json`
- report digest:
  `b9c3c9dac55c469be1e946c9ea2e7be9255dfa2f02a097d31df97bf9d64592d5`
- checker:
  `scripts/check-qwen35-08b-legal-mlx-lora-fixture.sh`

The run trained six LoRA iterations over the public-safe legal seed set.
Validation loss moved from `3.595` to `3.223`, with `1.804M` trainable
parameters and peak observed memory of `4.251 GB`. The adapter loaded through
`mlx_lm.generate` and through the MLX HTTP server on
`POST /v1/chat/completions`.

This is the first real Qwen-family adapter artifact for the Harvey legal lane,
but it is still not the retained model target. It is single local-worker SFT,
not RL, not live Nexus settlement, and not a retained Harvey score claim.

Serve it locally for Harvey smoke work with:

```bash
scripts/run-qwen35-08b-legal-mlx-lora-server.sh
```

Use this OpenAI-compatible route while the server is running:

```text
base_url: http://127.0.0.1:18088/v1
model: Qwen/Qwen3.5-0.8B
```

The request model must be the real base model id. The current MLX server tries
to resolve arbitrary aliases as Hugging Face repos.

The current adapter has also passed the Rust legal benchmark agent smoke:

```bash
scripts/run-qwen35-08b-legal-mlx-lora-harvey-smoke.sh
```

Recorded result:

- run id:
  `run.legal.qwen35_08b_mlx_lora.harvey_tool_smoke.f2972e6fead2.qwen35-08b-mlx-lora-2026-05-20`
- terminal state: `submitted`
- output artifact count: `1`
- tool receipt count: `1`
- generated deliverable:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002/harvey_agent_smoke/output/outputs/memo.md`
- run record hash:
  `3463444d89f01b57a7d25304cce0a3033665fa01e8ed3c130613db456fd026db`
- smoke report digest:
  `13c28f8ff6f3e8fad7b81b537947dfa029295449aa913b8c0b57800be76d90c9`
- deterministic score report digest:
  `610eba2cc13ad7a16069d60eee9dbfa95f829ca1a4dfa20bb45108d5f004ac2d`
- training record bundle digest:
  `db28588382457abf216b31e00d6875c0525e026a706c3448e78e26c6497e74e3`

This is now a real local Qwen LoRA plus a receipt-backed Rust benchmark-agent
trajectory. The fixture shares the workspace and output roots only for this
local smoke because the small Qwen adapter selects the `workspace` root even
when instructed to use `output`. Production Harvey tasks should keep separate
workspace and output roots and tune the tool-use policy on retained slices.
The result exports one canonical legal benchmark training record and is usable
as a seed trajectory for legal RL ingestion, but it is not itself an RL-trained
model update or a retained Harvey score claim.

## Rust API

The implementation lives in:

- `crates/psionic-train/src/qwen_legal_adapter_sft.rs`
- `crates/psionic-train/src/open_adapter.rs`
- `crates/psionic-train/src/train_runtime.rs`

The canonical lane id is:

```text
qwen_legal_adapter_sft_v1
```

The first target set is deliberately narrow:

```text
qwen3.5-4b.legal.lm_head_lora.v1
```

Only `lm_head` LoRA is admitted in the smoke. Attention and MLP LoRA targets
should be added after the benchmark record, checkpoint, export, and
score-import loop is proven end to end.

## Data Contract

The dataset binding must point at a `LegalBenchmarkTrainingRecordBundle` whose
record schema version is:

```text
psionic.legal_benchmark_training_record.v1
```

The trainer refuses dataset drift before training. Hidden benchmark criteria
must remain excluded from model-visible examples by the record exporter.

The run emits:

- base model id and served model id
- base artifact digest
- tokenizer contract digest
- prompt-template digest
- dataset digest
- eval-pack digest
- adapter artifact digest
- adapter identity digest
- final checkpoint id
- score-import bundle digest
- RL hillclimb plan digest
- RL benchmark readiness report digest
- RL optimization window report digest
- RL perfect-score push report digest

## Artifact Gate

The unit smoke does not require full Qwen weights. It uses
`SyntheticHiddenStateSmoke` and the explicit synthetic digest:

```text
sha256:synthetic-qwen35-4b-legal-smoke
```

Real-model execution must use `RealArtifactRequired`, a non-synthetic base
artifact digest, and an explicit materialized artifact path. The binding offers
`validate_real_artifact_materialized()` for local launch preflight.

This prevents silently swapping in another Qwen row or falling back from a real
model run to the synthetic smoke fixture.

## Local Smoke

Run the focused tests from the repo root:

```bash
cargo test -p psionic-train --lib qwen_legal
cargo test -p psionic-train --lib live_rl_update
```

The `qwen_legal` fixture runs a four-step deterministic adapter update,
exports a loadable LM-head LoRA artifact, saves a checkpoint, restores from a
midpoint checkpoint, emits an Autopilot4 score-import bundle, and materializes
the next-phase RL hillclimb plan plus local benchmark reports. The
`live_rl_update` fixture materializes rollout evidence and promotes a new
revision only when teacher-logprob alignment is valid.

The 2026-05-20 local run executed the broader filters with binary targets
included:

- `cargo test -p psionic-train qwen_legal`: 14 passed
- `cargo test -p psionic-train live_rl_update`: 2 passed

Those are still local training/RL substrate tests. They are not retained Harvey
score claims.

## RL Hillclimb Plan

The smoke lane now emits
`QwenLegalRlHillclimbPlan` with schema
`psionic.qwen_legal_rl_hillclimb_plan.v1`. This does not claim a retained
Harvey score, but it gives Pylon/Nexus a concrete plan for the next phase:

- retain `Qwen/Qwen3.5-4B` as the smallest smoke lane while targeting
  `Qwen/Qwen3.6-35B-A3B` for retained scoring
- require a retained 20-task Harvey slice before any public retained-score
  claim
- collect at least 60 accepted legal-agent rollouts with no more than 12
  quarantined rollouts in the window
- connect the run to
  `blueprint://harvey_legal_qwen_optimizer_frontier/optimizer_frontier_001`
- assign failure families to GRPO, GEPA trace selection, MIPRO prompt search,
  and supervised fine-tune refresh work

The current target families are:

| Failure family | Method | Blueprint module |
| --- | --- | --- |
| `document_coverage` | MIPRO prompt search | `harvey_legal.document_inventory` |
| `citation_evidence` | GEPA trace selection | `harvey_legal.evidence_mapping` |
| `legal_reasoning` | GRPO | `harvey_legal.issue_fact_extraction` |
| `spreadsheet_reasoning` | GRPO | `harvey_legal.evidence_mapping` |
| `missing_fact` | supervised fine-tune refresh | `harvey_legal.issue_fact_extraction` |
| `pre_submit_self_check` | GEPA trace selection | `harvey_legal.final_self_check` |

Each target carries a reward signal, baseline miss value, target lift value,
and dataset request ref. Operators should treat the plan as a routing and
admission contract for Pylon work, not as proof that RL training has already
improved the retained score.

## Offline RL Benchmark Report

The lane now also emits `QwenLegalRlBenchmarkReadinessReport` with schema
`psionic.qwen_legal_rl_benchmark_report.v1`. It turns the plan into the local
benchmark numbers Autopilot4 can publish safely:

- baseline retained slice: 5260 bps
- phase-two conservative target: 7000 bps
- unconstrained projection if all family lifts land: above 9000 bps
- minimum accepted rollouts: 60
- quarantine budget: 12
- required method mix: GRPO, GEPA trace selection, MIPRO prompt search, and
  supervised fine-tune refresh
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-002`

This report is still local readiness evidence. A retained score claim requires
the actual retained 20-task run, immutable score reports, and Autopilot4
release-gate approval.

## Phase-Three RL Optimization Window

The lane also emits `QwenLegalRlOptimizationWindowReport` with schema
`psionic.qwen_legal_rl_optimization_window.v1`. It consumes the phase-two
readiness report and the Blueprint shadow-eval shortlist:

- phase-two target carried forward: 7000 bps
- phase-three conservative target: 7800 bps
- accepted rollout minimum: 84
- quarantine budget: 16
- holdout regression allowance: 0 bps
- Blueprint shortlist ref:
  `blueprint://harvey_legal_qwen_phase_three_shadow_eval_shortlist/optimizer_shortlist.harvey_legal_qwen.phase_003.shadow_eval`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-003`

This is the next work packet for Pylon/Nexus. It is still not a live Harvey
score claim; it is the bounded local benchmark target and evidence contract
that the retained run must satisfy.

## Phase-Four Perfect-Score Push

The lane now also emits `QwenLegalRlPerfectScorePushReport` with schema
`psionic.qwen_legal_rl_perfect_score_push.v1`. It consumes the phase-three
optimization window and the Blueprint perfect-score push plan:

- phase-three target carried forward: 7800 bps
- phase-four conservative target: 8500 bps
- accepted rollout minimum: 140
- quarantine budget: 20
- holdout regression allowance: 0 bps
- calibrated judge disagreement budget: 75 bps
- family coverage: all nine Blueprint optimizer frontier families
- Blueprint plan ref:
  `blueprint://harvey_legal_qwen_phase_four_perfect_score_push_plan/optimizer_plan.harvey_legal_qwen.phase_004.perfect_score_push`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-004`

This is the next bounded work packet before any perfect-score campaign. It adds
deliverable completeness, fine-tune data selection, and task-intake routing to
the phase-three RL window while preserving judge-adjudication and scorecard
requirements for every family.

## Phase-Five Retained Rehearsal

The lane now also emits `QwenLegalRlRetainedRehearsalReport` with schema
`psionic.qwen_legal_rl_retained_rehearsal.v1`. It consumes the phase-four
perfect-score push report and the Blueprint retained rehearsal plan:

- phase-four target carried forward: 8500 bps
- phase-five conservative target: 9000 bps
- retained rehearsal task-runs: 60
- accepted rollout minimum: 194
- quarantine budget: 24
- adversarial holdout task-runs: 36
- holdout regression allowance: 0 bps
- calibrated judge disagreement budget: 50 bps
- family coverage: all nine Blueprint optimizer frontier families
- Blueprint plan ref:
  `blueprint://harvey_legal_qwen_phase_five_retained_rehearsal_plan/optimizer_plan.harvey_legal_qwen.phase_005.retained_rehearsal`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-005`

This is the first high-confidence rehearsal gate after the 8500 bps target. It
requires three retained-slice passes, an adversarial holdout, and a tighter
judge panel before Autopilot4 should import a candidate as ready for a public
retained campaign.

## Phase-Six Expanded Corpus Dry Run

The lane now also emits `QwenLegalRlExpandedCorpusReport` with schema
`psionic.qwen_legal_rl_expanded_corpus.v1`. It consumes the phase-five
retained rehearsal report and the Blueprint expanded corpus plan:

- phase-five target carried forward: 9000 bps
- phase-six conservative target: 9500 bps
- expanded stratified slice: 125 Harvey tasks
- practice-area coverage: all 24 audited practice areas
- accepted rollout minimum: 266
- quarantine budget: 30
- adversarial holdout task-runs: 72
- holdout regression allowance: 0 bps
- calibrated judge disagreement budget: 35 bps
- family coverage: all nine Blueprint optimizer frontier families
- Blueprint plan ref:
  `blueprint://harvey_legal_qwen_phase_six_expanded_corpus_plan/optimizer_plan.harvey_legal_qwen.phase_006.expanded_corpus`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-006`

This dry run expands beyond the retained 20-task slice before any all-task
campaign. It is readiness evidence for a broader benchmark run, not a retained
score claim.

## Phase-Seven Full-Corpus Matrix Dry Run

The lane now also emits `QwenLegalRlFullCorpusMatrixReport` with schema
`psionic.qwen_legal_rl_full_corpus_matrix.v1`. It consumes the phase-six
expanded corpus report and the Blueprint full-corpus matrix plan:

- phase-six target carried forward: 9500 bps
- phase-seven conservative target: 9800 bps
- full corpus: 1251 Harvey tasks
- practice-area coverage: all 24 audited practice areas
- Qwen/Blueprint/RL matrix cells: 48
- accepted rollout minimum: 410
- quarantine budget: 38
- adversarial holdout task-runs: 144
- holdout regression allowance: 0 bps
- calibrated judge disagreement budget: 25 bps
- family coverage: all nine Blueprint optimizer frontier families
- Blueprint plan ref:
  `blueprint://harvey_legal_qwen_phase_seven_full_corpus_matrix_plan/optimizer_plan.harvey_legal_qwen.phase_007.full_corpus_matrix`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-007`

This dry run is the first local matrix contract over the full Harvey corpus. It
is still readiness evidence only; public score claims require imported score
reports and release-gate approval.

## Phase-Eight Residual Burn-Down Dry Run

The lane now also emits `QwenLegalRlResidualBurnDownReport` with schema
`psionic.qwen_legal_rl_residual_burn_down.v1`. It consumes the phase-seven
full-corpus matrix report and the Blueprint residual burn-down plan:

- phase-seven target carried forward: 9800 bps
- phase-eight conservative target: 9900 bps
- full corpus: 1251 Harvey tasks
- practice-area coverage: all 24 audited practice areas
- Qwen/Blueprint/RL matrix cells: 96
- residual miss cluster budget: 24
- accepted rollout minimum: 626
- quarantine budget: 48
- adversarial holdout task-runs: 288
- holdout regression allowance: 0 bps
- calibrated judge disagreement budget: 15 bps
- family coverage: all nine Blueprint optimizer frontier families
- Blueprint plan ref:
  `blueprint://harvey_legal_qwen_phase_eight_residual_burn_down_plan/optimizer_plan.harvey_legal_qwen.phase_008.residual_burn_down`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-008`

This dry run narrows the final residual error budget before any perfect-score
push. It is still readiness evidence only; public score claims require imported
score reports and release-gate approval.

## Phase-Nine Final-Campaign Rehearsal Dry Run

The lane now also emits `QwenLegalRlFinalCampaignReport` with schema
`psionic.qwen_legal_rl_final_campaign.v1`. It consumes the phase-eight residual
burn-down report and the Blueprint final-campaign rehearsal plan:

- phase-eight target carried forward: 9900 bps
- phase-nine conservative target: 9950 bps
- full corpus: 1251 Harvey tasks
- practice-area coverage: all 24 audited practice areas
- Qwen/Blueprint/RL matrix cells: 144
- residual miss cluster budget: 12
- human-adjudicated sample task-runs: 96
- accepted rollout minimum: 950
- quarantine budget: 52
- adversarial holdout task-runs: 432
- holdout regression allowance: 0 bps
- calibrated judge disagreement budget: 10 bps
- family coverage: all nine Blueprint optimizer frontier families
- Blueprint plan ref:
  `blueprint://harvey_legal_qwen_phase_nine_final_campaign_plan/optimizer_plan.harvey_legal_qwen.phase_009.final_campaign_rehearsal`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-009`

This dry run is the last rehearsal before a separate perfect-score campaign.
It is still readiness evidence only; public score claims require imported score
reports and release-gate approval.

## Pylon Network SFT Smoke

The lane now has a first network-shaped Qwen legal training result:
`qwen_legal_pylon_network_sft_v1`.

This is not the final Qwen3.6 fine-tune. It is the smallest honest replacement
for the CS336-first proof path:

- objective id: `harvey_legal_qwen_finetune_v1`
- parent lane id: `qwen_legal_adapter_sft_v1`
- base model binding: `Qwen/Qwen3.5-4B`
- retained target model: `Qwen/Qwen3.6-35B-A3B`
- artifact mode: `synthetic_hidden_state_smoke`
- contributors: two logical Pylon workers
- aggregation rule: `trusted_weighted_lora_factor_average_v1`
- aggregate artifact:
  `fixtures/qwen_legal/pylon_network_sft/aggregate-qwen-legal-lm-head-lora.safetensors`
- report:
  `fixtures/qwen_legal/pylon_network_sft/pylon_network_sft_report_v1.json`
- generator:
  `crates/psionic-train/examples/qwen_legal_pylon_network_sft_fixture.rs`
- checker:
  `scripts/check-qwen-legal-pylon-network-sft.sh`

The retained report records per-contributor assignment ids, worker ids, node
pubkeys, shard refs, sample ids, training losses, checkpoint digests, adapter
artifact digests, contribution receipt digests, aggregate receipt digest,
model-progress participant count, and aggregate adapter identity digest.

The aggregate adapter is loadable as an LM-head LoRA safetensors artifact. The
two contributors train different legal smoke shards and produce different
adapter digests before trusted aggregation. That gives the legal lane a real
multi-contributor trained artifact while keeping the claim boundary explicit:
it proves Pylon-network training shape and aggregation, not real Qwen3.6
full-weight fine-tuning or retained Harvey score lift.

Run it locally with:

```bash
scripts/check-qwen-legal-pylon-network-sft.sh
```

## Runtime Admission

`train_runtime.rs` admits the lane as a CUDA adapter-training machine lane:

```text
release_id: psionic-train.qwen_legal_adapter_sft.release.v1
environment_ref: psionic.environment.qwen_legal_adapter_sft.cuda.operator@v1
backend_family: cuda
topology_class: single_host_cuda_adapter_smoke
minimum_machine_class: strong_cuda_trainer
```

Pylon/Nexus dispatch should use this lane only for smoke jobs until the real
artifact gate and retained eval import are wired into the operator path.
