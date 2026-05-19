# Qwen Legal Adapter Fine-Tune Lane

> Status: implemented smoke lane for `psionic-train` on 2026-05-19.

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
cargo test -p psionic-train qwen_legal_adapter
```

The test fixture runs a four-step deterministic adapter update, exports a
loadable LM-head LoRA artifact, saves a checkpoint, restores from a midpoint
checkpoint, emits an Autopilot4 score-import bundle, and materializes the
next-phase RL hillclimb plan.

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
