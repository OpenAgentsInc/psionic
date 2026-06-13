# Psion General Instruct SFT Lane

> Status: canonical psionic#1117 record, written 2026-06-12 after landing the
> bounded general instruct SFT lane: owned versioned chat template with
> generation masking, masked-loss LoRA training over the reused legal SFT
> machinery, applied warmup/cosine learning-rate schedules, a bit-exact
> checkpoint/resume drill, and a provenance-disciplined example instruct
> corpus.

This document freezes the bounded general instruct SFT lane that extends the
legal SFT lane toward the Psion post-training arc. The lane is a Psion
statistical lane: deterministic fixture-scale evidence only.

## Claim Boundary

- Bounded fixture-scale lane evidence only.
- No general instruct capability claim.
- No real training-run claim. The committed lane report comes from a
  deterministic smoke run over repo-owned example records.
- Naming discipline: this is a Psion statistical lane.

## Canonical Artifacts

- `crates/psionic-train/src/psion_instruct_sft_lane.rs` owns the typed
  contracts: chat template, generation mask, corpus provenance manifest, lane
  config, lane report, and the lane runner.
- `crates/psionic-train/examples/psion_instruct_sft_lane_fixtures.rs`
  regenerates every committed fixture from the typed contracts.
- `fixtures/psion/instruct/psion_chat_template_v1.json` is the owned,
  digest-pinned chat template contract.
- `fixtures/psion/instruct/psion_instruct_example_corpus_v1.jsonl` is the
  clearly labeled repo-owned example instruct corpus.
- `fixtures/psion/instruct/psion_instruct_corpus_manifest_v1.json` is the
  corpus provenance manifest.
- `fixtures/psion/instruct/psion_instruct_generation_mask_fixture_v1.json`
  pins the token-level generation masks for every example record.
- `fixtures/psion/instruct/psion_instruct_sft_lane_report_v1.json` is the
  bounded lane report with the applied learning-rate schedule and the
  checkpoint/resume drill outcome.
- `scripts/check-psion-instruct-sft-lane.sh` regenerates the fixtures and
  fails on any drift or invariant violation.

## Owned Chat Template (`psion.chat_template.v1`)

`PsionChatTemplate::v1()` owns five marker tokens: `<|psion_system|>`,
`<|psion_user|>`, `<|psion_assistant|>`, `<|psion_end|>`, and
`<|psion_eos|>`. The template is digest-pinned: `template_digest` is a stable
SHA-256 over the schema version, identity, and marker set, and `validate()`
refuses any template whose digest no longer matches its contents.

Generation-masking semantics, the part the playbook flags as a silent killer:

- Role-open markers never train, including the assistant role-open marker.
- System and user content never trains.
- Assistant content tokens train.
- The turn-close after assistant content trains; after system or user content
  it does not.
- The final end-of-sequence token trains.
- A conversation that does not end with an assistant turn is refused
  (`MissingFinalAssistantTurn`), so the trainable end-of-sequence token can
  never be silently dropped.
- Content containing any template marker is refused (`RoleBleed`), so user
  text can never unmask itself by impersonating an assistant span.

Adversarial tests in the module pin the exact token-level mask vector,
the missing-eos refusal, role bleed in both user and assistant content, and
the span-boundary off-by-one cases (assistant role-open masked, first content
token trainable, assistant turn-close trainable, next user role-open masked).

## Corpus Provenance Discipline

The instruct corpus enters through the same digest discipline as the
pretraining data: each record carries `source_id`, `source_family_id`,
`rights_posture`, `canonical_reference`, and a pinned `content_digest`,
mirroring the field shapes of
`fixtures/psion/corpus_admission/psion_source_admission_manifest_v1.json`.
The manifest pins the corpus JSONL artifact as `{path, sha256}` exactly like
`PsionActualPretrainingArtifactRef` in the pretraining recipe bundle, plus
per-record mask digests and trainable/masked token counts under the pinned
template digest.

The committed corpus is a clearly labeled example: `example_corpus: true`,
`source_family_id: psion_instruct_example_seed`, and
`rights_posture: repo_owned_example_text`. All record text is authored in
this repository.

## Lane Config And Learning Rate

`PsionInstructSftLaneConfig` (`psion.instruct_sft_lane_config.v1`) reuses the
legal SFT machinery end to end: the deterministic
`OpenAdapterTrainingExecutionBackend`, `TrainingOptimizerConfig` (AdamW or
SGD), `TrainingLoopBudget`, and the LoRA LM-head target, exactly as
`legal_sft_cli.rs` wires them.

The learning-rate default is ten times below the recorded pretraining
reference:

- Pretraining reference: `3e-4`, recorded as the dense-rank optimizer
  `TrainingOptimizerConfig::adamw(0.0003, ...)` in
  `crates/psionic-train/examples/psion_trusted_cluster_run_fixtures.rs` and as
  `effective_learning_rate: 0.0003` in the dense-rank group telemetry of
  `fixtures/psion/trusted_cluster/psion_trusted_cluster_run_bundle_v1.json`.
- Instruct default: `3e-5`
  (`PSION_INSTRUCT_SFT_DEFAULT_LEARNING_RATE`).
- Hard ceiling: configs above `reference / 5` are refused.

## Schedulers (Wired, Not Rebuilt)

The lane reuses the existing `TrainingSchedulerConfig` /
`TrainingSchedulerBinding` machinery from the core loop. The lane runner
attaches the configured scheduler binding to both LoRA parameter groups, so
`apply_group_step` resolves and applies the scheduled learning rate on every
step and records it in the step-receipt group telemetry. The lane report's
`learning_rate_schedule` rows are read back from that applied telemetry, not
recomputed on the side. Default schedule: cosine annealing from `3e-5` to
`3e-6` over the bounded budget; linear warmup is covered by tests.

## Checkpoint/Resume Drill

`FixedBudgetTrainingRun` serializes its complete state: parameters, optimizer
moments, scheduler state, and completed steps. The drill trains to
`checkpoint_at_step`, serializes the run, restores it from those bytes,
finishes the budget, and requires:

- bit-exact final LoRA parameters against the uninterrupted run
  (`resume_bit_exact`), verified for both AdamW and SGD, and
- post-resume step-receipt digests identical to the uninterrupted run.

The checkpoint state digest is recorded in the lane report.

## Masked-Loss Path

`derive_psion_instruct_supervision_samples` converts rendered conversations
into `OpenAdapterHiddenStateSample`s for trainable tokens only. System and
user spans produce no supervision samples, so they contribute nothing to the
loss by construction. Tests assert that the sample set covers exactly the
trainable token positions of every record.

## External Reference

`tetherto/qvac-fabric-llm.cpp` `llama-finetune-lora` is the read-only
external reference for this feature set (assistant-only masked loss,
checkpoint/resume, schedulers, Jinja chat templates) demonstrated on
consumer and mobile GPUs. Study and port ideas only; never vendor. If cloned
locally it lives under `projects/tether/repos/`.

## Verification

- `cargo test -p psionic-train --lib psion_instruct_sft_lane` runs the lane's
  module tests (template adversarial cases, masked-loss coverage, schedule
  application, resume drill, determinism).
- `scripts/check-psion-instruct-sft-lane.sh` regenerates all fixtures and
  fails on drift, mask-invariant violations, learning-rate-ratio drift, or a
  non-bit-exact resume drill.

Authored by Fable (claude-fable-5).
