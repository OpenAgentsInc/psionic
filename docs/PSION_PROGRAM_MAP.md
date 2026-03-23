# Psion Program Map

> Status: canonical `PSION-1` / `#357` learned-model program-map contract,
> written.

## Why This Doc Exists

The `Psion` learned-model lane now has canonical docs for governance, data,
training, evaluation, serving, rollback, and bounded decentralized follow-on
work. This doc is the umbrella map that ties those contracts together in one
dependency-ordered public-safe program.

It exists so the repo can point at one durable, reviewable learned-lane map
instead of asking readers to reconstruct the program from scattered issue
titles or doc filenames.

## Claim Boundary

This tranche is for a learned bounded-success systems model with explicit route
and refusal behavior.

It is not:

- CPU-reference exactness
- compiled exact execution
- proof-backed verification
- implicit execution hidden behind language answers

Exact executor closure remains separate work under the `Tassadar` lane and its
canonical docs.

The current reference pilot is also a CPU-reference lane. It remains a valid
bounded operator, receipt, checkpoint, and promotion fixture, but it is not a
valid GPU training proof target. Future Google GPU audits must name the exact
trainer command and execution backend explicitly instead of treating
GPU-hosted reference runs as accelerator-backed training proof.

## Dependency Order

The `Psion` program is dependency-ordered in four tracks:

1. governance critical path: `PSION-2`, `PSION-3`, `PSION-4`, `PSION-5`,
   `PSION-6`
2. data/tokenizer critical path: `PSION-7`, `PSION-8`, `PSION-9`, `PSION-10`
3. model/training critical path: `PSION-11`, `PSION-12`, `PSION-13`,
   `PSION-14`, `PSION-15`, `PSION-16`, `PSION-17`, `PSION-18`
4. eval/route/serve critical path: `PSION-19`, `PSION-20`, `PSION-21`,
   `PSION-22`, `PSION-23`, `PSION-24`, `PSION-25`, `PSION-26`, `PSION-27`,
   `PSION-28`, `PSION-29`, `PSION-30`

The ordering matters because later work must bind back to earlier governance,
acceptance, capability, and rollback truth instead of silently widening claims.

## Canonical Track Map

### Program / Governance

- `PSION-2` / `#358`: `docs/PSION_CORPUS_ADMISSION.md`
- `PSION-3` / `#359`: `docs/PSION_SOURCE_LIFECYCLE.md`
- `PSION-4` / `#360`: `docs/PSION_BENCHMARK_ISOLATION.md`
- `PSION-5` / `#361`: `docs/PSION_ACCEPTANCE_MATRIX.md`
- `PSION-6` / `#362`: `docs/PSION_CAPABILITY_MATRIX.md`

### Data / Tokenizer

- `PSION-7` / `#363`: `docs/PSION_RAW_SOURCE_INGESTION.md`
- `PSION-8` / `#364`: `docs/PSION_TOKENIZER_TRAINING.md`
- `PSION-9` / `#365`: `docs/PSION_TOKENIZED_CORPUS.md`
- `PSION-10` / `#366`: `docs/PSION_SAMPLING_POLICY.md`

### Model / Training

- `PSION-11` / `#367`: `docs/PSION_COMPACT_DECODER.md`
- `PSION-12` / `#368`: `docs/PSION_PRETRAIN_STAGE.md`
- `PSION-13` / `#369`: `docs/PSION_RUN_OBSERVABILITY.md`
- `PSION-14` / `#370`: `docs/PSION_PILOT_PRETRAINING_RUN.md`
- `PSION-15` / `#371`: `docs/PSION_CHECKPOINT_RECOVERY.md`
- `PSION-16` / `#372`: `docs/PSION_RENTED_CLUSTER_RUNBOOK.md`
- `PSION-17` / `#373`: `docs/PSION_TRUSTED_CLUSTER_RUN.md`
- `PSION-18` / `#374`: `docs/PSION_REASONING_SFT.md`

### Eval / Route / Serve

- `PSION-19` / `#375`: `docs/PSION_BENCHMARK_PACKAGES.md`
- `PSION-20` / `#376`: `docs/PSION_BENCHMARK_LABEL_GENERATION.md`
- `PSION-21` / `#377`: `docs/PSION_ARCHITECTURE_REASONING_BENCHMARK.md`
- `PSION-22` / `#378`: `docs/PSION_NORMATIVE_SPEC_READING_BENCHMARK.md`
- `PSION-23` / `#379`:
  `docs/PSION_ENGINEERING_SPEC_INTERPRETATION_BENCHMARK.md`
- `PSION-24` / `#380`: `docs/PSION_MEMORIZATION_VS_REASONING_PROBES.md`
- `PSION-25` / `#381`: `docs/PSION_ROUTE_CLASS_EVALUATION.md`
- `PSION-26` / `#382`: `docs/PSION_REFUSAL_CALIBRATION.md`
- `PSION-27` / `#383`: `docs/PSION_SERVED_EVIDENCE.md`
- `PSION-28` / `#384`: `docs/PSION_SERVED_OUTPUT_CLAIMS.md`
- `PSION-29` / `#385`: `docs/PSION_CAPABILITY_WITHDRAWAL.md`
- `PSION-30` / `#386`: `docs/PSION_DECENTRALIZED_CONTRIBUTION.md`

## Operating Rules

Every later `Psion` artifact must preserve the following:

- governance and rights state stay explicit and versioned
- held-out and contamination boundaries remain reviewable
- promotion stays bound to `docs/PSION_ACCEPTANCE_MATRIX.md`
- publication and refusal posture stay bound to
  `docs/PSION_CAPABILITY_MATRIX.md`
- rollback and downgrade history stay bound to
  `docs/PSION_CAPABILITY_WITHDRAWAL.md`
- CPU-reference lanes may not be used as the primary proof target for Google
  GPU training claims
- future GPU audits must state the trainer path and delivered execution backend
  explicitly

No part of this map authorizes widening the learned lane into hidden executor
claims or using decentralized participation as a shortcut around evaluation,
rollback, or provenance discipline.
