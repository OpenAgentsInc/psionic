# Qwen Replacement Model Conformance

> Status: implemented representative fixture/report layer on 2026-05-19.

This document records the Psionic conformance posture for the Qwen rows
recommended for legal fine-tuning and Harvey benchmark hill climbing.

The code lives in:

- `crates/psionic-models/src/qwen_replacement_conformance.rs`

## Required Rows

Psionic tracks the following replacement set:

- `Qwen3.5-4B`
- `Qwen3.5-9B-Base`
- `Qwen3.5-35B-A3B-Base`
- `Qwen3.6-27B`
- `Qwen3.6-35B-A3B`
- `Qwen3.5-397B-A17B`

The 4B row is the smoke SFT base. The first serious retained-score target is
`Qwen3.6-35B-A3B`. The 397B row is hosted teacher/judge material, not a local
Pylon target.

## Family Admission

Existing Psionic runtime support is `qwen35`. Qwen3.6 rows are not silently
treated as ordinary Qwen3.5 rows. They carry:

```text
accepted_family_label = qwen36_alias_qwen35
family_admission_decision = versioned_qwen36_alias_to_qwen35
```

That is an explicit compatibility decision until real Qwen3.6 artifact metadata
requires a new runtime branch. The conformance tests refuse a Qwen3.6 fixture
that tries to pass with plain `qwen35`.

## Artifact Facts

Each fixture records:

- public model id
- artifact format
- evidence level
- architecture or safetensors config family
- tokenizer/pretokenizer
- tokenizer vocabulary size
- chat/template digest
- context window
- vision posture
- MoE expert facts where applicable
- local Pylon realism
- hosted/Tinker-only posture
- legal role

Current rows are representative or hosted metadata contracts unless a local
artifact has been materialized. They are enough for scheduling, refusal tests,
and operator planning, not enough to claim local serving for hosted large rows.

## Local Smoke

Run the focused model tests:

```bash
cargo test -p psionic-models qwen_replacement
```

The tests prove:

- every required replacement model is represented
- Qwen3.6 rows carry the versioned alias decision
- a Qwen3.6 row cannot silently use plain `qwen35`
- the representative `Qwen3.6-35B-A3B` probe admits to
  `qwen36_alias_qwen35`

## Operator Guidance

Use the rows as follows:

- `Qwen3.5-4B`: first local smoke SFT and base-plus-adapter eval path.
- `Qwen3.5-9B-Base`: small base-model research and tokenizer/template checks.
- `Qwen3.5-35B-A3B-Base`: hosted/Tinker MoE base research.
- `Qwen3.6-27B`: dense retained-score fallback target.
- `Qwen3.6-35B-A3B`: first serious retained-score fine-tune target.
- `Qwen3.5-397B-A17B`: hosted teacher/judge/distillation row.

Before scheduling real training, materialize artifact probes and replace the
placeholder template digests with real metadata. If a Qwen3.6 artifact diverges
from qwen35 tokenizer/template/runtime facts, add a real `qwen36` branch instead
of widening the alias.
