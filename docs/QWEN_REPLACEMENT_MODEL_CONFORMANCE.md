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

`Qwen3.6-27B` also has a concrete target-path smoke now:

```bash
cargo run -p psionic-serve --example qwen36_legal_prompt_smoke -- \
  --model Qwen3.6-27B \
  --prompt fixtures/legal/smoke.prompt
```

That path loads the Qwen3.6-27B config fixture, tokenizer fixture, and a
safetensors shard in Rust, then renders the Qwen3.6 direct-answer legal prompt
and records a receipt. It is a target-path and artifact-loading proof, not full
27B weight inference.

The matching Rust-only adapter SFT smoke is:

```bash
cargo run -p psionic-train -- sft --config configs/legal/qwen36_27b_sft_smoke.json
```

That run loaded the config/tokenizer artifacts, improved smoke loss from
`5.5182295` to `2.2927308`, wrote
`target/legal/qwen36_27b_sft_smoke/adapter.safetensors`, recorded
`python_invoked: false`, and evaluated through the deterministic Harvey
public-three fixture at `10000` adapter bps. This is still public-fixture
proof, not performance on private Harvey tasks.

`Qwen3.6-35B-A3B` now has a MoE-safe target-path smoke:

```bash
cargo run -p psionic-serve --example qwen36_legal_prompt_smoke -- \
  --model Qwen3.6-35B-A3B \
  --prompt fixtures/legal/smoke.prompt
```

The matching SFT smoke is:

```bash
cargo run -p psionic-train -- sft --config configs/legal/qwen36_35b_a3b_sft_smoke.json
```

That run loads a `Qwen3MoeForCausalLM` config fixture, tokenizer fixture, and
expert safetensors shard; allows LoRA only on `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `up_proj`, and `down_proj`; refuses router/gate targets; records
active experts and usage counts; and proves the router hash is unchanged before
and after the adapter update. The recorded smoke improved loss from
`5.5602503` to `1.5024384`, wrote
`target/legal/qwen36_35b_a3b_sft_smoke/adapter.safetensors`, and scored
`10000` adapter bps on the deterministic Harvey public-three fixture. This is
still a small Rust smoke, not full 35B-A3B weight inference or performance on
private Harvey tasks.

## Operator Guidance

Use the rows as follows:

- `Qwen3.5-4B`: first local smoke SFT and base-plus-adapter eval path.
- `Qwen3.5-9B-Base`: small base-model research and tokenizer/template checks.
- `Qwen3.5-35B-A3B-Base`: hosted/Tinker MoE base research.
- `Qwen3.6-27B`: dense retained-score fallback target; now has a Rust config,
  tokenizer, safetensors, prompt-render, SFT, and public-fixture eval smoke.
- `Qwen3.6-35B-A3B`: first serious retained-score fine-tune target; now has a
  Rust MoE-safe config, expert-shard, prompt-render, adapter-SFT, frozen-router
  safety receipt, and public-fixture eval smoke.
- `Qwen3.5-397B-A17B`: hosted teacher/judge/distillation row.

Before scheduling real training, materialize artifact probes and replace the
placeholder template digests with real metadata. If a Qwen3.6 artifact diverges
from qwen35 tokenizer/template/runtime facts, add a real `qwen36` branch instead
of widening the alias.
