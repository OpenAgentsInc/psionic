# Qwen3.6-27B Real Weight Load 001

## Status

- source model: `Qwen/Qwen3.6-27B`
- local model directory: `target/models/qwen/Qwen3.6-27B`
- local directory tracked by git: `false`
- config loaded: `true`
- tokenizer loaded: `true`
- safetensors shards loaded: `15`
- safetensors bytes read: `55563006400`
- index tensor-data bytes: `55562855904`
- prompt tokens: `70`
- output report:
  `target/legal/qwen36_27b_real_weight_load/report.json`
- output report sha256:
  `efa51f06cf0d7d4e182e06ae20b669789107c9684c3d9000bba3063eddb3a8a7`

## Download

The full BF16 checkpoint was downloaded with:

```bash
hf download Qwen/Qwen3.6-27B \
  config.json generation_config.json tokenizer.json tokenizer_config.json \
  model.safetensors.index.json '*.safetensors' \
  --local-dir target/models/qwen/Qwen3.6-27B \
  --max-workers 4
```

The local model directory contains `config.json`, tokenizer files,
`model.safetensors.index.json`, and `model-00001-of-00015.safetensors` through
`model-00015-of-00015.safetensors`.

## Rust Run

The full shard load was run with:

```bash
mkdir -p target/legal/qwen36_27b_real_weight_load
cargo run -p psionic-serve --example qwen36_legal_prompt_smoke -- \
  --model Qwen/Qwen3.6-27B \
  --prompt fixtures/legal/smoke.prompt \
  --model-dir target/models/qwen/Qwen3.6-27B \
  > target/legal/qwen36_27b_real_weight_load/report.json
```

The Rust path now accepts the real Hugging Face config shape, where the
language model config is nested under `text_config`. The loaded language facts
are:

- top-level architecture: `Qwen3_5ForConditionalGeneration`
- text model type: `qwen3_5_text`
- hidden size: `5120`
- intermediate size: `17408`
- layers: `64`
- attention heads: `24`
- key-value heads: `4`
- vocab size: `248320`
- max positions: `262144`
- dtype: `bfloat16`

## Boundary

This is a real checkpoint load, not the old tiny smoke shard. It proves the
Rust code can read the actual Qwen3.6-27B config, tokenizer, index, and all 15
BF16 safetensors shards from disk.

It does not yet run a full Qwen3.6-27B forward pass or train LoRA against live
27B activations. That still requires implementing the Qwen3.6 `qwen3_5_text`
forward path in Psionic, including the linear-attention layers and the extra
MTP weights present in this checkpoint.
