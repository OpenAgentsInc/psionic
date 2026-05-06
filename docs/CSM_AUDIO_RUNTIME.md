# CSM Audio Runtime

Status: partial

This document tracks the Psionic-owned CSM speech-generation lane for Lyra.
CSM is a contextual speech generator. It is not the Lyra conversation runtime,
STT engine, LLM, transport, or product authority layer.

The current implementation state is phase 0:

- Psionic has a committed Python-reference parity corpus at
  `fixtures/csm/python_reference/csm_python_parity_v1.json`.
- `psionic-models` validates that fixture through
  `csm_python_parity_fixture()` and `validate_csm_python_parity_fixture(...)`.
- The fixture freezes prompt audio hashes, Llama tokenizer examples, 33-lane
  CSM text-frame masks, compact Mimi prompt-codebook prefixes, and a
  three-frame greedy generated-codebook prefix.
- The local Python repo at `/Users/christopherdavid/code/csm` remains a
  reference harness and parity source only. It is not a production Psionic
  runtime and it is not embedded in Lyra.

## Fixture Source

The frozen corpus was derived from the local CSM demo described in:

- `/Users/christopherdavid/code/csm/DEMO_RUN.md`
- root audit:
  `/Users/christopherdavid/work/docs/2026-05-06-csm-rust-lyra-psionic-audit.md`

The reference demo command is:

```bash
NO_TORCH_COMPILE=1 .venv/bin/python run_csm.py
```

The fixture records no Hugging Face token, provider key, full prompt audio,
full model weights, or full codebook tensors.

## Current Fixture Contents

The fixture binds:

- CSM repo: `sesame/csm-1b`
- Llama tokenizer repo: `meta-llama/Llama-3.2-1B`
- Mimi repo: `kyutai/moshiko-pytorch-bf16`
- Mimi weight: `tokenizer-e351c8d8-checkpoint125.safetensors`
- prompt profiles: `conversational_a`, `conversational_b`
- source prompt WAVs: 44.1 kHz mono, 30 seconds each
- CSM runtime sample rate: 24 kHz
- CSM text/audio frame width: 33 lanes
- CSM audio codebook count: 32
- deterministic generation prefix sampling: `greedy_argmax_topk1`

## Validation

Run the focused fixture validation with:

```bash
cargo test -p psionic-models csm_python_parity_fixture
```

The validator checks:

- fixture schema and artifact digest shapes
- required prompt profile ids
- prompt WAV metadata
- tokenizer frame dimensions and text-lane mask semantics
- Mimi codebook prefix dimensions and token bounds
- deterministic generation frame dimensions and token bounds
- explicit secret-redaction markers

## Next Phases

The phase sequence lives in GitHub under `OpenAgentsInc/psionic#959`.

Next work:

1. Add a Psionic speech API and honest Python-reference CSM service mode.
2. Implement Rust tokenizer, prompt framing, and artifact descriptors.
3. Implement Mimi decode and approved voice-profile codebook support.
4. Implement CPU CSM generation with parity tests.
5. Add accelerated serving, residency/refusal truth, and streaming chunks.

Cartesia remains Lyra's production TTS provider until CSM has measured quality,
latency, approved voice-profile governance, and watermark posture.

