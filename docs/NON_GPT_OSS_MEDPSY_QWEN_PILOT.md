# Non-GPT-OSS MedPsy Qwen3 Pilot

> Status: `implemented_early` for artifact admission, tokenizer/prompt fixture,
> quantization policy, medical safety metadata, and the first Rust-native
> Qwen3 BF16 safetensors plus GGUF CPU load/generate paths. A feature-gated
> Candle CUDA backend is available for the same direct benchmark harness on
> NVIDIA hosts with `--features medpsy-cuda`. OpenAI-compatible
> serving now publishes MedPsy medical policy metadata and refuses obvious
> diagnosis/prescribing/emergency-triage prompts when a MedPsy model is loaded.
> A local direct MedPsy benchmark harness exists; comparator benchmark
> publication has one retained partial matrix with a llama.cpp timeout blocker.

This document records the bounded Psionic lane for QVAC MedPsy support.

The lane targets Rust-native Psionic support for:

- `qvac/MedPsy-1.7B`
- `qvac/MedPsy-4B`
- `qvac/MedPsy-1.7B-GGUF`
- `qvac/MedPsy-4B-GGUF`

This lane does not use QVAC SDK, `llama.cpp`, vLLM, Transformers, Python, or a
C++ sidecar as Psionic runtime substrate. Those runtimes are comparator or
reference paths for benchmark work only.

## Source Facts

Primary public references:

- `https://huggingface.co/blog/qvac/medpsy`
- `https://huggingface.co/qvac/MedPsy-1.7B`
- `https://huggingface.co/qvac/MedPsy-4B`
- `https://huggingface.co/qvac/MedPsy-1.7B-GGUF`
- `https://huggingface.co/qvac/MedPsy-4B-GGUF`

`MedPsy-1.7B` source facts:

```text
architecture: Qwen3ForCausalLM
base: Qwen3-1.7B thinking mode
hidden_size: 2048
ffn_hidden_size: 6144
layers: 28
attention_heads: 16
kv_groups: 8
vocab_size: 151936
max_position_embeddings: 40960
rope_theta: 1000000
precision: bfloat16
```

`MedPsy-4B` source facts:

```text
architecture: Qwen3ForCausalLM
base: Qwen3-4B-Thinking-2507
hidden_size: 2560
ffn_hidden_size: 9728
layers: 36
attention_heads: 32
kv_groups: 8
vocab_size: 151936
max_position_embeddings: 262144
rope_theta: 5000000
precision: bfloat16
```

Tokenizer and prompt facts:

- tokenizer class: `Qwen2Tokenizer`
- model architecture: `Qwen3ForCausalLM`
- prompt persona: `You are MedPsy, a medical and healthcare AI assistant developed by QVAC.`
- default template digest from published tokenizer config:
  `8d51e8f9694b24924c7795050ecb7a605fcbd0d7980b40c56ad3e0561d465de7`

## Current Landed Scope

The first landed scope is metadata and admission only:

- `GgufDecoderFamily::Qwen3` is a distinct family label.
- `GgufTokenizerPretokenizer::Qwen3` is admitted when a GGUF declares that
  pretokenizer explicitly.
- The published MedPsy Qwen3 template digest maps to the Qwen3 prompt family.
- Golden tokenizer and prompt fixtures exist under `medpsy_qwen3`.
- `MedPsyModelSupportDescriptor` records the two public size rows.
- `MedPsyMedicalSafetyPolicy` records the baseline medical safety posture.
- `medpsy_quantization_admission` records the default medical-domain
  quantization policy.
- `MedPsyQwen3CandleGenerator` can load BF16 safetensors through
  Rust-native Candle Qwen3 and run greedy token-id generation on CPU when the
  operator supplies a local `PSIONIC_MEDPSY_17B_SAFETENSORS_PATH` artifact.
- `MedPsyQwen3GgufGenerator` can load GGUF MedPsy Qwen3 artifacts through
  Rust-native Candle quantized Qwen3 and run greedy token-id generation on CPU
  when the operator supplies a local `PSIONIC_MEDPSY_17B_Q4_K_M_GGUF_PATH`
  artifact.
- The generic OpenAI-compatible model card can publish `psionic_medical_policy`
  for MedPsy/Qwen3 rows, and the chat completions path refuses direct diagnosis,
  prescribing, dosage, and emergency-triage prompt shapes under
  `medical_model_use.medpsy.v1`.
- `crates/psionic-models/examples/medpsy_bench.rs` records local direct runtime
  benchmark JSON for BF16 safetensors or GGUF artifacts, and
  `scripts/release/run-medpsy-local-bench.sh` writes retained manual reports
  under `fixtures/medpsy/benchmarks/manual/` when local artifacts are supplied.
- `fixtures/medpsy/benchmarks/medpsy_comparator_matrix_20260511_local.json`
  records one completed Psionic CPU row, one completed Psionic CUDA row on the
  Tailnet `archlinux` RTX 4080, and one completed Ollama llama.cpp-runner row on
  the same host. The current CUDA row is below the comparator and does not prove
  competitive parity.
- `fixtures/medpsy/capability/medpsy_capability_matrix_v1.json` publishes the
  bounded capability/refusal envelope and ties the lane to
  `scripts/release/check-psionic-medpsy-pilot.sh`.

This landed scope claims BF16 safetensors and GGUF CPU execution paths when the
operator supplies local artifacts, exposes a feature-gated direct CUDA benchmark
path, plus model-card medical policy publication and first-pass chat safety
refusals on the OpenAI-compatible surface. It does not claim CUDA parity,
Metal acceleration, or benchmark parity against Tether-recommended runtime paths
yet.

## Quantization Policy

The default medical-domain policy is:

```text
preferred: BF16 | Q8_0 | Q5_K_M | Q4_K_M
allowed_with_warning: IQ4_NL | IQ4_XS | 4B IQ3_M
blocked_by_default: 1.7B IQ3_M | any IQ3_XXS
requires_explicit_override: any 3-bit medical route
requires_receipt_warning: all quantized medical routes
```

This policy follows the MedPsy model cards. Lower-bit aggregate benchmark deltas
do not prove safety on rare medical cases. Blocked variants must remain refused
until a dedicated benchmark and safety gate admits them.

## Medical Safety Policy

MedPsy is a medical-domain model. Psionic must not present it as clinical
authority.

Baseline policy:

```text
default_classification: medical_information_not_diagnosis
disclaimer_required: true
emergency_referral_required: true
direct_diagnosis_allowed: false
prescribing_or_treatment_authority_allowed: false
human_review_required_for_clinical_workflows: true
source_citation_required_for_claims: true
quantization_warning_required: true
```

The model cards state that MedPsy is not a substitute for professional medical
judgment, is English-only, is text-only, has no real-time medical knowledge, and
can hallucinate. Psionic serving must preserve those boundaries.

## Remaining Work

The implementation sequence is tracked in GitHub issues:

- `#977` artifact admission, tokenizer fixtures, and medical safety policy
- `#978` Rust-native Qwen3 BF16 safetensors load/generate
- `#979` MedPsy GGUF admission and quantized Rust inference
- `#980` OpenAI-compatible serving with metadata and refusals
- `#981` MedPsy benchmark and medical-safety eval harness
- `#982` benchmark Psionic against Tether-recommended runtimes
- `#983` publish the final capability envelope, docs, fixtures, and release gate

## Claim Boundary

It is honest to say this checkout has initial MedPsy/Qwen3 admission metadata,
policy fixtures, and Rust-native Candle Qwen3 BF16 safetensors plus GGUF CPU
load/generate paths that run when local artifacts are supplied.

It is not yet honest to say:

- Psionic supports accelerated CUDA or Metal MedPsy execution.
- Psionic is competitive with QVAC SDK, `llama.cpp`, Ollama's llama.cpp runner,
  vLLM, or Transformers for MedPsy.
- MedPsy can be used for clinical decision-making.

Those claims require the later issues in the sequence to land with retained
benchmark and safety evidence.

## Release Gate

Run:

```bash
scripts/release/check-psionic-medpsy-pilot.sh
```

The gate validates the retained admission policy, capability matrix, comparator
matrix, MedPsy docs, focused MedPsy model tests, and the benchmark example
compile path. It is green for the bounded metadata, CPU runtime, medical-policy,
and partial-comparator lane. It does not allow a competitive MedPsy claim or a
clinical-use claim.
