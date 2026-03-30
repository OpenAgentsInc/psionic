# PSION RVLLM GPU Logits Selection

> Status: landed on 2026-03-30 as the third retained RVLLM runtime-harvest
> packet.

This document records the current admitted GPU-resident token-selection lane
for native `qwen35` and native `gpt_oss`.

The important truth here is not that Psionic suddenly learned GPU argmax on
March 30. It already had that. The gap was that the repo did not yet retain
one explicit packet saying which lanes avoid dense host logits copies and which
lanes still fall back to raw logits on purpose.

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_gpu_logits_selection_v1.json`

Current retained truth:

* packet digest `25f0eced35a620a2ca5e646c2cfd90908c4c7210d34eaaa9cb7202054bcf4f3e`
* runtime scope:
  - `qwen35.native_cuda_decode`
  - `gpt_oss.native_cuda_decode`
* qwen35 admitted GPU-resident lanes:
  - `argmax_only` with `readback_bytes = 8`, no host logits copy
  - `top_k_candidates:40` with `readback_bytes = 320`, no host logits copy
* qwen35 fallback lane:
  - `raw_logits` with `readback_bytes = 604160`, explicit dense fallback
* gpt-oss admitted GPU-resident lane:
  - `argmax_only` with `readback_bytes = 8`, no host logits copy
* gpt-oss fallback lane:
  - `raw_logits` with `readback_bytes = 645120`, explicit dense fallback

Operator surface:

* `qwen35_cuda_bench` remains the machine-readable qwen35 evidence path for:
  - `qwen35_output_modes`
  - `qwen35_readback_bytes`
  - `qwen35_raw_logits`
* the runtime packet now binds that same claim to the existing native `gpt_oss`
  device-argmax path too

Claim boundary:

* This packet does **not** claim every sampling mode is GPU-resident.
* It does claim that admitted argmax-only and bounded candidate lanes avoid
  dense logits copies.
* It keeps raw-logits fallback explicit when the request leaves that envelope.
