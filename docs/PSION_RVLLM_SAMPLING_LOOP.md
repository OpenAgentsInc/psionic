# PSION RVLLM Sampling Loop

> Status: landed on 2026-03-30 as the fourth retained RVLLM runtime-harvest
> packet.

This document records the current admitted sampling-loop fast path for native
`qwen35` and native `gpt_oss`.

The important truth here is not that Psionic suddenly gained a seeded sampler.
It already had one. The gap was that the repo did not yet retain one explicit
packet describing how the hot path stays narrow: admitted device-resident
selection when possible, reused scratch state for penalties and sparse gathers,
and unchanged seeded replay semantics when the request falls back to dense
logits.

Canonical packet:

* `fixtures/psion/serve/psion_rvllm_sampling_loop_v1.json`

Current retained truth:

* packet digest `19d245f31b7cbad7134b4fe2e5cb537d7c741ba9fecabc0feab13248142c6e7d`
* runtime scope:
  - `generation_sampler.seeded_replay`
  - `qwen35.native_cuda_decode`
  - `gpt_oss.native_cuda_decode`
* qwen35 admitted hot path:
  - device argmax or exact bounded candidates
  - sparse penalty history encoded into reused device buffers
  - seeded replay preserved
  - structured-output posture preserved
* qwen35 explicit fallback:
  - dense host logits remain explicit when the request leaves the admitted
    candidate envelope
* gpt-oss admitted hot path:
  - device argmax with seeded host replay
  - sampling time remains explicit in `stage_timings.sampling_ns`
* qwen35 reused scratch fields:
  - `penalty_token_ids_scratch`
  - `penalty_token_counts_scratch`
  - `sparse_logit_indices_scratch`
  - `top_k_indices_buffer`
  - `top_k_values_buffer`

Seeded parity checks retained by this packet:

* `seeded_sampling_is_replayable`
* `bounded_candidate_sampling_matches_dense_sampling_when_candidate_set_is_exact`
* `bounded_candidate_sampling_matches_dense_sampling_with_penalties_when_candidate_set_is_exact`

Claim boundary:

* This packet does **not** claim every sampling mode stays device-resident.
* It does claim that admitted argmax and exact bounded-candidate lanes already
  avoid avoidable dense copies and preserve seeded replay semantics.
* It keeps dense fallback explicit when the request leaves that envelope.
