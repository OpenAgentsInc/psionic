# Qwen3.6 Real Pylon Rehearsal 001

## Status

- run id: `qwen36-27b-real-pylon-rehearsal-001`
- workers accepted: `2` / `2`
- model: `Qwen/Qwen3.6-27B`
- activation path: `sampled_embed_lm_head_projection_v1`
- trainable target: `lm_head.weight`
- merged adapter: `target/legal/qwen36_27b_real_pylon_rehearsal_001/merge/qwen36-27b-real-pylon-aggregate.safetensors`
- merged adapter sha256: `1b4828c2780a9f5352ce12e63e22015493b8de91ef6aedb41eed3de9780fa692`
- public eval candidate score: `10000` bps
- public eval delta: `6667` bps
- promotion decision: `Promote`
- serve admission verified: `true`
- payment gate: `DeferredByOperator`

## What Ran

Two local loopback Pylon identities each ran the Rust Qwen3.6 sampled-projection LoRA trainer against downloaded Qwen3.6-27B safetensor rows. Each worker produced a signed Pylon receipt, a payable decision, and an adapter. The adapters were merged, evaluated with the Rust public Harvey suite, loaded through the serving adapter path, and closed with an explicit deferred-payment proof.

## Workers

| worker | shard | loss | adapter sha256 | receipt | payment |
| --- | --- | ---: | --- | --- | --- |
| `worker.loopback.macbook-pro-m2.qwen36-real.01` | `shard.harvey-public-three.real-qwen.even` | `2.090468 -> 2.072220` | `e685ea94405f1ccfe2d566d5f2c7472057d32a9febb0bdfe2de7bc93f6ad9541` | `fcd41e420189f1423a3e48b7a9fd550f6e2f834fbf9c2bc61d2c9b35b5cc7ea3` | `Payable` |
| `worker.loopback.imac-pro-bertha.qwen36-real.02` | `shard.harvey-public-three.real-qwen.odd` | `2.090466 -> 2.054379` | `7bbfd3f66c6a1fe9ff38e4cf5f6c54d69603949ebeb738df019a4de930dcf14c` | `dcfd08df0eb889d073ce7fae844c861fe1b47b66f4404f9584e945ffab2f6703` | `Payable` |

## Merge And Eval

- merge manifest: `target/legal/qwen36_27b_real_pylon_rehearsal_001/merge/merge_manifest.json`
- merge manifest sha256: `071a2be66f221f95e9993ff0d79c5586a18672fd047d2b85c9faeaf9b78ec3cf`
- merge receipt: `target/legal/qwen36_27b_real_pylon_rehearsal_001/merge/qwen36-27b-real-pylon-aggregate.merge-receipt.json`
- merge receipt hash: `5a066ac6f75b24777a266b9707826105b316951af7a2e177ac549a47567faad9`
- eval output dir: `target/legal/qwen36_27b_real_pylon_rehearsal_001/eval`
- eval report hash: `b4d6213799aa19ba4ed3a7b091985ffbe7c6a4d970d0fe2325c160139d4beeb7`
- champion score: `3333` bps
- candidate score: `10000` bps
- score delta: `6667` bps

## Recovery And Payment

- checkpoint recovery report: `target/legal/qwen36_27b_real_pylon_rehearsal_001/checkpoint_recovery/qwen_legal_checkpoint_recovery_report.json`
- recovery exact match: `true`
- treasury handoff: `target/legal/qwen36_27b_real_pylon_rehearsal_001/treasury_handoff.json`
- payment closeout: `target/legal/qwen36_27b_real_pylon_rehearsal_001/payment_closeout.json`
- accepted work count: `2`
- deferred payment count: `2`
- failed payment count: `0`
- no wallet secrets present: `true`

## Serve Admission

- route id: `route.qwen36-27b-real-pylon-rehearsal-001.qwen36_real_pylon`
- served model id: `qwen3.6-27b`
- adapter identity digest: `d7675402af763dc7c24473e3653788c6d15b11be65341bd21979b6a39cb800ef`
- benchmark runner arg: `--adapter target/legal/qwen36_27b_real_pylon_rehearsal_001/merge/qwen36-27b-real-pylon-aggregate.safetensors`

## Boundaries

This is a local loopback two-Pylon rehearsal over real downloaded Qwen3.6-27B safetensor rows through the sampled-projection LoRA path. It proves two signed worker contributions, adapter merge, public Rust Harvey eval, serve-adapter admission, and deferred Bitcoin/Lightning payment closeout. It does not prove private Harvey benchmark performance, remote tailnet execution, or full transformer backprop.

No private Harvey gate is available in this repo. This report only claims public training-allowed suite evidence.

- machine-readable report: `target/legal/qwen36_27b_real_pylon_rehearsal_001/rehearsal_report.json`
- report digest: `38c17902de61f78f54cb5b3f6216a93b5283700f1d2b1ccd2f0ec4e15d64f993`
