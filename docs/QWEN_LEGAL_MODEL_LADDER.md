# Qwen Legal Model Ladder

This ladder keeps legal fine-tuning work in a practical order. Small-model
success is plumbing proof only. Strong legal-model claims start at the dense
27B rung and still need the promotion, payment, and benchmark gates defined in
`docs/QWEN_LEGAL_ACCEPTANCE_TARGETS.md`.

| Rung | Model | Why It Exists | Proves | Does Not Prove | Expected Memory | Quantization | Pylons | Method | Acceptance Target |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| `smoke-qwen35-08b` | `Qwen/Qwen3.5-0.8B` | Fast local correctness and command-surface tests. | Dataset, receipt, replay, scoring, and feed plumbing works. | Strong legal reasoning or retained benchmark quality. | 2-6 GB local RAM or VRAM. | q4/q8 serving or tiny LoRA smoke. | 1 | SFT | All schema, receipt, replay, and public-three regression checks pass. |
| `plumbing-qwen35-4b` | `Qwen/Qwen3.5-4B` | Small model for cheap multi-step plumbing beyond smoke. | SFT, DPO, GRPO, Pylon, settlement, and promotion paths compose. | Production legal quality or strong-model behavior. | 8-16 GB local RAM or VRAM. | q4/q8 serving, LoRA training smoke. | 2 | GRPO | Beats local smoke baseline without public-three regression. |
| `dense-qwen36-27b` | `Qwen/Qwen3.6-27B` | First serious dense legal target; the checkpoint is local and avoids MoE router training. | Dense legal adapter training can improve heldout legal tasks. | MoE router safety or very-large-model economics. | 64-96 GB unified memory or accelerator memory. | bf16/fp16 adapter training, q4/q8 serving checks. | 4 | GRPO | Meets the configured hillclimb target and has zero public-three regression. |
| `moe-qwen36-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | Later sparse target after dense 27B training and MoE-safe serving are stable. | Active-expert sparse training and serving can preserve legal gains. | Very-large distributed training reliability. | 80-128 GB aggregate memory with MoE-aware serving. | MoE-safe adapter training, q4/q8 serving rehearsal. | 6 | GRPO | Beats the dense 27B champion with no router, serving, or public regression failures. |
| `large-serving-eval-only` | Qwen large legal serving/eval target | Serving and evaluation rehearsal after distributed training gates are reliable. | Promotion, rollback, and evaluation paths can handle large models. | That Psionic can train the very large model yet. | 160 GB+ aggregate memory, distributed serving preferred. | Serving quantization only until training gates mature. | 8 | Hybrid RL | Serving/eval parity first; training only after distributed payment and promotion gates are reliable. |

## Operator Rule

Use `smoke-qwen35-08b` and `plumbing-qwen35-4b` to prove the pipes. Do not
describe either result as strong legal-model evidence. Use
`dense-qwen36-27b` as the first serious target because the local checkpoint
keeps iteration practical and dense training avoids MoE router-specific risk.
Move to `moe-qwen36-35b-a3b` only after dense 27B training, MoE-safe serving,
settlement closeout, and promotion gates are reliable. Treat very large models
as serving/evaluation targets until distributed training, payment, and
promotion gates are routine.

The hillclimb controller accepts a rung in the plan under
`model_ladder_rung`, or from the command line with `--rung <rung-name>`.
When a rung is selected, the controller validates the plan against the ladder
model id, adapter target, Pylon count, and training method before writing a
registry record or Autopilot4 progress feed.
