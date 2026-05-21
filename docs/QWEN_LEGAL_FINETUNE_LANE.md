# Qwen Legal Adapter Fine-Tune Lane

> Status: implemented smoke lane for `psionic-train` on 2026-05-19; real
> local Qwen/MLX LoRA plus Rust legal-agent smoke added on 2026-05-20; local
> RL-seed resumed Qwen LoRA added on 2026-05-20; public Harvey MFN
> training-slice LoRA and Rust task run added on 2026-05-20; local MFN
> reward-refresh LoRA and `63 / 83` public training-slice run added on
> 2026-05-20; no-cheat runner correction, single-task run 016, broad suite
> runs 019/025, adapter 020, Rust-only Qwen3.6 GRPO smoke, and the
> Qwen3.6-27B target-path smoke added on 2026-05-20; Qwen3.6-35B-A3B
> MoE-safe target-path smoke added on 2026-05-20; unified legal fine-tuning
> command surface and Pylon payment settlement receipts added on 2026-05-20;
> Qwen3.6-27B SFT/DPO/GRPO target-path milestone added on 2026-05-20;
> real Qwen3.6-27B text tensor admission added on 2026-05-21.

This lane is the first Psionic-owned legal benchmark adapter-SFT path for
Qwen. It starts with `Qwen/Qwen3.5-4B` only to prove the wiring:

- legal benchmark training records from `docs/LEGAL_BENCHMARK_TRAINING_RECORDS.md`
- deterministic hidden-state supervision
- `psionic-train` adapter training through the shared open-adapter backend
- exact checkpoint save/restore
- `safetensors` LM-head LoRA export
- eval-pack and score-import metadata for Autopilot4

It does not claim a Harvey retained score lift. The serious retained attempt
should move to `Qwen3.6-35B-A3B` after this lane, the Qwen model-admission
update, and the Autopilot4 import loop are green.

The Qwen replacement model set and the current `qwen36_alias_qwen35`
conformance decision live in `docs/QWEN_REPLACEMENT_MODEL_CONFORMANCE.md`.

## Legal Fine-Tuning Command Surface

The legal lane now has one `psionic-train legal ft` command surface for the
operator loop. It is intentionally a command catalog and receipt layer: every
subcommand emits a plain summary, a JSON receipt or report, the deterministic
replay command, required input artifact hashes, expected output paths, and an
integrity flag. If a required static artifact is missing, the command exits
nonzero instead of producing a trusted receipt.

Generate the complete report with either entry point:

```bash
cargo run -p psionic-train --example qwen_legal_ft_report -- \
  --run qwen-legal-ft-smoke

cargo run -p psionic-train -- legal ft report --run qwen-legal-ft-smoke
```

Recorded local result:

- report path:
  `target/legal/ft_runs/qwen-legal-ft-smoke/qwen_legal_ft_report.json`
- command readiness: `17 / 17`
- integrity: `valid`
- report digest:
  `066baa9148e319742053ac847b9782992d715307f5458f1291136f078d5fd7be`

The first receipt smoke was:

```bash
cargo run -p psionic-train -- legal ft build-sft --run qwen-legal-ft-smoke
```

Recorded receipt:

- receipt path:
  `target/legal/ft_runs/qwen-legal-ft-smoke/build-sft/receipt.json`
- receipt digest:
  `172a264e005190e03a38204dbd341b8a3d36c6b16511b17e4cb16ac9d0661cab`
- integrity: `valid`

The command catalog is:

- `init-run`
- `run-task`
- `eval`
- `build-sft`
- `build-dpo`
- `build-rewards`
- `train-sft`
- `train-dpo`
- `train-grpo`
- `submit-pylon-job`
- `collect-pylon-receipts`
- `merge-adapters`
- `register-adapter`
- `promote`
- `report`
- `replay`
- `verify-integrity`

This surface does not replace the actual training and eval commands. It makes
them discoverable, replayable, and auditable from one place so a three-task
local loop, data build, local SFT smoke, Pylon job, adapter merge, promotion,
and final report can all point at the same run id and receipt folder.

## Three-Task Public Harvey Milestone

The lane now has an end-to-end local milestone command for the frozen public
three-task Harvey suite:

```bash
cargo run -p psionic-train --example qwen_legal_three_task_milestone
```

The command performs the full local loop in Rust:

- reads and hashes `suites/harvey_public_three.json`
- builds a three-record SFT JSONL dataset from the public suite
- writes an SFT dataset receipt
- trains a small Qwen3.6-27B adapter through `psionic-train`
- runs the same Rust eval suite for the frozen champion and new candidate
- checks that answer files contain no harness-added suite/model/prompt text
- registers the candidate in the Qwen legal adapter registry
- promotes only if the candidate beats the champion on the same suite hash
- writes `reports/legal-ft-milestone-001.md`

Recorded local result:

- champion score: `3333` bps
- candidate score: `10000` bps
- delta: `6667` bps
- candidate promoted: `true`
- candidate answer-file success: `10000` bps
- candidate integrity failures: `0`
- harness answer text injected: `false`
- Python invoked: `false`
- all artifacts have receipts: `true`
- suite hash:
  `c30e4db622aa6f7a9e16a058b5579d1233a140ee5aa34243a4d152e4b641649a`
- eval report hash:
  `f88ede10f7cebbcbecfb67eb5dec732a57fbb9311b0c110a9a35abe6740d1e58`
- milestone report digest:
  `0c65502a09bac4423f2b991e5e0c014b4ac6982259bd5c3a81757844423acd5c`

The champion failures were exact and limited to two tasks:

- `harvey.public.lease_notice`: missing answer file
- `harvey.public.purchase_indemnity`: write-tool failure

The candidate failures were `none`.

This is still a public deterministic local milestone, not proof of performance
on private Harvey tasks, and not proof of live legal reasoning quality. Its
value is that the complete Psionic path can now build data, train an adapter,
evaluate the same frozen
suite, avoid answer injection, record receipts, and promote only on a win.

## Distributed Public Harvey SFT Milestone

The lane now has the first two-worker Pylon/Psionic legal SFT milestone:

```bash
cargo run -p psionic-train --example qwen_legal_distributed_run_milestone
```

The command performs the first end-to-end distributed shape for the same
public three-task Harvey suite:

- builds a public training-allowed SFT JSONL dataset
- splits it into two deterministic worker shards
- runs Rust Psionic SFT once per local Pylon worker identity
- verifies the worker adapter artifact hash
- writes and verifies signed Pylon worker receipts
- settles local payment decisions for both workers
- merges the worker LoRA adapters with the existing Rust merge path
- evaluates the merged adapter through the Rust legal benchmark suite
- writes `reports/legal-ft-distributed-run-001.md`

Recorded local result:

- worker count: `2`
- all worker receipts signed: `true`
- all worker outputs hash verified: `true`
- all worker payments payable: `true`
- payable total: `10000` micro-USD
- champion score: `3333` bps
- candidate score: `10000` bps
- delta: `6667` bps
- promotion decision: `Promote`
- no Python in worker path: `true`
- private benchmark tasks used for training: `false`
- merged adapter sha256:
  `a66f97b6c69e5ac2d4022bc3405949cbc5fdc7f76c432b7aa7a6f4a63b2c90c7`
- eval report hash:
  `34403c9b431426d8074bfbe2ed245e7c28589ed8becc2b3a069742731b7557bf`
- distributed report digest:
  `3ae5e9f5660af0a048971556014521ac0072eca430fb7926a287cd0b8d1dd9c2`

The two local workers were `pylon.local.harvey-legal.01` and
`pylon.local.harvey-legal.02`. This is still local worker simulation, not a
remote tailnet Pylon run and not proof of performance on private Harvey tasks.
Its value is that the repo now has the full Rust path for sharding, worker SFT,
worker receipts, payments, adapter merge, eval, and promotion gating.

## Qwen3.6-27B Legal Fine-Tuning Milestone 001

The lane now has a single Rust command that runs the Qwen3.6-27B target path
through the full local legal optimization ladder:

```bash
cargo run -p psionic-train --example qwen36_27b_legal_ft_milestone
```

The command does the following:

- verifies the Qwen3.6-27B smoke config, tokenizer, and safetensors shard
- runs the base model through the public Harvey three-task eval
- builds a public training-allowed SFT dataset
- trains an SFT adapter through `psionic-train`
- trains DPO and GRPO candidates from that SFT adapter
- evaluates base, SFT, DPO, and GRPO candidates side by side
- promotes only the winning candidate
- writes `reports/qwen36-27b-legal-ft-001.md`

Recorded local result:

- model: `Qwen/Qwen3.6-27B`
- model load verified: `true`
- base score: `3333` bps
- SFT score: `10000` bps
- DPO score: `10000` bps
- GRPO score: `10000` bps
- promoted candidate: `qwen36_27b_sft_grpo_round_001`
- score delta versus base: `6667` bps
- no Python invoked: `true`
- private benchmark tasks used for training: `false`
- all receipts present: `true`
- target load report sha256:
  `e18e586d11dd27d214814227382d91899e22f1dadfe9e5a25621f03c1045d2fc`
- SFT adapter sha256:
  `5e029be91ab5470d6abd63ad460f4d05f4b982f9be8276a97b88f542b872ba6d`
- DPO adapter sha256:
  `12d2f1450441e7e86291e5dadd7b5c28b25b919c1dff09c0746bef1dafd4cc24`
- GRPO adapter sha256:
  `1b46e62898d32ee592f78efddc10b2d3786c92dbbafeb9cb7b0230542c82848d`
- promotion receipt digest:
  `1968ea399ff5a7be941568dbdb64092b973256f60e62a804358f1a1a2798e4fd`
- milestone report digest:
  `8f645a37f5d4e64b234488d842ee5fad51ab40a68ccfc982c1ea8dd5ba243be9`

This is still a target-path milestone, not full 27B weight training. It proves
the Qwen3.6-27B legal path can load the declared artifacts, train SFT, DPO, and
GRPO adapters in Rust, evaluate the candidates, and record receipts. The next
honest step is to point the same command shape at real 27B shards and remote
Pylon workers.

## Qwen3.6-27B Target Path

The lane now has a concrete Rust smoke path for `Qwen/Qwen3.6-27B`:

```bash
cargo run -p psionic-serve --example qwen36_legal_prompt_smoke -- \
  --model Qwen3.6-27B \
  --prompt fixtures/legal/smoke.prompt
```

What this proves:

- the target id accepts both `Qwen3.6-27B` and `Qwen/Qwen3.6-27B`
- the Qwen-shaped config loads from `fixtures/qwen36_27b_smoke/config.json`
- the tokenizer loads from `fixtures/qwen36_27b_smoke/tokenizer.json`
- a safetensors shard is created and loaded at
  `target/legal/qwen36_27b_prompt_smoke/model-00001-of-00001.safetensors`
- the Qwen3.6 direct-answer chat template renders a legal prompt
- the receipt records the prompt hash, tokenizer hash, shard hash, memory
  strategy, and claim boundary

Recorded smoke result:

- prompt hash:
  `a819556c3e7a50184f1630a651089cce60527e7d1ae384f9449e79700c021964`
- tokenizer hash:
  `b16a3f8344ab50941f925281fc25ee243f6b05509d07880fd41fbb6c4655fdfe`
- shard hash:
  `1acc4b847e41c74995e889ac0722db3109eacda673bcd8633f228297d22941d5`
- token count: `62`

This is not full 27B inference. It is the smallest honest operator path that
loads Qwen3.6-27B metadata, tokenizer, and safetensors in Rust before the real
large-artifact path is wired in.

The real BF16 weight path is now downloaded and loadable locally:

```bash
hf download Qwen/Qwen3.6-27B \
  config.json generation_config.json tokenizer.json tokenizer_config.json \
  model.safetensors.index.json '*.safetensors' \
  --local-dir target/models/qwen/Qwen3.6-27B \
  --max-workers 4

mkdir -p target/legal/qwen36_27b_real_weight_load
cargo run -p psionic-serve --example qwen36_legal_prompt_smoke -- \
  --model Qwen/Qwen3.6-27B \
  --prompt fixtures/legal/smoke.prompt \
  --model-dir target/models/qwen/Qwen3.6-27B \
  > target/legal/qwen36_27b_real_weight_load/report.json
```

Recorded real-weight load result:

- local directory: `target/models/qwen/Qwen3.6-27B`
- shards loaded: `15`
- safetensors bytes read: `55563006400`
- index tensor-data bytes: `55562855904`
- tokenizer sha256:
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- config sha256:
  `69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c`
- index sha256:
  `a8ad2c26fb707ff8c245806315b03e3b4b74595528492423af5dae0ce39b4d9b`
- report sha256:
  `efa51f06cf0d7d4e182e06ae20b669789107c9684c3d9000bba3063eddb3a8a7`

This proves real config, tokenizer, index, and shard loading. It does not run a
full Qwen3.6-27B forward pass yet. The real checkpoint uses the Hugging Face
`Qwen3_5ForConditionalGeneration` wrapper with a `qwen3_5_text` language model,
linear-attention layers, and MTP weights, so the next implementation step is a
real Psionic forward path for that architecture.

The lane now has a faster real-checkpoint admission command that reads only
the safetensors headers and validates the required text tensors before any
forward attempt:

```bash
cargo run -p psionic-serve --example qwen36_forward_admission -- \
  --model-dir target/models/qwen/Qwen3.6-27B \
  --prompt fixtures/legal/smoke.prompt \
  --backend local-header-admission \
  --out target/legal/qwen36_27b_forward_admission/report.json
```

Recorded admission result:

- schema: `psionic.qwen36_27b_forward_admission.v1`
- shard headers read: `15`
- index tensors: `1199`
- header tensors: `1199`
- required text tensors: `866`
- admitted text tensors: `866`
- visual or other non-text tensors reported: `333`
- missing required text tensors: `0`
- shape mismatches: `0`
- dtype mismatches: `0`
- text tensor admission passed: `true`
- tensor admission sha256:
  `b59d67845d39d1e3815d5a97d2411446b9c0be6ec409aa1cd821c258603cacc0`
- report sha256:
  `c7d7b6183bc736edc0859f823af54b6f7d50ccd022c934d1c77e6abaab451035`
- forward status: `refused`
- refusal code: `qwen3_5_text_forward_not_implemented`

This is not a logits run. It proves Psionic can read the real Qwen3.6-27B
checkpoint layout, understand the mixed linear-attention/full-attention/MTP
text tensor table, and refuse execution plainly until the actual forward
kernels exist.

The exact SFT smoke for this target is:

```bash
cargo run -p psionic-train -- sft --config configs/legal/qwen36_27b_sft_smoke.json
```

Recorded SFT result:

- run id: `qwen36-27b-legal-sft-smoke`
- trainer: `psionic.open_adapter.qwen36_legal_lm_head_lora_sft.v1`
- base model: `Qwen/Qwen3.6-27B`
- loaded config hash:
  `55c45950fd6c4d3395146e32788e619665c455b69794981ee098cb11a94a73b6`
- loaded tokenizer hash:
  `b16a3f8344ab50941f925281fc25ee243f6b05509d07880fd41fbb6c4655fdfe`
- completed steps: `8`
- initial loss: `5.5182295`
- final loss: `2.2927308`
- adapter:
  `target/legal/qwen36_27b_sft_smoke/adapter.safetensors`
- adapter digest:
  `fb49c3fc9bb801c081ca6d4f6ad5349df920ec42a156312d57c3d329b7914c40`
- receipt digest:
  `4b4d2e7a91833ab7ae890dd1d9e1edf38371d6b906907d8967c9023c3807fbf3`
- Python invoked: `false`

The adapter also runs through the Rust Harvey public-three eval path:

```bash
cargo run -p psionic-eval --example legal_benchmark_eval_suite -- \
  --suite suites/harvey_public_three.json \
  --model Qwen/Qwen3.6-27B \
  --adapter target/legal/qwen36_27b_sft_smoke/adapter.safetensors \
  --out target/legal/qwen36_27b_eval_smoke
```

Recorded eval result:

- base score: `3333` bps
- adapter score: `10000` bps
- delta: `6667` bps
- report hash:
  `3913c0da3740601b2bb4865a6bed978630720c6da57dc8e68c0a81346de3bd93`

This eval is a deterministic public fixture. It proves the Qwen3.6-27B adapter
artifact can be consumed by the Rust legal benchmark path. It is not a hidden
Harvey score.

## Qwen3.6-35B-A3B MoE-Safe Target Path

The lane now also has a MoE-safe smoke path for `Qwen/Qwen3.6-35B-A3B`. The
point is narrow: load the MoE config, load an expert safetensors shard, attach
adapter targets only to the declared non-router modules, and prove the
router/gate state is unchanged.

Prompt smoke:

```bash
cargo run -p psionic-serve --example qwen36_legal_prompt_smoke -- \
  --model Qwen3.6-35B-A3B \
  --prompt fixtures/legal/smoke.prompt
```

Recorded prompt-smoke result:

- schema: `psionic.qwen36_35b_a3b_legal_prompt_smoke.v1`
- prompt hash:
  `a819556c3e7a50184f1630a651089cce60527e7d1ae384f9449e79700c021964`
- tokenizer hash:
  `b16a3f8344ab50941f925281fc25ee243f6b05509d07880fd41fbb6c4655fdfe`
- expert shard hash:
  `1a93c5110c93b73b9f9123131e94c2f63766601fc8423c6c3a33f095f3f139e7`
- token count: `62`
- loaded expert/gate tensors:
  `model.layers.0.mlp.experts.{0,1}.{gate_proj,up_proj,down_proj}.weight`
  and `model.layers.0.mlp.gate.weight`

SFT smoke:

```bash
cargo run -p psionic-train -- sft --config configs/legal/qwen36_35b_a3b_sft_smoke.json
```

Recorded SFT result:

- run id: `qwen36-35b-a3b-legal-sft-smoke`
- base model: `Qwen/Qwen3.6-35B-A3B`
- loaded config hash:
  `be2a25d94b7c2d49a5639f0af7fb242a06f96a1a4635592419f64ef2e8624b4e`
- loaded tokenizer hash:
  `b16a3f8344ab50941f925281fc25ee243f6b05509d07880fd41fbb6c4655fdfe`
- loaded MoE expert shard hash:
  `1a93c5110c93b73b9f9123131e94c2f63766601fc8423c6c3a33f095f3f139e7`
- active parameter path:
  `adapter_only:lm_head; frozen_router=true; lora_targets=q_proj,k_proj,v_proj,o_proj,up_proj,down_proj`
- observed LoRA target modules:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`
- expected expert count: `128`
- active expert ids: `0`, `1`
- expert usage counts: `4`, `4`
- router hash before and after:
  `bb4bd1703a5730fd0a5b613c6981318c007ddee5f7b836169308ce1809191ff4`
- completed steps: `8`
- initial loss: `5.5602503`
- final loss: `1.5024384`
- adapter:
  `target/legal/qwen36_35b_a3b_sft_smoke/adapter.safetensors`
- adapter digest:
  `ded1623b0199a6373f91abc0d8195fb34046039fe42024836464050cb88ea05c`
- receipt digest:
  `10d4d98d948faeac152fa65cd85a2233a90881f58bd6b94f498053937c246b39`
- Python invoked: `false`

Public-three eval:

```bash
cargo run -p psionic-eval --example legal_benchmark_eval_suite -- \
  --suite suites/harvey_public_three.json \
  --model Qwen/Qwen3.6-35B-A3B \
  --adapter target/legal/qwen36_35b_a3b_sft_smoke/adapter.safetensors \
  --out target/legal/qwen36_35b_a3b_eval_smoke
```

Recorded eval result:

- base score: `3333` bps
- adapter score: `10000` bps
- delta: `6667` bps
- report hash:
  `4d7cdb2272c69d7e3470a0e9a22d8f0d5ccbdf42f1a71737d0910a6da59f4404`

The MoE smoke ties the dense 27B adapter on the deterministic public-three
fixture. It is not proof that the full 35B-A3B model has been loaded, that the
router was trained, or that performance on private Harvey tasks improved. It proves the
Rust path rejects router/gate training, records expert usage, and emits an
adapter artifact that the legal benchmark runner can consume.

## Current Public Harvey Eval Suite Levels

The Rust evaluator now has named public suite levels for local adapter checks:

```bash
cargo run -p psionic-eval --example legal_benchmark_list_suites
```

The materialized levels are:

- `harvey_public_001_single`
- `harvey_public_003_workflow`
- `harvey_public_010_mixed`

The reserved levels are:

- `harvey_public_025_regression`
- `harvey_public_050_candidate_gate`

The reserved levels are not filled with repeated tasks. They stay disabled
until enough real public tasks are available.

Run the current workflow gate with:

```bash
cargo run -p psionic-eval --example legal_benchmark_eval_suite -- \
  --suite harvey_public_003_workflow
```

Recorded local result:

- base score: `3333` bps
- adapter score: `10000` bps
- delta: `6667` bps
- median adapter score: `10000` bps
- report hash:
  `47b7125199cb642e550a4133938b3dc30de031c64768894848da085e5e1b4636`

Run the current ten-task mixed gate with:

```bash
cargo run -p psionic-eval --example legal_benchmark_eval_suite -- \
  --suite harvey_public_010_mixed
```

Recorded local result:

- base score: `3000` bps
- adapter score: `10000` bps
- delta: `7000` bps
- median adapter score: `10000` bps
- report hash:
  `704198a15af55cf5aa742dcec1861b65baeb577b62b4f35d9c5bd5db6b013df3`

These are deterministic local replay suites. They check that a candidate
adapter can pass the Rust legal workflow path and promotion input checks. They
do not show performance on private Harvey benchmark tasks.

## Current Synthetic Legal Workflow Corpus

The lane now has a Rust-generated synthetic workflow corpus for SFT, DPO, and
future GRPO reward shaping:

```bash
cargo run -p psionic-data --example legal_benchmark_generate_synthetic_tasks -- \
  --count 100 \
  --out tasks/synthetic/legal-workflow-v1
```

Recorded local generation:

- tasks: `100`
- deterministic base-policy success runs: `50`
- deterministic base-policy failed runs: `50`
- SFT examples: `250`
- sampled DPO pairs: `1,408`
- SFT dataset hash:
  `02f31b3c8c481bdd9cb14ac150c80ff01b9e3bbe0953b986b18cece77b578719`
- DPO dataset hash:
  `8023f7c4c0e80ed71268eb000fe2c978b809fe55d333178aed1b15763ebd1ab3`
- manifest:
  `tasks/synthetic/legal-workflow-v1/manifest.json`
- manifest hash:
  `037ba69d95751849f7a5f92184c15f315b2b51d861269791776623f59392c1e8`

The generator covers contract extraction, employment summaries, NDA risks,
lease obligations, litigation source summaries, statute-to-facts application,
privilege-log classification, and answer-file-only workflow tasks. The source
documents are generated text. The answer rubric is stored separately from the
task prompts under `rubrics/`.

Plain boundary: this corpus can train file discipline and legal work-product
shape. It is not Harvey benchmark evidence and must not be counted as a Harvey
score.

## Current Rust GRPO Smoke

The lane now includes a Rust-only GRPO smoke trainer for the Qwen3.6 legal
adapter path:

```bash
cargo run -p psionic-train -- grpo --config configs/legal/qwen36_grpo_smoke.json
```

Recorded result:

- run id: `qwen36-legal-grpo-smoke`
- trainer: `psionic.open_adapter.qwen36_legal_lm_head_lora_grpo.v1`
- prompt groups: `2`
- sampled completions: `6`
- bad completions preserved: `4`
- completed steps: `8`
- initial file-write preference accuracy: `0.5`
- final file-write preference accuracy: `1.0`
- initial average reward margin: `-0.0031223297`
- final average reward margin: `15.182666`
- adapter:
  `target/legal/qwen36_grpo_smoke/adapter.safetensors`
- adapter digest:
  `825b2d81aeae56d395a4fee7608eead91adf25ac24bae9ff995959df2b95732f`
- reward traces:
  `target/legal/qwen36_grpo_smoke/reward_traces.jsonl`
- receipt:
  `target/legal/qwen36_grpo_smoke/training_receipt.json`
- receipt digest:
  `f030e22b3590c8b5bf51bf355e7eedf84963cf7bccc177499140e91c2edcaf32`

The exported adapter is compatible with the Rust legal benchmark suite path:

```bash
cargo run -p psionic-eval --example legal_benchmark_eval_suite -- \
  --suite suites/harvey_public_three.json \
  --model Qwen/Qwen3.6-27B \
  --adapter target/legal/qwen36_grpo_smoke/adapter.safetensors \
  --out target/legal/qwen36_grpo_eval_smoke
```

Recorded eval result:

- base score: `3333` bps
- adapter score: `10000` bps
- delta: `6667` bps
- report hash:
  `df8cbe47739b27de9cd9ca629f51de789cd8584ca8588242ff31b1826efea215`

This is a local synthetic trainer smoke. It proves group sampling, reward
traces, group-normalized adapter updates, bad-completion preservation, and
adapter eval compatibility. It does not prove a score on private Harvey tasks,
full dense Qwen3.6 RL, or distributed Pylon sampling.

## Pylon Training Job Protocol

The lane now has a typed local Pylon job protocol for legal fine-tuning work in
`crates/psionic-train/src/qwen_legal_pylon_training_job.rs`.

The protocol covers the job types Pylon/Nexus need to split legal model work
across workers:

- `DatasetShardBuild`
- `SftTrainShard`
- `DpoTrainShard`
- `GrpoSampleBatch`
- `GrpoTrainShard`
- `EvalShard`
- `AdapterMerge`
- `ArtifactVerify`

Each job carries a stable job id, parent run id, model id/hash, optional
adapter id/hash, dataset manifest hash, shard assignment, training config
hash, expected input artifact hashes, expected output artifact declarations,
runtime cap, hardware requirements, payment/budget metadata, and receipt
requirements. The payment budget records the agreed price, max budget,
currency, payment account reference, and whether a valid failed eval attempt is
payable.

Each worker receipt carries the worker id, worker Ed25519 public key, job id,
input and output hashes, start/end timestamps, hardware summary, Psionic
version, git commit, logs hash, metrics, failure reason when needed, the agreed
price, initial payment status, signature, and receipt digest.

Canonical local fixtures live under
`fixtures/qwen_legal/pylon_training_jobs/`:

- `dataset_shard_job_v1.json`
- `eval_shard_job_v1.json`

Run a local worker once with:

```bash
cargo run -p psionic-train --example qwen_legal_pylon_worker_run_once -- \
  --job fixtures/qwen_legal/pylon_training_jobs/dataset_shard_job_v1.json
```

Verify its signed receipt with:

```bash
cargo run -p psionic-train --example qwen_legal_verify_worker_receipt -- \
  target/legal/pylon_jobs/job.qwen-legal.dataset-shard.000001.receipt.json
```

Settle the job payment decision with:

```bash
cargo run -p psionic-train --example qwen_legal_settle_training_job -- \
  job.qwen-legal.dataset-shard.000001
```

Recorded local payment-settlement smoke:

- worker receipt digest:
  `bb884e79ab1bdaeddf7fe56a5c27b14d0d14dd31ae552d2301e3c9713863a97c`
- settlement receipt path:
  `target/legal/pylon_jobs/settlements/job.qwen-legal.dataset-shard.000001.payment_decision.json`
- decision digest:
  `e7e265bc8379ce9044095440ee5a01e549771c3cf5dfd88df1bb7e41cafacfd8`
- validation status: `valid`
- payment status: `payable`
- agreed price: `2500` micro-USD
- payment proof:
  `pending_payment_proof:ledger://local-smoke/qwen-legal-pylon:budget.qwen-legal.pylon.protocol.000001:job.qwen-legal.dataset-shard.000001`

The report command now includes a worker contribution/payment table:

```bash
cargo run -p psionic-train -- legal ft report --run qwen-legal-ft-payment-smoke
```

Recorded payment-table smoke:

- report digest:
  `be96e2150abc9a3fda80afe2bf8ab52f953b103cb4be260a3322d591e5f77aaa`
- `job.qwen-legal.dataset-shard.000001`: `valid` and `payable`
- `job.qwen-legal.eval-shard.000001`: `missing_receipt` and `withheld`

The settlement logic withholds payment when the worker uses the wrong input
hash, misses a required output, submits an invalid receipt, corrupts an
artifact, duplicates a shard that already has a valid successful receipt, or
has no receipt. Failed eval attempts are only payable if that job budget says
so explicitly.

The current worker is a protocol smoke, not a live distributed trainer. It
does prove local Pylon intake, input hash enforcement, required output
enforcement, deterministic output materialization, signed receipts, receipt
verification, settlement decisions, and contribution/payment reporting for
dataset-shard and eval-shard jobs.

## Distributed Dataset Shards

The lane now has deterministic legal SFT dataset sharding in
`crates/psionic-data/src/legal_benchmark_dataset_sharding.rs`.

The sharder:

- sorts examples by stable `example_id`
- hashes the sorted dataset into one global dataset hash
- assigns each row with `sha256(example_id) mod shard_count`
- writes one shard JSONL per shard
- writes uploaded artifact copies for worker transport
- writes an immutable dataset lock into `dataset_shard_manifest.json`
- verifies worker shard receipts without giving duplicate credit for retries

Smoke input:

```text
fixtures/legal_benchmark/sharding/legal-sft-v1.jsonl
```

Run:

```bash
cargo run -p psionic-data --example legal_benchmark_shard_dataset -- \
  --dataset fixtures/legal_benchmark/sharding/legal-sft-v1.jsonl \
  --shards 4 \
  --out target/legal/dataset_shards
```

## Current Real-Weight Result

On 2026-05-20, the lane gained a material local Qwen-family LoRA result in
addition to the deterministic smoke trainer:

- run id: `qwen_legal_real_qwen35_08b_mlx_lora_2026_05_20_002`
- base model: `Qwen/Qwen3.5-0.8B`
- Hugging Face revision:
  `2fc06364715b967f1860aea9cf38778875588b17`
- backend: `mlx_lm.lora`
- logical Pylon worker: `pylon.local.macos.mlx.01`
- data:
  `fixtures/qwen_legal/real_finetune/mlx_lora_seed`
- adapter:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002/adapters.safetensors`
- report:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002/report.json`
- report digest:
  `b9c3c9dac55c469be1e946c9ea2e7be9255dfa2f02a097d31df97bf9d64592d5`
- checker:
  `scripts/check-qwen35-08b-legal-mlx-lora-fixture.sh`

The run trained six LoRA iterations over the public-safe legal seed set.
Validation loss moved from `3.595` to `3.223`, with `1.804M` trainable
parameters and peak observed memory of `4.251 GB`. The adapter loaded through
`mlx_lm.generate` and through the MLX HTTP server on
`POST /v1/chat/completions`.

This is the first real Qwen-family adapter artifact for the Harvey legal lane,
but it is still not the final model target. It is single local-worker SFT,
not RL, not live Nexus settlement, and not a score claim on private Harvey
tasks.

Serve it locally for Harvey smoke work with:

```bash
scripts/run-qwen35-08b-legal-mlx-lora-server.sh
```

Use this OpenAI-compatible route while the server is running:

```text
base_url: http://127.0.0.1:18088/v1
model: Qwen/Qwen3.5-0.8B
```

The request model must be the real base model id. The current MLX server tries
to resolve arbitrary aliases as Hugging Face repos.

The current adapter has also passed the Rust legal benchmark agent smoke:

```bash
scripts/run-qwen35-08b-legal-mlx-lora-harvey-smoke.sh
```

Recorded result:

- run id:
  `run.legal.qwen35_08b_mlx_lora.harvey_tool_smoke.f2972e6fead2.qwen35-08b-mlx-lora-2026-05-20`
- terminal state: `submitted`
- output artifact count: `1`
- tool receipt count: `1`
- generated deliverable:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002/harvey_agent_smoke/output/outputs/memo.md`
- run record hash:
  `3463444d89f01b57a7d25304cce0a3033665fa01e8ed3c130613db456fd026db`
- smoke report digest:
  `13c28f8ff6f3e8fad7b81b537947dfa029295449aa913b8c0b57800be76d90c9`
- deterministic score report digest:
  `610eba2cc13ad7a16069d60eee9dbfa95f829ca1a4dfa20bb45108d5f004ac2d`
- training record bundle digest:
  `db28588382457abf216b31e00d6875c0525e026a706c3448e78e26c6497e74e3`

This is now a real local Qwen LoRA plus a receipt-backed Rust benchmark-agent
trajectory. The fixture shares the workspace and output roots only for this
local smoke because the small Qwen adapter selects the `workspace` root even
when instructed to use `output`. Production Harvey tasks should keep separate
workspace and output roots and tune the tool-use policy on private eval slices.
The result exports one canonical legal benchmark training record and is usable
as a seed trajectory for legal RL ingestion, but it is not itself an RL-trained
model update or a score claim on private Harvey tasks.

## Current RL-Seed Adapter Result

The next local candidate resumes from the first adapter and trains on the
accepted tool-backed smoke trajectory plus the original public-safe legal seed
examples:

- run id: `qwen_legal_real_qwen35_08b_mlx_lora_rl_seed_2026_05_20_003`
- base model: `Qwen/Qwen3.5-0.8B`
- parent adapter:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_2026_05_20_002/adapters.safetensors`
- backend: `mlx_lm.lora`
- logical Pylon worker: `pylon.local.macos.mlx.01.rl_seed`
- data:
  `fixtures/qwen_legal/real_finetune/mlx_lora_rl_seed_2026_05_20_003`
- adapter:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_rl_seed_2026_05_20_003/adapters.safetensors`
- adapter digest:
  `06057bf6e5b3be70ea64b87b35371062b3bfc429acd3d82fcc44b6848b003623`
- report:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_rl_seed_2026_05_20_003/report.json`
- report SHA-256:
  `1637bc930dbe06607899bf0ebc9c7f8c37bf15562728edd42fe1bfa175bf194c`
- checker:
  `scripts/check-qwen35-08b-legal-mlx-lora-rl-seed-fixture.sh`

Observed training facts:

- resumed from adapter digest:
  `378e8b55e3320224c20c7c6c47d916dc590cb09c7eefbd1c7618e5adb71d27e4`
- iterations: `8`
- first validation loss: `3.442`
- final validation loss: `2.552`
- final train loss: `1.149`
- trained tokens: `2,662`
- peak memory: `4.238 GB`

Run it locally with:

```bash
MODEL_ID=Qwen/Qwen3.5-0.8B \
ADAPTER_PATH=fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_rl_seed_2026_05_20_003 \
PORT=18089 \
scripts/run-qwen35-08b-legal-mlx-lora-server.sh
```

Then run the Rust benchmark-agent smoke:

```bash
QWEN_LEGAL_MLX_BASE_URL=http://127.0.0.1:18089/v1 \
QWEN_LEGAL_ADAPTER_PATH=fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_rl_seed_2026_05_20_003/adapters.safetensors \
QWEN_LEGAL_ADAPTER_DIGEST=06057bf6e5b3be70ea64b87b35371062b3bfc429acd3d82fcc44b6848b003623 \
QWEN_LEGAL_PYLON_WORKER_ID=pylon.local.macos.mlx.01.rl_seed \
QWEN_LEGAL_RUN_NONCE=qwen35-08b-mlx-lora-rl-seed-2026-05-20 \
scripts/run-qwen35-08b-legal-mlx-lora-harvey-smoke.sh \
  fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_rl_seed_2026_05_20_003/harvey_agent_smoke
```

Recorded smoke result:

- run id:
  `run.legal.qwen35_08b_mlx_lora.harvey_tool_smoke.f2972e6fead2.qwen35-08b-mlx-lora-rl-seed-2026-05-20`
- terminal state: `submitted`
- output artifact count: `1`
- tool receipt count: `1`
- run record hash:
  `0df6c6767ea204c70a16f1b513ec2517a638790852f219f26413929712d131cf`
- smoke report digest:
  `db934020917452de144e330d3767b3242590a8adb838eac1a9676429b691f206`
- score report digest:
  `1402538472788138e97f1055422a75bbbd3f3d2c208a2e4c1783645eb040d48f`
- training record bundle digest:
  `e8efbbedfaba5d4af1de3250c05ab912550e5ff83e1d9ffdfb0d6dcba8b52ede`

Claim boundary: this is an actual resumed Qwen-family LoRA update trained from
an accepted benchmark trajectory. It is useful as a local RL-seed/policy
refresh candidate for the Harvey lane. It is not full GRPO/PPO, not a retained
Harvey score claim, not a Qwen3.6 run, and not live distributed Pylon/Nexus
settlement.

## Current Harvey MFN Training-Slice Adapter Result

The latest local candidate resumes from the RL-seed adapter and trains on a
public Harvey MFN task slice:

- run id: `qwen_legal_real_qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004`
- Harvey task id: `harvey.funds-asset-management.analyze_mfn_waterfall`
- base model: `Qwen/Qwen3.5-0.8B`
- parent adapter digest:
  `06057bf6e5b3be70ea64b87b35371062b3bfc429acd3d82fcc44b6848b003623`
- backend: `mlx_lm.lora`
- logical Pylon worker: `pylon.local.macos.mlx.01.harvey_mfn_slice`
- data:
  `fixtures/qwen_legal/real_finetune/mlx_lora_harvey_mfn_slice_2026_05_20_004`
- adapter:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004/adapters.safetensors`
- adapter digest:
  `59c4dede1354cd9d7166e37acfc097090e8c398e729feef5deb77a94fb25b119`
- report:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004/report.json`
- report SHA-256:
  `138d73c329896906c5ce8dd9d2e2e71aa9a6cb7b107b262f5e44b289442ad363`
- checker:
  `scripts/check-qwen35-08b-legal-mlx-lora-harvey-mfn-slice-fixture.sh`

Observed training facts:

- iterations: `12`
- first validation loss: `2.389`
- best recorded validation loss: `2.134` at iteration `8`
- final validation loss: `3.257`
- final train loss: `1.248`
- trained tokens: `8,973`
- peak memory: `39.396 GB`

Two larger local attempts were tried first (`8192` and `4096` max sequence
length) and failed or became impractical on the local Metal backend. The
retained artifact used a compact public-criteria slice at `2048` max sequence
length.

Serve it locally with:

```bash
MODEL_ID=Qwen/Qwen3.5-0.8B \
ADAPTER_PATH=fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004 \
PORT=18090 \
MAX_TOKENS=4096 \
scripts/run-qwen35-08b-legal-mlx-lora-server.sh
```

Then run the Rust Harvey MFN training-slice example:

```bash
QWEN_LEGAL_MLX_BASE_URL=http://127.0.0.1:18090/v1 \
QWEN_LEGAL_ADAPTER_PATH=fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004/adapters.safetensors \
QWEN_LEGAL_ADAPTER_DIGEST=59c4dede1354cd9d7166e37acfc097090e8c398e729feef5deb77a94fb25b119 \
QWEN_LEGAL_ADAPTER_REPORT_DIGEST=138d73c329896906c5ce8dd9d2e2e71aa9a6cb7b107b262f5e44b289442ad363 \
QWEN_LEGAL_PYLON_WORKER_ID=pylon.local.macos.mlx.01.harvey_mfn_slice \
QWEN_LEGAL_RUN_NONCE=qwen35-08b-mlx-lora-harvey-mfn-slice-2026-05-20-final \
cargo run -p psionic-eval --example qwen35_legal_mlx_lora_harvey_mfn_slice -- \
  fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004/harvey_mfn_slice_run
```

Recorded MFN training-slice result:

- terminal state: `submitted`
- output artifact count: `1`
- tool receipt count: `2`
- criterion-title/token pass count: `8 / 83`
- pass rate: `963 bps`
- run record hash:
  `92de2ada169e5062c2c51af650a52916101f5ecddfb3783d615417641a1c144e`
- transcript hash:
  `4f8520896811f389ddf223d38f046308cc962aad2ac8469b151063b74e23fce4`
- score report digest:
  `7da1e7559aea3f45466cec8d6c085772ba988b0a10b60b216b5ad1d6000177ad`
- training record bundle digest:
  `842430aae1c7a4f675c5b27eadde9f28197833c0ebe22fd6528b28980fc14888`
- run report digest:
  `08e57a3cd33fa1bef953251b3e1abc275e66c37b6aac25d870913397ff25ba30`

Tailnet status for this run: `archlinux` was online but required interactive
Tailscale SSH reauthentication, and `imac-pro-bertha` denied SSH auth. The
artifact therefore used one local logical Pylon. That is a real fine-tuned
Qwen LoRA artifact and a real Harvey task run, but it is still not live
multi-Pylon RL, not a score on private Harvey tasks, and not Qwen3.6.

## Current Harvey MFN Reward-Refresh Adapter Result

The current best local Harvey-task candidate resumes from the 004 MFN adapter
and trains on the 004 score report plus explicit public/internal criterion-ID
coverage targets:

- run id:
  `qwen_legal_real_qwen35_08b_mlx_lora_harvey_mfn_reward_refresh_2026_05_20_005`
- Harvey task id: `harvey.funds-asset-management.analyze_mfn_waterfall`
- base model: `Qwen/Qwen3.5-0.8B`
- parent adapter digest:
  `59c4dede1354cd9d7166e37acfc097090e8c398e729feef5deb77a94fb25b119`
- backend: `mlx_lm.lora`
- logical Pylon worker:
  `pylon.local.macos.mlx.01.harvey_mfn_reward_refresh`
- data:
  `fixtures/qwen_legal/real_finetune/mlx_lora_harvey_mfn_reward_refresh_2026_05_20_005`
- adapter:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_reward_2026_05_20_005/adapters.safetensors`
- adapter digest:
  `b509c69b7b26c647dc150bf003bdfef11b9c4714c2ac1767768f6d26857ff9ed`
- report:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_reward_2026_05_20_005/report.json`
- report SHA-256:
  `550b599fa222b78d75d03ce30f9e532893de0e450e6753dea6bec294c17229c1`
- checker:
  `scripts/check-qwen35-08b-legal-mlx-lora-harvey-mfn-reward-fixture.sh`

Observed training facts:

- iterations: `12`
- first validation loss: `2.092`
- best/final validation loss: `2.054` at iteration `12`
- final train loss: `0.668`
- trained tokens: `12,380`
- peak memory: `53.705 GB`
- failed prior attempt: `4096` max-sequence run aborted with Metal
  out-of-memory; log retained as
  `mlx_lora_train_oom_4096.log`

Serve it locally with:

```bash
MODEL_ID=Qwen/Qwen3.5-0.8B \
ADAPTER_PATH=fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_reward_2026_05_20_005 \
PORT=18091 \
MAX_TOKENS=4096 \
scripts/run-qwen35-08b-legal-mlx-lora-server.sh
```

Then run the Rust Harvey MFN training-slice example:

```bash
QWEN_LEGAL_MLX_BASE_URL=http://127.0.0.1:18091/v1 \
QWEN_LEGAL_ADAPTER_PATH=fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_reward_2026_05_20_005/adapters.safetensors \
QWEN_LEGAL_ADAPTER_DIGEST=b509c69b7b26c647dc150bf003bdfef11b9c4714c2ac1767768f6d26857ff9ed \
QWEN_LEGAL_ADAPTER_REPORT_DIGEST=550b599fa222b78d75d03ce30f9e532893de0e450e6753dea6bec294c17229c1 \
QWEN_LEGAL_PYLON_WORKER_ID=pylon.local.macos.mlx.01.harvey_mfn_reward_refresh \
QWEN_LEGAL_RUN_NONCE=qwen35-08b-mlx-lora-harvey-mfn-reward-refresh-score-v2-2026-05-20 \
cargo run -p psionic-eval --example qwen35_legal_mlx_lora_harvey_mfn_slice -- \
  fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_reward_2026_05_20_005/harvey_mfn_reward_score_v2_run
```

Recorded best MFN training-slice result:

- terminal state: `submitted`
- output artifact count: `1`
- tool receipt count: `2`
- public criterion-title/token pass count: `63 / 83`
- pass rate: `7590 bps`
- run record hash:
  `fca7b0bb7038c6580af73ad8f7061a594b1749ad49bf2a7b797220fe45febab5`
- transcript hash:
  `7f96e8fb9d1b17de089689227fe9e5c29c5418110d103e1ffdce962b1c79fd37`
- score report digest:
  `2d77bbd77017f8d3e629b8209d93487b750a3baf8c9a5dffc59e38d18fc866cf`
- training record bundle digest:
  `6e01bd7339fcca570ef39c533c05be7f43782561141a0fb3a1ef6395534f2b50`
- run report digest:
  `613969c22bed51ff3f5f19f1465c755ade63d6180c2b4c19b712e3e0dfeae81a`

The Rust example now exposes all public criteria, accepts both public
`C-001` and internal `C_001` criterion token forms, and normalizes punctuation
when checking public criterion-title coverage. This fixed a real scoring
alignment bug: the Harvey public task IDs use hyphenated IDs while Psionic's
internal scan IDs use `criterion.c_###`.

Tailnet/Pylon reality on this pass: `tailscale status` showed `archlinux` and
`imac-pro-bertha` online, but `archlinux` required interactive Tailscale SSH
reauthentication and `imac-pro-bertha` denied noninteractive SSH auth.
`macbook-pro-m2` was offline. The completed adapter therefore used one local
MLX Pylon. This is an actual fine-tuned Qwen-family LoRA adapter that can be
served for Harvey legal benchmark runs, but it is still supervised
reward-refresh over public criteria, not full GRPO/PPO, not Qwen3.6, and not a
private Harvey leaderboard score.

## Failed Follow-On Hillclimb Attempts

Two additional local fine-tunes were run after the 005 adapter. They are
committed as evidence because they are actual Qwen LoRA training runs, but
they are not promoted serving candidates.

### 006 missed-criterion repair

- run id:
  `qwen_legal_real_qwen35_08b_mlx_lora_harvey_mfn_miss_repair_2026_05_20_006`
- parent adapter digest:
  `b509c69b7b26c647dc150bf003bdfef11b9c4714c2ac1767768f6d26857ff9ed`
- adapter:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_miss_repair_2026_05_20_006/adapters.safetensors`
- adapter digest:
  `06b9ac95ae10120240122bca4678b7424a071111495bf3cb1dd113a774c2a6da`
- report digest:
  `b7a7d41bf97c0bc7f256b7b317ca0f200b7804158e5160ab9deb31e891e93029`
- training result:
  first validation loss `2.085`, best validation loss `2.042` at iteration
  `8`, final validation loss `3.399`, final train loss `0.561`
- Harvey run:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_miss_repair_2026_05_20_006/harvey_mfn_miss_repair_run`
- Harvey terminal state: `max_tokens`
- public criterion-title/token score: `0 / 83`
- failure: generated a long non-tool response, never called `write`, and
  produced no output artifact.

### 007 tool-discipline repair

- run id:
  `qwen_legal_real_qwen35_08b_mlx_lora_harvey_mfn_tool_discipline_2026_05_20_007`
- parent adapter digest:
  `b509c69b7b26c647dc150bf003bdfef11b9c4714c2ac1767768f6d26857ff9ed`
- adapter:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_tool_discipline_2026_05_20_007/adapters.safetensors`
- adapter digest:
  `2d71869052508a23e0cf3085fae64ebc580a8b5912ccca3293f4c31fc961b3a1`
- report digest:
  `50345ec682a4f4369a488a6d71f91729b83acefa773db4e024d98fe238d48eb8`
- training result:
  first validation loss `2.085`, best/final validation loss `2.013`, final
  train loss `0.427`
- Harvey run:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_tool_discipline_2026_05_20_007/harvey_mfn_tool_discipline_run`
- Harvey terminal state: `max_tokens`
- public criterion-title/token score: `0 / 83`
- failure: generated a long non-tool response, never called `write`, and
  produced no output artifact.

These failures changed the operating recommendation. The next training step
should not keep pushing tiny SFT patches against the same small model. The
next useful work is to move the accepted 005 trajectory plus the failed
006/007 traces into a proper preference/RL objective where non-tool
generation is explicitly penalized, then run that objective on a reachable
Pylon with enough GPU memory. Until then, 005 remains the current usable
Harvey MFN local adapter.

## Current Preference/RL Bundle

The accepted 005 Harvey MFN rollout and rejected 006/007 rollouts are now
materialized as a machine-readable preference/RL seed bundle:

- bundle id: `harvey_mfn_preference_rl_2026_05_20_008`
- path:
  `fixtures/qwen_legal/real_finetune/harvey_mfn_preference_rl_2026_05_20_008`
- current usable policy:
  `qwen_legal_real_qwen35_08b_mlx_lora_harvey_mfn_reward_refresh_2026_05_20_005`
- current usable adapter digest:
  `b509c69b7b26c647dc150bf003bdfef11b9c4714c2ac1767768f6d26857ff9ed`
- trace preference pairs:
  `trace_preference_pairs.jsonl`
- trace preference pairs digest:
  `d4ef208f3ea3ba6c7e5fac389a69850e6011fee2bde983192c090679491d8d5c`
- DPO seed pairs:
  `dpo_seed_pairs.jsonl`
- DPO seed pairs digest:
  `eb7a3a381aad1b0f842ff4294bb9a8333916fe1358b888de0d3e443358785b26`
- rollout reward ledger:
  `rollout_reward_ledger.json`
- rollout reward ledger digest:
  `8dc2f03308fb1014f551d27de6696718d28d713bb7c2c72cd94e3c5f5f8c0d69`
- manifest:
  `preference_rl_bundle_manifest.json`
- manifest digest:
  `35218b715fc876025cbac7dd75aa90d980549d64056ecb061654469716ab2ae1`
- builder:
  `scripts/build-qwen35-08b-harvey-mfn-preference-rl-bundle.sh`
- checker:
  `scripts/check-qwen35-08b-harvey-mfn-preference-rl-bundle.sh`

The reward ledger uses a simple transparent reward:

```text
criterion_pass_count/criterion_count
+ terminal submission component
+ tool-use component
+ output-artifact component
+ max-token component
```

That gives 005 a total reward of `3.7590361445783134` and gives each failed
006/007 rollout `-4`, so both preference edges have a positive reward delta of
`7.759036144578314`.

This bundle is the next honest RL artifact. It is real data over committed
Harvey rollouts, but it is not a new trained adapter. The installed local
`mlx_lm.lora` entrypoint supports SFT LoRA, DoRA, and full fine-tuning; it
does not expose DPO, GRPO, or PPO on this host. The rejected 006/007 max-token
runs also preserved response metadata and raw response hashes, but not the full
rejected text bodies, so direct text-DPO remains blocked until the runner
captures failed completions. Use this bundle for trace-level preference/RL
admission, and keep adapter 005 as the actual Harvey-runnable fine-tuned Qwen
model until a real preference/RL trainer produces a better candidate.

## Current Simulated-Pylon Real Fine-Tune Run

After bundle 008, the lane ran an actual local MLX LoRA fine-tune sequence
with three simulated logical Pylons on this Mac:

- run id:
  `qwen_legal_real_qwen35_08b_mlx_lora_harvey_mfn_simulated_pylons_2026_05_20_009`
- base model: `Qwen/Qwen3.5-0.8B`
- parent policy: 005 reward-refresh adapter
- source preference/RL bundle: 008
- data:
  `fixtures/qwen_legal/real_finetune/mlx_lora_harvey_mfn_simulated_pylons_2026_05_20_009`
- report:
  `fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_simulated_pylons_2026_05_20_009/report.json`
- report digest:
  `541ba01f6e5ecb22b07a83441d2b349ef7352c7f590705cbf9d880beb8ce6ffd`
- checker:
  `scripts/check-qwen35-08b-harvey-mfn-simulated-pylons-run.sh`

The three local logical Pylon phases were:

| Worker | Resume | Iterations | LR | Final Val | Adapter Digest |
| --- | --- | ---: | ---: | ---: | --- |
| `pylon.local.macos.mlx.sim.01.coverage` | 005 | 4 | `0.000002` | `2.036` | `201a2083883f8b6123d66e4317be69cc0b7e475395d742531f7f4421afcaf982` |
| `pylon.local.macos.mlx.sim.02.tool_discipline` | Pylon 01 | 4 | `0.000001` | `2.009` | `b592d4efccba0763b59a7d490346290f71f5f972f8a79460fc5c82d00dc6a3e0` |
| `pylon.local.macos.mlx.sim.03.score_push` | Pylon 02 | 6 | `0.000008` | `2.291` | `4c9e8981b74170f068ade64bba73fdbca313d5ef7eda3e8bf5905e1ad4b763fd` |

Both scored candidates were served through the MLX OpenAI-compatible server
and run through the Rust Harvey MFN slice:

- Pylon 02 final adapter: `submitted`, two tool receipts, one output artifact,
  `63 / 83`
- Pylon 03 score-push adapter: `submitted`, two tool receipts, one output
  artifact, `63 / 83`

Decision: this was a real fine-tune run, not a gate, and it successfully
avoided the `max_tokens` no-tool failure from 006/007. It did not improve over
005, so it is retained as empirical rejection evidence and not promoted.
Adapter 005 remains the current best Harvey-runnable local Qwen policy.

## Current No-Cheat Harvey Runner Result

Run 015 is retired as an invalid score. The runner added words to the model's
output file, so the `83 / 83` result only proves that inserted text can satisfy
the public scorer. It does not prove model quality and must not be used as a
benchmark improvement.

The Rust legal benchmark agent now rejects that design. Legacy marker metadata
is ignored for output mutation, and the old scaffold-transform checker was
deleted. The allowed no-cheat controls are:

- `max_output_tokens`, which changes only the provider request budget;
- `force_write_until_required_deliverables`, which keeps asking the model to
  write its own deliverable;
- `force_validate_after_write`, which requires model-authored validation
  before submission;
- `plain_text_tool_protocol`, which lets weak local models write JSON tool
  requests as text while the runner only executes the JSON the model wrote.

Current no-cheat evidence:

- 016: adapter 005, same Harvey MFN task, runner output mutation disabled,
  terminal state `submitted`, one model-written output artifact, two tool
  receipts, score `4 / 18` on a rubric-free MFN work-product proxy.
- 019: three public Harvey tasks, model-only and scaffold-assisted prompts
  side by side, runner output mutation disabled, no output artifacts produced,
  model-only average `1851` bps, scaffold-assisted average `1481` bps.
- 020: real local MLX LoRA fine-tune resumed from adapter 005 over clean
  no-cheat supervised tool trajectories, adapter digest
  `30ba107fe59d81a8871edc02aa25a56b7eb7bc126d2705bc6f515e601f6c27a1`,
  validation loss `3.083` to `2.232`.
- 025: the broad no-cheat suite rerun against adapter 020, no output artifacts
  produced, model-only average `1851` bps, scaffold-assisted average `1481`
  bps, signed delta `-370` bps.

Run checker:

```bash
scripts/check-qwen35-08b-harvey-no-cheat-suite-runs.sh
```

Current operating conclusion: adapter 020 learned the clean supervised data
distribution, but the broad suite did not improve. More prompt-only variants
are not the next useful move. The next useful move is preference/RL data over
model-written traces: use 016 and clean supervised trajectories as chosen
examples, use 010-014 plus 017-019 and 021-025 failures as rejected protocol
evidence, and train a policy that writes, validates, submits, and cites
sources without any runner-added answer text.

## Rust API

The implementation lives in:

- `crates/psionic-eval/src/legal_benchmark_reward_traces.rs`
- `crates/psionic-train/src/legal_dpo_cli.rs`
- `crates/psionic-train/src/qwen_legal_adapter_sft.rs`
- `crates/psionic-train/src/open_adapter.rs`
- `crates/psionic-train/src/train_runtime.rs`

The canonical lane id is:

```text
qwen_legal_adapter_sft_v1
```

The first target set is deliberately narrow:

```text
qwen3.5-4b.legal.lm_head_lora.v1
```

Only `lm_head` LoRA is admitted in the smoke. Attention and MLP LoRA targets
should be added after the benchmark record, checkpoint, export, and
score-import loop is proven end to end.

## Data Contract

The dataset binding must point at a `LegalBenchmarkTrainingRecordBundle` whose
record schema version is:

```text
psionic.legal_benchmark_training_record.v1
```

The trainer refuses dataset drift before training. Hidden benchmark criteria
must remain excluded from model-visible examples by the record exporter.

The run emits:

- base model id and served model id
- base artifact digest
- tokenizer contract digest
- prompt-template digest
- dataset digest
- eval-pack digest
- adapter artifact digest
- adapter identity digest
- final checkpoint id
- score-import bundle digest
- RL hillclimb plan digest
- RL benchmark readiness report digest
- RL optimization window report digest
- RL perfect-score push report digest

## Artifact Gate

The unit smoke does not require full Qwen weights. It uses
`SyntheticHiddenStateSmoke` and the explicit synthetic digest:

```text
sha256:synthetic-qwen35-4b-legal-smoke
```

Real-model execution must use `RealArtifactRequired`, a non-synthetic base
artifact digest, and an explicit materialized artifact path. The binding offers
`validate_real_artifact_materialized()` for local launch preflight.

This prevents silently swapping in another Qwen row or falling back from a real
model run to the synthetic smoke fixture.

## Local Smoke

Run the focused tests from the repo root:

```bash
cargo test -p psionic-eval --lib legal_reward
cargo test -p psionic-train --lib qwen_legal
cargo test -p psionic-train --lib legal_qwen36_dpo
cargo test -p psionic-train --lib live_rl_update
```

The `legal_reward` fixture builds verifier reward traces from run receipts,
score reports, answer-integrity reports, and output manifests. It checks that
missing required files receive low reward without exclusion, harness-created
answers are excluded, and replay produces the same trace digest and JSONL bytes.
The `qwen_legal` fixture runs a four-step deterministic adapter update,
exports a loadable LM-head LoRA artifact, saves a checkpoint, restores from a
midpoint checkpoint, emits an Autopilot4 score-import bundle, and materializes
the next-phase RL hillclimb plan plus local benchmark reports. The
`legal_qwen36_dpo` fixture loads a parent SFT adapter, renders legal DPO pairs
through the Qwen3.6 direct-answer template, runs chosen/rejected weighted
adapter updates, and checks that the final adapter improves the synthetic
chosen-over-rejected margin. The `live_rl_update` fixture materializes rollout
evidence and promotes a new revision only when teacher-logprob alignment is
valid.

Run the Rust-only command smoke from the repo root:

```bash
cargo run -p psionic-train -- dpo \
  --config configs/legal/qwen36_dpo_smoke.json
```

The command bootstraps the parent SFT adapter from
`configs/legal/qwen36_sft_smoke.json` when the local target artifact is
missing, then writes `adapter.safetensors`, `loss_curve.json`,
`checkpoint_summary.json`, and `training_receipt.json` under
`target/legal/qwen36_dpo_smoke`.

The 2026-05-20 local command run completed 6 Rust-only DPO steps over 22
checked smoke pairs. Synthetic preference accuracy moved from `0.59090906` to
`0.95454544`; average chosen-minus-rejected logprob margin moved from
`0.3191057` to `4.2714095`. This proves the DPO smoke trainer can push the
adapter toward file-writing answers over chat-only answers on the synthetic
surface. It is not a score claim on private Harvey tasks.

The DPO adapter was also accepted by the deterministic replay harness:
`harvey_public_three_deterministic_replay_v1` reported base `3333` bps,
adapter `10000` bps, delta `6667` bps, and report hash
`bd01ce5a8653414a2189d935c80c835c774f55f195ed6809021c135a352faa66`. Treat
that as replay compatibility evidence only.

Build reward traces for GRPO-style training with:

```bash
cargo run -p psionic-eval --example legal_benchmark_build_reward_traces -- \
  --runs target/legal/reward_trace_public_three \
  --out target/legal/reward_trace_public_three/legal-reward-v1.jsonl
```

The 2026-05-20 local public-three replay generated 6 traces from base and
adapter task runs. The builder wrote dataset hash
`e85352b832dceed154282ff5e6b0de510b6af808b0cce945140387b7710a3e26`, with
0 fatal exclusions. Adapter-pass traces scored total reward `12`; base
missing-answer and tool-failure traces scored `0`. The reward is computed from
receipts, answer integrity, score reports, and output manifests only. Source-use
reward requires model/run evidence such as document-root tool calls or
run-record coverage. Hidden scoring leakage and harness-created answers are
fatal exclusions.

The 2026-05-20 local run executed the broader filters with binary targets
included:

- `cargo test -p psionic-eval legal_reward`: 3 passed
- `cargo test -p psionic-train qwen_legal`: 14 passed
- `cargo test -p psionic-train legal_qwen36_dpo`: 1 passed
- `cargo test -p psionic-train live_rl_update`: 2 passed

Those are still local training/RL substrate tests. They are not score claims on
private Harvey tasks.

## RL Hillclimb Plan

The smoke lane now emits
`QwenLegalRlHillclimbPlan` with schema
`psionic.qwen_legal_rl_hillclimb_plan.v1`. This does not claim a retained
Harvey score, but it gives Pylon/Nexus a concrete plan for the next phase:

- retain `Qwen/Qwen3.5-4B` as the smallest smoke lane while targeting
  `Qwen/Qwen3.6-35B-A3B` for retained scoring
- require a retained 20-task Harvey slice before any public retained-score
  claim
- collect at least 60 accepted legal-agent rollouts with no more than 12
  quarantined rollouts in the window
- connect the run to
  `blueprint://harvey_legal_qwen_optimizer_frontier/optimizer_frontier_001`
- assign failure families to GRPO, GEPA trace selection, MIPRO prompt search,
  and supervised fine-tune refresh work

The current target families are:

| Failure family | Method | Blueprint module |
| --- | --- | --- |
| `document_coverage` | MIPRO prompt search | `harvey_legal.document_inventory` |
| `citation_evidence` | GEPA trace selection | `harvey_legal.evidence_mapping` |
| `legal_reasoning` | GRPO | `harvey_legal.issue_fact_extraction` |
| `spreadsheet_reasoning` | GRPO | `harvey_legal.evidence_mapping` |
| `missing_fact` | supervised fine-tune refresh | `harvey_legal.issue_fact_extraction` |
| `pre_submit_self_check` | GEPA trace selection | `harvey_legal.final_self_check` |

Each target carries a reward signal, baseline miss value, target lift value,
and dataset request ref. Operators should treat the plan as a routing and
admission contract for Pylon work, not as proof that RL training has already
improved the retained score.

## Offline RL Benchmark Report

The lane now also emits `QwenLegalRlBenchmarkReadinessReport` with schema
`psionic.qwen_legal_rl_benchmark_report.v1`. It turns the plan into the local
benchmark numbers Autopilot4 can publish safely:

- baseline retained slice: 5260 bps
- phase-two conservative target: 7000 bps
- unconstrained projection if all family lifts land: above 9000 bps
- minimum accepted rollouts: 60
- quarantine budget: 12
- required method mix: GRPO, GEPA trace selection, MIPRO prompt search, and
  supervised fine-tune refresh
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-002`

This report is still local readiness evidence. A retained score claim requires
the actual retained 20-task run, immutable score reports, and Autopilot4
release-gate approval.

## Phase-Three RL Optimization Window

The lane also emits `QwenLegalRlOptimizationWindowReport` with schema
`psionic.qwen_legal_rl_optimization_window.v1`. It consumes the phase-two
readiness report and the Blueprint shadow-eval shortlist:

- phase-two target carried forward: 7000 bps
- phase-three conservative target: 7800 bps
- accepted rollout minimum: 84
- quarantine budget: 16
- holdout regression allowance: 0 bps
- Blueprint shortlist ref:
  `blueprint://harvey_legal_qwen_phase_three_shadow_eval_shortlist/optimizer_shortlist.harvey_legal_qwen.phase_003.shadow_eval`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-003`

This is the next work packet for Pylon/Nexus. It is still not a live Harvey
score claim; it is the bounded local benchmark target and evidence contract
that the retained run must satisfy.

## Phase-Four Perfect-Score Push

The lane now also emits `QwenLegalRlPerfectScorePushReport` with schema
`psionic.qwen_legal_rl_perfect_score_push.v1`. It consumes the phase-three
optimization window and the Blueprint perfect-score push plan:

- phase-three target carried forward: 7800 bps
- phase-four conservative target: 8500 bps
- accepted rollout minimum: 140
- quarantine budget: 20
- holdout regression allowance: 0 bps
- calibrated judge disagreement budget: 75 bps
- family coverage: all nine Blueprint optimizer frontier families
- Blueprint plan ref:
  `blueprint://harvey_legal_qwen_phase_four_perfect_score_push_plan/optimizer_plan.harvey_legal_qwen.phase_004.perfect_score_push`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-004`

This is the next bounded work packet before any perfect-score campaign. It adds
deliverable completeness, fine-tune data selection, and task-intake routing to
the phase-three RL window while preserving judge-adjudication and scorecard
requirements for every family.

## Phase-Five Retained Rehearsal

The lane now also emits `QwenLegalRlRetainedRehearsalReport` with schema
`psionic.qwen_legal_rl_retained_rehearsal.v1`. It consumes the phase-four
perfect-score push report and the Blueprint retained rehearsal plan:

- phase-four target carried forward: 8500 bps
- phase-five conservative target: 9000 bps
- retained rehearsal task-runs: 60
- accepted rollout minimum: 194
- quarantine budget: 24
- adversarial holdout task-runs: 36
- holdout regression allowance: 0 bps
- calibrated judge disagreement budget: 50 bps
- family coverage: all nine Blueprint optimizer frontier families
- Blueprint plan ref:
  `blueprint://harvey_legal_qwen_phase_five_retained_rehearsal_plan/optimizer_plan.harvey_legal_qwen.phase_005.retained_rehearsal`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-005`

This is the first high-confidence rehearsal gate after the 8500 bps target. It
requires three retained-slice passes, an adversarial holdout, and a tighter
judge panel before Autopilot4 should import a candidate as ready for a public
retained campaign.

## Phase-Six Expanded Corpus Dry Run

The lane now also emits `QwenLegalRlExpandedCorpusReport` with schema
`psionic.qwen_legal_rl_expanded_corpus.v1`. It consumes the phase-five
retained rehearsal report and the Blueprint expanded corpus plan:

- phase-five target carried forward: 9000 bps
- phase-six conservative target: 9500 bps
- expanded stratified slice: 125 Harvey tasks
- practice-area coverage: all 24 audited practice areas
- accepted rollout minimum: 266
- quarantine budget: 30
- adversarial holdout task-runs: 72
- holdout regression allowance: 0 bps
- calibrated judge disagreement budget: 35 bps
- family coverage: all nine Blueprint optimizer frontier families
- Blueprint plan ref:
  `blueprint://harvey_legal_qwen_phase_six_expanded_corpus_plan/optimizer_plan.harvey_legal_qwen.phase_006.expanded_corpus`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-006`

This dry run expands beyond the retained 20-task slice before any all-task
campaign. It is readiness evidence for a broader benchmark run, not a retained
score claim.

## Phase-Seven Full-Corpus Matrix Dry Run

The lane now also emits `QwenLegalRlFullCorpusMatrixReport` with schema
`psionic.qwen_legal_rl_full_corpus_matrix.v1`. It consumes the phase-six
expanded corpus report and the Blueprint full-corpus matrix plan:

- phase-six target carried forward: 9500 bps
- phase-seven conservative target: 9800 bps
- full corpus: 1251 Harvey tasks
- practice-area coverage: all 24 audited practice areas
- Qwen/Blueprint/RL matrix cells: 48
- accepted rollout minimum: 410
- quarantine budget: 38
- adversarial holdout task-runs: 144
- holdout regression allowance: 0 bps
- calibrated judge disagreement budget: 25 bps
- family coverage: all nine Blueprint optimizer frontier families
- Blueprint plan ref:
  `blueprint://harvey_legal_qwen_phase_seven_full_corpus_matrix_plan/optimizer_plan.harvey_legal_qwen.phase_007.full_corpus_matrix`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-007`

This dry run is the first local matrix contract over the full Harvey corpus. It
is still readiness evidence only; public score claims require imported score
reports and release-gate approval.

## Phase-Eight Residual Burn-Down Dry Run

The lane now also emits `QwenLegalRlResidualBurnDownReport` with schema
`psionic.qwen_legal_rl_residual_burn_down.v1`. It consumes the phase-seven
full-corpus matrix report and the Blueprint residual burn-down plan:

- phase-seven target carried forward: 9800 bps
- phase-eight conservative target: 9900 bps
- full corpus: 1251 Harvey tasks
- practice-area coverage: all 24 audited practice areas
- Qwen/Blueprint/RL matrix cells: 96
- residual miss cluster budget: 24
- accepted rollout minimum: 626
- quarantine budget: 48
- adversarial holdout task-runs: 288
- holdout regression allowance: 0 bps
- calibrated judge disagreement budget: 15 bps
- family coverage: all nine Blueprint optimizer frontier families
- Blueprint plan ref:
  `blueprint://harvey_legal_qwen_phase_eight_residual_burn_down_plan/optimizer_plan.harvey_legal_qwen.phase_008.residual_burn_down`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-008`

This dry run narrows the final residual error budget before any perfect-score
push. It is still readiness evidence only; public score claims require imported
score reports and release-gate approval.

## Phase-Nine Final-Campaign Rehearsal Dry Run

The lane now also emits `QwenLegalRlFinalCampaignReport` with schema
`psionic.qwen_legal_rl_final_campaign.v1`. It consumes the phase-eight residual
burn-down report and the Blueprint final-campaign rehearsal plan:

- phase-eight target carried forward: 9900 bps
- phase-nine conservative target: 9950 bps
- full corpus: 1251 Harvey tasks
- practice-area coverage: all 24 audited practice areas
- Qwen/Blueprint/RL matrix cells: 144
- residual miss cluster budget: 12
- human-adjudicated sample task-runs: 96
- accepted rollout minimum: 950
- quarantine budget: 52
- adversarial holdout task-runs: 432
- holdout regression allowance: 0 bps
- calibrated judge disagreement budget: 10 bps
- family coverage: all nine Blueprint optimizer frontier families
- Blueprint plan ref:
  `blueprint://harvey_legal_qwen_phase_nine_final_campaign_plan/optimizer_plan.harvey_legal_qwen.phase_009.final_campaign_rehearsal`
- export ref:
  `autopilot4://benchmarks/harvey/progress/phase-009`

This dry run is the last rehearsal before a separate perfect-score campaign.
It is still readiness evidence only; public score claims require imported score
reports and release-gate approval.

## Pylon Network SFT Smoke

The lane now has a first network-shaped Qwen legal training result:
`qwen_legal_pylon_network_sft_v1`.

This is not the final Qwen3.6 fine-tune. It is the smallest honest replacement
for the CS336-first proof path:

- objective id: `harvey_legal_qwen_finetune_v1`
- parent lane id: `qwen_legal_adapter_sft_v1`
- base model binding: `Qwen/Qwen3.5-4B`
- retained target model: `Qwen/Qwen3.6-35B-A3B`
- artifact mode: `synthetic_hidden_state_smoke`
- contributors: two logical Pylon workers
- aggregation rule: `trusted_weighted_lora_factor_average_v1`
- aggregate artifact:
  `fixtures/qwen_legal/pylon_network_sft/aggregate-qwen-legal-lm-head-lora.safetensors`
- report:
  `fixtures/qwen_legal/pylon_network_sft/pylon_network_sft_report_v1.json`
- generator:
  `crates/psionic-train/examples/qwen_legal_pylon_network_sft_fixture.rs`
- checker:
  `scripts/check-qwen-legal-pylon-network-sft.sh`

The retained report records per-contributor assignment ids, worker ids, node
pubkeys, shard refs, sample ids, training losses, checkpoint digests, adapter
artifact digests, contribution receipt digests, aggregate receipt digest,
model-progress participant count, and aggregate adapter identity digest.

The aggregate adapter is loadable as an LM-head LoRA safetensors artifact. The
two contributors train different legal smoke shards and produce different
adapter digests before trusted aggregation. That gives the legal lane a real
multi-contributor trained artifact while keeping the claim boundary explicit:
it proves Pylon-network training shape and aggregation, not real Qwen3.6
full-weight fine-tuning or a score lift on private Harvey tasks.

Run it locally with:

```bash
scripts/check-qwen-legal-pylon-network-sft.sh
```

## Distributed LoRA Merge Command

The Pylon worker aggregation path now has a reusable command:

```bash
cargo run -p psionic-train -- merge-lora \
  --manifest merge/legal-sft-round-001.json
```

The manifest is intentionally small and explicit. It names the parent adapter
hash, the base model binding, each worker adapter artifact, each worker's
dataset shard hash, each worker's token count, and the output adapter path.
The command verifies every worker artifact hash before loading it, refuses
shape drift, merges real LoRA safetensors factors, writes the aggregate
adapter, runs the local Rust eval suite when requested, and writes a
machine-readable merge receipt next to the output adapter.

Two merge modes are supported:

- `delta_averaging`: workers start from the same parent adapter and the
  aggregator averages LoRA factors by token count.
- `shard_sequential_training`: workers form a chain and the output is the last
  adapter after validating that each worker continued from the previous
  adapter hash.

The smoke manifest uses the two retained Pylon worker adapters from
`fixtures/qwen_legal/pylon_network_sft/`, writes the aggregate to
`target/legal/qwen_lora_merge/legal-sft-round-001/`, and checks the aggregate
against the retained digest
`8e8dea3bc639ed2c147d6901f6ceda9b5f1a176034dc7bb65219daf7dd33116d`.
It then runs `suites/harvey_public_three.json` locally and records a promotion
gate. The gate is simple: the merged adapter is only promotable if it beats
the declared champion score on the same local suite and has no integrity,
tool, or timeout failures. The command does not mutate the adapter registry;
registry promotion remains a separate operator action.

## Runtime Admission

`train_runtime.rs` admits the lane as a CUDA adapter-training machine lane:

```text
release_id: psionic-train.qwen_legal_adapter_sft.release.v1
environment_ref: psionic.environment.qwen_legal_adapter_sft.cuda.operator@v1
backend_family: cuda
topology_class: single_host_cuda_adapter_smoke
minimum_machine_class: strong_cuda_trainer
```

Pylon/Nexus dispatch should use this lane only for smoke jobs until the real
artifact gate and retained eval import are wired into the operator path.
