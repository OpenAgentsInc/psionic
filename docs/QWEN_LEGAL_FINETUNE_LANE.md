# Qwen Legal Adapter Fine-Tune Lane

> Status: implemented smoke lane for `psionic-train` on 2026-05-19; real
> local Qwen/MLX LoRA plus Rust legal-agent smoke added on 2026-05-20; local
> RL-seed resumed Qwen LoRA added on 2026-05-20; public Harvey MFN
> training-slice LoRA and Rust task run added on 2026-05-20; local MFN
> reward-refresh LoRA and `63 / 83` public training-slice run added on
> 2026-05-20; no-cheat runner correction, single-task run 016, broad suite
> runs 019/025, and adapter 020 added on 2026-05-20.

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
but it is still not the retained model target. It is single local-worker SFT,
not RL, not live Nexus settlement, and not a retained Harvey score claim.

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
workspace and output roots and tune the tool-use policy on retained slices.
The result exports one canonical legal benchmark training record and is usable
as a seed trajectory for legal RL ingestion, but it is not itself an RL-trained
model update or a retained Harvey score claim.

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
multi-Pylon RL, not a retained Harvey score, and not Qwen3.6.

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
retained Harvey leaderboard score.

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
surface. It is not a retained Harvey score claim.

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

Those are still local training/RL substrate tests. They are not retained Harvey
score claims.

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
full-weight fine-tuning or retained Harvey score lift.

Run it locally with:

```bash
scripts/check-qwen-legal-pylon-network-sft.sh
```

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
