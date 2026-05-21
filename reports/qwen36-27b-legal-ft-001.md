# Qwen3.6-27B Legal Fine-Tuning Milestone 001

## Status

- model: `Qwen/Qwen3.6-27B`
- model load verified: `true`
- base score: `3333` bps
- promoted candidate: `qwen36_27b_sft_grpo_round_001`
- promoted score: `10000` bps
- score delta: `6667` bps
- decision: `Promote`
- no Python invoked: `true`
- hidden benchmark training: `false`
- all receipts present: `true`

## Candidate Ladder

| candidate | stage | score bps | answer-file bps | hard failures | python | adapter sha256 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `qwen36_27b_base` | `Base` | `3333` | `3333` | `3` | `false` | `none` |
| `qwen36_27b_sft_round_001` | `Sft` | `10000` | `10000` | `0` | `false` | `5e029be91ab5470d6abd63ad460f4d05f4b982f9be8276a97b88f542b872ba6d` |
| `qwen36_27b_sft_dpo_round_001` | `Dpo` | `10000` | `10000` | `0` | `false` | `12d2f1450441e7e86291e5dadd7b5c28b25b919c1dff09c0746bef1dafd4cc24` |
| `qwen36_27b_sft_grpo_round_001` | `Grpo` | `10000` | `10000` | `0` | `false` | `1b46e62898d32ee592f78efddc10b2d3786c92dbbafeb9cb7b0230542c82848d` |

## Target Artifacts

- target load report: `target/legal/qwen36-27b-legal-ft-001/qwen36_27b_target_load_report.json`
- target load report sha256: `4fe429f16d9cd4e91fffdd46564e9ea794cae47e28d03dc2169bebeed911eda7`
- base eval report hash: `876bea7dd9fbeedafc4a2b0e01b36485665250764c7f11ce4952dd373a213552`

| role | path | sha256 | bytes |
| --- | --- | --- | ---: |
| `model_config` | `fixtures/qwen36_27b_smoke/config.json` | `55c45950fd6c4d3395146e32788e619665c455b69794981ee098cb11a94a73b6` | `522` |
| `tokenizer` | `fixtures/qwen36_27b_smoke/tokenizer.json` | `b16a3f8344ab50941f925281fc25ee243f6b05509d07880fd41fbb6c4655fdfe` | `1832` |
| `safetensors_shard` | `target/legal/qwen36_27b_prompt_smoke/model-00001-of-00001.safetensors` | `1acc4b847e41c74995e889ac0722db3109eacda673bcd8633f228297d22941d5` | `224` |

## Data And Training

- suite: `suites/harvey_public_three.json`
- suite hash: `c30e4db622aa6f7a9e16a058b5579d1233a140ee5aa34243a4d152e4b641649a`
- SFT dataset: `target/legal/qwen36-27b-legal-ft-001/dataset/qwen36-27b-legal-ft-001-sft.jsonl`
- SFT dataset sha256: `c088d0c738c51103d8ef1476b5f02f80359078782590dbb94b766aa6f6d9b8f4`
- SFT dataset receipt: `b0ccf600fbf9cdf9ef09a510ced728fe61aee2023bc12c8d33ca12e6cf2be2a0`
- SFT config: `target/legal/qwen36-27b-legal-ft-001/sft_config.json`
- SFT config sha256: `96d1c36ffabc9dfe1be0399eb7bb8b7892a62f2ef2aafb286e2613027b6214d2`

## Promotion

- receipt: `target/legal/qwen36-27b-legal-ft-001/promotion_receipt.json`
- receipt digest: `1968ea399ff5a7be941568dbdb64092b973256f60e62a804358f1a1a2798e4fd`
- reason: qwen36_27b_sft_grpo_round_001 beats qwen36_27b_base by 6667 bps on the same public suite
- reason: stage tie-break prefers the latest successful optimization stage when scores tie

## Boundary

This is a Qwen3.6-27B target-path legal fine-tuning milestone over public training-allowed Harvey fixtures. It loads the Qwen3.6-27B smoke target artifacts, runs Rust SFT, DPO, and GRPO adapter updates, evaluates the candidate ladder, and records receipts. It does not claim full 27B weight loading, hidden Harvey performance, or production leaderboard standing.

## Report Receipt

- report digest: `1e1d9d203b3a3c7c33ddf1abeace28484318372c73355ee2f903d6951bcd8ce7`
