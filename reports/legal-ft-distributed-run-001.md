# Legal Distributed Fine-Tuning Run 001

## Status

- workers: `2`
- all worker receipts signed: `true`
- all worker outputs hash verified: `true`
- all worker payments payable: `true`
- champion score: `3333` bps
- candidate score: `10000` bps
- delta: `6667` bps
- promotion decision: `Promote`
- no Python in worker path: `true`
- hidden benchmark training: `false`

## What Ran

This run trained two local Pylon worker shards with the Rust Psionic legal SFT trainer. Each worker produced a LoRA adapter, a Psionic training receipt, a signed Pylon worker receipt, and a local payment decision. The aggregator merged the two adapters and evaluated the merged adapter on the same public three-task Harvey suite.

## Worker Table

| worker | shard | tasks | adapter sha256 | signed | output verified | payment |
| --- | --- | --- | --- | --- | --- | --- |
| `pylon.local.harvey-legal.01` | `shard.dataset.qwen-legal.harvey-public-three.distributed-001.1` | `harvey.public.lease_notice, harvey.public.privilege_log` | `99d9f0a5100acbd822611a4b2f3c89ad95b5dde5c9d99a32a65873ecc33e0e1a` | `true` | `true` | `Payable` |
| `pylon.local.harvey-legal.02` | `shard.dataset.qwen-legal.harvey-public-three.distributed-001.2` | `harvey.public.purchase_indemnity` | `c23dae04c1b533904c34a82a9c122aa0ee7af404d4f84ca328299d7bcbbd64bd` | `true` | `true` | `Payable` |

## Shard Table

| shard | samples | tokens | manifest | training receipt | worker receipt |
| --- | --- | ---: | --- | --- | --- |
| `shard.dataset.qwen-legal.harvey-public-three.distributed-001.1` | `harvey.public.lease_notice.distributed.sft, harvey.public.privilege_log.distributed.sft` | `96` | `376fe5aabc6a0bb9a74583f0bb202dd14dc7dca0bb7a4dd2711f62512ddd6b63` | `7810c3a0c42fe8f57487c48fcc474f023311a0a0d5ee3d7485ce56812b6522ed` | `f48ff4489f97157232f90f94428d67dc55b6c52e6828bfcab5b84b2d6103d9b5` |
| `shard.dataset.qwen-legal.harvey-public-three.distributed-001.2` | `harvey.public.purchase_indemnity.distributed.sft` | `38` | `7bb93b0d69a01a146d27dd405babfbe7e3559f6ae6b8f7092c0c2a39f775dda4` | `5c545425071c301d4d305e3dfcc7ec7a300a8dd06c38cb0d90a18b7cc24a1d1f` | `771e9e792f4a5752764108ee19d5bb6c75d3247a285c88b4e01f65561ae55ad4` |

## Adapter Merge

- merge manifest: `target/legal/legal-ft-distributed-run-001/merge/merge_manifest.json`
- merge manifest sha256: `be1fb7c70c9b0d45badc49ae86d1893dcea48a58cd95acad28696228d1e65b21`
- merge receipt: `target/legal/legal-ft-distributed-run-001/merge/qwen36-27b-legal-distributed-run-001.merge-receipt.json`
- merge receipt hash: `15a38aa2c79eae15bff0f99347cc4a0ee0542219c289163d12f7e939c9614b0a`
- merged adapter: `target/legal/legal-ft-distributed-run-001/merge/qwen36-27b-legal-distributed-run-001.safetensors`
- merged adapter sha256: `a66f97b6c69e5ac2d4022bc3405949cbc5fdc7f76c432b7aa7a6f4a63b2c90c7`

## Eval Result

- eval output dir: `target/legal/legal-ft-distributed-run-001/eval`
- eval report hash: `34403c9b431426d8074bfbe2ed245e7c28589ed8becc2b3a069742731b7557bf`
- candidate answer-file success: `10000` bps
- candidate integrity failures: `0`
- candidate tool failures: `0`
- candidate timeouts: `0`

## Promotion Decision

- decision: `Promote`
- champion score: `3333` bps
- candidate score: `10000` bps
- score delta: `6667` bps
- reason: candidate beats champion on the same local eval suite

## Payment And Budget Receipts

- payable total: `10000` micro-USD
- all worker payments payable: `true`

| worker | decision | digest | proof | amount micro-USD |
| --- | --- | --- | --- | ---: |
| `pylon.local.harvey-legal.01` | `target/legal/legal-ft-distributed-run-001/workers/pylon.local.harvey-legal.01/pylon/settlements/job.legal-ft-distributed-run-001.sft-shard-1.payment_decision.json` | `fcea1a645fb979adb1dfc176d7508897d2716bc5caa158b0e5dd3e7fdc5cc8d6` | `pending_payment_proof:ledger://local-smoke/qwen-legal-distributed:budget.qwen-legal.distributed-run-001:job.legal-ft-distributed-run-001.sft-shard-1` | `5000` |
| `pylon.local.harvey-legal.02` | `target/legal/legal-ft-distributed-run-001/workers/pylon.local.harvey-legal.02/pylon/settlements/job.legal-ft-distributed-run-001.sft-shard-2.payment_decision.json` | `a84b6987cbde19f7dcc382fc4b75cea570e782656fc28a879e2f5c4069f29e53` | `pending_payment_proof:ledger://local-smoke/qwen-legal-distributed:budget.qwen-legal.distributed-run-001:job.legal-ft-distributed-run-001.sft-shard-2` | `5000` |

## Boundary

This is a local two-worker Pylon/Psionic legal fine-tuning milestone over the public training-allowed Harvey three-task fixture. It proves Rust sharding, local worker SFT, signed receipts, payment decisions, adapter merge, and replay eval. It does not prove hidden Harvey benchmark performance or remote tailnet worker execution.

## Report Receipt

- report digest: `3ae5e9f5660af0a048971556014521ac0072eca430fb7926a287cd0b8d1dd9c2`
- all artifacts have receipts: `true`
