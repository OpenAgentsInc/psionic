# Legal Fine-Tuning Milestone 001

## Status

- candidate promoted: `true`
- champion score: `3333` bps
- candidate score: `10000` bps
- delta: `6667` bps
- candidate answer files: `10000` bps
- candidate integrity failures: `0`
- Python invoked: `false`
- harness answer text injected: `false`

## What Improved

- Legal score moved from 3333 bps to 10000 bps on the same frozen three-task public suite.
- Answer-file success moved from 3333 bps to 10000 bps.
- The candidate wrote the required answer file on all three tasks with valid answer integrity.
- The previous missing-answer and write-tool-failure cases are gone in this local replay.

## What Did Not Improve

- This does not prove performance on private Harvey tasks.
- This does not prove live Qwen legal reasoning quality; the eval is a deterministic public replay fixture.
- This does not yet replace the larger Pylon-distributed fine-tuning run.

## Exact Scores

| model | score bps | answer-file bps | integrity failures | tool failures | timeout failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| champion | 3333 | 3333 | 2 | 1 | 0 |
| candidate | 10000 | 10000 | 0 | 0 | 0 |

## Exact Failures

Champion failures:

- `harvey.public.lease_notice`: outcome `MissingAnswer`, score `0` bps, classes `integrity_failure, missing_answer`
- `harvey.public.purchase_indemnity`: outcome `ToolFailure`, score `0` bps, classes `integrity_failure, tool_failure`

Candidate failures:

- none

## Promotion

- eval gate decision: `Promote`
- registry decision: `Promote`
- promotion receipt: `target/legal/legal-ft-milestone-001/registry/promotion_harvey_public_three_deterministic_replay_v1_qwen36-27b-legal-ft-milestone-001.json`

## Why This Is Honest

- The suite hash is frozen in the report before training and eval.
- The SFT adapter is produced by the Rust Psionic trainer; the training receipt says python_invoked=false.
- The answer writer no longer appends suite, model, or prompt metadata to answer files.
- Promotion happens through the Qwen legal adapter registry only after the candidate beats the champion on the same suite hash.
- The report is explicit that this is a public local milestone, not proof on private benchmark tasks.

## Receipts

- suite: `suites/harvey_public_three.json`
- suite hash: `c30e4db622aa6f7a9e16a058b5579d1233a140ee5aa34243a4d152e4b641649a`
- SFT dataset receipt: `a86794afd1c0504ce182c8884db5045566dd2c69e0748ea042092f9a269d9204`
- SFT training receipt: `8ea5f750d3593fe8a005d4d8690f0e8535986818b16063a529882ec7bb280521`
- eval report: `target/legal/legal-ft-milestone-001/eval/eval_report.json`
- eval report hash: `f88ede10f7cebbcbecfb67eb5dec732a57fbb9311b0c110a9a35abe6740d1e58`
- registry: `target/legal/legal-ft-milestone-001/registry/registry.json`
- all artifacts have receipts: `true`
- report digest: `0c65502a09bac4423f2b991e5e0c014b4ac6982259bd5c3a81757844423acd5c`

| artifact | sha256 | receipt | receipt sha256 |
| --- | --- | --- | --- |
| `target/legal/legal-ft-milestone-001/candidate_adapter/adapter.safetensors` | `3566320d28c66cb7e769ccd932dfe3806a35b59942c0ce465ea3943b0d1e5094` | `target/legal/legal-ft-milestone-001/candidate_adapter/training_receipt.json` | `46142daa9ffe70dc580b19d3b740b91d6ed23b0f402577b2f8b367342b118e47` |
| `target/legal/legal-ft-milestone-001/registry/candidate_registration_receipt.json` | `d5df465018e9b4aa94dddb7315a9f3391dfcb56a7b17678e5793fbf7b3fb3a42` | `target/legal/legal-ft-milestone-001/registry/candidate_registration_receipt.json` | `d5df465018e9b4aa94dddb7315a9f3391dfcb56a7b17678e5793fbf7b3fb3a42` |
| `target/legal/legal-ft-milestone-001/candidate_adapter/checkpoint_summary.json` | `00df034d9b1b0e271162aa3f0df18fb9a47850aa31c72bdf24d564859da749b9` | `target/legal/legal-ft-milestone-001/candidate_adapter/training_receipt.json` | `46142daa9ffe70dc580b19d3b740b91d6ed23b0f402577b2f8b367342b118e47` |
| `target/legal/legal-ft-milestone-001/eval/eval_report.json` | `214f6569826a4d54e5f1e575365b84b06596cc17e6c8aa8161c39d1dc5a39a1b` | `target/legal/legal-ft-milestone-001/eval/replay_receipt.json` | `147399ed36409bb5379d44ca50713ff75ed9f1ba240d9649393f5002d339d28f` |
| `target/legal/legal-ft-milestone-001/candidate_adapter/loss_curve.json` | `af253110647854af55a97b5f61b10b6ba338741ec655914884484c89eecac36c` | `target/legal/legal-ft-milestone-001/candidate_adapter/training_receipt.json` | `46142daa9ffe70dc580b19d3b740b91d6ed23b0f402577b2f8b367342b118e47` |
| `target/legal/legal-ft-milestone-001/eval/promotion_gate_input.json` | `ebe7f72e5f58121c022a9676fb9d91a019ddd52046567f4a43068477c437831a` | `target/legal/legal-ft-milestone-001/eval/replay_receipt.json` | `147399ed36409bb5379d44ca50713ff75ed9f1ba240d9649393f5002d339d28f` |
| `target/legal/legal-ft-milestone-001/registry/promotion_harvey_public_three_deterministic_replay_v1_qwen36-27b-legal-ft-milestone-001.json` | `877e9dcf47a194abbc9ab6265afc0385f12ad51cab6885cfd61ae2dce2a6a3dc` | `target/legal/legal-ft-milestone-001/registry/promotion_harvey_public_three_deterministic_replay_v1_qwen36-27b-legal-ft-milestone-001.json` | `877e9dcf47a194abbc9ab6265afc0385f12ad51cab6885cfd61ae2dce2a6a3dc` |
| `target/legal/legal-ft-milestone-001/sft/sft_config.json` | `971dbd455c989530434a3931757faf9764743916314e9c6a60dcaccbb5e36d81` | `target/legal/legal-ft-milestone-001/sft/sft_dataset_receipt.json` | `b0f4c6fac40635bb7c6295bbb078506375545aed1ef7e2c77c5acba265d026b1` |
| `target/legal/legal-ft-milestone-001/sft/legal-public-three-sft.jsonl` | `ed5f3ae5ae2c26a705bf71b5b22e65c9922365b6ce3dc7b91a32453e3f615dc6` | `target/legal/legal-ft-milestone-001/sft/sft_dataset_receipt.json` | `b0f4c6fac40635bb7c6295bbb078506375545aed1ef7e2c77c5acba265d026b1` |
