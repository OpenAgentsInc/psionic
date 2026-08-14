# Qwen3.8 GGUF Download Plan

> Status: `implemented` on 2026-08-14 for the transfer plan. Artifact
> qualification and execution remain `planned`.

## Source

All planned files come from `unsloth/Qwen3.8-27B-GGUF` at immutable revision
`fdd03b8bbd279c1694563650e79d85a2373d9934`. Sizes and SHA-256 values are the
Hugging Face LFS facts returned for that revision.

The local root is:

```text
target/models/qwen/unsloth/Qwen3.8-27B-GGUF
```

This directory is ignored through `/target/`. Model weights are never
committed.

## Materialization Plan

| Artifact | Exact bytes | Expected SHA-256 | Classification | Local disposition |
| --- | ---: | --- | --- | --- |
| `Qwen3.8-27B-UD-Q3_K_XL.gguf` | 13,441,059,904 | `00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2` | Primary text artifact | Materialized and verified at the local root |
| `Qwen3.8-27B-Q3_K_M.gguf` | 13,818,690,528 | `7f3b845b563888ec3abc269474cf744bf703a7ce8766dbb7f696c63975facfd7` | Standard K-quant compatibility and quality baseline | Download to the local root with one worker, then verify size and SHA-256 |
| `Qwen3.8-27B-Q4_K_M.gguf` | 17,106,775,008 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | CPU-offload quality comparator | Download to the local root with one worker, then verify size and SHA-256 |

Each planned model is one GGUF file rather than a split artifact. The transfer
therefore has no additional weight shards.

## Excluded Companions

| Artifact | Exact bytes | Expected SHA-256 | Classification | Disposition |
| --- | ---: | --- | --- | --- |
| `mmproj-BF16.gguf` | 931,146,432 | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` | Vision projector | Excluded from the R5 text plan; reserved for R11 |
| `mmproj-F16.gguf` | 927,607,488 | `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e` | Vision projector | Excluded from the R5 text plan; reserved for R11 |

The repository has no separate MTP companion. Its single NextN layer is
embedded as `blk.64.*` in each text GGUF. R5 inventories those 15 tensors and
marks them skipped for standard generation. R9A owns MTP execution.

## Transfer Commands

Run transfers serially so disk use, progress, and failure identity remain
unambiguous:

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-Q3_K_M.gguf \
  --revision fdd03b8bbd279c1694563650e79d85a2373d9934 \
  --local-dir target/models/qwen/unsloth/Qwen3.8-27B-GGUF \
  --max-workers 1

hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-Q4_K_M.gguf \
  --revision fdd03b8bbd279c1694563650e79d85a2373d9934 \
  --local-dir target/models/qwen/unsloth/Qwen3.8-27B-GGUF \
  --max-workers 1
```

After each transfer, compare the observed byte size and SHA-256 with the table
before using the artifact for qualification or comparison.

## Admission Boundary

The profile name does not define the concrete tensor storage types. Psionic
must inspect each GGUF tensor table and refuse any required storage type that
lacks native loader and backend support. Download and digest verification do
not establish runtime admission.
