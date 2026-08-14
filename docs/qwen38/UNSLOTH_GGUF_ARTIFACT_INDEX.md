# Unsloth Qwen3.8-27B GGUF Artifact Index

> Status: `implemented` for source inventory, selected local artifacts, and R5
> native storage admission on 2026-08-14. Token generation remains `planned`.

This index pins `unsloth/Qwen3.8-27B-GGUF` at revision
`fdd03b8bbd279c1694563650e79d85a2373d9934`. It records the complete source
tree and the deliberately bounded local materialization used by the first
Psionic text lane.

The local artifact root is:

```text
target/models/qwen/unsloth/Qwen3.8-27B-GGUF
```

That directory is ignored through `/target/`. Model bytes are not committed.
All links below use the pinned revision rather than upstream `main`.

## Contents

- [Local materialization](#local-materialization)
- [Repository metadata](#repository-metadata)
- [Primary and comparison artifacts](#primary-and-comparison-artifacts)
- [Other standard quantizations](#other-standard-quantizations)
- [Other Dynamic quantizations](#other-dynamic-quantizations)
- [Vision projectors](#vision-projectors)
- [BF16 GGUF shards](#bf16-gguf-shards)
- [Psionic consumption order](#psionic-consumption-order)

## Local Materialization

Disk preflight reported 361 GiB free on `/home`. Downloading every quantized
variant plus BF16 and vision companions would consume most of that capacity
and would not improve the first implementation gate. The local materialization
therefore contains only these four upstream files:

| File | Bytes | Identity | Purpose |
| --- | ---: | --- | --- |
| [`.gitattributes`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/.gitattributes) | 3,175 | Git blob `5b36aa9079a4e144a6547dcb8b8e7b417933d78a` | LFS transport rules; not a runtime input. |
| [`README.md`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/README.md) | 6,583 | Git blob `992ca5f077d4514e193063181de944c594033f89` | Quantization catalog, usage guidance, and published size labels. |
| [`config.json`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/config.json) | 3,760 | Git blob `c2bb5cf2a82d965d5b11aa078ba9571fe949a4d5` | Companion Transformers topology. Psionic still treats the pinned official Qwen repository as architecture authority. |
| [`Qwen3.8-27B-UD-Q3_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-UD-Q3_K_XL.gguf) | 13,441,059,904 | SHA-256 `00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2` | Primary text-generation candidate. R5 admits its actual tensor types, metadata, tokenizer, template, converter layout, MTP disposition, and 4,096-token CUDA preflight. |

The Hugging Face CLI creates `.cache/huggingface/` download state inside the
local root. It is not part of the upstream tree or the Psionic artifact
contract.

The materialization and verification commands were:

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q3_K_XL.gguf README.md config.json .gitattributes \
  --revision fdd03b8bbd279c1694563650e79d85a2373d9934 \
  --local-dir target/models/qwen/unsloth/Qwen3.8-27B-GGUF

hf cache verify unsloth/Qwen3.8-27B-GGUF \
  --revision fdd03b8bbd279c1694563650e79d85a2373d9934 \
  --local-dir target/models/qwen/unsloth/Qwen3.8-27B-GGUF

sha256sum \
  target/models/qwen/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf
```

Hugging Face verification checked the four selected upstream files. It warned
that 24 remote files were intentionally absent and that local CLI cache state
was not present upstream. The independent SHA-256 matched the pinned LFS
digest exactly.

The first local execution lane uses only the primary weight artifact. The two
comparison weights are materialized for R5 qualification but remain outside
the primary CUDA residency claim. Both vision projectors and both BF16 shards
remain excluded. R5 admits storage and memory preflight only; generation starts
in R6.

## Repository Metadata

| Entry | Purpose |
| --- | --- |
| [`.gitattributes`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/.gitattributes) | Declares GGUF and other large-file transport through Git LFS. |
| [`README.md`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/README.md) | Repository card describing the conversion and quantization catalog. Published rounded sizes are discovery data; exact byte sizes below control materialization. |
| [`config.json`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/config.json) | Transformers-compatible companion configuration. It is checked against, not substituted for, the official pinned Qwen configuration. |
| [`BF16/`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/fdd03b8bbd279c1694563650e79d85a2373d9934/BF16) | Directory containing the two full-precision GGUF shards. |

## Primary And Comparison Artifacts

| File | Bytes | SHA-256 | Psionic role |
| --- | ---: | --- | --- |
| [`Qwen3.8-27B-UD-Q3_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-UD-Q3_K_XL.gguf) | 13,441,059,904 | `00cf92e666c6af6566996c38c89a44ccdb6449ea25ef0f112a452c853b2a71e2` | Downloaded primary target. The profile name does not prove concrete tensor-type support. |
| [`Qwen3.8-27B-Q3_K_M.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-Q3_K_M.gguf) | 13,818,690,528 | `7f3b845b563888ec3abc269474cf744bf703a7ce8766dbb7f696c63975facfd7` | Downloaded standard K-quant compatibility and output-quality baseline. |
| [`Qwen3.8-27B-Q4_K_M.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-Q4_K_M.gguf) | 17,106,775,008 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | Downloaded CPU-offload quality comparator; not a replacement for the primary CUDA residency target. |

## Other Standard Quantizations

These variants are indexed for future qualification. None is downloaded or
admitted by the first lane.

| File | Bytes | SHA-256 | Profile purpose |
| --- | ---: | --- | --- |
| [`Qwen3.8-27B-IQ4_NL.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-IQ4_NL.gguf) | 16,337,628,128 | `466c6714b0eca21c032690c801391a3c1e8f464ef01bbf420b70840027590c38` | Importance-matrix IQ4 non-linear profile. Requires independent IQ4_NL loader/runtime admission. |
| [`Qwen3.8-27B-IQ4_XS.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-IQ4_XS.gguf) | 15,705,861,088 | `9fd40d7036f5e0918e20aaeebf11468fafd06bb53d4d980eef6bb7e4e4ace666` | Smaller IQ4 profile. Requires independent IQ4_XS admission. |
| [`Qwen3.8-27B-Q3_K_S.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-Q3_K_S.gguf) | 12,574,489,568 | `0fc041075efd255732ce6de77617ac31520b35a8dbffc06ef56cb80e5c8762ca` | Smaller standard Q3 K-quant profile. |
| [`Qwen3.8-27B-Q4_0.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-Q4_0.gguf) | 16,056,478,688 | `ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d` | Legacy uniform Q4_0 compatibility profile. |
| [`Qwen3.8-27B-Q4_1.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-Q4_1.gguf) | 17,540,705,248 | `3e020514545c310dfc511dc8d3ddc23482b645189cb9287816d84bed6eddd4ac` | Legacy Q4_1 compatibility profile. |
| [`Qwen3.8-27B-Q4_K_S.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-Q4_K_S.gguf) | 16,121,359,328 | `22200efcd98a7aeeaf83f59b0f1400b055d9e0437900e26b930ef2d42a3eb3f9` | Smaller standard Q4 K-quant profile. |
| [`Qwen3.8-27B-Q5_K_M.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-Q5_K_M.gguf) | 19,834,055,648 | `07deb7fa91bf751d3000774fe5bb8afae5ffb41255fd19980147468052e07177` | Medium standard Q5 K-quant profile. |
| [`Qwen3.8-27B-Q5_K_S.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-Q5_K_S.gguf) | 19,270,036,448 | `a272bd49a992e38c2aa30216a966c9d1334afff0a0812b9837bb222e31d14b00` | Smaller standard Q5 K-quant profile. |
| [`Qwen3.8-27B-Q6_K.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-Q6_K.gguf) | 22,884,408,288 | `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` | Standard Q6 K-quant profile with a larger memory requirement. |
| [`Qwen3.8-27B-Q8_0.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-Q8_0.gguf) | 29,047,086,048 | `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348` | Q8_0 profile for higher-quality, higher-residency comparisons. |

## Other Dynamic Quantizations

The `UD` filenames describe Unsloth Dynamic profiles. They do not identify all
concrete GGML tensor types stored inside each file. Every future artifact must
pass the same tensor-table and converter-provenance gate as the primary file.

| File | Bytes | SHA-256 | Profile purpose |
| --- | ---: | --- | --- |
| [`Qwen3.8-27B-UD-IQ2_M.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-UD-IQ2_M.gguf) | 10,319,907,904 | `04a89ef4fa9c8726d09331433346809bbab692b4851d49d0738ba8d58a1ae740` | Dynamic IQ2 medium profile. |
| [`Qwen3.8-27B-UD-IQ2_XXS.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-UD-IQ2_XXS.gguf) | 9,010,048,064 | `8d1b37297d6cf98303cd396896f35e01089ddcc904053a9c6997f7a1c35b8524` | Smallest published dynamic profile in this pinned tree. |
| [`Qwen3.8-27B-UD-IQ3_XXS.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-UD-IQ3_XXS.gguf) | 11,913,559,104 | `0a6129dcbbbe72f423dc67e0e3bbfbbdf3e923981a3637687ebb96a46c59d6be` | Dynamic IQ3 profile. |
| [`Qwen3.8-27B-UD-Q2_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-UD-Q2_K_XL.gguf) | 10,676,423,744 | `46151b52a5cad673d90a00222103254864326c251130b8fc4381d6f34386b3c8` | Dynamic Q2 K-quant profile. |
| [`Qwen3.8-27B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-UD-Q4_K_XL.gguf) | 17,923,394,624 | `bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372` | Dynamic Q4 K-quant profile. |
| [`Qwen3.8-27B-UD-Q5_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-UD-Q5_K_XL.gguf) | 20,218,178,624 | `176a6a3f034e9cdc447c10cd00329fc9b31002e6589b9295f2ad4f1eefe0f6ab` | Dynamic Q5 K-quant profile. |
| [`Qwen3.8-27B-UD-Q6_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-UD-Q6_K_XL.gguf) | 25,924,152,384 | `739202186fd9389bb58497c58b56c8a0d4253d99d20131e6a0427e363e678fc8` | Dynamic Q6 K-quant profile. |
| [`Qwen3.8-27B-UD-Q8_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/Qwen3.8-27B-UD-Q8_K_XL.gguf) | 31,457,991,680 | `af36ecb6b5db1407953345b746c14ac93f0657dda413910b4348683a2d990377` | Largest published Dynamic profile in this pinned tree. |

## Vision Projectors

Both projector choices are excluded from the first text lane. R11 must select,
download, digest-bind, inspect, and memory-qualify one before native media
support can be claimed.

| File | Bytes | SHA-256 | Purpose |
| --- | ---: | --- | --- |
| [`mmproj-BF16.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/mmproj-BF16.gguf) | 931,146,432 | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` | BF16 vision encoder/projector companion. |
| [`mmproj-F16.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/mmproj-F16.gguf) | 927,607,488 | `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e` | F16 vision encoder/projector companion. |

## BF16 GGUF Shards

The BF16 GGUF export is not downloaded. The already verified official
safetensors repository remains the source-tensor authority for R1-R4 and
sampled converter checks.

| File | Bytes | SHA-256 | Purpose |
| --- | ---: | --- | --- |
| [`BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf) | 49,986,159,616 | `b9966e82b7a4d87028b5eae061d578ee826305ebf8baea5bfc6e09bad0ba191f` | First BF16 GGUF shard. |
| [`BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/fdd03b8bbd279c1694563650e79d85a2373d9934/BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf) | 4,671,576,000 | `92e3943c4f9bd6292a7bef82369f65fed9bfed088b9df0fb2fa2ce17c9edfa02` | Second BF16 GGUF shard. |

## Psionic Consumption Order

1. Bind repository id, immutable revision, exact filename, byte size, and
   SHA-256 before opening the GGUF.
2. Compare `config.json` and GGUF family metadata with the pinned official
   configuration. The official artifact controls disagreement.
3. Parse the GGUF metadata and tensor table without allocating model storage.
   Record architecture, model name, context, RoPE/MRoPE, tokenizer,
   pre-tokenizer, template, tensor names, shapes, types, and MTP inventory.
4. Validate converter provenance and sampled transforms against the official
   BF16 source. Unknown V-head layout refuses.
5. Map every concrete stored type to an implemented loader and native CPU/CUDA
   runtime. A profile label cannot satisfy this gate. R5 covers the
   materialized artifacts only.
6. Build explicit weights, KV, recurrent-state, scratch, graph, and allocator
   memory plans for each admitted context and backend.
7. Admit standard text generation while explicitly skipping and reporting MTP
   tensors. Vision projectors remain absent and media inputs refuse.
8. Materialize comparison artifacts only from a reviewed exact download plan
   when their R5 or R13 evidence is ready to run.
