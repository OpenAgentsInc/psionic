# Psionic Ternary TQ-Class Formats

> Status: canonical `psionic#1115` record, 2026-06-12, after landing the
> owned TQ1_0/TQ2_0 ternary quantization formats with parity fixtures and
> schema-versioned cross-backend determinism receipts (CPU recorded;
> Metal/CUDA/Vulkan explicitly pending).

Authored by Fable (claude-fable-5) for psionic#1115.

## What This Is

Psionic owns a Rust implementation of the TQ1_0 (1.6875 bits per weight) and
TQ2_0 (2.0625 bits per weight) ternary `{-1, 0, +1}` block quantization
formats from the llama.cpp/BitNet lineage, the formats Tether's QVAC BitNet
lane serves through its Vulkan/Metal kernels.

- `crates/psionic-core/src/ternary.rs` — pack (quantize), unpack
  (dequantize), exact-integer trit extraction, the exact-integer
  ternary-times-i8 dot path, the committed workload, and the typed receipts
- `crates/psionic-core/src/bin/ternary_tq_fixture.rs` — fixture generator
- `fixtures/quant/ternary_tq_formats_v1.json` — committed parity fixture:
  layout spec, known-answer blocks, full packed workload bytes, digests, and
  per-backend determinism receipts
- `scripts/check-ternary-tq-formats.sh` — regenerates the fixture, then
  re-derives every packed byte and digest with an independent Python
  reimplementation of the published reference algorithms
- contract id: `psionic.core.ternary_tq_formats.v1`
- receipt schema: `psionic.core.ternary_tq_determinism_receipt.v1`

## Why Ternary Serving Formats

Tether's QVAC BitNet lane (`projects/tether/repos/qvac-rnd-fabric-llm-bitnet`,
read-only reference) publishes lossless-verification evidence for ternary
serving: 99.04% same-top-token rate, mean KL divergence below 0.0003, and
claimed bit-exact equivalence between its Vulkan TQ kernels and the CPU
reference. Those numbers are external reference claims, not assertions this
module makes. What they demonstrate is that an integer/ternary serving format
can make quantized inference outputs bit-reproducible across heterogeneous
backends.

That property changes the verification economics. Today the OpenAgents
verification map routes quantized-inference work to `seeded_replication`
plus statistical checks. A ternary serving lane whose cross-backend
determinism is proven with receipts moves those rows to
`deterministic_recompute` — digest spot-checks, the cheapest verification
rung. See openagents `docs/training/2026-06-10-qvac-edge-stack-analysis.md`
(sections 2.1 and 5) and
`docs/training/2026-06-10-psion-full-pipeline-buildout-plan.md` (sections 9
and 11). This serves `training.verification_classes.v1` and
`pylon.compute_revenue_modes.v1`.

Scope boundary: this is the SERVING format lane only. BitNet-b1.58 QAT
(training ternary models) stays a deferred derisking-ledger candidate on the
openagents side and is explicitly out of scope here.

## Wire Layout and Compatibility Verdict

Both formats are byte-compatible with the ggml `block_tq1_0` /
`block_tq2_0` wire layout (`QK_K = 256`), as specified by
`ggml/src/ggml-common.h` and the `quantize_row_tq{1,2}_0_ref` /
`dequantize_row_tq{1,2}_0` reference algorithms in
`ggml/src/ggml-quants.c` (read-only at `projects/repos/llama.cpp`, studied
and transcribed, never vendored).

| Format | Block | Body | Tail | Scale | Bits/weight |
| --- | --- | --- | --- | --- | --- |
| TQ1_0 | 54 B / 256 elems | `qs[48]`: 5 trits per byte, base-3 with ceiling-division encode (`(q*256 + 242)/243`) | `qh[4]`: 4 trits per byte, shifted to the high digits | fp16 LE at offset 52 | 1.6875 |
| TQ2_0 | 66 B / 256 elems | `qs[64]`: 2 bits per trit, element `g*128 + m + n*32` in bits `2n..=2n+1` of byte `g*32 + m` | — | fp16 LE at offset 64 | 2.0625 |

The compatibility pin has three independent legs in
`scripts/check-ternary-tq-formats.sh`:

1. Hand-derived wire bytes for the all-zero block (TQ1_0: 48 bytes of
   `0x80`, 4 bytes of `0x7F`, fp16 zero; TQ2_0: 64 bytes of `0x55`, fp16
   zero), computed from the published layout on paper, not from either
   implementation.
2. Known-answer blocks (`all_zero`, `unit_cycle`, `half_cycle`) with packed
   bytes pinned in the fixture.
3. A full Philox-derived workload whose packed bytes and digests are
   re-derived by an independent Python transcription of the reference
   algorithms.

What is not claimed: no GGUF file produced by upstream llama.cpp has been
decoded byte-for-byte against this implementation on this host. The
compatibility claim is at the algorithm-and-layout level, pinned by the
hand-derived bytes and the independent reimplementation. Decoding a real
upstream TQ GGUF is a recorded follow-up.

## Exact-Integer Versus Float, Stated Precisely

The determinism claim lives or dies on where floating point enters, so the
boundary is explicit:

| Step | Arithmetic | Order dependence |
| --- | --- | --- |
| Trit packing (base-3 ceil-div encode, 2-bit fields) | exact integer | none |
| Trit unpacking (`(byte * pow3[n] mod 256) * 3 >> 8`) | exact integer | none |
| Per-block trit×i8 dot accumulation (`isum`) | exact integer (i32) | none — the hot accumulation path is integer |
| f16 encode/decode of the block scale | exact integer bit manipulation (RNE encode, exact decode) | none |
| Block `amax` scan | f32 compares only | none — max is order-independent |
| `id = 1.0 / d` | one correctly-rounded IEEE-754 f32 division | none — elementwise |
| Trit selection (`lroundf(x * id)`) | one f32 multiply + round-half-away-from-zero per element | none — elementwise, no accumulation |
| Dequantized value (`trit * d`) | exact — multiplier is -1, 0, or +1 | none |
| Dot combine (`((isum as f32) * d) * activation_scale`, summed) | f32, ascending block order | pinned by specification; every backend must reproduce this order |

So: pack and unpack are deterministic on every conforming IEEE-754 host
because every float step is a single correctly-rounded elementwise
operation; the matmul hot path is pure integer; and the only accumulation
over floats (cross-block dot combine) is order-pinned, which makes it
deterministic by specification rather than by accident.

Non-finite inputs are refused with a typed error rather than encoded.

## Determinism Receipts

The receipt is a typed, schema-versioned record
(`TernaryDeterminismReceipt`, schema
`psionic.core.ternary_tq_determinism_receipt.v1`): same committed workload
in, same digests out, per backend. The committed workload derives 4 rows of
1024 f32 inputs and 4 rows of i8 activations from the house Philox 4x32-10
stream (psionic#1116, seed `0x11157E40C0DE2026`), then digests the packed
bytes, the dequantized f32 outputs, and the per-block i32 isums plus per-row
f32 dots for both formats.

CPU receipt (recorded for real on this host, an arm64 macOS machine, and
reproduced by the independent Python reimplementation in the check script):

```
tq1_0_packed_sha256      = 383a0236f3e9e39af5cfd3c0c0d58dbd3de862dfe82cffce1d54d87b5b6f8216
tq1_0_dequantized_sha256 = cb03951af1841cb688954f020fbd1f87e9171f707bef52b7c8735703fdf78770
tq1_0_dot_sha256         = f6e10e00863ad33e943d4e31677bce10f01287dc7819efcd5131ed163485ac69
tq2_0_packed_sha256      = d1cfd501516d50938bb5f7d5744cddbc5a233849afa05d4c4cfdf1a20fb83174
tq2_0_dequantized_sha256 = cb03951af1841cb688954f020fbd1f87e9171f707bef52b7c8735703fdf78770
tq2_0_dot_sha256         = f6e10e00863ad33e943d4e31677bce10f01287dc7819efcd5131ed163485ac69
```

The TQ1_0 and TQ2_0 dequantized and dot digests are identical by
construction: both formats encode exactly the same trits and the same fp16
scale, so their decoded outputs are bitwise equal — a cross-format
consistency property the fixture now pins.

Backend status:

| Backend | Status | Meaning |
| --- | --- | --- |
| cpu | recorded | digests above, produced by the owned Rust implementation and reproduced by the independent Python reimplementation |
| metal | pending | no Metal ternary kernel run exists; nothing is claimed |
| cuda | pending | no CUDA ternary kernel run exists; nothing is claimed |
| vulkan | pending | no Vulkan ternary kernel run exists; nothing is claimed (Vulkan is the QVAC-demonstrated cross-vendor target) |

A pending receipt is a typed contract, not evidence. A backend moves to
recorded only when the committed workload executes through a psionic kernel
on that backend and its digests are pinned from that run. Never claim a
backend that did not run.

## Verification-Class Consequence

With a recorded receipt per backend pair, quantized-inference work classes
become `deterministic_recompute` candidates on the OpenAgents verification
map: a validator on any conforming device recomputes the committed workload
(or a sampled slice of real serving work in this format) and compares
digests, instead of running seeded replication with statistical checks. The
CPU receipt plus this module is the substrate half of that move; the
cross-backend receipts are the missing half and stay honestly pending until
the kernels exist.

## Claim Boundary

From openagents `docs/training/2026-06-10-qvac-edge-stack-analysis.md`
section 5: BitNet is a trained, statistical model whose integer-quantized
forward pass happens to be reproducible. Quantization-fidelity metrics
(same-top-token rate, KL divergence) are not correctness proofs, and this
lane never borrows Tassadar exact-lane language. What this module claims is
reproducibility of the quantized forward path, recorded per backend in
typed receipts — nothing about model quality, and nothing exact-by-
construction about the model itself.

## Recorded Follow-Ups

1. Execute the committed workload through `psionic-backend-metal` and
   `psionic-backend-cuda` ternary kernels (which do not exist yet) and pin
   recorded receipts from those runs; Vulkan follows when psionic grows a
   Vulkan backend.
2. Decode a real upstream TQ1_0/TQ2_0 GGUF tensor against this
   implementation to upgrade the wire-compatibility pin from
   algorithm-level to artifact-level.
3. Wire `TernaryFormat` into `QuantizationMode` and the serving-path
   quantization ladder once a serving lane consumes these formats; the enum
   was deliberately not extended in this change because roughly ten crates
   match on it exhaustively and no serving path consumes ternary yet.
4. Register the `deterministic_recompute` candidacy on the OpenAgents
   verification map once at least one cross-backend receipt pair is
   recorded.
