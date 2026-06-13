#!/usr/bin/env bash
# Ternary TQ-class format parity and determinism-receipt check (psionic#1115).
#
# 1. Regenerates the fixture with the owned Rust generator and compares it
#    against the committed copy.
# 2. Recomputes the known-answer blocks, the full packed workload bytes, and
#    every committed digest with an independent Python reimplementation of
#    the published ggml TQ1_0/TQ2_0 reference algorithms (and of the house
#    Philox 4x32-10 stream), so the fixture is checked without trusting the
#    Rust code under test.
# 3. Verifies the hand-derived all-zero wire bytes, the receipt statuses
#    (cpu recorded, metal/cuda/vulkan explicitly pending), and the layout
#    constants.
# 4. Runs the pinned Rust tests in psionic-core.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/quant/ternary_tq_formats_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/ternary_tq_formats.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/ternary_tq_formats_v1.json"
(cd "${repo_root}" && cargo run -q -p psionic-core --bin ternary_tq_fixture -- "${generated_path}")

python3 - "${fixture_path}" "${generated_path}" <<'PY'
import hashlib
import json
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path

committed = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
generated = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


if committed != generated:
    fail("ternary check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.core.ternary_tq_formats.v1":
    fail("ternary check: schema_version drifted")
if committed["contract_id"] != "psionic.core.ternary_tq_formats.v1":
    fail("ternary check: contract_id drifted")
if committed["issue"] != "psionic#1115":
    fail("ternary check: issue drifted")
if committed["doc"] != "docs/PSIONIC_TERNARY_TQ_FORMATS.md":
    fail("ternary check: doc path drifted")
if committed["check_script"] != "scripts/check-ternary-tq-formats.sh":
    fail("ternary check: check_script path drifted")

layout = committed["layout"]
if layout["block_elements"] != 256:
    fail("ternary check: block_elements drifted")
if (layout["tq1_0"]["bytes_per_block"], layout["tq1_0"]["qs_bytes"],
        layout["tq1_0"]["qh_bytes"], layout["tq1_0"]["scale_offset"]) != (54, 48, 4, 52):
    fail("ternary check: tq1_0 layout drifted")
if (layout["tq2_0"]["bytes_per_block"], layout["tq2_0"]["qs_bytes"],
        layout["tq2_0"]["qh_bytes"], layout["tq2_0"]["scale_offset"]) != (66, 64, 0, 64):
    fail("ternary check: tq2_0 layout drifted")

# --- exact f32 helpers (every op is one correctly-rounded IEEE-754 step) ---

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def f32_mul(a, b):
    # product of two f32 values is exact in f64, so one struct round = f32 mul
    return f32(a * b)


def f32_add(a, b):
    # sum of two f32 values is exact in f64
    return f32(a + b)


def f32_div(a, b):
    # correctly rounded positive f32 quotient via Fraction comparison,
    # immune to f64 double rounding
    naive = f32(a / b)
    target = Fraction(a) / Fraction(b)
    nb = f32_bits(naive)
    best = None
    best_err = None
    for cb in {nb, nb + 1, max(nb - 1, 0)}:
        c = bits_f32(cb)
        err = abs(Fraction(c) - target)
        if best is None or err < best_err or (err == best_err and (cb & 1) == 0):
            best, best_err = c, err
    return best


def f16_pack(x):
    # f64 holding an exact f32 value -> one RNE rounding to binary16
    return struct.pack("<e", x)


def f16_unpack(two_bytes):
    return f32(struct.unpack("<e", two_bytes)[0])


# --- independent Philox 4x32-10 (psionic#1116 layout) ---

M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
MASK = 0xFFFFFFFF


def philox4x32_10(counter, key):
    c, k = list(counter), list(key)
    for r in range(10):
        p0 = M0 * c[0]
        p1 = M1 * c[2]
        c = [
            ((p1 >> 32) ^ c[1] ^ k[0]) & MASK,
            p1 & MASK,
            ((p0 >> 32) ^ c[3] ^ k[1]) & MASK,
            p0 & MASK,
        ]
        if r < 9:
            k = [(k[0] + W0) & MASK, (k[1] + W1) & MASK]
    return c


def philox_block(seed, stream, counter):
    return philox4x32_10(
        [counter & MASK, (counter >> 32) & MASK, stream & MASK, (stream >> 32) & MASK],
        [seed & MASK, (seed >> 32) & MASK],
    )


def u32_at(seed, stream, index):
    return philox_block(seed, stream, index // 4)[index % 4]


def u64_at(seed, stream, index):
    block = philox_block(seed, stream, index // 2)
    lane = (index % 2) * 2
    return (block[lane + 1] << 32) | block[lane]


def unit_f64_at(seed, stream, index):
    return (u64_at(seed, stream, index) >> 11) / (1 << 53)


# --- independent TQ1_0 / TQ2_0 reimplementation (transcribed from the
#     published ggml-quants.c reference algorithms, read-only) ---

def encode_trit(value, inverse_scale):
    product = f32_mul(value, inverse_scale)
    # lroundf: round half away from zero (exact: f32 value held in f64)
    rounded = math.floor(product + 0.5) if product >= 0 else math.ceil(product - 0.5)
    return int(rounded) + 1


def block_scale_pair(block):
    amax = 0.0
    for value in block:
        amax = max(amax, abs(value))
    inverse = f32_div(1.0, amax) if amax != 0.0 else 0.0
    return amax, inverse


def quantize_tq1_block(block):
    amax, inverse = block_scale_pair(block)
    out = bytearray()
    for m in range(32):
        q = 0
        for n in range(5):
            q = q * 3 + encode_trit(block[m + n * 32], inverse)
        out.append((q * 256 + 242) // 243)
    for m in range(16):
        q = 0
        for n in range(5):
            q = q * 3 + encode_trit(block[160 + m + n * 16], inverse)
        out.append((q * 256 + 242) // 243)
    for j in range(4):
        q = 0
        for m in range(4):
            q = q * 3 + encode_trit(block[240 + j + m * 4], inverse)
        q *= 3
        out.append((q * 256 + 242) // 243)
    out += f16_pack(amax)
    return bytes(out)


def quantize_tq2_block(block):
    amax, inverse = block_scale_pair(block)
    out = bytearray()
    for group in range(2):
        for m in range(32):
            byte = 0
            for n in range(4):
                byte |= (encode_trit(block[group * 128 + m + n * 32], inverse) & 3) << (2 * n)
            out.append(byte)
    out += f16_pack(amax)
    return bytes(out)


TRIT_MULS = [1, 3, 9, 27, 81]


def unpack_tq1_trits(block54):
    trits = [0] * 256
    for m in range(32):
        for n in range(5):
            q = (block54[m] * TRIT_MULS[n]) & 0xFF
            trits[m + n * 32] = ((q * 3) >> 8) - 1
    for m in range(16):
        for n in range(5):
            q = (block54[32 + m] * TRIT_MULS[n]) & 0xFF
            trits[160 + m + n * 16] = ((q * 3) >> 8) - 1
    for j in range(4):
        for n in range(4):
            q = (block54[48 + j] * TRIT_MULS[n]) & 0xFF
            trits[240 + j + n * 4] = ((q * 3) >> 8) - 1
    return trits


def unpack_tq2_trits(block66):
    trits = [0] * 256
    for group in range(2):
        for m in range(32):
            byte = block66[group * 32 + m]
            for n in range(4):
                trits[group * 128 + m + n * 32] = ((byte >> (2 * n)) & 3) - 1
    return trits


def quantize_rows(values, rows, row_elements, block_fn, bytes_per_block):
    packed = bytearray()
    for r in range(rows):
        row = values[r]
        for b in range(row_elements // 256):
            packed += block_fn(row[b * 256:(b + 1) * 256])
    return bytes(packed)


# --- known-answer blocks ---

def kat_inputs(case_id):
    if case_id == "all_zero":
        return [0.0] * 256
    if case_id == "unit_cycle":
        return [[1.0, -1.0, 0.0, 1.0, -1.0][i % 5] for i in range(256)]
    if case_id == "half_cycle":
        return [f32(((i % 3) - 1.0) * 0.5) for i in range(256)]
    fail(f"ternary check: unknown known-answer case {case_id}")


kats = committed["known_answer_blocks"]
if [k["id"] for k in kats] != ["all_zero", "unit_cycle", "half_cycle"]:
    fail("ternary check: known-answer case set drifted")
for kat in kats:
    inputs = kat_inputs(kat["id"])
    if quantize_tq1_block(inputs).hex() != kat["tq1_0_packed_hex"]:
        fail(f"ternary check: tq1_0 known-answer mismatch for {kat['id']}")
    if quantize_tq2_block(inputs).hex() != kat["tq2_0_packed_hex"]:
        fail(f"ternary check: tq2_0 known-answer mismatch for {kat['id']}")

# Hand-derived ggml wire bytes for the all-zero block: every trit is 1
# (zero), so each tq1_0 qs byte is (121*256+242)//243 = 128, each qh byte is
# (120*256+242)//243 = 127, and each tq2_0 qs byte is 0b01010101 = 0x55.
if kats[0]["tq1_0_packed_hex"] != "80" * 48 + "7f" * 4 + "0000":
    fail("ternary check: all-zero tq1_0 wire bytes drifted from hand-derived layout")
if kats[0]["tq2_0_packed_hex"] != "55" * 64 + "0000":
    fail("ternary check: all-zero tq2_0 wire bytes drifted from hand-derived layout")

# --- the committed determinism workload ---

workload = committed["workload"]
seed = int(workload["seed"], 16)
rows = workload["rows"]
row_elements = workload["row_elements"]
if seed != 0x11157E40C0DE2026 or rows != 4 or row_elements != 1024:
    fail("ternary check: workload parameters drifted")

inputs = [
    [f32(2.0 * unit_f64_at(seed, r, i) - 1.0) for i in range(row_elements)]
    for r in range(rows)
]
activations = [
    [((u32_at(seed, 256 + r, i) >> 8) % 255) - 127 for i in range(row_elements)]
    for r in range(rows)
]
activation_scale = 0.015625

tq1_packed = quantize_rows(inputs, rows, row_elements, quantize_tq1_block, 54)
tq2_packed = quantize_rows(inputs, rows, row_elements, quantize_tq2_block, 66)
if tq1_packed.hex() != workload["tq1_0_packed_hex"]:
    fail("ternary check: tq1_0 workload packed bytes mismatch (independent reimplementation)")
if tq2_packed.hex() != workload["tq2_0_packed_hex"]:
    fail("ternary check: tq2_0 workload packed bytes mismatch (independent reimplementation)")


def workload_digests(packed, bytes_per_block, unpack_fn):
    blocks_per_row = row_elements // 256
    row_bytes = blocks_per_row * bytes_per_block
    packed_hash = hashlib.sha256()
    dequant_hash = hashlib.sha256()
    dot_hash = hashlib.sha256()
    for r in range(rows):
        row_packed = packed[r * row_bytes:(r + 1) * row_bytes]
        packed_hash.update(row_packed)
        acc = 0.0
        for b in range(blocks_per_row):
            block = row_packed[b * bytes_per_block:(b + 1) * bytes_per_block]
            d = f16_unpack(block[bytes_per_block - 2:bytes_per_block])
            trits = unpack_fn(block)
            isum = 0
            for i, trit in enumerate(trits):
                value = trit * d  # exact: multiplier is -1, 0, or +1
                dequant_hash.update(struct.pack("<f", value))
                isum += trit * activations[r][b * 256 + i]
            dot_hash.update(struct.pack("<i", isum))
            term = f32_mul(f32_mul(f32(float(isum)), d), activation_scale)
            acc = f32_add(acc, term)
        dot_hash.update(struct.pack("<f", acc))
    return packed_hash.hexdigest(), dequant_hash.hexdigest(), dot_hash.hexdigest()


digests = workload["digests"]
tq1_digests = workload_digests(tq1_packed, 54, unpack_tq1_trits)
tq2_digests = workload_digests(tq2_packed, 66, unpack_tq2_trits)
if tq1_digests != (digests["tq1_0_packed_sha256"], digests["tq1_0_dequantized_sha256"],
                   digests["tq1_0_dot_sha256"]):
    fail("ternary check: tq1_0 workload digests mismatch (independent reimplementation)")
if tq2_digests != (digests["tq2_0_packed_sha256"], digests["tq2_0_dequantized_sha256"],
                   digests["tq2_0_dot_sha256"]):
    fail("ternary check: tq2_0 workload digests mismatch (independent reimplementation)")

# --- receipts: cpu recorded for real, every other backend explicitly pending ---

receipts = committed["determinism_receipts"]
if [(r["backend"], r["status"]) for r in receipts] != [
    ("cpu", "recorded"), ("metal", "pending"), ("cuda", "pending"), ("vulkan", "pending")
]:
    fail("ternary check: receipt backend/status set drifted")
for receipt in receipts:
    if receipt["schema_version"] != "psionic.core.ternary_tq_determinism_receipt.v1":
        fail("ternary check: receipt schema_version drifted")
    if receipt["status"] == "recorded":
        if receipt.get("digests") != digests or receipt.get("pending_reason") is not None:
            fail("ternary check: recorded receipt payload drifted")
    else:
        if receipt.get("digests") is not None or not receipt.get("pending_reason"):
            fail("ternary check: pending receipt must carry a reason and no digests")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "tq1_0_packed_sha256": digests["tq1_0_packed_sha256"],
    "tq2_0_packed_sha256": digests["tq2_0_packed_sha256"],
    "receipts": [(r["backend"], r["status"]) for r in receipts],
}
print(json.dumps(summary, indent=2))
PY

(cd "${repo_root}" && cargo test -q -p psionic-core ternary)
echo "ternary check: psionic-core ternary tests ok"
