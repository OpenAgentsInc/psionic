#!/usr/bin/env bash
# Philox 4x32-10 seeded work-class RNG parity check (psionic#1116).
#
# 1. Recomputes the published random123 known-answer vectors and the committed
#    determinism-receipt digest with an independent Python reimplementation,
#    so the fixture is checked without trusting the Rust code under test.
# 2. Runs the pinned Rust tests in psionic-core.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/rng/philox4x32_reference_vectors.json"

python3 - "${fixture_path}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

fixture = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


if fixture["schema_version"] != "psionic.core.philox_rng.v1":
    fail("philox check: schema_version drifted")
if fixture["contract_id"] != "psionic.core.philox_rng.v1":
    fail("philox check: contract_id drifted")
if fixture["algorithm"] != "philox4x32-10":
    fail("philox check: algorithm drifted")

M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
MASK = 0xFFFFFFFF


def philox_round(c, k):
    p0 = M0 * c[0]
    p1 = M1 * c[2]
    return [
        ((p1 >> 32) ^ c[1] ^ k[0]) & MASK,
        p1 & MASK,
        ((p0 >> 32) ^ c[3] ^ k[1]) & MASK,
        p0 & MASK,
    ]


def philox4x32_10(counter, key):
    c, k = list(counter), list(key)
    for r in range(10):
        c = philox_round(c, k)
        if r < 9:
            k = [(k[0] + W0) & MASK, (k[1] + W1) & MASK]
    return c


def words(values):
    return [int(v, 16) for v in values]


vectors = fixture["reference_vectors"]
if len(vectors) != 3:
    fail("philox check: expected the three published random123 vectors")
for vector in vectors:
    got = philox4x32_10(words(vector["counter"]), words(vector["key"]))
    if got != words(vector["expected"]):
        fail(f"philox check: known-answer vector mismatch: {vector}")

receipt = fixture["determinism_receipt"]
seed = int(receipt["seed"], 16)
streams = receipt["streams"]
draws = receipt["draws_per_stream"]


def u64_at(stream, index):
    ctr = index // 2
    block = philox4x32_10(
        [ctr & MASK, (ctr >> 32) & MASK, stream & MASK, (stream >> 32) & MASK],
        [seed & MASK, (seed >> 32) & MASK],
    )
    lane = (index % 2) * 2
    return (block[lane + 1] << 32) | block[lane]


hasher = hashlib.sha256()
for stream in range(streams):
    for counter in range(draws):
        hasher.update(u64_at(stream, counter).to_bytes(8, "little"))
if hasher.hexdigest() != receipt["sha256"]:
    fail("philox check: determinism-receipt digest mismatch")

print("philox check: fixture parity ok (independent reimplementation)")
PY

(cd "${repo_root}" && cargo test -q -p psionic-core philox)
echo "philox check: psionic-core philox tests ok"
