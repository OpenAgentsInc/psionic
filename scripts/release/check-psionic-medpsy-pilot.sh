#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

required_files=(
  "docs/NON_GPT_OSS_MEDPSY_QWEN_PILOT.md"
  "docs/MEDPSY_BENCHMARK.md"
  "fixtures/medpsy/capability/medpsy_admission_policy_v1.json"
  "fixtures/medpsy/capability/medpsy_capability_matrix_v1.json"
  "fixtures/medpsy/benchmarks/medpsy_comparator_matrix_20260511_local.json"
  "crates/psionic-models/src/medpsy_qwen3.rs"
  "crates/psionic-serve/examples/medpsy_bench.rs"
)

for rel in "${required_files[@]}"; do
  if [[ ! -f "${repo_root}/${rel}" ]]; then
    printf 'missing required MedPsy pilot file: %s\n' "${rel}" >&2
    exit 1
  fi
done

python3 - <<'PY' "${repo_root}"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cap = json.loads((root / "fixtures/medpsy/capability/medpsy_capability_matrix_v1.json").read_text())
admission = json.loads((root / "fixtures/medpsy/capability/medpsy_admission_policy_v1.json").read_text())
matrix = json.loads((root / "fixtures/medpsy/benchmarks/medpsy_comparator_matrix_20260511_local.json").read_text())

assert cap["schema"] == "psionic.medpsy.capability_matrix.v1"
assert cap["release_gate"]["competitive_claim_allowed"] is False
assert cap["release_gate"]["clinical_claim_allowed"] is False
assert admission["medical_policy"]["direct_diagnosis_allowed"] is False
assert admission["medical_policy"]["prescribing_or_treatment_authority_allowed"] is False
assert matrix["schema"] == "psionic.medpsy.comparator_matrix.v1"
assert matrix["decision"]["can_claim_competitive_medpsy"] is False
assert any(row["runtime"] == "psionic" and row["status"] == "completed" for row in matrix["rows"])
assert any(row["runtime"] == "llama.cpp" and row["status"] == "timeout" for row in matrix["rows"])
PY

cargo test -p psionic-models medpsy -- --nocapture
cargo check -p psionic-serve --example medpsy_bench

printf 'MedPsy bounded pilot gate passed.\n'
