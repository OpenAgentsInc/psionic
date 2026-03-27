#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/curated_decentralized_run_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/curated_decentralized_run_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/curated_decentralized_run_contract_v1.json"
cargo run -q -p psionic-train --bin curated_decentralized_run_contract -- "${generated_path}" >/dev/null

python3 - "${contract_path}" "${generated_path}" <<'PY'
import json, sys
from pathlib import Path
committed = json.loads(Path(sys.argv[1]).read_text())
generated = json.loads(Path(sys.argv[2]).read_text())
assert committed == generated
assert len(committed["participants"]) == 4
print(json.dumps({"contract_digest": committed["contract_digest"], "verdict": "verified"}, indent=2))
PY
