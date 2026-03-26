#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/decentralized_network_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/decentralized_network_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/decentralized_network_contract_v1.json"
cargo run -q -p psionic-train --bin decentralized_network_contract -- "${generated_path}" >/dev/null

python3 - "${contract_path}" "${generated_path}" <<'PY'
import json
import sys
from pathlib import Path

committed = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
generated = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


if committed != generated:
    fail("decentralized network contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.decentralized_network_contract.v1":
    fail("decentralized network contract check: schema_version drifted")
if committed["contract_id"] != "psionic.decentralized_network_contract.v1":
    fail("decentralized network contract check: contract_id drifted")
if committed["network_id"] != "psionic.decentralized_training.testnet.v1":
    fail("decentralized network contract check: network_id drifted")
if committed["governance_revision"]["governance_revision_id"] != "psionic.decentralized_training_governance.v1":
    fail("decentralized network contract check: governance_revision_id drifted")
if committed["governance_revision"]["registration_mode"] != "permissioned_testnet":
    fail("decentralized network contract check: registration_mode drifted")
if committed["settlement_backend"]["backend_kind"] != "signed_ledger_bundle":
    fail("decentralized network contract check: settlement backend drifted")
if committed["checkpoint_authority_policy"]["promotion_mode"] != "multi_validator_quorum":
    fail("decentralized network contract check: checkpoint promotion mode drifted")
expected_roles = {
    "public_miner",
    "public_validator",
    "relay",
    "checkpoint_authority",
    "aggregator",
}
if {binding["role_class"] for binding in committed["role_bindings"]} != expected_roles:
    fail("decentralized network contract check: role binding set drifted")
relay_binding = next(
    (binding for binding in committed["role_bindings"] if binding["role_class"] == "relay"),
    None,
)
if relay_binding is None or relay_binding["binding_kind"] != "network_only_support_role":
    fail("decentralized network contract check: relay binding drifted")
if committed["authority_paths"]["reference_doc_path"] != "docs/DECENTRALIZED_NETWORK_CONTRACT_REFERENCE.md":
    fail("decentralized network contract check: reference_doc_path drifted")
if committed["authority_paths"]["check_script_path"] != "scripts/check-decentralized-network-contract.sh":
    fail("decentralized network contract check: check_script_path drifted")

summary = {
    "verdict": "verified",
    "network_id": committed["network_id"],
    "contract_digest": committed["contract_digest"],
    "role_classes": [binding["role_class"] for binding in committed["role_bindings"]],
    "registration_mode": committed["governance_revision"]["registration_mode"],
}
print(json.dumps(summary, indent=2))
PY
