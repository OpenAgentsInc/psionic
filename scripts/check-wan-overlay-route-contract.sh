#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/wan_overlay_route_contract_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/wan_overlay_route_contract.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/wan_overlay_route_contract_v1.json"
cargo run -q -p psionic-train --bin wan_overlay_route_contract -- "${generated_path}" >/dev/null

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
    fail("wan overlay route contract check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.wan_overlay_route_contract.v1":
    fail("wan overlay route contract check: schema_version drifted")
if committed["contract_id"] != "psionic.wan_overlay_route_contract.v1":
    fail("wan overlay route contract check: contract_id drifted")
if len(committed["nat_postures"]) != 4:
    fail("wan overlay route contract check: expected four nat postures")
if len(committed["route_quality_samples"]) != 4:
    fail("wan overlay route contract check: expected four route-quality samples")
if len(committed["route_records"]) != 4:
    fail("wan overlay route contract check: expected four route records")
if len(committed["failover_receipts"]) != 1:
    fail("wan overlay route contract check: expected one failover receipt")

transports = {route["selected_transport"] for route in committed["route_records"]}
if transports != {"direct", "relayed", "overlay_tunnel"}:
    fail("wan overlay route contract check: direct, relayed, and overlay transports must all remain present")

failover = committed["failover_receipts"][0]
if failover["next_route_id"] != "route.public_miner.local_rtx4080_local_mlx.overlay_failover":
    fail("wan overlay route contract check: failover target drifted")

summary = {
    "verdict": "verified",
    "contract_id": committed["contract_id"],
    "contract_digest": committed["contract_digest"],
    "route_ids": [route["route_id"] for route in committed["route_records"]],
}
print(json.dumps(summary, indent=2))
PY
