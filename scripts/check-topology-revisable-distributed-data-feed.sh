#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/topology_revisable_distributed_data_feed_report_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/topology_revisable_distributed_data_feed.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/topology_revisable_distributed_data_feed_report_v1.json"
cargo run -q -p psionic-data --bin topology_revisable_distributed_data_feed_report -- "${generated_path}" >/dev/null

python3 - "${fixture_path}" "${generated_path}" <<'PY'
import json
import sys
from pathlib import Path

fixture_path = Path(sys.argv[1])
generated_path = Path(sys.argv[2])

fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

if fixture != generated:
    fail("topology-revisable data-feed check: committed fixture drifted from generator output")

if fixture["current_scope_window"] != "psionic_topology_revisable_distributed_data_feed_v1":
    fail("topology-revisable data-feed check: scope window drifted")

cases = {case["case_id"]: case for case in fixture["cases"]}
replace_provider = cases.get("dense_rank.provider_loss.replace_rank2")
replace_node = cases.get("dense_rank.node_loss.replace_rank1")
refused = cases.get("dense_rank.shrink_world.refused")

if replace_provider is None or replace_provider["status"] != "supported":
    fail("topology-revisable data-feed check: missing supported provider-loss replacement case")
if replace_node is None or replace_node["status"] != "supported":
    fail("topology-revisable data-feed check: missing supported node-loss replacement case")
if refused is None or refused["status"] != "refused":
    fail("topology-revisable data-feed check: missing refused shrink-world case")
if not replace_provider["reassigned_shard_keys"]:
    fail("topology-revisable data-feed check: provider-loss replacement lost reassigned shard evidence")
if replace_provider["baseline_global_order_digest"] != replace_provider["revised_global_order_digest"]:
    fail("topology-revisable data-feed check: provider-loss replacement drifted replay order")
if "requested_world_size" not in refused["refusal"]:
    fail("topology-revisable data-feed check: shrink-world refusal detail drifted")

summary = {
    "verdict": "verified",
    "case_count": len(fixture["cases"]),
    "supported_cases": [
        case["case_id"]
        for case in fixture["cases"]
        if case["status"] == "supported"
    ],
}
print(json.dumps(summary, indent=2))
PY
