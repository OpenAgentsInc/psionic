#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/cross_provider_runtime_binder_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/cross_provider_runtime_binder.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/cross_provider_runtime_binder_v1.json"
cargo run -q -p psionic-train --bin cross_provider_runtime_binder -- "${generated_path}" >/dev/null

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
    fail("runtime binder check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.cross_provider_runtime_binder.v1":
    fail("runtime binder check: schema version drifted")
if len(fixture["binding_records"]) != 4:
    fail("runtime binder check: binding record count drifted")

binding_ids = {binding["binding_id"] for binding in fixture["binding_records"]}
if len(binding_ids) != 4:
    fail("runtime binder check: binding ids lost uniqueness")

adapter_kinds = {binding["adapter_kind"] for binding in fixture["binding_records"]}
expected_adapters = {
    "google_host",
    "google_configured_peer_cluster",
    "run_pod_remote_pod",
    "local_trusted_lan_bundle",
}
if adapter_kinds != expected_adapters:
    fail(f"runtime binder check: adapter set drifted: {sorted(adapter_kinds)}")

for binding in fixture["binding_records"]:
    hook_kinds = {hook["hook_kind"] for hook in binding["provider_hooks"]}
    if "launch" not in hook_kinds or "finalizer" not in hook_kinds and "evidence_seal" not in hook_kinds:
        fail(f"runtime binder check: binding {binding['binding_id']} lost launch/finalizer coverage")
    env_names = {entry["name"] for entry in binding["bound_runtime_env"]}
    required_envs = {
        "PSION_PROGRAM_MANIFEST_ID",
        "PSION_PROGRAM_MANIFEST_DIGEST",
        "PSION_RUN_ID",
        "PSION_EXECUTION_CLASS",
        "PSION_RUN_ROOT",
        "PSION_LAUNCH_ROOT",
        "PSION_CHECKPOINT_ROOT",
        "PSION_METRICS_ROOT",
        "PSION_VISUALIZATION_ROOT",
        "PSION_FINAL_ROOT",
    }
    if not required_envs.issubset(env_names):
        fail(f"runtime binder check: binding {binding['binding_id']} lost one or more shared env vars")
    artifact_classes = {entry["artifact_class"] for entry in binding["bound_artifact_classes"]}
    if artifact_classes != {"checkpoint", "log_bundle", "metrics_bundle", "final_evidence_bundle"}:
        fail(f"runtime binder check: binding {binding['binding_id']} artifact coverage drifted")

summary = {
    "verdict": "verified",
    "binding_count": len(fixture["binding_records"]),
    "adapter_count": len(adapter_kinds),
    "source_count": len(fixture["admitted_source_ids"]),
}
print(json.dumps(summary, indent=2))
PY
