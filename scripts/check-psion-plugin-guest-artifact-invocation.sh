#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo run -q -p psionic-runtime --example psion_plugin_guest_artifact_invocation

jq -e '
  .schema_version == "psionic.psion.plugin_guest_artifact_invocation.v1"
  and .tool_projection.tool_name == "plugin_example_echo_guest"
  and .success_case.status == "exact_success"
  and (.success_case.receipt_binding_preserved == true)
  and (.refusal_cases | length) >= 3
' fixtures/psion/plugins/guest_artifact/psion_plugin_guest_artifact_invocation_v1.json >/dev/null

echo "guest-artifact invocation bundle is present and receipt-equivalent"
