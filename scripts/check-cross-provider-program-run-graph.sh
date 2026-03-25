#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_path="${repo_root}/fixtures/training/cross_provider_program_run_graph_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/cross_provider_program_run_graph.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/cross_provider_program_run_graph_v1.json"
cargo run -q -p psionic-train --bin cross_provider_program_run_graph -- "${generated_path}" >/dev/null

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
    fail("cross-provider whole-program run graph check: committed fixture drifted from generator output")

if fixture["schema_version"] != "psionic.cross_provider_program_run_graph.v1":
    fail("cross-provider whole-program run graph check: schema version drifted")

classes = {participant["execution_class"] for participant in fixture["participants"]}
expected = {
    "dense_full_model_rank",
    "validated_contributor_window",
    "validator",
    "checkpoint_writer",
    "eval_worker",
    "data_builder",
}
if classes != expected:
    fail("cross-provider whole-program run graph check: participant execution-class coverage drifted")

role_window = fixture["role_windows"][0]
if len(fixture["role_windows"]) != 1:
    fail("cross-provider whole-program run graph check: role-window count drifted")
if len(role_window["active_dense_rank_participant_ids"]) != 8:
    fail("cross-provider whole-program run graph check: dense-rank count drifted")
if len(role_window["active_contributor_window_participant_ids"]) != 3:
    fail("cross-provider whole-program run graph check: contributor-window count drifted")
if len(role_window["active_validator_participant_ids"]) != 1 or len(role_window["quarantined_validator_participant_ids"]) != 1:
    fail("cross-provider whole-program run graph check: validator split drifted")
if len(role_window["active_checkpoint_writer_participant_ids"]) != 1 or len(role_window["standby_checkpoint_writer_participant_ids"]) != 1:
    fail("cross-provider whole-program run graph check: checkpoint-writer split drifted")
if len(role_window["active_eval_worker_participant_ids"]) != 2:
    fail("cross-provider whole-program run graph check: eval-worker count drifted")
if len(role_window["active_data_builder_participant_ids"]) != 2:
    fail("cross-provider whole-program run graph check: data-builder count drifted")

if fixture["run_id"] != fixture["orchestrator"]["run"]["run_id"]:
    fail("cross-provider whole-program run graph check: run id drifted from orchestrator run")

summary = {
    "verdict": "verified",
    "participant_count": len(fixture["participants"]),
    "transition_count": len(fixture["transition_log"]),
    "dense_rank_count": len(role_window["active_dense_rank_participant_ids"]),
    "execution_class_count": len(classes),
}
print(json.dumps(summary, indent=2))
PY
