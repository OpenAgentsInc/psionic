#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
contract_path="${repo_root}/fixtures/training/cross_provider_training_program_manifest_v1.json"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/cross_provider_training_program_manifest.XXXXXX")"
trap 'rm -rf -- "${tmpdir}"' EXIT

generated_path="${tmpdir}/cross_provider_training_program_manifest_v1.json"
cargo run -q -p psionic-train --bin cross_provider_training_program_manifest -- "${generated_path}" >/dev/null

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
    fail("cross-provider training program manifest check: committed fixture drifted from generator output")
if committed["schema_version"] != "psionic.cross_provider_training_program_manifest.v1":
    fail("cross-provider training program manifest check: schema_version drifted")
if committed["program_manifest_id"] != "psionic-cross-provider-pretraining-program-v1":
    fail("cross-provider training program manifest check: program_manifest_id drifted")
if committed["stage_authority"]["stage_id"] != "psion_pretrain":
    fail("cross-provider training program manifest check: stage_id drifted")
if committed["stage_authority"]["stage_kind"] != "pretrain":
    fail("cross-provider training program manifest check: stage_kind drifted")
if committed["checkpoint_family"] != "psion.cross_provider.pretrain.v1":
    fail("cross-provider training program manifest check: checkpoint_family drifted")
if committed["environment"]["environment_ref"] != "psion.pretrain.reference_runtime":
    fail("cross-provider training program manifest check: environment_ref drifted")
if committed["environment"]["version"] != "v1":
    fail("cross-provider training program manifest check: environment version drifted")
expected_sources = {
    "google_cloud",
    "runpod",
    "local_workstation",
    "trusted_lan_cluster",
}
if set(committed["admitted_compute_source_classes"]) != expected_sources:
    fail("cross-provider training program manifest check: compute-source class set drifted")
expected_execution = {
    "dense_full_model_rank",
    "validated_contributor_window",
    "validator",
    "checkpoint_writer",
    "eval_worker",
    "data_builder",
}
if set(committed["admitted_execution_classes"]) != expected_execution:
    fail("cross-provider training program manifest check: execution-class set drifted")
if committed["artifact_roots"]["final_root_template"] != "runs/${RUN_ID}/final":
    fail("cross-provider training program manifest check: final_root_template drifted")
if committed["authority_paths"]["reference_doc_path"] != "docs/TRAIN_PROGRAM_MANIFEST_REFERENCE.md":
    fail("cross-provider training program manifest check: reference_doc_path drifted")
if committed["authority_paths"]["check_script_path"] != "scripts/check-cross-provider-training-program-manifest.sh":
    fail("cross-provider training program manifest check: check_script_path drifted")
if committed["evidence_authority"]["evidence_family_id"] != "psionic.training_execution_evidence_bundle.v1.reserved":
    fail("cross-provider training program manifest check: evidence_family_id drifted")
if len(committed["baseline_artifacts"]) < 3:
    fail("cross-provider training program manifest check: baseline_artifacts shrank")

summary = {
    "verdict": "verified",
    "program_manifest_id": committed["program_manifest_id"],
    "program_manifest_digest": committed["program_manifest_digest"],
    "compute_source_classes": committed["admitted_compute_source_classes"],
    "execution_classes": committed["admitted_execution_classes"],
}
print(json.dumps(summary, indent=2))
PY
