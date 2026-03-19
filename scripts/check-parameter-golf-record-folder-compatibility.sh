#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
input_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_record_folder_compatibility.json"
parameter_golf_root="~/code/parameter-golf"
submission_dir=""
report_path=""
track_id=""
skip_entrypoint_dry_run=0

usage() {
    cat <<'EOF' >&2
Usage: scripts/check-parameter-golf-record-folder-compatibility.sh [--input <path>] [--parameter-golf-root <path>] [--submission-dir <path>] [--track-id <track_id>] [--report <path>] [--skip-entrypoint-dry-run]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            [[ $# -ge 2 ]] || {
                echo "missing path after --input" >&2
                usage
                exit 1
            }
            input_path="$2"
            shift 2
            ;;
        --parameter-golf-root)
            [[ $# -ge 2 ]] || {
                echo "missing path after --parameter-golf-root" >&2
                usage
                exit 1
            }
            parameter_golf_root="$2"
            shift 2
            ;;
        --submission-dir)
            [[ $# -ge 2 ]] || {
                echo "missing path after --submission-dir" >&2
                usage
                exit 1
            }
            submission_dir="$2"
            shift 2
            ;;
        --track-id)
            [[ $# -ge 2 ]] || {
                echo "missing track id after --track-id" >&2
                usage
                exit 1
            }
            track_id="$2"
            shift 2
            ;;
        --report)
            [[ $# -ge 2 ]] || {
                echo "missing path after --report" >&2
                usage
                exit 1
            }
            report_path="$2"
            shift 2
            ;;
        --skip-entrypoint-dry-run)
            skip_entrypoint_dry_run=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

python3 - "$input_path" "$parameter_golf_root" "$submission_dir" "$track_id" "$report_path" "$skip_entrypoint_dry_run" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

INPUT_PATH = Path(os.path.expanduser(sys.argv[1]))
PARAMETER_GOLF_ROOT = Path(os.path.expanduser(sys.argv[2]))
SUBMISSION_DIR = Path(os.path.expanduser(sys.argv[3])) if sys.argv[3] else None
TRACK_ID = sys.argv[4]
REPORT_PATH = Path(os.path.expanduser(sys.argv[5])) if sys.argv[5] else None
SKIP_ENTRYPOINT_DRY_RUN = sys.argv[6] == "1"


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def stable_digest(payload: dict, prefix: str = "") -> str:
    material = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    hasher = hashlib.sha256()
    hasher.update(prefix.encode("utf-8"))
    hasher.update(material)
    return hasher.hexdigest()


def validate_report(data: dict) -> None:
    required_top = [
        "schema_version",
        "report_id",
        "compatibility_status",
        "challenge_repo_readme_ref",
        "psionic_parameter_golf_spec_ref",
        "baseline_submission_package_version",
        "baseline_submission_id",
        "baseline_record_folder_relpath",
        "required_submission_json_core_keys",
        "track_contracts",
        "required_top_level_files",
        "dependency_posture",
        "public_repo_allows_additional_folder_contents",
        "dependency_detail",
        "current_psionic_extra_submission_paths",
        "verifier_runner",
        "verifier_example_command",
        "claim_boundary",
        "report_digest",
    ]
    for key in required_top:
        if key not in data:
            fail(f"parameter golf record-folder compatibility error: missing top-level key `{key}`")

    if data["schema_version"] != 1:
        fail("parameter golf record-folder compatibility error: `schema_version` must be 1")
    if data["report_id"] != "parameter_golf.record_folder_compatibility.v1":
        fail("parameter golf record-folder compatibility error: unexpected `report_id`")
    if data["compatibility_status"] != "compatible":
        fail("parameter golf record-folder compatibility error: `compatibility_status` must be `compatible`")
    if data["challenge_repo_readme_ref"] != "README.md":
        fail("parameter golf record-folder compatibility error: unexpected challenge repo README ref")
    if data["psionic_parameter_golf_spec_ref"] != "PSIONIC_PARAMETER_GOLF_SPEC.md":
        fail("parameter golf record-folder compatibility error: unexpected Psionic spec ref")
    if data["dependency_posture"] != "folder_local_self_contained":
        fail("parameter golf record-folder compatibility error: unexpected dependency posture")
    if data["public_repo_allows_additional_folder_contents"] is not True:
        fail("parameter golf record-folder compatibility error: public repo should allow additional folder contents")
    if data["verifier_runner"] != "scripts/check-parameter-golf-record-folder-compatibility.sh":
        fail("parameter golf record-folder compatibility error: unexpected verifier runner")

    core_keys = data["required_submission_json_core_keys"]
    if not isinstance(core_keys, list) or not core_keys:
        fail("parameter golf record-folder compatibility error: `required_submission_json_core_keys` must be a non-empty array")
    if len(set(core_keys)) != len(core_keys):
        fail("parameter golf record-folder compatibility error: duplicate core submission.json keys are not allowed")

    contracts = data["track_contracts"]
    if not isinstance(contracts, list) or len(contracts) != 2:
        fail("parameter golf record-folder compatibility error: `track_contracts` must contain the record and non-record contracts")
    seen_track_ids = set()
    for contract in contracts:
        for key in [
            "track_id",
            "records_relpath",
            "psionic_required_submission_json_keys",
            "current_example_readme_ref",
            "detail",
        ]:
            if key not in contract:
                fail(f"parameter golf record-folder compatibility error: track contract missing `{key}`")
        seen_track_ids.add(contract["track_id"])
    if seen_track_ids != {"record_10min_16mb", "non_record_16mb"}:
        fail("parameter golf record-folder compatibility error: unexpected track ids")

    required_files = data["required_top_level_files"]
    if not isinstance(required_files, list) or not required_files:
        fail("parameter golf record-folder compatibility error: `required_top_level_files` must be a non-empty array")
    expected_files = ["README.md", "submission.json", "train.log", "train_gpt.py"]
    actual_files = [item.get("file_name") for item in required_files]
    if actual_files != expected_files:
        fail("parameter golf record-folder compatibility error: required top-level files must stay in canonical order")

    digestible = dict(data)
    digestible["report_digest"] = ""
    expected_digest = stable_digest(
        digestible,
        "psionic_parameter_golf_record_folder_compatibility_report|",
    )
    if data["report_digest"] != expected_digest:
        fail("parameter golf record-folder compatibility error: `report_digest` does not match the canonical payload")


def validate_parameter_golf_root(root: Path, report: dict) -> None:
    if not root.is_dir():
        fail(f"parameter golf record-folder compatibility error: parameter-golf root `{root}` is not a directory")
    readme_path = root / report["challenge_repo_readme_ref"]
    if not readme_path.is_file():
        fail(f"parameter golf record-folder compatibility error: missing challenge README `{readme_path}`")
    for contract in report["track_contracts"]:
        records_path = root / contract["records_relpath"]
        if not records_path.is_dir():
            fail(f"parameter golf record-folder compatibility error: missing records directory `{records_path}`")
        example_readme = root / contract["current_example_readme_ref"]
        if not example_readme.is_file():
            fail(f"parameter golf record-folder compatibility error: missing public example README `{example_readme}`")


def load_submission_json(submission_root: Path) -> dict:
    try:
        return json.loads((submission_root / "submission.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"parameter golf record-folder compatibility error: missing required top-level file `submission.json` in `{submission_root}`")
    except json.JSONDecodeError as error:
        fail(f"parameter golf record-folder compatibility error: invalid submission.json: {error}")


def resolve_track(report: dict, submission_json: dict) -> dict:
    contracts = report["track_contracts"]
    if TRACK_ID:
        matches = [contract for contract in contracts if contract["track_id"] == TRACK_ID]
        if len(matches) != 1:
            fail(f"parameter golf record-folder compatibility error: unknown track id `{TRACK_ID}`")
        return matches[0]
    track_value = submission_json.get("track")
    if track_value is None:
        matches = [contract for contract in contracts if contract.get("submission_json_track_value") is None]
    else:
        matches = [
            contract
            for contract in contracts
            if contract.get("submission_json_track_value") == track_value
        ]
    if len(matches) != 1:
        fail("parameter golf record-folder compatibility error: unable to resolve submission track from submission.json; pass --track-id explicitly")
    return matches[0]


def ensure_no_external_symlinks(submission_root: Path) -> None:
    for path in submission_root.rglob("*"):
        if path.is_symlink():
            fail(f"parameter golf record-folder compatibility error: symlinks are not allowed inside the submission folder: `{path}`")


def verify_required_files(report: dict, submission_root: Path) -> list[str]:
    checked = []
    for item in report["required_top_level_files"]:
        path = submission_root / item["file_name"]
        if not path.exists():
            fail(f"parameter golf record-folder compatibility error: missing required top-level file `{item['file_name']}` in `{submission_root}`")
        if not path.is_file():
            fail(f"parameter golf record-folder compatibility error: required top-level file `{item['file_name']}` must be a regular file")
        checked.append(item["file_name"])
    return checked


def verify_submission_json_keys(report: dict, track: dict, submission_json: dict) -> None:
    missing = [
        key
        for key in report["required_submission_json_core_keys"] + track["psionic_required_submission_json_keys"]
        if key not in submission_json
    ]
    if missing:
        fail(
            "parameter golf record-folder compatibility error: submission.json is missing required key(s): "
            + ", ".join(missing)
        )


def relative_paths(submission_root: Path) -> tuple[list[str], list[str]]:
    top_level = []
    nested = []
    for path in sorted(submission_root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(submission_root).as_posix()
            if "/" in relative:
                nested.append(relative)
            else:
                top_level.append(relative)
    return top_level, nested


def dry_run_entrypoint(submission_root: Path) -> dict:
    if SKIP_ENTRYPOINT_DRY_RUN:
        return {
            "executed": False,
            "command": ["python3", "train_gpt.py"],
            "exit_code": None,
            "stdout_digest": None,
            "stdout_preview": "",
            "stderr_preview": "",
        }
    completed = subprocess.run(
        ["python3", "train_gpt.py"],
        cwd=submission_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            "parameter golf record-folder compatibility error: entrypoint dry-run failed with exit code "
            + str(completed.returncode)
            + "\n"
            + completed.stderr
        )
    stdout = completed.stdout
    stderr = completed.stderr
    return {
        "executed": True,
        "command": ["python3", "train_gpt.py"],
        "exit_code": completed.returncode,
        "stdout_digest": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stdout_preview": "\n".join(stdout.strip().splitlines()[:4]),
        "stderr_preview": "\n".join(stderr.strip().splitlines()[:4]),
    }


def verification_report(report: dict, submission_root: Path, track: dict, required_files: list[str], entrypoint_report: dict) -> dict:
    top_level_files, nested_files = relative_paths(submission_root)
    payload = {
        "schema_version": 1,
        "runner": report["verifier_runner"],
        "compatibility_report_digest": report["report_digest"],
        "parameter_golf_root": str(PARAMETER_GOLF_ROOT.resolve()),
        "submission_dir": str(submission_root.resolve()),
        "track_id": track["track_id"],
        "records_relpath": track["records_relpath"],
        "required_top_level_files": required_files,
        "extra_top_level_files": [path for path in top_level_files if path not in required_files],
        "nested_submission_paths": nested_files,
        "entrypoint_dry_run": entrypoint_report,
        "verdict": "compatible",
    }
    digestible = dict(payload)
    digestible["report_digest"] = ""
    payload["report_digest"] = stable_digest(
        digestible,
        "psionic_parameter_golf_record_folder_verification_report|",
    )
    return payload


with INPUT_PATH.open("r", encoding="utf-8") as handle:
    report = json.load(handle)

validate_report(report)
validate_parameter_golf_root(PARAMETER_GOLF_ROOT, report)

if SUBMISSION_DIR is None:
    encoded = json.dumps(report, indent=2) + "\n"
    if REPORT_PATH:
        REPORT_PATH.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    sys.exit(0)

if not SUBMISSION_DIR.is_dir():
    fail(f"parameter golf record-folder compatibility error: submission dir `{SUBMISSION_DIR}` is not a directory")

ensure_no_external_symlinks(SUBMISSION_DIR)
required_files = verify_required_files(report, SUBMISSION_DIR)
submission_json = load_submission_json(SUBMISSION_DIR)
track = resolve_track(report, submission_json)
verify_submission_json_keys(report, track, submission_json)
entrypoint_report = dry_run_entrypoint(SUBMISSION_DIR)
verification = verification_report(report, SUBMISSION_DIR, track, required_files, entrypoint_report)

encoded = json.dumps(verification, indent=2) + "\n"
if REPORT_PATH:
    REPORT_PATH.write_text(encoded, encoding="utf-8")
else:
    sys.stdout.write(encoded)
PY
