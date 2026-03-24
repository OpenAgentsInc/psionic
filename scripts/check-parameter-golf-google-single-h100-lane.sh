#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
report_path=""

usage() {
  cat <<'EOF' >&2
Usage: check-parameter-golf-google-single-h100-lane.sh [--report <path>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      report_path="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${report_path}" ]]; then
  report_path="$(mktemp "${TMPDIR:-/tmp}/parameter_golf_google_single_h100_rehearsal.XXXXXX.json")"
  cleanup_report=1
else
  cleanup_report=0
fi

cleanup() {
  if [[ "${cleanup_report}" -eq 1 ]]; then
    rm -f -- "${report_path}"
  fi
}
trap cleanup EXIT

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"; cleanup' EXIT

mock_root="${tmpdir}/mock-gcloud"
mock_bin="${tmpdir}/mock-bin"
generated_dir="${tmpdir}/generated"
mkdir -p "${mock_root}/storage" "${mock_bin}" "${generated_dir}"

descriptor_uri="$(jq -r '.descriptor_uri' "${repo_root}/fixtures/parameter_golf/google/parameter_golf_google_input_package_policy_v1.json")"
archive_uri="$(jq -r '.archive_uri' "${repo_root}/fixtures/parameter_golf/google/parameter_golf_google_input_package_policy_v1.json")"

mock_uri_path() {
  local uri="$1"
  local without_scheme="${uri#gs://}"
  printf '%s/%s' "${mock_root}/storage" "${without_scheme}"
}

mkdir -p "$(dirname "$(mock_uri_path "${descriptor_uri}")")" "$(dirname "$(mock_uri_path "${archive_uri}")")"
cp \
  "${repo_root}/fixtures/parameter_golf/google/parameter_golf_google_input_package_descriptor_v1.json" \
  "$(mock_uri_path "${descriptor_uri}")"
cp \
  "${repo_root}/fixtures/parameter_golf/google/parameter_golf_google_input_package_v1.tar.gz" \
  "$(mock_uri_path "${archive_uri}")"

cat > "${mock_bin}/gcloud" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

mock_root="${MOCK_GCLOUD_ROOT:?}"
bucket_root="${mock_root}/storage"

uri_to_path() {
  local uri="$1"
  local without_scheme="${uri#gs://}"
  printf '%s/%s' "${bucket_root}" "${without_scheme}"
}

if [[ "$#" -ge 2 && "$1" == "version" && "$2" == "--format=json" ]]; then
  printf '{"Google Cloud SDK":"556.0.0","bq":"2.1.28"}\n'
  exit 0
fi

if [[ "$#" -ge 3 && "$1" == "config" && "$2" == "get-value" && "$3" == "account" ]]; then
  printf 'mock-operator@example.com\n'
  exit 0
fi

if [[ "$#" -ge 3 && "$1" == "config" && "$2" == "get-value" && "$3" == "project" ]]; then
  printf 'openagentsgemini\n'
  exit 0
fi

if [[ "$#" -ge 2 && "$1" == "auth" && "$2" == "print-access-token" ]]; then
  printf 'mock-access-token\n'
  exit 0
fi

if [[ "$#" -ge 3 && "$1" == "storage" && "$2" == "ls" ]]; then
  target="$(uri_to_path "$3")"
  if [[ -e "${target}" ]]; then
    printf '%s\n' "$3"
    exit 0
  fi
  exit 1
fi

if [[ "$#" -ge 3 && "$1" == "storage" && "$2" == "cat" ]]; then
  cat "$(uri_to_path "$3")"
  exit 0
fi

if [[ "$#" -ge 5 && "$1" == "storage" && "$2" == "cp" ]]; then
  src="$4"
  dest="$5"
  if [[ "$3" != "--quiet" ]]; then
    src="$3"
    dest="$4"
  fi
  if [[ "${dest}" == gs://* ]]; then
    dest_path="$(uri_to_path "${dest}")"
    mkdir -p "$(dirname "${dest_path}")"
    cp "${src}" "${dest_path}"
  else
    src_path="$(uri_to_path "${src}")"
    mkdir -p "$(dirname "${dest}")"
    cp "${src_path}" "${dest}"
  fi
  exit 0
fi

if [[ "$#" -ge 5 && "$1" == "storage" && "$2" == "buckets" && "$3" == "get-iam-policy" ]]; then
  cat <<'JSON'
{"bindings":[{"role":"roles/storage.objectAdmin","members":["serviceAccount:psion-train-single-node@openagentsgemini.iam.gserviceaccount.com"]},{"role":"roles/storage.legacyBucketReader","members":["serviceAccount:psion-train-single-node@openagentsgemini.iam.gserviceaccount.com"]}]}
JSON
  exit 0
fi

if [[ "$#" -ge 4 && "$1" == "iam" && "$2" == "service-accounts" && "$3" == "describe" ]]; then
  email="$4"
  if [[ "${*: -1}" == "value(email)" ]]; then
    printf '%s\n' "${email}"
  else
    printf '{"email":"%s"}\n' "${email}"
  fi
  exit 0
fi

if [[ "$#" -ge 4 && "$1" == "projects" && "$2" == "get-iam-policy" ]]; then
  role="$(printf '%s\n' "$*" | sed -n 's/.*bindings.role:\([^ ]*\).*/\1/p')"
  printf '%s\n' "${role}"
  exit 0
fi

if [[ "$#" -ge 7 && "$1" == "compute" && "$2" == "machine-types" && "$3" == "describe" ]]; then
  printf '{"name":"%s"}\n' "$4"
  exit 0
fi

if [[ "$#" -ge 7 && "$1" == "compute" && "$2" == "accelerator-types" && "$3" == "describe" ]]; then
  printf '{"name":"%s"}\n' "$4"
  exit 0
fi

if [[ "$#" -ge 6 && "$1" == "compute" && "$2" == "images" && "$3" == "describe-from-family" ]]; then
  family="$4"
  cat <<JSON
{"name":"${family}-mock-image","selfLink":"https://example.invalid/${family}-mock-image","id":"123456789","creationTimestamp":"2026-03-23T00:00:00.000-00:00"}
JSON
  exit 0
fi

if [[ "$#" -ge 6 && "$1" == "compute" && "$2" == "regions" && "$3" == "describe" ]]; then
  cat <<'JSON'
{"quotas":[{"metric":"INSTANCES","limit":10,"usage":0},{"metric":"SSD_TOTAL_GB","limit":2000,"usage":0},{"metric":"GPU_FAMILY:NVIDIA_H100","limit":4,"usage":0},{"metric":"PREEMPTIBLE_NVIDIA_H100_GPUS","limit":4,"usage":0}]}
JSON
  exit 0
fi

echo "unsupported mock gcloud invocation: $*" >&2
exit 1
EOF

cat > "${mock_bin}/bq" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ge 2 && "$1" == "show" ]]; then
  printf '{}\n'
  exit 0
fi
echo "unsupported mock bq invocation: $*" >&2
exit 1
EOF

chmod +x "${mock_bin}/gcloud" "${mock_bin}/bq"

MOCK_GCLOUD_ROOT="${mock_root}" \
PATH="${mock_bin}:${PATH}" \
bash "${repo_root}/scripts/parameter-golf-google-package-inputs.sh" \
  --output-dir "${generated_dir}" >/dev/null

export MOCK_GCLOUD_ROOT="${mock_root}"
export PATH="${mock_bin}:${PATH}"

preflight_json="$(
  bash "${repo_root}/scripts/parameter-golf-google-operator-preflight.sh" \
    --profile a3_h100_single_node_parameter_golf
)"

launch_output_file="${tmpdir}/launch.out"
bash "${repo_root}/scripts/parameter-golf-google-launch-single-node.sh" \
  --profile a3_h100_single_node_parameter_golf \
  --run-id parameter-golf-google-rehearsal \
  --instance-name parameter-golf-google-rehearsal \
  --manifest-only > "${launch_output_file}"

python3 - "${repo_root}" "${generated_dir}" "${report_path}" "${launch_output_file}" "${preflight_json}" "${descriptor_uri}" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
generated_dir = Path(sys.argv[2])
report_path = Path(sys.argv[3])
launch_output_file = Path(sys.argv[4])
preflight = json.loads(sys.argv[5])
descriptor_uri = sys.argv[6]

committed_dir = repo_root / "fixtures" / "parameter_golf" / "google"

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)

for name in [
    "parameter_golf_google_input_contract_v1.json",
    "parameter_golf_google_input_package_manifest_v1.json",
    "parameter_golf_google_input_package_descriptor_v1.json",
]:
    committed = json.loads((committed_dir / name).read_text(encoding="utf-8"))
    generated = json.loads((generated_dir / name).read_text(encoding="utf-8"))
    if committed != generated:
        fail(f"generated {name} does not match committed truth")

committed_archive = (committed_dir / "parameter_golf_google_input_package_v1.tar.gz").read_bytes()
generated_archive = (generated_dir / "parameter_golf_google_input_package_v1.tar.gz").read_bytes()
if committed_archive != generated_archive:
    fail("generated input package archive does not match committed truth")

if preflight.get("result") != "ready":
    fail("operator preflight did not report ready")
if preflight.get("profile_id") != "a3_h100_single_node_parameter_golf":
    fail("operator preflight drifted to the wrong profile")
quota_preflight = preflight.get("quota_preflight") or {}
if quota_preflight.get("result") != "ready":
    fail("quota preflight did not report ready")

launch_lines = launch_output_file.read_text(encoding="utf-8").splitlines()
try:
    json_start = next(i for i, line in enumerate(launch_lines) if line.startswith("{"))
except StopIteration:
    fail("launch output did not contain a manifest JSON object")
manifest = json.loads("\n".join(launch_lines[json_start:]))

if manifest.get("profile_id") != "a3_h100_single_node_parameter_golf":
    fail("launch manifest drifted to the wrong profile")
if manifest.get("trainer_lane_id") != "parameter_golf_single_h100":
    fail("launch manifest drifted to the wrong trainer lane")
if manifest.get("expected_execution_backend") != "cuda":
    fail("launch manifest lost the cuda execution claim")

machine = manifest.get("machine", {})
if machine.get("machine_type") != "a3-highgpu-1g":
    fail("launch manifest lost the A3 High machine type")
if machine.get("accelerator_type") != "nvidia-h100-80gb":
    fail("launch manifest lost the H100 accelerator type")
if machine.get("provisioning_model") != "FLEX_START":
    fail("launch manifest lost the explicit FLEX_START provisioning model")

input_package = manifest.get("input_package", {})
if input_package.get("descriptor_uri") != descriptor_uri:
    fail("launch manifest drifted from the committed input package descriptor uri")

training = manifest.get("training", {})
command = training.get("command", "")
pre_training_command = training.get("pre_training_command", "")
if "parameter_golf_single_h100_train" not in command:
    fail("launch manifest training command does not invoke the Rust single-H100 trainer")
if "parameter-golf-materialize-public-cache.sh" not in pre_training_command:
    fail("launch manifest pre-training command does not materialize the public cache")
if "check-parameter-golf-single-h100-bringup.sh" not in pre_training_command:
    fail("launch manifest pre-training command no longer validates the cache contract")

report = {
    "schema_version": "parameter_golf.google_single_h100_operator_rehearsal.v1",
    "runner": "scripts/check-parameter-golf-google-single-h100-lane.sh",
    "profile_id": manifest["profile_id"],
    "trainer_lane_id": manifest["trainer_lane_id"],
    "preflight_result": preflight["result"],
    "quota_preflight_result": quota_preflight["result"],
    "machine_type": machine["machine_type"],
    "accelerator_type": machine["accelerator_type"],
    "provisioning_model": machine["provisioning_model"],
    "input_package_descriptor_uri": input_package["descriptor_uri"],
    "training_command": command,
    "pre_training_command": pre_training_command,
    "generated_package_matches_committed_truth": True,
}
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
