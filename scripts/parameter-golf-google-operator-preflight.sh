#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

export POLICY_FILE="${repo_root}/fixtures/parameter_golf/google/parameter_golf_google_operator_preflight_policy_v1.json"
export IDENTITY_PROFILE_FILE="${repo_root}/fixtures/psion/google/psion_google_training_identity_profile_v1.json"
export QUOTA_PREFLIGHT="${script_dir}/parameter-golf-google-quota-preflight.sh"

exec bash "${script_dir}/psion-google-operator-preflight.sh" "$@"
