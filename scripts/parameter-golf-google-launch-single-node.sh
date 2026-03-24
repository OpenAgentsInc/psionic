#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

export LAUNCH_FILE="${repo_root}/fixtures/parameter_golf/google/parameter_golf_google_single_node_launch_profiles_v1.json"
export OBSERVABILITY_FILE="${repo_root}/fixtures/psion/google/psion_google_host_observability_policy_v1.json"
export STARTUP_SCRIPT="${script_dir}/psion-google-single-node-startup.sh"
export QUOTA_PREFLIGHT="${script_dir}/parameter-golf-google-quota-preflight.sh"

exec bash "${script_dir}/psion-google-launch-single-node.sh" "$@"
