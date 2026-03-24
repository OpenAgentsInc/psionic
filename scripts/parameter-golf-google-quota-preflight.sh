#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

export GUARDRAIL_FILE="${repo_root}/fixtures/parameter_golf/google/parameter_golf_google_billing_guardrails_v1.json"

exec bash "${script_dir}/psion-google-quota-preflight.sh" "$@"
