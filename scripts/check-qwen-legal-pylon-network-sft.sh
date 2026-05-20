#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-${repo_root}/fixtures/qwen_legal/pylon_network_sft}"

cd "${repo_root}"

cargo test -p psionic-train --lib qwen_legal_pylon_network_sft -- --nocapture
cargo run -q -p psionic-train --example qwen_legal_pylon_network_sft_fixture -- "${output_dir}"

test -s "${output_dir}/pylon_network_sft_report_v1.json"
test -s "${output_dir}/aggregate-qwen-legal-lm-head-lora.safetensors"
test -s "${output_dir}/contributor-pylon-local-legal-cuda-01.safetensors"
test -s "${output_dir}/contributor-pylon-local-legal-metal-01.safetensors"
