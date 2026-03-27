#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_homegolf_multiseed_package.json"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/parameter_golf_homegolf_multiseed_package.XXXXXX")"
generated_path="${tmpdir}/parameter_golf_homegolf_multiseed_package.json"

cargo run -q -p psionic-train --bin parameter_golf_homegolf_multiseed_package -- "${generated_path}" >/dev/null
cmp -s "${fixture_path}" "${generated_path}"
echo "HOMEGOLF multi-seed package matches fixture"
