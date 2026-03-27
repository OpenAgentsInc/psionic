#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_homegolf_competitive_ablation.json"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/parameter_golf_homegolf_competitive_ablation.XXXXXX")"
generated_path="${tmpdir}/parameter_golf_homegolf_competitive_ablation.json"

cargo run -q -p psionic-train --bin parameter_golf_homegolf_competitive_ablation -- "${generated_path}" >/dev/null

if ! cmp -s "${fixture_path}" "${generated_path}"; then
  echo "HOMEGOLF competitive ablation drifted from committed fixture" >&2
  diff -u "${fixture_path}" "${generated_path}" >&2 || true
  exit 1
fi

echo "HOMEGOLF competitive ablation matches fixture"
