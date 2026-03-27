#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_homegolf_clustered_run_surface.json"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/parameter_golf_homegolf_clustered_run_surface.XXXXXX")"
generated_path="${tmpdir}/parameter_golf_homegolf_clustered_run_surface.json"

cargo run -q -p psionic-train --bin parameter_golf_homegolf_clustered_run_surface -- "${generated_path}" >/dev/null

if ! cmp -s "${fixture_path}" "${generated_path}"; then
  echo "clustered HOMEGOLF surface drifted from committed fixture" >&2
  diff -u "${fixture_path}" "${generated_path}" >&2 || true
  exit 1
fi

echo "clustered HOMEGOLF surface matches fixture"
