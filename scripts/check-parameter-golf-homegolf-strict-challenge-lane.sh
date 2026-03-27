#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_path="${repo_root}/fixtures/parameter_golf/reports/parameter_golf_homegolf_strict_challenge_lane.json"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/parameter_golf_homegolf_strict_challenge_lane.XXXXXX")"
generated_path="${tmpdir}/parameter_golf_homegolf_strict_challenge_lane.json"

cargo run -q -p psionic-train --bin parameter_golf_homegolf_strict_challenge_lane -- "${generated_path}" >/dev/null

if ! cmp -s "${fixture_path}" "${generated_path}"; then
  echo "strict HOMEGOLF challenge lane drifted from committed fixture" >&2
  diff -u "${fixture_path}" "${generated_path}" >&2 || true
  exit 1
fi

echo "strict HOMEGOLF challenge lane matches fixture"
