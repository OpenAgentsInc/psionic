#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
review_path="${1:-${repo_root}/fixtures/tassadar/reports/tassadar_public_disclosure_release_review.json}"

python3 - "${review_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)

required_keys = [
    "review_id",
    "status",
    "review_scope",
    "source_private_refs",
    "public_surface_refs",
    "checklist",
    "red_team_findings",
]
required_checklist = [
    "private_naming_removed",
    "private_product_framing_removed",
    "benchmark_claims_bounded",
    "dependency_markers_preserved",
    "private_language_refused",
    "public_only_claim_language_used",
]

errors = []
for key in required_keys:
    if key not in data:
        errors.append(f"missing top-level key `{key}`")

checklist = data.get("checklist", {})
for key in required_checklist:
    if key not in checklist:
        errors.append(f"missing checklist key `{key}`")

status = data.get("status")
if status not in {"approved", "refused"}:
    errors.append("status must be `approved` or `refused`")

if status == "approved":
    for key in required_checklist:
        if checklist.get(key) is not True:
            errors.append(f"approved reviews require checklist `{key}` to be true")
    if data.get("red_team_findings"):
        errors.append("approved reviews must not carry red-team findings")

if status == "refused":
    reason = data.get("blocked_publication_reason", "")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("refused reviews require a non-empty `blocked_publication_reason`")
    if checklist.get("private_language_refused") is not True:
        errors.append("refused reviews still require `private_language_refused=true`")

if errors:
    for error in errors:
        print(f"disclosure review error: {error}", file=sys.stderr)
    sys.exit(1)

print(
    f"validated Tassadar public disclosure review `{data['review_id']}` with status `{status}`"
)
PY
