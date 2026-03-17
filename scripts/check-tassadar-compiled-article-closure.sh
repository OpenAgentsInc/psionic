#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-research --example tassadar_compiled_article_closure_report -- "$@"
jq -e '.passed == true' fixtures/tassadar/reports/tassadar_compiled_article_closure_report.json >/dev/null
