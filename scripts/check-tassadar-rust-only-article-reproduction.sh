#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo run -p psionic-serve --example tassadar_rust_only_article_reproduction

jq -e '
  .all_components_green == true
  and .component_count == 8
  and .green_component_count == 8
' fixtures/tassadar/reports/tassadar_rust_only_article_reproduction_report.json >/dev/null
