#!/usr/bin/env bash
set -euo pipefail

cargo run -q -p psionic-train --bin compiled_agent_default_row_contract -- --check-fixture
