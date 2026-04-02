#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

cargo test -p psionic-train --lib live_rl_run_service -- --nocapture
