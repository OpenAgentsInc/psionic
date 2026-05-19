#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo test -p psionic-eval --no-default-features --lib legal_benchmark_ci
cargo test -p psionic-eval --no-default-features --lib legal_benchmark_coverage
cargo test -p psionic-eval --no-default-features --lib legal_benchmark_tools
cargo test -p psionic-eval --no-default-features --lib legal_benchmark_reports
cargo test -p psionic-eval --no-default-features --lib legal_benchmark_sweeps

python3 -m json.tool fixtures/legal_benchmark/harvey_corpus_metadata.json >/dev/null
python3 -m json.tool fixtures/legal_benchmark/normalization_snapshot_minimal.json >/dev/null
python3 -m json.tool fixtures/legal_benchmark/evaluator_mock_score_report.json >/dev/null
python3 -m json.tool fixtures/legal_benchmark/report_export_mock.json >/dev/null
python3 -m json.tool fixtures/legal_benchmark/sweep_smoke_config.json >/dev/null
