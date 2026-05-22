#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

cd "${repo_root}"

export RUSTFLAGS="${RUSTFLAGS:--Awarnings}"

grep -q '^version = "0.2.0"$' Cargo.toml

scripts/check-training-execution-evidence-bundle.sh
scripts/check-cross-provider-program-run-graph.sh
scripts/check-decentralized-network-contract.sh
scripts/check-signed-node-identity-contract-set.sh
scripts/check-public-network-registry-contract.sh
scripts/check-public-work-assignment-contract.sh
scripts/check-public-dataset-authority-contract.sh
scripts/check-public-miner-protocol-contract.sh
scripts/check-validator-challenge-scoring-contract.sh
scripts/check-multi-validator-consensus-contract.sh
scripts/check-fraud-quarantine-slashing-contract.sh
scripts/check-reward-ledger-contract.sh
scripts/check-settlement-publication-contract.sh
scripts/check-operator-bootstrap-package-contract.sh
scripts/check-public-run-explorer-contract.sh
scripts/check-public-testnet-readiness-contract.sh
scripts/check-curated-decentralized-run-contract.sh
scripts/check-open-public-decentralized-run-contract.sh
scripts/check-incentivized-decentralized-run-contract.sh

cargo check -q -p psionic-train --bin qwen_legal_pylon_worker_server
cargo test -q -p psionic-train qwen_legal_pylon_dispatch --lib -- --nocapture
cargo test -q -p psionic-train qwen_legal_pylon_training_job --lib -- --nocapture
scripts/check-qwen-legal-pylon-network-sft.sh
