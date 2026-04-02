#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

cargo test -p psionic-train sampler_service -- --nocapture
