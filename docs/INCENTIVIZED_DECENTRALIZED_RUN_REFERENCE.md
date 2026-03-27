# Incentivized Decentralized Run Reference

> Status: canonical `XTRAIN-47` / `#591` record, updated 2026-03-26 after
> landing the first incentivized decentralized run contract.

The new contract lives in:

- `crates/psionic-train/src/incentivized_decentralized_run_contract.rs`
- `crates/psionic-train/src/bin/incentivized_decentralized_run_contract.rs`
- `fixtures/training/incentivized_decentralized_run_contract_v1.json`
- `scripts/check-incentivized-decentralized-run-contract.sh`

The first retained incentivized run now freezes:

- three paid participants
- one settlement-backed payout publication
- published validator weights
- one incentives-focused audit

This issue proves Psionic can now point at one truthful rewarded decentralized
window with retained accounting and settlement artifacts.
