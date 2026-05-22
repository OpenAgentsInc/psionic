# Qwen Legal v0.2 Pylon Release Readiness Run

Date: 2026-05-22
Command: `scripts/check-v0.2-pylon-release.sh`
Result: PASS

## Contract Gate

The release gate verified:

- provider-neutral training evidence bundle
- cross-provider whole-program run graph
- decentralized network contract
- signed node identity contract set
- public network registry
- public work assignment
- public dataset authority
- public miner protocol
- validator challenge scoring
- multi-validator consensus
- fraud quarantine/slashing
- reward ledger
- settlement publication
- operator bootstrap package
- public run explorer
- public testnet readiness
- curated decentralized run
- open public decentralized run
- incentivized decentralized run

Representative digests from the passing run:

- decentralized network: `071310191273b8161bf37aba2c0771863281f2a0f1d5d9a07d2f37f1b87a329b`
- public network registry: `13c54dbe05c8be670629d69e6c84d8457e8ce507f075fe4b70c4569634899fd2`
- reward ledger: `df6d23ba9aaaa5186f0ba51145c48a77d470530a2eab0ab7c62e166aec0e993a`
- settlement publication: `4cb9e8a965cff73e25c9dd85df54b91cd4387866b422aa27e2699e686d2f64f9`
- public testnet readiness: `10837daca68e69513fd2b291a30e44b0eb0dac7efc03db878c2df584e885e25f`
- incentivized decentralized run: `0185b3721cb30d3c0590be9220749b0beb29ed81ca984e70f9670d66dda0655e`

## Runtime Gate

The release gate compiled the TCP worker server and ran:

```text
running 4 tests
....
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 1126 filtered out

running 19 tests
...................
test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 1111 filtered out

running 2 tests
test qwen_legal_pylon_network_sft::tests::qwen_legal_pylon_network_sft_emits_two_contributor_aggregate ... ok
test qwen_legal_pylon_network_sft::tests::qwen_legal_pylon_network_sft_fixture_writes_loadable_artifacts ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 1128 filtered out
```

The Qwen legal Pylon network SFT fixture completed with aggregate digest:

```text
8e8dea3bc639ed2c147d6901f6ceda9b5f1a176034dc7bb65219daf7dd33116d
```

## Payment Gate

The `qwen_legal_pylon_training_job` test bucket covered payable worker
decisions, missing and invalid receipts, duplicate shard withholding,
operator-deferred closeout, failed settlement proof blocking, live-small-value
operator-approved proof acceptance, duplicate proof rejection, bad proof digest
rejection, unknown authorization rejection, secret-looking proof rejection, and
amount mismatch rejection.

The passing release boundary is the Psionic-side payment boundary: signed
worker receipt validation, payment decision receipts, Treasury handoff,
settlement proof validation, and promotion-gate status. Wallet execution and
secret custody remain Treasury/Nexus responsibilities.
