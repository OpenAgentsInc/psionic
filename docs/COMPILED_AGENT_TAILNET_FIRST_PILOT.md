# Compiled Agent Tailnet First Pilot

> Status: first governed dual-node run completed on 2026-03-29 across the local
> M5 coordinator and the `archlinux` NVIDIA RTX 4080 worker.

## Why This Exists

Phase 5 does not widen the task family. It proves that the first "external"
contributor path can be our own Tailnet while still obeying the same bounded
receipt, quarantine, replay, validator, promotion, and rollback contracts used
by the internal compiled-agent loop.

## Nodes

- coordinator: `contrib.tailnet.m5`
  - bundle id: `compiled_agent.tailnet_node_bundle.m5.v1`
  - bundle digest:
    `8a1755d9ffaa3af2aff425129dde232b318c8bf839f707324409e4af390b0c3a`
- worker: `contrib.tailnet.archlinux_rtx4080`
  - bundle id: `compiled_agent.tailnet_node_bundle.archlinux.v1`
  - bundle digest:
    `c47df518233bd687bd88b442002840f8fedc45960db8298795db8562a097022d`

Both bundles are now anchored to the same retained external benchmark contract
digest:

- `9a2a53cc95fdb1a674a0da0612dda1a013718a5756fa7764bde305b49b4174f4`

That was the key seam to close. The Tailnet run only became stable once the
remote node stopped recomputing its own local benchmark contract and started
consuming the same retained artifact as the M5 node.

## Retained Outputs

- staging ledger digest:
  `5d05f3500e0ca5bdfd8291e1b0ffd3bfd95a4d99b3bc854d03db6636183151b2`
- quarantine report digest:
  `90492578580c808d8639f7a920b641fe8e717446509c74e9126d5e0c6d91c6c4`
- tailnet learning ledger digest:
  `f0c2e7cc9a884dec8d030da6f30c7f7155c761415a28d4685501b040b44275a3`
- tailnet replay bundle digest:
  `05889986e497765869edcc8b5edc0066ab4bbe916bb867ec29e2ddf085cb0908`
- governed run digest:
  `dc9ab99b00fa05ae990693b5e758cc728d7d06dcef36bb51b86bf769c7f18b37`

Preview training accuracy from the governed Tailnet run:

- route preview accuracy: `1.0`
- grounded-answer preview accuracy: `1.0`

## What The Pilot Proved

- the M5 and RTX 4080 machines can now contribute to one governed compiled-agent
  run under the same retained contract
- external benchmark runs and runtime disagreement receipts can enter the same
  staging and quarantine flow without changing the evidence boundary
- the bounded learning loop can consume the broader evidence set and still make
  an honest promotion decision

## Follow-On Truth

After the Tailnet pilot, the bounded XTRAIN loop was rerun on the widened
receipt base and now retains:

- route decision: `promote`
- grounded-answer decision: `promote`
- XTRAIN receipt digest:
  `4f7655b1b65931c538c3fbea643452a8a16e1ad7738ae4a9e12896ef722cef45`
- promoted-artifact contract digest:
  `80b130858c414d13f2351a2ff3a2b4e7597ad7b20cd467922883c5ce90981720`

That means the Tailnet pilot was not just telemetry. It contributed evidence
that the bounded route path could eventually clear the unchanged held-out gate.

## Commands

Local M5 node bundle:

```bash
cargo run -q -p psionic-train --bin compiled_agent_tailnet_node_bundle -- \
  --profile tailnet_m5_mlx \
  --output fixtures/compiled_agent/tailnet/compiled_agent_tailnet_m5_node_bundle_v1.json
```

Remote RTX 4080 node bundle:

```bash
ssh christopherdavid@archlinux 'bash -ic "
  cd ~/psionic-phase5-tailnet-remote &&
  TMPDIR=\$HOME/.tmp-rust \
  CARGO_TARGET_DIR=\$HOME/psionic-phase5-target \
  cargo run -q -p psionic-train --bin compiled_agent_tailnet_node_bundle -- \
    --profile tailnet_archlinux_rtx4080_cuda \
    --output /home/christopherdavid/psionic-phase5-tailnet-remote/fixtures/compiled_agent/tailnet/compiled_agent_tailnet_archlinux_node_bundle_v1.json
"'
```

Merged governed run:

```bash
cargo run -q -p psionic-train --bin compiled_agent_tailnet_governed_run -- \
  --local-bundle fixtures/compiled_agent/tailnet/compiled_agent_tailnet_m5_node_bundle_v1.json \
  --remote-bundle fixtures/compiled_agent/tailnet/compiled_agent_tailnet_archlinux_node_bundle_v1.json \
  --staging-output fixtures/compiled_agent/tailnet/compiled_agent_tailnet_submission_staging_ledger_v1.json \
  --quarantine-output fixtures/compiled_agent/tailnet/compiled_agent_tailnet_quarantine_report_v1.json \
  --run-output fixtures/compiled_agent/tailnet/compiled_agent_tailnet_governed_run_v1.json
```

## Boundary

This pilot stays strictly on the learned compiled-agent lane. It does not
collapse learned evidence into stronger-evidence claims, it does not require
Tassadar, and it does not give the external node promotion authority.
