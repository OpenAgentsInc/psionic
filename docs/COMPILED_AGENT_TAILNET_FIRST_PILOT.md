# Compiled Agent Tailnet First Pilot

> Status: Tailnet-first governed dual-node run retained and rerun on 2026-03-29
> across the local M5 coordinator and the `archlinux` NVIDIA RTX 4080 worker.

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

- current local M5 bundle digest:
  `5a8f3fa13bfd67c1609251f4f9a4ad9c514e688d892c3f6f16e155a46ed9dda9`
- current remote RTX 4080 bundle digest:
  `f2682730f546c68715e66bd0bdb2144a792c1eacb44838feefa4ab59b4b90ec4`
- current staging ledger digest:
  `5f4ce9b6126629220a69083eaa62f029bfd4054af38161cee52092489d0dabf8`
- current quarantine report digest:
  `1b71d21d1f1f0ca2ea2b2420709c91b8f1d6012cdcd884ee91bfe1431ddf00a3`
- current tailnet learning ledger digest:
  `22f790143a9da07527548333d4b9d53e811fbecc74b191722a3f4e94fb1d4b9e`
- current tailnet replay bundle digest:
  `85d108ecebb9ae71bb005709c8b643c1b7eca86058786c01d7501faa40ca9036`
- current governed run digest:
  `a6f3caf208fb510793d048fa44e8bab8f2761282f0a1c6d42d0845fe8208f2dc`

The retained phase-six cadence surface now also compares the current run
against the phase-five baseline:

- previous governed run digest:
  `dc9ab99b00fa05ae990693b5e758cc728d7d06dcef36bb51b86bf769c7f18b37`
- previous staging ledger digest:
  `5d05f3500e0ca5bdfd8291e1b0ffd3bfd95a4d99b3bc854d03db6636183151b2`
- previous quarantine report digest:
  `90492578580c808d8639f7a920b641fe8e717446509c74e9126d5e0c6d91c6c4`

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

After the retained phase-six rerun, the bounded XTRAIN loop now retains:

- route decision: `promote`
- grounded-answer decision: `promote`
- XTRAIN receipt digest:
  `b432ca5f00bffae428592411712bcae980262038bd684aa7ad2f6f39b8d49073`
- promoted-artifact contract digest:
  `5835f484b83deb27ac7a7a96ae909011dd02a0c74582109b11ae04b3dfbeada4`
- phase-six operational report digest:
  `b0d4061de01e35a21b83ff6c2f57fb5737905420cfb110d031c0116c3fabad86`

That means the Tailnet loop is no longer just a first pilot. It now compares
run-to-run digests, surfaces alert states, and proves the bounded route path
can stay promotable on a harder evidence base without relaxing the gate.

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
Tassadar, and it does not give the external node promotion authority. The
phase-six operational report in `docs/COMPILED_AGENT_PHASE_SIX.md` is the
retained proof that this loop now reruns instead of decaying into demo state.
