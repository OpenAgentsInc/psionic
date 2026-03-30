# Psion Executor MLX Forward/Load Parity

> Status: canonical `PSION-0201` / `#720` retained MLX forward/load parity
> packet for the executor lane, updated 2026-03-30.

## Canonical Fixture

- `fixtures/psion/executor/psion_executor_mlx_forward_load_parity_v1.json`

## Canonical Generator

Run from the repo root:

```bash
cargo run -q -p psionic-train --example psion_executor_mlx_forward_load_parity_fixtures
```

## What Landed

The executor lane now has one typed MLX forward/load parity packet grounded in
the already-shipped Mac bring-up path:

- entrypoint command:
  `cargo run -q -p psionic-train --bin swarm_mac_mlx_bringup -- fixtures/swarm/reports/swarm_mac_mlx_bringup_v1.json`
- entrypoint source:
  `crates/psionic-train/src/bin/swarm_mac_mlx_bringup.rs`
- retained source report:
  `fixtures/swarm/reports/swarm_mac_mlx_bringup_v1.json`

The packet binds three truths into the executor roadmap:

- the Mac profile can load the bounded converted-equivalent MLX lane through
  `open_adapter_backend.mlx.metal.gpt_oss_lm_head`
- the admitted forward surface is no longer implicit: it is the retained
  `constant -> matmul -> add` proof on `metal:0`
- parity gaps are explicit instead of hidden

## Retained Forward/Load Truth

The committed packet keeps the first executor-lane MLX facts visible:

- admitted profile id: `local_mac_mlx_aarch64`
- backend label: `open_adapter_backend.mlx.metal.gpt_oss_lm_head`
- logical device: `metal:0`
- adapter family: `gpt_oss.decoder_lm_head_lora`
- checkpoint family: `swarm.open_adapter.mlx.same_node`
- retained forward probe output: `[1.5, 2.5]`
- retained adapter artifact digest:
  `44a868bbf20e5b80cbe52f620a0aa5f94b117a7bbd66906d3bff25dd59228631`

## Explicit Parity Gaps

Phase-one MLX parity is intentionally narrow.

The committed packet keeps the current non-closure explicit:

- reshape-backed graphs still refuse outside the admitted Metal slice
- bf16 mixed precision still refuses on this bounded MLX load path

That is the right phase-one boundary. The packet proves one real executor-lane
MLX load/forward surface without pretending the repo has blanket MLX executor
closure.
