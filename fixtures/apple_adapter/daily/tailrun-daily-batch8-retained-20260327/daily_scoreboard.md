# Tailrun Daily Scoreboard

- Run id: `tailrun-daily-batch8-retained-20260327`
- Overall verdict: `throughput_improved`
- M5 throughput verdict: `meaningful_improvement` at `304.3399208380537` steps/s
- RTX 4080 throughput verdict: `meaningful_improvement` at `122.27966003430116` steps/s
- Held-out quality verdict: `noise_band` at best loss `15.942383766174316`
- Near-equivalent bridge verdict: `passed` with served token `37`

## Ordering

- Run the local M5 MLX same-node lane first.
- Run the remote archlinux RTX 4080 CUDA lane in the same matrix second.
- Run the PGOLF-ish held-out quality compare on the just-produced bundles third.
- Run the M5 near-equivalent infer/serve bridge fourth.
- Treat the M2 as opportunistic only and do not block the daily loop on it.

## Artifact Roots

- Matrix report: `/Users/christopherdavid/work/psionic/fixtures/apple_adapter/daily/tailrun-daily-batch8-retained-20260327/matrix/matrix_report.json`
- Quality report: `/Users/christopherdavid/work/psionic/fixtures/apple_adapter/daily/tailrun-daily-batch8-retained-20260327/quality/quality_report.json`
- Near-equivalent report: `/Users/christopherdavid/work/psionic/fixtures/apple_adapter/daily/tailrun-daily-batch8-retained-20260327/near_equivalent/near_equivalent_report.json`
