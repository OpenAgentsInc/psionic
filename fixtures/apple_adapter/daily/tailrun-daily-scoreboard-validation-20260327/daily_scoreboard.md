# Tailrun Daily Scoreboard

- Run id: `tailrun-daily-scoreboard-validation-20260327`
- Overall verdict: `stable_no_clear_gain`
- M5 throughput verdict: `noise_band` at `162.53061053630358` steps/s
- RTX 4080 throughput verdict: `noise_band` at `82.40252049829174` steps/s
- Held-out quality verdict: `noise_band` at best loss `15.942383766174316`
- Near-equivalent bridge verdict: `passed` with served token `37`

## Ordering

- Run the local M5 MLX same-node lane first.
- Run the remote archlinux RTX 4080 CUDA lane in the same matrix second.
- Run the PGOLF-ish held-out quality compare on the just-produced bundles third.
- Run the M5 near-equivalent infer/serve bridge fourth.
- Treat the M2 as opportunistic only and do not block the daily loop on it.

## Artifact Roots

- Matrix report: `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/matrix_report.json`
- Quality report: `/Users/christopherdavid/work/psionic/fixtures/apple_adapter/daily/tailrun-daily-scoreboard-validation-20260327/quality/quality_report.json`
- Near-equivalent report: `/Users/christopherdavid/work/psionic/fixtures/apple_adapter/daily/tailrun-daily-scoreboard-validation-20260327/near_equivalent/near_equivalent_report.json`
