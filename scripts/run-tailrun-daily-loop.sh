#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

run_id=""
root_dir=""
target_seconds="600"
remote_host="archlinux"
matrix_root=""
quality_root=""
near_equivalent_root=""
skip_matrix="0"
skip_quality="0"
skip_near_equivalent="0"
baseline_matrix_report="${repo_root}/fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/matrix_report.json"
baseline_quality_report="${repo_root}/fixtures/apple_adapter/runs/tailrun_pgolfish_quality_compare_20260327/quality_report.json"
baseline_near_equivalent_report="${repo_root}/fixtures/apple_adapter/promoted/tailrun_m5_near_equivalent_20260327/near_equivalent_report.json"
admitted_home_summary="${repo_root}/fixtures/swarm/runs/tailrun-home-admitted-20260327e/tailrun_admitted_home_run_summary.json"
throughput_gain_threshold_pct="5"
heldout_loss_improvement_threshold_pct="1"

usage() {
  cat <<'EOF' >&2
Usage: scripts/run-tailrun-daily-loop.sh [options]

Options:
  --run-id <id>                         Stable daily run identifier.
  --root-dir <path>                     Daily artifact root. Default: fixtures/apple_adapter/daily/<run_id>
  --target-seconds <seconds>            Shared matrix wallclock. Default: 600
  --remote-host <host>                  Remote admitted CUDA host. Default: archlinux
  --matrix-root <path>                  Matrix artifact root. Default: <root-dir>/matrix
  --quality-root <path>                 Quality artifact root. Default: <root-dir>/quality
  --near-equivalent-root <path>         Infer/serve artifact root. Default: <root-dir>/near_equivalent
  --baseline-matrix-report <path>       Baseline matrix report for scorekeeping.
  --baseline-quality-report <path>      Baseline quality report for scorekeeping.
  --baseline-near-equivalent-report <path>
                                        Baseline near-equivalent report for scorekeeping.
  --admitted-home-summary <path>        Mixed-device summary context for quality compare.
  --skip-matrix                         Reuse an existing matrix root instead of running the matrix.
  --skip-quality                        Reuse an existing quality root instead of running the quality compare.
  --skip-near-equivalent                Reuse an existing near-equivalent root instead of running the infer/serve bridge.
  --help|-h                             Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      run_id="$2"
      shift 2
      ;;
    --root-dir)
      root_dir="$2"
      shift 2
      ;;
    --target-seconds)
      target_seconds="$2"
      shift 2
      ;;
    --remote-host)
      remote_host="$2"
      shift 2
      ;;
    --matrix-root)
      matrix_root="$2"
      shift 2
      ;;
    --quality-root)
      quality_root="$2"
      shift 2
      ;;
    --near-equivalent-root)
      near_equivalent_root="$2"
      shift 2
      ;;
    --baseline-matrix-report)
      baseline_matrix_report="$2"
      shift 2
      ;;
    --baseline-quality-report)
      baseline_quality_report="$2"
      shift 2
      ;;
    --baseline-near-equivalent-report)
      baseline_near_equivalent_report="$2"
      shift 2
      ;;
    --admitted-home-summary)
      admitted_home_summary="$2"
      shift 2
      ;;
    --skip-matrix)
      skip_matrix="1"
      shift
      ;;
    --skip-quality)
      skip_quality="1"
      shift
      ;;
    --skip-near-equivalent)
      skip_near_equivalent="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${run_id}" ]]; then
  run_id="tailrun-daily-$(date -u +%Y%m%dT%H%M%SZ)"
fi

if [[ -z "${root_dir}" ]]; then
  root_dir="${repo_root}/fixtures/apple_adapter/daily/${run_id}"
fi
mkdir -p "${root_dir}"
root_dir="$(cd "${root_dir}" && pwd)"

if [[ -z "${matrix_root}" ]]; then
  matrix_root="${root_dir}/matrix"
fi
if [[ -z "${quality_root}" ]]; then
  quality_root="${root_dir}/quality"
fi
if [[ -z "${near_equivalent_root}" ]]; then
  near_equivalent_root="${root_dir}/near_equivalent"
fi

matrix_report="${matrix_root}/matrix_report.json"
quality_report="${quality_root}/quality_report.json"
near_equivalent_report="${near_equivalent_root}/near_equivalent_report.json"
scoreboard_json="${root_dir}/daily_scoreboard.json"
scoreboard_md="${root_dir}/daily_scoreboard.md"

if [[ "${skip_matrix}" != "1" ]]; then
  "${repo_root}/scripts/run-open-adapter-tailnet-matrix.sh" \
    --run-id "${run_id}" \
    --bundle-dir "${matrix_root}" \
    --target-seconds "${target_seconds}" \
    --remote-host "${remote_host}"
fi

if [[ "${skip_quality}" != "1" ]]; then
  mkdir -p "${quality_root}"
  (
    cd "${repo_root}"
    cargo run -q -p psionic-train --bin open_adapter_pgolfish_quality_compare -- \
      --output-root "${quality_root}" \
      --m5-report "${matrix_root}/m5_mlx/report.json" \
      --m5-bundle "${matrix_root}/m5_mlx/portable_bundle.safetensors" \
      --cuda-report "${matrix_root}/archlinux_cuda/report.json" \
      --cuda-bundle "${matrix_root}/archlinux_cuda/portable_bundle.safetensors" \
      --admitted-home-summary "${admitted_home_summary}"
  )
fi

if [[ "${skip_near_equivalent}" != "1" ]]; then
  mkdir -p "${near_equivalent_root}"
  (
    cd "${repo_root}"
    cargo run -q -p psionic-serve --example tailrun_open_adapter_near_equivalent_operator -- \
      --source-report "${matrix_root}/m5_mlx/report.json" \
      --source-bundle "${matrix_root}/m5_mlx/portable_bundle.safetensors" \
      --output-root "${near_equivalent_root}"
  )
fi

for required_path in \
  "${matrix_report}" \
  "${quality_report}" \
  "${near_equivalent_report}" \
  "${baseline_matrix_report}" \
  "${baseline_quality_report}" \
  "${baseline_near_equivalent_report}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "error: missing required report ${required_path}" >&2
    exit 1
  fi
done

jq -n \
  --arg schema_version "psionic.tailrun_daily_scoreboard.v1" \
  --arg run_id "${run_id}" \
  --arg created_at_utc "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  --arg target_seconds "${target_seconds}" \
  --arg matrix_report "${matrix_report}" \
  --arg quality_report "${quality_report}" \
  --arg near_equivalent_report "${near_equivalent_report}" \
  --arg baseline_matrix_report "${baseline_matrix_report}" \
  --arg baseline_quality_report "${baseline_quality_report}" \
  --arg baseline_near_equivalent_report "${baseline_near_equivalent_report}" \
  --arg throughput_gain_threshold_pct "${throughput_gain_threshold_pct}" \
  --arg heldout_loss_improvement_threshold_pct "${heldout_loss_improvement_threshold_pct}" \
  --slurpfile current_matrix "${matrix_report}" \
  --slurpfile current_quality "${quality_report}" \
  --slurpfile current_near "${near_equivalent_report}" \
  --slurpfile baseline_matrix "${baseline_matrix_report}" \
  --slurpfile baseline_quality "${baseline_quality_report}" \
  --slurpfile baseline_near "${baseline_near_equivalent_report}" \
  '
  def pct_delta(new; old):
    if old == 0 then null else ((new - old) / old) * 100 end;
  def pct_improvement_lower_is_better(new; old):
    if old == 0 then null else ((old - new) / old) * 100 end;
  def status_higher(new; old; threshold):
    if old == 0 then "unscored"
    elif pct_delta(new; old) >= threshold then "meaningful_improvement"
    elif pct_delta(new; old) <= (-threshold) then "meaningful_regression"
    else "noise_band"
    end;
  def status_lower(new; old; threshold):
    if old == 0 then "unscored"
    elif pct_improvement_lower_is_better(new; old) >= threshold then "meaningful_improvement"
    elif pct_improvement_lower_is_better(new; old) <= (-threshold) then "meaningful_regression"
    else "noise_band"
    end;
  def overlay_token(report):
    report.overlay_served_output_tokens[0];
  def bridge_passed(report):
    (report.direct_inference_predicted_token_id == report.expected_target_token_id)
    and (overlay_token(report) == report.expected_target_token_id);
  def any_improvement(values):
    any(values[]; . == "meaningful_improvement");
  def any_regression(values):
    any(values[]; . == "meaningful_regression");
  def overall(m5; cuda; quality; bridge):
    if bridge != "passed" then "bridge_failed"
    elif any_improvement([m5, cuda, quality]) then
      if quality == "meaningful_improvement" and any_improvement([m5, cuda]) then
        "quality_and_throughput_improved"
      elif quality == "meaningful_improvement" then
        "quality_improved"
      else
        "throughput_improved"
      end
    elif any_regression([m5, cuda, quality]) then
      "regression"
    else
      "stable_no_clear_gain"
    end;
  ($throughput_gain_threshold_pct | tonumber) as $throughput_threshold
  | ($heldout_loss_improvement_threshold_pct | tonumber) as $quality_threshold
  | ($current_matrix[0]) as $cm
  | ($current_quality[0]) as $cq
  | ($current_near[0]) as $cn
  | ($baseline_matrix[0]) as $bm
  | ($baseline_quality[0]) as $bq
  | ($baseline_near[0]) as $bn
  | ($cq.evaluated_artifacts[] | select(.artifact_id == "same_node_m5_mlx")) as $current_m5_quality
  | ($cq.evaluated_artifacts[] | select(.artifact_id == "same_node_rtx4080_cuda")) as $current_cuda_quality
  | ($bq.evaluated_artifacts[] | select(.artifact_id == "same_node_m5_mlx")) as $baseline_m5_quality
  | ($bq.evaluated_artifacts[] | select(.artifact_id == "same_node_rtx4080_cuda")) as $baseline_cuda_quality
  | status_higher($cm.local_report.retained_run.steps_per_second; $bm.local_report.retained_run.steps_per_second; $throughput_threshold) as $m5_status
  | status_higher($cm.remote_report.retained_run.steps_per_second; $bm.remote_report.retained_run.steps_per_second; $throughput_threshold) as $cuda_status
  | status_lower($cq.comparison.best_heldout_mean_loss; $bq.comparison.best_heldout_mean_loss; $quality_threshold) as $quality_status
  | (if bridge_passed($cn) then "passed" else "failed" end) as $bridge_status
  | {
      schema_version: $schema_version,
      run_id: $run_id,
      created_at_utc: $created_at_utc,
      target_wallclock_seconds: ($target_seconds | tonumber),
      admitted_daily_operator_ordering: [
        "Run the local M5 MLX same-node lane first.",
        "Run the remote archlinux RTX 4080 CUDA lane in the same matrix second.",
        "Run the PGOLF-ish held-out quality compare on the just-produced bundles third.",
        "Run the M5 near-equivalent infer/serve bridge fourth.",
        "Treat the M2 as opportunistic only and do not block the daily loop on it."
      ],
      command_contract: {
        matrix: "scripts/run-open-adapter-tailnet-matrix.sh --run-id <run_id> --bundle-dir <matrix_root> --target-seconds 600 --remote-host archlinux",
        quality: "cargo run -q -p psionic-train --bin open_adapter_pgolfish_quality_compare -- --output-root <quality_root> --m5-report <matrix_root>/m5_mlx/report.json --m5-bundle <matrix_root>/m5_mlx/portable_bundle.safetensors --cuda-report <matrix_root>/archlinux_cuda/report.json --cuda-bundle <matrix_root>/archlinux_cuda/portable_bundle.safetensors --admitted-home-summary fixtures/swarm/runs/tailrun-home-admitted-20260327e/tailrun_admitted_home_run_summary.json",
        near_equivalent: "cargo run -q -p psionic-serve --example tailrun_open_adapter_near_equivalent_operator -- --source-report <matrix_root>/m5_mlx/report.json --source-bundle <matrix_root>/m5_mlx/portable_bundle.safetensors --output-root <near_equivalent_root>"
      },
      current_paths: {
        matrix_report: $matrix_report,
        quality_report: $quality_report,
        near_equivalent_report: $near_equivalent_report
      },
      baseline_paths: {
        matrix_report: $baseline_matrix_report,
        quality_report: $baseline_quality_report,
        near_equivalent_report: $baseline_near_equivalent_report
      },
      improvement_thresholds: {
        throughput_gain_pct_min: $throughput_threshold,
        heldout_loss_improvement_pct_min: $quality_threshold,
        infer_bridge_requires_direct_and_served_target_match: true
      },
      current_metrics: {
        matrix: {
          m5_steps_per_second: $cm.local_report.retained_run.steps_per_second,
          cuda_steps_per_second: $cm.remote_report.retained_run.steps_per_second,
          local_over_remote_steps_gain_pct: $cm.comparison.steps_per_second_gain_pct_local_over_remote
        },
        quality: {
          best_heldout_artifact_id: $cq.comparison.best_heldout_artifact_id,
          best_heldout_mean_loss: $cq.comparison.best_heldout_mean_loss,
          m5_heldout_mean_loss: $current_m5_quality.heldout_mean_loss,
          cuda_heldout_mean_loss: $current_cuda_quality.heldout_mean_loss
        },
        near_equivalent: {
          expected_target_token_id: $cn.expected_target_token_id,
          direct_inference_predicted_token_id: $cn.direct_inference_predicted_token_id,
          overlay_served_output_token_id: overlay_token($cn),
          runtime_support_level: $cn.runtime_support_level
        }
      },
      baseline_metrics: {
        matrix: {
          m5_steps_per_second: $bm.local_report.retained_run.steps_per_second,
          cuda_steps_per_second: $bm.remote_report.retained_run.steps_per_second
        },
        quality: {
          best_heldout_artifact_id: $bq.comparison.best_heldout_artifact_id,
          best_heldout_mean_loss: $bq.comparison.best_heldout_mean_loss,
          m5_heldout_mean_loss: $baseline_m5_quality.heldout_mean_loss,
          cuda_heldout_mean_loss: $baseline_cuda_quality.heldout_mean_loss
        },
        near_equivalent: {
          expected_target_token_id: $bn.expected_target_token_id,
          direct_inference_predicted_token_id: $bn.direct_inference_predicted_token_id,
          overlay_served_output_token_id: overlay_token($bn)
        }
      },
      comparison: {
        m5_steps_gain_pct_vs_baseline:
          pct_delta($cm.local_report.retained_run.steps_per_second; $bm.local_report.retained_run.steps_per_second),
        cuda_steps_gain_pct_vs_baseline:
          pct_delta($cm.remote_report.retained_run.steps_per_second; $bm.remote_report.retained_run.steps_per_second),
        best_heldout_loss_improvement_pct_vs_baseline:
          pct_improvement_lower_is_better($cq.comparison.best_heldout_mean_loss; $bq.comparison.best_heldout_mean_loss),
        near_equivalent_direct_match:
          ($cn.direct_inference_predicted_token_id == $cn.expected_target_token_id),
        near_equivalent_served_match:
          (overlay_token($cn) == $cn.expected_target_token_id)
      },
      verdict: {
        m5_throughput: $m5_status,
        cuda_throughput: $cuda_status,
        heldout_quality: $quality_status,
        near_equivalent_bridge: $bridge_status,
        overall: overall($m5_status; $cuda_status; $quality_status; $bridge_status)
      },
      claim_boundary:
        "This scoreboard tracks the admitted daily home-Tailnet operator loop over the local M5 MLX lane, the remote archlinux RTX 4080 CUDA lane, the PGOLF-ish held-out comparison, and the same-node M5 near-equivalent infer/serve bridge. It does not claim that the M2 participated, that the mixed-device swarm artifact already has the same inferable bundle path, or that this lane is exact PGOLF or full decentralized internet training."
    }' >"${scoreboard_json}"

jq -r '
  "# Tailrun Daily Scoreboard",
  "",
  "- Run id: `\(.run_id)`",
  "- Overall verdict: `\(.verdict.overall)`",
  "- M5 throughput verdict: `\(.verdict.m5_throughput)` at `\(.current_metrics.matrix.m5_steps_per_second)` steps/s",
  "- RTX 4080 throughput verdict: `\(.verdict.cuda_throughput)` at `\(.current_metrics.matrix.cuda_steps_per_second)` steps/s",
  "- Held-out quality verdict: `\(.verdict.heldout_quality)` at best loss `\(.current_metrics.quality.best_heldout_mean_loss)`",
  "- Near-equivalent bridge verdict: `\(.verdict.near_equivalent_bridge)` with served token `\(.current_metrics.near_equivalent.overlay_served_output_token_id)`",
  "",
  "## Ordering",
  "",
  (.admitted_daily_operator_ordering[] | "- " + .),
  "",
  "## Artifact Roots",
  "",
  "- Matrix report: `\(.current_paths.matrix_report)`",
  "- Quality report: `\(.current_paths.quality_report)`",
  "- Near-equivalent report: `\(.current_paths.near_equivalent_report)`"
' "${scoreboard_json}" >"${scoreboard_md}"

echo "wrote daily Tailrun scoreboard ${scoreboard_json}"
echo "wrote daily Tailrun summary ${scoreboard_md}"
