#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
PROJECT_NUMBER="${PROJECT_NUMBER:-157437760789}"
BILLING_ACCOUNT_ID="${BILLING_ACCOUNT_ID:-01D15C-64524A-1062EA}"
BQ_LOCATION="${BQ_LOCATION:-US}"
BUCKET_URL="${BUCKET_URL:-gs://openagentsgemini-psion-train-us-central1}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
GUARDRAIL_FILE="${REPO_ROOT}/fixtures/psion/google/psion_google_billing_guardrails_v1.json"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

if ! command -v bq >/dev/null 2>&1; then
  echo "error: bq is required" >&2
  exit 1
fi

budget_display_name="$(
  jq -r '.budget.display_name' "${GUARDRAIL_FILE}"
)"
budget_amount_usd="$(
  jq -r '.budget.amount_usd' "${GUARDRAIL_FILE}"
)"
pubsub_topic="$(
  jq -r '.budget.pubsub_topic' "${GUARDRAIL_FILE}"
)"
pull_subscription="$(
  jq -r '.budget.pull_subscription' "${GUARDRAIL_FILE}"
)"
topic_id="${pubsub_topic##*/}"
subscription_id="${pull_subscription##*/}"
dataset_id="$(
  jq -r '.machine_queryable_cost_sink.dataset_id' "${GUARDRAIL_FILE}"
)"
table_id="$(
  jq -r '.machine_queryable_cost_sink.price_profiles_table' "${GUARDRAIL_FILE}"
)"
dataset_ref="${PROJECT_ID}:${dataset_id}"
table_ref="${dataset_ref}.${table_id}"

wait_for_pubsub_topic() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if gcloud pubsub topics describe "${topic_id}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "error: topic ${pubsub_topic} did not become visible" >&2
  exit 1
}

wait_for_billing_budget() {
  local budget_name="$1"
  local attempt
  for attempt in 1 2 3 4 5; do
    if gcloud billing budgets describe "${budget_name}" --billing-account="${BILLING_ACCOUNT_ID}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "error: budget ${budget_name} did not become visible" >&2
  exit 1
}

wait_for_manifest_upload() {
  local manifest_path="${BUCKET_URL}/manifests/psion_google_billing_guardrails_v1.json"
  local attempt
  for attempt in 1 2 3 4 5; do
    if gcloud storage ls "${manifest_path}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "error: billing guardrails manifest upload did not become visible" >&2
  exit 1
}

ensure_topic() {
  if ! gcloud pubsub topics describe "${topic_id}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud pubsub topics create "${topic_id}" --project="${PROJECT_ID}" >/dev/null
  fi
  wait_for_pubsub_topic
}

ensure_subscription() {
  if ! gcloud pubsub subscriptions describe "${subscription_id}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud pubsub subscriptions create "${subscription_id}" \
      --topic="${topic_id}" \
      --project="${PROJECT_ID}" >/dev/null
  fi
}

ensure_dataset() {
  if ! bq show --format=prettyjson "${dataset_ref}" >/dev/null 2>&1; then
    bq --location="${BQ_LOCATION}" mk --dataset \
      --description "Psion Google single-node training FinOps tables." \
      "${dataset_ref}" >/dev/null
  fi
}

fetch_catalog_jsonl() {
  local outfile="$1"
  local token
  local page_token=""
  token="$(gcloud auth print-access-token)"
  : > "${outfile}"
  while true; do
    local url="https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?pageSize=5000&currencyCode=USD"
    if [[ -n "${page_token}" ]]; then
      url="${url}&pageToken=${page_token}"
    fi
    local response
    response="$(curl -sf -H "Authorization: Bearer ${token}" "${url}")"
    jq -c '.skus[]' <<<"${response}" >> "${outfile}"
    page_token="$(jq -r '.nextPageToken // empty' <<<"${response}")"
    if [[ -z "${page_token}" ]]; then
      break
    fi
  done
}

price_from_description() {
  local catalog_file="$1"
  local description="$2"
  jq -r --arg desc "${description}" '
    select(.serviceRegions | index("us-central1"))
    | select(.description == $desc)
    | .pricingInfo[0].pricingExpression.tieredRates[0].unitPrice
    | ((.units // "0") | tonumber) + (((.nanos // 0) | tonumber) / 1000000000)
  ' "${catalog_file}" | head -n 1
}

hourly_from_rate_and_units() {
  local rate="$1"
  local units="$2"
  awk -v rate="${rate}" -v units="${units}" 'BEGIN { printf "%.9f", rate * units }'
}

hourly_from_monthly_gb_rate() {
  local rate="$1"
  local gb="$2"
  awk -v rate="${rate}" -v gb="${gb}" 'BEGIN { printf "%.9f", (rate * gb) / 730.0 }'
}

sum_costs() {
  printf '%s\n' "$@" | awk '{ total += $1 } END { printf "%.9f", total }'
}

profile_rows_jsonl() {
  local catalog_file="$1"
  local outfile="$2"
  local captured_at_utc="$3"
  local snapshot_id
  snapshot_id="psion-google-price-snapshot-${captured_at_utc}"

  local g2_cpu_rate g2_ram_rate l4_rate a2_cpu_rate a2_ram_rate a100_rate disk_monthly_rate
  g2_cpu_rate="$(price_from_description "${catalog_file}" "G2 Instance Core running in Americas")"
  g2_ram_rate="$(price_from_description "${catalog_file}" "G2 Instance Ram running in Americas")"
  l4_rate="$(price_from_description "${catalog_file}" "Nvidia L4 GPU running in Americas")"
  a2_cpu_rate="$(price_from_description "${catalog_file}" "A2 Instance Core running in Americas")"
  a2_ram_rate="$(price_from_description "${catalog_file}" "A2 Instance Ram running in Americas")"
  a100_rate="$(price_from_description "${catalog_file}" "Nvidia Tesla A100 GPU running in Americas")"
  disk_monthly_rate="$(price_from_description "${catalog_file}" "Balanced PD Capacity")"

  if [[ -z "${g2_cpu_rate}" || -z "${g2_ram_rate}" || -z "${l4_rate}" || -z "${a2_cpu_rate}" || -z "${a2_ram_rate}" || -z "${a100_rate}" || -z "${disk_monthly_rate}" ]]; then
    echo "error: failed to resolve one or more us-central1 Compute Engine price catalog rows" >&2
    exit 1
  fi

  local g2_cpu_hourly g2_ram_hourly l4_disk_hourly g2_hourly g2_estimated_run
  g2_cpu_hourly="$(hourly_from_rate_and_units "${g2_cpu_rate}" 8)"
  g2_ram_hourly="$(hourly_from_rate_and_units "${g2_ram_rate}" 32)"
  l4_disk_hourly="$(hourly_from_monthly_gb_rate "${disk_monthly_rate}" 200)"
  g2_hourly="$(sum_costs "${g2_cpu_hourly}" "${g2_ram_hourly}" "${l4_rate}" "${l4_disk_hourly}")"
  g2_estimated_run="$(hourly_from_rate_and_units "${g2_hourly}" 12)"

  local a2_cpu_hourly a2_ram_hourly a100_disk_hourly a100_hourly a100_estimated_run
  a2_cpu_hourly="$(hourly_from_rate_and_units "${a2_cpu_rate}" 12)"
  a2_ram_hourly="$(hourly_from_rate_and_units "${a2_ram_rate}" 85)"
  a100_disk_hourly="$(hourly_from_monthly_gb_rate "${disk_monthly_rate}" 250)"
  a100_hourly="$(sum_costs "${a2_cpu_hourly}" "${a2_ram_hourly}" "${a100_rate}" "${a100_disk_hourly}")"
  a100_estimated_run="$(hourly_from_rate_and_units "${a100_hourly}" 12)"

  jq -c -n \
    --arg snapshot_id "${snapshot_id}" \
    --arg captured_at_utc "${captured_at_utc}" \
    --arg project_id "${PROJECT_ID}" \
    --arg profile_id "g2_l4_single_node" \
    --arg machine_type "g2-standard-8" \
    --arg accelerator_type "nvidia-l4" \
    --argjson boot_disk_gb 200 \
    --argjson max_runtime_hours 12 \
    --argjson declared_run_cost_ceiling_usd 15 \
    --arg cpu_sku_description "G2 Instance Core running in Americas" \
    --arg ram_sku_description "G2 Instance Ram running in Americas" \
    --arg accelerator_sku_description "Nvidia L4 GPU running in Americas" \
    --arg disk_sku_description "Balanced PD Capacity" \
    --argjson cpu_hourly_usd "${g2_cpu_hourly}" \
    --argjson ram_hourly_usd "${g2_ram_hourly}" \
    --argjson accelerator_hourly_usd "${l4_rate}" \
    --argjson boot_disk_hourly_usd "${l4_disk_hourly}" \
    --argjson estimated_hourly_usd "${g2_hourly}" \
    --argjson estimated_run_cost_usd "${g2_estimated_run}" \
    '{
      snapshot_id: $snapshot_id,
      captured_at_utc: $captured_at_utc,
      project_id: $project_id,
      region: "us-central1",
      profile_id: $profile_id,
      machine_type: $machine_type,
      accelerator_type: $accelerator_type,
      accelerator_count: 1,
      boot_disk_type: "pd-balanced",
      boot_disk_gb: $boot_disk_gb,
      max_runtime_hours: $max_runtime_hours,
      declared_run_cost_ceiling_usd: $declared_run_cost_ceiling_usd,
      cpu_hourly_usd: $cpu_hourly_usd,
      ram_hourly_usd: $ram_hourly_usd,
      accelerator_hourly_usd: $accelerator_hourly_usd,
      boot_disk_hourly_usd: $boot_disk_hourly_usd,
      estimated_hourly_usd: $estimated_hourly_usd,
      estimated_run_cost_usd: $estimated_run_cost_usd,
      cpu_sku_description: $cpu_sku_description,
      ram_sku_description: $ram_sku_description,
      accelerator_sku_description: $accelerator_sku_description,
      disk_sku_description: $disk_sku_description
    }' > "${outfile}"

  jq -c -n \
    --arg snapshot_id "${snapshot_id}" \
    --arg captured_at_utc "${captured_at_utc}" \
    --arg project_id "${PROJECT_ID}" \
    --arg profile_id "g2_l4_single_node_accelerated" \
    --arg machine_type "g2-standard-8" \
    --arg accelerator_type "nvidia-l4" \
    --argjson boot_disk_gb 200 \
    --argjson max_runtime_hours 12 \
    --argjson declared_run_cost_ceiling_usd 20 \
    --arg cpu_sku_description "G2 Instance Core running in Americas" \
    --arg ram_sku_description "G2 Instance Ram running in Americas" \
    --arg accelerator_sku_description "Nvidia L4 GPU running in Americas" \
    --arg disk_sku_description "Balanced PD Capacity" \
    --argjson cpu_hourly_usd "${g2_cpu_hourly}" \
    --argjson ram_hourly_usd "${g2_ram_hourly}" \
    --argjson accelerator_hourly_usd "${l4_rate}" \
    --argjson boot_disk_hourly_usd "${l4_disk_hourly}" \
    --argjson estimated_hourly_usd "${g2_hourly}" \
    --argjson estimated_run_cost_usd "${g2_estimated_run}" \
    '{
      snapshot_id: $snapshot_id,
      captured_at_utc: $captured_at_utc,
      project_id: $project_id,
      region: "us-central1",
      profile_id: $profile_id,
      machine_type: $machine_type,
      accelerator_type: $accelerator_type,
      accelerator_count: 1,
      boot_disk_type: "pd-balanced",
      boot_disk_gb: $boot_disk_gb,
      max_runtime_hours: $max_runtime_hours,
      declared_run_cost_ceiling_usd: $declared_run_cost_ceiling_usd,
      cpu_hourly_usd: $cpu_hourly_usd,
      ram_hourly_usd: $ram_hourly_usd,
      accelerator_hourly_usd: $accelerator_hourly_usd,
      boot_disk_hourly_usd: $boot_disk_hourly_usd,
      estimated_hourly_usd: $estimated_hourly_usd,
      estimated_run_cost_usd: $estimated_run_cost_usd,
      cpu_sku_description: $cpu_sku_description,
      ram_sku_description: $ram_sku_description,
      accelerator_sku_description: $accelerator_sku_description,
      disk_sku_description: $disk_sku_description
    }' >> "${outfile}"

  jq -c -n \
    --arg snapshot_id "${snapshot_id}" \
    --arg captured_at_utc "${captured_at_utc}" \
    --arg project_id "${PROJECT_ID}" \
    --arg profile_id "g2_l4_single_node_plugin_host_native_accelerated" \
    --arg machine_type "g2-standard-8" \
    --arg accelerator_type "nvidia-l4" \
    --argjson boot_disk_gb 200 \
    --argjson max_runtime_hours 12 \
    --argjson declared_run_cost_ceiling_usd 25 \
    --arg cpu_sku_description "G2 Instance Core running in Americas" \
    --arg ram_sku_description "G2 Instance Ram running in Americas" \
    --arg accelerator_sku_description "Nvidia L4 GPU running in Americas" \
    --arg disk_sku_description "Balanced PD Capacity" \
    --argjson cpu_hourly_usd "${g2_cpu_hourly}" \
    --argjson ram_hourly_usd "${g2_ram_hourly}" \
    --argjson accelerator_hourly_usd "${l4_rate}" \
    --argjson boot_disk_hourly_usd "${l4_disk_hourly}" \
    --argjson estimated_hourly_usd "${g2_hourly}" \
    --argjson estimated_run_cost_usd "${g2_estimated_run}" \
    '{
      snapshot_id: $snapshot_id,
      captured_at_utc: $captured_at_utc,
      project_id: $project_id,
      region: "us-central1",
      profile_id: $profile_id,
      machine_type: $machine_type,
      accelerator_type: $accelerator_type,
      accelerator_count: 1,
      boot_disk_type: "pd-balanced",
      boot_disk_gb: $boot_disk_gb,
      max_runtime_hours: $max_runtime_hours,
      declared_run_cost_ceiling_usd: $declared_run_cost_ceiling_usd,
      cpu_hourly_usd: $cpu_hourly_usd,
      ram_hourly_usd: $ram_hourly_usd,
      accelerator_hourly_usd: $accelerator_hourly_usd,
      boot_disk_hourly_usd: $boot_disk_hourly_usd,
      estimated_hourly_usd: $estimated_hourly_usd,
      estimated_run_cost_usd: $estimated_run_cost_usd,
      cpu_sku_description: $cpu_sku_description,
      ram_sku_description: $ram_sku_description,
      accelerator_sku_description: $accelerator_sku_description,
      disk_sku_description: $disk_sku_description
    }' >> "${outfile}"

  jq -c -n \
    --arg snapshot_id "${snapshot_id}" \
    --arg captured_at_utc "${captured_at_utc}" \
    --arg project_id "${PROJECT_ID}" \
    --arg profile_id "a2_a100_single_node" \
    --arg machine_type "a2-highgpu-1g" \
    --arg accelerator_type "nvidia-tesla-a100" \
    --argjson boot_disk_gb 250 \
    --argjson max_runtime_hours 12 \
    --argjson declared_run_cost_ceiling_usd 50 \
    --arg cpu_sku_description "A2 Instance Core running in Americas" \
    --arg ram_sku_description "A2 Instance Ram running in Americas" \
    --arg accelerator_sku_description "Nvidia Tesla A100 GPU running in Americas" \
    --arg disk_sku_description "Balanced PD Capacity" \
    --argjson cpu_hourly_usd "${a2_cpu_hourly}" \
    --argjson ram_hourly_usd "${a2_ram_hourly}" \
    --argjson accelerator_hourly_usd "${a100_rate}" \
    --argjson boot_disk_hourly_usd "${a100_disk_hourly}" \
    --argjson estimated_hourly_usd "${a100_hourly}" \
    --argjson estimated_run_cost_usd "${a100_estimated_run}" \
    '{
      snapshot_id: $snapshot_id,
      captured_at_utc: $captured_at_utc,
      project_id: $project_id,
      region: "us-central1",
      profile_id: $profile_id,
      machine_type: $machine_type,
      accelerator_type: $accelerator_type,
      accelerator_count: 1,
      boot_disk_type: "pd-balanced",
      boot_disk_gb: $boot_disk_gb,
      max_runtime_hours: $max_runtime_hours,
      declared_run_cost_ceiling_usd: $declared_run_cost_ceiling_usd,
      cpu_hourly_usd: $cpu_hourly_usd,
      ram_hourly_usd: $ram_hourly_usd,
      accelerator_hourly_usd: $accelerator_hourly_usd,
      boot_disk_hourly_usd: $boot_disk_hourly_usd,
      estimated_hourly_usd: $estimated_hourly_usd,
      estimated_run_cost_usd: $estimated_run_cost_usd,
      cpu_sku_description: $cpu_sku_description,
      ram_sku_description: $ram_sku_description,
      accelerator_sku_description: $accelerator_sku_description,
      disk_sku_description: $disk_sku_description
    }' >> "${outfile}"
}

load_cost_profiles() {
  local data_file="$1"
  local schema_file="$2"

  cat > "${schema_file}" <<'EOF'
[
  {"name":"snapshot_id","type":"STRING"},
  {"name":"captured_at_utc","type":"TIMESTAMP"},
  {"name":"project_id","type":"STRING"},
  {"name":"region","type":"STRING"},
  {"name":"profile_id","type":"STRING"},
  {"name":"machine_type","type":"STRING"},
  {"name":"accelerator_type","type":"STRING"},
  {"name":"accelerator_count","type":"INTEGER"},
  {"name":"boot_disk_type","type":"STRING"},
  {"name":"boot_disk_gb","type":"INTEGER"},
  {"name":"max_runtime_hours","type":"FLOAT"},
  {"name":"declared_run_cost_ceiling_usd","type":"FLOAT"},
  {"name":"cpu_hourly_usd","type":"FLOAT"},
  {"name":"ram_hourly_usd","type":"FLOAT"},
  {"name":"accelerator_hourly_usd","type":"FLOAT"},
  {"name":"boot_disk_hourly_usd","type":"FLOAT"},
  {"name":"estimated_hourly_usd","type":"FLOAT"},
  {"name":"estimated_run_cost_usd","type":"FLOAT"},
  {"name":"cpu_sku_description","type":"STRING"},
  {"name":"ram_sku_description","type":"STRING"},
  {"name":"accelerator_sku_description","type":"STRING"},
  {"name":"disk_sku_description","type":"STRING"}
]
EOF

  if ! bq show --format=prettyjson "${table_ref}" >/dev/null 2>&1; then
    bq mk --table "${table_ref}" "${schema_file}" >/dev/null
  fi

  bq load \
    --replace \
    --source_format=NEWLINE_DELIMITED_JSON \
    "${table_ref}" \
    "${data_file}" \
    "${schema_file}" >/dev/null
}

ensure_budget() {
  local current_budget_name
  local current_budget_json
  local desired_budget_json
  local normalized_current_json
  local budget_name
  budget_name="$(
    gcloud billing budgets list --billing-account="${BILLING_ACCOUNT_ID}" --format=json | \
      jq -r --arg display_name "${budget_display_name}" '.[] | select(.displayName == $display_name) | .name'
  )"

  if [[ -n "${budget_name}" ]]; then
    current_budget_name="${budget_name}"
    current_budget_json="$(
      gcloud billing budgets describe "${current_budget_name}" \
        --billing-account="${BILLING_ACCOUNT_ID}" \
        --format=json
    )"
    normalized_current_json="$(
      jq -c '
        {
          amount_usd: (
            ((.amount.specifiedAmount.units // "0") | tonumber)
            + (((.amount.specifiedAmount.nanos // 0) | tonumber) / 1000000000)
          ),
          projects: ((.budgetFilter.projects // []) | sort),
          topic: (.notificationsRule.pubsubTopic // null),
          thresholds: (
            (.thresholdRules // [])
            | map({
                basis: (.spendBasis | ascii_downcase | gsub("_"; "-")),
                percent: .thresholdPercent
              })
            | sort_by(.basis, .percent)
          )
        }
      ' <<<"${current_budget_json}"
    )"
    desired_budget_json="$(
      jq -c \
        --arg project_number "projects/${PROJECT_NUMBER}" \
        --arg topic "${pubsub_topic}" \
        --argjson amount_usd "${budget_amount_usd}" \
        '{
          amount_usd: $amount_usd,
          projects: [$project_number],
          topic: $topic,
          thresholds: (.budget.threshold_rules | sort_by(.basis, .percent))
        }' "${GUARDRAIL_FILE}"
    )"

    if [[ "$(
      jq -n \
        --argjson current "${normalized_current_json}" \
        --argjson desired "${desired_budget_json}" \
        '$current == $desired'
    )" != "true" ]]; then
      gcloud billing budgets delete "${current_budget_name}" \
        --billing-account="${BILLING_ACCOUNT_ID}" \
        --quiet >/dev/null
      budget_name=""
    fi
  fi

  if [[ -z "${budget_name}" ]]; then
    budget_name="$(
      gcloud billing budgets create \
        --billing-account="${BILLING_ACCOUNT_ID}" \
        --display-name="${budget_display_name}" \
        --budget-amount="${budget_amount_usd}USD" \
        --calendar-period=month \
        --filter-projects="projects/${PROJECT_ID}" \
        --ownership-scope=all-users \
        --threshold-rule=percent=0.50 \
        --threshold-rule=percent=0.75 \
        --threshold-rule=percent=0.90 \
        --threshold-rule=percent=1.00,basis=forecasted-spend \
        --notifications-rule-pubsub-topic="${pubsub_topic}" \
        --format='value(name)'
    )"
  fi

  wait_for_billing_budget "${budget_name}"
  printf '%s\n' "${budget_name}"
}

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

gcloud services enable billingbudgets.googleapis.com pubsub.googleapis.com bigquery.googleapis.com \
  --project="${PROJECT_ID}" >/dev/null

ensure_topic
ensure_subscription
ensure_dataset

catalog_file="${tmpdir}/compute_engine_catalog.jsonl"
price_rows_file="${tmpdir}/single_node_price_profiles.jsonl"
schema_file="${tmpdir}/single_node_price_profiles.schema.json"
captured_at_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

fetch_catalog_jsonl "${catalog_file}"
profile_rows_jsonl "${catalog_file}" "${price_rows_file}" "${captured_at_utc}"
load_cost_profiles "${price_rows_file}" "${schema_file}"
budget_name="$(ensure_budget)"

gcloud storage cp --quiet \
  "${GUARDRAIL_FILE}" \
  "${BUCKET_URL}/manifests/psion_google_billing_guardrails_v1.json" >/dev/null
wait_for_manifest_upload

echo "billing guardrails summary:"
gcloud billing budgets describe "${budget_name}" \
  --billing-account="${BILLING_ACCOUNT_ID}" \
  --format=json | jq '{name, displayName, amount, budgetFilter, thresholdRules, notificationsRule}'

echo "machine-queryable cost sink summary:"
bq query --use_legacy_sql=false --format=prettyjson \
  "SELECT profile_id, estimated_hourly_usd, estimated_run_cost_usd, declared_run_cost_ceiling_usd FROM \`${PROJECT_ID}.${dataset_id}.${table_id}\` ORDER BY profile_id" \
  | jq '.'
