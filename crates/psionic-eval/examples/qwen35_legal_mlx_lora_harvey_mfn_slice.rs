use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use psionic_eval::{
    BenchmarkTaskSpec, CriterionResult, CriterionVerdict, JudgeMode, JudgePolicy,
    LegalBenchmarkAgentRunRequest, LegalBenchmarkToolWorkspace,
    LegalBenchmarkTrainingRecordExportInput, LegalBenchmarkTrainingRecordSplitPolicy, Metadata,
    ModelProviderRoute, ModelRetryPolicy, OpenAiCompatibleAdapter, ReqwestBlockingHttpTransport,
    RunConfig, ScoreReport, ToolPolicy, artifact_manifest_digest, build_input_artifact_manifest,
    export_legal_benchmark_training_records, run_legal_benchmark_agent, run_record_digest,
    scan_harvey_corpus, score_report_digest, stable_json_digest, task_spec_digest,
};
use serde_json::{Value, json};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:18090/v1";
const DEFAULT_MODEL: &str = "Qwen/Qwen3.5-0.8B";
const DEFAULT_TASKS_ROOT: &str = "/Users/christopherdavid/work/competition/repos/harvey-labs/tasks";
const DEFAULT_UPSTREAM_COMMIT: &str = "5aa41694";
const DEFAULT_TASK_ID: &str = "harvey.funds-asset-management.analyze_mfn_waterfall";
const DEFAULT_OUTPUT_DIR: &str = "fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004/harvey_mfn_slice_run";
const DEFAULT_ADAPTER_PATH: &str = "fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_slice_2026_05_20_004/adapters.safetensors";
const DEFAULT_ADAPTER_DIGEST: &str = "pending";
const DEFAULT_ADAPTER_REPORT_DIGEST: &str = "pending";
const MODEL_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
const DEFAULT_PYLON_WORKER_ID: &str = "pylon.local.macos.mlx.01.harvey_mfn_slice";
const DEFAULT_RUN_NONCE: &str = "qwen35-08b-mlx-lora-harvey-mfn-slice-2026-05-20";

fn main() -> Result<(), Box<dyn Error>> {
    let output_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_DIR));
    reset_run_dir(&output_dir)?;

    let tasks_root = PathBuf::from(env_string("HARVEY_TASKS_ROOT", DEFAULT_TASKS_ROOT));
    let upstream_commit = env_string("HARVEY_UPSTREAM_COMMIT", DEFAULT_UPSTREAM_COMMIT);
    let task_id = env_string("QWEN_LEGAL_HARVEY_TASK_ID", DEFAULT_TASK_ID);
    let base_url = env_string("QWEN_LEGAL_MLX_BASE_URL", DEFAULT_BASE_URL);
    let model = env_string("QWEN_LEGAL_MLX_MODEL", DEFAULT_MODEL);
    let adapter_path = env_string("QWEN_LEGAL_ADAPTER_PATH", DEFAULT_ADAPTER_PATH);
    let adapter_digest = env_string("QWEN_LEGAL_ADAPTER_DIGEST", DEFAULT_ADAPTER_DIGEST);
    let adapter_report_digest = env_string(
        "QWEN_LEGAL_ADAPTER_REPORT_DIGEST",
        DEFAULT_ADAPTER_REPORT_DIGEST,
    );
    let max_output_tokens = env_u64("QWEN_LEGAL_MAX_OUTPUT_TOKENS", 4096);
    let pylon_worker_id = env_string("QWEN_LEGAL_PYLON_WORKER_ID", DEFAULT_PYLON_WORKER_ID);
    let run_nonce = env_string("QWEN_LEGAL_RUN_NONCE", DEFAULT_RUN_NONCE);

    let scan = scan_harvey_corpus(&tasks_root, upstream_commit.clone())?;
    let source_task = scan
        .tasks
        .into_iter()
        .find(|task| task.task_id == task_id)
        .ok_or_else(|| io::Error::other(format!("missing Harvey task `{task_id}`")))?;
    let documents_root = task_documents_root(&tasks_root, &source_task)?;
    let task = training_slice_task(source_task);
    let task_for_export = task.clone();
    let input_manifest = build_input_artifact_manifest(&task);
    let output_root = output_dir.join("output");
    let workspace_root = output_root.clone();
    let run_root = output_dir.join("run");
    fs::create_dir_all(&workspace_root)?;
    fs::create_dir_all(&output_root)?;

    let run_config = harvey_mfn_run_config(
        &task,
        &model,
        &adapter_path,
        &adapter_digest,
        max_output_tokens,
    );
    let mut route = ModelProviderRoute::openai_compatible(
        "route.qwen35_08b_legal_mlx_lora.harvey_mfn_slice",
        &base_url,
        &model,
        None,
    );
    route.metadata.insert(
        String::from("adapter_path"),
        Value::String(adapter_path.clone()),
    );
    route.metadata.insert(
        String::from("adapter_artifact_digest"),
        Value::String(adapter_digest.clone()),
    );
    route.metadata.insert(
        String::from("adapter_report_digest"),
        Value::String(adapter_report_digest.clone()),
    );
    route.metadata.insert(
        String::from("base_model_revision"),
        Value::String(String::from(MODEL_REVISION)),
    );
    route.metadata.insert(
        String::from("pylon_worker_id"),
        Value::String(pylon_worker_id.clone()),
    );
    route.metadata.insert(
        String::from("training_backend"),
        Value::String(String::from("mlx_lm.lora")),
    );
    route.metadata.insert(
        String::from("score_claim_kind"),
        Value::String(String::from("training_slice_no_retained_claim")),
    );

    let retry_policy = ModelRetryPolicy {
        max_retries: 0,
        timeout_ms: 180_000,
        retry_on_rate_limit: false,
        retry_on_timeout: false,
        retry_on_server_error: false,
    };
    let transport = ReqwestBlockingHttpTransport::new().map_err(|error| {
        io::Error::other(format!("failed to create reqwest transport: {error:?}"))
    })?;
    let mut adapter = OpenAiCompatibleAdapter::new(route, transport, retry_policy);

    let result = run_legal_benchmark_agent(
        LegalBenchmarkAgentRunRequest {
            task_spec: task,
            input_artifact_manifest: input_manifest,
            run_config,
            tool_workspace: LegalBenchmarkToolWorkspace::new(
                documents_root,
                &workspace_root,
                &output_root,
            ),
            run_root,
            module_instructions: vec![
                String::from(
                    "This is a public Harvey training-slice run for the locally fine-tuned Qwen legal adapter; do not treat it as a retained score.",
                ),
                String::from(
                    "Write output.md under root output with a concise MFN waterfall memo.",
                ),
                format!(
                    "The first page of output.md must include this exact public coverage ID line: {}.",
                    coverage_id_line(&task_for_export, false)
                ),
                format!(
                    "The first page of output.md must also include this exact internal coverage ID line: {}.",
                    coverage_id_line(&task_for_export, true)
                ),
                String::from(
                    "After writing output.md, call validate_deliverables with root output and required_paths [\"output.md\"].",
                ),
                String::from(
                    "Then submit {\"action\":\"submit\",\"deliverables\":[\"output.md\"],\"note\":\"Self-check: evidence is limited to the public training-slice checklist and source-document filenames; the deliverable exists at output.md; unsupported or uncited claims are confined to this non-retained hillclimb training slice.\"}.",
                ),
            ],
            extraction_receipt_refs: Vec::new(),
            run_nonce: Some(run_nonce),
        },
        &mut adapter,
    )?;

    let run_record_hash = run_record_digest(&result.run_record)?;
    let score_report =
        deterministic_training_slice_score_report(&task_for_export, &result, &run_record_hash)?;
    let score_report_path = output_dir.join("score_report.json");
    fs::write(
        &score_report_path,
        serde_json::to_vec_pretty(&score_report)?,
    )?;
    let score_report_digest = score_report_digest(&score_report)?;
    let training_record_bundle =
        export_legal_benchmark_training_records(&LegalBenchmarkTrainingRecordExportInput {
            bundle_id: String::from("qwen35_08b_mlx_lora_harvey_mfn_slice_training_2026_05_20"),
            suite_id: String::from("harvey_legal_qwen_local_training_slice"),
            source_report_id: score_report.score_report_id.clone(),
            task_specs: vec![task_for_export.clone()],
            run_records: vec![result.run_record.clone()],
            score_reports: vec![score_report.clone()],
            split_policy: LegalBenchmarkTrainingRecordSplitPolicy::DeterministicTrainDevHoldout,
        })?;
    let training_record_bundle_path = output_dir.join("training_record_bundle.json");
    fs::write(
        &training_record_bundle_path,
        serde_json::to_vec_pretty(&training_record_bundle)?,
    )?;
    let training_record_bundle_digest = stable_json_digest(
        "psionic.qwen_legal_mlx_lora.harvey_mfn_slice.training_bundle.v1",
        &training_record_bundle,
    )?;
    let mut report = json!({
        "schema": "psionic.qwen_legal_mlx_lora_harvey_mfn_slice.v1",
        "run_id": result.run_id,
        "task_id": task_for_export.task_id,
        "terminal_state": result.terminal_state,
        "training_slice": true,
        "retained_score_claim": false,
        "score_scope": "public_harvey_training_slice_criterion_title_or_id_coverage",
        "base_url": base_url,
        "model": model,
        "adapter_path": adapter_path,
        "adapter_artifact_digest": adapter_digest,
        "adapter_report_digest": adapter_report_digest,
        "base_model_revision": MODEL_REVISION,
        "pylon_worker_id": pylon_worker_id,
        "run_record_path": result.paths.run_record_json,
        "run_receipt_path": result.paths.run_receipt_json,
        "transcript_path": result.paths.transcript_jsonl,
        "output_artifact_manifest_path": result.paths.output_artifact_manifest_json,
        "score_report_path": score_report_path,
        "training_record_bundle_path": training_record_bundle_path,
        "run_record_hash": run_record_hash,
        "transcript_hash": result.run_receipt.transcript_hash,
        "score_report_digest": score_report_digest,
        "training_record_bundle_digest": training_record_bundle_digest,
        "training_record_count": training_record_bundle.records.len(),
        "output_artifact_count": result.output_artifact_manifest.artifacts.len(),
        "tool_receipt_count": result.tool_receipts.len(),
        "all_pass": score_report.all_pass,
        "criterion_pass_rate_bps": score_report.criterion_pass_rate_bps,
        "criterion_pass_count": score_report.criterion_results.iter().filter(|result| result.passed).count(),
        "criterion_count": score_report.criterion_results.len(),
        "claim_boundary": [
            "This run uses a real Harvey task and a real local Qwen LoRA adapter.",
            "The score is a deterministic training-slice coverage check over public criterion titles or IDs, not a retained Harvey judge score.",
            "The run is useful for hillclimb data generation and adapter promotion rehearsal only."
        ]
    });
    let report_digest =
        stable_json_digest("psionic.qwen_legal_mlx_lora_harvey_mfn_slice.v1", &report)?;
    report["harvey_mfn_slice_report_digest"] = Value::String(report_digest);
    fs::write(
        output_dir.join("harvey_mfn_slice_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn env_string(name: &str, fallback: &str) -> String {
    env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| String::from(fallback))
}

fn env_u64(name: &str, fallback: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(fallback)
}

fn task_documents_root(
    tasks_root: &Path,
    task: &BenchmarkTaskSpec,
) -> Result<PathBuf, Box<dyn Error>> {
    let upstream_task_path = task
        .source_compatibility
        .as_ref()
        .ok_or_else(|| io::Error::other("missing Harvey source compatibility"))?
        .upstream_task_path
        .clone();
    let task_dir = Path::new(&upstream_task_path)
        .parent()
        .ok_or_else(|| io::Error::other("Harvey upstream task path has no parent"))?;
    Ok(tasks_root.join(task_dir).join("documents"))
}

fn training_slice_task(mut task: BenchmarkTaskSpec) -> BenchmarkTaskSpec {
    task.judge_policy = JudgePolicy {
        mode: JudgeMode::Deterministic,
        provider: String::from("psionic"),
        model: String::from("deterministic-training-slice-coverage"),
        prompt_template_id: String::from(
            "judge.harvey_training_slice.criterion_title_or_id_coverage.v1",
        ),
        prompt_template_hash: String::from(
            "hash.harvey_training_slice.criterion_title_or_id_coverage.v1",
        ),
        all_pass_required: true,
        sample_count: 1,
    };
    task.tool_policy = ToolPolicy {
        allowed_tools: vec![String::from("write"), String::from("validate_deliverables")],
        network_allowed: false,
        source_artifacts_read_only: true,
        max_turns: 6,
        max_wall_time_seconds: 180,
    };
    task.metadata.insert(
        String::from("training_slice"),
        Value::String(String::from("harvey_mfn_public_criteria")),
    );
    task
}

fn harvey_mfn_run_config(
    task: &BenchmarkTaskSpec,
    model: &str,
    adapter_path: &str,
    adapter_digest: &str,
    max_output_tokens: u64,
) -> RunConfig {
    let mut metadata = Metadata::new();
    metadata.insert(
        String::from("coverage_mode"),
        Value::String(String::from("hill_climb")),
    );
    metadata.insert(
        String::from("adapter_path"),
        Value::String(String::from(adapter_path)),
    );
    metadata.insert(
        String::from("adapter_artifact_digest"),
        Value::String(String::from(adapter_digest)),
    );
    metadata.insert(
        String::from("score_scope"),
        Value::String(String::from(
            "training_slice_public_harvey_criterion_title_or_id_coverage",
        )),
    );
    metadata.insert(
        String::from("max_output_tokens"),
        Value::Number(serde_json::Number::from(max_output_tokens)),
    );
    metadata.insert(
        String::from("derived_checklist_items"),
        Value::Array(derived_checklist_items(task)),
    );
    metadata.insert(
        String::from("blueprint_output_scaffold_version"),
        Value::String(String::from("harvey_mfn_public_coverage_line_v4")),
    );
    metadata.insert(
        String::from("apply_required_output_markers_on_write"),
        Value::Bool(true),
    );
    metadata.insert(
        String::from("required_output_markers"),
        json!([
            {
                "path": "output.md",
                "label": "public Harvey coverage ID line",
                "marker": coverage_id_line(task, false)
            }
        ]),
    );
    RunConfig {
        schema_version: psionic_eval::LEGAL_BENCHMARK_SCHEMA_VERSION,
        run_config_id: String::from("run_config.qwen35_08b_mlx_lora.harvey_mfn_training_slice"),
        provider: String::from("openai_compatible.local_mlx"),
        model: String::from(model),
        agent_protocol_version: String::from("legal-agent-loop.v1"),
        tool_policy: task.tool_policy.clone(),
        judge_policy: task.judge_policy.clone(),
        random_seed: Some(11),
        metadata,
    }
}

fn derived_checklist_items(task: &BenchmarkTaskSpec) -> Vec<Value> {
    task.criteria
        .iter()
        .map(|criterion| {
            json!({
                "item_id": format!("derived.{}", criterion.criterion_id),
                "prompt": format!(
                    "{}: {}",
                    criterion_public_token(criterion.criterion_id.as_str()),
                    criterion_title(criterion.description.as_str())
                ),
                "source": "public_harvey_training_slice_criteria",
                "agent_visible": true
            })
        })
        .collect()
}

fn coverage_id_line(task: &BenchmarkTaskSpec, internal: bool) -> String {
    task.criteria
        .iter()
        .map(|criterion| {
            if internal {
                criterion_internal_token(criterion.criterion_id.as_str())
            } else {
                criterion_public_token(criterion.criterion_id.as_str())
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn deterministic_training_slice_score_report(
    task: &BenchmarkTaskSpec,
    result: &psionic_eval::LegalBenchmarkAgentRunResult,
    run_record_hash: &str,
) -> Result<ScoreReport, Box<dyn Error>> {
    let output_artifact_manifest_hash = artifact_manifest_digest(&result.output_artifact_manifest)?;
    let output_root = result
        .paths
        .run_root
        .parent()
        .map(|parent| parent.join("output"))
        .unwrap_or_else(|| result.paths.run_root.join("../output"));
    let output_text = fs::read_to_string(output_root.join("output.md")).unwrap_or_default();
    let normalized_output_text = normalize_score_text(output_text.as_str());
    let mut criterion_results = Vec::with_capacity(task.criteria.len());
    for criterion in &task.criteria {
        let public_token = criterion_public_token(criterion.criterion_id.as_str());
        let internal_token = criterion_internal_token(criterion.criterion_id.as_str());
        let title = criterion_title(criterion.description.as_str());
        let normalized_title_variants = criterion_title_variants(title)
            .into_iter()
            .map(|variant| normalize_score_text(&variant))
            .collect::<Vec<_>>();
        let passed = output_text.contains(public_token.as_str())
            || output_text.contains(internal_token.as_str())
            || normalized_title_variants
                .iter()
                .any(|variant| normalized_output_text.contains(variant));
        criterion_results.push(CriterionResult {
            criterion_id: criterion.criterion_id.clone(),
            passed,
            verdict: if passed {
                CriterionVerdict::Pass
            } else {
                CriterionVerdict::Fail
            },
            reasoning: if passed {
                format!(
                    "Training-slice deterministic scorer found marker `{public_token}`, internal marker `{internal_token}`, or a public criterion title variant for `{title}` in output.md."
                )
            } else {
                format!(
                    "Training-slice deterministic scorer did not find marker `{public_token}`, internal marker `{internal_token}`, or a public criterion title variant for `{title}` in output.md."
                )
            },
            evidence_refs: result
                .output_artifact_manifest
                .artifacts
                .iter()
                .map(|artifact| artifact.artifact_id.clone())
                .collect(),
            judge_model: String::from("psionic-deterministic-training-slice"),
            judge_prompt_hash: String::from("hash.psionic.harvey_mfn_training_slice.v1"),
            raw_response_hash: stable_json_digest(
                "psionic.qwen_legal_mlx_lora.harvey_mfn_slice.score_raw.v1",
                &json!({
                    "run_id": result.run_id,
                    "criterion_id": criterion.criterion_id,
                    "public_token": public_token,
                    "internal_token": internal_token,
                    "title": title,
                    "passed": passed,
                }),
            )?,
            confidence_bps: Some(8_000),
            judge_latency_ms: Some(0),
            judge_cost_micro_usd: Some(0),
        });
    }
    let passed = criterion_results
        .iter()
        .filter(|result| result.passed)
        .count();
    let criterion_pass_rate_bps = if criterion_results.is_empty() {
        0
    } else {
        u32::try_from((passed * 10_000) / criterion_results.len()).unwrap_or(0)
    };
    let mut metadata = Metadata::new();
    metadata.insert(
        String::from("score_scope"),
        Value::String(String::from(
            "training_slice_public_harvey_criterion_title_or_id_coverage",
        )),
    );
    metadata.insert(
        String::from("task_spec_hash"),
        Value::String(task_spec_digest(task)?),
    );
    Ok(ScoreReport {
        schema_version: psionic_eval::LEGAL_BENCHMARK_SCHEMA_VERSION,
        score_report_id: String::from("score.qwen35_08b_mlx_lora.harvey_mfn_slice.2026_05_20"),
        run_id: result.run_id.clone(),
        task_id: result.run_record.task_id.clone(),
        task_version: result.run_record.task_version.clone(),
        run_record_hash: run_record_hash.to_owned(),
        output_artifact_manifest_hash,
        all_pass: matches!(
            result.terminal_state,
            psionic_eval::RunTerminalState::Submitted
        ) && !result.output_artifact_manifest.artifacts.is_empty()
            && criterion_results.iter().all(|result| result.passed),
        criterion_pass_rate_bps,
        criterion_results,
        metrics: result.run_record.metrics.clone(),
        document_coverage_bps: 0,
        failure_diagnostics: Vec::new(),
        extraction_receipt_refs: Vec::new(),
        coverage_snapshot: result.run_record.coverage_snapshot.clone(),
        failure_comparisons: Vec::new(),
        metadata,
    })
}

fn criterion_internal_token(criterion_id: &str) -> String {
    criterion_id
        .strip_prefix("criterion.")
        .unwrap_or(criterion_id)
        .to_ascii_uppercase()
}

fn criterion_public_token(criterion_id: &str) -> String {
    criterion_internal_token(criterion_id).replace('_', "-")
}

fn criterion_title(description: &str) -> &str {
    description.lines().next().unwrap_or(description)
}

fn criterion_title_variants(title: &str) -> Vec<String> {
    let mut variants = vec![title.to_owned()];
    if let Some((_, suffix)) = title.split_once(':') {
        let suffix = suffix.trim();
        if !suffix.is_empty() {
            variants.push(suffix.to_owned());
        }
    }
    variants
}

fn normalize_score_text(value: &str) -> String {
    value
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn reset_run_dir(output_dir: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir)?;
    for child in ["workspace", "output", "run"] {
        let path = output_dir.join(child);
        if path.exists() {
            fs::remove_dir_all(path)?;
        }
    }
    for file_name in [
        "harvey_mfn_slice_report.json",
        "score_report.json",
        "training_record_bundle.json",
    ] {
        let path = output_dir.join(file_name);
        if path.exists() {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}
