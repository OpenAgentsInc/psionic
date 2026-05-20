use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use psionic_eval::{
    BenchmarkTaskSpec, CriterionResult, CriterionVerdict, DeliverableKind, DeliverableSpec,
    JudgeMode, JudgePolicy, LegalBenchmarkAgentRunRequest, LegalBenchmarkToolWorkspace, Metadata,
    ModelProviderRoute, ModelRetryPolicy, OpenAiCompatibleAdapter, ReqwestBlockingHttpTransport,
    RunConfig, RunTerminalState, ScoreReport, ToolPolicy, artifact_manifest_digest,
    build_input_artifact_manifest, run_legal_benchmark_agent, run_record_digest,
    scan_harvey_corpus, score_report_digest, stable_json_digest, task_spec_digest,
};
use serde_json::{Value, json};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:18090/v1";
const DEFAULT_MODEL: &str = "Qwen/Qwen3.5-0.8B";
const DEFAULT_TASKS_ROOT: &str = "/Users/christopherdavid/work/competition/repos/harvey-labs/tasks";
const DEFAULT_UPSTREAM_COMMIT: &str = "5aa41694";
const DEFAULT_OUTPUT_DIR: &str =
    "fixtures/qwen_legal/real_finetune/harvey_no_cheat_suite_2026_05_20_019";
const DEFAULT_TASK_IDS: &str = concat!(
    "harvey.funds-asset-management.analyze_mfn_waterfall,",
    "harvey.corporate-ma.identify_earnout_issues,",
    "harvey.data-privacy-cybersecurity.assess_breach_notification_obligations_across_affected_jurisdictions",
);
const DEFAULT_ADAPTER_PATH: &str = "fixtures/qwen_legal/real_finetune/qwen35_08b_mlx_lora_harvey_mfn_reward_2026_05_20_005/adapters.safetensors";
const DEFAULT_ADAPTER_DIGEST: &str =
    "b509c69b7b26c647dc150bf003bdfef11b9c4714c2ac1767768f6d26857ff9ed";
const DEFAULT_ADAPTER_REPORT_DIGEST: &str =
    "550b599fa222b78d75d03ce30f9e532893de0e450e6753dea6bec294c17229c1";
const MODEL_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SuiteMode {
    ModelOnly,
    BlueprintScaffold,
}

impl SuiteMode {
    fn id(self) -> &'static str {
        match self {
            Self::ModelOnly => "model_only",
            Self::BlueprintScaffold => "blueprint_scaffold",
        }
    }

    fn display(self) -> &'static str {
        match self {
            Self::ModelOnly => "Model-only",
            Self::BlueprintScaffold => "Blueprint scaffold",
        }
    }
}

#[derive(Clone, Debug)]
struct SuiteRunSummary {
    task_id: String,
    title: String,
    mode: SuiteMode,
    terminal_state: RunTerminalState,
    score_report_digest: String,
    run_record_hash: String,
    transcript_hash: String,
    pass_count: usize,
    check_count: usize,
    pass_rate_bps: u32,
    output_artifact_count: usize,
    tool_receipt_count: usize,
    output_md_sha256: Option<String>,
    run_dir: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let output_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_DIR));
    reset_suite_dir(&output_dir)?;

    let tasks_root = PathBuf::from(env_string("HARVEY_TASKS_ROOT", DEFAULT_TASKS_ROOT));
    let upstream_commit = env_string("HARVEY_UPSTREAM_COMMIT", DEFAULT_UPSTREAM_COMMIT);
    let task_ids = env_list("HARVEY_NO_CHEAT_TASK_IDS", DEFAULT_TASK_IDS);
    let base_url = env_string("QWEN_LEGAL_MLX_BASE_URL", DEFAULT_BASE_URL);
    let model = env_string("QWEN_LEGAL_MLX_MODEL", DEFAULT_MODEL);
    let adapter_path = env_string("QWEN_LEGAL_ADAPTER_PATH", DEFAULT_ADAPTER_PATH);
    let adapter_digest = env_string("QWEN_LEGAL_ADAPTER_DIGEST", DEFAULT_ADAPTER_DIGEST);
    let adapter_report_digest = env_string(
        "QWEN_LEGAL_ADAPTER_REPORT_DIGEST",
        DEFAULT_ADAPTER_REPORT_DIGEST,
    );
    let max_output_tokens = env_u64("QWEN_LEGAL_MAX_OUTPUT_TOKENS", 2048);

    let scan = scan_harvey_corpus(&tasks_root, upstream_commit.clone())?;
    let mut summaries = Vec::new();
    for task_id in task_ids {
        let source_task = scan
            .tasks
            .iter()
            .find(|task| task.task_id == task_id)
            .cloned()
            .ok_or_else(|| io::Error::other(format!("missing Harvey task `{task_id}`")))?;
        for mode in [SuiteMode::ModelOnly, SuiteMode::BlueprintScaffold] {
            let summary = run_one_mode(
                &output_dir,
                &tasks_root,
                source_task.clone(),
                mode,
                &base_url,
                &model,
                &adapter_path,
                &adapter_digest,
                &adapter_report_digest,
                max_output_tokens,
            )?;
            summaries.push(summary);
        }
    }

    let report = suite_report(
        &summaries,
        &output_dir,
        &tasks_root,
        &upstream_commit,
        &base_url,
        &model,
        &adapter_path,
        &adapter_digest,
        &adapter_report_digest,
        max_output_tokens,
    )?;
    fs::write(
        output_dir.join("harvey_no_cheat_suite_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_one_mode(
    output_dir: &Path,
    tasks_root: &Path,
    source_task: BenchmarkTaskSpec,
    mode: SuiteMode,
    base_url: &str,
    model: &str,
    adapter_path: &str,
    adapter_digest: &str,
    adapter_report_digest: &str,
    max_output_tokens: u64,
) -> Result<SuiteRunSummary, Box<dyn Error>> {
    let task = suite_task(source_task.clone(), mode);
    let documents_root = task_documents_root(tasks_root, &source_task)?;
    let run_dir = output_dir
        .join(mode.id())
        .join(stable_path_part(source_task.task_id.as_str()));
    reset_run_dir(&run_dir)?;
    let workspace_root = run_dir.join("workspace");
    let output_root = run_dir.join("output");
    let run_root = run_dir.join("run");
    fs::create_dir_all(&workspace_root)?;
    fs::create_dir_all(&output_root)?;

    let input_manifest = build_input_artifact_manifest(&task);
    let run_config = suite_run_config(
        &task,
        mode,
        model,
        adapter_path,
        adapter_digest,
        max_output_tokens,
    );
    let mut route = ModelProviderRoute::openai_compatible(
        format!(
            "route.qwen35_08b_legal_mlx_lora.harvey_no_cheat_suite.{}",
            mode.id()
        ),
        base_url,
        model,
        None,
    );
    route.metadata.insert(
        String::from("adapter_path"),
        Value::String(adapter_path.to_owned()),
    );
    route.metadata.insert(
        String::from("adapter_artifact_digest"),
        Value::String(adapter_digest.to_owned()),
    );
    route.metadata.insert(
        String::from("adapter_report_digest"),
        Value::String(adapter_report_digest.to_owned()),
    );
    route.metadata.insert(
        String::from("base_model_revision"),
        Value::String(String::from(MODEL_REVISION)),
    );
    route.metadata.insert(
        String::from("runner_content_mutation_allowed"),
        Value::Bool(false),
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
            task_spec: task.clone(),
            input_artifact_manifest: input_manifest,
            run_config,
            tool_workspace: LegalBenchmarkToolWorkspace::new(
                documents_root,
                &workspace_root,
                &output_root,
            ),
            run_root,
            module_instructions: module_instructions(&source_task, mode),
            extraction_receipt_refs: Vec::new(),
            run_nonce: Some(format!(
                "qwen35-08b-mlx-lora-harvey-no-cheat-suite-{}-{}",
                mode.id(),
                stable_path_part(source_task.task_id.as_str())
            )),
        },
        &mut adapter,
    )?;

    let run_record_hash = run_record_digest(&result.run_record)?;
    let score_report = suite_score_report(&task, &result, &run_record_hash, mode)?;
    let score_report_path = run_dir.join("score_report.json");
    fs::write(
        &score_report_path,
        serde_json::to_vec_pretty(&score_report)?,
    )?;
    let score_digest = score_report_digest(&score_report)?;
    let output_md_sha256 = result
        .output_artifact_manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.relative_path == "output.md")
        .map(|artifact| artifact.sha256.clone());
    let pass_count = score_report
        .criterion_results
        .iter()
        .filter(|result| result.passed)
        .count();
    Ok(SuiteRunSummary {
        task_id: source_task.task_id,
        title: source_task.title,
        mode,
        terminal_state: result.terminal_state,
        score_report_digest: score_digest,
        run_record_hash,
        transcript_hash: result.run_receipt.transcript_hash,
        pass_count,
        check_count: score_report.criterion_results.len(),
        pass_rate_bps: score_report.criterion_pass_rate_bps,
        output_artifact_count: result.output_artifact_manifest.artifacts.len(),
        tool_receipt_count: result.tool_receipts.len(),
        output_md_sha256,
        run_dir,
    })
}

fn suite_task(mut task: BenchmarkTaskSpec, mode: SuiteMode) -> BenchmarkTaskSpec {
    task.judge_policy = JudgePolicy {
        mode: JudgeMode::Deterministic,
        provider: String::from("psionic"),
        model: String::from("deterministic-no-cheat-suite"),
        prompt_template_id: String::from("judge.harvey.no_cheat_suite.v1"),
        prompt_template_hash: String::from("hash.harvey.no_cheat_suite.v1"),
        all_pass_required: false,
        sample_count: 1,
    };
    task.tool_policy = ToolPolicy {
        allowed_tools: vec![String::from("write"), String::from("validate_deliverables")],
        network_allowed: false,
        source_artifacts_read_only: true,
        max_turns: 6,
        max_wall_time_seconds: 180,
    };
    task.deliverables = vec![DeliverableSpec {
        deliverable_id: String::from("deliverable.output_md"),
        deliverable_kind: DeliverableKind::Markdown,
        required_path: String::from("output.md"),
        description: String::from("Plain Markdown legal work product for local no-cheat scoring"),
        required: true,
    }];
    task.criteria.clear();
    task.metadata.insert(
        String::from("harvey_no_cheat_suite_mode"),
        Value::String(mode.id().to_owned()),
    );
    task.metadata.insert(
        String::from("runner_content_mutation_allowed"),
        Value::Bool(false),
    );
    task
}

fn suite_run_config(
    task: &BenchmarkTaskSpec,
    mode: SuiteMode,
    model: &str,
    adapter_path: &str,
    adapter_digest: &str,
    max_output_tokens: u64,
) -> RunConfig {
    let mut metadata = Metadata::new();
    metadata.insert(
        String::from("coverage_mode"),
        Value::String(String::from("integrity_no_public_rubric")),
    );
    metadata.insert(
        String::from("score_scope"),
        Value::String(String::from("rubric_free_protocol_and_work_product")),
    );
    metadata.insert(String::from("mode"), Value::String(mode.id().to_owned()));
    metadata.insert(
        String::from("adapter_path"),
        Value::String(String::from(adapter_path)),
    );
    metadata.insert(
        String::from("adapter_artifact_digest"),
        Value::String(String::from(adapter_digest)),
    );
    metadata.insert(
        String::from("runner_content_mutation_allowed"),
        Value::Bool(false),
    );
    metadata.insert(
        String::from("max_output_tokens"),
        Value::Number(serde_json::Number::from(max_output_tokens)),
    );
    if mode == SuiteMode::BlueprintScaffold {
        metadata.insert(String::from("plain_text_tool_protocol"), Value::Bool(true));
        metadata.insert(
            String::from("force_write_until_required_deliverables"),
            Value::Bool(true),
        );
        metadata.insert(
            String::from("force_validate_after_write"),
            Value::Bool(true),
        );
    }
    RunConfig {
        schema_version: psionic_eval::LEGAL_BENCHMARK_SCHEMA_VERSION,
        run_config_id: format!(
            "run_config.qwen35_08b_mlx_lora.harvey_no_cheat_suite.{}",
            mode.id()
        ),
        provider: String::from("openai_compatible.local_mlx"),
        model: String::from(model),
        agent_protocol_version: String::from("legal-agent-loop.v1.no-cheat-suite"),
        tool_policy: task.tool_policy.clone(),
        judge_policy: task.judge_policy.clone(),
        random_seed: Some(19),
        metadata,
    }
}

fn module_instructions(task: &BenchmarkTaskSpec, mode: SuiteMode) -> Vec<String> {
    let mut instructions = vec![
        String::from(
            "This is a public Harvey training-slice run. It is not a retained benchmark score.",
        ),
        String::from(
            "The runner must not and will not add text to your output. Any score comes only from what the model writes.",
        ),
        String::from("Write output.md under root output, validate output.md, then submit it."),
        String::from(
            "Do not include Harvey criterion IDs, rubric labels, C-IDs, checklist IDs, or scoring tokens.",
        ),
        String::from(
            "If you did not actually inspect a source document, say that plainly as a source limit instead of pretending.",
        ),
    ];
    if mode == SuiteMode::BlueprintScaffold {
        instructions.push(String::from(
            "Use plain JSON tool messages exactly like {\"tool\":\"write\",\"input\":{...}} and {\"tool\":\"validate_deliverables\",\"input\":{...}}; the runner will execute only JSON you wrote.",
        ));
        instructions.push(generic_blueprint_work_product_scaffold(task));
    }
    instructions
}

fn generic_blueprint_work_product_scaffold(task: &BenchmarkTaskSpec) -> String {
    let source_names = task
        .source_artifacts
        .iter()
        .take(12)
        .map(|artifact| artifact.original_filename.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let work_type_instruction = match task.work_type.as_str() {
        "analyze" => {
            "Produce an issue map, explain the governing legal/business standard, compare source facts to that standard, and end with recommendations and open questions."
        }
        "review" => {
            "Produce a review memo with material issues, risk severity, proposed edits or follow-up questions, and source limits."
        }
        "draft" => {
            "Produce a drafting plan or draft text with assumptions, clause structure, unresolved business points, and source limits."
        }
        "research" => {
            "Produce a research memo with question presented, rule summary, application, authorities or missing authorities, and next steps."
        }
        _ => {
            "Produce a concise legal work product with facts, analysis, recommendation, and source limits."
        }
    };
    format!(
        "Blueprint work-product scaffold: task `{}` is {} work in {}. Do not copy rubric text. Build the memo around product requirements: source map; material facts from visible source names; legal/business issues; analysis; recommendation; risks; assumptions; and source limits. {} Visible source names include: {}.",
        task.title, task.work_type, task.practice_area, work_type_instruction, source_names
    )
}

fn suite_score_report(
    task: &BenchmarkTaskSpec,
    result: &psionic_eval::LegalBenchmarkAgentRunResult,
    run_record_hash: &str,
    mode: SuiteMode,
) -> Result<ScoreReport, Box<dyn Error>> {
    let output_artifact_manifest_hash = artifact_manifest_digest(&result.output_artifact_manifest)?;
    let output_root = result
        .paths
        .run_root
        .parent()
        .map(|parent| parent.join("output"))
        .unwrap_or_else(|| result.paths.run_root.join("../output"));
    let output_text = fs::read_to_string(output_root.join("output.md")).unwrap_or_default();
    let normalized = normalize_score_text(output_text.as_str());
    let tool_names = result
        .tool_receipts
        .iter()
        .map(|receipt| receipt.tool_name.as_str())
        .collect::<Vec<_>>();
    let checks = [
        (
            "protocol.submitted",
            matches!(result.terminal_state, RunTerminalState::Submitted),
            "run reached submitted terminal state",
        ),
        (
            "protocol.output_artifact",
            !result.output_artifact_manifest.artifacts.is_empty(),
            "run produced an output artifact",
        ),
        (
            "protocol.write_tool",
            tool_names.iter().any(|tool| *tool == "write"),
            "model called write",
        ),
        (
            "protocol.validate_tool",
            tool_names
                .iter()
                .any(|tool| *tool == "validate_deliverables"),
            "model called validate_deliverables",
        ),
        (
            "integrity.no_public_criterion_ids",
            !contains_criterion_id_text(output_text.as_str()),
            "output does not contain public Harvey criterion IDs",
        ),
        (
            "work_product.task_topic",
            topic_terms(task)
                .iter()
                .any(|term| normalized.contains(normalize_score_text(term).as_str())),
            "output mentions the task topic",
        ),
        (
            "work_product.source_limits",
            [
                "source limit",
                "assumption",
                "provided source",
                "did not inspect",
                "based on",
            ]
            .iter()
            .any(|term| normalized.contains(normalize_score_text(term).as_str())),
            "output states source limits or assumptions",
        ),
        (
            "work_product.legal_action",
            legal_action_terms(task)
                .iter()
                .any(|term| normalized.contains(normalize_score_text(term).as_str())),
            "output contains legal analysis or action language",
        ),
        (
            "work_product.minimum_substance",
            output_text.split_whitespace().count() >= 120,
            "output has enough substance for a first-pass memo",
        ),
    ];
    let evidence_refs = result
        .output_artifact_manifest
        .artifacts
        .iter()
        .map(|artifact| artifact.artifact_id.clone())
        .collect::<Vec<_>>();
    let mut criterion_results = Vec::with_capacity(checks.len());
    for (check_id, passed, description) in checks {
        criterion_results.push(CriterionResult {
            criterion_id: String::from(check_id),
            passed,
            verdict: if passed {
                CriterionVerdict::Pass
            } else {
                CriterionVerdict::Fail
            },
            reasoning: if passed {
                format!("Passed: {description}.")
            } else {
                format!("Missing or failed: {description}.")
            },
            evidence_refs: evidence_refs.clone(),
            judge_model: String::from("psionic-deterministic-no-cheat-suite"),
            judge_prompt_hash: String::from("hash.psionic.harvey.no_cheat_suite.v1"),
            raw_response_hash: stable_json_digest(
                "psionic.harvey.no_cheat_suite.score_check.v1",
                &json!({
                    "run_id": result.run_id,
                    "task_id": result.run_record.task_id,
                    "mode": mode.id(),
                    "check_id": check_id,
                    "passed": passed,
                }),
            )?,
            confidence_bps: Some(7_500),
            judge_latency_ms: Some(0),
            judge_cost_micro_usd: Some(0),
        });
    }
    let pass_count = criterion_results
        .iter()
        .filter(|result| result.passed)
        .count();
    let criterion_pass_rate_bps =
        u32::try_from((pass_count * 10_000) / criterion_results.len()).unwrap_or(0);
    let mut metadata = Metadata::new();
    metadata.insert(
        String::from("score_scope"),
        Value::String(String::from("rubric_free_protocol_and_work_product")),
    );
    metadata.insert(String::from("mode"), Value::String(mode.id().to_owned()));
    metadata.insert(
        String::from("runner_content_mutation_allowed"),
        Value::Bool(false),
    );
    metadata.insert(
        String::from("task_spec_hash"),
        Value::String(task_spec_digest(task)?),
    );
    Ok(ScoreReport {
        schema_version: psionic_eval::LEGAL_BENCHMARK_SCHEMA_VERSION,
        score_report_id: format!(
            "score.qwen35_08b_mlx_lora.harvey_no_cheat_suite.{}.2026_05_20",
            mode.id()
        ),
        run_id: result.run_id.clone(),
        task_id: result.run_record.task_id.clone(),
        task_version: result.run_record.task_version.clone(),
        run_record_hash: run_record_hash.to_owned(),
        output_artifact_manifest_hash,
        all_pass: criterion_results.iter().all(|result| result.passed),
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

fn suite_report(
    summaries: &[SuiteRunSummary],
    output_dir: &Path,
    tasks_root: &Path,
    upstream_commit: &str,
    base_url: &str,
    model: &str,
    adapter_path: &str,
    adapter_digest: &str,
    adapter_report_digest: &str,
    max_output_tokens: u64,
) -> Result<Value, Box<dyn Error>> {
    let by_task = summaries
        .iter()
        .map(|summary| {
            json!({
                "task_id": summary.task_id,
                "title": summary.title,
                "mode": summary.mode.id(),
                "mode_label": summary.mode.display(),
                "terminal_state": summary.terminal_state,
                "score": {
                    "pass_count": summary.pass_count,
                    "check_count": summary.check_count,
                    "pass_rate_bps": summary.pass_rate_bps
                },
                "output_artifact_count": summary.output_artifact_count,
                "tool_receipt_count": summary.tool_receipt_count,
                "output_md_sha256": summary.output_md_sha256,
                "score_report_digest": summary.score_report_digest,
                "run_record_hash": summary.run_record_hash,
                "transcript_hash": summary.transcript_hash,
                "run_dir": summary.run_dir,
            })
        })
        .collect::<Vec<_>>();
    let model_only_avg = average_bps(summaries, SuiteMode::ModelOnly);
    let scaffold_avg = average_bps(summaries, SuiteMode::BlueprintScaffold);
    let suite_id = output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("harvey_no_cheat_suite");
    let mut report = json!({
        "schema": "psionic.harvey_no_cheat_suite.v1",
        "suite_id": suite_id,
        "tasks_root": tasks_root,
        "upstream_commit": upstream_commit,
        "training_slice": true,
        "retained_score_claim": false,
        "runner_content_mutation_allowed": false,
        "base_url": base_url,
        "model": model,
        "adapter_path": adapter_path,
        "adapter_artifact_digest": adapter_digest,
        "adapter_report_digest": adapter_report_digest,
        "base_model_revision": MODEL_REVISION,
        "max_output_tokens": max_output_tokens,
        "mode_average_pass_rate_bps": {
            "model_only": model_only_avg,
            "blueprint_scaffold": scaffold_avg,
            "delta_scaffold_minus_model_only": i64::from(scaffold_avg) - i64::from(model_only_avg)
        },
        "runs": by_task,
        "claim_boundary": [
            "These are local public Harvey training-slice runs through the Rust legal agent loop.",
            "No runner path adds or rewrites model output.",
            "Scaffold-assisted means prompt/module requirements only; it does not inject answer text.",
            "Scores are local protocol and work-product checks, not official Harvey retained scores."
        ]
    });
    let digest = stable_json_digest("psionic.harvey_no_cheat_suite.v1", &report)?;
    report["suite_report_digest"] = Value::String(digest);
    Ok(report)
}

fn average_bps(summaries: &[SuiteRunSummary], mode: SuiteMode) -> u32 {
    let matching = summaries
        .iter()
        .filter(|summary| summary.mode == mode)
        .collect::<Vec<_>>();
    if matching.is_empty() {
        return 0;
    }
    let total = matching
        .iter()
        .map(|summary| usize::try_from(summary.pass_rate_bps).unwrap_or(0))
        .sum::<usize>();
    u32::try_from(total / matching.len()).unwrap_or(0)
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

fn topic_terms(task: &BenchmarkTaskSpec) -> Vec<String> {
    task.title
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|term| term.len() > 4)
        .take(8)
        .map(ToOwned::to_owned)
        .collect()
}

fn legal_action_terms(task: &BenchmarkTaskSpec) -> Vec<&'static str> {
    match task.work_type.as_str() {
        "draft" => vec!["draft", "clause", "provision", "revise", "language"],
        "review" => vec!["issue", "risk", "revise", "comment", "redline"],
        "research" => vec![
            "rule",
            "authority",
            "jurisdiction",
            "analysis",
            "conclusion",
        ],
        _ => vec!["analyze", "recommend", "risk", "issue", "conclusion"],
    }
}

fn contains_criterion_id_text(output_text: &str) -> bool {
    (1..=999).any(|index| {
        let hyphen = format!("C-{index:03}");
        let underscore = format!("C_{index:03}");
        output_text.contains(hyphen.as_str()) || output_text.contains(underscore.as_str())
    })
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

fn stable_path_part(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
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

fn env_list(name: &str, fallback: &str) -> Vec<String> {
    env_string(name, fallback)
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn reset_suite_dir(output_dir: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir)?;
    for child in ["model_only", "blueprint_scaffold"] {
        let path = output_dir.join(child);
        if path.exists() {
            fs::remove_dir_all(path)?;
        }
    }
    let report = output_dir.join("harvey_no_cheat_suite_report.json");
    if report.exists() {
        fs::remove_file(report)?;
    }
    Ok(())
}

fn reset_run_dir(run_dir: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(run_dir)?;
    for child in ["workspace", "output", "run"] {
        let path = run_dir.join(child);
        if path.exists() {
            fs::remove_dir_all(path)?;
        }
    }
    let report = run_dir.join("score_report.json");
    if report.exists() {
        fs::remove_file(report)?;
    }
    Ok(())
}
