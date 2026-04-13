use std::{
    env, fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    Cs336A1ReferenceTrainer, Cs336A1ReferenceTrainingStepReport,
    PSION_CS336_A1_DEMO_CHECKPOINT_LABEL, PSION_CS336_A1_DEMO_CLAIM_BOUNDARY,
    PSION_CS336_A1_DEMO_CLOSEOUT_BUNDLE_SCHEMA_VERSION,
    PSION_CS336_A1_DEMO_CURRENT_RUN_STATUS_SCHEMA_VERSION, PSION_CS336_A1_DEMO_LANE_ID,
    PSION_CS336_A1_DEMO_RETAINED_SUMMARY_SCHEMA_VERSION, PSION_CS336_A1_DEMO_TRAINING_STEP_COUNT,
    PSIONIC_TRAIN_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    PSIONIC_TRAIN_CHECKPOINT_POINTER_SCHEMA_VERSION, PsionCs336A1DemoAutomaticExecutionRequest,
    PsionCs336A1DemoCurrentRunStatus, PsionCs336A1DemoLaunchManifest,
    PsionCs336A1DemoRetainedSummary, PsionicTrainCheckpointManifest, PsionicTrainCheckpointPointer,
    PsionicTrainInvocationManifest, PsionicTrainOperation, PsionicTrainRole,
    build_psion_cs336_a1_demo_launch_manifest, inspect_psionic_train_checkpoint_surface,
    load_cs336_a1_reference_checkpoint, psion_cs336_a1_demo_retained_paths,
};

#[derive(Debug, Error)]
pub enum PsionCs336A1DemoOperatorError {
    #[error("the bounded A1 demo lane expected `{expected}` but found `{actual}`")]
    LaneMismatch { expected: String, actual: String },
    #[error("the bounded A1 demo lane does not support operation `{0}` yet")]
    UnsupportedOperation(String),
    #[error("missing required field `{0}`")]
    MissingField(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Training(#[from] crate::Cs336A1ReferenceTrainingError),
    #[error(transparent)]
    Launcher(#[from] crate::PsionCs336A1DemoLauncherError),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsionCs336A1DemoCloseoutBundle {
    pub schema_version: String,
    pub lane_id: String,
    pub run_id: String,
    pub outcome: String,
    pub claim_boundary: String,
    pub corpus_fixture_path: String,
    pub training_step_count: u64,
    pub launch_manifest: PsionCs336A1DemoLaunchManifest,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub checkpoint_label: String,
    pub checkpoint_ref: String,
    pub checkpoint_digest: String,
    pub model_state_digest: String,
    pub optimizer_state_digest: String,
    pub step_reports: Vec<Cs336A1ReferenceTrainingStepReport>,
}

pub fn run_psion_cs336_a1_demo_cli(args: Vec<String>) -> Result<(), PsionCs336A1DemoOperatorError> {
    if args.is_empty() || matches!(args[0].as_str(), "--help" | "-h" | "help") {
        print_usage();
        return Ok(());
    }

    match args[0].as_str() {
        "start" | "rehearse-base-lane" => {
            let parsed = parse_start_like_args(args[0].as_str(), &args[1..])?;
            let request = PsionCs336A1DemoAutomaticExecutionRequest {
                schema_version: String::from(
                    crate::PSION_CS336_A1_DEMO_AUTOMATIC_EXECUTION_REQUEST_SCHEMA_VERSION,
                ),
                role: PsionicTrainRole::Worker,
                operation: parsed.operation,
                coordination: crate::PsionicTrainCoordinationContext {
                    node_pubkey: Some(String::from("npub1-local-cs336-a1-demo")),
                    ..Default::default()
                },
                build_digest: String::from("local-rehearsal"),
                run_id: parsed.run_id,
                output_root: Some(parsed.output_root),
                run_root: None,
                selected_git_ref: parsed.selected_git_ref,
                allow_dirty_tree: parsed.allow_dirty_tree,
                dry_run: parsed.dry_run,
            };
            let manifest = request.to_invocation_manifest()?;
            run_psion_cs336_a1_demo_manifest(&manifest)
        }
        "status" => {
            let run_root = parse_status_args(&args[1..])?;
            print_status(Path::new(run_root.as_str()))
        }
        other => Err(PsionCs336A1DemoOperatorError::UnsupportedOperation(
            other.to_string(),
        )),
    }
}

pub fn run_psion_cs336_a1_demo_manifest(
    manifest: &PsionicTrainInvocationManifest,
) -> Result<(), PsionCs336A1DemoOperatorError> {
    if manifest.lane_id != PSION_CS336_A1_DEMO_LANE_ID {
        return Err(PsionCs336A1DemoOperatorError::LaneMismatch {
            expected: String::from(PSION_CS336_A1_DEMO_LANE_ID),
            actual: manifest.lane_id.clone(),
        });
    }
    if manifest.role != PsionicTrainRole::Worker {
        return Err(PsionCs336A1DemoOperatorError::UnsupportedOperation(
            String::from("only worker role is packaged for the bounded A1 lane"),
        ));
    }

    match manifest.operation {
        PsionicTrainOperation::Start => run_start_like(manifest, false),
        PsionicTrainOperation::RehearseBaseLane => run_start_like(manifest, true),
        other => Err(PsionCs336A1DemoOperatorError::UnsupportedOperation(
            other.cli_subcommand().to_string(),
        )),
    }
}

fn run_start_like(
    manifest: &PsionicTrainInvocationManifest,
    rehearsal: bool,
) -> Result<(), PsionCs336A1DemoOperatorError> {
    let run_root =
        PathBuf::from(manifest.output_root.as_deref().ok_or_else(|| {
            PsionCs336A1DemoOperatorError::MissingField(String::from("output_root"))
        })?);
    let retained_paths = psion_cs336_a1_demo_retained_paths();
    let launch_manifest = build_psion_cs336_a1_demo_launch_manifest(
        manifest
            .run_id
            .clone()
            .unwrap_or_else(|| String::from("psion-cs336-a1-demo")),
        manifest
            .selected_git_ref
            .clone()
            .unwrap_or_else(|| String::from("HEAD")),
        rehearsal,
    );

    create_run_dirs(&run_root)?;
    write_json_pretty(
        run_root
            .join(&retained_paths.launch_manifest_path)
            .as_path(),
        &launch_manifest,
    )?;
    append_log(
        run_root.join(&retained_paths.launcher_log_path).as_path(),
        format!(
            "phase=launch_manifest_written run_id={} surface_id={}",
            launch_manifest.run_id, launch_manifest.surface_id
        )
        .as_str(),
    )?;

    let mut current_status = PsionCs336A1DemoCurrentRunStatus {
        schema_version: String::from(PSION_CS336_A1_DEMO_CURRENT_RUN_STATUS_SCHEMA_VERSION),
        lane_id: String::from(PSION_CS336_A1_DEMO_LANE_ID),
        run_id: launch_manifest.run_id.clone(),
        phase: if manifest.dry_run {
            String::from("dry_run_materialized")
        } else {
            String::from("launching")
        },
        completed_steps: 0,
        total_steps: PSION_CS336_A1_DEMO_TRAINING_STEP_COUNT,
        latest_loss: None,
        checkpoint_ref: None,
        detail: String::from(
            "Bounded CS336 A1 demo lane is materializing retained launcher surfaces.",
        ),
    };
    write_json_pretty(
        run_root.join(&retained_paths.current_status_path).as_path(),
        &current_status,
    )?;

    if manifest.dry_run {
        let summary = PsionCs336A1DemoRetainedSummary {
            schema_version: String::from(PSION_CS336_A1_DEMO_RETAINED_SUMMARY_SCHEMA_VERSION),
            lane_id: String::from(PSION_CS336_A1_DEMO_LANE_ID),
            run_id: launch_manifest.run_id.clone(),
            claim_boundary: String::from(PSION_CS336_A1_DEMO_CLAIM_BOUNDARY),
            corpus_fixture_path: launch_manifest.corpus_fixture_path.clone(),
            training_config: launch_manifest.training_config.clone(),
            total_steps: PSION_CS336_A1_DEMO_TRAINING_STEP_COUNT,
            initial_loss: None,
            final_loss: None,
            latest_checkpoint_label: None,
            latest_checkpoint_ref: None,
            latest_checkpoint_path: None,
            model_state_digest: None,
            optimizer_state_digest: None,
            checkpoint_digest: None,
            detail: String::from(
                "Dry-run rehearsal materialized the packaged A1 lane without executing training.",
            ),
        };
        write_json_pretty(
            run_root
                .join(&retained_paths.retained_summary_path)
                .as_path(),
            &summary,
        )?;
        append_log(
            run_root.join(&retained_paths.launcher_log_path).as_path(),
            "phase=dry_run_complete detail=retained_launcher_surfaces_only",
        )?;
        return Ok(());
    }

    let corpus_path = repo_root().join(launch_manifest.corpus_fixture_path.as_str());
    let mut trainer = Cs336A1ReferenceTrainer::from_corpus_path(
        &corpus_path,
        launch_manifest.training_config.clone(),
    )?;
    let initial_loss = trainer.current_loss()?;
    append_log(
        run_root.join(&retained_paths.launcher_log_path).as_path(),
        format!(
            "phase=training_started run_id={} initial_loss={initial_loss:.6}",
            launch_manifest.run_id
        )
        .as_str(),
    )?;

    let mut step_reports = Vec::with_capacity(PSION_CS336_A1_DEMO_TRAINING_STEP_COUNT as usize);
    for _ in 0..PSION_CS336_A1_DEMO_TRAINING_STEP_COUNT {
        let report = trainer.step()?;
        current_status.phase = String::from("training");
        current_status.completed_steps = report.step_number;
        current_status.latest_loss = Some(report.loss_after);
        current_status.detail = format!(
            "Bounded CS336 A1 demo lane completed step {} of {}.",
            report.step_number, PSION_CS336_A1_DEMO_TRAINING_STEP_COUNT
        );
        write_json_pretty(
            run_root.join(&retained_paths.current_status_path).as_path(),
            &current_status,
        )?;
        append_log(
            run_root.join(&retained_paths.launcher_log_path).as_path(),
            format!(
                "phase=step_complete step={} loss_before={:.6} loss_after={:.6}",
                report.step_number, report.loss_before, report.loss_after
            )
            .as_str(),
        )?;
        step_reports.push(report);
    }

    let checkpoint_path = run_root.join(&retained_paths.checkpoint_payload_path);
    let checkpoint_receipt = trainer.save_checkpoint(&checkpoint_path)?;
    let checkpoint_total_bytes = fs::metadata(&checkpoint_path)?.len();
    let checkpoint_ref = format!(
        "checkpoint://psion/cs336_a1_demo/{}/step-{:06}",
        launch_manifest.run_id, PSION_CS336_A1_DEMO_TRAINING_STEP_COUNT
    );
    let mut checkpoint_manifest = PsionicTrainCheckpointManifest {
        schema_version: String::from(PSIONIC_TRAIN_CHECKPOINT_MANIFEST_SCHEMA_VERSION),
        lane_id: String::from(PSION_CS336_A1_DEMO_LANE_ID),
        run_id: launch_manifest.run_id.clone(),
        window_id: manifest.coordination.window_id.clone(),
        assignment_id: manifest.coordination.assignment_id.clone(),
        grouped_stage_assignment: manifest.grouped_stage_assignment.clone(),
        checkpoint_label: String::from(PSION_CS336_A1_DEMO_CHECKPOINT_LABEL),
        optimizer_step: PSION_CS336_A1_DEMO_TRAINING_STEP_COUNT,
        checkpoint_ref: checkpoint_ref.clone(),
        relative_manifest_path: retained_paths.checkpoint_manifest_path.clone(),
        checkpoint_object_digest: checkpoint_receipt.checkpoint_digest.clone(),
        checkpoint_total_bytes,
        manifest_digest: String::new(),
    };
    checkpoint_manifest.manifest_digest = checkpoint_manifest.stable_manifest_digest();

    let checkpoint_pointer = PsionicTrainCheckpointPointer {
        schema_version: String::from(PSIONIC_TRAIN_CHECKPOINT_POINTER_SCHEMA_VERSION),
        lane_id: String::from(PSION_CS336_A1_DEMO_LANE_ID),
        run_id: launch_manifest.run_id.clone(),
        window_id: manifest.coordination.window_id.clone(),
        assignment_id: manifest.coordination.assignment_id.clone(),
        grouped_stage_assignment: manifest.grouped_stage_assignment.clone(),
        pointer_state: String::from("accepted"),
        checkpoint_label: String::from(PSION_CS336_A1_DEMO_CHECKPOINT_LABEL),
        optimizer_step: PSION_CS336_A1_DEMO_TRAINING_STEP_COUNT,
        checkpoint_ref: checkpoint_ref.clone(),
        checkpoint_manifest_relative_path: retained_paths.checkpoint_manifest_path.clone(),
        detail: String::from(
            "Bounded CS336 A1 demo lane retained one accepted generic checkpoint for Pylon and Nexus closeout ingestion.",
        ),
    };
    write_json_pretty(
        run_root
            .join(&retained_paths.checkpoint_manifest_path)
            .as_path(),
        &checkpoint_manifest,
    )?;
    write_json_pretty(
        run_root
            .join(&retained_paths.checkpoint_pointer_path)
            .as_path(),
        &checkpoint_pointer,
    )?;

    let final_loss = step_reports
        .last()
        .map(|value| value.loss_after)
        .unwrap_or(initial_loss);
    let checkpoint = load_cs336_a1_reference_checkpoint(&checkpoint_path)?;
    let summary = PsionCs336A1DemoRetainedSummary {
        schema_version: String::from(PSION_CS336_A1_DEMO_RETAINED_SUMMARY_SCHEMA_VERSION),
        lane_id: String::from(PSION_CS336_A1_DEMO_LANE_ID),
        run_id: launch_manifest.run_id.clone(),
        claim_boundary: String::from(PSION_CS336_A1_DEMO_CLAIM_BOUNDARY),
        corpus_fixture_path: launch_manifest.corpus_fixture_path.clone(),
        training_config: checkpoint.config.clone(),
        total_steps: PSION_CS336_A1_DEMO_TRAINING_STEP_COUNT,
        initial_loss: Some(initial_loss),
        final_loss: Some(final_loss),
        latest_checkpoint_label: Some(String::from(PSION_CS336_A1_DEMO_CHECKPOINT_LABEL)),
        latest_checkpoint_ref: Some(checkpoint_ref.clone()),
        latest_checkpoint_path: Some(checkpoint_path.display().to_string()),
        model_state_digest: Some(checkpoint_receipt.model_state_digest.clone()),
        optimizer_state_digest: Some(checkpoint_receipt.optimizer_state_digest.clone()),
        checkpoint_digest: Some(checkpoint_receipt.checkpoint_digest.clone()),
        detail: if rehearsal {
            String::from(
                "Rehearsal completed the packaged bounded A1 lane and retained one accepted checkpoint and closeout bundle.",
            )
        } else {
            String::from(
                "Packaged bounded A1 lane completed and retained one accepted checkpoint and closeout bundle for shared Pylon ingestion.",
            )
        },
    };
    write_json_pretty(
        run_root
            .join(&retained_paths.retained_summary_path)
            .as_path(),
        &summary,
    )?;

    let closeout = PsionCs336A1DemoCloseoutBundle {
        schema_version: String::from(PSION_CS336_A1_DEMO_CLOSEOUT_BUNDLE_SCHEMA_VERSION),
        lane_id: String::from(PSION_CS336_A1_DEMO_LANE_ID),
        run_id: launch_manifest.run_id.clone(),
        outcome: String::from("accepted"),
        claim_boundary: String::from(PSION_CS336_A1_DEMO_CLAIM_BOUNDARY),
        corpus_fixture_path: launch_manifest.corpus_fixture_path.clone(),
        training_step_count: PSION_CS336_A1_DEMO_TRAINING_STEP_COUNT,
        launch_manifest,
        initial_loss,
        final_loss,
        checkpoint_label: String::from(PSION_CS336_A1_DEMO_CHECKPOINT_LABEL),
        checkpoint_ref: checkpoint_ref.clone(),
        checkpoint_digest: checkpoint_receipt.checkpoint_digest.clone(),
        model_state_digest: checkpoint_receipt.model_state_digest.clone(),
        optimizer_state_digest: checkpoint_receipt.optimizer_state_digest.clone(),
        step_reports,
    };
    write_json_pretty(
        run_root
            .join(&retained_paths.closeout_bundle_path)
            .as_path(),
        &closeout,
    )?;

    current_status.phase = if rehearsal {
        String::from("rehearsed")
    } else {
        String::from("completed")
    };
    current_status.completed_steps = PSION_CS336_A1_DEMO_TRAINING_STEP_COUNT;
    current_status.latest_loss = Some(final_loss);
    current_status.checkpoint_ref = Some(checkpoint_ref);
    current_status.detail = String::from(
        "Bounded CS336 A1 demo lane completed with one accepted checkpoint and one closeout bundle.",
    );
    write_json_pretty(
        run_root.join(&retained_paths.current_status_path).as_path(),
        &current_status,
    )?;
    append_log(
        run_root.join(&retained_paths.launcher_log_path).as_path(),
        format!(
            "phase=complete run_id={} final_loss={final_loss:.6} checkpoint_digest={}",
            summary.run_id,
            summary
                .checkpoint_digest
                .as_deref()
                .unwrap_or("unknown_checkpoint_digest")
        )
        .as_str(),
    )?;
    Ok(())
}

fn print_status(run_root: &Path) -> Result<(), PsionCs336A1DemoOperatorError> {
    let retained_paths = psion_cs336_a1_demo_retained_paths();
    let current_status: PsionCs336A1DemoCurrentRunStatus =
        load_json_file(run_root.join(retained_paths.current_status_path).as_path())?;
    let summary: PsionCs336A1DemoRetainedSummary = load_json_file(
        run_root
            .join(retained_paths.retained_summary_path)
            .as_path(),
    )?;
    let closeout_path = run_root.join(retained_paths.closeout_bundle_path);
    let closeout_exists = closeout_path.is_file();
    let checkpoint_surface = inspect_psionic_train_checkpoint_surface(
        run_root,
        PsionicTrainRole::Worker,
        PsionicTrainOperation::Start,
    )
    .ok()
    .flatten();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "run_root": run_root.display().to_string(),
            "current_status": current_status,
            "retained_summary": summary,
            "closeout_bundle_path": closeout_exists.then(|| closeout_path.display().to_string()),
            "checkpoint_surface": checkpoint_surface,
        }))?
    );
    Ok(())
}

#[derive(Clone, Debug)]
struct ParsedStartLikeArgs {
    operation: PsionicTrainOperation,
    run_id: String,
    output_root: String,
    selected_git_ref: String,
    allow_dirty_tree: bool,
    dry_run: bool,
}

fn parse_start_like_args(
    command: &str,
    args: &[String],
) -> Result<ParsedStartLikeArgs, PsionCs336A1DemoOperatorError> {
    let mut run_id = None;
    let mut output_root = None;
    let mut selected_git_ref = Some(String::from("HEAD"));
    let mut allow_dirty_tree = false;
    let mut dry_run = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--run-id" => {
                index += 1;
                run_id = Some(args.get(index).cloned().ok_or_else(|| {
                    PsionCs336A1DemoOperatorError::MissingField(String::from("run-id"))
                })?);
            }
            "--output-root" => {
                index += 1;
                output_root = Some(args.get(index).cloned().ok_or_else(|| {
                    PsionCs336A1DemoOperatorError::MissingField(String::from("output-root"))
                })?);
            }
            "--git-ref" => {
                index += 1;
                selected_git_ref = Some(args.get(index).cloned().ok_or_else(|| {
                    PsionCs336A1DemoOperatorError::MissingField(String::from("git-ref"))
                })?);
            }
            "--allow-dirty-tree" => allow_dirty_tree = true,
            "--dry-run" => dry_run = true,
            other => {
                return Err(PsionCs336A1DemoOperatorError::UnsupportedOperation(
                    format!("unknown argument `{other}`"),
                ));
            }
        }
        index += 1;
    }

    let run_id = run_id.unwrap_or_else(default_run_id);
    let output_root = output_root.unwrap_or_else(|| default_output_root(run_id.as_str()));
    Ok(ParsedStartLikeArgs {
        operation: if command == "rehearse-base-lane" {
            PsionicTrainOperation::RehearseBaseLane
        } else {
            PsionicTrainOperation::Start
        },
        run_id,
        output_root,
        selected_git_ref: selected_git_ref.unwrap_or_else(|| String::from("HEAD")),
        allow_dirty_tree,
        dry_run,
    })
}

fn parse_status_args(args: &[String]) -> Result<String, PsionCs336A1DemoOperatorError> {
    let mut run_root = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--run-root" => {
                index += 1;
                run_root = Some(args.get(index).cloned().ok_or_else(|| {
                    PsionCs336A1DemoOperatorError::MissingField(String::from("run-root"))
                })?);
            }
            other => {
                return Err(PsionCs336A1DemoOperatorError::UnsupportedOperation(
                    format!("unknown argument `{other}`"),
                ));
            }
        }
        index += 1;
    }
    run_root.ok_or_else(|| PsionCs336A1DemoOperatorError::MissingField(String::from("run-root")))
}

fn create_run_dirs(run_root: &Path) -> Result<(), std::io::Error> {
    for relative in [
        "manifests",
        "status",
        "checkpoints/manifests",
        "checkpoints/step-000004",
        "closeout",
        "logs",
    ] {
        fs::create_dir_all(run_root.join(relative))?;
    }
    Ok(())
}

fn write_json_pretty(
    path: &Path,
    value: &impl Serialize,
) -> Result<(), PsionCs336A1DemoOperatorError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn load_json_file<T: serde::de::DeserializeOwned>(
    path: &Path,
) -> Result<T, PsionCs336A1DemoOperatorError> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn append_log(path: &Path, line: &str) -> Result<(), PsionCs336A1DemoOperatorError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut message = if path.is_file() {
        fs::read_to_string(path)?
    } else {
        String::new()
    };
    message.push_str(line);
    message.push('\n');
    fs::write(path, message)?;
    Ok(())
}

fn default_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis())
        .unwrap_or(0);
    format!("psion-cs336-a1-demo-{millis}")
}

fn default_output_root(run_id: &str) -> String {
    let base = env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root().join("scratch"));
    base.join("scratch/psion_cs336_a1_demo_runs")
        .join(run_id)
        .display()
        .to_string()
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

fn print_usage() {
    println!(
        "Usage:\n  psionic-train cs336-a1-demo start [--run-id <id>] [--output-root <path>] [--git-ref <ref>] [--allow-dirty-tree] [--dry-run]\n  psionic-train cs336-a1-demo rehearse-base-lane [--run-id <id>] [--output-root <path>] [--git-ref <ref>] [--allow-dirty-tree] [--dry-run]\n  psionic-train cs336-a1-demo status --run-root <path>\n\nThis is the bounded packaged CS336 A1 demo lane. It always uses the admitted tiny corpus and the fixed four-step training budget."
    );
}

#[cfg(test)]
mod tests {
    use super::{
        PSION_CS336_A1_DEMO_CLOSEOUT_BUNDLE_SCHEMA_VERSION, PsionCs336A1DemoCloseoutBundle,
        run_psion_cs336_a1_demo_manifest,
    };
    use crate::{
        PSION_CS336_A1_DEMO_AUTOMATIC_EXECUTION_REQUEST_SCHEMA_VERSION,
        PsionCs336A1DemoAutomaticExecutionRequest, PsionicTrainCoordinationContext,
        PsionicTrainOperation, PsionicTrainRole,
    };
    use tempfile::tempdir;

    fn request(output_root: &str) -> PsionCs336A1DemoAutomaticExecutionRequest {
        PsionCs336A1DemoAutomaticExecutionRequest {
            schema_version: String::from(
                PSION_CS336_A1_DEMO_AUTOMATIC_EXECUTION_REQUEST_SCHEMA_VERSION,
            ),
            role: PsionicTrainRole::Worker,
            operation: PsionicTrainOperation::Start,
            coordination: PsionicTrainCoordinationContext {
                node_pubkey: Some(String::from("npub1-demo-runner")),
                ..Default::default()
            },
            build_digest: String::from("sha256:test-build"),
            run_id: String::from("psion-cs336-a1-demo-test"),
            output_root: Some(String::from(output_root)),
            run_root: None,
            selected_git_ref: String::from("HEAD"),
            allow_dirty_tree: false,
            dry_run: false,
        }
    }

    #[test]
    fn packaged_demo_manifest_writes_checkpoint_and_closeout()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let run_root = temp.path().join("run");
        let manifest = request(run_root.to_str().expect("temp run root should be utf8"))
            .to_invocation_manifest()?;
        run_psion_cs336_a1_demo_manifest(&manifest)?;
        let surface = crate::inspect_psionic_train_checkpoint_surface(
            &run_root,
            manifest.role,
            manifest.operation,
        )?
        .expect("checkpoint surface should exist after a real bounded run");
        assert_eq!(surface.pointer_state.as_deref(), Some("accepted"));
        let closeout: PsionCs336A1DemoCloseoutBundle = serde_json::from_slice(&std::fs::read(
            run_root.join("closeout/closeout_bundle.json"),
        )?)?;
        assert_eq!(
            closeout.schema_version,
            PSION_CS336_A1_DEMO_CLOSEOUT_BUNDLE_SCHEMA_VERSION
        );
        assert!(
            closeout.final_loss < closeout.initial_loss,
            "the bounded demo run should descend over four steps"
        );
        Ok(())
    }

    #[test]
    fn packaged_demo_dry_run_skips_checkpoint_and_closeout()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let run_root = temp.path().join("run");
        let mut request = request(run_root.to_str().expect("temp run root should be utf8"));
        request.dry_run = true;
        let manifest = request.to_invocation_manifest()?;
        run_psion_cs336_a1_demo_manifest(&manifest)?;
        assert!(
            !run_root
                .join("checkpoints/latest_accepted_checkpoint_pointer.json")
                .is_file()
        );
        assert!(!run_root.join("closeout/closeout_bundle.json").is_file());
        Ok(())
    }
}
