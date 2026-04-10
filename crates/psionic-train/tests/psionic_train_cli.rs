use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use psionic_train::{
    runtime_build_digest, PsionicTrainAdmissionIdentity, PsionicTrainCheckpointHandoffReceipt,
    PsionicTrainCheckpointHandoffSourceKind, PsionicTrainCheckpointSurface,
    PsionicTrainContributionArtifactManifest, PsionicTrainCoordinationContext,
    PsionicTrainInvocationManifest,
    PsionicTrainMembershipRevisionReceipt, PsionicTrainOperation, PsionicTrainOutcomeKind,
    PsionicTrainRefusalClass, PsionicTrainRole, PsionicTrainRunStatusPacket,
    PsionicTrainSealedWindowBundle, PsionicTrainStatusPacket,
    PsionicTrainValidatorScoreArtifact, PsionicTrainValidatorScoreReceipt,
    PsionicTrainWindowExecution,
    PsionicTrainWindowStatusPacket,
    TrainingExecutionValidatorDisposition,
    PSIONIC_TRAIN_ACTUAL_PRETRAINING_ENVIRONMENT_REF, PSIONIC_TRAIN_ACTUAL_PRETRAINING_RELEASE_ID,
    PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION, PSIONIC_TRAIN_RUNTIME_SURFACE_ID,
};
use sha2::{Digest, Sha256};
use tempfile::tempdir;

fn binary_path() -> PathBuf {
    PathBuf::from(
        std::env::var("CARGO_BIN_EXE_psionic-train")
            .expect("cargo should provide the psionic-train binary path"),
    )
}

fn git_head() -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("git rev-parse HEAD should run");
    assert!(output.status.success(), "git rev-parse HEAD should succeed");
    String::from_utf8(output.stdout)
        .expect("git output should be utf8")
        .trim()
        .to_string()
}

fn dirty_tree_build_inputs() -> (String, Option<String>) {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let porcelain = Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["status", "--porcelain"])
        .output()
        .expect("git status --porcelain should run");
    assert!(
        porcelain.status.success(),
        "git status --porcelain should succeed"
    );
    if porcelain.stdout.is_empty() {
        return (String::from("refuse_by_default"), None);
    }
    let status_snapshot = Command::new("git")
        .arg("-C")
        .arg(&repo_root)
        .args(["status", "--short", "--branch"])
        .output()
        .expect("git status --short --branch should run");
    assert!(
        status_snapshot.status.success(),
        "git status --short --branch should succeed"
    );
    let status_snapshot = String::from_utf8(status_snapshot.stdout)
        .expect("git status snapshot should be utf8")
        .trim()
        .to_string();
    let mut digest = Sha256::new();
    digest.update(status_snapshot.as_bytes());
    (
        String::from("allowed_by_operator_override"),
        Some(format!("{:x}", digest.finalize())),
    )
}

fn admitted_identity(git_commit_sha: &str) -> PsionicTrainAdmissionIdentity {
    let (dirty_tree_admission, workspace_status_sha256) = dirty_tree_build_inputs();
    PsionicTrainAdmissionIdentity {
        release_id: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_RELEASE_ID),
        build_digest: runtime_build_digest(
            PSIONIC_TRAIN_ACTUAL_PRETRAINING_RELEASE_ID,
            PSIONIC_TRAIN_RUNTIME_SURFACE_ID,
            psionic_train::PSION_ACTUAL_PRETRAINING_LANE_ID,
            git_commit_sha,
            dirty_tree_admission.as_str(),
            workspace_status_sha256.as_deref(),
            PSIONIC_TRAIN_ACTUAL_PRETRAINING_ENVIRONMENT_REF,
        ),
        environment_ref: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_ENVIRONMENT_REF),
    }
}

fn base_manifest() -> PsionicTrainInvocationManifest {
    let git_commit_sha = git_head();
    PsionicTrainInvocationManifest {
        schema_version: String::from(PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION),
        runtime_surface_id: String::from(PSIONIC_TRAIN_RUNTIME_SURFACE_ID),
        lane_id: String::from(psionic_train::PSION_ACTUAL_PRETRAINING_LANE_ID),
        role: PsionicTrainRole::Worker,
        operation: PsionicTrainOperation::Start,
        coordination: PsionicTrainCoordinationContext {
            network_id: Some(String::from("network.psionic.cli-test")),
            window_id: None,
            assignment_id: None,
            challenge_id: None,
            node_pubkey: Some(String::from("npub1-psionic-cli-test")),
            membership_revision: None,
        },
        admission_identity: admitted_identity(git_commit_sha.as_str()),
        run_id: Some(String::from("psion-train-cli-test")),
        output_root: None,
        run_root: None,
        peer_node_pubkey: None,
        peer_checkpoint_handoff_receipt_path: None,
        validator_target_contribution_receipt_path: None,
        validator_target_contribution_artifact_manifest_path: None,
        selected_git_ref: Some(String::from("HEAD")),
        hardware_observation_path: None,
        run_shape_observation_path: None,
        allow_dirty_tree: false,
        dry_run: true,
        checkpoint_label: None,
        optimizer_step: None,
        checkpoint_ref: None,
        checkpoint_object_digest: None,
        checkpoint_total_bytes: None,
        inject_failed_upload: false,
        inject_eval_worker_unavailable: false,
        manifest_digest: None,
    }
}

fn add_admitted_observations(manifest: &mut PsionicTrainInvocationManifest) {
    manifest.hardware_observation_path = Some(String::from(
        "fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json",
    ));
    manifest.run_shape_observation_path = Some(String::from(
        "fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json",
    ));
}

fn write_manifest(path: &Path, manifest: &mut PsionicTrainInvocationManifest) {
    manifest
        .populate_manifest_digest()
        .expect("manifest digest should populate");
    fs::write(
        path,
        serde_json::to_string_pretty(manifest).expect("manifest should serialize"),
    )
    .expect("manifest should write");
}

fn run_machine_manifest(manifest_path: &Path) -> Output {
    Command::new(binary_path())
        .args(["manifest", "--manifest"])
        .arg(manifest_path)
        .output()
        .expect("psionic-train should run")
}

fn parse_json<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> T {
    serde_json::from_slice(&fs::read(path.as_ref()).expect("retained artifact should be readable"))
        .expect("retained artifact should parse")
}

fn build_launch_manifest(run_root: &Path) -> PsionicTrainInvocationManifest {
    let mut manifest = base_manifest();
    manifest.output_root = Some(run_root.display().to_string());
    manifest.allow_dirty_tree = true;
    add_admitted_observations(&mut manifest);
    manifest
}

fn build_retained_operation_manifest(
    run_root: &Path,
    role: PsionicTrainRole,
    operation: PsionicTrainOperation,
) -> PsionicTrainInvocationManifest {
    let mut manifest = base_manifest();
    manifest.role = role;
    manifest.operation = operation;
    manifest.run_id = None;
    manifest.output_root = None;
    manifest.run_root = Some(run_root.display().to_string());
    manifest.peer_node_pubkey = None;
    manifest.peer_checkpoint_handoff_receipt_path = None;
    manifest.allow_dirty_tree = true;
    add_admitted_observations(&mut manifest);
    manifest
}

fn bind_window_context(
    manifest: &mut PsionicTrainInvocationManifest,
    window_id: &str,
    assignment_id: &str,
    membership_revision: u64,
) {
    manifest.coordination.window_id = Some(String::from(window_id));
    manifest.coordination.assignment_id = Some(String::from(assignment_id));
    manifest.coordination.membership_revision = Some(membership_revision);
}

fn build_validator_manifest(
    run_root: &Path,
    contribution_receipt_path: &Path,
    contribution_artifact_manifest_path: &Path,
    window_id: &str,
    assignment_id: &str,
    challenge_id: &str,
) -> PsionicTrainInvocationManifest {
    let mut manifest = build_retained_operation_manifest(
        run_root,
        PsionicTrainRole::Validator,
        PsionicTrainOperation::ValidateContribution,
    );
    manifest.coordination.window_id = Some(String::from(window_id));
    manifest.coordination.assignment_id = Some(String::from(assignment_id));
    manifest.coordination.challenge_id = Some(String::from(challenge_id));
    manifest.coordination.node_pubkey = Some(String::from("npub1-psionic-validator-cli-test"));
    manifest.validator_target_contribution_receipt_path =
        Some(contribution_receipt_path.display().to_string());
    manifest.validator_target_contribution_artifact_manifest_path =
        Some(contribution_artifact_manifest_path.display().to_string());
    manifest
}

#[test]
fn machine_manifest_dry_run_emits_success_status_packet() {
    let tempdir = tempdir().expect("tempdir should exist");
    let run_root = tempdir.path().join("run");
    let manifest_path = tempdir.path().join("invocation.json");
    let mut manifest = build_launch_manifest(&run_root);
    write_manifest(&manifest_path, &mut manifest);

    let output = run_machine_manifest(&manifest_path);

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stdout).expect("status packet should parse");
    assert_eq!(packet.outcome, PsionicTrainOutcomeKind::Succeeded);
    assert_eq!(packet.run_id.as_deref(), Some("psion-train-cli-test"));
    assert_eq!(
        packet.run_root.as_deref(),
        Some(run_root.to_string_lossy().as_ref())
    );
    assert!(packet.runtime_attestation.is_some());
    assert!(packet.capability_projection.is_some());
    assert!(packet.run_status_packet_path.is_some());
    assert!(packet.window_status_packet_path.is_some());
    assert!(run_root.join("manifests/launch_manifest.json").is_file());
    assert!(run_root.join("status/current_run_status.json").is_file());
    let run_status: PsionicTrainRunStatusPacket = serde_json::from_slice(
        &fs::read(
            packet
                .run_status_packet_path
                .as_ref()
                .expect("run status packet path should exist"),
        )
        .expect("run status packet should be readable"),
    )
    .expect("run status packet should parse");
    assert_eq!(run_status.run_id.as_deref(), Some("psion-train-cli-test"));
    let checkpoint_surface_path = run_status
        .artifacts
        .checkpoint_surface_path
        .as_ref()
        .expect("checkpoint surface path should exist");
    let checkpoint_surface: PsionicTrainCheckpointSurface = parse_json(checkpoint_surface_path);
    assert_eq!(
        checkpoint_surface.pointer_state.as_deref(),
        Some("pending_first_checkpoint")
    );
    assert_eq!(
        checkpoint_surface.checkpoint_label.as_deref(),
        Some("pending_first_checkpoint")
    );
    assert_eq!(checkpoint_surface.optimizer_step, None);
    let membership_revision_path = run_status
        .artifacts
        .membership_revision_path
        .as_ref()
        .expect("membership revision path should exist");
    let membership_receipt: PsionicTrainMembershipRevisionReceipt = serde_json::from_slice(
        &fs::read(membership_revision_path)
            .expect("membership revision receipt should be readable"),
    )
    .expect("membership revision receipt should parse");
    assert_eq!(
        membership_receipt.node_pubkey.as_str(),
        "npub1-psionic-cli-test"
    );
    assert_eq!(membership_receipt.run_id.as_str(), "psion-train-cli-test");
    let window_status: PsionicTrainWindowStatusPacket = serde_json::from_slice(
        &fs::read(
            packet
                .window_status_packet_path
                .as_ref()
                .expect("window status packet path should exist"),
        )
        .expect("window status packet should be readable"),
    )
    .expect("window status packet should parse");
    assert_eq!(
        window_status.run_id.as_deref(),
        Some("psion-train-cli-test")
    );
}

#[test]
fn machine_manifest_record_checkpoint_persists_checkpoint_surface() {
    let tempdir = tempdir().expect("tempdir should exist");
    let run_root = tempdir.path().join("run");

    let launch_manifest_path = tempdir.path().join("launch.json");
    let mut launch_manifest = build_launch_manifest(&run_root);
    write_manifest(&launch_manifest_path, &mut launch_manifest);
    let launch_output = run_machine_manifest(&launch_manifest_path);
    assert!(
        launch_output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&launch_output.stdout),
        String::from_utf8_lossy(&launch_output.stderr)
    );

    let checkpoint_manifest_path = tempdir.path().join("record-checkpoint.json");
    let mut checkpoint_manifest = build_retained_operation_manifest(
        &run_root,
        PsionicTrainRole::Worker,
        PsionicTrainOperation::RecordCheckpoint,
    );
    checkpoint_manifest.checkpoint_label = Some(String::from("broader-pretrain-final"));
    checkpoint_manifest.optimizer_step = Some(16_384);
    checkpoint_manifest.checkpoint_ref =
        Some(String::from("checkpoint://psion/broad/pretrain/final"));
    write_manifest(&checkpoint_manifest_path, &mut checkpoint_manifest);
    let output = run_machine_manifest(&checkpoint_manifest_path);

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stdout).expect("status packet should parse");
    assert_eq!(packet.outcome, PsionicTrainOutcomeKind::Succeeded);

    let run_status: PsionicTrainRunStatusPacket = parse_json(
        packet
            .run_status_packet_path
            .as_ref()
            .expect("run status packet path should exist"),
    );
    let checkpoint_surface: PsionicTrainCheckpointSurface = parse_json(
        run_status
            .artifacts
            .checkpoint_surface_path
            .as_ref()
            .expect("checkpoint surface path should exist"),
    );
    assert_eq!(
        checkpoint_surface.current_phase.as_deref(),
        Some("checkpoint_evaluated")
    );
    assert_eq!(
        checkpoint_surface.pointer_state.as_deref(),
        Some("accepted")
    );
    assert_eq!(
        checkpoint_surface.checkpoint_label.as_deref(),
        Some("broader-pretrain-final")
    );
    assert_eq!(checkpoint_surface.optimizer_step, Some(16_384));
    assert_eq!(
        checkpoint_surface.checkpoint_ref.as_deref(),
        Some("checkpoint://psion/broad/pretrain/final")
    );
    assert!(checkpoint_surface.checkpoint_manifest_digest.is_some());
    assert!(checkpoint_surface.checkpoint_object_digest.is_some());
    assert!(checkpoint_surface.checkpoint_total_bytes.is_some());
    assert_eq!(
        checkpoint_surface.backup_state.as_deref(),
        Some("backed_up")
    );
    assert_eq!(
        checkpoint_surface.upload_outcome.as_deref(),
        Some("succeeded")
    );
    assert!(Path::new(
        checkpoint_surface
            .artifacts
            .checkpoint_manifest_path
            .as_deref()
            .expect("checkpoint manifest path should exist"),
    )
    .is_file());
    assert!(Path::new(
        checkpoint_surface
            .artifacts
            .checkpoint_backup_receipt_path
            .as_deref()
            .expect("checkpoint backup receipt path should exist"),
    )
    .is_file());
    assert!(Path::new(
        checkpoint_surface
            .artifacts
            .checkpoint_backup_pointer_path
            .as_deref()
            .expect("checkpoint backup pointer path should exist"),
    )
    .is_file());
    assert!(Path::new(
        checkpoint_surface
            .artifacts
            .checkpoint_backup_manifest_path
            .as_deref()
            .expect("checkpoint backup manifest path should exist"),
    )
    .is_file());
}

#[test]
fn machine_manifest_resume_recovers_primary_pointer_from_backup() {
    let tempdir = tempdir().expect("tempdir should exist");
    let run_root = tempdir.path().join("run");

    let launch_manifest_path = tempdir.path().join("launch.json");
    let mut launch_manifest = build_launch_manifest(&run_root);
    write_manifest(&launch_manifest_path, &mut launch_manifest);
    let launch_output = run_machine_manifest(&launch_manifest_path);
    assert!(launch_output.status.success(), "launch should succeed");

    let checkpoint_manifest_path = tempdir.path().join("record-checkpoint.json");
    let mut checkpoint_manifest = build_retained_operation_manifest(
        &run_root,
        PsionicTrainRole::Worker,
        PsionicTrainOperation::RecordCheckpoint,
    );
    checkpoint_manifest.checkpoint_label = Some(String::from("broader-pretrain-final"));
    checkpoint_manifest.optimizer_step = Some(16_384);
    checkpoint_manifest.checkpoint_ref =
        Some(String::from("checkpoint://psion/broad/pretrain/final"));
    write_manifest(&checkpoint_manifest_path, &mut checkpoint_manifest);
    let checkpoint_output = run_machine_manifest(&checkpoint_manifest_path);
    assert!(
        checkpoint_output.status.success(),
        "record-checkpoint should succeed"
    );

    let primary_pointer_path = run_root.join("checkpoints/latest_accepted_checkpoint_pointer.json");
    fs::remove_file(&primary_pointer_path).expect("primary pointer should be removable");
    assert!(
        !primary_pointer_path.is_file(),
        "primary pointer should be gone"
    );

    let resume_manifest_path = tempdir.path().join("resume.json");
    let mut resume_manifest = build_retained_operation_manifest(
        &run_root,
        PsionicTrainRole::RecoverySource,
        PsionicTrainOperation::Resume,
    );
    resume_manifest.dry_run = true;
    write_manifest(&resume_manifest_path, &mut resume_manifest);
    let output = run_machine_manifest(&resume_manifest_path);

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stdout).expect("status packet should parse");
    let run_status: PsionicTrainRunStatusPacket = parse_json(
        packet
            .run_status_packet_path
            .as_ref()
            .expect("run status packet path should exist"),
    );
    let checkpoint_surface: PsionicTrainCheckpointSurface = parse_json(
        run_status
            .artifacts
            .checkpoint_surface_path
            .as_ref()
            .expect("checkpoint surface path should exist"),
    );
    assert_eq!(
        checkpoint_surface.recovery_resolution_state.as_deref(),
        Some("recovered_from_backup")
    );
    assert_eq!(
        checkpoint_surface.recovery_source_kind.as_deref(),
        Some("backup_receipt")
    );
    assert_eq!(checkpoint_surface.restored_primary_pointer, Some(true));
    assert!(Path::new(
        checkpoint_surface
            .artifacts
            .checkpoint_pointer_path
            .as_deref()
            .expect("checkpoint pointer path should exist"),
    )
    .is_file());
    assert!(Path::new(
        checkpoint_surface
            .artifacts
            .auto_resume_receipt_path
            .as_deref()
            .expect("auto-resume receipt path should exist"),
    )
    .is_file());
}

#[test]
fn machine_manifest_resume_refuses_without_any_admitted_checkpoint() {
    let tempdir = tempdir().expect("tempdir should exist");
    let run_root = tempdir.path().join("run");

    let launch_manifest_path = tempdir.path().join("launch.json");
    let mut launch_manifest = build_launch_manifest(&run_root);
    write_manifest(&launch_manifest_path, &mut launch_manifest);
    let launch_output = run_machine_manifest(&launch_manifest_path);
    assert!(launch_output.status.success(), "launch should succeed");

    let resume_manifest_path = tempdir.path().join("resume.json");
    let mut resume_manifest = build_retained_operation_manifest(
        &run_root,
        PsionicTrainRole::RecoverySource,
        PsionicTrainOperation::Resume,
    );
    resume_manifest.dry_run = true;
    write_manifest(&resume_manifest_path, &mut resume_manifest);
    let output = run_machine_manifest(&resume_manifest_path);

    assert!(!output.status.success(), "resume should be refused");
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stderr).expect("refusal packet should parse");
    assert_eq!(
        packet.refusal_class,
        Some(PsionicTrainRefusalClass::CheckpointMissing)
    );
    let run_status: PsionicTrainRunStatusPacket = parse_json(
        packet
            .run_status_packet_path
            .as_ref()
            .expect("run status packet path should exist"),
    );
    let checkpoint_surface: PsionicTrainCheckpointSurface = parse_json(
        run_status
            .artifacts
            .checkpoint_surface_path
            .as_ref()
            .expect("checkpoint surface path should exist"),
    );
    assert_eq!(
        checkpoint_surface.recovery_resolution_state.as_deref(),
        Some("refused")
    );
    assert_eq!(
        checkpoint_surface.recovery_source_kind.as_deref(),
        Some("none")
    );
    assert_eq!(checkpoint_surface.restored_primary_pointer, Some(false));
}

#[test]
fn machine_manifest_window_context_emits_window_and_contribution_artifacts() {
    let tempdir = tempdir().expect("tempdir should exist");
    let run_root = tempdir.path().join("windowed-run");
    let manifest_path = tempdir.path().join("windowed-launch.json");
    let mut manifest = build_launch_manifest(&run_root);
    bind_window_context(&mut manifest, "window-0001", "assignment-0001", 1);
    write_manifest(&manifest_path, &mut manifest);

    let output = run_machine_manifest(&manifest_path);

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stdout).expect("status packet should parse");
    let run_status: PsionicTrainRunStatusPacket = parse_json(
        packet
            .run_status_packet_path
            .as_ref()
            .expect("run status packet path should exist"),
    );
    assert!(run_status.artifacts.window_execution_path.is_some());
    assert!(run_status.artifacts.contribution_receipt_path.is_some());
    assert!(run_status.artifacts.contribution_artifact_manifest_path.is_some());
    assert!(run_status.artifacts.sealed_window_bundle_path.is_some());

    let window_execution: PsionicTrainWindowExecution = parse_json(
        run_status
            .artifacts
            .window_execution_path
            .as_ref()
            .expect("window execution path should exist"),
    );
    assert_eq!(window_execution.window_id.as_str(), "window-0001");
    assert_eq!(
        window_execution.current_assignment.assignment_id.as_str(),
        "assignment-0001"
    );

    let sealed_window: PsionicTrainSealedWindowBundle = parse_json(
        run_status
            .artifacts
            .sealed_window_bundle_path
            .as_ref()
            .expect("sealed window bundle path should exist"),
    );
    assert_eq!(sealed_window.window_id.as_str(), "window-0001");
    assert_eq!(sealed_window.contribution_count, 1);
    assert_eq!(sealed_window.artifact_manifest_count, 1);
    assert_eq!(sealed_window.contributions.len(), 1);
}

#[test]
fn machine_manifest_second_assignment_updates_sealed_window_rollup() {
    let tempdir = tempdir().expect("tempdir should exist");
    let run_root = tempdir.path().join("windowed-run");

    let launch_manifest_path = tempdir.path().join("windowed-launch.json");
    let mut launch_manifest = build_launch_manifest(&run_root);
    bind_window_context(&mut launch_manifest, "window-0001", "assignment-0001", 1);
    write_manifest(&launch_manifest_path, &mut launch_manifest);
    let launch_output = run_machine_manifest(&launch_manifest_path);
    assert!(launch_output.status.success(), "launch should succeed");

    let checkpoint_manifest_path = tempdir.path().join("windowed-record-checkpoint.json");
    let mut checkpoint_manifest = build_retained_operation_manifest(
        &run_root,
        PsionicTrainRole::Worker,
        PsionicTrainOperation::RecordCheckpoint,
    );
    bind_window_context(&mut checkpoint_manifest, "window-0001", "assignment-0002", 2);
    checkpoint_manifest.checkpoint_label = Some(String::from("broader-pretrain-final"));
    checkpoint_manifest.optimizer_step = Some(16_384);
    checkpoint_manifest.checkpoint_ref =
        Some(String::from("checkpoint://psion/broad/pretrain/final"));
    write_manifest(&checkpoint_manifest_path, &mut checkpoint_manifest);
    let output = run_machine_manifest(&checkpoint_manifest_path);

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stdout).expect("status packet should parse");
    let run_status: PsionicTrainRunStatusPacket = parse_json(
        packet
            .run_status_packet_path
            .as_ref()
            .expect("run status packet path should exist"),
    );
    let sealed_window: PsionicTrainSealedWindowBundle = parse_json(
        run_status
            .artifacts
            .sealed_window_bundle_path
            .as_ref()
            .expect("sealed window bundle path should exist"),
    );
    assert_eq!(sealed_window.window_id.as_str(), "window-0001");
    assert_eq!(sealed_window.contribution_count, 2);
    assert_eq!(sealed_window.artifact_manifest_count, 2);
    assert_eq!(sealed_window.contributions.len(), 2);
    assert_eq!(
        sealed_window.contributions[0].assignment_id.as_str(),
        "assignment-0001"
    );
    assert_eq!(
        sealed_window.contributions[1].assignment_id.as_str(),
        "assignment-0002"
    );
}

#[test]
fn machine_manifest_serve_checkpoint_retains_primary_handoff_receipt() {
    let tempdir = tempdir().expect("tempdir should exist");
    let source_run_root = tempdir.path().join("source-run");

    let launch_manifest_path = tempdir.path().join("launch.json");
    let mut launch_manifest = build_launch_manifest(&source_run_root);
    write_manifest(&launch_manifest_path, &mut launch_manifest);
    let launch_output = run_machine_manifest(&launch_manifest_path);
    assert!(launch_output.status.success(), "launch should succeed");

    let checkpoint_manifest_path = tempdir.path().join("record-checkpoint.json");
    let mut checkpoint_manifest = build_retained_operation_manifest(
        &source_run_root,
        PsionicTrainRole::Worker,
        PsionicTrainOperation::RecordCheckpoint,
    );
    checkpoint_manifest.checkpoint_label = Some(String::from("broader-pretrain-final"));
    checkpoint_manifest.optimizer_step = Some(16_384);
    checkpoint_manifest.checkpoint_ref =
        Some(String::from("checkpoint://psion/broad/pretrain/final"));
    write_manifest(&checkpoint_manifest_path, &mut checkpoint_manifest);
    let checkpoint_output = run_machine_manifest(&checkpoint_manifest_path);
    assert!(
        checkpoint_output.status.success(),
        "record-checkpoint should succeed"
    );

    let serve_manifest_path = tempdir.path().join("serve-checkpoint.json");
    let mut serve_manifest = build_retained_operation_manifest(
        &source_run_root,
        PsionicTrainRole::RecoverySource,
        PsionicTrainOperation::ServeCheckpoint,
    );
    serve_manifest.peer_node_pubkey = Some(String::from("npub1-late-joiner"));
    write_manifest(&serve_manifest_path, &mut serve_manifest);
    let output = run_machine_manifest(&serve_manifest_path);

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stdout).expect("status packet should parse");
    let run_status: PsionicTrainRunStatusPacket = parse_json(
        packet
            .run_status_packet_path
            .as_ref()
            .expect("run status packet path should exist"),
    );
    let handoff_receipt: PsionicTrainCheckpointHandoffReceipt = parse_json(
        run_status
            .artifacts
            .checkpoint_handoff_receipt_path
            .as_ref()
            .expect("checkpoint handoff receipt path should exist"),
    );
    assert_eq!(
        handoff_receipt.source_kind,
        PsionicTrainCheckpointHandoffSourceKind::LivePrimaryPointer
    );
    assert_eq!(
        handoff_receipt.peer_node_pubkey.as_str(),
        "npub1-late-joiner"
    );
    assert_eq!(
        handoff_receipt.serving_node_pubkey.as_str(),
        "npub1-psionic-cli-test"
    );
    assert_eq!(handoff_receipt.optimizer_step, 16_384);
}

#[test]
fn machine_manifest_resume_can_seed_from_peer_checkpoint_handoff() {
    let tempdir = tempdir().expect("tempdir should exist");
    let source_run_root = tempdir.path().join("source-run");

    let launch_manifest_path = tempdir.path().join("launch.json");
    let mut launch_manifest = build_launch_manifest(&source_run_root);
    write_manifest(&launch_manifest_path, &mut launch_manifest);
    let launch_output = run_machine_manifest(&launch_manifest_path);
    assert!(launch_output.status.success(), "launch should succeed");

    let checkpoint_manifest_path = tempdir.path().join("record-checkpoint.json");
    let mut checkpoint_manifest = build_retained_operation_manifest(
        &source_run_root,
        PsionicTrainRole::Worker,
        PsionicTrainOperation::RecordCheckpoint,
    );
    checkpoint_manifest.checkpoint_label = Some(String::from("broader-pretrain-final"));
    checkpoint_manifest.optimizer_step = Some(16_384);
    checkpoint_manifest.checkpoint_ref =
        Some(String::from("checkpoint://psion/broad/pretrain/final"));
    write_manifest(&checkpoint_manifest_path, &mut checkpoint_manifest);
    let checkpoint_output = run_machine_manifest(&checkpoint_manifest_path);
    assert!(
        checkpoint_output.status.success(),
        "record-checkpoint should succeed"
    );

    let serve_manifest_path = tempdir.path().join("serve-checkpoint.json");
    let mut serve_manifest = build_retained_operation_manifest(
        &source_run_root,
        PsionicTrainRole::RecoverySource,
        PsionicTrainOperation::ServeCheckpoint,
    );
    serve_manifest.peer_node_pubkey = Some(String::from("npub1-late-joiner"));
    write_manifest(&serve_manifest_path, &mut serve_manifest);
    let serve_output = run_machine_manifest(&serve_manifest_path);
    assert!(
        serve_output.status.success(),
        "serve-checkpoint should succeed"
    );
    let serve_packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&serve_output.stdout).expect("status packet should parse");
    let serve_run_status: PsionicTrainRunStatusPacket = parse_json(
        serve_packet
            .run_status_packet_path
            .as_ref()
            .expect("run status packet path should exist"),
    );
    let handoff_receipt_path = serve_run_status
        .artifacts
        .checkpoint_handoff_receipt_path
        .as_ref()
        .expect("checkpoint handoff receipt path should exist")
        .clone();

    let joiner_run_root = tempdir.path().join("joiner-run");
    let resume_manifest_path = tempdir.path().join("resume-from-peer.json");
    let mut resume_manifest = build_retained_operation_manifest(
        &joiner_run_root,
        PsionicTrainRole::RecoverySource,
        PsionicTrainOperation::Resume,
    );
    resume_manifest.coordination.node_pubkey = Some(String::from("npub1-late-joiner"));
    resume_manifest.peer_checkpoint_handoff_receipt_path = Some(handoff_receipt_path);
    resume_manifest.dry_run = true;
    write_manifest(&resume_manifest_path, &mut resume_manifest);
    let output = run_machine_manifest(&resume_manifest_path);

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stdout).expect("status packet should parse");
    let run_status: PsionicTrainRunStatusPacket = parse_json(
        packet
            .run_status_packet_path
            .as_ref()
            .expect("run status packet path should exist"),
    );
    assert_eq!(run_status.run_id.as_deref(), Some("psion-train-cli-test"));
    let checkpoint_surface: PsionicTrainCheckpointSurface = parse_json(
        run_status
            .artifacts
            .checkpoint_surface_path
            .as_ref()
            .expect("checkpoint surface path should exist"),
    );
    assert_eq!(
        checkpoint_surface.pointer_state.as_deref(),
        Some("accepted")
    );
    assert_eq!(
        checkpoint_surface.recovery_resolution_state.as_deref(),
        Some("accepted_primary_pointer")
    );
    assert!(Path::new(
        run_status
            .artifacts
            .checkpoint_handoff_receipt_path
            .as_deref()
            .expect("joiner checkpoint handoff receipt path should exist"),
    )
    .is_file());
    assert!(Path::new(
        checkpoint_surface
            .artifacts
            .checkpoint_pointer_path
            .as_deref()
            .expect("checkpoint pointer path should exist"),
    )
    .is_file());
    assert!(Path::new(
        checkpoint_surface
            .artifacts
            .checkpoint_manifest_path
            .as_deref()
            .expect("checkpoint manifest path should exist"),
    )
    .is_file());
}

#[test]
fn machine_manifest_serve_checkpoint_falls_back_to_backup_when_primary_is_missing() {
    let tempdir = tempdir().expect("tempdir should exist");
    let source_run_root = tempdir.path().join("source-run");

    let launch_manifest_path = tempdir.path().join("launch.json");
    let mut launch_manifest = build_launch_manifest(&source_run_root);
    write_manifest(&launch_manifest_path, &mut launch_manifest);
    let launch_output = run_machine_manifest(&launch_manifest_path);
    assert!(launch_output.status.success(), "launch should succeed");

    let checkpoint_manifest_path = tempdir.path().join("record-checkpoint.json");
    let mut checkpoint_manifest = build_retained_operation_manifest(
        &source_run_root,
        PsionicTrainRole::Worker,
        PsionicTrainOperation::RecordCheckpoint,
    );
    checkpoint_manifest.checkpoint_label = Some(String::from("broader-pretrain-final"));
    checkpoint_manifest.optimizer_step = Some(16_384);
    checkpoint_manifest.checkpoint_ref =
        Some(String::from("checkpoint://psion/broad/pretrain/final"));
    write_manifest(&checkpoint_manifest_path, &mut checkpoint_manifest);
    let checkpoint_output = run_machine_manifest(&checkpoint_manifest_path);
    assert!(
        checkpoint_output.status.success(),
        "record-checkpoint should succeed"
    );

    fs::remove_file(source_run_root.join("checkpoints/latest_accepted_checkpoint_pointer.json"))
        .expect("primary pointer should be removable");

    let serve_manifest_path = tempdir.path().join("serve-checkpoint.json");
    let mut serve_manifest = build_retained_operation_manifest(
        &source_run_root,
        PsionicTrainRole::RecoverySource,
        PsionicTrainOperation::ServeCheckpoint,
    );
    serve_manifest.peer_node_pubkey = Some(String::from("npub1-late-joiner"));
    write_manifest(&serve_manifest_path, &mut serve_manifest);
    let output = run_machine_manifest(&serve_manifest_path);

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stdout).expect("status packet should parse");
    let run_status: PsionicTrainRunStatusPacket = parse_json(
        packet
            .run_status_packet_path
            .as_ref()
            .expect("run status packet path should exist"),
    );
    let handoff_receipt: PsionicTrainCheckpointHandoffReceipt = parse_json(
        run_status
            .artifacts
            .checkpoint_handoff_receipt_path
            .as_ref()
            .expect("checkpoint handoff receipt path should exist"),
    );
    assert_eq!(
        handoff_receipt.source_kind,
        PsionicTrainCheckpointHandoffSourceKind::DurableBackupCopy
    );
    assert!(handoff_receipt.restored_from_backup);
}

#[test]
fn validator_manifest_emits_accepted_score_receipt_for_valid_contribution() {
    let tempdir = tempdir().expect("tempdir should exist");
    let worker_run_root = tempdir.path().join("worker-run");
    let worker_launch_manifest_path = tempdir.path().join("worker-windowed-launch.json");
    let mut worker_launch_manifest = build_launch_manifest(&worker_run_root);
    bind_window_context(&mut worker_launch_manifest, "window-0001", "assignment-0001", 1);
    write_manifest(&worker_launch_manifest_path, &mut worker_launch_manifest);
    let worker_launch_output = run_machine_manifest(&worker_launch_manifest_path);
    assert!(worker_launch_output.status.success(), "worker launch should succeed");

    let worker_checkpoint_manifest_path = tempdir.path().join("worker-record-checkpoint.json");
    let mut worker_checkpoint_manifest = build_retained_operation_manifest(
        &worker_run_root,
        PsionicTrainRole::Worker,
        PsionicTrainOperation::RecordCheckpoint,
    );
    bind_window_context(
        &mut worker_checkpoint_manifest,
        "window-0001",
        "assignment-0001",
        2,
    );
    worker_checkpoint_manifest.checkpoint_label = Some(String::from("validator-target"));
    worker_checkpoint_manifest.optimizer_step = Some(4_096);
    worker_checkpoint_manifest.checkpoint_ref =
        Some(String::from("checkpoint://psion/validator/target"));
    write_manifest(
        &worker_checkpoint_manifest_path,
        &mut worker_checkpoint_manifest,
    );
    let worker_output = run_machine_manifest(&worker_checkpoint_manifest_path);
    assert!(worker_output.status.success(), "worker checkpoint should succeed");

    let worker_packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&worker_output.stdout).expect("worker packet should parse");
    let worker_run_status: PsionicTrainRunStatusPacket = parse_json(
        worker_packet
            .run_status_packet_path
            .as_ref()
            .expect("worker run status path should exist"),
    );

    let validator_run_root = tempdir.path().join("validator-run");
    let manifest_path = tempdir.path().join("validator-invocation.json");
    let mut manifest = build_validator_manifest(
        &validator_run_root,
        Path::new(
            worker_run_status
                .artifacts
                .contribution_receipt_path
                .as_deref()
                .expect("worker contribution receipt path should exist"),
        ),
        Path::new(
            worker_run_status
                .artifacts
                .contribution_artifact_manifest_path
                .as_deref()
                .expect("worker contribution artifact manifest path should exist"),
        ),
        "window-0001",
        "assignment-0001",
        "challenge-0001",
    );
    write_manifest(&manifest_path, &mut manifest);

    let output = run_machine_manifest(&manifest_path);

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stdout).expect("validator packet should parse");
    let run_status: PsionicTrainRunStatusPacket = parse_json(
        packet
            .run_status_packet_path
            .as_ref()
            .expect("validator run status path should exist"),
    );
    let score_receipt: PsionicTrainValidatorScoreReceipt = parse_json(
        run_status
            .artifacts
            .validator_score_receipt_path
            .as_ref()
            .expect("validator score receipt path should exist"),
    );
    assert_eq!(
        score_receipt.disposition,
        TrainingExecutionValidatorDisposition::Accepted
    );
    assert_eq!(score_receipt.score_bps, 10_000);

    let score_artifact: PsionicTrainValidatorScoreArtifact =
        parse_json(&score_receipt.score_artifact_path);
    assert_eq!(
        score_artifact.disposition,
        TrainingExecutionValidatorDisposition::Accepted
    );
    assert_eq!(
        score_artifact.challenge_id.as_str(),
        "challenge-0001"
    );
}

#[test]
fn validator_manifest_emits_rejected_score_receipt_for_refused_contribution() {
    let tempdir = tempdir().expect("tempdir should exist");
    let worker_run_root = tempdir.path().join("worker-run");
    let worker_manifest_path = tempdir.path().join("worker-refused-launch.json");
    let mut worker_manifest = build_launch_manifest(&worker_run_root);
    bind_window_context(&mut worker_manifest, "window-0001", "assignment-0001", 1);
    worker_manifest.admission_identity.build_digest = String::from("sha256:not-the-real-build");
    write_manifest(&worker_manifest_path, &mut worker_manifest);
    let worker_output = run_machine_manifest(&worker_manifest_path);
    assert!(!worker_output.status.success(), "worker launch should be refused");

    let worker_packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&worker_output.stderr).expect("worker refusal packet should parse");
    let worker_run_status: PsionicTrainRunStatusPacket = parse_json(
        worker_packet
            .run_status_packet_path
            .as_ref()
            .expect("worker run status path should exist"),
    );

    let validator_run_root = tempdir.path().join("validator-run");
    let validator_manifest_path = tempdir.path().join("validator-rejected.json");
    let mut validator_manifest = build_validator_manifest(
        &validator_run_root,
        Path::new(
            worker_run_status
                .artifacts
                .contribution_receipt_path
                .as_deref()
                .expect("worker contribution receipt path should exist"),
        ),
        Path::new(
            worker_run_status
                .artifacts
                .contribution_artifact_manifest_path
                .as_deref()
                .expect("worker contribution artifact manifest path should exist"),
        ),
        "window-0001",
        "assignment-0001",
        "challenge-0002",
    );
    write_manifest(&validator_manifest_path, &mut validator_manifest);

    let output = run_machine_manifest(&validator_manifest_path);

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stdout).expect("validator packet should parse");
    let run_status: PsionicTrainRunStatusPacket = parse_json(
        packet
            .run_status_packet_path
            .as_ref()
            .expect("validator run status path should exist"),
    );
    let score_receipt: PsionicTrainValidatorScoreReceipt = parse_json(
        run_status
            .artifacts
            .validator_score_receipt_path
            .as_ref()
            .expect("validator score receipt path should exist"),
    );
    assert_eq!(
        score_receipt.disposition,
        TrainingExecutionValidatorDisposition::Rejected
    );
    assert_eq!(score_receipt.score_bps, 0);
}

#[test]
fn validator_manifest_refuses_stale_assignment_targets() {
    let tempdir = tempdir().expect("tempdir should exist");
    let worker_run_root = tempdir.path().join("worker-run");
    let worker_manifest_path = tempdir.path().join("worker-windowed-launch.json");
    let mut worker_manifest = build_launch_manifest(&worker_run_root);
    bind_window_context(&mut worker_manifest, "window-0001", "assignment-0001", 1);
    write_manifest(&worker_manifest_path, &mut worker_manifest);
    let worker_output = run_machine_manifest(&worker_manifest_path);
    assert!(worker_output.status.success(), "worker launch should succeed");

    let worker_packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&worker_output.stdout).expect("worker packet should parse");
    let worker_run_status: PsionicTrainRunStatusPacket = parse_json(
        worker_packet
            .run_status_packet_path
            .as_ref()
            .expect("worker run status path should exist"),
    );

    let validator_run_root = tempdir.path().join("validator-run");
    let manifest_path = tempdir.path().join("validator-stale-assignment.json");
    let mut manifest = build_validator_manifest(
        &validator_run_root,
        Path::new(
            worker_run_status
                .artifacts
                .contribution_receipt_path
                .as_deref()
                .expect("worker contribution receipt path should exist"),
        ),
        Path::new(
            worker_run_status
                .artifacts
                .contribution_artifact_manifest_path
                .as_deref()
                .expect("worker contribution artifact manifest path should exist"),
        ),
        "window-0001",
        "assignment-stale",
        "challenge-0003",
    );
    write_manifest(&manifest_path, &mut manifest);

    let output = run_machine_manifest(&manifest_path);

    assert!(!output.status.success(), "stale validator target should be refused");
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stderr).expect("refusal packet should parse");
    assert_eq!(
        packet.refusal_class,
        Some(PsionicTrainRefusalClass::StaleAssignment)
    );
    assert_eq!(packet.outcome, PsionicTrainOutcomeKind::Refused);
    assert_eq!(
        output.status.code(),
        Some(PsionicTrainRefusalClass::StaleAssignment.exit_code() as i32)
    );
}

#[test]
fn validator_manifest_refuses_missing_replay_inputs() {
    let tempdir = tempdir().expect("tempdir should exist");
    let run_root = tempdir.path().join("validator-run");
    let manifest_path = tempdir.path().join("validator-missing-inputs.json");
    let mut manifest = build_validator_manifest(
        &run_root,
        &tempdir.path().join("missing-contribution-receipt.json"),
        &tempdir.path().join("missing-artifact-manifest.json"),
        "window-0001",
        "assignment-0001",
        "challenge-0004",
    );
    write_manifest(&manifest_path, &mut manifest);

    let output = run_machine_manifest(&manifest_path);

    assert!(!output.status.success(), "missing validator inputs should be refused");
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stderr).expect("refusal packet should parse");
    assert_eq!(
        packet.refusal_class,
        Some(PsionicTrainRefusalClass::ArtifactIncomplete)
    );
}

#[test]
fn validator_manifest_refuses_artifact_manifest_digest_drift() {
    let tempdir = tempdir().expect("tempdir should exist");
    let worker_run_root = tempdir.path().join("worker-run");
    let worker_manifest_path = tempdir.path().join("worker-windowed-launch.json");
    let mut worker_manifest = build_launch_manifest(&worker_run_root);
    bind_window_context(&mut worker_manifest, "window-0001", "assignment-0001", 1);
    write_manifest(&worker_manifest_path, &mut worker_manifest);
    let worker_output = run_machine_manifest(&worker_manifest_path);
    assert!(worker_output.status.success(), "worker launch should succeed");

    let worker_packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&worker_output.stdout).expect("worker packet should parse");
    let worker_run_status: PsionicTrainRunStatusPacket = parse_json(
        worker_packet
            .run_status_packet_path
            .as_ref()
            .expect("worker run status path should exist"),
    );
    let artifact_manifest_path = PathBuf::from(
        worker_run_status
            .artifacts
            .contribution_artifact_manifest_path
            .as_deref()
            .expect("worker contribution artifact manifest path should exist"),
    );
    let mut artifact_manifest: PsionicTrainContributionArtifactManifest =
        parse_json(&artifact_manifest_path);
    artifact_manifest.artifact_manifest_digest = String::from("drifted-artifact-manifest-digest");
    fs::write(
        &artifact_manifest_path,
        serde_json::to_vec_pretty(&artifact_manifest).expect("artifact manifest should serialize"),
    )
    .expect("artifact manifest should rewrite");

    let validator_run_root = tempdir.path().join("validator-run");
    let manifest_path = tempdir.path().join("validator-drifted-manifest.json");
    let mut manifest = build_validator_manifest(
        &validator_run_root,
        Path::new(
            worker_run_status
                .artifacts
                .contribution_receipt_path
                .as_deref()
                .expect("worker contribution receipt path should exist"),
        ),
        &artifact_manifest_path,
        "window-0001",
        "assignment-0001",
        "challenge-0005",
    );
    write_manifest(&manifest_path, &mut manifest);

    let output = run_machine_manifest(&manifest_path);

    assert!(!output.status.success(), "drifted validator input should be refused");
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stderr).expect("refusal packet should parse");
    assert_eq!(
        packet.refusal_class,
        Some(PsionicTrainRefusalClass::ArtifactDigestMismatch)
    );
}

#[test]
fn build_identity_mismatch_is_refused_before_launch() {
    let tempdir = tempdir().expect("tempdir should exist");
    let manifest_path = tempdir.path().join("mismatched-build.json");
    let mut manifest = base_manifest();
    manifest.output_root = Some(
        tempdir
            .path()
            .join("mismatched-build-run")
            .display()
            .to_string(),
    );
    manifest.allow_dirty_tree = true;
    manifest.admission_identity.build_digest = String::from("sha256:not-the-real-build");
    write_manifest(&manifest_path, &mut manifest);

    let output = run_machine_manifest(&manifest_path);

    assert!(
        !output.status.success(),
        "mismatched build should be refused"
    );
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stderr).expect("refusal packet should parse");
    assert_eq!(
        packet.refusal_class,
        Some(PsionicTrainRefusalClass::BuildRevoked)
    );
}
