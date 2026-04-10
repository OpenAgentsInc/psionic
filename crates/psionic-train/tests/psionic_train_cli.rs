use std::{fs, path::PathBuf, process::Command};

use psionic_train::{
    runtime_build_digest, PsionicTrainAdmissionIdentity, PsionicTrainCoordinationContext,
    PsionicTrainInvocationManifest, PsionicTrainOperation, PsionicTrainOutcomeKind,
    PsionicTrainRefusalClass, PsionicTrainRole, PsionicTrainRunStatusPacket,
    PsionicTrainStatusPacket, PsionicTrainWindowStatusPacket,
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
        coordination: PsionicTrainCoordinationContext::default(),
        admission_identity: admitted_identity(git_commit_sha.as_str()),
        run_id: Some(String::from("psion-train-cli-test")),
        output_root: None,
        run_root: None,
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

#[test]
fn machine_manifest_dry_run_emits_success_status_packet() {
    let tempdir = tempdir().expect("tempdir should exist");
    let run_root = tempdir.path().join("run");
    let manifest_path = tempdir.path().join("invocation.json");
    let mut manifest = base_manifest();
    manifest.output_root = Some(run_root.display().to_string());
    manifest.allow_dirty_tree = true;
    manifest.hardware_observation_path = Some(String::from(
        "fixtures/psion/pretrain/psion_actual_pretraining_hardware_observation_admitted_v1.json",
    ));
    manifest.run_shape_observation_path = Some(String::from(
        "fixtures/psion/pretrain/psion_actual_pretraining_run_shape_observation_admitted_v1.json",
    ));
    manifest
        .populate_manifest_digest()
        .expect("digest should populate");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).expect("manifest should serialize"),
    )
    .expect("manifest should write");

    let output = Command::new(binary_path())
        .args(["manifest", "--manifest"])
        .arg(&manifest_path)
        .output()
        .expect("psionic-train should run");

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
fn validator_role_is_reported_as_machine_refusal() {
    let tempdir = tempdir().expect("tempdir should exist");
    let manifest_path = tempdir.path().join("validator-invocation.json");
    let mut manifest = base_manifest();
    manifest.role = PsionicTrainRole::Validator;
    manifest.output_root = Some(tempdir.path().join("validator-run").display().to_string());
    manifest
        .populate_manifest_digest()
        .expect("digest should populate");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).expect("manifest should serialize"),
    )
    .expect("manifest should write");

    let output = Command::new(binary_path())
        .args(["manifest", "--manifest"])
        .arg(&manifest_path)
        .output()
        .expect("psionic-train should run");

    assert!(!output.status.success(), "validator role should be refused");
    let packet: PsionicTrainStatusPacket =
        serde_json::from_slice(&output.stderr).expect("refusal packet should parse");
    assert_eq!(
        packet.refusal_class,
        Some(PsionicTrainRefusalClass::BadConfig)
    );
    assert_eq!(packet.outcome, PsionicTrainOutcomeKind::Refused);
    assert_eq!(
        output.status.code(),
        Some(PsionicTrainRefusalClass::BadConfig.exit_code() as i32)
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
    manifest
        .populate_manifest_digest()
        .expect("digest should populate");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).expect("manifest should serialize"),
    )
    .expect("manifest should write");

    let output = Command::new(binary_path())
        .args(["manifest", "--manifest"])
        .arg(&manifest_path)
        .output()
        .expect("psionic-train should run");

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
