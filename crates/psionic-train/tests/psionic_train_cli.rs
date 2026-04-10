use std::{fs, path::PathBuf, process::Command};

use psionic_train::{
    PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION, PSIONIC_TRAIN_RUNTIME_SURFACE_ID,
    PsionicTrainInvocationManifest, PsionicTrainOperation, PsionicTrainOutcomeKind,
    PsionicTrainRefusalClass, PsionicTrainRole, PsionicTrainStatusPacket,
};
use tempfile::tempdir;

fn binary_path() -> PathBuf {
    PathBuf::from(
        std::env::var("CARGO_BIN_EXE_psionic-train")
            .expect("cargo should provide the psionic-train binary path"),
    )
}

fn base_manifest() -> PsionicTrainInvocationManifest {
    PsionicTrainInvocationManifest {
        schema_version: String::from(PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION),
        runtime_surface_id: String::from(PSIONIC_TRAIN_RUNTIME_SURFACE_ID),
        lane_id: String::from(psionic_train::PSION_ACTUAL_PRETRAINING_LANE_ID),
        role: PsionicTrainRole::Worker,
        operation: PsionicTrainOperation::Start,
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
    assert!(run_root.join("manifests/launch_manifest.json").is_file());
    assert!(run_root.join("status/current_run_status.json").is_file());
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
