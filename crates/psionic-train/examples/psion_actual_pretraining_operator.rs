use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use psionic_train::{
    PSION_ACTUAL_PRETRAINING_CHECKPOINT_POINTER_SCHEMA_VERSION,
    PSION_ACTUAL_PRETRAINING_CLOSEOUT_BUNDLE_SCHEMA_VERSION,
    PSION_ACTUAL_PRETRAINING_CONTINUATION_HANDOFF_PATH,
    PSION_ACTUAL_PRETRAINING_CURRENT_RUN_STATUS_SCHEMA_VERSION,
    PSION_ACTUAL_PRETRAINING_DRY_RUN_SURFACE_ID, PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT_ID,
    PSION_ACTUAL_PRETRAINING_LANE_ID, PSION_ACTUAL_PRETRAINING_LAUNCH_MANIFEST_SCHEMA_VERSION,
    PSION_ACTUAL_PRETRAINING_RECIPE_ID, PSION_ACTUAL_PRETRAINING_RESUME_MANIFEST_SCHEMA_VERSION,
    PSION_ACTUAL_PRETRAINING_RESUME_SURFACE_ID,
    PSION_ACTUAL_PRETRAINING_RETAINED_SUMMARY_SCHEMA_VERSION,
    PSION_ACTUAL_PRETRAINING_START_SURFACE_ID, PSION_ACTUAL_PRETRAINING_STATUS_SURFACE_ID,
    PSION_ACTUAL_PRETRAINING_TOPOLOGY_STORAGE_BUNDLE_ID, PsionActualPretrainingArtifactRef,
    PsionActualPretrainingBaselineToolsBundle, PsionActualPretrainingCheckpointPointer,
    PsionActualPretrainingCloseoutBundle, PsionActualPretrainingContinuationHandoff,
    PsionActualPretrainingCredentialBinding, PsionActualPretrainingCurrentRunStatus,
    PsionActualPretrainingDataBundle, PsionActualPretrainingEvidenceContract,
    PsionActualPretrainingEvidenceContractError, PsionActualPretrainingLaneSpec,
    PsionActualPretrainingLaunchManifest, PsionActualPretrainingLauncherContractRefs,
    PsionActualPretrainingLauncherSurfaces, PsionActualPretrainingRecipeBundle,
    PsionActualPretrainingResumeManifest, PsionActualPretrainingRetainedPathSet,
    PsionActualPretrainingRetainedSummary, PsionActualPretrainingRunRoots,
    PsionActualPretrainingScalingBundle, PsionActualPretrainingSystemsBundle,
    PsionActualPretrainingTopologyStorageBundle, PsionPluginConditionedSftStageManifest,
    record_psion_actual_pretraining_continuation_handoff,
};
use sha2::{Digest, Sha256};

enum Cli {
    Start {
        run_id: String,
        run_root: PathBuf,
        selected_git_ref: String,
        allow_dirty_tree: bool,
        dry_run: bool,
    },
    Resume {
        run_root: PathBuf,
        selected_git_ref: String,
        allow_dirty_tree: bool,
        dry_run: bool,
    },
}

struct FrozenContracts {
    lane_spec_ref: PsionActualPretrainingArtifactRef,
    recipe_bundle_ref: PsionActualPretrainingArtifactRef,
    baseline_tools_bundle_ref: PsionActualPretrainingArtifactRef,
    scaling_bundle_ref: PsionActualPretrainingArtifactRef,
    data_bundle_ref: PsionActualPretrainingArtifactRef,
    systems_bundle_ref: PsionActualPretrainingArtifactRef,
    topology_storage_bundle_ref: PsionActualPretrainingArtifactRef,
    evidence_contract_ref: PsionActualPretrainingArtifactRef,
    recipe_bundle: PsionActualPretrainingRecipeBundle,
    plugin_conditioned_stage_manifest: PsionPluginConditionedSftStageManifest,
    topology: PsionActualPretrainingTopologyStorageBundle,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = parse_cli()?;
    let root = workspace_root()?;
    let repo_root = root.as_path();
    let contracts = load_frozen_contracts(repo_root)?;
    let launcher_surfaces = launcher_surfaces();
    launcher_surfaces.validate()?;
    let retained_paths_set = retained_paths();
    retained_paths_set.validate()?;

    match cli {
        Cli::Start {
            run_id,
            run_root,
            selected_git_ref,
            allow_dirty_tree,
            dry_run,
        } => {
            let git_commit_sha = git_output(repo_root, &["rev-parse", selected_git_ref.as_str()])?;
            let (dirty_tree_admission, workspace_status_sha256) =
                dirty_tree_posture(repo_root, allow_dirty_tree)?;
            let run_roots = run_roots(&run_root, &run_id, &contracts.topology);
            let contract_refs = PsionActualPretrainingLauncherContractRefs {
                lane_spec: contracts.lane_spec_ref.clone(),
                recipe_bundle: contracts.recipe_bundle_ref.clone(),
                baseline_tools_bundle: contracts.baseline_tools_bundle_ref.clone(),
                scaling_bundle: contracts.scaling_bundle_ref.clone(),
                data_bundle: contracts.data_bundle_ref.clone(),
                systems_bundle: contracts.systems_bundle_ref.clone(),
                topology_storage_bundle: contracts.topology_storage_bundle_ref.clone(),
                evidence_contract: contracts.evidence_contract_ref.clone(),
            };
            let credential_sources = credential_bindings(&contracts.topology);
            let launch_manifest = PsionActualPretrainingLaunchManifest {
                schema_version: String::from(
                    PSION_ACTUAL_PRETRAINING_LAUNCH_MANIFEST_SCHEMA_VERSION,
                ),
                surface_id: String::from(if dry_run {
                    PSION_ACTUAL_PRETRAINING_DRY_RUN_SURFACE_ID
                } else {
                    PSION_ACTUAL_PRETRAINING_START_SURFACE_ID
                }),
                lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
                recipe_id: String::from(PSION_ACTUAL_PRETRAINING_RECIPE_ID),
                topology_storage_bundle_id: String::from(
                    PSION_ACTUAL_PRETRAINING_TOPOLOGY_STORAGE_BUNDLE_ID,
                ),
                evidence_contract_id: String::from(PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT_ID),
                run_id: run_id.clone(),
                retained_paths: retained_paths_set.clone(),
                launcher_surfaces: launcher_surfaces.clone(),
                run_roots: run_roots.clone(),
                contract_refs,
                selected_git_ref: selected_git_ref.clone(),
                git_commit_sha: git_commit_sha.clone(),
                dirty_tree_admission: dirty_tree_admission.clone(),
                workspace_status_sha256: workspace_status_sha256.clone(),
                credential_sources,
                claim_boundary: String::from(
                    "The actual-lane launcher materializes the frozen launch manifest, retained status surfaces, checkpoint pointer, and provisional closeout bundle. It does not by itself execute the distributed broader-pretraining run.",
                ),
                detail: String::from(
                    "Launch manifest binds the actual pretraining operator command to the frozen lane, recipe, baseline-tools, scaling, data, systems, topology/storage, evidence, and git-provenance surfaces.",
                ),
            };
            launch_manifest.validate()?;

            let checkpoint_pointer = PsionActualPretrainingCheckpointPointer {
                schema_version: String::from(
                    PSION_ACTUAL_PRETRAINING_CHECKPOINT_POINTER_SCHEMA_VERSION,
                ),
                lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
                run_id: run_id.clone(),
                pointer_state: String::from("pending_first_checkpoint"),
                checkpoint_label: String::from("pending_first_checkpoint"),
                optimizer_step: 0,
                checkpoint_ref: None,
                checkpoint_manifest_relative_path: None,
                detail: String::from(
                    "Launch-time checkpoint pointer records that the actual lane has not yet admitted the first durable checkpoint.",
                ),
            };
            checkpoint_pointer.validate()?;

            let phase = if dry_run {
                "dry_run_planned"
            } else {
                "launch_staged"
            };
            let current_status = PsionActualPretrainingCurrentRunStatus {
                schema_version: String::from(
                    PSION_ACTUAL_PRETRAINING_CURRENT_RUN_STATUS_SCHEMA_VERSION,
                ),
                lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
                run_id: run_id.clone(),
                phase: String::from(phase),
                current_status_path: retained_paths_set.current_status_path.clone(),
                retained_summary_path: retained_paths_set.retained_summary_path.clone(),
                latest_checkpoint_pointer_path: retained_paths_set
                    .latest_checkpoint_pointer_path
                    .clone(),
                continuation_handoff_path: retained_paths_set.continuation_handoff_path.clone(),
                latest_checkpoint_label: String::from("pending_first_checkpoint"),
                last_completed_step: 0,
                launcher_surfaces: launcher_surfaces.clone(),
                updated_at_utc: now_utc(repo_root)?,
                detail: String::from(
                    "Current status records the canonical actual-lane launch state before the first accepted checkpoint exists.",
                ),
            };
            current_status.validate()?;

            let retained_summary = PsionActualPretrainingRetainedSummary {
                schema_version: String::from(
                    PSION_ACTUAL_PRETRAINING_RETAINED_SUMMARY_SCHEMA_VERSION,
                ),
                lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
                run_id: run_id.clone(),
                last_known_phase: String::from(phase),
                selected_git_ref: selected_git_ref.clone(),
                git_commit_sha: git_commit_sha.clone(),
                dirty_tree_admission: dirty_tree_admission.clone(),
                current_status_path: retained_paths_set.current_status_path.clone(),
                latest_checkpoint_pointer_path: retained_paths_set
                    .latest_checkpoint_pointer_path
                    .clone(),
                continuation_handoff_path: retained_paths_set.continuation_handoff_path.clone(),
                launcher_surfaces: launcher_surfaces.clone(),
                claim_boundary: String::from(
                    "The retained summary records actual-lane start, dry-run, resume, and status surfaces plus the last known operator state. It does not claim that cluster execution, automatic eval, or durable backup are finished.",
                ),
                detail: String::from(
                    "Retained summary keeps the last known actual-lane operator state legible outside the launch manifest.",
                ),
            };
            retained_summary.validate()?;

            let closeout_bundle = PsionActualPretrainingCloseoutBundle {
                schema_version: String::from(
                    PSION_ACTUAL_PRETRAINING_CLOSEOUT_BUNDLE_SCHEMA_VERSION,
                ),
                lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
                run_id: run_id.clone(),
                closeout_state: String::from(phase),
                retained_paths: retained_paths_set.clone(),
                selected_git_ref: selected_git_ref,
                git_commit_sha,
                dirty_tree_admission,
                workspace_status_sha256,
                claim_boundary: String::from(
                    "This provisional closeout bundle repeats launcher provenance early so later closeout work can extend the same evidence family without losing source-state identity. It does not claim completed training.",
                ),
                detail: String::from(
                    "Launch-time closeout bundle repeats the selected ref, git SHA, and dirty-tree posture inside the retained evidence family.",
                ),
            };
            closeout_bundle.validate()?;

            write_launcher_bundle(
                &run_root,
                Some(&launch_manifest),
                None,
                &current_status,
                &retained_summary,
                &checkpoint_pointer,
                None,
                &closeout_bundle,
                &format!(
                    "{} phase={} surface_id={} git_commit_sha={}\n",
                    now_utc(repo_root)?,
                    phase,
                    launch_manifest.surface_id,
                    launch_manifest.git_commit_sha,
                ),
            )?;

            println!("status={phase}");
            println!("surface_id={}", launch_manifest.surface_id);
            println!("run_id={run_id}");
            println!("run_root={}", run_root.display());
            println!(
                "launch_manifest={}",
                run_root
                    .join(&retained_paths_set.launch_manifest_path)
                    .display()
            );
            println!(
                "current_status={}",
                run_root
                    .join(&retained_paths_set.current_status_path)
                    .display()
            );
            println!(
                "retained_summary={}",
                run_root
                    .join(&retained_paths_set.retained_summary_path)
                    .display()
            );
            println!(
                "checkpoint_pointer={}",
                run_root
                    .join(&retained_paths_set.latest_checkpoint_pointer_path)
                    .display()
            );
            println!(
                "closeout_bundle={}",
                run_root
                    .join(&retained_paths_set.closeout_bundle_path)
                    .display()
            );
            println!(
                "launcher_log={}",
                run_root
                    .join(&retained_paths_set.launcher_log_path)
                    .display()
            );
        }
        Cli::Resume {
            run_root,
            selected_git_ref,
            allow_dirty_tree,
            dry_run,
        } => {
            let git_commit_sha = git_output(repo_root, &["rev-parse", selected_git_ref.as_str()])?;
            let (dirty_tree_admission, workspace_status_sha256) =
                dirty_tree_posture(repo_root, allow_dirty_tree)?;
            let retained_paths = retained_paths();
            let pointer_path = run_root.join(&retained_paths.latest_checkpoint_pointer_path);
            let checkpoint_pointer: PsionActualPretrainingCheckpointPointer =
                load_json(&pointer_path)?;
            checkpoint_pointer.validate()?;
            if checkpoint_pointer.pointer_state != "accepted" {
                return Err(std::io::Error::other(
                    "resume requires an accepted checkpoint pointer under checkpoints/latest_accepted_checkpoint_pointer.json",
                )
                .into());
            }
            let run_roots = run_roots(&run_root, &checkpoint_pointer.run_id, &contracts.topology);
            let resume_manifest = PsionActualPretrainingResumeManifest {
                schema_version: String::from(
                    PSION_ACTUAL_PRETRAINING_RESUME_MANIFEST_SCHEMA_VERSION,
                ),
                surface_id: String::from(PSION_ACTUAL_PRETRAINING_RESUME_SURFACE_ID),
                lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
                recipe_id: String::from(PSION_ACTUAL_PRETRAINING_RECIPE_ID),
                topology_storage_bundle_id: String::from(
                    PSION_ACTUAL_PRETRAINING_TOPOLOGY_STORAGE_BUNDLE_ID,
                ),
                evidence_contract_id: String::from(PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT_ID),
                run_id: checkpoint_pointer.run_id.clone(),
                retained_paths: retained_paths.clone(),
                launcher_surfaces: launcher_surfaces.clone(),
                run_roots: run_roots.clone(),
                contract_refs: PsionActualPretrainingLauncherContractRefs {
                    lane_spec: contracts.lane_spec_ref.clone(),
                    recipe_bundle: contracts.recipe_bundle_ref.clone(),
                    baseline_tools_bundle: contracts.baseline_tools_bundle_ref.clone(),
                    scaling_bundle: contracts.scaling_bundle_ref.clone(),
                    data_bundle: contracts.data_bundle_ref.clone(),
                    systems_bundle: contracts.systems_bundle_ref.clone(),
                    topology_storage_bundle: contracts.topology_storage_bundle_ref.clone(),
                    evidence_contract: contracts.evidence_contract_ref.clone(),
                },
                selected_git_ref: selected_git_ref.clone(),
                git_commit_sha: git_commit_sha.clone(),
                dirty_tree_admission: dirty_tree_admission.clone(),
                workspace_status_sha256: workspace_status_sha256.clone(),
                latest_checkpoint_pointer_path: retained_paths
                    .latest_checkpoint_pointer_path
                    .clone(),
                checkpoint_label: checkpoint_pointer.checkpoint_label.clone(),
                optimizer_step: checkpoint_pointer.optimizer_step,
                checkpoint_ref: checkpoint_pointer
                    .checkpoint_ref
                    .clone()
                    .expect("accepted checkpoint pointer must retain checkpoint_ref"),
                claim_boundary: String::from(
                    "The actual-lane resume manifest binds the canonical resume command to the accepted checkpoint pointer inside the frozen evidence family. It does not claim post-resume training success by itself.",
                ),
                detail: String::from(
                    "Resume manifest records the exact accepted checkpoint selection and repeats launcher provenance plus the frozen baseline-tools, scaling, data, and systems bundles for restart decisions.",
                ),
            };
            resume_manifest.validate()?;

            let phase = if dry_run {
                "resume_dry_run_planned"
            } else {
                "resume_staged"
            };
            let current_status = PsionActualPretrainingCurrentRunStatus {
                schema_version: String::from(
                    PSION_ACTUAL_PRETRAINING_CURRENT_RUN_STATUS_SCHEMA_VERSION,
                ),
                lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
                run_id: checkpoint_pointer.run_id.clone(),
                phase: String::from(phase),
                current_status_path: retained_paths.current_status_path.clone(),
                retained_summary_path: retained_paths.retained_summary_path.clone(),
                latest_checkpoint_pointer_path: retained_paths
                    .latest_checkpoint_pointer_path
                    .clone(),
                continuation_handoff_path: retained_paths.continuation_handoff_path.clone(),
                latest_checkpoint_label: checkpoint_pointer.checkpoint_label.clone(),
                last_completed_step: checkpoint_pointer.optimizer_step,
                launcher_surfaces: launcher_surfaces.clone(),
                updated_at_utc: now_utc(repo_root)?,
                detail: String::from(
                    "Current status records the accepted checkpoint selected by the canonical actual-lane resume command.",
                ),
            };
            current_status.validate()?;

            let retained_summary = PsionActualPretrainingRetainedSummary {
                schema_version: String::from(
                    PSION_ACTUAL_PRETRAINING_RETAINED_SUMMARY_SCHEMA_VERSION,
                ),
                lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
                run_id: checkpoint_pointer.run_id.clone(),
                last_known_phase: String::from(phase),
                selected_git_ref: selected_git_ref.clone(),
                git_commit_sha: git_commit_sha.clone(),
                dirty_tree_admission: dirty_tree_admission.clone(),
                current_status_path: retained_paths.current_status_path.clone(),
                latest_checkpoint_pointer_path: retained_paths
                    .latest_checkpoint_pointer_path
                    .clone(),
                continuation_handoff_path: retained_paths.continuation_handoff_path.clone(),
                launcher_surfaces: launcher_surfaces.clone(),
                claim_boundary: String::from(
                    "The retained summary records actual-lane start, dry-run, resume, and status surfaces plus the last known operator state. It does not claim that cluster execution, automatic eval, or durable backup are finished.",
                ),
                detail: String::from(
                    "Retained summary keeps the accepted resume checkpoint legible for operator review.",
                ),
            };
            retained_summary.validate()?;

            let closeout_bundle = PsionActualPretrainingCloseoutBundle {
                schema_version: String::from(
                    PSION_ACTUAL_PRETRAINING_CLOSEOUT_BUNDLE_SCHEMA_VERSION,
                ),
                lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
                run_id: checkpoint_pointer.run_id.clone(),
                closeout_state: String::from(phase),
                retained_paths: retained_paths.clone(),
                selected_git_ref: selected_git_ref,
                git_commit_sha,
                dirty_tree_admission,
                workspace_status_sha256,
                claim_boundary: String::from(
                    "This provisional closeout bundle repeats launcher provenance early so later closeout work can extend the same evidence family without losing source-state identity. It does not claim completed training.",
                ),
                detail: String::from(
                    "Resume-time closeout bundle repeats the selected ref, git SHA, and dirty-tree posture inside the retained evidence family.",
                ),
            };
            closeout_bundle.validate()?;
            let continuation_handoff = record_psion_actual_pretraining_continuation_handoff(
                &checkpoint_pointer,
                &contracts.recipe_bundle,
                &contracts.plugin_conditioned_stage_manifest,
            )?;

            write_launcher_bundle(
                &run_root,
                None,
                Some(&resume_manifest),
                &current_status,
                &retained_summary,
                &checkpoint_pointer,
                Some(&continuation_handoff),
                &closeout_bundle,
                &format!(
                    "{} phase={} surface_id={} git_commit_sha={}\n",
                    now_utc(repo_root)?,
                    phase,
                    resume_manifest.surface_id,
                    resume_manifest.git_commit_sha,
                ),
            )?;

            println!("status={phase}");
            println!("surface_id={}", resume_manifest.surface_id);
            println!("run_id={}", checkpoint_pointer.run_id);
            println!("run_root={}", run_root.display());
            println!(
                "resume_manifest={}",
                run_root
                    .join(&retained_paths.resume_manifest_path)
                    .display()
            );
            println!(
                "current_status={}",
                run_root.join(&retained_paths.current_status_path).display()
            );
            println!(
                "retained_summary={}",
                run_root
                    .join(&retained_paths.retained_summary_path)
                    .display()
            );
            println!(
                "checkpoint_pointer={}",
                run_root
                    .join(&retained_paths.latest_checkpoint_pointer_path)
                    .display()
            );
            println!(
                "continuation_handoff={}",
                run_root
                    .join(&retained_paths.continuation_handoff_path)
                    .display()
            );
            println!(
                "closeout_bundle={}",
                run_root
                    .join(&retained_paths.closeout_bundle_path)
                    .display()
            );
            println!(
                "launcher_log={}",
                run_root.join(&retained_paths.launcher_log_path).display()
            );
        }
    }

    Ok(())
}

fn parse_cli() -> Result<Cli, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        usage();
        return Err(std::io::Error::other("missing subcommand").into());
    };

    let mut run_id = String::new();
    let mut output_root = String::new();
    let mut run_root = String::new();
    let mut git_ref = String::new();
    let mut allow_dirty_tree = false;
    let mut dry_run = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--run-id" => {
                run_id = args
                    .next()
                    .ok_or_else(|| std::io::Error::other("--run-id requires a value"))?;
            }
            "--output-root" => {
                output_root = args
                    .next()
                    .ok_or_else(|| std::io::Error::other("--output-root requires a value"))?;
            }
            "--run-root" => {
                run_root = args
                    .next()
                    .ok_or_else(|| std::io::Error::other("--run-root requires a value"))?;
            }
            "--git-ref" => {
                git_ref = args
                    .next()
                    .ok_or_else(|| std::io::Error::other("--git-ref requires a value"))?;
            }
            "--allow-dirty-tree" => allow_dirty_tree = true,
            "--dry-run" => dry_run = true,
            "--help" | "-h" => {
                usage();
                std::process::exit(0);
            }
            other => {
                return Err(std::io::Error::other(format!("unknown argument `{other}`")).into());
            }
        }
    }

    let repo_root = workspace_root()?;
    let selected_git_ref = if git_ref.is_empty() {
        match git_output(repo_root.as_path(), &["symbolic-ref", "-q", "HEAD"]) {
            Ok(value) if !value.is_empty() => value,
            _ => String::from("HEAD"),
        }
    } else {
        git_ref
    };

    match command.as_str() {
        "start" => {
            let run_id = if run_id.is_empty() {
                format!(
                    "psion-actual-pretraining-{}",
                    timestamp_utc(repo_root.as_path())?
                )
            } else {
                run_id
            };
            let run_root = if output_root.is_empty() {
                PathBuf::from(env::var("HOME").unwrap_or_else(|_| String::from(".")))
                    .join("scratch/psion_actual_pretraining_runs")
                    .join(&run_id)
            } else {
                PathBuf::from(output_root)
            };
            Ok(Cli::Start {
                run_id,
                run_root,
                selected_git_ref,
                allow_dirty_tree,
                dry_run,
            })
        }
        "resume" => {
            if run_root.is_empty() {
                return Err(std::io::Error::other("resume requires --run-root <path>").into());
            }
            Ok(Cli::Resume {
                run_root: PathBuf::from(run_root),
                selected_git_ref,
                allow_dirty_tree,
                dry_run,
            })
        }
        _ => {
            usage();
            Err(std::io::Error::other(format!("unsupported subcommand `{command}`")).into())
        }
    }
}

fn usage() {
    eprintln!(
        "Usage:\n  psion_actual_pretraining_operator start [--run-id <id>] [--output-root <path>] [--git-ref <ref>] [--allow-dirty-tree] [--dry-run]\n  psion_actual_pretraining_operator resume --run-root <path> [--git-ref <ref>] [--allow-dirty-tree] [--dry-run]"
    );
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .ancestors()
        .nth(2)
        .map(PathBuf::from)
        .ok_or_else(|| "failed to resolve workspace root".into())
}

fn load_frozen_contracts(repo_root: &Path) -> Result<FrozenContracts, Box<dyn Error>> {
    let pretrain_dir = repo_root.join("fixtures/psion/pretrain");
    let lane_spec_path = pretrain_dir.join("psion_actual_pretraining_lane_spec_v1.json");
    let recipe_path = pretrain_dir.join("psion_actual_pretraining_recipe_bundle_v1.json");
    let baseline_tools_path =
        pretrain_dir.join("psion_actual_pretraining_baseline_tools_bundle_v1.json");
    let scaling_path = pretrain_dir.join("psion_actual_pretraining_scaling_bundle_v1.json");
    let data_path = pretrain_dir.join("psion_actual_pretraining_data_bundle_v1.json");
    let topology_path =
        pretrain_dir.join("psion_actual_pretraining_topology_storage_bundle_v1.json");
    let systems_path = pretrain_dir.join("psion_actual_pretraining_systems_bundle_v1.json");
    let evidence_path = pretrain_dir.join("psion_actual_pretraining_evidence_contract_v1.json");
    let lane_spec: PsionActualPretrainingLaneSpec = load_json(&lane_spec_path)?;
    lane_spec.validate()?;
    let recipe: PsionActualPretrainingRecipeBundle = load_json(&recipe_path)?;
    recipe.validate()?;
    let baseline_tools_bundle: PsionActualPretrainingBaselineToolsBundle =
        load_json(&baseline_tools_path)?;
    baseline_tools_bundle.validate()?;
    let scaling_bundle: PsionActualPretrainingScalingBundle = load_json(&scaling_path)?;
    scaling_bundle.validate()?;
    let data_bundle: PsionActualPretrainingDataBundle = load_json(&data_path)?;
    data_bundle.validate()?;
    let plugin_stage_manifest_path = repo_root.join(
        &recipe
            .continuation_target
            .plugin_conditioned_stage_manifest
            .path,
    );
    let plugin_conditioned_stage_manifest: PsionPluginConditionedSftStageManifest =
        load_json(&plugin_stage_manifest_path)?;
    let topology: PsionActualPretrainingTopologyStorageBundle = load_json(&topology_path)?;
    topology.validate()?;
    let systems_bundle: PsionActualPretrainingSystemsBundle = load_json(&systems_path)?;
    systems_bundle.validate()?;
    let evidence: PsionActualPretrainingEvidenceContract = load_json(&evidence_path)?;
    evidence.validate().map_err(map_evidence_error)?;

    Ok(FrozenContracts {
        lane_spec_ref: artifact_ref(repo_root, &lane_spec_path)?,
        recipe_bundle_ref: artifact_ref(repo_root, &recipe_path)?,
        baseline_tools_bundle_ref: artifact_ref(repo_root, &baseline_tools_path)?,
        scaling_bundle_ref: artifact_ref(repo_root, &scaling_path)?,
        data_bundle_ref: artifact_ref(repo_root, &data_path)?,
        systems_bundle_ref: artifact_ref(repo_root, &systems_path)?,
        topology_storage_bundle_ref: artifact_ref(repo_root, &topology_path)?,
        evidence_contract_ref: artifact_ref(repo_root, &evidence_path)?,
        recipe_bundle: recipe,
        plugin_conditioned_stage_manifest,
        topology,
    })
}

fn map_evidence_error(error: PsionActualPretrainingEvidenceContractError) -> Box<dyn Error> {
    Box::new(std::io::Error::other(error.to_string()))
}

fn retained_paths() -> PsionActualPretrainingRetainedPathSet {
    PsionActualPretrainingRetainedPathSet {
        launch_manifest_path: String::from("manifests/launch_manifest.json"),
        resume_manifest_path: String::from("manifests/resume_manifest.json"),
        current_status_path: String::from("status/current_run_status.json"),
        retained_summary_path: String::from("status/retained_summary.json"),
        latest_checkpoint_pointer_path: String::from(
            "checkpoints/latest_accepted_checkpoint_pointer.json",
        ),
        continuation_handoff_path: String::from(PSION_ACTUAL_PRETRAINING_CONTINUATION_HANDOFF_PATH),
        closeout_bundle_path: String::from("closeout/closeout_bundle.json"),
        launcher_log_path: String::from("logs/launcher.log"),
    }
}

fn launcher_surfaces() -> PsionActualPretrainingLauncherSurfaces {
    PsionActualPretrainingLauncherSurfaces {
        start_surface_id: String::from(PSION_ACTUAL_PRETRAINING_START_SURFACE_ID),
        dry_run_surface_id: String::from(PSION_ACTUAL_PRETRAINING_DRY_RUN_SURFACE_ID),
        resume_surface_id: String::from(PSION_ACTUAL_PRETRAINING_RESUME_SURFACE_ID),
        status_surface_id: String::from(PSION_ACTUAL_PRETRAINING_STATUS_SURFACE_ID),
    }
}

fn run_roots(
    run_root: &Path,
    run_id: &str,
    topology: &PsionActualPretrainingTopologyStorageBundle,
) -> PsionActualPretrainingRunRoots {
    PsionActualPretrainingRunRoots {
        local_run_root: run_root.display().to_string(),
        remote_run_root: topology
            .remote_run_root_template
            .replace("<run_id>", run_id),
        remote_checkpoint_root: topology
            .remote_checkpoint_root_template
            .replace("<run_id>", run_id),
        remote_manifest_root: topology
            .remote_manifest_root_template
            .replace("<run_id>", run_id),
        remote_log_root: topology
            .remote_log_root_template
            .replace("<run_id>", run_id),
    }
}

fn credential_bindings(
    topology: &PsionActualPretrainingTopologyStorageBundle,
) -> Vec<PsionActualPretrainingCredentialBinding> {
    topology
        .credential_sources
        .iter()
        .map(|source| PsionActualPretrainingCredentialBinding {
            kind: source.kind.clone(),
            source_name: source.source_name.clone(),
            retained_redaction: source.retained_redaction.clone(),
        })
        .collect()
}

fn dirty_tree_posture(
    repo_root: &Path,
    allow_dirty_tree: bool,
) -> Result<(String, Option<String>), Box<dyn Error>> {
    let porcelain = git_output(repo_root, &["status", "--porcelain"])?;
    if porcelain.is_empty() {
        return Ok((String::from("refuse_by_default"), None));
    }
    if !allow_dirty_tree {
        return Err(std::io::Error::other(
            "dirty working trees are refused by default; rerun with --allow-dirty-tree to override",
        )
        .into());
    }
    let status_snapshot = git_output(repo_root, &["status", "--short", "--branch"])?;
    Ok((
        String::from("allowed_by_operator_override"),
        Some(sha256_hex(status_snapshot.as_bytes())),
    ))
}

fn git_output(repo_root: &Path, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "git command failed: git -C {} {}",
            repo_root.display(),
            args.join(" ")
        ))
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn now_utc(repo_root: &Path) -> Result<String, Box<dyn Error>> {
    let output = Command::new("date")
        .arg("-u")
        .arg("+%Y-%m-%dT%H:%M:%SZ")
        .current_dir(repo_root)
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other("failed to get UTC time").into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn timestamp_utc(repo_root: &Path) -> Result<String, Box<dyn Error>> {
    let output = Command::new("date")
        .arg("-u")
        .arg("+%Y%m%dT%H%M%SZ")
        .current_dir(repo_root)
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other("failed to get UTC timestamp").into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn load_json<T>(path: &Path) -> Result<T, Box<dyn Error>>
where
    T: serde::de::DeserializeOwned,
{
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn artifact_ref(
    root: &Path,
    path: &Path,
) -> Result<PsionActualPretrainingArtifactRef, Box<dyn Error>> {
    let relative = path
        .strip_prefix(root)?
        .to_string_lossy()
        .replace('\\', "/");
    Ok(PsionActualPretrainingArtifactRef {
        path: relative,
        sha256: file_sha256(path)?,
    })
}

fn file_sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn write_launcher_bundle(
    run_root: &Path,
    launch_manifest: Option<&PsionActualPretrainingLaunchManifest>,
    resume_manifest: Option<&PsionActualPretrainingResumeManifest>,
    current_status: &PsionActualPretrainingCurrentRunStatus,
    retained_summary: &PsionActualPretrainingRetainedSummary,
    checkpoint_pointer: &PsionActualPretrainingCheckpointPointer,
    continuation_handoff: Option<&PsionActualPretrainingContinuationHandoff>,
    closeout_bundle: &PsionActualPretrainingCloseoutBundle,
    launcher_log_line: &str,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(run_root.join("manifests"))?;
    fs::create_dir_all(run_root.join("status"))?;
    fs::create_dir_all(run_root.join("checkpoints"))?;
    fs::create_dir_all(run_root.join("continuation"))?;
    fs::create_dir_all(run_root.join("closeout"))?;
    fs::create_dir_all(run_root.join("logs"))?;
    if let Some(launch_manifest) = launch_manifest {
        fs::write(
            run_root.join("manifests/launch_manifest.json"),
            serde_json::to_string_pretty(launch_manifest)?,
        )?;
    }
    if let Some(resume_manifest) = resume_manifest {
        fs::write(
            run_root.join("manifests/resume_manifest.json"),
            serde_json::to_string_pretty(resume_manifest)?,
        )?;
    }
    fs::write(
        run_root.join("status/current_run_status.json"),
        serde_json::to_string_pretty(current_status)?,
    )?;
    fs::write(
        run_root.join("status/retained_summary.json"),
        serde_json::to_string_pretty(retained_summary)?,
    )?;
    fs::write(
        run_root.join("checkpoints/latest_accepted_checkpoint_pointer.json"),
        serde_json::to_string_pretty(checkpoint_pointer)?,
    )?;
    if let Some(continuation_handoff) = continuation_handoff {
        fs::write(
            run_root.join(PSION_ACTUAL_PRETRAINING_CONTINUATION_HANDOFF_PATH),
            serde_json::to_string_pretty(continuation_handoff)?,
        )?;
    }
    fs::write(
        run_root.join("closeout/closeout_bundle.json"),
        serde_json::to_string_pretty(closeout_bundle)?,
    )?;
    fs::write(run_root.join("logs/launcher.log"), launcher_log_line)?;
    Ok(())
}
