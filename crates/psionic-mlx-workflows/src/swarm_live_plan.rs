use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use psionic_cluster::{
    AdmissionToken, ClusterBackendReadinessStatus, ClusterId, ClusterMembershipRecord,
    ClusterMembershipStatus, ClusterNamespace, ClusterNodeIdentity, ClusterNodeTelemetry,
    ClusterSnapshot, ClusterStabilityPosture, ClusterState, NodeEpoch, NodeId, NodeRole,
};
use psionic_core::{DType, Device, Shape, TensorData};
use psionic_environments::EnvironmentPackageKey;
use psionic_mlx_recipes::{MlxAdapterRecipe, MlxRecipeConfig, MlxRecipeMethod, MlxRecipePlan};
use psionic_runtime::TrainingCheckpointReference;
use psionic_train::{
    AdapterClusterMembershipReceipt, AdapterClusterWindowPlanReceipt,
    AdapterContributorBackendCapability, AdapterContributorCapabilityPolicy,
    AdapterContributorEligibility, AdapterDatasetSliceIdentity, AdapterTargetIdentity,
    AdapterTrainingClusterCoordinator, CheckpointPointer, CheckpointScopeBinding,
    CheckpointScopeKind, OPEN_ADAPTER_CUDA_BACKEND_LABEL, OPEN_ADAPTER_MLX_METAL_BACKEND_LABEL,
    OptimizerStateResidency, PortableModelBundle, PortableModelStateDict,
    PortableTokenizerAssetFormat, PortableTokenizerBinding, TrainingOptimizerConfig,
    TrainingOptimizerResidencyPolicy, TrainingOptimizerState, TrainingParameterClass,
    TrainingParameterGroupState, TrainingTensorBuffer, first_swarm_open_adapter_receipt_contract,
    first_swarm_run_contract, first_swarm_tokenizer_digest,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    MlxPublishConfig, MlxPublishTarget, MlxSyntheticDatasetArtifact, MlxSyntheticDatasetReport,
    MlxSyntheticSftDatasetSpec, MlxSyntheticSftSample, MlxSyntheticSftSplit, MlxWorkflowError,
    MlxWorkflowWorkspace,
};

/// Stable scope window for the first swarm live workflow plan.
pub const FIRST_SWARM_LIVE_WORKFLOW_PLAN_SCOPE_WINDOW: &str = "first_swarm_live_workflow_plan_v1";
/// Stable fixture path for the first swarm live workflow plan.
pub const FIRST_SWARM_LIVE_WORKFLOW_PLAN_FIXTURE_PATH: &str =
    "fixtures/swarm/first_swarm_live_workflow_plan_v1.json";
/// Stable scope window for the first swarm local snapshot publication proof.
pub const FIRST_SWARM_LOCAL_SNAPSHOT_PUBLICATION_SCOPE_WINDOW: &str =
    "first_swarm_local_snapshot_publication_v1";
/// Stable fixture root for the retained first swarm local snapshot publication proof.
pub const FIRST_SWARM_LOCAL_SNAPSHOT_PUBLICATION_FIXTURE_ROOT: &str = "fixtures/swarm/publications";
/// Stable fixture path for the retained first swarm local snapshot publication report.
pub const FIRST_SWARM_LOCAL_SNAPSHOT_PUBLICATION_REPORT_FIXTURE_PATH: &str =
    "fixtures/swarm/publications/first_swarm_local_snapshot_publication_v1.json";

/// One machine-legible contributor assignment in the first swarm live plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FirstSwarmWorkflowContributorAssignment {
    /// Stable swarm role id from the frozen run contract.
    pub role_id: String,
    /// Stable platform label from the frozen run contract.
    pub platform: String,
    /// Stable node id selected for the role.
    pub contributor_node_id: String,
    /// Backend label that satisfied the contributor capability policy.
    pub matched_backend_label: String,
    /// Whether the role may validate contribution uploads.
    pub validator_eligible: bool,
    /// Whether the role may aggregate accepted deltas.
    pub aggregation_eligible: bool,
    /// Deterministic contributor priority at selection time.
    pub priority_bps: u16,
    /// Deterministic contributor reliability at selection time.
    pub reliability_bps: u16,
    /// Dataset slice assigned through the live adapter-cluster window planner.
    pub dataset_slice: AdapterDatasetSliceIdentity,
    /// Source sample ids consumed from the MLX workflow dataset artifact.
    pub source_sample_ids: Vec<String>,
    /// Source split object digest consumed from the MLX workflow report.
    pub source_split_object_digest: String,
    /// Stable contribution id emitted by the adapter window planner.
    pub contribution_id: String,
    /// Stable upload reference expected for this contributor.
    pub expected_upload_reference: String,
    /// Stable upload-manifest digest expected for this contributor.
    pub expected_upload_manifest_digest: String,
}

/// Publish expectation for the first swarm lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FirstSwarmWorkflowPublishExpectation {
    /// Stable role id that may publish a local snapshot.
    pub publisher_role_id: String,
    /// Stable publish identifier.
    pub publish_id: String,
    /// Publish target admitted by the workflow package.
    pub target: MlxPublishTarget,
    /// Repository identifier from the workflow publish config.
    pub repo_id: String,
    /// Expected local snapshot directory for an accepted mergeable outcome.
    pub expected_local_snapshot_directory: String,
    /// Frozen publish posture from the swarm contract.
    pub publish_posture: String,
}

/// Machine-legible first swarm live workflow plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FirstSwarmLiveWorkflowPlan {
    /// Stable schema version.
    pub schema_version: u32,
    /// Stable scope window.
    pub scope_window: String,
    /// Stable run family id from the swarm contract.
    pub run_family_id: String,
    /// Stable first-swarm contract digest.
    pub swarm_contract_digest: String,
    /// MLX recipe plan consumed by the live adapter-cluster bridge.
    pub recipe_plan: MlxRecipePlan,
    /// Workflow dataset report consumed by the same live bridge.
    pub dataset_report: MlxSyntheticDatasetReport,
    /// Workflow publish config consumed by the same live bridge.
    pub publish_config: MlxPublishConfig,
    /// Contributor capability policy projected into the live cluster coordinator.
    pub capability_policy: AdapterContributorCapabilityPolicy,
    /// Membership receipt emitted by the live adapter-cluster coordinator.
    pub membership_receipt: AdapterClusterMembershipReceipt,
    /// Window plan emitted by the live adapter-cluster coordinator.
    pub window_plan: AdapterClusterWindowPlanReceipt,
    /// Role-aware contributor assignments for the first swarm lane.
    pub contributor_assignments: Vec<FirstSwarmWorkflowContributorAssignment>,
    /// Publish expectation for a later accepted mergeable outcome.
    pub publish_expectation: FirstSwarmWorkflowPublishExpectation,
    /// Honest claim boundary for the plan.
    pub claim_boundary: String,
    /// Deterministic planning notes.
    pub notes: Vec<String>,
    /// Stable plan digest.
    pub plan_digest: String,
}

impl FirstSwarmLiveWorkflowPlan {
    /// Returns the stable digest over the live workflow plan.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.plan_digest.clear();
        stable_digest(b"psionic_first_swarm_live_workflow_plan|", &clone)
    }
}

/// Retained publication proof for one truthful first-swarm local snapshot directory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FirstSwarmLocalSnapshotPublicationReport {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable scope window.
    pub scope_window: String,
    /// Stable run family identifier.
    pub run_family_id: String,
    /// Stable workflow-plan digest that bound the publish expectation.
    pub workflow_plan_digest: String,
    /// Stable publish identifier.
    pub publish_id: String,
    /// Publish target admitted by the workflow package.
    pub target: MlxPublishTarget,
    /// Logical repository identifier for the local snapshot.
    pub repo_id: String,
    /// Expected relative snapshot directory from the workflow plan.
    pub expected_local_snapshot_directory: String,
    /// Relative snapshot root actually written by the proof.
    pub published_snapshot_root: String,
    /// Stable merge report digest for the published portable bundle.
    pub merge_report_digest: String,
    /// Stable publish manifest digest.
    pub publish_manifest_digest: String,
    /// Stable merged bundle state-dict digest.
    pub merged_state_dict_digest: String,
    /// Stable merged artifact digest.
    pub merged_artifact_digest: String,
    /// Stable base state-dict digest used for the merge proof.
    pub base_state_dict_digest: String,
    /// Stable tuned state-dict digest used for the merge proof.
    pub tuned_state_dict_digest: String,
    /// Ordered payload files emitted by the published snapshot.
    pub published_files: Vec<crate::MlxWorkflowFile>,
    /// Honest bounded notes.
    pub notes: Vec<String>,
    /// Honest claim boundary.
    pub claim_boundary: String,
    /// Stable report digest.
    pub report_digest: String,
}

impl FirstSwarmLocalSnapshotPublicationReport {
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.report_digest.clear();
        stable_digest(
            b"psionic_first_swarm_local_snapshot_publication_report|",
            &clone,
        )
    }
}

#[derive(Clone, Debug)]
struct PlannedRoleAssignment {
    role_id: String,
    platform: String,
    contributor_node_id: String,
    matched_backend_label: String,
    validator_eligible: bool,
    aggregation_eligible: bool,
    priority_bps: u16,
    reliability_bps: u16,
    dataset_slice: AdapterDatasetSliceIdentity,
    source_sample_ids: Vec<String>,
    source_split_object_digest: String,
}

impl MlxWorkflowWorkspace {
    /// Builds the canonical first swarm synthetic SFT dataset artifact.
    pub fn build_first_swarm_synthetic_sft_dataset(
        &self,
    ) -> Result<MlxSyntheticDatasetArtifact, MlxWorkflowError> {
        self.build_synthetic_sft_dataset(&first_swarm_synthetic_sft_spec())
    }

    /// Binds MLX recipe, dataset, and publish planning into one live adapter-cluster plan.
    pub fn plan_first_swarm_live_adapter_cluster(
        &self,
        recipe: &MlxRecipeConfig,
        dataset: &MlxSyntheticDatasetArtifact,
        publish: &MlxPublishConfig,
        cluster_state: &ClusterState,
        observed_at_ms: u64,
        planned_at_ms: u64,
    ) -> Result<FirstSwarmLiveWorkflowPlan, MlxWorkflowError> {
        if publish.target != MlxPublishTarget::HuggingFaceSnapshot {
            return Err(MlxWorkflowError::FirstSwarmPlan {
                message: String::from(
                    "the first swarm lane may only plan a local Hugging Face-style snapshot publish target",
                ),
            });
        }

        let recipe_plan = self.recipe_workspace.plan(recipe)?;
        validate_first_swarm_recipe_plan(&recipe_plan)?;
        validate_first_swarm_dataset(dataset)?;

        let swarm_contract = first_swarm_run_contract();
        let capability_policy = first_swarm_capability_policy()?;
        let adapter_target = first_swarm_adapter_target_identity(&recipe_plan)?;
        let checkpoint_pointer = first_swarm_checkpoint_pointer(
            recipe_plan.run_graph.run_id.as_str(),
            cluster_state.cluster_id().as_str(),
            recipe_plan.run_graph.checkpoint_family.as_str(),
            planned_at_ms,
        )?;

        let mut coordinator = AdapterTrainingClusterCoordinator::new(
            recipe_plan.run_graph.clone(),
            adapter_target,
            recipe_plan.policy_revision.clone(),
            checkpoint_pointer,
            capability_policy.clone(),
        );
        let membership_receipt = coordinator
            .observe_cluster_state(cluster_state, observed_at_ms)
            .map_err(|error| MlxWorkflowError::FirstSwarmPlan {
                message: error.to_string(),
            })?
            .clone();
        let planned_role_assignments =
            plan_role_assignments(&swarm_contract, &membership_receipt, dataset)?;
        let window = coordinator
            .plan_next_window_with_selected_nodes(
                planned_role_assignments
                    .iter()
                    .map(|assignment| assignment.dataset_slice.clone())
                    .collect(),
                planned_role_assignments
                    .iter()
                    .map(|assignment| assignment.contributor_node_id.clone())
                    .collect(),
                planned_at_ms,
            )
            .map_err(|error| MlxWorkflowError::FirstSwarmPlan {
                message: error.to_string(),
            })?;
        let contribution_ids = window
            .window
            .contributions
            .iter()
            .map(|contribution| {
                (
                    contribution.assignment.binding.contributor_node_id.clone(),
                    contribution.assignment.binding.contribution_id.clone(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let contributor_assignments = planned_role_assignments
            .into_iter()
            .map(|assignment| {
                let contribution_id = contribution_ids
                    .get(&assignment.contributor_node_id)
                    .cloned()
                    .ok_or_else(|| MlxWorkflowError::FirstSwarmPlan {
                        message: format!(
                            "live adapter window did not emit a contribution id for node `{}`",
                            assignment.contributor_node_id
                        ),
                    })?;
                Ok(FirstSwarmWorkflowContributorAssignment {
                    role_id: assignment.role_id.clone(),
                    platform: assignment.platform,
                    contributor_node_id: assignment.contributor_node_id.clone(),
                    matched_backend_label: assignment.matched_backend_label,
                    validator_eligible: assignment.validator_eligible,
                    aggregation_eligible: assignment.aggregation_eligible,
                    priority_bps: assignment.priority_bps,
                    reliability_bps: assignment.reliability_bps,
                    dataset_slice: assignment.dataset_slice.clone(),
                    source_sample_ids: assignment.source_sample_ids,
                    source_split_object_digest: assignment.source_split_object_digest,
                    contribution_id: contribution_id.clone(),
                    expected_upload_reference: format!(
                        "object://swarm/{}/{}/adapter_delta.safetensors",
                        window.plan.window_id, contribution_id
                    ),
                    expected_upload_manifest_digest: stable_expected_upload_manifest_digest(
                        publish.publish_id.as_str(),
                        publish.repo_id.as_str(),
                        contribution_id.as_str(),
                        assignment.dataset_slice.slice_digest.as_str(),
                    ),
                })
            })
            .collect::<Result<Vec<_>, MlxWorkflowError>>()?;
        let publish_expectation = FirstSwarmWorkflowPublishExpectation {
            publisher_role_id: swarm_contract
                .node_roles
                .iter()
                .find(|role| role.validator_eligible)
                .map(|role| role.role_id.clone())
                .ok_or_else(|| MlxWorkflowError::FirstSwarmPlan {
                    message: String::from(
                        "first swarm contract is missing a validator-capable role",
                    ),
                })?,
            publish_id: publish.publish_id.clone(),
            target: publish.target,
            repo_id: publish.repo_id.clone(),
            expected_local_snapshot_directory: format!(
                "local_publish/{}/{}",
                sanitize_label(publish.repo_id.as_str()),
                publish.publish_id
            ),
            publish_posture: swarm_contract.governance.publish_posture.clone(),
        };
        let notes = vec![
            String::from(
                "This plan proves the MLX recipe planner now seeds the live adapter-cluster run graph instead of stopping at a notebook-like planning artifact.",
            ),
            String::from(
                "The same plan consumes the workflow dataset report and publish config to produce deterministic dataset slices, contributor assignments, and upload expectations through the existing adapter window substrate.",
            ),
            String::from(
                "The first swarm lane remains one train architecture because contributor selection and window planning still flow through AdapterTrainingClusterCoordinator and the existing contributor-set revision path.",
            ),
        ];
        let mut plan = FirstSwarmLiveWorkflowPlan {
            schema_version: 1,
            scope_window: String::from(FIRST_SWARM_LIVE_WORKFLOW_PLAN_SCOPE_WINDOW),
            run_family_id: swarm_contract.run_family_id,
            swarm_contract_digest: swarm_contract.contract_digest,
            recipe_plan,
            dataset_report: dataset.report.clone(),
            publish_config: publish.clone(),
            capability_policy,
            membership_receipt,
            window_plan: window.plan,
            contributor_assignments,
            publish_expectation,
            claim_boundary: String::from(
                "This plan binds one MLX LoRA recipe, one MLX workflow dataset artifact, and one local snapshot publish config into the existing live adapter-cluster planner for the first swarm lane. It does not claim a second trainer runtime, full-model mixed-backend all-reduce, or automatic publication without validator, replay, and merge truth.",
            ),
            notes,
            plan_digest: String::new(),
        };
        plan.plan_digest = plan.stable_digest();
        Ok(plan)
    }
}

/// Returns the canonical first swarm synthetic SFT dataset spec.
#[must_use]
pub fn first_swarm_synthetic_sft_spec() -> MlxSyntheticSftDatasetSpec {
    let contract = first_swarm_run_contract();
    MlxSyntheticSftDatasetSpec {
        workflow_id: String::from("swarm.workflow.synthetic_sft"),
        dataset: contract.dataset.dataset_key.clone(),
        display_name: String::from("First Swarm Open Adapter SFT"),
        tokenizer: contract.dataset.tokenizer.clone(),
        context_window_tokens: Some(512),
        splits: vec![
            MlxSyntheticSftSplit {
                split_name: String::from("train"),
                kind: psionic_data::DatasetSplitKind::Train,
                samples: vec![
                    swarm_sample(
                        "swarm-train-001",
                        "Bind the Mac contributor into the trusted-LAN swarm lane.",
                        "Keep the MLX contributor bound to one explicit dataset slice and one upload expectation.",
                        &["swarm", "mlx"],
                    ),
                    swarm_sample(
                        "swarm-train-002",
                        "Bind the Linux contributor into the trusted-LAN swarm lane.",
                        "Keep the CUDA contributor bound to one explicit dataset slice and one upload expectation.",
                        &["swarm", "cuda"],
                    ),
                    swarm_sample(
                        "swarm-train-003",
                        "What must the validator check before aggregation?",
                        "It must keep replay, dataset, and contributor receipts explicit before accepting the window.",
                        &["validator"],
                    ),
                    swarm_sample(
                        "swarm-train-004",
                        "What is the publish boundary for the first swarm run?",
                        "Only a local snapshot may publish after accepted mergeable outputs exist.",
                        &["publish"],
                    ),
                ],
            },
            MlxSyntheticSftSplit {
                split_name: String::from("validation"),
                kind: psionic_data::DatasetSplitKind::Validation,
                samples: vec![
                    swarm_sample(
                        "swarm-val-001",
                        "What happens when a contributor upload disagrees with the manifest?",
                        "The lane must refuse promotion and keep the disagreement explicit.",
                        &["validation"],
                    ),
                    swarm_sample(
                        "swarm-val-002",
                        "What happens when a worker stops heartbeating?",
                        "The coordinator records stale-worker truth and replans instead of hiding the loss.",
                        &["validation", "stale-worker"],
                    ),
                ],
            },
        ],
    }
}

/// Returns the canonical recipe config for the first swarm live planner.
pub fn first_swarm_recipe_config(
    run_id: impl Into<String>,
    cluster_id: impl Into<String>,
) -> Result<MlxRecipeConfig, MlxWorkflowError> {
    let receipt_contract = first_swarm_open_adapter_receipt_contract();
    Ok(MlxRecipeConfig::new(
        run_id,
        cluster_id,
        "swarm.open_adapter",
        EnvironmentPackageKey::new("env.swarm.local_open_adapter", "2026.03.24"),
        MlxRecipeMethod::Lora,
    )?
    .with_adapter(MlxAdapterRecipe {
        method: MlxRecipeMethod::Lora,
        rank: receipt_contract.lora_rank,
        alpha: receipt_contract.lora_alpha,
        quantization: None,
    }))
}

/// Returns the canonical publish config for the first swarm live planner.
#[must_use]
pub fn first_swarm_publish_config() -> MlxPublishConfig {
    MlxPublishConfig {
        publish_id: String::from("first-swarm-local-snapshot"),
        target: MlxPublishTarget::HuggingFaceSnapshot,
        repo_id: String::from("openagents/swarm-local-open-adapter"),
    }
}

/// Returns one deterministic sample cluster state for the first swarm live planner.
#[must_use]
pub fn sample_first_swarm_live_cluster_state() -> ClusterState {
    let swarm_contract = first_swarm_run_contract();
    let cluster_id = ClusterId::new(
        &ClusterNamespace::new(swarm_contract.cluster_namespace.as_str()),
        &AdmissionToken::new("swarm-local-shared-secret"),
    );
    let mut snapshot = ClusterSnapshot::new(cluster_id.clone());
    snapshot.memberships = BTreeMap::from([
        (
            NodeId::new("swarm-mac-a"),
            ClusterMembershipRecord::new(
                ClusterNodeIdentity {
                    cluster_id: cluster_id.clone(),
                    node_id: NodeId::new("swarm-mac-a"),
                    node_epoch: NodeEpoch::initial(),
                    role: NodeRole::Mixed,
                    auth_public_key: String::from("swarm-mac-a-pk"),
                    attestation: None,
                },
                Some(std::net::SocketAddr::new(
                    std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
                    34_100,
                )),
                ClusterMembershipStatus::Ready,
            ),
        ),
        (
            NodeId::new("swarm-linux-4080-a"),
            ClusterMembershipRecord::new(
                ClusterNodeIdentity {
                    cluster_id: cluster_id.clone(),
                    node_id: NodeId::new("swarm-linux-4080-a"),
                    node_epoch: NodeEpoch::initial(),
                    role: NodeRole::ExecutorOnly,
                    auth_public_key: String::from("swarm-linux-4080-a-pk"),
                    attestation: None,
                },
                Some(std::net::SocketAddr::new(
                    std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
                    34_101,
                )),
                ClusterMembershipStatus::Ready,
            ),
        ),
    ]);
    snapshot.telemetry = BTreeMap::from([
        (
            NodeId::new("swarm-mac-a"),
            ClusterNodeTelemetry::new(NodeId::new("swarm-mac-a"))
                .with_memory(Some(24 * 1024 * 1024 * 1024), Some(24 * 1024 * 1024 * 1024))
                .with_accelerator_count(1)
                .with_backend_readiness(
                    String::from(OPEN_ADAPTER_MLX_METAL_BACKEND_LABEL),
                    ClusterBackendReadinessStatus::Ready,
                )
                .with_stability_posture(ClusterStabilityPosture::Stable),
        ),
        (
            NodeId::new("swarm-linux-4080-a"),
            ClusterNodeTelemetry::new(NodeId::new("swarm-linux-4080-a"))
                .with_memory(Some(20 * 1024 * 1024 * 1024), Some(20 * 1024 * 1024 * 1024))
                .with_accelerator_count(1)
                .with_backend_readiness(
                    String::from(OPEN_ADAPTER_CUDA_BACKEND_LABEL),
                    ClusterBackendReadinessStatus::Ready,
                )
                .with_stability_posture(ClusterStabilityPosture::Stable),
        ),
    ]);
    ClusterState::from_snapshot(snapshot)
}

/// Writes the canonical first swarm live workflow plan to one JSON path.
pub fn write_first_swarm_live_workflow_plan(
    output_path: impl AsRef<Path>,
) -> Result<FirstSwarmLiveWorkflowPlan, MlxWorkflowError> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| MlxWorkflowError::Io {
            path: parent.display().to_string(),
            message: error.to_string(),
        })?;
    }
    let workspace = MlxWorkflowWorkspace::default();
    let cluster_state = sample_first_swarm_live_cluster_state();
    let recipe =
        first_swarm_recipe_config("first-swarm-live-plan", cluster_state.cluster_id().as_str())?;
    let dataset = workspace.build_first_swarm_synthetic_sft_dataset()?;
    let publish = first_swarm_publish_config();
    let plan = workspace.plan_first_swarm_live_adapter_cluster(
        &recipe,
        &dataset,
        &publish,
        &cluster_state,
        1_774_409_200_000,
        1_774_409_201_000,
    )?;
    let encoded =
        serde_json::to_string_pretty(&plan).map_err(|error| MlxWorkflowError::Serialization {
            context: "first swarm live workflow plan export",
            message: error.to_string(),
        })?;
    fs::write(output_path, format!("{encoded}\n")).map_err(|error| MlxWorkflowError::Io {
        path: output_path.display().to_string(),
        message: error.to_string(),
    })?;
    Ok(plan)
}

/// Writes the retained first-swarm local snapshot publication proof into one root directory.
pub fn write_first_swarm_local_snapshot_publication(
    output_root: impl AsRef<Path>,
) -> Result<FirstSwarmLocalSnapshotPublicationReport, MlxWorkflowError> {
    let output_root = output_root.as_ref();
    fs::create_dir_all(output_root).map_err(|error| MlxWorkflowError::Io {
        path: output_root.display().to_string(),
        message: error.to_string(),
    })?;

    let workspace = MlxWorkflowWorkspace::default();
    let cluster_state = sample_first_swarm_live_cluster_state();
    let recipe = first_swarm_recipe_config(
        "first-swarm-local-publication",
        cluster_state.cluster_id().as_str(),
    )?;
    let dataset = workspace.build_first_swarm_synthetic_sft_dataset()?;
    let publish = first_swarm_publish_config();
    let plan = workspace.plan_first_swarm_live_adapter_cluster(
        &recipe,
        &dataset,
        &publish,
        &cluster_state,
        1_774_409_200_000,
        1_774_409_201_000,
    )?;

    let (base_bundle, tuned_bundle) = first_swarm_publication_reference_bundles()?;
    let merged = workspace.merge_adapter(
        &crate::MlxAdapterMergeConfig {
            merge_id: String::from("first-swarm-local-snapshot-merge"),
            adapter_id: String::from("first-swarm-open-adapter"),
        },
        &base_bundle,
        &tuned_bundle,
    )?;

    let snapshot_relative_root = plan
        .publish_expectation
        .expected_local_snapshot_directory
        .clone();
    let snapshot_root = output_root.join(&snapshot_relative_root);
    let manifest = workspace.publish_bundle(&publish, &merged.merged_bundle, &snapshot_root)?;

    let mut report = FirstSwarmLocalSnapshotPublicationReport {
        schema_version: String::from("swarm.first_local_snapshot_publication_report.v1"),
        scope_window: String::from(FIRST_SWARM_LOCAL_SNAPSHOT_PUBLICATION_SCOPE_WINDOW),
        run_family_id: plan.run_family_id.clone(),
        workflow_plan_digest: plan.plan_digest.clone(),
        publish_id: publish.publish_id.clone(),
        target: publish.target,
        repo_id: publish.repo_id.clone(),
        expected_local_snapshot_directory: plan
            .publish_expectation
            .expected_local_snapshot_directory
            .clone(),
        published_snapshot_root: snapshot_relative_root,
        merge_report_digest: merged.report.report_digest.clone(),
        publish_manifest_digest: manifest.manifest_digest.clone(),
        merged_state_dict_digest: merged.merged_bundle.state_dict.digest.clone(),
        merged_artifact_digest: manifest.source_artifact_receipt.artifact_digest.clone(),
        base_state_dict_digest: base_bundle.state_dict.digest.clone(),
        tuned_state_dict_digest: tuned_bundle.state_dict.digest.clone(),
        published_files: manifest.files.clone(),
        notes: vec![
            String::from(
                "This proof uses the exact first-swarm publish target and the existing `psionic-mlx-workflows::MlxWorkflowWorkspace::publish_bundle` surface to retain one real local Hugging Face-style snapshot directory.",
            ),
            String::from(
                "The published bundle is deterministic and machine-checkable, but it is a bounded publication proof built from a retained portable merge pair rather than the already-closed trusted-LAN live run.",
            ),
            String::from(
                "The retained first-swarm live bundle still truthfully stays `publish_disposition=refused` because that real mixed-hardware run ended at merge truth plus a promotion hold instead of an earned promoted snapshot.",
            ),
        ],
        claim_boundary: String::from(
            "This report proves that the first-swarm lane now has one truthful retained local snapshot publication path for the frozen `first-swarm-local-snapshot` target. It does not claim that the retained mixed-hardware live run earned publication, that publication is automatic after live promotion, or that any remote/public registry publish has been completed.",
        ),
        report_digest: String::new(),
    };
    report.report_digest = report.stable_digest();

    let report_path = first_swarm_local_snapshot_publication_report_path(output_root);
    let encoded =
        serde_json::to_string_pretty(&report).map_err(|error| MlxWorkflowError::Serialization {
            context: "first swarm local snapshot publication report export",
            message: error.to_string(),
        })?;
    fs::write(&report_path, format!("{encoded}\n")).map_err(|error| MlxWorkflowError::Io {
        path: report_path.display().to_string(),
        message: error.to_string(),
    })?;
    Ok(report)
}

fn plan_role_assignments(
    swarm_contract: &psionic_train::FirstSwarmRunContract,
    membership_receipt: &AdapterClusterMembershipReceipt,
    dataset: &MlxSyntheticDatasetArtifact,
) -> Result<Vec<PlannedRoleAssignment>, MlxWorkflowError> {
    let train_split = dataset
        .report
        .splits
        .iter()
        .find(|split| split.split_name == "train")
        .ok_or_else(|| MlxWorkflowError::FirstSwarmPlan {
            message: String::from(
                "first swarm workflow dataset report is missing the `train` split",
            ),
        })?;
    let train_sample_ids = split_sample_ids(dataset, "train")?;
    if train_sample_ids.len() < swarm_contract.node_roles.len() {
        return Err(MlxWorkflowError::FirstSwarmPlan {
            message: format!(
                "first swarm workflow train split only carries {} sample(s) for {} contributor role(s)",
                train_sample_ids.len(),
                swarm_contract.node_roles.len()
            ),
        });
    }
    let mut used_nodes = BTreeSet::new();
    let mut assignments = Vec::with_capacity(swarm_contract.node_roles.len());
    for (index, role) in swarm_contract.node_roles.iter().enumerate() {
        let status = select_status_for_role(role, membership_receipt, &used_nodes)?;
        used_nodes.insert(status.node_id.clone());
        let sample_ids = evenly_partitioned_sample_ids(
            train_sample_ids.as_slice(),
            index,
            swarm_contract.node_roles.len(),
        );
        assignments.push(PlannedRoleAssignment {
            role_id: role.role_id.clone(),
            platform: role.platform.clone(),
            contributor_node_id: status.node_id.clone(),
            matched_backend_label: status.matched_backend_label.clone().ok_or_else(|| {
                MlxWorkflowError::FirstSwarmPlan {
                    message: format!(
                        "selected role `{}` did not carry a matched backend label",
                        role.role_id
                    ),
                }
            })?,
            validator_eligible: role.validator_eligible,
            aggregation_eligible: role.aggregation_eligible,
            priority_bps: status.priority_bps,
            reliability_bps: status.reliability_bps,
            dataset_slice: AdapterDatasetSliceIdentity::new(
                dataset.report.dataset_storage_key.clone(),
                train_split.split_name.clone(),
                format!("{}-{}", sanitize_label(role.role_id.as_str()), index + 1),
                stable_first_swarm_slice_digest(
                    dataset.report.dataset_storage_key.as_str(),
                    train_split.split_name.as_str(),
                    role.role_id.as_str(),
                    sample_ids.as_slice(),
                    train_split.object_digest.as_str(),
                ),
            )
            .map_err(|error| MlxWorkflowError::FirstSwarmPlan {
                message: error.to_string(),
            })?,
            source_sample_ids: sample_ids,
            source_split_object_digest: train_split.object_digest.clone(),
        });
    }
    Ok(assignments)
}

fn select_status_for_role<'a>(
    role: &psionic_train::FirstSwarmNodeRoleContract,
    membership_receipt: &'a AdapterClusterMembershipReceipt,
    used_nodes: &BTreeSet<String>,
) -> Result<&'a psionic_train::AdapterClusterContributorStatus, MlxWorkflowError> {
    let mut candidates = membership_receipt
        .contributor_statuses
        .iter()
        .filter(|status| {
            status.eligibility == AdapterContributorEligibility::Eligible
                && status.matched_backend_label.as_deref() == Some(role.backend_label.as_str())
                && !used_nodes.contains(&status.node_id)
                && role_matches_cluster_role(role, status.role)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .priority_bps
            .cmp(&left.priority_bps)
            .then_with(|| right.reliability_bps.cmp(&left.reliability_bps))
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| MlxWorkflowError::FirstSwarmPlan {
            message: format!(
                "no eligible contributor satisfied role `{}` on backend `{}`",
                role.role_id, role.backend_label
            ),
        })
}

fn role_matches_cluster_role(
    role: &psionic_train::FirstSwarmNodeRoleContract,
    cluster_role: Option<NodeRole>,
) -> bool {
    match (
        role.validator_eligible || role.aggregation_eligible,
        cluster_role,
    ) {
        (true, Some(NodeRole::Mixed)) => true,
        (true, _) => false,
        (false, Some(NodeRole::CoordinatorOnly)) | (false, None) => false,
        (false, Some(NodeRole::ExecutorOnly | NodeRole::Mixed)) => true,
    }
}

fn validate_first_swarm_recipe_plan(recipe_plan: &MlxRecipePlan) -> Result<(), MlxWorkflowError> {
    let contract = first_swarm_run_contract();
    let receipt_contract = first_swarm_open_adapter_receipt_contract();
    if recipe_plan.method != MlxRecipeMethod::Lora {
        return Err(MlxWorkflowError::FirstSwarmPlan {
            message: format!(
                "first swarm live planner requires `lora`, but the recipe plan used `{}`",
                recipe_plan.method.as_str()
            ),
        });
    }
    let adapter_execution = recipe_plan.adapter_execution.as_ref().ok_or_else(|| {
        MlxWorkflowError::FirstSwarmPlan {
            message: String::from(
                "first swarm live planner requires an adapter execution plan from the MLX recipe",
            ),
        }
    })?;
    if adapter_execution.adapter_family != contract.adapter_family {
        return Err(MlxWorkflowError::FirstSwarmPlan {
            message: format!(
                "first swarm live planner expected adapter family `{}` but the MLX recipe planned `{}`",
                contract.adapter_family, adapter_execution.adapter_family
            ),
        });
    }
    if adapter_execution.adapter_format != contract.adapter_format {
        return Err(MlxWorkflowError::FirstSwarmPlan {
            message: format!(
                "first swarm live planner expected adapter format `{}` but the MLX recipe planned `{}`",
                contract.adapter_format, adapter_execution.adapter_format
            ),
        });
    }
    if adapter_execution.rank != receipt_contract.lora_rank {
        return Err(MlxWorkflowError::FirstSwarmPlan {
            message: format!(
                "first swarm live planner expected LoRA rank `{}` but the MLX recipe planned `{}`",
                receipt_contract.lora_rank, adapter_execution.rank
            ),
        });
    }
    if adapter_execution.alpha != format!("{:.4}", receipt_contract.lora_alpha) {
        return Err(MlxWorkflowError::FirstSwarmPlan {
            message: format!(
                "first swarm live planner expected LoRA alpha `{:.4}` but the MLX recipe planned `{}`",
                receipt_contract.lora_alpha, adapter_execution.alpha
            ),
        });
    }
    if adapter_execution.quantization.is_some() {
        return Err(MlxWorkflowError::FirstSwarmPlan {
            message: String::from(
                "first swarm live planner refuses quantized adapter recipes because the shared contributor receipt contract is f32-only",
            ),
        });
    }
    Ok(())
}

fn validate_first_swarm_dataset(
    dataset: &MlxSyntheticDatasetArtifact,
) -> Result<(), MlxWorkflowError> {
    let contract = first_swarm_run_contract();
    if dataset.report.dataset_storage_key != contract.dataset.dataset_storage_key {
        return Err(MlxWorkflowError::FirstSwarmPlan {
            message: format!(
                "first swarm live planner expected dataset storage key `{}` but the workflow artifact reported `{}`",
                contract.dataset.dataset_storage_key, dataset.report.dataset_storage_key
            ),
        });
    }
    if dataset.dataset_manifest.tokenizer.stable_digest()
        != contract.dataset.tokenizer.stable_digest()
    {
        return Err(MlxWorkflowError::FirstSwarmPlan {
            message: String::from(
                "first swarm live planner observed tokenizer drift between the workflow dataset artifact and the frozen swarm contract",
            ),
        });
    }
    for contract_split in &contract.dataset.splits {
        let observed_sample_ids = split_sample_ids(dataset, contract_split.split_name.as_str())?;
        if observed_sample_ids != contract_split.sample_ids {
            return Err(MlxWorkflowError::FirstSwarmPlan {
                message: format!(
                    "first swarm live planner observed sample-id drift for split `{}`",
                    contract_split.split_name
                ),
            });
        }
    }
    Ok(())
}

fn first_swarm_capability_policy() -> Result<AdapterContributorCapabilityPolicy, MlxWorkflowError> {
    let contract = first_swarm_run_contract();
    let mac_role = contract
        .node_roles
        .iter()
        .find(|role| role.backend_label == OPEN_ADAPTER_MLX_METAL_BACKEND_LABEL)
        .ok_or_else(|| MlxWorkflowError::FirstSwarmPlan {
            message: String::from("first swarm contract is missing the Mac MLX role"),
        })?;
    let linux_role = contract
        .node_roles
        .iter()
        .find(|role| role.backend_label == OPEN_ADAPTER_CUDA_BACKEND_LABEL)
        .ok_or_else(|| MlxWorkflowError::FirstSwarmPlan {
            message: String::from("first swarm contract is missing the Linux CUDA role"),
        })?;
    Ok(AdapterContributorCapabilityPolicy {
        backend_label: mac_role.backend_label.clone(),
        minimum_free_memory_bytes: mac_role.minimum_free_memory_bytes,
        require_accelerator: true,
        allow_degraded_backend: false,
        additional_backend_capabilities: vec![AdapterContributorBackendCapability::new(
            linux_role.backend_label.clone(),
            linux_role.minimum_free_memory_bytes,
            true,
            false,
        )],
        allow_flaky_nodes: false,
    })
}

fn first_swarm_adapter_target_identity(
    recipe_plan: &MlxRecipePlan,
) -> Result<AdapterTargetIdentity, MlxWorkflowError> {
    let receipt_contract = first_swarm_open_adapter_receipt_contract();
    let adapter_execution =
        recipe_plan
            .adapter_execution
            .as_ref()
            .ok_or_else(|| MlxWorkflowError::FirstSwarmPlan {
                message: String::from(
                    "first swarm adapter target requires one MLX adapter execution plan",
                ),
            })?;
    AdapterTargetIdentity::new(
        adapter_execution.target_id.clone(),
        adapter_execution.adapter_family.clone(),
        format!(
            "{}@{}",
            receipt_contract.base_model_id, receipt_contract.base_model_revision
        ),
        adapter_execution.adapter_format.clone(),
    )
    .map_err(|error| MlxWorkflowError::FirstSwarmPlan {
        message: error.to_string(),
    })
}

fn first_swarm_checkpoint_pointer(
    run_id: &str,
    cluster_id: &str,
    checkpoint_family: &str,
    planned_at_ms: u64,
) -> Result<CheckpointPointer, MlxWorkflowError> {
    let checkpoint = TrainingCheckpointReference::new(
        checkpoint_family,
        format!("stream://swarm/{run_id}/policy"),
        format!("manifest://swarm/{run_id}/policy"),
        format!("object://swarm/{run_id}/policy"),
        "swarm-mac-a",
        1,
        stable_string_digest(["cluster", cluster_id, run_id].as_slice()),
        stable_string_digest(["topology", run_id, cluster_id].as_slice()),
        planned_at_ms,
    )
    .with_checkpoint_ref(format!("checkpoint://swarm/{run_id}/policy"))
    .with_step(12);
    CheckpointPointer::new(
        CheckpointScopeBinding::new(CheckpointScopeKind::Window, format!("{run_id}-window-1")),
        checkpoint_family,
        checkpoint,
        stable_string_digest(["manifest", run_id, checkpoint_family].as_slice()),
        planned_at_ms,
    )
    .map_err(|error| MlxWorkflowError::FirstSwarmPlan {
        message: error.to_string(),
    })
}

fn evenly_partitioned_sample_ids(
    sample_ids: &[String],
    partition_index: usize,
    partition_count: usize,
) -> Vec<String> {
    let start = partition_index * sample_ids.len() / partition_count;
    let end = (partition_index + 1) * sample_ids.len() / partition_count;
    sample_ids[start..end].to_vec()
}

fn split_sample_ids(
    dataset: &MlxSyntheticDatasetArtifact,
    split_name: &str,
) -> Result<Vec<String>, MlxWorkflowError> {
    let split = dataset
        .split_artifacts
        .iter()
        .find(|split| split.split_name == split_name)
        .ok_or_else(|| MlxWorkflowError::FirstSwarmPlan {
            message: format!("first swarm workflow artifact is missing split `{split_name}`"),
        })?;
    std::str::from_utf8(split.jsonl_bytes.as_slice())
        .map_err(|error| MlxWorkflowError::Serialization {
            context: "first swarm workflow sample-id decode",
            message: error.to_string(),
        })?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let value: Value = serde_json::from_str(line).map_err(|error| MlxWorkflowError::Serialization {
                context: "first swarm workflow sample-id parse",
                message: error.to_string(),
            })?;
            value
                .get("sample_id")
                .and_then(Value::as_str)
                .map(String::from)
                .ok_or_else(|| MlxWorkflowError::FirstSwarmPlan {
                    message: format!(
                        "first swarm workflow split `{split_name}` carried one record without `sample_id`"
                    ),
                })
        })
        .collect()
}

fn stable_first_swarm_slice_digest(
    dataset_storage_key: &str,
    split_name: &str,
    role_id: &str,
    sample_ids: &[String],
    source_split_object_digest: &str,
) -> String {
    let mut hasher = Sha256::new();
    for part in [
        "first_swarm_dataset_slice",
        dataset_storage_key,
        split_name,
        role_id,
        source_split_object_digest,
    ] {
        hasher.update(part.as_bytes());
        hasher.update(b"|");
    }
    for sample_id in sample_ids {
        hasher.update(sample_id.as_bytes());
        hasher.update(b"|");
    }
    format!("{:x}", hasher.finalize())
}

fn stable_expected_upload_manifest_digest(
    publish_id: &str,
    repo_id: &str,
    contribution_id: &str,
    dataset_slice_digest: &str,
) -> String {
    let mut hasher = Sha256::new();
    for part in [
        "first_swarm_expected_upload_manifest",
        publish_id,
        repo_id,
        contribution_id,
        dataset_slice_digest,
    ] {
        hasher.update(part.as_bytes());
        hasher.update(b"|");
    }
    format!("{:x}", hasher.finalize())
}

fn stable_string_digest(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.as_bytes());
        hasher.update(b"|");
    }
    format!("{:x}", hasher.finalize())
}

fn stable_digest(label: &[u8], value: &impl Serialize) -> String {
    let mut hasher = Sha256::new();
    hasher.update(label);
    hasher.update(
        serde_json::to_vec(value).expect("first swarm live workflow plan should serialize"),
    );
    format!("{:x}", hasher.finalize())
}

fn sanitize_label(label: &str) -> String {
    label
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch,
            _ => '_',
        })
        .collect()
}

fn first_swarm_local_snapshot_publication_report_path(output_root: &Path) -> PathBuf {
    output_root.join("first_swarm_local_snapshot_publication_v1.json")
}

fn first_swarm_publication_reference_bundle() -> Result<PortableModelBundle, MlxWorkflowError> {
    let receipt_contract = first_swarm_open_adapter_receipt_contract();
    let map_training_error = |detail: String| MlxWorkflowError::FirstSwarmPlan {
        message: format!(
            "first swarm publication proof could not build the retained portable bundle: {detail}"
        ),
    };
    let mut embedding = TrainingParameterGroupState::new(
        "embedding",
        TrainingParameterClass::Embedding,
        TrainingTensorBuffer::from_f32(
            "embedding",
            psionic_core::TensorSpec::new(Shape::new(vec![2, 2]), DType::F32, Device::cpu()),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .map_err(|error| map_training_error(error.to_string()))?,
        TrainingOptimizerConfig::sgd(0.05).with_momentum(0.8),
        TrainingOptimizerResidencyPolicy::new(
            OptimizerStateResidency::DeviceResident,
            OptimizerStateResidency::HostResident,
        ),
    )
    .map_err(|error| map_training_error(error.to_string()))?;
    embedding.optimizer_state = TrainingOptimizerState::Sgd {
        momentum_buffer: Some(vec![0.01, 0.02, 0.03, 0.04]),
    };
    embedding.optimizer_residency = OptimizerStateResidency::DeviceResident;
    embedding.applied_steps = 2;

    let mut decoder_head = TrainingParameterGroupState::new(
        "decoder.head",
        TrainingParameterClass::Head,
        TrainingTensorBuffer::from_f32(
            "decoder.head",
            psionic_core::TensorSpec::new(Shape::new(vec![2]), DType::F32, Device::cpu()),
            vec![0.5, -0.5],
        )
        .map_err(|error| map_training_error(error.to_string()))?,
        TrainingOptimizerConfig::adamw(0.01, 0.9, 0.999, 1e-8).with_weight_decay(0.01),
        TrainingOptimizerResidencyPolicy::new(
            OptimizerStateResidency::DeviceResident,
            OptimizerStateResidency::Offloaded,
        ),
    )
    .map_err(|error| map_training_error(error.to_string()))?;
    decoder_head.optimizer_state = TrainingOptimizerState::AdamW {
        first_moment: vec![0.01, -0.02],
        second_moment: vec![0.03, 0.04],
    };
    decoder_head.optimizer_residency = OptimizerStateResidency::Offloaded;
    decoder_head.applied_steps = 4;

    Ok(PortableModelBundle::from_training_groups(
        receipt_contract.adapter_family,
        receipt_contract.base_model_revision,
        String::from("swarm-local-open-adapter"),
        Some(String::from(
            "checkpoint://swarm/first_swarm_local_snapshot_publication",
        )),
        &[embedding, decoder_head],
        PortableTokenizerBinding::new(
            first_swarm_tokenizer_digest(),
            PortableTokenizerAssetFormat::TokenizerJson,
            String::from("gpt-oss-20b@swarm-local-v1"),
        )
        .with_special_tokens(Some(1), vec![2], Some(0), Some(3), true, false),
        Some(String::from("swarm-open-adapter-template-v1")),
    )?)
}

fn first_swarm_publication_reference_bundles()
-> Result<(PortableModelBundle, PortableModelBundle), MlxWorkflowError> {
    let base = first_swarm_publication_reference_bundle()?;
    let mut tuned = base.clone();
    let TensorData::F32(values) = &mut tuned
        .state_dict
        .tensors
        .get_mut("model.decoder.head.parameter")
        .ok_or_else(|| MlxWorkflowError::FirstSwarmPlan {
            message: String::from(
                "first swarm publication proof could not find the decoder-head parameter tensor",
            ),
        })?
        .data
    else {
        return Err(MlxWorkflowError::FirstSwarmPlan {
            message: String::from(
                "first swarm publication proof expected a dense f32 decoder-head tensor",
            ),
        });
    };
    values[0] += 0.25;
    values[1] -= 0.5;
    tuned.state_dict = PortableModelStateDict::new(
        tuned.state_dict.model_family.clone(),
        tuned.state_dict.revision.clone(),
        tuned.state_dict.checkpoint_family.clone(),
        tuned.state_dict.checkpoint_ref.clone(),
        tuned.state_dict.source_format,
        tuned.state_dict.groups.clone(),
        tuned.state_dict.tensors.clone(),
    )?;
    Ok((base, tuned))
}

fn swarm_sample(
    sample_id: &str,
    prompt: &str,
    response: &str,
    tags: &[&str],
) -> MlxSyntheticSftSample {
    MlxSyntheticSftSample {
        sample_id: String::from(sample_id),
        prompt: String::from(prompt),
        response: String::from(response),
        tags: tags.iter().map(|tag| String::from(*tag)).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FIRST_SWARM_LIVE_WORKFLOW_PLAN_FIXTURE_PATH,
        FIRST_SWARM_LOCAL_SNAPSHOT_PUBLICATION_REPORT_FIXTURE_PATH, first_swarm_publish_config,
        first_swarm_recipe_config, first_swarm_synthetic_sft_spec,
        sample_first_swarm_live_cluster_state, write_first_swarm_live_workflow_plan,
        write_first_swarm_local_snapshot_publication,
    };
    use crate::{MlxWorkflowError, MlxWorkflowWorkspace};

    #[test]
    fn first_swarm_live_plan_consumes_recipe_dataset_and_publish_truth()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace = MlxWorkflowWorkspace::default();
        let cluster_state = sample_first_swarm_live_cluster_state();
        let recipe = first_swarm_recipe_config(
            "first-swarm-live-plan-test",
            cluster_state.cluster_id().as_str(),
        )?;
        let dataset = workspace.build_synthetic_sft_dataset(&first_swarm_synthetic_sft_spec())?;
        let publish = first_swarm_publish_config();
        let plan = workspace.plan_first_swarm_live_adapter_cluster(
            &recipe,
            &dataset,
            &publish,
            &cluster_state,
            1_000,
            1_010,
        )?;

        assert_eq!(plan.membership_receipt.eligible_node_ids.len(), 2);
        assert_eq!(
            plan.window_plan.selected_node_ids,
            vec![
                String::from("swarm-mac-a"),
                String::from("swarm-linux-4080-a")
            ]
        );
        assert_eq!(plan.contributor_assignments.len(), 2);
        assert_eq!(
            plan.contributor_assignments[0].matched_backend_label,
            psionic_train::OPEN_ADAPTER_MLX_METAL_BACKEND_LABEL
        );
        assert_eq!(
            plan.contributor_assignments[1].matched_backend_label,
            psionic_train::OPEN_ADAPTER_CUDA_BACKEND_LABEL
        );
        Ok(())
    }

    #[test]
    fn first_swarm_live_plan_refuses_quantized_recipe() -> Result<(), Box<dyn std::error::Error>> {
        let cluster_state = sample_first_swarm_live_cluster_state();
        let workspace = MlxWorkflowWorkspace::default();
        let dataset = workspace.build_synthetic_sft_dataset(&first_swarm_synthetic_sft_spec())?;
        let publish = first_swarm_publish_config();
        let recipe = psionic_mlx_recipes::MlxRecipeConfig::new(
            "first-swarm-bad-plan",
            cluster_state.cluster_id().as_str(),
            "swarm.open_adapter",
            psionic_environments::EnvironmentPackageKey::new(
                "env.swarm.local_open_adapter",
                "2026.03.24",
            ),
            psionic_mlx_recipes::MlxRecipeMethod::Qlora,
        )?
        .with_adapter(psionic_mlx_recipes::MlxAdapterRecipe {
            method: psionic_mlx_recipes::MlxRecipeMethod::Qlora,
            rank: 8,
            alpha: 16.0,
            quantization: Some(String::from("q4_k")),
        });
        let error = workspace
            .plan_first_swarm_live_adapter_cluster(
                &recipe,
                &dataset,
                &publish,
                &cluster_state,
                1_000,
                1_010,
            )
            .expect_err("quantized recipe should refuse");
        assert!(matches!(error, MlxWorkflowError::FirstSwarmPlan { .. }));
        Ok(())
    }

    #[test]
    fn retained_first_swarm_live_plan_fixture_matches_builder()
    -> Result<(), Box<dyn std::error::Error>> {
        let retained = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../")
                .join(FIRST_SWARM_LIVE_WORKFLOW_PLAN_FIXTURE_PATH),
        )?;
        let retained: super::FirstSwarmLiveWorkflowPlan = serde_json::from_str(&retained)?;
        let temp = tempfile::tempdir()?;
        let rebuilt = write_first_swarm_live_workflow_plan(
            temp.path().join("first_swarm_live_workflow_plan.json"),
        )?;
        assert_eq!(retained, rebuilt);
        Ok(())
    }

    #[test]
    fn first_swarm_local_snapshot_publication_proof_writes_expected_snapshot()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let report = write_first_swarm_local_snapshot_publication(temp.path())?;
        let snapshot_root = temp.path().join(&report.published_snapshot_root);
        assert!(snapshot_root.join("model.safetensors").exists());
        assert!(snapshot_root.join("publish_manifest.json").exists());
        assert_eq!(
            report.publish_id,
            String::from("first-swarm-local-snapshot")
        );
        assert_eq!(
            report.published_snapshot_root,
            report.expected_local_snapshot_directory
        );
        Ok(())
    }

    #[test]
    fn retained_first_swarm_local_snapshot_publication_fixture_matches_builder()
    -> Result<(), Box<dyn std::error::Error>> {
        let retained = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../")
                .join(FIRST_SWARM_LOCAL_SNAPSHOT_PUBLICATION_REPORT_FIXTURE_PATH),
        )?;
        let retained: super::FirstSwarmLocalSnapshotPublicationReport =
            serde_json::from_str(&retained)?;
        let temp = tempfile::tempdir()?;
        let rebuilt = write_first_swarm_local_snapshot_publication(temp.path())?;
        assert_eq!(retained, rebuilt);
        Ok(())
    }
}
