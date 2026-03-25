use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    canonical_cross_provider_compute_source_contracts, cross_provider_training_program_manifest,
    CrossProviderComputeSourceContract, CrossProviderComputeSourceContractError,
    CrossProviderExecutionClass, CrossProviderTrainingProgramManifest,
    CrossProviderTrainingProgramManifestError,
};

/// Stable schema version for the provider-neutral training execution evidence bundle.
pub const TRAINING_EXECUTION_EVIDENCE_BUNDLE_SCHEMA_VERSION: &str =
    "psionic.training_execution_evidence_bundle.v1";
/// Stable fixture path for the canonical provider-neutral evidence bundle.
pub const TRAINING_EXECUTION_EVIDENCE_BUNDLE_FIXTURE_PATH: &str =
    "fixtures/training/provider_neutral_training_execution_evidence_bundle_v1.json";
/// Stable checker path for the provider-neutral evidence bundle.
pub const TRAINING_EXECUTION_EVIDENCE_BUNDLE_CHECK_SCRIPT_PATH: &str =
    "scripts/check-training-execution-evidence-bundle.sh";
/// Stable reference doc path for the provider-neutral evidence bundle.
pub const TRAINING_EXECUTION_EVIDENCE_BUNDLE_DOC_PATH: &str =
    "docs/TRAINING_EXECUTION_EVIDENCE_REFERENCE.md";

/// Error surfaced while building, validating, or writing the evidence bundle.
#[derive(Debug, Error)]
pub enum TrainingExecutionEvidenceBundleError {
    #[error("failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("failed to create `{path}`: {error}")]
    CreateDir { path: String, error: std::io::Error },
    #[error("failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error(transparent)]
    Serialize(#[from] serde_json::Error),
    #[error(transparent)]
    ComputeSource(#[from] CrossProviderComputeSourceContractError),
    #[error(transparent)]
    ProgramManifest(#[from] CrossProviderTrainingProgramManifestError),
    #[error("training execution evidence bundle is invalid: {detail}")]
    InvalidBundle { detail: String },
}

/// Evidence posture for one retained artifact or validator outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingExecutionEvidencePosture {
    Measured,
    Derived,
    Refused,
}

/// Topology kind represented by one segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingExecutionTopologyKind {
    SingleNode,
    DenseDistributed,
    ContributorWindow,
    ValidatorOnly,
    Hybrid,
}

/// Final disposition carried by one segment or whole bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingExecutionDisposition {
    CompletedSuccess,
    DegradedSuccess,
    Refused,
    Failed,
}

/// Typed evidence reference for one launch, runtime, checkpoint, metric, visualization, or audit artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingExecutionEvidenceRef {
    /// Artifact role inside the evidence bundle.
    pub artifact_role: String,
    /// Repo-relative path or retained URI.
    pub artifact_path: String,
    /// SHA256 over the retained artifact bytes when the artifact is repo-local.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_digest: Option<String>,
    /// Whether the artifact is measured, derived, or refused proof.
    pub evidence_posture: TrainingExecutionEvidencePosture,
    /// Whether the artifact is authoritative for the fact family.
    pub authoritative: bool,
    /// Machine-legible detail.
    pub detail: String,
}

/// Typed validator outcome over one execution class.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingExecutionValidatorResult {
    /// Stable validator id.
    pub validator_id: String,
    /// Execution class this result applies to.
    pub execution_class: CrossProviderExecutionClass,
    /// Explicit result disposition.
    pub disposition: TrainingExecutionDisposition,
    /// Measured versus refused posture.
    pub evidence_posture: TrainingExecutionEvidencePosture,
    /// Machine-legible detail.
    pub detail: String,
}

/// One execution segment inside the provider-neutral final evidence bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingExecutionSegmentEvidence {
    /// Stable segment id.
    pub segment_id: String,
    /// Topology kind represented by this segment.
    pub topology_kind: TrainingExecutionTopologyKind,
    /// Source ids participating in the segment.
    pub source_ids: Vec<String>,
    /// Execution classes represented in the segment.
    pub execution_classes: Vec<CrossProviderExecutionClass>,
    /// Launch-fact references.
    pub launch_facts: Vec<TrainingExecutionEvidenceRef>,
    /// Runtime-fact references.
    pub runtime_facts: Vec<TrainingExecutionEvidenceRef>,
    /// Checkpoint-fact references.
    pub checkpoint_facts: Vec<TrainingExecutionEvidenceRef>,
    /// Metric-fact references.
    pub metric_facts: Vec<TrainingExecutionEvidenceRef>,
    /// Visualization references.
    pub visualization_refs: Vec<TrainingExecutionEvidenceRef>,
    /// Validator results over the segment.
    pub validator_results: Vec<TrainingExecutionValidatorResult>,
    /// Explicit segment disposition.
    pub segment_disposition: TrainingExecutionDisposition,
    /// Machine-legible detail.
    pub detail: String,
}

/// Final disposition for the whole run bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingExecutionFinalDisposition {
    /// Explicit final disposition.
    pub disposition: TrainingExecutionDisposition,
    /// Promotion posture for the run.
    pub promotion_outcome: String,
    /// Machine-legible detail.
    pub detail: String,
}

/// Canonical provider-neutral final evidence bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingExecutionEvidenceBundle {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable bundle id.
    pub bundle_id: String,
    /// Stable cross-provider program manifest id.
    pub program_manifest_id: String,
    /// Stable cross-provider program manifest digest.
    pub program_manifest_digest: String,
    /// Stable run id.
    pub run_id: String,
    /// Stable lane id.
    pub lane_id: String,
    /// Execution segments carried by this bundle family.
    pub segment_evidence: Vec<TrainingExecutionSegmentEvidence>,
    /// Final artifact refs retained after bundle closure.
    pub final_artifact_refs: Vec<TrainingExecutionEvidenceRef>,
    /// After-action audit or closeout refs.
    pub after_action_refs: Vec<TrainingExecutionEvidenceRef>,
    /// Explicit final disposition.
    pub final_disposition: TrainingExecutionFinalDisposition,
    /// Honest claim boundary.
    pub claim_boundary: String,
    /// Stable bundle digest.
    pub bundle_digest: String,
}

impl TrainingExecutionEvidenceBundle {
    /// Returns the stable digest for the bundle payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.bundle_digest.clear();
        stable_digest(b"psionic_training_execution_evidence_bundle|", &clone)
    }

    /// Validates the bundle against the canonical manifest and compute-source contracts.
    pub fn validate(
        &self,
        manifest: &CrossProviderTrainingProgramManifest,
        source_contracts: &[CrossProviderComputeSourceContract],
    ) -> Result<(), TrainingExecutionEvidenceBundleError> {
        if self.schema_version != TRAINING_EXECUTION_EVIDENCE_BUNDLE_SCHEMA_VERSION {
            return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                detail: format!(
                    "schema_version must stay `{}` but was `{}`",
                    TRAINING_EXECUTION_EVIDENCE_BUNDLE_SCHEMA_VERSION, self.schema_version
                ),
            });
        }
        if self.program_manifest_id != manifest.program_manifest_id {
            return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                detail: String::from("program_manifest_id drifted from the root manifest"),
            });
        }
        if self.program_manifest_digest != manifest.program_manifest_digest {
            return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                detail: String::from("program_manifest_digest drifted from the root manifest"),
            });
        }
        if self.segment_evidence.is_empty() {
            return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                detail: String::from("segment_evidence must not be empty"),
            });
        }
        let sources_by_id = source_contracts
            .iter()
            .map(|contract| (contract.source_id.as_str(), contract))
            .collect::<BTreeMap<_, _>>();
        let mut segment_ids = BTreeSet::new();
        let mut topology_coverage = BTreeSet::new();
        let mut class_coverage = BTreeSet::new();
        let mut bundle_checkpoint_facts = 0_usize;
        for segment in &self.segment_evidence {
            if !segment_ids.insert(segment.segment_id.as_str()) {
                return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                    detail: format!("duplicate segment id `{}`", segment.segment_id),
                });
            }
            if segment.source_ids.is_empty() || segment.execution_classes.is_empty() {
                return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                    detail: format!(
                        "segment `{}` must retain source_ids and execution_classes",
                        segment.segment_id
                    ),
                });
            }
            if segment.launch_facts.is_empty()
                || segment.runtime_facts.is_empty()
                || segment.metric_facts.is_empty()
                || segment.visualization_refs.is_empty()
                || segment.validator_results.is_empty()
                || segment.detail.trim().is_empty()
            {
                return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                    detail: format!(
                        "segment `{}` is missing one or more required evidence sections",
                        segment.segment_id
                    ),
                });
            }
            bundle_checkpoint_facts += segment.checkpoint_facts.len();
            topology_coverage.insert(segment.topology_kind);
            for source_id in &segment.source_ids {
                if !sources_by_id.contains_key(source_id.as_str()) {
                    return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                        detail: format!(
                            "segment `{}` references unknown source `{}`",
                            segment.segment_id, source_id
                        ),
                    });
                }
            }
            let mut unique_classes = BTreeSet::new();
            for execution_class in &segment.execution_classes {
                class_coverage.insert(*execution_class);
                unique_classes.insert(*execution_class);
            }
            validate_segment_topology(segment, &unique_classes)?;
            for section in [
                segment.launch_facts.as_slice(),
                segment.runtime_facts.as_slice(),
                segment.checkpoint_facts.as_slice(),
                segment.metric_facts.as_slice(),
                segment.visualization_refs.as_slice(),
            ] {
                for artifact_ref in section {
                    validate_artifact_ref(artifact_ref)?;
                }
            }
            for validator_result in &segment.validator_results {
                if validator_result.detail.trim().is_empty() {
                    return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                        detail: format!(
                            "segment `{}` has validator result with empty detail",
                            segment.segment_id
                        ),
                    });
                }
            }
        }
        if bundle_checkpoint_facts == 0 {
            return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                detail: String::from("bundle must retain at least one checkpoint fact"),
            });
        }
        if topology_coverage
            != BTreeSet::from([
                TrainingExecutionTopologyKind::SingleNode,
                TrainingExecutionTopologyKind::DenseDistributed,
                TrainingExecutionTopologyKind::ContributorWindow,
                TrainingExecutionTopologyKind::ValidatorOnly,
                TrainingExecutionTopologyKind::Hybrid,
            ])
        {
            return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                detail: String::from(
                    "canonical bundle must cover single_node, dense_distributed, contributor_window, validator_only, and hybrid topologies",
                ),
            });
        }
        let required_classes = BTreeSet::from([
            CrossProviderExecutionClass::DenseFullModelRank,
            CrossProviderExecutionClass::ValidatedContributorWindow,
            CrossProviderExecutionClass::Validator,
            CrossProviderExecutionClass::CheckpointWriter,
            CrossProviderExecutionClass::EvalWorker,
        ]);
        if !required_classes.is_subset(&class_coverage) {
            return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                detail: String::from(
                    "canonical bundle must cover dense_full_model_rank, validated_contributor_window, validator, checkpoint_writer, and eval_worker",
                ),
            });
        }
        if self.final_artifact_refs.is_empty() || self.after_action_refs.is_empty() {
            return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                detail: String::from(
                    "final_artifact_refs and after_action_refs must both stay non-empty",
                ),
            });
        }
        for artifact_ref in &self.final_artifact_refs {
            validate_artifact_ref(artifact_ref)?;
        }
        for artifact_ref in &self.after_action_refs {
            validate_artifact_ref(artifact_ref)?;
        }
        if self.final_disposition.detail.trim().is_empty()
            || self.final_disposition.promotion_outcome.trim().is_empty()
        {
            return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                detail: String::from(
                    "final_disposition must keep promotion_outcome and detail explicit",
                ),
            });
        }
        if self.bundle_digest != self.stable_digest() {
            return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                detail: String::from("bundle_digest drifted"),
            });
        }
        Ok(())
    }
}

/// Returns the canonical provider-neutral training execution evidence bundle.
pub fn canonical_training_execution_evidence_bundle(
) -> Result<TrainingExecutionEvidenceBundle, TrainingExecutionEvidenceBundleError> {
    let manifest = cross_provider_training_program_manifest()?;
    let source_contracts = canonical_cross_provider_compute_source_contracts()?;
    let mut bundle = TrainingExecutionEvidenceBundle {
        schema_version: String::from(TRAINING_EXECUTION_EVIDENCE_BUNDLE_SCHEMA_VERSION),
        bundle_id: String::from("psionic-cross-provider-training-execution-evidence-v1"),
        program_manifest_id: manifest.program_manifest_id.clone(),
        program_manifest_digest: manifest.program_manifest_digest.clone(),
        run_id: String::from("psion-xprovider-evidence-sample-1"),
        lane_id: String::from("psion.cross_provider.pretrain.reference"),
        segment_evidence: vec![
            TrainingExecutionSegmentEvidence {
                segment_id: String::from("google-single-node"),
                topology_kind: TrainingExecutionTopologyKind::SingleNode,
                source_ids: vec![String::from("google_l4_validator_node")],
                execution_classes: vec![CrossProviderExecutionClass::DenseFullModelRank],
                launch_facts: vec![repo_artifact_ref(
                    "launch_contract",
                    "fixtures/training/launch_contracts/google_single_node_accelerated_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    true,
                    "Single-node Google lane keeps one explicit launch contract instead of provider-local launcher logs.",
                )?],
                runtime_facts: vec![repo_artifact_ref(
                    "dense_runtime_contract",
                    "fixtures/training/dense_rank_runtime_reference_contract_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    true,
                    "Single-node dense execution still binds into the shared dense-rank runtime contract family.",
                )?],
                checkpoint_facts: vec![repo_artifact_ref(
                    "dense_checkpoint_artifact",
                    "fixtures/psion/checkpoint_recovery/psion_dense_checkpoint_artifact_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    true,
                    "Single-node checkpoint proof stays explicit instead of being hidden behind phase logs.",
                )?],
                metric_facts: vec![repo_artifact_ref(
                    "live_metric_bundle",
                    "fixtures/training_visualization/psion_google_live_remote_training_visualization_bundle_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    true,
                    "The one-second Google live bundle is the authoritative metric surface for the single-node lane.",
                )?],
                visualization_refs: vec![
                    repo_artifact_ref(
                        "visualization_bundle",
                        "fixtures/training_visualization/psion_google_live_remote_training_visualization_bundle_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        true,
                        "The app-facing visualization bundle is carried directly into the final evidence bundle.",
                    )?,
                    repo_artifact_ref(
                        "run_index",
                        "fixtures/training_visualization/remote_training_run_index_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        true,
                        "Run discovery remains explicit through the typed run index.",
                    )?,
                ],
                validator_results: vec![TrainingExecutionValidatorResult {
                    validator_id: String::from("google-single-node-validator"),
                    execution_class: CrossProviderExecutionClass::DenseFullModelRank,
                    disposition: TrainingExecutionDisposition::CompletedSuccess,
                    evidence_posture: TrainingExecutionEvidencePosture::Measured,
                    detail: String::from(
                        "Single-node Google execution preserved launch, runtime, checkpoint, metric, and visualization proof without degradation.",
                    ),
                }],
                segment_disposition: TrainingExecutionDisposition::CompletedSuccess,
                detail: String::from(
                    "This segment proves the bundle family can seal a successful single-node dense run without changing schema families.",
                ),
            },
            TrainingExecutionSegmentEvidence {
                segment_id: String::from("runpod-dense-distributed"),
                topology_kind: TrainingExecutionTopologyKind::DenseDistributed,
                source_ids: vec![String::from("runpod_8xh100_dense_node")],
                execution_classes: vec![
                    CrossProviderExecutionClass::DenseFullModelRank,
                    CrossProviderExecutionClass::CheckpointWriter,
                ],
                launch_facts: vec![repo_artifact_ref(
                    "launch_contract",
                    "fixtures/training/launch_contracts/runpod_8xh100_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    true,
                    "RunPod dense distributed execution keeps one typed launch contract instead of a lane-specific finalizer JSON family.",
                )?],
                runtime_facts: vec![
                    repo_artifact_ref(
                        "dense_runtime_contract",
                        "fixtures/training/dense_rank_runtime_reference_contract_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        true,
                        "Distributed dense execution still binds to the shared dense-rank runtime substrate.",
                    )?,
                    repo_artifact_ref(
                        "distributed_checkpoint_contract",
                        "fixtures/training/sharded_distributed_checkpoint_contract_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        true,
                        "Distributed checkpoint closure is carried directly into the final evidence bundle.",
                    )?,
                ],
                checkpoint_facts: vec![repo_artifact_ref(
                    "distributed_checkpoint_contract",
                    "fixtures/training/sharded_distributed_checkpoint_contract_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    true,
                    "The sharded checkpoint contract seals checkpoint and restore truth for the distributed lane.",
                )?],
                metric_facts: vec![repo_artifact_ref(
                    "visualization_metric_bundle",
                    "fixtures/training_visualization/parameter_golf_distributed_8xh100_remote_training_visualization_bundle_v1.json",
                    TrainingExecutionEvidencePosture::Derived,
                    true,
                    "The retained RunPod bundle is still post-run-only and keeps the missing live loss curve explicit.",
                )?],
                visualization_refs: vec![
                    repo_artifact_ref(
                        "visualization_bundle",
                        "fixtures/training_visualization/parameter_golf_distributed_8xh100_remote_training_visualization_bundle_v1.json",
                        TrainingExecutionEvidencePosture::Derived,
                        true,
                        "The provider-neutral visualization bundle is preserved even when the lane remains post-run-only.",
                    )?,
                    repo_artifact_ref(
                        "run_index",
                        "fixtures/training_visualization/remote_training_run_index_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        true,
                        "The shared run index still enumerates the distributed lane through the same discovery surface.",
                    )?,
                ],
                validator_results: vec![TrainingExecutionValidatorResult {
                    validator_id: String::from("distributed-lane-validator"),
                    execution_class: CrossProviderExecutionClass::DenseFullModelRank,
                    disposition: TrainingExecutionDisposition::DegradedSuccess,
                    evidence_posture: TrainingExecutionEvidencePosture::Derived,
                    detail: String::from(
                        "The distributed lane sealed checkpoint and visualization proof, but the missing live primary series keeps the outcome degraded instead of promoted.",
                    ),
                }],
                segment_disposition: TrainingExecutionDisposition::DegradedSuccess,
                detail: String::from(
                    "This segment proves the same schema can represent dense distributed proof with degraded-success posture instead of inventing a new RunPod-only bundle family.",
                ),
            },
            TrainingExecutionSegmentEvidence {
                segment_id: String::from("contributor-window"),
                topology_kind: TrainingExecutionTopologyKind::ContributorWindow,
                source_ids: vec![
                    String::from("local_mlx_mac_workstation"),
                    String::from("local_rtx4080_workstation"),
                ],
                execution_classes: vec![CrossProviderExecutionClass::ValidatedContributorWindow],
                launch_facts: vec![repo_artifact_ref(
                    "swarm_launch_contract",
                    "fixtures/training/launch_contracts/local_first_swarm_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    true,
                    "Contributor-window execution still uses one typed launch contract instead of provider-local orchestration logs.",
                )?],
                runtime_facts: vec![
                    repo_artifact_ref(
                        "swarm_run_contract",
                        "fixtures/swarm/first_swarm_run_contract_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        true,
                        "The open-adapter swarm run contract remains the runtime truth surface for contributor windows.",
                    )?,
                    repo_artifact_ref(
                        "swarm_receipt_contract",
                        "fixtures/swarm/first_swarm_open_adapter_receipt_contract_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        true,
                        "Contributor receipts stay typed even when the first live attempt was refused before remote execution.",
                    )?,
                ],
                checkpoint_facts: Vec::new(),
                metric_facts: vec![repo_artifact_ref(
                    "swarm_rehearsal_report",
                    "fixtures/swarm/reports/first_swarm_trusted_lan_rehearsal_v1.json",
                    TrainingExecutionEvidencePosture::Derived,
                    true,
                    "The current contributor-window lane only retains rehearsal-grade throughput and bottleneck metrics, so that derived posture stays explicit.",
                )?],
                visualization_refs: vec![repo_artifact_ref(
                    "swarm_evidence_bundle",
                    "fixtures/swarm/reports/first_swarm_trusted_lan_evidence_bundle_v1.json",
                    TrainingExecutionEvidencePosture::Refused,
                    true,
                    "The contributor-window lane preserves refusal and no-promotion posture instead of implying live two-node contributor execution.",
                )?],
                validator_results: vec![TrainingExecutionValidatorResult {
                    validator_id: String::from("swarm-validator"),
                    execution_class: CrossProviderExecutionClass::ValidatedContributorWindow,
                    disposition: TrainingExecutionDisposition::Refused,
                    evidence_posture: TrainingExecutionEvidencePosture::Refused,
                    detail: String::from(
                        "The first trusted-LAN attempt was refused before remote execution began, so the bundle keeps replay-required and no-promotion posture explicit.",
                    ),
                }],
                segment_disposition: TrainingExecutionDisposition::Refused,
                detail: String::from(
                    "This segment proves the bundle family can represent a refused contributor-window outcome without a lane-specific refusal JSON family.",
                ),
            },
            TrainingExecutionSegmentEvidence {
                segment_id: String::from("validator-only"),
                topology_kind: TrainingExecutionTopologyKind::ValidatorOnly,
                source_ids: vec![String::from("google_l4_validator_node")],
                execution_classes: vec![CrossProviderExecutionClass::Validator],
                launch_facts: vec![repo_artifact_ref(
                    "validator_launch_contract",
                    "fixtures/training/launch_contracts/google_single_node_accelerated_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    false,
                    "Validator-only proof still binds into the current Google launch surface instead of inventing a validator-only launcher family.",
                )?],
                runtime_facts: vec![repo_artifact_ref(
                    "remote_backend_contract",
                    "fixtures/training/remote_train_artifact_backend_contract_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    true,
                    "Validator proof depends on the shared remote artifact backend contract for finalizer and restore authority.",
                )?],
                checkpoint_facts: vec![repo_artifact_ref(
                    "dense_checkpoint_artifact",
                    "fixtures/psion/checkpoint_recovery/psion_dense_checkpoint_artifact_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    false,
                    "Validator checks can still cite checkpoint evidence without becoming dense execution proof themselves.",
                )?],
                metric_facts: vec![repo_artifact_ref(
                    "summary_visualization_bundle",
                    "fixtures/training_visualization/psion_google_summary_only_remote_training_visualization_bundle_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    true,
                    "Validator-only metrics can be summary-only and still remain truthful inside the shared bundle family.",
                )?],
                visualization_refs: vec![repo_artifact_ref(
                    "summary_visualization_bundle",
                    "fixtures/training_visualization/psion_google_summary_only_remote_training_visualization_bundle_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    true,
                    "Validator-only visualization refs stay summary-only instead of forcing fake loss curves.",
                )?],
                validator_results: vec![TrainingExecutionValidatorResult {
                    validator_id: String::from("google-validator-1"),
                    execution_class: CrossProviderExecutionClass::Validator,
                    disposition: TrainingExecutionDisposition::CompletedSuccess,
                    evidence_posture: TrainingExecutionEvidencePosture::Measured,
                    detail: String::from(
                        "Validator-only proof stays measured and explicit even when the underlying run class is not dense execution.",
                    ),
                }],
                segment_disposition: TrainingExecutionDisposition::CompletedSuccess,
                detail: String::from(
                    "This segment proves validator-only proof can be emitted through the same bundle family.",
                ),
            },
            TrainingExecutionSegmentEvidence {
                segment_id: String::from("hybrid-program"),
                topology_kind: TrainingExecutionTopologyKind::Hybrid,
                source_ids: vec![
                    String::from("runpod_8xh100_dense_node"),
                    String::from("google_l4_validator_node"),
                    String::from("local_mlx_mac_workstation"),
                ],
                execution_classes: vec![
                    CrossProviderExecutionClass::DenseFullModelRank,
                    CrossProviderExecutionClass::ValidatedContributorWindow,
                    CrossProviderExecutionClass::Validator,
                    CrossProviderExecutionClass::CheckpointWriter,
                    CrossProviderExecutionClass::EvalWorker,
                ],
                launch_facts: vec![
                    repo_artifact_ref(
                        "google_swarm_launch_contract",
                        "fixtures/training/launch_contracts/google_two_node_swarm_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        false,
                        "Hybrid proof can cite multiple launch seams through one shared evidence family.",
                    )?,
                    repo_artifact_ref(
                        "runpod_launch_contract",
                        "fixtures/training/launch_contracts/runpod_8xh100_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        false,
                        "The hybrid bundle can carry both dense-rank and mirror-side launch facts without lane-specific proof JSON.",
                    )?,
                ],
                runtime_facts: vec![
                    repo_artifact_ref(
                        "hybrid_plan",
                        "fixtures/training/hybrid_pretraining_plan_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        true,
                        "The hybrid planner provides the canonical mixed work-class runtime plan.",
                    )?,
                    repo_artifact_ref(
                        "distributed_checkpoint_contract",
                        "fixtures/training/sharded_distributed_checkpoint_contract_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        true,
                        "Hybrid closure reuses the shared distributed checkpoint contract instead of inventing a hybrid-only checkpoint proof family.",
                    )?,
                    repo_artifact_ref(
                        "remote_backend_contract",
                        "fixtures/training/remote_train_artifact_backend_contract_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        true,
                        "Hybrid closure reuses the shared remote artifact backend contract for finalizer projection and restore authority.",
                    )?,
                ],
                checkpoint_facts: vec![repo_artifact_ref(
                    "distributed_checkpoint_contract",
                    "fixtures/training/sharded_distributed_checkpoint_contract_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    true,
                    "Hybrid closure keeps one provider-neutral checkpoint family even when multiple execution classes contribute to the run.",
                )?],
                metric_facts: vec![repo_artifact_ref(
                    "run_index",
                    "fixtures/training_visualization/remote_training_run_index_v1.json",
                    TrainingExecutionEvidencePosture::Measured,
                    true,
                    "Hybrid proof keeps the run index explicit because the app still discovers hybrid lanes through the same typed surface.",
                )?],
                visualization_refs: vec![
                    repo_artifact_ref(
                        "google_live_visualization",
                        "fixtures/training_visualization/psion_google_live_remote_training_visualization_bundle_v1.json",
                        TrainingExecutionEvidencePosture::Measured,
                        false,
                        "Hybrid proof can cite live Google visualization state where available.",
                    )?,
                    repo_artifact_ref(
                        "runpod_distributed_visualization",
                        "fixtures/training_visualization/parameter_golf_distributed_8xh100_remote_training_visualization_bundle_v1.json",
                        TrainingExecutionEvidencePosture::Derived,
                        false,
                        "Hybrid proof can also cite post-run-only distributed visualization state while keeping the degraded posture explicit.",
                    )?,
                ],
                validator_results: vec![
                    TrainingExecutionValidatorResult {
                        validator_id: String::from("hybrid-validator-google"),
                        execution_class: CrossProviderExecutionClass::Validator,
                        disposition: TrainingExecutionDisposition::DegradedSuccess,
                        evidence_posture: TrainingExecutionEvidencePosture::Measured,
                        detail: String::from(
                            "The hybrid bundle preserves successful validator coverage but keeps missing always-live distributed telemetry explicit.",
                        ),
                    },
                    TrainingExecutionValidatorResult {
                        validator_id: String::from("hybrid-eval-worker"),
                        execution_class: CrossProviderExecutionClass::EvalWorker,
                        disposition: TrainingExecutionDisposition::CompletedSuccess,
                        evidence_posture: TrainingExecutionEvidencePosture::Measured,
                        detail: String::from(
                            "Eval-worker proof is retained under the same bundle family instead of a sidecar eval-only proof JSON.",
                        ),
                    },
                ],
                segment_disposition: TrainingExecutionDisposition::DegradedSuccess,
                detail: String::from(
                    "This segment proves the bundle family can seal a hybrid run that spans dense ranks, contributor windows, validator work, checkpoint writing, and eval work.",
                ),
            },
        ],
        final_artifact_refs: vec![
            repo_artifact_ref(
                "remote_artifact_backend_contract",
                "fixtures/training/remote_train_artifact_backend_contract_v1.json",
                TrainingExecutionEvidencePosture::Measured,
                true,
                "Remote backend placement and finalizer projection are part of final evidence closure.",
            )?,
            repo_artifact_ref(
                "hybrid_plan",
                "fixtures/training/hybrid_pretraining_plan_v1.json",
                TrainingExecutionEvidencePosture::Measured,
                true,
                "The final bundle cites the shared hybrid planner instead of a lane-specific closeout summary.",
            )?,
        ],
        after_action_refs: vec![repo_artifact_ref(
            "after_action_audit",
            "docs/audits/2026-03-25-cross-provider-pretraining-system-readiness-audit.md",
            TrainingExecutionEvidencePosture::Derived,
            true,
            "After-action direction remains explicit and machine-linkable instead of narrative-only closeout.",
        )?],
        final_disposition: TrainingExecutionFinalDisposition {
            disposition: TrainingExecutionDisposition::DegradedSuccess,
            promotion_outcome: String::from("no_global_promotion_yet"),
            detail: String::from(
                "The canonical bundle proves the shared schema can encode successful, degraded, and refused segments together. The current cross-provider program remains degraded overall because dense distributed live telemetry and full hybrid runtime closure are not finished.",
            ),
        },
        claim_boundary: String::from(
            "This bundle family proves one provider-neutral final evidence schema can represent single-node dense runs, dense distributed runs, contributor-window runs, validator-only runs, and hybrid runs without lane-specific proof JSON. It does not claim that every segment shown here has already been executed together in one real production run.",
        ),
        bundle_digest: String::new(),
    };
    bundle.bundle_digest = bundle.stable_digest();
    bundle.validate(&manifest, &source_contracts)?;
    Ok(bundle)
}

/// Writes the canonical provider-neutral training execution evidence bundle to the requested path.
pub fn write_training_execution_evidence_bundle(
    output_path: impl AsRef<Path>,
) -> Result<(), TrainingExecutionEvidenceBundleError> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| TrainingExecutionEvidenceBundleError::CreateDir {
            path: parent.display().to_string(),
            error,
        })?;
    }
    let bundle = canonical_training_execution_evidence_bundle()?;
    let bytes = serde_json::to_vec_pretty(&bundle)?;
    fs::write(output_path, bytes).map_err(|error| TrainingExecutionEvidenceBundleError::Write {
        path: output_path.display().to_string(),
        error,
    })?;
    Ok(())
}

fn validate_segment_topology(
    segment: &TrainingExecutionSegmentEvidence,
    unique_classes: &BTreeSet<CrossProviderExecutionClass>,
) -> Result<(), TrainingExecutionEvidenceBundleError> {
    match segment.topology_kind {
        TrainingExecutionTopologyKind::SingleNode => {
            if segment.source_ids.len() != 1
                || unique_classes != &BTreeSet::from([CrossProviderExecutionClass::DenseFullModelRank])
            {
                return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                    detail: format!(
                        "single-node segment `{}` must keep exactly one source and only dense_full_model_rank",
                        segment.segment_id
                    ),
                });
            }
        }
        TrainingExecutionTopologyKind::DenseDistributed => {
            if !unique_classes.contains(&CrossProviderExecutionClass::DenseFullModelRank) {
                return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                    detail: format!(
                        "dense-distributed segment `{}` must include dense_full_model_rank",
                        segment.segment_id
                    ),
                });
            }
        }
        TrainingExecutionTopologyKind::ContributorWindow => {
            if unique_classes
                != &BTreeSet::from([CrossProviderExecutionClass::ValidatedContributorWindow])
            {
                return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                    detail: format!(
                        "contributor-window segment `{}` must keep only validated_contributor_window",
                        segment.segment_id
                    ),
                });
            }
        }
        TrainingExecutionTopologyKind::ValidatorOnly => {
            if unique_classes != &BTreeSet::from([CrossProviderExecutionClass::Validator]) {
                return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                    detail: format!(
                        "validator-only segment `{}` must keep only validator",
                        segment.segment_id
                    ),
                });
            }
        }
        TrainingExecutionTopologyKind::Hybrid => {
            let required_hybrid_classes = BTreeSet::from([
                CrossProviderExecutionClass::DenseFullModelRank,
                CrossProviderExecutionClass::ValidatedContributorWindow,
                CrossProviderExecutionClass::Validator,
            ]);
            if !required_hybrid_classes.is_subset(unique_classes) || unique_classes.len() < 3 {
                return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
                    detail: format!(
                        "hybrid segment `{}` must keep dense, contributor, and validator execution classes explicit",
                        segment.segment_id
                    ),
                });
            }
        }
    }
    Ok(())
}

fn validate_artifact_ref(
    artifact_ref: &TrainingExecutionEvidenceRef,
) -> Result<(), TrainingExecutionEvidenceBundleError> {
    if artifact_ref.artifact_role.trim().is_empty()
        || artifact_ref.artifact_path.trim().is_empty()
        || artifact_ref.detail.trim().is_empty()
    {
        return Err(TrainingExecutionEvidenceBundleError::InvalidBundle {
            detail: String::from("artifact refs must keep role, path, and detail explicit"),
        });
    }
    Ok(())
}

fn repo_artifact_ref(
    artifact_role: &str,
    artifact_path: &str,
    evidence_posture: TrainingExecutionEvidencePosture,
    authoritative: bool,
    detail: &str,
) -> Result<TrainingExecutionEvidenceRef, TrainingExecutionEvidenceBundleError> {
    let digest = sha256_file(artifact_path)?;
    Ok(TrainingExecutionEvidenceRef {
        artifact_role: String::from(artifact_role),
        artifact_path: String::from(artifact_path),
        artifact_digest: Some(digest),
        evidence_posture,
        authoritative,
        detail: String::from(detail),
    })
}

fn sha256_file(path: &str) -> Result<String, TrainingExecutionEvidenceBundleError> {
    let bytes = fs::read(path).map_err(|error| TrainingExecutionEvidenceBundleError::Read {
        path: String::from(path),
        error,
    })?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("stable digest serialization should succeed");
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_training_execution_evidence_bundle, TrainingExecutionDisposition,
        TrainingExecutionEvidenceBundleError, TrainingExecutionTopologyKind,
    };

    #[test]
    fn canonical_bundle_covers_all_topology_kinds() -> Result<(), Box<dyn std::error::Error>> {
        let bundle = canonical_training_execution_evidence_bundle()?;
        let topology_kinds = bundle
            .segment_evidence
            .iter()
            .map(|segment| segment.topology_kind)
            .collect::<BTreeSet<_>>();
        assert!(topology_kinds.contains(&TrainingExecutionTopologyKind::Hybrid));
        assert!(topology_kinds.contains(&TrainingExecutionTopologyKind::SingleNode));
        Ok(())
    }

    #[test]
    fn canonical_bundle_rejects_missing_final_artifact_refs()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut bundle = canonical_training_execution_evidence_bundle()?;
        bundle.final_artifact_refs.clear();
        let manifest = crate::cross_provider_training_program_manifest()?;
        let sources = crate::canonical_cross_provider_compute_source_contracts()?;
        let err = bundle
            .validate(&manifest, &sources)
            .expect_err("missing final artifact refs must be rejected");
        assert!(matches!(
            err,
            TrainingExecutionEvidenceBundleError::InvalidBundle { .. }
        ));
        Ok(())
    }

    #[test]
    fn canonical_bundle_stays_degraded_until_all_segments_are_green()
    -> Result<(), Box<dyn std::error::Error>> {
        let bundle = canonical_training_execution_evidence_bundle()?;
        assert_eq!(
            bundle.final_disposition.disposition,
            TrainingExecutionDisposition::DegradedSuccess
        );
        Ok(())
    }
}
