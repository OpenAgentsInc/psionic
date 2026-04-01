//! Rust-native optimizer substrate contracts for Psionic.
//!
//! The first landing keeps candidate identity, lineage state, persisted run
//! state, and top-level run receipts explicit and machine-readable. Later
//! issues add evaluation contracts, search loops, reflection, merge, and proof
//! lanes above these core artifacts.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

const RUN_SPEC_PREFIX: &[u8] = b"psionic_optimize_run_spec|";
const CANDIDATE_MANIFEST_PREFIX: &[u8] = b"psionic_optimize_candidate_manifest|";
const CASE_MANIFEST_PREFIX: &[u8] = b"psionic_optimize_case_manifest|";
const LINEAGE_STATE_PREFIX: &[u8] = b"psionic_optimize_lineage_state|";
const CASE_RECEIPT_PREFIX: &[u8] = b"psionic_optimize_case_receipt|";
const BATCH_RECEIPT_PREFIX: &[u8] = b"psionic_optimize_batch_receipt|";
const FRONTIER_SNAPSHOT_PREFIX: &[u8] = b"psionic_optimize_frontier_snapshot|";
const RUN_RECEIPT_PREFIX: &[u8] = b"psionic_optimize_run_receipt|";

/// Stable frontier mode identifier declared by one optimization run spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationFrontierMode {
    /// One scalar objective and one retained frontier.
    Scalar,
    /// Multiple named objectives will later produce a hybrid frontier.
    Hybrid,
}

/// Final stop reason recorded by one optimization run receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationStopReason {
    /// The operator or caller stopped the run intentionally.
    Manual,
    /// The run reached its configured iteration budget.
    IterationBudgetReached,
    /// The run reached its configured candidate budget.
    CandidateBudgetReached,
    /// The run completed without entering a proposal loop.
    NoSearchRequired,
}

/// Stable run spec for one optimizer invocation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationRunSpec {
    /// Stable schema version.
    pub schema_version: u16,
    /// Stable run identifier.
    pub run_id: String,
    /// Candidate family under optimization.
    pub family_id: String,
    /// Ordered dataset or retained-surface refs bound to this run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dataset_refs: Vec<String>,
    /// Optional issue or queue ref that owns the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub issue_ref: Option<String>,
    /// Configured frontier mode.
    pub frontier_mode: OptimizationFrontierMode,
    /// Optional iteration budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iteration_budget: Option<u32>,
    /// Optional candidate budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_budget: Option<u32>,
    /// Stable digest over the run spec.
    pub spec_digest: String,
}

impl OptimizationRunSpec {
    /// Creates a run spec with default schema metadata and a stable digest.
    #[must_use]
    pub fn new(run_id: impl Into<String>, family_id: impl Into<String>) -> Self {
        Self {
            schema_version: 1,
            run_id: run_id.into(),
            family_id: family_id.into(),
            dataset_refs: Vec::new(),
            issue_ref: None,
            frontier_mode: OptimizationFrontierMode::Scalar,
            iteration_budget: None,
            candidate_budget: None,
            spec_digest: String::new(),
        }
        .with_stable_digest()
    }

    /// Returns a copy with the given dataset refs.
    #[must_use]
    pub fn with_dataset_refs(mut self, dataset_refs: Vec<String>) -> Self {
        self.dataset_refs = dataset_refs;
        self.with_stable_digest()
    }

    /// Returns a copy with the given issue ref.
    #[must_use]
    pub fn with_issue_ref(mut self, issue_ref: impl Into<String>) -> Self {
        self.issue_ref = Some(issue_ref.into());
        self.with_stable_digest()
    }

    /// Returns a copy with the given frontier mode.
    #[must_use]
    pub fn with_frontier_mode(mut self, frontier_mode: OptimizationFrontierMode) -> Self {
        self.frontier_mode = frontier_mode;
        self.with_stable_digest()
    }

    /// Returns a copy with the given iteration budget.
    #[must_use]
    pub fn with_iteration_budget(mut self, iteration_budget: u32) -> Self {
        self.iteration_budget = Some(iteration_budget);
        self.with_stable_digest()
    }

    /// Returns a copy with the given candidate budget.
    #[must_use]
    pub fn with_candidate_budget(mut self, candidate_budget: u32) -> Self {
        self.candidate_budget = Some(candidate_budget);
        self.with_stable_digest()
    }

    /// Returns the stable digest over the run spec payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.spec_digest.clear();
        stable_digest(RUN_SPEC_PREFIX, &digestible)
    }

    /// Populates the stable digest field.
    #[must_use]
    pub fn with_stable_digest(mut self) -> Self {
        self.spec_digest = self.stable_digest();
        self
    }
}

/// Stable manifest for one candidate artifact under optimization.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationCandidateManifest {
    /// Stable schema version.
    pub schema_version: u16,
    /// Stable candidate identifier.
    pub candidate_id: String,
    /// Candidate family under optimization.
    pub family_id: String,
    /// Stable run id that first materialized this candidate.
    pub originating_run_id: String,
    /// Text-renderable named component map for the candidate.
    pub components: BTreeMap<String, String>,
    /// Parent candidate ids in deterministic order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parent_candidate_ids: Vec<String>,
    /// Ordered provenance refs that justify how the candidate was materialized.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub provenance_refs: Vec<String>,
    /// Optional issue or queue ref that owns the candidate row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub issue_ref: Option<String>,
    /// Stable digest over the candidate manifest payload.
    pub manifest_digest: String,
}

impl OptimizationCandidateManifest {
    /// Creates a candidate manifest with stable digest metadata.
    #[must_use]
    pub fn new(
        candidate_id: impl Into<String>,
        family_id: impl Into<String>,
        originating_run_id: impl Into<String>,
        components: BTreeMap<String, String>,
    ) -> Self {
        Self {
            schema_version: 1,
            candidate_id: candidate_id.into(),
            family_id: family_id.into(),
            originating_run_id: originating_run_id.into(),
            components,
            parent_candidate_ids: Vec::new(),
            provenance_refs: Vec::new(),
            issue_ref: None,
            manifest_digest: String::new(),
        }
        .with_stable_digest()
    }

    /// Returns a copy with the given parent candidate ids.
    #[must_use]
    pub fn with_parent_candidate_ids(mut self, parent_candidate_ids: Vec<String>) -> Self {
        self.parent_candidate_ids = parent_candidate_ids;
        self.with_stable_digest()
    }

    /// Returns a copy with the given provenance refs.
    #[must_use]
    pub fn with_provenance_refs(mut self, provenance_refs: Vec<String>) -> Self {
        self.provenance_refs = provenance_refs;
        self.with_stable_digest()
    }

    /// Returns a copy with the given issue ref.
    #[must_use]
    pub fn with_issue_ref(mut self, issue_ref: impl Into<String>) -> Self {
        self.issue_ref = Some(issue_ref.into());
        self.with_stable_digest()
    }

    /// Returns the stable digest over the candidate manifest payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.manifest_digest.clear();
        stable_digest(CANDIDATE_MANIFEST_PREFIX, &digestible)
    }

    /// Populates the stable digest field.
    #[must_use]
    pub fn with_stable_digest(mut self) -> Self {
        self.manifest_digest = self.stable_digest();
        self
    }
}

/// Split membership for one retained evaluation case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationCaseSplit {
    /// Train-time batch sampling surface.
    Train,
    /// Held-out retained validation surface.
    Validation,
    /// Test-only surface.
    Test,
    /// Shadow-only comparison surface.
    Shadow,
}

/// Stable manifest for one retained evaluation case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationCaseManifest {
    /// Stable schema version.
    pub schema_version: u16,
    /// Stable case identifier.
    pub case_id: String,
    /// Split membership for the case.
    pub split: OptimizationCaseSplit,
    /// Optional label or expected outcome identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Structured string metadata for the case.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
    /// Ordered evidence refs that justify the case.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_refs: Vec<String>,
    /// Stable digest over the case manifest payload.
    pub case_digest: String,
}

impl OptimizationCaseManifest {
    /// Creates a case manifest with stable digest metadata.
    #[must_use]
    pub fn new(case_id: impl Into<String>, split: OptimizationCaseSplit) -> Self {
        Self {
            schema_version: 1,
            case_id: case_id.into(),
            split,
            label: None,
            metadata: BTreeMap::new(),
            evidence_refs: Vec::new(),
            case_digest: String::new(),
        }
        .with_stable_digest()
    }

    /// Returns a copy with the given label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self.with_stable_digest()
    }

    /// Returns a copy with the given metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, String>) -> Self {
        self.metadata = metadata;
        self.with_stable_digest()
    }

    /// Returns a copy with the given evidence refs.
    #[must_use]
    pub fn with_evidence_refs(mut self, evidence_refs: Vec<String>) -> Self {
        self.evidence_refs = evidence_refs;
        self.with_stable_digest()
    }

    /// Returns the stable digest over the case manifest payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.case_digest.clear();
        stable_digest(CASE_MANIFEST_PREFIX, &digestible)
    }

    /// Populates the stable digest field.
    #[must_use]
    pub fn with_stable_digest(mut self) -> Self {
        self.case_digest = self.stable_digest();
        self
    }
}

/// Shared typed evaluator feedback for one case.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationSharedFeedback {
    /// Short summary of the evaluation result.
    pub summary: String,
    /// Ordered detail rows.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub details: Vec<String>,
}

impl OptimizationSharedFeedback {
    /// Creates one shared feedback record.
    #[must_use]
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            details: Vec::new(),
        }
    }

    /// Returns a copy with the given detail rows.
    #[must_use]
    pub fn with_details(mut self, details: Vec<String>) -> Self {
        self.details = details;
        self
    }
}

/// Typed evaluator feedback scoped to one candidate component.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationComponentFeedback {
    /// Short summary of the component-local feedback.
    pub summary: String,
    /// Ordered detail rows for the component.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub details: Vec<String>,
}

impl OptimizationComponentFeedback {
    /// Creates one component feedback record.
    #[must_use]
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            details: Vec::new(),
        }
    }

    /// Returns a copy with the given detail rows.
    #[must_use]
    pub fn with_details(mut self, details: Vec<String>) -> Self {
        self.details = details;
        self
    }
}

/// Case-level evaluation receipt for one candidate against one retained case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationCaseEvaluationReceipt {
    /// Stable schema version.
    pub schema_version: u16,
    /// Candidate identifier under evaluation.
    pub candidate_id: String,
    /// Candidate manifest digest under evaluation.
    pub candidate_manifest_digest: String,
    /// Retained case identifier.
    pub case_id: String,
    /// Retained case digest.
    pub case_digest: String,
    /// Scalar score for the case.
    pub scalar_score: i64,
    /// Named objective scores for the case.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub objective_scores: BTreeMap<String, i64>,
    /// Shared evaluator feedback.
    pub shared_feedback: OptimizationSharedFeedback,
    /// Per-component feedback map.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub component_feedback: BTreeMap<String, OptimizationComponentFeedback>,
    /// Unified cache key used by the optimizer substrate.
    pub cache_key: String,
    /// Stable digest over the case receipt payload.
    pub receipt_digest: String,
}

impl OptimizationCaseEvaluationReceipt {
    /// Creates one case receipt with a stable cache key and digest.
    #[must_use]
    pub fn new(
        candidate: &OptimizationCandidateManifest,
        case: &OptimizationCaseManifest,
        scalar_score: i64,
        objective_scores: BTreeMap<String, i64>,
        shared_feedback: OptimizationSharedFeedback,
        component_feedback: BTreeMap<String, OptimizationComponentFeedback>,
    ) -> Self {
        Self {
            schema_version: 1,
            candidate_id: candidate.candidate_id.clone(),
            candidate_manifest_digest: candidate.manifest_digest.clone(),
            case_id: case.case_id.clone(),
            case_digest: case.case_digest.clone(),
            scalar_score,
            objective_scores,
            shared_feedback,
            component_feedback,
            cache_key: OptimizationEvaluationCache::cache_key(candidate, case),
            receipt_digest: String::new(),
        }
        .with_stable_digest()
    }

    /// Returns the stable digest over the case receipt payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.receipt_digest.clear();
        stable_digest(CASE_RECEIPT_PREFIX, &digestible)
    }

    /// Populates the stable digest field.
    #[must_use]
    pub fn with_stable_digest(mut self) -> Self {
        self.receipt_digest = self.stable_digest();
        self
    }
}

/// Unified case-evaluation cache for the optimizer substrate.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationEvaluationCache {
    /// Cached case receipts keyed by candidate-manifest digest plus case digest.
    pub entries: BTreeMap<String, OptimizationCaseEvaluationReceipt>,
}

impl OptimizationEvaluationCache {
    /// Builds the stable cache key for one candidate-plus-case pair.
    #[must_use]
    pub fn cache_key(
        candidate: &OptimizationCandidateManifest,
        case: &OptimizationCaseManifest,
    ) -> String {
        format!("{}:{}", candidate.manifest_digest, case.case_digest)
    }

    /// Returns one cached receipt when present.
    #[must_use]
    pub fn lookup(
        &self,
        candidate: &OptimizationCandidateManifest,
        case: &OptimizationCaseManifest,
    ) -> Option<&OptimizationCaseEvaluationReceipt> {
        self.entries.get(&Self::cache_key(candidate, case))
    }

    /// Inserts or replaces one cached case receipt.
    pub fn insert(
        &mut self,
        candidate: &OptimizationCandidateManifest,
        case: &OptimizationCaseManifest,
        receipt: OptimizationCaseEvaluationReceipt,
    ) {
        self.entries
            .insert(Self::cache_key(candidate, case), receipt);
    }
}

/// Aggregated batch receipt for one candidate across a retained case batch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationBatchEvaluationReceipt {
    /// Stable schema version.
    pub schema_version: u16,
    /// Stable report identifier.
    pub report_id: String,
    /// Stable run id.
    pub run_id: String,
    /// Candidate identifier under evaluation.
    pub candidate_id: String,
    /// Candidate manifest digest under evaluation.
    pub candidate_manifest_digest: String,
    /// Ordered case receipts for the batch.
    pub case_receipts: Vec<OptimizationCaseEvaluationReceipt>,
    /// Sum of scalar case scores for the batch.
    pub aggregated_scalar_score: i64,
    /// Sum of objective scores for the batch.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub aggregated_objective_scores: BTreeMap<String, i64>,
    /// Number of cache hits for the batch.
    pub cache_hit_count: u32,
    /// Number of cache misses for the batch.
    pub cache_miss_count: u32,
    /// Stable digest over the batch receipt payload.
    pub receipt_digest: String,
}

impl OptimizationBatchEvaluationReceipt {
    /// Creates one batch receipt and aggregates objective totals from the cases.
    #[must_use]
    pub fn new(
        run_id: impl Into<String>,
        candidate: &OptimizationCandidateManifest,
        case_receipts: Vec<OptimizationCaseEvaluationReceipt>,
        cache_hit_count: u32,
        cache_miss_count: u32,
    ) -> Self {
        let aggregated_scalar_score = case_receipts.iter().map(|case| case.scalar_score).sum();
        let mut aggregated_objective_scores = BTreeMap::new();
        for case_receipt in &case_receipts {
            for (objective_name, objective_score) in &case_receipt.objective_scores {
                *aggregated_objective_scores
                    .entry(objective_name.clone())
                    .or_insert(0) += objective_score;
            }
        }
        Self {
            schema_version: 1,
            report_id: String::from("psionic.optimize.batch_evaluation_receipt.v1"),
            run_id: run_id.into(),
            candidate_id: candidate.candidate_id.clone(),
            candidate_manifest_digest: candidate.manifest_digest.clone(),
            case_receipts,
            aggregated_scalar_score,
            aggregated_objective_scores,
            cache_hit_count,
            cache_miss_count,
            receipt_digest: String::new(),
        }
        .with_stable_digest()
    }

    /// Returns the stable digest over the batch receipt payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.receipt_digest.clear();
        stable_digest(BATCH_RECEIPT_PREFIX, &digestible)
    }

    /// Populates the stable digest field.
    #[must_use]
    pub fn with_stable_digest(mut self) -> Self {
        self.receipt_digest = self.stable_digest();
        self
    }
}

/// Winner row for one retained case in the frontier snapshot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationCaseFrontierRow {
    /// Stable case identifier.
    pub case_id: String,
    /// Winning candidate identifier.
    pub winning_candidate_id: String,
    /// Winning scalar score for the case.
    pub winning_scalar_score: i64,
}

/// Winner row for one named objective in the frontier snapshot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationObjectiveFrontierRow {
    /// Named objective identifier.
    pub objective_name: String,
    /// Winning candidate identifier.
    pub winning_candidate_id: String,
    /// Winning aggregated objective score.
    pub winning_score: i64,
}

/// Snapshot of the retained frontier for one optimizer run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationFrontierSnapshot {
    /// Stable schema version.
    pub schema_version: u16,
    /// Stable report identifier.
    pub report_id: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Frontier mode used to interpret the snapshot.
    pub frontier_mode: OptimizationFrontierMode,
    /// Source candidates considered for the frontier.
    pub source_candidate_ids: Vec<String>,
    /// Ordered per-case frontier rows.
    pub case_frontier: Vec<OptimizationCaseFrontierRow>,
    /// Ordered per-objective frontier rows.
    pub objective_frontier: Vec<OptimizationObjectiveFrontierRow>,
    /// Deduplicated candidate ids retained by the hybrid frontier.
    pub hybrid_candidate_ids: Vec<String>,
    /// Stable digest over the frontier snapshot payload.
    pub snapshot_digest: String,
}

impl OptimizationFrontierSnapshot {
    /// Builds a frontier snapshot from candidate batch receipts.
    #[must_use]
    pub fn from_batches(
        run_id: impl Into<String>,
        frontier_mode: OptimizationFrontierMode,
        batches: &[OptimizationBatchEvaluationReceipt],
    ) -> Self {
        let run_id = run_id.into();
        let mut case_winners = BTreeMap::<String, (String, i64)>::new();
        let mut objective_winners = BTreeMap::<String, (String, i64)>::new();
        let mut source_candidate_ids = Vec::new();

        for batch in batches {
            push_unique(&mut source_candidate_ids, batch.candidate_id.clone());
            for case_receipt in &batch.case_receipts {
                let candidate_score = case_receipt.scalar_score;
                let candidate_id = case_receipt.candidate_id.clone();
                case_winners
                    .entry(case_receipt.case_id.clone())
                    .and_modify(|winner| {
                        if candidate_score > winner.1 {
                            *winner = (candidate_id.clone(), candidate_score);
                        }
                    })
                    .or_insert((candidate_id, candidate_score));
            }
            for (objective_name, objective_score) in &batch.aggregated_objective_scores {
                let candidate_id = batch.candidate_id.clone();
                objective_winners
                    .entry(objective_name.clone())
                    .and_modify(|winner| {
                        if *objective_score > winner.1 {
                            *winner = (candidate_id.clone(), *objective_score);
                        }
                    })
                    .or_insert((candidate_id, *objective_score));
            }
        }

        let case_frontier = case_winners
            .into_iter()
            .map(|(case_id, (winning_candidate_id, winning_scalar_score))| {
                OptimizationCaseFrontierRow {
                    case_id,
                    winning_candidate_id,
                    winning_scalar_score,
                }
            })
            .collect::<Vec<_>>();
        let objective_frontier = objective_winners
            .into_iter()
            .map(|(objective_name, (winning_candidate_id, winning_score))| {
                OptimizationObjectiveFrontierRow {
                    objective_name,
                    winning_candidate_id,
                    winning_score,
                }
            })
            .collect::<Vec<_>>();

        let mut hybrid_candidate_ids = Vec::new();
        for row in &case_frontier {
            push_unique(&mut hybrid_candidate_ids, row.winning_candidate_id.clone());
        }
        if frontier_mode == OptimizationFrontierMode::Hybrid {
            for row in &objective_frontier {
                push_unique(&mut hybrid_candidate_ids, row.winning_candidate_id.clone());
            }
        }

        Self {
            schema_version: 1,
            report_id: String::from("psionic.optimize.frontier_snapshot.v1"),
            run_id,
            frontier_mode,
            source_candidate_ids,
            case_frontier,
            objective_frontier,
            hybrid_candidate_ids,
            snapshot_digest: String::new(),
        }
        .with_stable_digest()
    }

    /// Returns the stable digest over the frontier snapshot payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.snapshot_digest.clear();
        stable_digest(FRONTIER_SNAPSHOT_PREFIX, &digestible)
    }

    /// Populates the stable digest field.
    #[must_use]
    pub fn with_stable_digest(mut self) -> Self {
        self.snapshot_digest = self.stable_digest();
        self
    }
}

/// Ordered lineage state for one optimization run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationLineageState {
    /// Stable schema version.
    pub schema_version: u16,
    /// Run spec that owns the lineage state.
    pub run_spec: OptimizationRunSpec,
    /// Materialized candidates keyed by candidate id.
    pub candidates: BTreeMap<String, OptimizationCandidateManifest>,
    /// Candidate ids in discovery order.
    pub discovery_order: Vec<String>,
    /// Root candidates for the run in deterministic order.
    pub root_candidate_ids: Vec<String>,
    /// Retained candidate ids for the current run state.
    pub retained_candidate_ids: Vec<String>,
    /// Stable digest over the lineage state payload.
    pub state_digest: String,
}

impl OptimizationLineageState {
    /// Creates an empty lineage state for the given run.
    #[must_use]
    pub fn new(run_spec: OptimizationRunSpec) -> Self {
        Self {
            schema_version: 1,
            run_spec,
            candidates: BTreeMap::new(),
            discovery_order: Vec::new(),
            root_candidate_ids: Vec::new(),
            retained_candidate_ids: Vec::new(),
            state_digest: String::new(),
        }
        .with_stable_digest()
    }

    /// Registers one candidate in the lineage state.
    pub fn register_candidate(
        &mut self,
        candidate: OptimizationCandidateManifest,
    ) -> Result<(), OptimizationLineageStateError> {
        if candidate.family_id != self.run_spec.family_id {
            return Err(OptimizationLineageStateError::FamilyMismatch {
                expected_family_id: self.run_spec.family_id.clone(),
                actual_family_id: candidate.family_id,
            });
        }
        if candidate.originating_run_id != self.run_spec.run_id {
            return Err(OptimizationLineageStateError::RunMismatch {
                expected_run_id: self.run_spec.run_id.clone(),
                actual_run_id: candidate.originating_run_id,
            });
        }
        if self.candidates.contains_key(&candidate.candidate_id) {
            return Err(OptimizationLineageStateError::DuplicateCandidate {
                candidate_id: candidate.candidate_id,
            });
        }
        for parent_candidate_id in &candidate.parent_candidate_ids {
            if !self.candidates.contains_key(parent_candidate_id) {
                return Err(OptimizationLineageStateError::UnknownParent {
                    candidate_id: candidate.candidate_id.clone(),
                    parent_candidate_id: parent_candidate_id.clone(),
                });
            }
        }

        let candidate_id = candidate.candidate_id.clone();
        if candidate.parent_candidate_ids.is_empty() {
            push_unique(&mut self.root_candidate_ids, candidate_id.clone());
        }
        self.discovery_order.push(candidate_id.clone());
        self.candidates.insert(candidate_id.clone(), candidate);
        push_unique(&mut self.retained_candidate_ids, candidate_id);
        self.state_digest = self.stable_digest();
        Ok(())
    }

    /// Replaces the retained set with the supplied deterministic ids.
    pub fn set_retained_candidates(
        &mut self,
        retained_candidate_ids: Vec<String>,
    ) -> Result<(), OptimizationLineageStateError> {
        let mut ordered = Vec::new();
        let mut seen = BTreeSet::new();
        for candidate_id in retained_candidate_ids {
            if !self.candidates.contains_key(&candidate_id) {
                return Err(OptimizationLineageStateError::UnknownRetainedCandidate {
                    candidate_id,
                });
            }
            if seen.insert(candidate_id.clone()) {
                ordered.push(candidate_id);
            }
        }
        self.retained_candidate_ids = ordered;
        self.state_digest = self.stable_digest();
        Ok(())
    }

    /// Returns a candidate manifest by id when one exists.
    #[must_use]
    pub fn candidate(&self, candidate_id: &str) -> Option<&OptimizationCandidateManifest> {
        self.candidates.get(candidate_id)
    }

    /// Returns the stable digest over the lineage state payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.state_digest.clear();
        stable_digest(LINEAGE_STATE_PREFIX, &digestible)
    }

    /// Populates the stable digest field.
    #[must_use]
    pub fn with_stable_digest(mut self) -> Self {
        self.state_digest = self.stable_digest();
        self
    }

    /// Writes the lineage state as pretty JSON.
    pub fn write_json(&self, output_path: impl AsRef<Path>) -> Result<(), OptimizationIoError> {
        let output_path = output_path.as_ref();
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).map_err(|error| OptimizationIoError::CreateDir {
                path: parent.display().to_string(),
                error,
            })?;
        }
        let json = serde_json::to_string_pretty(self)?;
        fs::write(output_path, format!("{json}\n")).map_err(|error| OptimizationIoError::Write {
            path: output_path.display().to_string(),
            error,
        })
    }

    /// Loads the lineage state from one JSON artifact.
    pub fn read_json(input_path: impl AsRef<Path>) -> Result<Self, OptimizationIoError> {
        let input_path = input_path.as_ref();
        let body = fs::read_to_string(input_path).map_err(|error| OptimizationIoError::Read {
            path: input_path.display().to_string(),
            error,
        })?;
        let state: Self =
            serde_json::from_str(&body).map_err(|error| OptimizationIoError::Deserialize {
                path: input_path.display().to_string(),
                error,
            })?;
        Ok(state)
    }
}

/// Top-level run receipt that points at retained candidates and frontier refs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationRunReceipt {
    /// Stable schema version.
    pub schema_version: u16,
    /// Stable report identifier.
    pub report_id: String,
    /// Stable run id.
    pub run_id: String,
    /// Candidate family under optimization.
    pub family_id: String,
    /// Bound run spec digest.
    pub run_spec_digest: String,
    /// Final lineage state digest.
    pub lineage_state_digest: String,
    /// Total materialized candidate count.
    pub candidate_count: u32,
    /// Retained candidate ids in deterministic order.
    pub retained_candidate_ids: Vec<String>,
    /// Frontier snapshot refs retained for this run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frontier_snapshot_refs: Vec<String>,
    /// Explicit stop reason for the run.
    pub stop_reason: OptimizationStopReason,
    /// Plain-language claim boundary for the receipt.
    pub claim_boundary: String,
    /// Stable digest over the receipt payload.
    pub receipt_digest: String,
}

impl OptimizationRunReceipt {
    /// Builds a run receipt from the final lineage state.
    #[must_use]
    pub fn from_state(
        state: &OptimizationLineageState,
        frontier_snapshot_refs: Vec<String>,
        stop_reason: OptimizationStopReason,
    ) -> Self {
        Self {
            schema_version: 1,
            report_id: String::from("psionic.optimize.run_receipt.v1"),
            run_id: state.run_spec.run_id.clone(),
            family_id: state.run_spec.family_id.clone(),
            run_spec_digest: state.run_spec.spec_digest.clone(),
            lineage_state_digest: state.state_digest.clone(),
            candidate_count: state.candidates.len() as u32,
            retained_candidate_ids: state.retained_candidate_ids.clone(),
            frontier_snapshot_refs,
            stop_reason,
            claim_boundary: String::from(
                "This receipt records optimizer run identity, retained candidates, and frontier refs for one bounded offline optimization run. It does not claim evaluation quality, search validity, or runtime promotion by itself.",
            ),
            receipt_digest: String::new(),
        }
        .with_stable_digest()
    }

    /// Returns the stable digest over the receipt payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut digestible = self.clone();
        digestible.receipt_digest.clear();
        stable_digest(RUN_RECEIPT_PREFIX, &digestible)
    }

    /// Populates the stable digest field.
    #[must_use]
    pub fn with_stable_digest(mut self) -> Self {
        self.receipt_digest = self.stable_digest();
        self
    }
}

/// Failure while mutating lineage state.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum OptimizationLineageStateError {
    /// Candidate family did not match the bound run spec.
    #[error(
        "candidate family `{actual_family_id}` did not match run family `{expected_family_id}`"
    )]
    FamilyMismatch {
        /// Expected family id from the bound run spec.
        expected_family_id: String,
        /// Actual family id from the candidate manifest.
        actual_family_id: String,
    },
    /// Candidate originating run did not match the bound run spec.
    #[error("candidate run `{actual_run_id}` did not match run `{expected_run_id}`")]
    RunMismatch {
        /// Expected run id from the bound run spec.
        expected_run_id: String,
        /// Actual originating run id from the candidate manifest.
        actual_run_id: String,
    },
    /// Candidate id was already registered.
    #[error("candidate `{candidate_id}` is already registered in this run state")]
    DuplicateCandidate {
        /// Duplicate candidate id.
        candidate_id: String,
    },
    /// Parent candidate id was not present in the state.
    #[error("candidate `{candidate_id}` references unknown parent `{parent_candidate_id}`")]
    UnknownParent {
        /// Candidate that declared the unknown parent.
        candidate_id: String,
        /// Unknown parent id.
        parent_candidate_id: String,
    },
    /// Retained candidate id was not present in the state.
    #[error("retained candidate `{candidate_id}` is not registered in this run state")]
    UnknownRetainedCandidate {
        /// Unknown retained candidate id.
        candidate_id: String,
    },
}

/// Failure while persisting or loading optimizer JSON artifacts.
#[derive(Debug, Error)]
pub enum OptimizationIoError {
    /// Failed to create a parent directory.
    #[error("failed to create `{path}`: {error}")]
    CreateDir {
        /// Parent path.
        path: String,
        /// Underlying error.
        error: std::io::Error,
    },
    /// Failed to write one artifact.
    #[error("failed to write `{path}`: {error}")]
    Write {
        /// Output path.
        path: String,
        /// Underlying error.
        error: std::io::Error,
    },
    /// Failed to read one artifact.
    #[error("failed to read `{path}`: {error}")]
    Read {
        /// Input path.
        path: String,
        /// Underlying error.
        error: std::io::Error,
    },
    /// Failed to decode one artifact.
    #[error("failed to decode `{path}`: {error}")]
    Deserialize {
        /// Input path.
        path: String,
        /// Underlying error.
        error: serde_json::Error,
    },
    /// Failed to encode JSON.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

fn push_unique(target: &mut Vec<String>, value: String) {
    if !target.iter().any(|existing| existing == &value) {
        target.push(value);
    }
}

fn stable_digest<T>(prefix: &[u8], value: &T) -> String
where
    T: Serialize,
{
    let encoded = match serde_json::to_vec(value) {
        Ok(encoded) => encoded,
        Err(error) => error.to_string().into_bytes(),
    };
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use tempfile::tempdir;

    use super::{
        OptimizationBatchEvaluationReceipt, OptimizationCandidateManifest,
        OptimizationCaseManifest, OptimizationCaseSplit, OptimizationComponentFeedback,
        OptimizationEvaluationCache, OptimizationFrontierMode, OptimizationFrontierSnapshot,
        OptimizationLineageState, OptimizationLineageStateError, OptimizationRunReceipt,
        OptimizationRunSpec, OptimizationSharedFeedback, OptimizationStopReason,
    };

    fn route_components(route_name: &str) -> BTreeMap<String, String> {
        BTreeMap::from([
            (String::from("selected_tool"), String::from(route_name)),
            (
                String::from("reason"),
                String::from("route decision evidence goes here"),
            ),
        ])
    }

    fn route_case(case_id: &str, label: &str) -> OptimizationCaseManifest {
        OptimizationCaseManifest::new(case_id, OptimizationCaseSplit::Validation)
            .with_label(label)
            .with_metadata(BTreeMap::from([(
                String::from("decision_family"),
                String::from("tool_route"),
            )]))
    }

    #[test]
    fn run_spec_stable_digest_changes_with_fields() {
        let baseline = OptimizationRunSpec::new("run_a", "probe.tool_route")
            .with_frontier_mode(OptimizationFrontierMode::Scalar);
        let widened = baseline
            .clone()
            .with_frontier_mode(OptimizationFrontierMode::Hybrid);
        assert_ne!(baseline.spec_digest, widened.spec_digest);
    }

    #[test]
    fn lineage_state_registers_roots_and_children() {
        let run_spec = OptimizationRunSpec::new("run_a", "probe.tool_route");
        let mut state = OptimizationLineageState::new(run_spec);
        let baseline = OptimizationCandidateManifest::new(
            "baseline",
            "probe.tool_route",
            "run_a",
            route_components("read_file"),
        );
        state
            .register_candidate(baseline)
            .expect("register baseline");

        let candidate = OptimizationCandidateManifest::new(
            "candidate",
            "probe.tool_route",
            "run_a",
            route_components("apply_patch"),
        )
        .with_parent_candidate_ids(vec![String::from("baseline")]);
        state
            .register_candidate(candidate)
            .expect("register candidate");

        assert_eq!(state.root_candidate_ids, vec![String::from("baseline")]);
        assert_eq!(
            state.discovery_order,
            vec![String::from("baseline"), String::from("candidate")]
        );
        assert_eq!(state.retained_candidate_ids.len(), 2);
    }

    #[test]
    fn lineage_state_rejects_unknown_parent() {
        let run_spec = OptimizationRunSpec::new("run_a", "probe.tool_route");
        let mut state = OptimizationLineageState::new(run_spec);
        let candidate = OptimizationCandidateManifest::new(
            "candidate",
            "probe.tool_route",
            "run_a",
            route_components("apply_patch"),
        )
        .with_parent_candidate_ids(vec![String::from("missing_parent")]);
        let error = state
            .register_candidate(candidate)
            .expect_err("unknown parent should fail");
        assert_eq!(
            error,
            OptimizationLineageStateError::UnknownParent {
                candidate_id: String::from("candidate"),
                parent_candidate_id: String::from("missing_parent"),
            }
        );
    }

    #[test]
    fn lineage_state_round_trips_through_json() {
        let run_spec = OptimizationRunSpec::new("run_a", "probe.tool_route")
            .with_issue_ref("OpenAgentsInc/psionic#807")
            .with_iteration_budget(12)
            .with_candidate_budget(24);
        let mut state = OptimizationLineageState::new(run_spec);
        state
            .register_candidate(OptimizationCandidateManifest::new(
                "baseline",
                "probe.tool_route",
                "run_a",
                route_components("read_file"),
            ))
            .expect("register baseline");
        state
            .set_retained_candidates(vec![String::from("baseline")])
            .expect("retained set");

        let temp = tempdir().expect("tempdir");
        let path = temp.path().join("optimizer_state.json");
        state.write_json(&path).expect("write state");
        let loaded = OptimizationLineageState::read_json(&path).expect("read state");
        assert_eq!(loaded, state);
    }

    #[test]
    fn run_receipt_points_at_retained_candidates() {
        let run_spec = OptimizationRunSpec::new("run_a", "probe.tool_route");
        let mut state = OptimizationLineageState::new(run_spec);
        state
            .register_candidate(OptimizationCandidateManifest::new(
                "baseline",
                "probe.tool_route",
                "run_a",
                route_components("read_file"),
            ))
            .expect("register baseline");
        state
            .set_retained_candidates(vec![String::from("baseline")])
            .expect("retained set");

        let receipt = OptimizationRunReceipt::from_state(
            &state,
            vec![String::from("fixtures/optimizer/frontier_snapshot.json")],
            OptimizationStopReason::NoSearchRequired,
        );
        assert_eq!(receipt.run_spec_digest, state.run_spec.spec_digest);
        assert_eq!(receipt.lineage_state_digest, state.state_digest);
        assert_eq!(
            receipt.retained_candidate_ids,
            vec![String::from("baseline")]
        );
        assert!(!receipt.receipt_digest.is_empty());
    }

    #[test]
    fn evaluation_cache_round_trips_case_receipts_by_unified_key() {
        let candidate = OptimizationCandidateManifest::new(
            "baseline",
            "probe.tool_route",
            "run_a",
            route_components("read_file"),
        );
        let case = route_case("case_a", "read_file");
        let receipt = super::OptimizationCaseEvaluationReceipt::new(
            &candidate,
            &case,
            8200,
            BTreeMap::from([(String::from("correctness_bps"), 8200)]),
            OptimizationSharedFeedback::new("baseline route was correct")
                .with_details(vec![String::from("selected_tool matched retained label")]),
            BTreeMap::from([(
                String::from("selected_tool"),
                OptimizationComponentFeedback::new("route component matched retained label"),
            )]),
        );
        let mut cache = OptimizationEvaluationCache::default();
        cache.insert(&candidate, &case, receipt.clone());

        let cached = cache.lookup(&candidate, &case).expect("cache hit");
        assert_eq!(
            cached.cache_key,
            OptimizationEvaluationCache::cache_key(&candidate, &case)
        );
        assert_eq!(cached, &receipt);
    }

    #[test]
    fn batch_receipt_aggregates_scalar_and_objective_scores() {
        let candidate = OptimizationCandidateManifest::new(
            "baseline",
            "probe.tool_route",
            "run_a",
            route_components("read_file"),
        );
        let case_a = route_case("case_a", "read_file");
        let case_b = route_case("case_b", "apply_patch");
        let receipt_a = super::OptimizationCaseEvaluationReceipt::new(
            &candidate,
            &case_a,
            5000,
            BTreeMap::from([(String::from("correctness_bps"), 5000)]),
            OptimizationSharedFeedback::new("first case"),
            BTreeMap::new(),
        );
        let receipt_b = super::OptimizationCaseEvaluationReceipt::new(
            &candidate,
            &case_b,
            4200,
            BTreeMap::from([
                (String::from("correctness_bps"), 4200),
                (String::from("latency_budget_bps"), 9700),
            ]),
            OptimizationSharedFeedback::new("second case"),
            BTreeMap::new(),
        );
        let batch = OptimizationBatchEvaluationReceipt::new(
            "run_a",
            &candidate,
            vec![receipt_a, receipt_b],
            1,
            1,
        );
        assert_eq!(batch.aggregated_scalar_score, 9200);
        assert_eq!(
            batch.aggregated_objective_scores.get("correctness_bps"),
            Some(&9200)
        );
        assert_eq!(
            batch.aggregated_objective_scores.get("latency_budget_bps"),
            Some(&9700)
        );
        assert_eq!(batch.cache_hit_count, 1);
        assert_eq!(batch.cache_miss_count, 1);
        assert!(!batch.receipt_digest.is_empty());
    }

    #[test]
    fn frontier_snapshot_keeps_case_and_objective_winners_explicit() {
        let baseline = OptimizationCandidateManifest::new(
            "baseline",
            "probe.tool_route",
            "run_a",
            route_components("read_file"),
        );
        let candidate = OptimizationCandidateManifest::new(
            "candidate",
            "probe.tool_route",
            "run_a",
            route_components("apply_patch"),
        )
        .with_parent_candidate_ids(vec![String::from("baseline")]);
        let case_a = route_case("case_a", "read_file");
        let case_b = route_case("case_b", "apply_patch");
        let baseline_batch = OptimizationBatchEvaluationReceipt::new(
            "run_a",
            &baseline,
            vec![
                super::OptimizationCaseEvaluationReceipt::new(
                    &baseline,
                    &case_a,
                    8000,
                    BTreeMap::from([
                        (String::from("correctness_bps"), 8000),
                        (String::from("latency_budget_bps"), 9600),
                    ]),
                    OptimizationSharedFeedback::new("baseline case a"),
                    BTreeMap::new(),
                ),
                super::OptimizationCaseEvaluationReceipt::new(
                    &baseline,
                    &case_b,
                    4000,
                    BTreeMap::from([
                        (String::from("correctness_bps"), 4000),
                        (String::from("latency_budget_bps"), 9600),
                    ]),
                    OptimizationSharedFeedback::new("baseline case b"),
                    BTreeMap::new(),
                ),
            ],
            0,
            2,
        );
        let candidate_batch = OptimizationBatchEvaluationReceipt::new(
            "run_a",
            &candidate,
            vec![
                super::OptimizationCaseEvaluationReceipt::new(
                    &candidate,
                    &case_a,
                    7800,
                    BTreeMap::from([
                        (String::from("correctness_bps"), 7800),
                        (String::from("latency_budget_bps"), 9900),
                    ]),
                    OptimizationSharedFeedback::new("candidate case a"),
                    BTreeMap::new(),
                ),
                super::OptimizationCaseEvaluationReceipt::new(
                    &candidate,
                    &case_b,
                    9100,
                    BTreeMap::from([
                        (String::from("correctness_bps"), 9100),
                        (String::from("latency_budget_bps"), 9900),
                    ]),
                    OptimizationSharedFeedback::new("candidate case b"),
                    BTreeMap::new(),
                ),
            ],
            0,
            2,
        );

        let frontier = OptimizationFrontierSnapshot::from_batches(
            "run_a",
            OptimizationFrontierMode::Hybrid,
            &[baseline_batch, candidate_batch],
        );
        assert_eq!(frontier.case_frontier.len(), 2);
        assert_eq!(frontier.objective_frontier.len(), 2);
        assert!(frontier
            .case_frontier
            .iter()
            .any(|row| row.case_id == "case_a" && row.winning_candidate_id == "baseline"));
        assert!(frontier
            .case_frontier
            .iter()
            .any(|row| row.case_id == "case_b" && row.winning_candidate_id == "candidate"));
        assert!(frontier
            .objective_frontier
            .iter()
            .any(|row| row.objective_name == "latency_budget_bps"
                && row.winning_candidate_id == "candidate"));
        assert_eq!(
            frontier.hybrid_candidate_ids,
            vec![String::from("baseline"), String::from("candidate")]
        );
        assert!(!frontier.snapshot_digest.is_empty());
    }
}
