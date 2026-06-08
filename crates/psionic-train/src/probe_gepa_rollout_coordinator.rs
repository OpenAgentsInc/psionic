use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    ProbeGepaCandidateManifest, ProbeGepaCandidateManifestError,
    ProbeGepaCandidatePromotionState, ProbeGepaPolicyGateState,
    validate_probe_gepa_candidate_manifest,
};

pub const PROBE_GEPA_COORDINATOR_STATE_SCHEMA_VERSION: &str =
    "psionic.probe_gepa_rollout_coordinator_state.v1";
pub const PROBE_GEPA_ROLLOUT_ASSIGNMENT_SCHEMA_VERSION: &str =
    "psionic.probe_gepa_rollout_assignment.v1";
pub const PROBE_GEPA_ROLLOUT_RESULT_SCHEMA_VERSION: &str =
    "psionic.probe_gepa_rollout_result.v1";
pub const PROBE_GEPA_STAGE_0_CAMPAIGN_ID: &str = "probe_gepa.terminal_bench.stage_0_1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeGepaEvaluatorBackend {
    Local,
    PylonPending,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeGepaVerifierStatus {
    Passed,
    Failed,
    TimedOut,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeGepaRolloutStatus {
    Succeeded,
    AgentFailed,
    InfrastructureFailed,
    PolicyBlocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeGepaFailureClass {
    None,
    ServiceReadiness,
    ParserCorrectness,
    RunnerSupervision,
    AgentRegression,
    InfrastructureUnavailable,
    PolicyViolation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeGepaRolloutTask {
    pub task_id: String,
    pub dataset: String,
    pub split_ref: String,
    pub split: String,
    pub expected_failure_family: ProbeGepaFailureClass,
    pub selected_signature_refs: Vec<String>,
    pub tool_menu_ref: String,
    pub scorer_verifier_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeGepaRolloutAssignment {
    pub schema_version: String,
    pub rollout_ref: String,
    pub campaign_id: String,
    pub task_id: String,
    pub dataset: String,
    pub split_ref: String,
    pub split: String,
    pub probe_commit: String,
    pub agent_slug: String,
    pub backend_model_ref: String,
    pub candidate_hash: String,
    pub candidate_id: String,
    pub selected_signature_refs: Vec<String>,
    pub tool_menu_ref: String,
    pub evaluator_backend: ProbeGepaEvaluatorBackend,
    pub expected_artifact_refs: Vec<String>,
    pub timeout_policy_ref: String,
    pub budget_policy_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeGepaRolloutResult {
    pub schema_version: String,
    pub rollout_ref: String,
    pub task_id: String,
    pub dataset: String,
    pub split_ref: String,
    pub split: String,
    pub probe_commit: String,
    pub agent_slug: String,
    pub backend_model_ref: String,
    pub candidate_hash: String,
    pub candidate_id: String,
    pub selected_signature_refs: Vec<String>,
    pub tool_menu_ref: String,
    pub verifier_status: ProbeGepaVerifierStatus,
    pub rollout_status: ProbeGepaRolloutStatus,
    pub scalar_score_bps: u32,
    pub failure_family: ProbeGepaFailureClass,
    pub artifact_manifest_ref: Option<String>,
    pub proof_bundle_ref: Option<String>,
    pub resource_usage_ref: Option<String>,
    pub policy_findings: Vec<String>,
    pub duration_ms: u64,
    pub cost_micros: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeGepaEvaluationCache {
    pub results_by_cache_key: BTreeMap<String, ProbeGepaRolloutResult>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeGepaCandidateFrontierEntry {
    pub candidate_id: String,
    pub candidate_hash: String,
    pub parent_candidate_id: Option<String>,
    pub mean_score_bps: u32,
    pub successful_rollouts: u32,
    pub agent_failures: u32,
    pub infrastructure_failures: u32,
    pub policy_blocks: u32,
    pub accepted_for_reflection: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeGepaLineageEdge {
    pub parent_candidate_id: Option<String>,
    pub child_candidate_id: String,
    pub optimizer_run_id: String,
    pub reflection_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeGepaReflectionProposal {
    pub proposal_ref: String,
    pub source_candidate_id: String,
    pub source_candidate_hash: String,
    pub next_candidate_parent_id: String,
    pub failure_families_to_mutate: Vec<ProbeGepaFailureClass>,
    pub accepted: bool,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeGepaCoordinatorExports {
    pub probe_candidate_refs: Vec<String>,
    pub omega_candidate_refs: Vec<String>,
    pub artanis_candidate_refs: Vec<String>,
    pub benchmark_cloud_candidate_refs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeGepaIterationMetrics {
    pub iteration_index: u32,
    pub metric_call_count: u32,
    pub cache_hit_count: u32,
    pub local_rollout_count: u32,
    pub pylon_assignment_count: u32,
    pub infrastructure_failure_count: u32,
    pub agent_failure_count: u32,
    pub policy_block_count: u32,
    pub total_duration_ms: u64,
    pub total_cost_micros: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeGepaCoordinatorState {
    pub schema_version: String,
    pub campaign_id: String,
    pub iteration_index: u32,
    pub candidate_frontier: Vec<ProbeGepaCandidateFrontierEntry>,
    pub lineage: Vec<ProbeGepaLineageEdge>,
    pub completed_rollouts: Vec<ProbeGepaRolloutResult>,
    pub pending_pylon_assignments: Vec<ProbeGepaRolloutAssignment>,
    pub evaluation_cache: ProbeGepaEvaluationCache,
    pub reflection_proposals: Vec<ProbeGepaReflectionProposal>,
    pub exports: ProbeGepaCoordinatorExports,
    pub last_iteration_metrics: Option<ProbeGepaIterationMetrics>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeGepaCoordinatorConfig {
    pub campaign_id: String,
    pub probe_commit: String,
    pub agent_slug: String,
    pub backend_model_ref: String,
    pub local_first: bool,
    pub pylon_enabled: bool,
    pub target_metric_calls: u32,
}

#[derive(Clone, Debug, Error, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbeGepaCoordinatorError {
    #[error("candidate manifest invalid: {0}")]
    CandidateManifestInvalid(#[from] ProbeGepaCandidateManifestError),
    #[error("candidate `{candidate_id}` cannot advance because policy gate state is `{policy_gate_state:?}`")]
    CandidatePolicyBlocked {
        candidate_id: String,
        policy_gate_state: ProbeGepaPolicyGateState,
    },
    #[error("candidate `{candidate_id}` cannot advance from promotion state `{promotion_state:?}`")]
    CandidatePromotionBlocked {
        candidate_id: String,
        promotion_state: ProbeGepaCandidatePromotionState,
    },
    #[error("stage 0 requires 20 to 40 metric calls, found {target_metric_calls}")]
    InvalidStage0MetricCallCount { target_metric_calls: u32 },
}

impl Default for ProbeGepaCoordinatorState {
    fn default() -> Self {
        Self {
            schema_version: PROBE_GEPA_COORDINATOR_STATE_SCHEMA_VERSION.to_string(),
            campaign_id: PROBE_GEPA_STAGE_0_CAMPAIGN_ID.to_string(),
            iteration_index: 0,
            candidate_frontier: Vec::new(),
            lineage: Vec::new(),
            completed_rollouts: Vec::new(),
            pending_pylon_assignments: Vec::new(),
            evaluation_cache: ProbeGepaEvaluationCache::default(),
            reflection_proposals: Vec::new(),
            exports: ProbeGepaCoordinatorExports {
                probe_candidate_refs: Vec::new(),
                omega_candidate_refs: Vec::new(),
                artanis_candidate_refs: Vec::new(),
                benchmark_cloud_candidate_refs: Vec::new(),
            },
            last_iteration_metrics: None,
        }
    }
}

impl Default for ProbeGepaCoordinatorConfig {
    fn default() -> Self {
        Self {
            campaign_id: PROBE_GEPA_STAGE_0_CAMPAIGN_ID.to_string(),
            probe_commit: "probe.commit.local-stage0".to_string(),
            agent_slug: "probe".to_string(),
            backend_model_ref: "backend.local.probe.fake_metric_call.v1".to_string(),
            local_first: true,
            pylon_enabled: false,
            target_metric_calls: 20,
        }
    }
}

pub fn run_probe_gepa_stage0_iteration(
    state: &mut ProbeGepaCoordinatorState,
    candidate: &ProbeGepaCandidateManifest,
    config: &ProbeGepaCoordinatorConfig,
) -> Result<ProbeGepaIterationMetrics, ProbeGepaCoordinatorError> {
    validate_stage0_config(config)?;
    validate_candidate_can_advance(candidate)?;

    let tasks = canonical_probe_gepa_stage0_tasks(config.target_metric_calls);
    let mut metrics = ProbeGepaIterationMetrics {
        iteration_index: state.iteration_index,
        metric_call_count: 0,
        cache_hit_count: 0,
        local_rollout_count: 0,
        pylon_assignment_count: 0,
        infrastructure_failure_count: 0,
        agent_failure_count: 0,
        policy_block_count: 0,
        total_duration_ms: 0,
        total_cost_micros: 0,
    };

    for task in tasks {
        let assignment = build_probe_gepa_rollout_assignment(candidate, config, &task);
        let cache_key = rollout_cache_key(&assignment);
        let result = if let Some(cached) = state.evaluation_cache.results_by_cache_key.get(&cache_key) {
            metrics.cache_hit_count += 1;
            cached.clone()
        } else if config.local_first {
            metrics.local_rollout_count += 1;
            let result = run_local_probe_gepa_rollout(&assignment, &task);
            state
                .evaluation_cache
                .results_by_cache_key
                .insert(cache_key, result.clone());
            result
        } else {
            metrics.pylon_assignment_count += 1;
            state.pending_pylon_assignments.push(assignment);
            continue;
        };

        metrics.metric_call_count += 1;
        metrics.total_duration_ms += result.duration_ms;
        metrics.total_cost_micros += result.cost_micros;
        match result.rollout_status {
            ProbeGepaRolloutStatus::InfrastructureFailed => {
                metrics.infrastructure_failure_count += 1
            }
            ProbeGepaRolloutStatus::AgentFailed => metrics.agent_failure_count += 1,
            ProbeGepaRolloutStatus::PolicyBlocked => metrics.policy_block_count += 1,
            ProbeGepaRolloutStatus::Succeeded => {}
        }
        state.completed_rollouts.push(result);
    }

    let frontier_entry = summarize_candidate_frontier_entry(candidate, &state.completed_rollouts);
    upsert_frontier_entry(&mut state.candidate_frontier, frontier_entry.clone());
    let reflection = build_reflection_proposal(candidate, &state.completed_rollouts);
    state.lineage.push(ProbeGepaLineageEdge {
        parent_candidate_id: candidate.parent_candidate_id.clone(),
        child_candidate_id: candidate.candidate_id.clone(),
        optimizer_run_id: candidate.optimizer_run_id.clone(),
        reflection_ref: reflection.proposal_ref.clone(),
    });
    state.reflection_proposals.push(reflection);
    state.exports = build_candidate_exports(candidate);
    state.iteration_index += 1;
    state.last_iteration_metrics = Some(metrics.clone());
    Ok(metrics)
}

pub fn build_probe_gepa_rollout_assignment(
    candidate: &ProbeGepaCandidateManifest,
    config: &ProbeGepaCoordinatorConfig,
    task: &ProbeGepaRolloutTask,
) -> ProbeGepaRolloutAssignment {
    let rollout_ref = stable_rollout_ref(candidate.candidate_hash.as_str(), task.task_id.as_str());
    ProbeGepaRolloutAssignment {
        schema_version: PROBE_GEPA_ROLLOUT_ASSIGNMENT_SCHEMA_VERSION.to_string(),
        rollout_ref,
        campaign_id: config.campaign_id.clone(),
        task_id: task.task_id.clone(),
        dataset: task.dataset.clone(),
        split_ref: task.split_ref.clone(),
        split: task.split.clone(),
        probe_commit: config.probe_commit.clone(),
        agent_slug: config.agent_slug.clone(),
        backend_model_ref: config.backend_model_ref.clone(),
        candidate_hash: candidate.candidate_hash.clone(),
        candidate_id: candidate.candidate_id.clone(),
        selected_signature_refs: task.selected_signature_refs.clone(),
        tool_menu_ref: task.tool_menu_ref.clone(),
        evaluator_backend: if config.pylon_enabled && !config.local_first {
            ProbeGepaEvaluatorBackend::PylonPending
        } else {
            ProbeGepaEvaluatorBackend::Local
        },
        expected_artifact_refs: vec![
            "artifact_manifest.probe_gepa_rollout.required.v1".to_string(),
            "proof_bundle.probe_gepa_rollout.required.v1".to_string(),
            "resource_usage.probe_gepa_rollout.required.v1".to_string(),
        ],
        timeout_policy_ref: "timeout_policy.probe_gepa.stage0.v1".to_string(),
        budget_policy_ref: "budget_policy.probe_gepa.local_zero_cost.v1".to_string(),
    }
}

pub fn run_local_probe_gepa_rollout(
    assignment: &ProbeGepaRolloutAssignment,
    task: &ProbeGepaRolloutTask,
) -> ProbeGepaRolloutResult {
    let infrastructure_failure = task.task_id.contains("infra-unavailable");
    let agent_failure = task.task_id.contains("agent-regression");
    let policy_blocked = task.task_id.contains("policy-blocked");
    let rollout_status = if infrastructure_failure {
        ProbeGepaRolloutStatus::InfrastructureFailed
    } else if policy_blocked {
        ProbeGepaRolloutStatus::PolicyBlocked
    } else if agent_failure {
        ProbeGepaRolloutStatus::AgentFailed
    } else {
        ProbeGepaRolloutStatus::Succeeded
    };
    let verifier_status = match rollout_status {
        ProbeGepaRolloutStatus::Succeeded => ProbeGepaVerifierStatus::Passed,
        ProbeGepaRolloutStatus::AgentFailed => ProbeGepaVerifierStatus::Failed,
        ProbeGepaRolloutStatus::InfrastructureFailed => ProbeGepaVerifierStatus::Error,
        ProbeGepaRolloutStatus::PolicyBlocked => ProbeGepaVerifierStatus::Error,
    };
    let failure_family = match rollout_status {
        ProbeGepaRolloutStatus::Succeeded => ProbeGepaFailureClass::None,
        ProbeGepaRolloutStatus::AgentFailed => task.expected_failure_family,
        ProbeGepaRolloutStatus::InfrastructureFailed => ProbeGepaFailureClass::InfrastructureUnavailable,
        ProbeGepaRolloutStatus::PolicyBlocked => ProbeGepaFailureClass::PolicyViolation,
    };

    ProbeGepaRolloutResult {
        schema_version: PROBE_GEPA_ROLLOUT_RESULT_SCHEMA_VERSION.to_string(),
        rollout_ref: assignment.rollout_ref.clone(),
        task_id: assignment.task_id.clone(),
        dataset: assignment.dataset.clone(),
        split_ref: assignment.split_ref.clone(),
        split: assignment.split.clone(),
        probe_commit: assignment.probe_commit.clone(),
        agent_slug: assignment.agent_slug.clone(),
        backend_model_ref: assignment.backend_model_ref.clone(),
        candidate_hash: assignment.candidate_hash.clone(),
        candidate_id: assignment.candidate_id.clone(),
        selected_signature_refs: assignment.selected_signature_refs.clone(),
        tool_menu_ref: assignment.tool_menu_ref.clone(),
        verifier_status,
        rollout_status,
        scalar_score_bps: if rollout_status == ProbeGepaRolloutStatus::Succeeded {
            10_000
        } else {
            0
        },
        failure_family,
        artifact_manifest_ref: Some(format!("artifact_manifest.{}", assignment.rollout_ref)),
        proof_bundle_ref: Some(format!("proof_bundle.{}", assignment.rollout_ref)),
        resource_usage_ref: Some(format!("resource_usage.{}", assignment.rollout_ref)),
        policy_findings: if policy_blocked {
            vec!["candidate_or_assignment_policy_blocked".to_string()]
        } else {
            Vec::new()
        },
        duration_ms: if infrastructure_failure { 250 } else { 1_000 },
        cost_micros: 0,
    }
}

pub fn canonical_probe_gepa_stage0_tasks(target_metric_calls: u32) -> Vec<ProbeGepaRolloutTask> {
    let families = [
        ("configure-git-webserver", ProbeGepaFailureClass::ServiceReadiness),
        ("filter-js-from-html", ProbeGepaFailureClass::ParserCorrectness),
        ("gcode-to-text", ProbeGepaFailureClass::ParserCorrectness),
        ("runner-stall-supervision", ProbeGepaFailureClass::RunnerSupervision),
    ];
    (0..target_metric_calls)
        .map(|index| {
            let (base_task, family) = families[index as usize % families.len()];
            let split = if index < target_metric_calls / 2 {
                "retained"
            } else {
                "validation"
            };
            ProbeGepaRolloutTask {
                task_id: format!("{base_task}.metric_call_{index:02}"),
                dataset: "terminal_bench_2".to_string(),
                split_ref: "benchmark_split_manifest.terminal_bench_2.probe_gepa.stage_0_1.v1".to_string(),
                split: split.to_string(),
                expected_failure_family: family,
                selected_signature_refs: vec![format!(
                    "program_signature.probe.benchmark.{family:?}.v1"
                )],
                tool_menu_ref: format!("tool_menu.probe.terminal_bench.{base_task}.v1"),
                scorer_verifier_ref: format!("verifier.terminal_bench.{base_task}.v1"),
            }
        })
        .collect()
}

fn validate_stage0_config(config: &ProbeGepaCoordinatorConfig) -> Result<(), ProbeGepaCoordinatorError> {
    if !(20..=40).contains(&config.target_metric_calls) {
        return Err(ProbeGepaCoordinatorError::InvalidStage0MetricCallCount {
            target_metric_calls: config.target_metric_calls,
        });
    }
    Ok(())
}

fn validate_candidate_can_advance(
    candidate: &ProbeGepaCandidateManifest,
) -> Result<(), ProbeGepaCoordinatorError> {
    validate_probe_gepa_candidate_manifest(candidate)?;
    if candidate.safety_boundary.public_claim_upgrade_authority
        || !candidate.safety_boundary.no_new_runtime_authority
    {
        return Err(ProbeGepaCoordinatorError::CandidatePolicyBlocked {
            candidate_id: candidate.candidate_id.clone(),
            policy_gate_state: candidate.policy_gate_state,
        });
    }
    if matches!(candidate.policy_gate_state, ProbeGepaPolicyGateState::Failed | ProbeGepaPolicyGateState::Blocked) {
        return Err(ProbeGepaCoordinatorError::CandidatePolicyBlocked {
            candidate_id: candidate.candidate_id.clone(),
            policy_gate_state: candidate.policy_gate_state,
        });
    }
    if matches!(
        candidate.promotion_state,
        ProbeGepaCandidatePromotionState::Active | ProbeGepaCandidatePromotionState::Reverted
    ) {
        return Err(ProbeGepaCoordinatorError::CandidatePromotionBlocked {
            candidate_id: candidate.candidate_id.clone(),
            promotion_state: candidate.promotion_state,
        });
    }
    Ok(())
}

fn summarize_candidate_frontier_entry(
    candidate: &ProbeGepaCandidateManifest,
    completed_rollouts: &[ProbeGepaRolloutResult],
) -> ProbeGepaCandidateFrontierEntry {
    let candidate_results = completed_rollouts
        .iter()
        .filter(|result| result.candidate_hash == candidate.candidate_hash)
        .collect::<Vec<_>>();
    let count = candidate_results.len().max(1) as u32;
    let total_score = candidate_results
        .iter()
        .map(|result| result.scalar_score_bps)
        .sum::<u32>();
    let infrastructure_failures = candidate_results
        .iter()
        .filter(|result| result.rollout_status == ProbeGepaRolloutStatus::InfrastructureFailed)
        .count() as u32;
    let agent_failures = candidate_results
        .iter()
        .filter(|result| result.rollout_status == ProbeGepaRolloutStatus::AgentFailed)
        .count() as u32;
    let policy_blocks = candidate_results
        .iter()
        .filter(|result| result.rollout_status == ProbeGepaRolloutStatus::PolicyBlocked)
        .count() as u32;

    ProbeGepaCandidateFrontierEntry {
        candidate_id: candidate.candidate_id.clone(),
        candidate_hash: candidate.candidate_hash.clone(),
        parent_candidate_id: candidate.parent_candidate_id.clone(),
        mean_score_bps: total_score / count,
        successful_rollouts: candidate_results
            .iter()
            .filter(|result| result.rollout_status == ProbeGepaRolloutStatus::Succeeded)
            .count() as u32,
        agent_failures,
        infrastructure_failures,
        policy_blocks,
        accepted_for_reflection: policy_blocks == 0,
    }
}

fn build_reflection_proposal(
    candidate: &ProbeGepaCandidateManifest,
    completed_rollouts: &[ProbeGepaRolloutResult],
) -> ProbeGepaReflectionProposal {
    let mut families = BTreeSet::new();
    for result in completed_rollouts
        .iter()
        .filter(|result| result.candidate_hash == candidate.candidate_hash)
        .filter(|result| result.rollout_status == ProbeGepaRolloutStatus::AgentFailed)
    {
        families.insert(result.failure_family);
    }
    ProbeGepaReflectionProposal {
        proposal_ref: format!("gepa_reflection.{}", short_hash(candidate.candidate_hash.as_str())),
        source_candidate_id: candidate.candidate_id.clone(),
        source_candidate_hash: candidate.candidate_hash.clone(),
        next_candidate_parent_id: candidate.candidate_id.clone(),
        failure_families_to_mutate: families.into_iter().collect(),
        accepted: true,
        detail: "Central GEPA reflection/proposal step retained for the next candidate mutation.".to_string(),
    }
}

fn build_candidate_exports(candidate: &ProbeGepaCandidateManifest) -> ProbeGepaCoordinatorExports {
    ProbeGepaCoordinatorExports {
        probe_candidate_refs: vec![
            candidate.probe_import.prompt_candidate_ref.clone(),
            candidate.probe_import.blueprint_candidate_ref.clone(),
            candidate.probe_import.tool_menu_candidate_ref.clone(),
            candidate.probe_import.loop_policy_candidate_ref.clone(),
        ],
        omega_candidate_refs: vec![format!("omega.probe_candidate_projection.{}", candidate.candidate_id)],
        artanis_candidate_refs: vec![format!("artanis.probe_candidate_projection.{}", candidate.candidate_id)],
        benchmark_cloud_candidate_refs: candidate
            .benchmark_cloud_import
            .benchmark_run_manifest_refs
            .clone(),
    }
}

fn upsert_frontier_entry(
    frontier: &mut Vec<ProbeGepaCandidateFrontierEntry>,
    entry: ProbeGepaCandidateFrontierEntry,
) {
    if let Some(existing) = frontier
        .iter_mut()
        .find(|existing| existing.candidate_hash == entry.candidate_hash)
    {
        *existing = entry;
    } else {
        frontier.push(entry);
    }
}

fn rollout_cache_key(assignment: &ProbeGepaRolloutAssignment) -> String {
    stable_sha256(format!(
        "{}|{}|{}|{}|{}",
        assignment.candidate_hash,
        assignment.task_id,
        assignment.probe_commit,
        assignment.backend_model_ref,
        assignment.evaluator_backend as u8
    ).as_bytes())
}

fn stable_rollout_ref(candidate_hash: &str, task_id: &str) -> String {
    format!(
        "probe_gepa_rollout.{}",
        short_hash(stable_sha256(format!("{candidate_hash}|{task_id}").as_bytes()).as_str())
    )
}

fn stable_sha256(bytes: &[u8]) -> String {
    format!("sha256:{}", hex::encode(Sha256::digest(bytes)))
}

fn short_hash(hash: &str) -> &str {
    hash.strip_prefix("sha256:")
        .and_then(|suffix| suffix.get(..16))
        .unwrap_or(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_probe_gepa_stage_0_1_candidate_manifest;

    #[test]
    fn local_evaluator_runs_stage0_metric_calls_before_pylon() {
        let candidate = canonical_probe_gepa_stage_0_1_candidate_manifest().unwrap();
        let mut state = ProbeGepaCoordinatorState::default();
        let config = ProbeGepaCoordinatorConfig::default();

        let metrics = run_probe_gepa_stage0_iteration(&mut state, &candidate, &config).unwrap();

        assert_eq!(metrics.metric_call_count, 20);
        assert_eq!(metrics.local_rollout_count, 20);
        assert_eq!(metrics.pylon_assignment_count, 0);
        assert_eq!(state.completed_rollouts.len(), 20);
        assert_eq!(state.evaluation_cache.results_by_cache_key.len(), 20);
        assert_eq!(state.candidate_frontier.len(), 1);
        assert_eq!(state.candidate_frontier[0].candidate_hash, candidate.candidate_hash);
        assert!(state.pending_pylon_assignments.is_empty());
        assert!(state.exports.probe_candidate_refs.contains(&candidate.probe_import.prompt_candidate_ref));
    }

    #[test]
    fn coordinator_resumes_from_evaluation_cache() {
        let candidate = canonical_probe_gepa_stage_0_1_candidate_manifest().unwrap();
        let mut state = ProbeGepaCoordinatorState::default();
        let config = ProbeGepaCoordinatorConfig::default();

        run_probe_gepa_stage0_iteration(&mut state, &candidate, &config).unwrap();
        state.completed_rollouts.clear();
        state.iteration_index = 1;
        let metrics = run_probe_gepa_stage0_iteration(&mut state, &candidate, &config).unwrap();

        assert_eq!(metrics.metric_call_count, 20);
        assert_eq!(metrics.cache_hit_count, 20);
        assert_eq!(metrics.local_rollout_count, 0);
        assert_eq!(state.completed_rollouts.len(), 20);
        assert_eq!(state.iteration_index, 2);
    }

    #[test]
    fn rollout_status_distinguishes_infrastructure_from_agent_failure() {
        let candidate = canonical_probe_gepa_stage_0_1_candidate_manifest().unwrap();
        let config = ProbeGepaCoordinatorConfig::default();
        let infra_task = ProbeGepaRolloutTask {
            task_id: "infra-unavailable.metric_call_00".to_string(),
            dataset: "terminal_bench_2".to_string(),
            split_ref: "split.ref".to_string(),
            split: "retained".to_string(),
            expected_failure_family: ProbeGepaFailureClass::ServiceReadiness,
            selected_signature_refs: vec!["signature.ref".to_string()],
            tool_menu_ref: "tool.ref".to_string(),
            scorer_verifier_ref: "verifier.ref".to_string(),
        };
        let agent_task = ProbeGepaRolloutTask {
            task_id: "agent-regression.metric_call_01".to_string(),
            ..infra_task.clone()
        };

        let infra = run_local_probe_gepa_rollout(
            &build_probe_gepa_rollout_assignment(&candidate, &config, &infra_task),
            &infra_task,
        );
        let agent = run_local_probe_gepa_rollout(
            &build_probe_gepa_rollout_assignment(&candidate, &config, &agent_task),
            &agent_task,
        );

        assert_eq!(infra.rollout_status, ProbeGepaRolloutStatus::InfrastructureFailed);
        assert_eq!(infra.failure_family, ProbeGepaFailureClass::InfrastructureUnavailable);
        assert_eq!(agent.rollout_status, ProbeGepaRolloutStatus::AgentFailed);
        assert_eq!(agent.failure_family, ProbeGepaFailureClass::ServiceReadiness);
    }

    #[test]
    fn policy_violating_candidates_cannot_advance() {
        let mut candidate = canonical_probe_gepa_stage_0_1_candidate_manifest().unwrap();
        candidate.safety_boundary.public_claim_upgrade_authority = true;
        let mut state = ProbeGepaCoordinatorState::default();
        let config = ProbeGepaCoordinatorConfig::default();

        assert!(matches!(
            run_probe_gepa_stage0_iteration(&mut state, &candidate, &config),
            Err(ProbeGepaCoordinatorError::CandidateManifestInvalid(_))
        ));
        assert!(state.completed_rollouts.is_empty());
        assert!(state.candidate_frontier.is_empty());
    }
}
