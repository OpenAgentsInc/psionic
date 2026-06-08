use std::error::Error;

use psionic_train::{
    canonical_probe_gepa_stage_0_1_candidate_manifest, import_probe_gepa_live_closeout,
    ProbeGepaCandidatePromotionState, ProbeGepaCoordinatorState, ProbeGepaFailureClass,
    ProbeGepaLiveCloseoutImport, ProbeGepaLiveCloseoutState, ProbeGepaLivePaymentMode,
    ProbeGepaLivePublicClaim, PROBE_GEPA_LIVE_CLOSEOUT_IMPORT_SCHEMA_VERSION,
};

fn main() -> Result<(), Box<dyn Error>> {
    let candidate = canonical_probe_gepa_stage_0_1_candidate_manifest()?;
    let mut state = ProbeGepaCoordinatorState::default();
    let import = ProbeGepaLiveCloseoutImport {
        schema_version: PROBE_GEPA_LIVE_CLOSEOUT_IMPORT_SCHEMA_VERSION.to_string(),
        assignment_id: "benchmark_run.probe.shc_harbor.db_wal_recovery.20260608".to_string(),
        closeout_ref: "probe_closeout.probe.shc_harbor.db_wal_recovery.20260608".to_string(),
        campaign_id: candidate.campaign_id.clone(),
        task_id: "terminal-bench/db-wal-recovery".to_string(),
        dataset: "terminal_bench_2".to_string(),
        split_ref: candidate.split_refs[0].clone(),
        split: "retained".to_string(),
        probe_commit: "probe.main.archived-refactor".to_string(),
        agent_slug: "probe".to_string(),
        backend_model_ref: "backend.codex.gpt-5.5.harbor.v1".to_string(),
        candidate_hash: candidate.candidate_hash.clone(),
        candidate_id: candidate.candidate_id.clone(),
        selected_signature_refs: vec![
            "program_signature.probe.benchmark.runner_supervision.v1".to_string()
        ],
        tool_menu_ref: "tool_menu.probe.terminal_bench.db_wal_recovery.v1".to_string(),
        verifier_refs: vec![
            "harbor_job.e487217a-715e-448c-8d45-e528b76980e7".to_string(),
            "harbor_trial.a6c6c245-b9c0-44a8-a8c0-0c7fe5cc3383".to_string(),
            "harbor_result.sha256:1a5bba9286f507a5e6923c0a67d94a0a9d8801cd016f50fec84ff49c08629528"
                .to_string(),
        ],
        closeout_state: ProbeGepaLiveCloseoutState::AgentFailure,
        payment_mode: ProbeGepaLivePaymentMode::UnpaidSmoke,
        settlement_receipt_refs: Vec::new(),
        scalar_score_bps: 0,
        failure_family: ProbeGepaFailureClass::RunnerSupervision,
        artifact_manifest_ref: "artifact_manifest.probe.shc_harbor.db_wal_recovery.20260608"
            .to_string(),
        proof_bundle_ref: "proof_bundle.probe.shc_harbor.db_wal_recovery.20260608".to_string(),
        resource_usage_ref: "resource_usage_unavailable.probe.shc_harbor.db_wal_recovery.20260608"
            .to_string(),
        route_scorecard_ref: "route_scorecard.probe.shc_harbor.db_wal_recovery.20260608"
            .to_string(),
        policy_findings: vec!["no_public_benchmark_score_claimed".to_string()],
        duration_ms: 61_000,
        cost_micros: 0,
        public_claim: ProbeGepaLivePublicClaim::None,
        runtime_promotion_claim: ProbeGepaCandidatePromotionState::Draft,
        model_training_authority_claimed: false,
    };

    let receipt = import_probe_gepa_live_closeout(&mut state, &candidate, import)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
