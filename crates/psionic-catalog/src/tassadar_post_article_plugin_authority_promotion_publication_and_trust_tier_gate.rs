use std::{
    fs,
    path::{Path, PathBuf},
};

#[cfg(test)]
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const TASSADAR_POST_ARTICLE_PLUGIN_AUTHORITY_PROMOTION_PUBLICATION_AND_TRUST_TIER_GATE_REPORT_REF:
    &str =
    "fixtures/tassadar/reports/tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate_report.json";
pub const TASSADAR_POST_ARTICLE_PLUGIN_AUTHORITY_PROMOTION_PUBLICATION_AND_TRUST_TIER_GATE_CHECKER_REF:
    &str =
    "scripts/check-tassadar-post-article-plugin-authority-promotion-publication-and-trust-tier-gate.sh";

const CONTROLLER_EVAL_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_post_article_weighted_plugin_controller_trace_and_refusal_aware_model_loop_eval_report.json";
const MANIFEST_CONTRACT_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_post_article_plugin_manifest_identity_contract_report.json";
const MODULE_PROMOTION_STATE_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_module_promotion_state_report.json";
const MODULE_TRUST_ISOLATION_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_module_trust_isolation_report.json";
const BROAD_INTERNAL_COMPUTE_PROFILE_PUBLICATION_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_broad_internal_compute_profile_publication_report.json";
const BROAD_INTERNAL_COMPUTE_ROUTE_POLICY_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_broad_internal_compute_route_policy_report.json";
const CANONICAL_MACHINE_CLOSURE_BUNDLE_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_post_article_canonical_machine_closure_bundle_report.json";
const STARTER_PLUGIN_CATALOG_RUNTIME_BUNDLE_REF: &str =
    "fixtures/tassadar/runs/tassadar_post_article_starter_plugin_catalog_v1/tassadar_post_article_starter_plugin_catalog_bundle.json";
const LOCAL_PLUGIN_SYSTEM_SPEC_REF: &str = "~/code/alpha/tassadar/plugin-system.md";
const PLUGIN_SYSTEM_AUDIT_REF: &str =
    "docs/audits/2026-03-20-tassadar-plugin-system-and-turing-completeness-audit.md";
const CANONICAL_ARCHITECTURE_BOUNDARY_REF: &str =
    "docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md";
const CANONICAL_ARCHITECTURE_ANCHOR_CRATE: &str = "psionic-transformer";

const ARTICLE_CLOSEOUT_PROFILE_ID: &str = "tassadar.internal_compute.article_closeout.v1";
const DETERMINISTIC_IMPORT_PROFILE_ID: &str =
    "tassadar.internal_compute.deterministic_import_subset.v1";
const RUNTIME_SUPPORT_PROFILE_ID: &str = "tassadar.internal_compute.runtime_support_subset.v1";
const PORTABLE_BROAD_PROFILE_ID: &str = "tassadar.internal_compute.portable_broad_family.v1";
const PUBLIC_BROAD_PROFILE_ID: &str = "tassadar.internal_compute.public_broad_family.v1";

const ROUTE_DETERMINISTIC_IMPORT: &str = "route.deterministic_import.subset";
const ROUTE_RUNTIME_SUPPORT: &str = "route.runtime_support.linked_bundle";
const ROUTE_PORTABLE_BROAD: &str = "route.portable_broad_family.declared_matrix";
const ROUTE_PUBLIC_BROAD: &str = "route.public_broad_family.publication";
const GUEST_ARTIFACT_PLUGIN_ID: &str = "plugin.example.echo_guest";
const GUEST_ARTIFACT_REPLAY_CLASS_ID: &str = "guest_artifact_digest_replay_only.v1";
const GUEST_ARTIFACT_TRUST_TIER_ID: &str =
    "operator_reviewed_guest_artifact_digest_bound_internal_only";
const GUEST_ARTIFACT_EVIDENCE_POSTURE_ID: &str =
    "evidence.manifest_digest_invocation_receipt_bound.v1";
const GUEST_ARTIFACT_PUBLICATION_NOT_CLAIMED_NEGATIVE_CLAIM_ID: &str =
    "guest_artifact_publication_not_claimed";
const GUEST_ARTIFACT_ARBITRARY_WASM_NOT_CLAIMED_NEGATIVE_CLAIM_ID: &str =
    "arbitrary_wasm_plugin_support_not_claimed";
const GUEST_ARTIFACT_SECRET_OR_STATEFUL_NOT_CLAIMED_NEGATIVE_CLAIM_ID: &str =
    "secret_or_stateful_guest_capability_not_claimed";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarPostArticlePluginAuthorityPromotionPublicationAndTrustTierGateStatus {
    Green,
    Incomplete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarPostArticlePluginAuthorityDependencyClass {
    ControllerPrecedent,
    CatalogPrecedent,
    PromotionDependency,
    TrustDependency,
    PublicationDependency,
    RoutePolicyDependency,
    ClosureBundle,
    DesignInput,
    AuditInput,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPostArticlePluginAuthorityMachineIdentityBinding {
    pub machine_identity_id: String,
    pub canonical_model_id: String,
    pub canonical_route_id: String,
    pub canonical_route_descriptor_digest: String,
    pub canonical_weight_bundle_digest: String,
    pub canonical_weight_primary_artifact_sha256: String,
    pub continuation_contract_id: String,
    pub continuation_contract_digest: String,
    pub computational_model_statement_id: String,
    pub controller_eval_report_id: String,
    pub controller_eval_report_digest: String,
    pub control_trace_contract_id: String,
    pub closure_bundle_report_id: String,
    pub closure_bundle_report_digest: String,
    pub closure_bundle_digest: String,
    pub manifest_contract_report_id: String,
    pub manifest_contract_report_digest: String,
    pub canonical_architecture_anchor_crate: String,
    pub canonical_architecture_boundary_ref: String,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPostArticlePluginAuthorityDependencyRow {
    pub dependency_id: String,
    pub dependency_class: TassadarPostArticlePluginAuthorityDependencyClass,
    pub satisfied: bool,
    pub source_ref: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound_report_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound_report_digest: Option<String>,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPostArticlePluginAuthorityPromotionPublicationAndTrustTierGate {
    pub status: TassadarPostArticlePluginAuthorityPromotionPublicationAndTrustTierGateStatus,
    pub machine_identity_bindings: Vec<TassadarPostArticlePluginAuthorityMachineIdentityBinding>,
    pub dependencies: Vec<TassadarPostArticlePluginAuthorityDependencyRow>,
}