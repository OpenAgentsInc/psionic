use std::{
    fs,
    path::{Path, PathBuf},
};

#[cfg(test)]
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const TASSADAR_POST_ARTICLE_PLUGIN_MANIFEST_IDENTITY_CONTRACT_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_post_article_plugin_manifest_identity_contract_report.json";
pub const TASSADAR_POST_ARTICLE_PLUGIN_MANIFEST_IDENTITY_CONTRACT_CHECKER_REF: &str =
    "scripts/check-tassadar-post-article-plugin-manifest-identity-contract.sh";

const PLUGIN_CHARTER_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_post_article_plugin_charter_authority_boundary_report.json";
const MODULE_TRUST_ISOLATION_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_module_trust_isolation_report.json";
const MODULE_PROMOTION_STATE_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_module_promotion_state_report.json";
const INTERNAL_COMPUTE_PACKAGE_MANAGER_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_internal_compute_package_manager_report.json";
const INTERNAL_COMPUTE_PACKAGE_ROUTE_POLICY_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_internal_compute_package_route_policy_report.json";
const LOCAL_PLUGIN_SYSTEM_SPEC_REF: &str = "~/code/alpha/tassadar/plugin-system.md";
const CANONICAL_ARCHITECTURE_BOUNDARY_REF: &str =
    "docs/TASSADAR_ARTICLE_TRANSFORMER_STACK_BOUNDARY.md";
const CANONICAL_ARCHITECTURE_ANCHOR_CRATE: &str = "psionic-transformer";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarPostArticlePluginManifestIdentityContractStatus {
    Green,
    Incomplete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarPostArticlePluginManifestDependencyClass {
    ProofCarrying,
    GovernanceDependency,
    CatalogDependency,
    RoutePolicyDependency,
    DesignInput,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPostArticlePluginManifestMachineIdentityBinding {
    pub machine_identity_id: String,
    pub canonical_model_id: String,
    pub canonical_route_id: String,
    pub canonical_route_descriptor_digest: String,
    pub canonical_weight_bundle_digest: String,
    pub canonical_weight_primary_artifact_sha256: String,
    pub continuation_contract_id: String,
    pub continuation_contract_digest: String,
    pub computational_model_statement_id: String,
    pub plugin_charter_report_id: String,
    pub plugin_charter_report_digest: String,
    pub canonical_architecture_anchor_crate: String,
    pub canonical_architecture_boundary_ref: String,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPostArticlePluginManifestDependencyRow {
    pub dependency_id: String,
    pub dependency_class: TassadarPostArticlePluginManifestDependencyClass,
    pub satisfied: bool,
    pub source_ref: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound_report_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound_report_digest: Option<String>,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPostArticlePluginManifestFieldRow {
    pub field_id: String,
    pub current_posture: String,
    pub green: bool,
    pub source_refs: Vec<String>,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPostArticlePluginInvocationIdentityRow {
    pub identity_id: String,
    pub required_fields: Vec<String>,
    pub current_posture: String,
    pub green: bool,
    pub source_refs: Vec<String>,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPostArticlePluginHotSwapRuleRow {
    pub rule_id: String,
    pub current_posture: String,
    pub green: bool,
    pub source_refs: Vec<String>,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPostArticlePluginPackagingRow {
    pub packaging_id: String,
    pub current_posture: String,
    pub green: bool,
    pub source_refs: Vec<String>,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPostArticlePluginManifestValidationRow {
    pub validation_id: String,
    pub green: bool,
    pub source_refs: Vec<String>,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarPostArticlePluginManifestIdentityContractReport {
    pub schema_version: u16,
    pub report_id: String,
    pub checker_script_ref: String,
    pub plugin_charter_report_ref: String,
    pub module_trust_isolation_report_ref: String,
    pub module_promotion_state_report_ref: String,
    pub internal_compute_package_manager_report_ref: String,
    pub internal_compute_package_route_policy_report_ref: String,
    pub local_plugin_system_spec_ref: String,
    pub supporting_material_refs: Vec<String>,
    pub machine_identity_binding: TassadarPostArticlePluginManifestMachineIdentityBinding,
    pub dependency_rows: Vec<TassadarPostArticlePluginManifestDependencyRow>,
    pub manifest_field_rows: Vec<TassadarPostArticlePluginManifestFieldRow>,
    pub invocation_identity_rows: Vec<TassadarPostArticlePluginInvocationIdentityRow>,
    pub hot_swap_rule_rows: Vec<TassadarPostArticlePluginHotSwapRuleRow>,
    pub packaging_rows: Vec<TassadarPostArticlePluginPackagingRow>,
    pub validation_rows: Vec<TassadarPostArticlePluginManifestValidationRow>,
    pub contract_status: TassadarPostArticlePluginManifestIdentityContractStatus,
    pub contract_green: bool,
    pub operator_internal_only_posture: bool,
    pub manifest_fields_frozen: bool,
    pub canonical_invocation_identity_frozen: bool,
    pub hot_swap_rules_frozen: bool,
}