use psionic_runtime::{
    TassadarModuleExecutionCapabilityReport, tassadar_module_execution_capability_report,
};
use serde::{Deserialize, Serialize};

use crate::TassadarExecutorFixture;

/// Repo-facing publication for the bounded module-execution lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarModuleExecutionCapabilityPublication {
    /// Stable publication identifier.
    pub publication_id: String,
    /// Stable model identifier owning the publication.
    pub model_id: String,
    /// Coarse claim class for the publication.
    pub claim_class: String,
    /// Runtime capability report projected into model/publication space.
    pub runtime_capability: TassadarModuleExecutionCapabilityReport,
    /// Stable seeded case ids that anchor the publication today.
    pub seeded_case_ids: Vec<String>,
    /// Plain-language claim boundary for the publication.
    pub claim_boundary: String,
}

/// Builds the repo-facing publication for the bounded module-execution lane.
#[must_use]
pub fn tassadar_module_execution_capability_publication(
    fixture: &TassadarExecutorFixture,
) -> TassadarModuleExecutionCapabilityPublication {
    TassadarModuleExecutionCapabilityPublication {
        publication_id: format!(
            "tassadar.module_execution_capability.{}.v1",
            fixture.descriptor().model.model_id
        ),
        model_id: fixture.descriptor().model.model_id.clone(),
        claim_class: String::from("capability_truth"),
        runtime_capability: tassadar_module_execution_capability_report(),
        seeded_case_ids: vec![
            String::from("global_state_parity"),
            String::from("call_indirect_dispatch"),
            String::from("deterministic_import_stub"),
            String::from("unsupported_host_import_refusal"),
        ],
        claim_boundary: String::from(
            "this publication covers bounded module execution with i32 globals, funcref tables, zero-parameter indirect calls, and deterministic host-import stubs only; arbitrary host calls and arbitrary Wasm remain explicitly unsupported",
        ),
    }
}

impl TassadarExecutorFixture {
    /// Returns the repo-facing publication for the bounded module-execution lane.
    #[must_use]
    pub fn module_execution_capability_publication(
        &self,
    ) -> TassadarModuleExecutionCapabilityPublication {
        tassadar_module_execution_capability_publication(self)
    }
}

#[cfg(test)]
mod tests {
    use super::tassadar_module_execution_capability_publication;
    use crate::TassadarExecutorFixture;

    #[test]
    fn module_execution_capability_publication_is_machine_legible() {
        let fixture = TassadarExecutorFixture::article_i32_compute_v1();
        let publication = tassadar_module_execution_capability_publication(&fixture);
        assert_eq!(
            publication.model_id,
            TassadarExecutorFixture::ARTICLE_I32_COMPUTE_MODEL_ID
        );
        assert!(publication.runtime_capability.supports_call_indirect);
        assert_eq!(publication.seeded_case_ids.len(), 4);
        assert_eq!(publication.claim_class, "capability_truth");
    }
}
