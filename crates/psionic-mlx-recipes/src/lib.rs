//! Bounded MLX-style training recipe layer above `psionic-train`.

use psionic_environments::EnvironmentPackageKey;
use psionic_train::{
    OPEN_ADAPTER_REFERENCE_ADAPTER_FAMILY, OPEN_ADAPTER_REFERENCE_ADAPTER_FORMAT,
    OpenAdapterAdmissibleModelFamily, PolicyRevision, RolloutValidatorPolicy, TrainingCoreError,
    TrainingLoopBudget, TrainingOptimizerConfig, TrainingRunGraphError, TrainingRunState,
    TrainingStageKind,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Human-readable crate ownership summary.
pub const CRATE_ROLE: &str = "bounded MLX-style training recipe layer above psionic-train";

/// Supported MLX-style recipe methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxRecipeMethod {
    /// Plain supervised fine-tuning.
    Sft,
    /// LoRA adapter fine-tuning.
    Lora,
    /// DoRA adapter fine-tuning.
    Dora,
    /// Quantized LoRA fine-tuning.
    Qlora,
    /// Direct preference optimization.
    Dpo,
    /// Conservative preference optimization.
    Cpo,
    /// Odds-ratio preference optimization.
    Orpo,
    /// Group relative policy optimization.
    Grpo,
    /// Online DPO.
    OnlineDpo,
    /// Experience preference optimization.
    Xpo,
    /// Proximal policy optimization.
    Ppo,
}

impl MlxRecipeMethod {
    /// Returns the stable method label used by CLI and receipts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sft => "sft",
            Self::Lora => "lora",
            Self::Dora => "dora",
            Self::Qlora => "qlora",
            Self::Dpo => "dpo",
            Self::Cpo => "cpo",
            Self::Orpo => "orpo",
            Self::Grpo => "grpo",
            Self::OnlineDpo => "online_dpo",
            Self::Xpo => "xpo",
            Self::Ppo => "ppo",
        }
    }

    /// Returns whether the method requires an explicit adapter config.
    #[must_use]
    pub const fn requires_adapter(self) -> bool {
        matches!(self, Self::Lora | Self::Dora | Self::Qlora)
    }

    /// Returns whether the method materializes rollout-validator posture.
    #[must_use]
    pub const fn uses_rollout_validator(self) -> bool {
        matches!(self, Self::Grpo | Self::OnlineDpo | Self::Xpo | Self::Ppo)
    }
}

/// One machine-readable method summary surfaced by the package.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxRecipeMethodSummary {
    /// Stable method id.
    pub method: MlxRecipeMethod,
    /// Ordered stage sequence for the method.
    pub stage_sequence: Vec<TrainingStageKind>,
    /// Whether the caller must supply an adapter config.
    pub requires_adapter: bool,
    /// Whether the plan will emit rollout-validator posture.
    pub uses_rollout_validator: bool,
    /// Honest bounded notes for the method family.
    pub notes: Vec<String>,
}

/// Bounded adapter strategy surfaced by the recipe layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxAdapterRecipe {
    /// Adapter method family.
    pub method: MlxRecipeMethod,
    /// Target adapter rank.
    pub rank: usize,
    /// Target adapter alpha.
    pub alpha: f32,
    /// Optional quantization label for QLoRA-class plans.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
}

/// Adapter execution plan reused from the open adapter lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxAdapterExecutionPlan {
    /// Admissible open adapter family.
    pub admissible_model_family: OpenAdapterAdmissibleModelFamily,
    /// Adapter family label.
    pub adapter_family: String,
    /// Adapter artifact format.
    pub adapter_format: String,
    /// Stable target identifier.
    pub target_id: String,
    /// Requested adapter rank.
    pub rank: usize,
    /// Requested adapter alpha.
    pub alpha: String,
    /// Optional quantization label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
}

/// One reusable recipe config.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxRecipeConfig {
    /// Stable run id.
    pub run_id: String,
    /// Stable cluster id for the run graph.
    pub cluster_id: String,
    /// Stable checkpoint family.
    pub checkpoint_family: String,
    /// Environment package key.
    pub environment: EnvironmentPackageKey,
    /// Selected recipe method.
    pub method: MlxRecipeMethod,
    /// Fixed-budget loop policy.
    pub budget: TrainingLoopBudget,
    /// Base policy family.
    pub policy_family: String,
    /// Optional adapter plan.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter: Option<MlxAdapterRecipe>,
}

impl MlxRecipeConfig {
    /// Creates one recipe config with a validated default budget.
    pub fn new(
        run_id: impl Into<String>,
        cluster_id: impl Into<String>,
        checkpoint_family: impl Into<String>,
        environment: EnvironmentPackageKey,
        method: MlxRecipeMethod,
    ) -> Result<Self, MlxRecipeError> {
        Ok(Self {
            run_id: run_id.into(),
            cluster_id: cluster_id.into(),
            checkpoint_family: checkpoint_family.into(),
            environment,
            method,
            budget: TrainingLoopBudget::new(64, 8, 4)?,
            policy_family: String::from("mlx.recipe.policy"),
            adapter: None,
        })
    }

    /// Attaches one adapter recipe.
    #[must_use]
    pub fn with_adapter(mut self, adapter: MlxAdapterRecipe) -> Self {
        self.adapter = Some(adapter);
        self
    }
}

/// Planned recipe execution over the shared train substrate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxRecipePlan {
    /// Stable config digest.
    pub config_digest: String,
    /// Selected method.
    pub method: MlxRecipeMethod,
    /// Current run graph root.
    pub run_graph: TrainingRunState,
    /// Ordered stage sequence.
    pub stage_sequence: Vec<TrainingStageKind>,
    /// Optimizer chosen for the recipe.
    pub optimizer: TrainingOptimizerConfig,
    /// Input policy revision for rollouts and promotion.
    pub policy_revision: PolicyRevision,
    /// Optional adapter execution projection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_execution: Option<MlxAdapterExecutionPlan>,
    /// Optional rollout-validator policy for RL-style methods.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rollout_validator_policy: Option<RolloutValidatorPolicy>,
    /// Honest bounded notes.
    pub notes: Vec<String>,
}

/// Errors returned by the recipe layer.
#[derive(Debug, Error)]
pub enum MlxRecipeError {
    /// The selected method does not allow one attached adapter strategy.
    #[error("recipe method `{method:?}` does not accept an adapter configuration")]
    AdapterNotAllowed {
        /// Selected method.
        method: MlxRecipeMethod,
    },
    /// One adapter strategy is missing for an adapter method.
    #[error("recipe method `{method:?}` requires an adapter configuration")]
    AdapterRequired {
        /// Selected method.
        method: MlxRecipeMethod,
    },
    /// One adapter config is invalid.
    #[error("{0}")]
    InvalidAdapter(String),
    /// Fixed-budget validation failed.
    #[error(transparent)]
    TrainingCore(#[from] TrainingCoreError),
    /// Run-graph initialization failed.
    #[error(transparent)]
    RunGraph(#[from] TrainingRunGraphError),
}

/// Workspace for the recipe planner.
#[derive(Clone, Debug, Default)]
pub struct MlxRecipeWorkspace;

impl MlxRecipeWorkspace {
    /// Returns the machine-readable supported method inventory.
    #[must_use]
    pub fn methods(&self) -> Vec<MlxRecipeMethodSummary> {
        supported_methods()
    }

    /// Plans one recipe over the existing train substrate.
    pub fn plan(&self, config: &MlxRecipeConfig) -> Result<MlxRecipePlan, MlxRecipeError> {
        validate_adapter_choice(config.method, config.adapter.as_ref())?;
        let stage_sequence = stage_sequence_for_method(config.method);
        let current_stage = *stage_sequence
            .last()
            .expect("stage sequence must be non-empty");
        let stage_id = format!("stage-{}", stage_label(current_stage));
        let run_graph = TrainingRunState::new(
            config.run_id.clone(),
            stage_id,
            config.cluster_id.clone(),
            config.checkpoint_family.clone(),
            config.environment.clone(),
        )?;
        let policy_revision = PolicyRevision::new(
            config.policy_family.clone(),
            format!("{}-r1", config.run_id),
            stable_recipe_policy_digest(config),
            1_000,
        )
        .with_revision_number(1);
        let adapter_execution = config
            .adapter
            .as_ref()
            .map(build_adapter_plan)
            .transpose()?;
        let rollout_validator_policy = rl_method(config.method).then(|| RolloutValidatorPolicy {
            policy_id: format!("{}.validator", config.run_id),
            ..RolloutValidatorPolicy::default()
        });
        Ok(MlxRecipePlan {
            config_digest: stable_recipe_config_digest(config),
            method: config.method,
            run_graph,
            stage_sequence,
            optimizer: optimizer_for_method(config.method),
            policy_revision,
            adapter_execution,
            rollout_validator_policy,
            notes: notes_for_method(config.method),
        })
    }
}

fn validate_adapter_choice(
    method: MlxRecipeMethod,
    adapter: Option<&MlxAdapterRecipe>,
) -> Result<(), MlxRecipeError> {
    match (method.requires_adapter(), adapter) {
        (true, None) => Err(MlxRecipeError::AdapterRequired { method }),
        (false, Some(_)) => Err(MlxRecipeError::AdapterNotAllowed { method }),
        (true, Some(adapter)) => {
            if adapter.method != method {
                return Err(MlxRecipeError::InvalidAdapter(format!(
                    "adapter method `{}` does not match recipe method `{}`",
                    adapter.method.as_str(),
                    method.as_str()
                )));
            }
            if adapter.rank == 0 {
                return Err(MlxRecipeError::InvalidAdapter(String::from(
                    "adapter rank must be greater than zero",
                )));
            }
            if !adapter.alpha.is_finite() || adapter.alpha <= 0.0 {
                return Err(MlxRecipeError::InvalidAdapter(String::from(
                    "adapter alpha must be finite and greater than zero",
                )));
            }
            match method {
                MlxRecipeMethod::Qlora if adapter.quantization.is_none() => {
                    return Err(MlxRecipeError::InvalidAdapter(String::from(
                        "qlora recipes require an explicit quantization label",
                    )));
                }
                MlxRecipeMethod::Lora | MlxRecipeMethod::Dora if adapter.quantization.is_some() => {
                    return Err(MlxRecipeError::InvalidAdapter(String::from(
                        "only qlora recipes may attach an explicit quantization label",
                    )));
                }
                _ => {}
            }
            Ok(())
        }
        (false, None) => Ok(()),
    }
}

fn stage_sequence_for_method(method: MlxRecipeMethod) -> Vec<TrainingStageKind> {
    match method {
        MlxRecipeMethod::Sft
        | MlxRecipeMethod::Lora
        | MlxRecipeMethod::Dora
        | MlxRecipeMethod::Qlora => vec![TrainingStageKind::GeneralSft],
        MlxRecipeMethod::Dpo | MlxRecipeMethod::Cpo | MlxRecipeMethod::Orpo => {
            vec![TrainingStageKind::GeneralSft, TrainingStageKind::AgenticSft]
        }
        MlxRecipeMethod::Grpo
        | MlxRecipeMethod::OnlineDpo
        | MlxRecipeMethod::Xpo
        | MlxRecipeMethod::Ppo => vec![
            TrainingStageKind::GeneralSft,
            TrainingStageKind::AgenticSft,
            TrainingStageKind::Rl,
        ],
    }
}

fn optimizer_for_method(method: MlxRecipeMethod) -> TrainingOptimizerConfig {
    match method {
        MlxRecipeMethod::Sft
        | MlxRecipeMethod::Lora
        | MlxRecipeMethod::Dora
        | MlxRecipeMethod::Qlora => TrainingOptimizerConfig::adamw(2e-4, 0.9, 0.95, 1e-8),
        MlxRecipeMethod::Dpo
        | MlxRecipeMethod::Cpo
        | MlxRecipeMethod::Orpo
        | MlxRecipeMethod::OnlineDpo
        | MlxRecipeMethod::Xpo => TrainingOptimizerConfig::adam(1e-5, 0.9, 0.99, 1e-8),
        MlxRecipeMethod::Grpo | MlxRecipeMethod::Ppo => {
            TrainingOptimizerConfig::adam(3e-6, 0.9, 0.999, 1e-8)
        }
    }
}

fn build_adapter_plan(
    adapter: &MlxAdapterRecipe,
) -> Result<MlxAdapterExecutionPlan, MlxRecipeError> {
    Ok(MlxAdapterExecutionPlan {
        admissible_model_family: OpenAdapterAdmissibleModelFamily::GptOssDecoderLmHeadLora,
        adapter_family: String::from(OPEN_ADAPTER_REFERENCE_ADAPTER_FAMILY),
        adapter_format: String::from(OPEN_ADAPTER_REFERENCE_ADAPTER_FORMAT),
        target_id: String::from("mlx.adapter.target"),
        rank: adapter.rank,
        alpha: format!("{:.4}", adapter.alpha),
        quantization: adapter.quantization.clone(),
    })
}

fn rl_method(method: MlxRecipeMethod) -> bool {
    method.uses_rollout_validator()
}

fn stage_label(kind: TrainingStageKind) -> &'static str {
    match kind {
        TrainingStageKind::Pretrain => "pretrain",
        TrainingStageKind::GeneralSft => "general_sft",
        TrainingStageKind::AgenticSft => "agentic_sft",
        TrainingStageKind::Rl => "rl",
    }
}

fn stable_recipe_config_digest(config: &MlxRecipeConfig) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_mlx_recipe_config|");
    hasher.update(config.run_id.as_bytes());
    hasher.update(b"|");
    hasher.update(config.cluster_id.as_bytes());
    hasher.update(b"|");
    hasher.update(config.checkpoint_family.as_bytes());
    hasher.update(b"|");
    hasher.update(config.environment.storage_key().as_bytes());
    hasher.update(b"|");
    hasher.update(format!("{:?}", config.method).as_bytes());
    hasher.update(b"|");
    hasher.update(config.policy_family.as_bytes());
    if let Some(adapter) = &config.adapter {
        hasher.update(b"|adapter|");
        hasher.update(format!("{:?}", adapter.method).as_bytes());
        hasher.update(adapter.rank.to_string().as_bytes());
        hasher.update(format!("{:.6}", adapter.alpha).as_bytes());
        if let Some(quantization) = &adapter.quantization {
            hasher.update(quantization.as_bytes());
        }
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn stable_recipe_policy_digest(config: &MlxRecipeConfig) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_mlx_recipe_policy|");
    hasher.update(stable_recipe_config_digest(config).as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn notes_for_method(method: MlxRecipeMethod) -> Vec<String> {
    let mut notes = vec![String::from(
        "This package is a recipe planner above psionic-train: it compiles MLX-style method choices into the existing run-graph, optimizer, policy, and adapter/rollout primitives instead of adding a second trainer architecture.",
    )];
    if rl_method(method) {
        notes.push(String::from(
            "RL-style methods currently materialize a rollout-validator policy and an explicit `general_sft -> agentic_sft -> rl` stage sequence over the shared train substrate.",
        ));
    }
    if matches!(
        method,
        MlxRecipeMethod::Lora | MlxRecipeMethod::Dora | MlxRecipeMethod::Qlora
    ) {
        notes.push(String::from(
            "Adapter methods currently target the repo-owned open adapter lane and keep adapter family/format explicit instead of hiding them behind notebook-side defaults.",
        ));
    }
    notes
}

fn supported_methods() -> Vec<MlxRecipeMethodSummary> {
    [
        MlxRecipeMethod::Sft,
        MlxRecipeMethod::Lora,
        MlxRecipeMethod::Dora,
        MlxRecipeMethod::Qlora,
        MlxRecipeMethod::Dpo,
        MlxRecipeMethod::Cpo,
        MlxRecipeMethod::Orpo,
        MlxRecipeMethod::Grpo,
        MlxRecipeMethod::OnlineDpo,
        MlxRecipeMethod::Xpo,
        MlxRecipeMethod::Ppo,
    ]
    .into_iter()
    .map(|method| MlxRecipeMethodSummary {
        method,
        stage_sequence: stage_sequence_for_method(method),
        requires_adapter: method.requires_adapter(),
        uses_rollout_validator: method.uses_rollout_validator(),
        notes: notes_for_method(method),
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::{MlxAdapterRecipe, MlxRecipeConfig, MlxRecipeMethod, MlxRecipeWorkspace};
    use psionic_environments::EnvironmentPackageKey;
    use psionic_train::TrainingStageKind;

    #[test]
    fn sft_recipe_stays_in_general_sft() -> Result<(), Box<dyn std::error::Error>> {
        let plan = MlxRecipeWorkspace.plan(&MlxRecipeConfig::new(
            "recipe-sft",
            "cluster-a",
            "checkpoint.recipe",
            EnvironmentPackageKey::new("env.sft", "v1"),
            MlxRecipeMethod::Sft,
        )?)?;

        assert_eq!(plan.stage_sequence, vec![TrainingStageKind::GeneralSft]);
        assert!(plan.adapter_execution.is_none());
        assert!(plan.rollout_validator_policy.is_none());
        Ok(())
    }

    #[test]
    fn qlora_recipe_reuses_open_adapter_lane() -> Result<(), Box<dyn std::error::Error>> {
        let config = MlxRecipeConfig::new(
            "recipe-qlora",
            "cluster-a",
            "checkpoint.recipe",
            EnvironmentPackageKey::new("env.sft", "v1"),
            MlxRecipeMethod::Qlora,
        )?
        .with_adapter(MlxAdapterRecipe {
            method: MlxRecipeMethod::Qlora,
            rank: 16,
            alpha: 32.0,
            quantization: Some(String::from("q4_k")),
        });
        let plan = MlxRecipeWorkspace.plan(&config)?;

        assert_eq!(plan.stage_sequence, vec![TrainingStageKind::GeneralSft]);
        assert_eq!(
            plan.adapter_execution
                .as_ref()
                .expect("adapter plan")
                .quantization
                .as_deref(),
            Some("q4_k")
        );
        Ok(())
    }

    #[test]
    fn ppo_recipe_emits_rl_stage_and_rollout_policy() -> Result<(), Box<dyn std::error::Error>> {
        let plan = MlxRecipeWorkspace.plan(&MlxRecipeConfig::new(
            "recipe-ppo",
            "cluster-b",
            "checkpoint.recipe",
            EnvironmentPackageKey::new("env.rl", "v1"),
            MlxRecipeMethod::Ppo,
        )?)?;

        assert_eq!(
            plan.stage_sequence,
            vec![
                TrainingStageKind::GeneralSft,
                TrainingStageKind::AgenticSft,
                TrainingStageKind::Rl
            ]
        );
        assert!(plan.rollout_validator_policy.is_some());
        Ok(())
    }

    #[test]
    fn qlora_recipe_requires_quantization_label() {
        let config = MlxRecipeConfig::new(
            "recipe-qlora-invalid",
            "cluster-a",
            "checkpoint.recipe",
            EnvironmentPackageKey::new("env.sft", "v1"),
            MlxRecipeMethod::Qlora,
        )
        .expect("config")
        .with_adapter(MlxAdapterRecipe {
            method: MlxRecipeMethod::Qlora,
            rank: 16,
            alpha: 32.0,
            quantization: None,
        });

        let error = MlxRecipeWorkspace
            .plan(&config)
            .expect_err("expected refusal");
        assert!(
            error
                .to_string()
                .contains("qlora recipes require an explicit quantization label")
        );
    }

    #[test]
    fn method_inventory_marks_adapter_and_rl_surfaces() {
        let methods = MlxRecipeWorkspace.methods();
        let qlora = methods
            .iter()
            .find(|method| method.method == MlxRecipeMethod::Qlora)
            .expect("qlora summary");
        assert!(qlora.requires_adapter);
        assert!(!qlora.uses_rollout_validator);

        let ppo = methods
            .iter()
            .find(|method| method.method == MlxRecipeMethod::Ppo)
            .expect("ppo summary");
        assert!(!ppo.requires_adapter);
        assert!(ppo.uses_rollout_validator);
        assert_eq!(
            ppo.stage_sequence,
            vec![
                TrainingStageKind::GeneralSft,
                TrainingStageKind::AgenticSft,
                TrainingStageKind::Rl
            ]
        );
    }
}
