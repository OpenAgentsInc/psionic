//! Bounded CPU-budget guard for the student training/eval entrypoints
//! (psionic#1123).
//!
//! Training runs default to **one core** unless the owner explicitly
//! opts into more, either with the `--cpu-budget <cores>` flag or the
//! `PSIONIC_TRAIN_CPU_BUDGET` environment variable. Agents launching
//! training must treat the default as binding; the launch banner
//! records the effective budget and how it was set so transcripts show
//! whether a run was authorized to go wide.

/// Environment variable carrying an explicit owner CPU-budget opt-in.
pub const CPU_BUDGET_ENV: &str = "PSIONIC_TRAIN_CPU_BUDGET";

/// Bounded default per psionic#1123: one core.
pub const DEFAULT_CPU_BUDGET_CORES: usize = 1;

/// How the effective CPU budget was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetSource {
    /// Bounded default; nobody opted into more.
    Default,
    /// Explicit `PSIONIC_TRAIN_CPU_BUDGET` environment override.
    EnvVar,
    /// Explicit `--cpu-budget` (or legacy `--threads`) flag.
    Flag,
}

impl BudgetSource {
    /// Stable label for banners and receipts.
    pub fn label(self) -> &'static str {
        match self {
            BudgetSource::Default => "default",
            BudgetSource::EnvVar => "env PSIONIC_TRAIN_CPU_BUDGET (explicit owner opt-in)",
            BudgetSource::Flag => "flag --cpu-budget (explicit owner opt-in)",
        }
    }
}

/// Effective CPU budget for one entrypoint launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuBudget {
    /// Worker-thread cap applied to the global rayon pool.
    pub cores: usize,
    /// Where the cap came from.
    pub source: BudgetSource,
}

impl CpuBudget {
    /// Launch banner line; printed to stderr before any work starts.
    pub fn banner(&self) -> String {
        format!(
            "cpu-budget cores={} source={} (bounded default {} core per psionic#1123; \
             widen only via --cpu-budget N or {}=N)",
            self.cores,
            self.source.label(),
            DEFAULT_CPU_BUDGET_CORES,
            CPU_BUDGET_ENV
        )
    }
}

/// Resolves the effective budget. Precedence: explicit flag, then the
/// environment variable, then the bounded one-core default.
pub fn resolve_cpu_budget(
    flag_cores: Option<usize>,
    env_value: Option<&str>,
) -> Result<CpuBudget, String> {
    if let Some(cores) = flag_cores {
        if cores == 0 {
            return Err(String::from("--cpu-budget must be >= 1 core"));
        }
        return Ok(CpuBudget {
            cores,
            source: BudgetSource::Flag,
        });
    }
    if let Some(raw) = env_value {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let cores: usize = trimmed
                .parse()
                .map_err(|error| format!("bad {CPU_BUDGET_ENV}={trimmed:?}: {error}"))?;
            if cores == 0 {
                return Err(format!("{CPU_BUDGET_ENV} must be >= 1 core"));
            }
            return Ok(CpuBudget {
                cores,
                source: BudgetSource::EnvVar,
            });
        }
    }
    Ok(CpuBudget {
        cores: DEFAULT_CPU_BUDGET_CORES,
        source: BudgetSource::Default,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_one_core() {
        let budget = resolve_cpu_budget(None, None).unwrap();
        assert_eq!(budget.cores, 1);
        assert_eq!(budget.source, BudgetSource::Default);
    }

    #[test]
    fn empty_env_falls_back_to_default() {
        let budget = resolve_cpu_budget(None, Some("  ")).unwrap();
        assert_eq!(budget.cores, 1);
        assert_eq!(budget.source, BudgetSource::Default);
    }

    #[test]
    fn env_opt_in_is_honored() {
        let budget = resolve_cpu_budget(None, Some("5")).unwrap();
        assert_eq!(budget.cores, 5);
        assert_eq!(budget.source, BudgetSource::EnvVar);
    }

    #[test]
    fn flag_beats_env() {
        let budget = resolve_cpu_budget(Some(3), Some("8")).unwrap();
        assert_eq!(budget.cores, 3);
        assert_eq!(budget.source, BudgetSource::Flag);
    }

    #[test]
    fn zero_flag_is_rejected() {
        assert!(resolve_cpu_budget(Some(0), None).is_err());
    }

    #[test]
    fn zero_env_is_rejected() {
        assert!(resolve_cpu_budget(None, Some("0")).is_err());
    }

    #[test]
    fn garbage_env_is_rejected() {
        let error = resolve_cpu_budget(None, Some("lots")).unwrap_err();
        assert!(error.contains(CPU_BUDGET_ENV), "{error}");
    }

    #[test]
    fn banner_names_budget_source_and_overrides() {
        let default = resolve_cpu_budget(None, None).unwrap().banner();
        assert!(default.contains("cores=1"), "{default}");
        assert!(default.contains("source=default"), "{default}");
        assert!(default.contains("psionic#1123"), "{default}");
        assert!(default.contains(CPU_BUDGET_ENV), "{default}");

        let flagged = resolve_cpu_budget(Some(5), None).unwrap().banner();
        assert!(flagged.contains("cores=5"), "{flagged}");
        assert!(flagged.contains("explicit owner opt-in"), "{flagged}");
    }
}
