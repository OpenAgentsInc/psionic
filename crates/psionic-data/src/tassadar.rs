use std::collections::{BTreeMap, BTreeSet};

use psionic_datastream::{DatastreamEncoding, DatastreamManifest, DatastreamSubjectKind};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    DatasetContractError, DatasetKey, DatasetManifest, DatasetPackingPlan, DatasetPackingPolicy,
    DatasetRecordEncoding, DatasetSequenceDescriptor, DatasetShardManifest,
    DatasetSplitDeclaration, DatasetSplitKind, TokenizerDigest,
};

/// Stable ABI version for Tassadar token-sequence dataset contracts.
pub const TASSADAR_SEQUENCE_DATASET_ABI_VERSION: &str = "psionic.tassadar.sequence_dataset.v1";
/// Stable ABI version for Tassadar benchmark-package-set contracts.
pub const TASSADAR_BENCHMARK_PACKAGE_SET_ABI_VERSION: &str =
    "psionic.tassadar.benchmark_package_set.v1";
/// Stable ABI version for Tassadar numeric-opcode ladder contracts.
pub const TASSADAR_NUMERIC_OPCODE_LADDER_ABI_VERSION: &str =
    "psionic.tassadar.numeric_opcode_ladder.v1";
/// Stable ABI version for Tassadar trace-family-set contracts.
pub const TASSADAR_TRACE_FAMILY_SET_ABI_VERSION: &str = "psionic.tassadar.trace_family_set.v1";

/// Benchmark-family taxonomy for the public Tassadar package set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarBenchmarkFamily {
    /// Exact arithmetic kernels and microprograms.
    Arithmetic,
    /// CLRS-adjacent benchmark subset currently seeded by shortest-path witnesses.
    ClrsSubset,
    /// Sudoku-class exact-search workloads.
    Sudoku,
    /// Hungarian or matching-class workloads.
    Hungarian,
    /// Explicit trace-length and horizon-stress workloads.
    TraceLengthStress,
}

/// Canonical benchmark-summary axes for Tassadar benchmark packages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarBenchmarkAxis {
    /// Exactness or equivalence facts.
    Exactness,
    /// Length-generalization or horizon-scaling posture.
    LengthGeneralization,
    /// Whether the family is useful for planner or route selection rather than only systems work.
    PlannerUsefulness,
}

/// Numeric-opcode family tracked by the public Tassadar widening ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarNumericOpcodeFamily {
    /// Core i32 arithmetic already admitted by the bounded lane.
    I32CoreArithmetic,
    /// Full i32 comparison and equality family, including `eqz`.
    I32Comparisons,
    /// i32 bitwise and shift family.
    I32BitOps,
    /// i64 integer family that remains out of scope today.
    I64Integer,
    /// Floating-point family that remains out of scope today.
    FloatingPoint,
}

/// Current support posture for one numeric-opcode family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarNumericOpcodeFamilyStatus {
    /// The family has exact bounded coverage today.
    Implemented,
    /// The family still refuses explicitly today.
    Refused,
}

/// Family-level contract row for the public numeric-opcode widening ladder.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarNumericOpcodeFamilyContract {
    /// Stable numeric family.
    pub family: TassadarNumericOpcodeFamily,
    /// Current support posture for the family.
    pub status: TassadarNumericOpcodeFamilyStatus,
    /// Human-readable family summary.
    pub summary: String,
    /// Explicit claim boundary for the family.
    pub claim_boundary: String,
    /// Opcodes implemented exactly for the family today.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported_opcodes: Vec<String>,
    /// Typed refusal classes still expected for the family today.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refusal_kinds: Vec<String>,
    /// Stable seeded case identifiers that anchor the family today.
    pub case_ids: Vec<String>,
}

impl TassadarNumericOpcodeFamilyContract {
    fn validate(&self) -> Result<(), TassadarNumericOpcodeLadderError> {
        if self.summary.trim().is_empty() {
            return Err(TassadarNumericOpcodeLadderError::MissingFamilySummary {
                family: self.family,
            });
        }
        if self.claim_boundary.trim().is_empty() {
            return Err(TassadarNumericOpcodeLadderError::MissingClaimBoundary {
                family: self.family,
            });
        }
        if self.case_ids.is_empty() {
            return Err(TassadarNumericOpcodeLadderError::MissingCaseIds {
                family: self.family,
            });
        }

        match self.status {
            TassadarNumericOpcodeFamilyStatus::Implemented if self.supported_opcodes.is_empty() => {
                return Err(TassadarNumericOpcodeLadderError::MissingSupportedOpcodes {
                    family: self.family,
                });
            }
            TassadarNumericOpcodeFamilyStatus::Refused if self.refusal_kinds.is_empty() => {
                return Err(TassadarNumericOpcodeLadderError::MissingRefusalKinds {
                    family: self.family,
                });
            }
            _ => {}
        }

        let mut seen_supported_opcodes = BTreeSet::new();
        for opcode in &self.supported_opcodes {
            if opcode.trim().is_empty() {
                return Err(TassadarNumericOpcodeLadderError::MissingSupportedOpcode {
                    family: self.family,
                });
            }
            if !seen_supported_opcodes.insert(opcode.clone()) {
                return Err(TassadarNumericOpcodeLadderError::DuplicateSupportedOpcode {
                    family: self.family,
                    opcode: opcode.clone(),
                });
            }
        }

        let mut seen_refusal_kinds = BTreeSet::new();
        for refusal_kind in &self.refusal_kinds {
            if refusal_kind.trim().is_empty() {
                return Err(TassadarNumericOpcodeLadderError::MissingRefusalKind {
                    family: self.family,
                });
            }
            if !seen_refusal_kinds.insert(refusal_kind.clone()) {
                return Err(TassadarNumericOpcodeLadderError::DuplicateRefusalKind {
                    family: self.family,
                    refusal_kind: refusal_kind.clone(),
                });
            }
        }

        let mut seen_case_ids = BTreeSet::new();
        for case_id in &self.case_ids {
            if case_id.trim().is_empty() {
                return Err(TassadarNumericOpcodeLadderError::MissingCaseId {
                    family: self.family,
                });
            }
            if !seen_case_ids.insert(case_id.clone()) {
                return Err(TassadarNumericOpcodeLadderError::DuplicateCaseId {
                    family: self.family,
                    case_id: case_id.clone(),
                });
            }
        }

        Ok(())
    }
}

/// Public contract for the current numeric-opcode widening ladder.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarNumericOpcodeLadderContract {
    /// Stable ABI version.
    pub abi_version: String,
    /// Stable ladder reference.
    pub ladder_ref: String,
    /// Immutable ladder version.
    pub version: String,
    /// Family rows covered by the ladder.
    pub families: Vec<TassadarNumericOpcodeFamilyContract>,
}

impl TassadarNumericOpcodeLadderContract {
    /// Creates and validates a numeric-opcode ladder contract.
    pub fn new(
        ladder_ref: impl Into<String>,
        version: impl Into<String>,
        families: Vec<TassadarNumericOpcodeFamilyContract>,
    ) -> Result<Self, TassadarNumericOpcodeLadderError> {
        let contract = Self {
            abi_version: String::from(TASSADAR_NUMERIC_OPCODE_LADDER_ABI_VERSION),
            ladder_ref: ladder_ref.into(),
            version: version.into(),
            families,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Validates the contract.
    pub fn validate(&self) -> Result<(), TassadarNumericOpcodeLadderError> {
        if self.abi_version != TASSADAR_NUMERIC_OPCODE_LADDER_ABI_VERSION {
            return Err(TassadarNumericOpcodeLadderError::UnsupportedAbiVersion {
                abi_version: self.abi_version.clone(),
            });
        }
        if self.ladder_ref.trim().is_empty() {
            return Err(TassadarNumericOpcodeLadderError::MissingLadderRef);
        }
        if self.version.trim().is_empty() {
            return Err(TassadarNumericOpcodeLadderError::MissingLadderVersion);
        }
        if self.families.is_empty() {
            return Err(TassadarNumericOpcodeLadderError::MissingFamilies);
        }

        let mut seen_families = BTreeSet::new();
        for family in &self.families {
            family.validate()?;
            if !seen_families.insert(family.family) {
                return Err(TassadarNumericOpcodeLadderError::DuplicateFamily {
                    family: family.family,
                });
            }
        }

        Ok(())
    }

    /// Returns a stable digest over the contract.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(b"psionic_tassadar_numeric_opcode_ladder_contract|", self)
    }
}

/// Returns the current public numeric-opcode widening ladder contract.
#[must_use]
pub fn tassadar_numeric_opcode_ladder_contract() -> TassadarNumericOpcodeLadderContract {
    TassadarNumericOpcodeLadderContract::new(
        "tassadar://numeric_opcode_ladder/structured_control_v1",
        "2026.03.17",
        vec![
            TassadarNumericOpcodeFamilyContract {
                family: TassadarNumericOpcodeFamily::I32CoreArithmetic,
                status: TassadarNumericOpcodeFamilyStatus::Implemented,
                summary: String::from(
                    "core i32 arithmetic remains exact in the bounded structured-control lane",
                ),
                claim_boundary: String::from(
                    "compiled_bounded_exactness over zero-parameter i32-only Wasm functions with empty block types",
                ),
                supported_opcodes: vec![
                    String::from("i32.add"),
                    String::from("i32.sub"),
                    String::from("i32.mul"),
                ],
                refusal_kinds: Vec::new(),
                case_ids: vec![String::from("i32_core_arithmetic_suite")],
            },
            TassadarNumericOpcodeFamilyContract {
                family: TassadarNumericOpcodeFamily::I32Comparisons,
                status: TassadarNumericOpcodeFamilyStatus::Implemented,
                summary: String::from(
                    "full i32 equality and comparison coverage now exists, including `eqz` and signed-vs-unsigned forms",
                ),
                claim_boundary: String::from(
                    "compiled_bounded_exactness for the current zero-parameter i32-only structured-control lowering lane",
                ),
                supported_opcodes: vec![
                    String::from("i32.eqz"),
                    String::from("i32.eq"),
                    String::from("i32.ne"),
                    String::from("i32.lt_s"),
                    String::from("i32.lt_u"),
                    String::from("i32.gt_s"),
                    String::from("i32.gt_u"),
                    String::from("i32.le_s"),
                    String::from("i32.le_u"),
                    String::from("i32.ge_s"),
                    String::from("i32.ge_u"),
                ],
                refusal_kinds: Vec::new(),
                case_ids: vec![String::from("i32_comparison_suite")],
            },
            TassadarNumericOpcodeFamilyContract {
                family: TassadarNumericOpcodeFamily::I32BitOps,
                status: TassadarNumericOpcodeFamilyStatus::Implemented,
                summary: String::from(
                    "i32 bitwise logic and shift operations now lower exactly in the bounded lane",
                ),
                claim_boundary: String::from(
                    "compiled_bounded_exactness for the current zero-parameter i32-only structured-control lowering lane",
                ),
                supported_opcodes: vec![
                    String::from("i32.and"),
                    String::from("i32.or"),
                    String::from("i32.xor"),
                    String::from("i32.shl"),
                    String::from("i32.shr_s"),
                    String::from("i32.shr_u"),
                ],
                refusal_kinds: Vec::new(),
                case_ids: vec![String::from("i32_bit_ops_suite")],
            },
            TassadarNumericOpcodeFamilyContract {
                family: TassadarNumericOpcodeFamily::I64Integer,
                status: TassadarNumericOpcodeFamilyStatus::Refused,
                summary: String::from(
                    "i64 result and instruction families remain outside the current i32-only lowering lane",
                ),
                claim_boundary: String::from(
                    "typed refusal only; this issue does not claim i64 closure or mixed-width execution",
                ),
                supported_opcodes: Vec::new(),
                refusal_kinds: vec![String::from("unsupported_instruction")],
                case_ids: vec![String::from("i64_refusal")],
            },
            TassadarNumericOpcodeFamilyContract {
                family: TassadarNumericOpcodeFamily::FloatingPoint,
                status: TassadarNumericOpcodeFamilyStatus::Refused,
                summary: String::from(
                    "floating-point lowering remains explicitly gated outside the current structured-control lane",
                ),
                claim_boundary: String::from(
                    "typed refusal only; no floating-point exactness or conformance claim is made here",
                ),
                supported_opcodes: Vec::new(),
                refusal_kinds: vec![String::from("unsupported_instruction")],
                case_ids: vec![String::from("float_refusal")],
            },
        ],
    )
    .expect("current Tassadar numeric-opcode ladder contract should stay valid")
}

/// Numeric-opcode-ladder validation failure.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TassadarNumericOpcodeLadderError {
    /// Unsupported ABI version.
    #[error("unsupported Tassadar numeric-opcode ladder ABI version `{abi_version}`")]
    UnsupportedAbiVersion {
        /// Observed ABI version.
        abi_version: String,
    },
    /// Missing ladder ref.
    #[error("Tassadar numeric-opcode ladder is missing `ladder_ref`")]
    MissingLadderRef,
    /// Missing ladder version.
    #[error("Tassadar numeric-opcode ladder is missing `version`")]
    MissingLadderVersion,
    /// No families declared.
    #[error("Tassadar numeric-opcode ladder must contain at least one family contract")]
    MissingFamilies,
    /// One family repeated.
    #[error("Tassadar numeric-opcode ladder repeated family `{family:?}`")]
    DuplicateFamily {
        /// Repeated family.
        family: TassadarNumericOpcodeFamily,
    },
    /// Family summary missing.
    #[error("Tassadar numeric-opcode family `{family:?}` is missing `summary`")]
    MissingFamilySummary {
        /// Family.
        family: TassadarNumericOpcodeFamily,
    },
    /// Family claim boundary missing.
    #[error("Tassadar numeric-opcode family `{family:?}` is missing `claim_boundary`")]
    MissingClaimBoundary {
        /// Family.
        family: TassadarNumericOpcodeFamily,
    },
    /// Implemented family lacks supported opcode list.
    #[error(
        "implemented Tassadar numeric-opcode family `{family:?}` must declare supported opcodes"
    )]
    MissingSupportedOpcodes {
        /// Family.
        family: TassadarNumericOpcodeFamily,
    },
    /// Refused family lacks refusal list.
    #[error("refused Tassadar numeric-opcode family `{family:?}` must declare refusal kinds")]
    MissingRefusalKinds {
        /// Family.
        family: TassadarNumericOpcodeFamily,
    },
    /// One supported opcode was empty.
    #[error("Tassadar numeric-opcode family `{family:?}` contains an empty supported opcode")]
    MissingSupportedOpcode {
        /// Family.
        family: TassadarNumericOpcodeFamily,
    },
    /// One refusal kind was empty.
    #[error("Tassadar numeric-opcode family `{family:?}` contains an empty refusal kind")]
    MissingRefusalKind {
        /// Family.
        family: TassadarNumericOpcodeFamily,
    },
    /// No case ids declared.
    #[error("Tassadar numeric-opcode family `{family:?}` must declare at least one case id")]
    MissingCaseIds {
        /// Family.
        family: TassadarNumericOpcodeFamily,
    },
    /// One case id was empty.
    #[error("Tassadar numeric-opcode family `{family:?}` contains an empty case id")]
    MissingCaseId {
        /// Family.
        family: TassadarNumericOpcodeFamily,
    },
    /// One supported opcode repeated.
    #[error("Tassadar numeric-opcode family `{family:?}` repeated supported opcode `{opcode}`")]
    DuplicateSupportedOpcode {
        /// Family.
        family: TassadarNumericOpcodeFamily,
        /// Repeated opcode.
        opcode: String,
    },
    /// One refusal kind repeated.
    #[error("Tassadar numeric-opcode family `{family:?}` repeated refusal kind `{refusal_kind}`")]
    DuplicateRefusalKind {
        /// Family.
        family: TassadarNumericOpcodeFamily,
        /// Repeated refusal kind.
        refusal_kind: String,
    },
    /// One case id repeated.
    #[error("Tassadar numeric-opcode family `{family:?}` repeated case id `{case_id}`")]
    DuplicateCaseId {
        /// Family.
        family: TassadarNumericOpcodeFamily,
        /// Repeated case id.
        case_id: String,
    },
}

/// One benchmark package that participates in the public Tassadar package set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarBenchmarkPackageBinding {
    /// Stable benchmark reference.
    pub benchmark_ref: String,
    /// Immutable benchmark package version.
    pub version: String,
    /// Environment package that owns benchmark execution.
    pub environment_ref: String,
    /// Bound dataset identity.
    pub dataset: DatasetKey,
    /// Optional dataset split.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,
}

impl TassadarBenchmarkPackageBinding {
    fn validate(&self) -> Result<(), TassadarBenchmarkPackageSetError> {
        if self.benchmark_ref.trim().is_empty() {
            return Err(TassadarBenchmarkPackageSetError::MissingBenchmarkRef);
        }
        if self.version.trim().is_empty() {
            return Err(TassadarBenchmarkPackageSetError::MissingBenchmarkVersion {
                benchmark_ref: self.benchmark_ref.clone(),
            });
        }
        if self.environment_ref.trim().is_empty() {
            return Err(TassadarBenchmarkPackageSetError::MissingEnvironmentRef {
                benchmark_ref: self.benchmark_ref.clone(),
            });
        }
        if self.dataset.dataset_ref.trim().is_empty() || self.dataset.version.trim().is_empty() {
            return Err(TassadarBenchmarkPackageSetError::MissingDatasetBinding {
                benchmark_ref: self.benchmark_ref.clone(),
            });
        }
        Ok(())
    }
}

/// Family-level benchmark-package-set contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarBenchmarkFamilyContract {
    /// Stable benchmark family.
    pub family: TassadarBenchmarkFamily,
    /// Human-readable family summary.
    pub summary: String,
    /// Benchmark packages that currently cover the family.
    pub benchmark_packages: Vec<TassadarBenchmarkPackageBinding>,
    /// Canonical reporting axes that the family supports.
    pub axis_coverage: Vec<TassadarBenchmarkAxis>,
    /// Stable benchmark case identifiers that seed the family today.
    pub case_ids: Vec<String>,
}

impl TassadarBenchmarkFamilyContract {
    fn validate(&self) -> Result<(), TassadarBenchmarkPackageSetError> {
        if self.summary.trim().is_empty() {
            return Err(TassadarBenchmarkPackageSetError::MissingFamilySummary {
                family: self.family,
            });
        }
        if self.benchmark_packages.is_empty() {
            return Err(TassadarBenchmarkPackageSetError::MissingBenchmarkPackages {
                family: self.family,
            });
        }
        if self.axis_coverage.is_empty() {
            return Err(TassadarBenchmarkPackageSetError::MissingAxisCoverage {
                family: self.family,
            });
        }
        if self.case_ids.is_empty() {
            return Err(TassadarBenchmarkPackageSetError::MissingCaseIds {
                family: self.family,
            });
        }

        let mut seen_axes = BTreeSet::new();
        for axis in &self.axis_coverage {
            if !seen_axes.insert(*axis) {
                return Err(TassadarBenchmarkPackageSetError::DuplicateAxis {
                    family: self.family,
                    axis: *axis,
                });
            }
        }

        let mut seen_packages = BTreeSet::new();
        for package in &self.benchmark_packages {
            package.validate()?;
            let package_key = format!("{}@{}", package.benchmark_ref, package.version);
            if !seen_packages.insert(package_key.clone()) {
                return Err(
                    TassadarBenchmarkPackageSetError::DuplicateBenchmarkPackage {
                        family: self.family,
                        package_key,
                    },
                );
            }
        }

        let mut seen_case_ids = BTreeSet::new();
        for case_id in &self.case_ids {
            if case_id.trim().is_empty() {
                return Err(TassadarBenchmarkPackageSetError::MissingCaseId {
                    family: self.family,
                });
            }
            if !seen_case_ids.insert(case_id.clone()) {
                return Err(TassadarBenchmarkPackageSetError::DuplicateCaseId {
                    family: self.family,
                    case_id: case_id.clone(),
                });
            }
        }

        Ok(())
    }
}

/// Public package-set contract for the current Tassadar benchmark families.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarBenchmarkPackageSetContract {
    /// Stable ABI version.
    pub abi_version: String,
    /// Stable package-set reference.
    pub package_set_ref: String,
    /// Immutable package-set version.
    pub version: String,
    /// Family contracts covered by the package set.
    pub families: Vec<TassadarBenchmarkFamilyContract>,
}

impl TassadarBenchmarkPackageSetContract {
    /// Creates and validates a package-set contract.
    pub fn new(
        package_set_ref: impl Into<String>,
        version: impl Into<String>,
        families: Vec<TassadarBenchmarkFamilyContract>,
    ) -> Result<Self, TassadarBenchmarkPackageSetError> {
        let contract = Self {
            abi_version: String::from(TASSADAR_BENCHMARK_PACKAGE_SET_ABI_VERSION),
            package_set_ref: package_set_ref.into(),
            version: version.into(),
            families,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Validates the contract.
    pub fn validate(&self) -> Result<(), TassadarBenchmarkPackageSetError> {
        if self.abi_version != TASSADAR_BENCHMARK_PACKAGE_SET_ABI_VERSION {
            return Err(TassadarBenchmarkPackageSetError::UnsupportedAbiVersion {
                abi_version: self.abi_version.clone(),
            });
        }
        if self.package_set_ref.trim().is_empty() {
            return Err(TassadarBenchmarkPackageSetError::MissingPackageSetRef);
        }
        if self.version.trim().is_empty() {
            return Err(TassadarBenchmarkPackageSetError::MissingPackageSetVersion);
        }
        if self.families.is_empty() {
            return Err(TassadarBenchmarkPackageSetError::MissingFamilies);
        }

        let mut seen_families = BTreeSet::new();
        for family in &self.families {
            family.validate()?;
            if !seen_families.insert(family.family) {
                return Err(TassadarBenchmarkPackageSetError::DuplicateFamily {
                    family: family.family,
                });
            }
        }

        Ok(())
    }

    /// Returns a stable digest over the contract.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(b"psionic_tassadar_benchmark_package_set_contract|", self)
    }
}

/// Benchmark-package-set validation failure.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TassadarBenchmarkPackageSetError {
    /// Unsupported ABI version.
    #[error("unsupported Tassadar benchmark package set ABI version `{abi_version}`")]
    UnsupportedAbiVersion {
        /// Observed ABI version.
        abi_version: String,
    },
    /// Missing package-set ref.
    #[error("Tassadar benchmark package set is missing `package_set_ref`")]
    MissingPackageSetRef,
    /// Missing package-set version.
    #[error("Tassadar benchmark package set is missing `version`")]
    MissingPackageSetVersion,
    /// No family contracts.
    #[error("Tassadar benchmark package set must contain at least one family contract")]
    MissingFamilies,
    /// Family repeated.
    #[error("Tassadar benchmark package set repeated family `{family:?}`")]
    DuplicateFamily {
        /// Repeated family.
        family: TassadarBenchmarkFamily,
    },
    /// Benchmark ref missing.
    #[error("Tassadar benchmark package binding is missing `benchmark_ref`")]
    MissingBenchmarkRef,
    /// Package version missing.
    #[error("Tassadar benchmark package `{benchmark_ref}` is missing `version`")]
    MissingBenchmarkVersion {
        /// Benchmark reference.
        benchmark_ref: String,
    },
    /// Environment ref missing.
    #[error("Tassadar benchmark package `{benchmark_ref}` is missing `environment_ref`")]
    MissingEnvironmentRef {
        /// Benchmark reference.
        benchmark_ref: String,
    },
    /// Dataset binding missing.
    #[error("Tassadar benchmark package `{benchmark_ref}` is missing a stable dataset binding")]
    MissingDatasetBinding {
        /// Benchmark reference.
        benchmark_ref: String,
    },
    /// Family summary missing.
    #[error("Tassadar benchmark family `{family:?}` is missing `summary`")]
    MissingFamilySummary {
        /// Benchmark family.
        family: TassadarBenchmarkFamily,
    },
    /// Family has no packages.
    #[error("Tassadar benchmark family `{family:?}` must reference at least one benchmark package")]
    MissingBenchmarkPackages {
        /// Benchmark family.
        family: TassadarBenchmarkFamily,
    },
    /// Family has no axes.
    #[error("Tassadar benchmark family `{family:?}` must declare axis coverage")]
    MissingAxisCoverage {
        /// Benchmark family.
        family: TassadarBenchmarkFamily,
    },
    /// Family has no cases.
    #[error("Tassadar benchmark family `{family:?}` must declare at least one case id")]
    MissingCaseIds {
        /// Benchmark family.
        family: TassadarBenchmarkFamily,
    },
    /// Empty case id.
    #[error("Tassadar benchmark family `{family:?}` contains an empty case id")]
    MissingCaseId {
        /// Benchmark family.
        family: TassadarBenchmarkFamily,
    },
    /// Repeated axis.
    #[error("Tassadar benchmark family `{family:?}` repeated axis `{axis:?}`")]
    DuplicateAxis {
        /// Benchmark family.
        family: TassadarBenchmarkFamily,
        /// Repeated axis.
        axis: TassadarBenchmarkAxis,
    },
    /// Repeated package.
    #[error("Tassadar benchmark family `{family:?}` repeated benchmark package `{package_key}`")]
    DuplicateBenchmarkPackage {
        /// Benchmark family.
        family: TassadarBenchmarkFamily,
        /// Repeated package key.
        package_key: String,
    },
    /// Repeated case id.
    #[error("Tassadar benchmark family `{family:?}` repeated case id `{case_id}`")]
    DuplicateCaseId {
        /// Benchmark family.
        family: TassadarBenchmarkFamily,
        /// Repeated case id.
        case_id: String,
    },
}

/// Topology classification for one public Tassadar trace family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarTraceFamilyTopology {
    /// Canonical CPU-style append-only trace authority.
    SequentialCpuReference,
    /// Parallel or wavefront-style target family.
    ParallelWavefront,
    /// Parallel frontier-style target family.
    ParallelFrontier,
}

/// Honest reconstruction scope admitted by one public Tassadar trace family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarTraceFamilyAuthorityScope {
    /// Full CPU-trace reconstruction is preserved.
    FullCpuTrace,
    /// Only final outputs are preserved exactly.
    FinalOutputsOnly,
}

/// One workload currently bound to one public Tassadar trace family.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarTraceFamilyWorkloadBinding {
    /// Stable workload or dataset reference.
    pub workload_ref: String,
    /// Honest claim boundary currently attached to this workload/family pairing.
    pub claim_boundary: String,
}

impl TassadarTraceFamilyWorkloadBinding {
    fn validate(&self, family_label: &str) -> Result<(), TassadarTraceFamilySetError> {
        if self.workload_ref.trim().is_empty() {
            return Err(TassadarTraceFamilySetError::MissingWorkloadRef {
                family_label: family_label.to_string(),
            });
        }
        if self.claim_boundary.trim().is_empty() {
            return Err(TassadarTraceFamilySetError::MissingWorkloadClaimBoundary {
                family_label: family_label.to_string(),
                workload_ref: self.workload_ref.clone(),
            });
        }
        Ok(())
    }
}

/// Public contract for one comparable Tassadar trace family.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarTraceFamilyContract {
    /// Stable trace-family label.
    pub family_label: String,
    /// High-level topology for the family.
    pub topology: TassadarTraceFamilyTopology,
    /// Human-readable family summary.
    pub summary: String,
    /// Optional dataset suffix used when this family materializes alternate targets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_suffix: Option<String>,
    /// Honest reconstruction scope for the family.
    pub authority_scope: TassadarTraceFamilyAuthorityScope,
    /// Workloads currently covered by this family.
    pub workloads: Vec<TassadarTraceFamilyWorkloadBinding>,
}

impl TassadarTraceFamilyContract {
    fn validate(&self) -> Result<(), TassadarTraceFamilySetError> {
        if self.family_label.trim().is_empty() {
            return Err(TassadarTraceFamilySetError::MissingFamilyLabel);
        }
        if self.summary.trim().is_empty() {
            return Err(TassadarTraceFamilySetError::MissingFamilySummary {
                family_label: self.family_label.clone(),
            });
        }
        if self
            .dataset_suffix
            .as_ref()
            .is_some_and(|suffix| suffix.trim().is_empty())
        {
            return Err(TassadarTraceFamilySetError::EmptyDatasetSuffix {
                family_label: self.family_label.clone(),
            });
        }
        if self.workloads.is_empty() {
            return Err(TassadarTraceFamilySetError::MissingWorkloads {
                family_label: self.family_label.clone(),
            });
        }

        let mut seen_workloads = BTreeSet::new();
        for workload in &self.workloads {
            workload.validate(self.family_label.as_str())?;
            if !seen_workloads.insert(workload.workload_ref.clone()) {
                return Err(TassadarTraceFamilySetError::DuplicateWorkloadRef {
                    family_label: self.family_label.clone(),
                    workload_ref: workload.workload_ref.clone(),
                });
            }
        }

        Ok(())
    }
}

/// Public package-set contract for comparable Tassadar trace families.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarTraceFamilySetContract {
    /// Stable ABI version.
    pub abi_version: String,
    /// Stable trace-family-set reference.
    pub trace_family_set_ref: String,
    /// Immutable set version.
    pub version: String,
    /// Public trace families in the set.
    pub families: Vec<TassadarTraceFamilyContract>,
}

impl TassadarTraceFamilySetContract {
    /// Creates and validates a comparable trace-family-set contract.
    pub fn new(
        trace_family_set_ref: impl Into<String>,
        version: impl Into<String>,
        families: Vec<TassadarTraceFamilyContract>,
    ) -> Result<Self, TassadarTraceFamilySetError> {
        let contract = Self {
            abi_version: String::from(TASSADAR_TRACE_FAMILY_SET_ABI_VERSION),
            trace_family_set_ref: trace_family_set_ref.into(),
            version: version.into(),
            families,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Validates the contract.
    pub fn validate(&self) -> Result<(), TassadarTraceFamilySetError> {
        if self.abi_version != TASSADAR_TRACE_FAMILY_SET_ABI_VERSION {
            return Err(TassadarTraceFamilySetError::UnsupportedAbiVersion {
                abi_version: self.abi_version.clone(),
            });
        }
        if self.trace_family_set_ref.trim().is_empty() {
            return Err(TassadarTraceFamilySetError::MissingTraceFamilySetRef);
        }
        if self.version.trim().is_empty() {
            return Err(TassadarTraceFamilySetError::MissingTraceFamilySetVersion);
        }
        if self.families.is_empty() {
            return Err(TassadarTraceFamilySetError::MissingFamilies);
        }

        let mut seen_family_labels = BTreeSet::new();
        for family in &self.families {
            family.validate()?;
            if !seen_family_labels.insert(family.family_label.clone()) {
                return Err(TassadarTraceFamilySetError::DuplicateFamilyLabel {
                    family_label: family.family_label.clone(),
                });
            }
        }

        Ok(())
    }

    /// Returns a stable digest over the contract.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(b"psionic_tassadar_trace_family_set_contract|", self)
    }
}

/// Trace-family-set validation failure.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TassadarTraceFamilySetError {
    /// Unsupported ABI version.
    #[error("unsupported Tassadar trace family set ABI version `{abi_version}`")]
    UnsupportedAbiVersion {
        /// Observed ABI version.
        abi_version: String,
    },
    /// Missing trace-family-set ref.
    #[error("Tassadar trace family set is missing `trace_family_set_ref`")]
    MissingTraceFamilySetRef,
    /// Missing version.
    #[error("Tassadar trace family set is missing `version`")]
    MissingTraceFamilySetVersion,
    /// No families were declared.
    #[error("Tassadar trace family set must contain at least one family contract")]
    MissingFamilies,
    /// Missing family label.
    #[error("Tassadar trace family contract is missing `family_label`")]
    MissingFamilyLabel,
    /// Missing family summary.
    #[error("Tassadar trace family `{family_label}` is missing `summary`")]
    MissingFamilySummary {
        /// Family label.
        family_label: String,
    },
    /// Empty dataset suffix.
    #[error("Tassadar trace family `{family_label}` contains an empty `dataset_suffix`")]
    EmptyDatasetSuffix {
        /// Family label.
        family_label: String,
    },
    /// Family has no workloads.
    #[error("Tassadar trace family `{family_label}` must declare at least one workload")]
    MissingWorkloads {
        /// Family label.
        family_label: String,
    },
    /// One workload ref was missing.
    #[error("Tassadar trace family `{family_label}` contains an empty `workload_ref`")]
    MissingWorkloadRef {
        /// Family label.
        family_label: String,
    },
    /// One workload claim boundary was missing.
    #[error(
        "Tassadar trace family `{family_label}` is missing `claim_boundary` for workload `{workload_ref}`"
    )]
    MissingWorkloadClaimBoundary {
        /// Family label.
        family_label: String,
        /// Workload reference.
        workload_ref: String,
    },
    /// One family label repeated.
    #[error("Tassadar trace family set repeated family `{family_label}`")]
    DuplicateFamilyLabel {
        /// Repeated family label.
        family_label: String,
    },
    /// One workload ref repeated inside a family.
    #[error("Tassadar trace family `{family_label}` repeated workload `{workload_ref}`")]
    DuplicateWorkloadRef {
        /// Family label.
        family_label: String,
        /// Repeated workload reference.
        workload_ref: String,
    },
}

/// Split identity used by the canonical Sudoku-v0 token-sequence dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarSequenceSplit {
    /// Main training split.
    Train,
    /// Held-out validation split.
    Validation,
    /// Final test split.
    Test,
}

impl TassadarSequenceSplit {
    /// Returns the stable split name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }

    fn dataset_kind(self) -> DatasetSplitKind {
        match self {
            Self::Train => DatasetSplitKind::Train,
            Self::Validation => DatasetSplitKind::Validation,
            Self::Test => DatasetSplitKind::Test,
        }
    }
}

/// Per-example lineage and curriculum metadata carried alongside one tokenized sequence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarSequenceExampleMetadata {
    /// Stable corpus case identifier.
    pub case_id: String,
    /// Stable puzzle identity derived from the raw puzzle cells.
    pub puzzle_digest: String,
    /// Stable validated-program identifier.
    pub program_id: String,
    /// Stable validated-program digest.
    pub program_digest: String,
    /// Stable program-artifact digest for CPU-reference truth.
    pub program_artifact_digest: String,
    /// Stable append-only trace digest.
    pub trace_digest: String,
    /// Stable behavior digest over the full execution.
    pub behavior_digest: String,
    /// Stable split assignment.
    pub split: TassadarSequenceSplit,
    /// Number of given Sudoku clues in the source puzzle.
    pub given_count: u32,
    /// Tokens in the program prompt prefix.
    pub prompt_token_count: u32,
    /// Tokens in the predicted trace suffix.
    pub target_token_count: u32,
    /// Total tokens in the full sequence.
    pub total_token_count: u32,
    /// Exact CPU-reference trace step count.
    pub trace_step_count: u32,
    /// Count of taken backward branches in the reference trace.
    pub backward_branch_count: u32,
    /// Maximum stack depth observed across the reference trace.
    pub max_stack_depth: u32,
}

/// One fully tokenized program-plus-trace example.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarSequenceExample {
    /// Stable sequence identifier.
    pub sequence_id: String,
    /// Stable ordered token ids in little-endian `u32` space.
    pub token_ids: Vec<u32>,
    /// Typed lineage and curriculum metadata.
    pub metadata: TassadarSequenceExampleMetadata,
}

impl TassadarSequenceExample {
    fn validate(&self) -> Result<(), TassadarSequenceDatasetError> {
        if self.sequence_id.trim().is_empty() {
            return Err(TassadarSequenceDatasetError::MissingSequenceId);
        }
        if self.token_ids.is_empty() {
            return Err(TassadarSequenceDatasetError::SequenceHasNoTokens {
                sequence_id: self.sequence_id.clone(),
            });
        }
        if self.metadata.total_token_count != self.token_ids.len() as u32 {
            return Err(TassadarSequenceDatasetError::TokenCountMismatch {
                sequence_id: self.sequence_id.clone(),
                declared: self.metadata.total_token_count,
                actual: self.token_ids.len() as u32,
            });
        }
        if self.metadata.prompt_token_count + self.metadata.target_token_count
            != self.metadata.total_token_count
        {
            return Err(TassadarSequenceDatasetError::PromptTargetBoundaryMismatch {
                sequence_id: self.sequence_id.clone(),
                prompt_tokens: self.metadata.prompt_token_count,
                target_tokens: self.metadata.target_token_count,
                total_tokens: self.metadata.total_token_count,
            });
        }
        if self.metadata.case_id.trim().is_empty() {
            return Err(TassadarSequenceDatasetError::MissingCaseId {
                sequence_id: self.sequence_id.clone(),
            });
        }
        if self.metadata.program_digest.trim().is_empty()
            || self.metadata.trace_digest.trim().is_empty()
            || self.metadata.behavior_digest.trim().is_empty()
            || self.metadata.program_artifact_digest.trim().is_empty()
        {
            return Err(TassadarSequenceDatasetError::MissingLineageDigest {
                sequence_id: self.sequence_id.clone(),
            });
        }
        Ok(())
    }
}

/// Full versioned dataset contract for Tassadar token sequences.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarSequenceDatasetContract {
    /// Stable ABI version.
    pub abi_version: String,
    /// Canonical dataset manifest.
    pub manifest: DatasetManifest,
    /// Stable digest over the vocabulary/tokenizer contract used to produce the token ids.
    pub vocabulary_digest: String,
    /// All tokenized examples across splits.
    pub examples: Vec<TassadarSequenceExample>,
}

impl TassadarSequenceDatasetContract {
    /// Builds the canonical dataset manifest and split shards from tokenized examples.
    pub fn from_examples(
        key: DatasetKey,
        display_name: impl Into<String>,
        tokenizer: TokenizerDigest,
        vocabulary_digest: impl Into<String>,
        examples: Vec<TassadarSequenceExample>,
    ) -> Result<Self, TassadarSequenceDatasetError> {
        let vocabulary_digest = vocabulary_digest.into();
        if vocabulary_digest.trim().is_empty() {
            return Err(TassadarSequenceDatasetError::MissingVocabularyDigest);
        }
        if examples.is_empty() {
            return Err(TassadarSequenceDatasetError::DatasetHasNoExamples);
        }

        let mut split_examples =
            BTreeMap::<TassadarSequenceSplit, Vec<TassadarSequenceExample>>::new();
        for example in examples.iter().cloned() {
            split_examples
                .entry(example.metadata.split)
                .or_default()
                .push(example);
        }

        let max_tokens = examples
            .iter()
            .map(|example| example.token_ids.len() as u32)
            .max()
            .unwrap_or(1);

        let mut split_declarations = Vec::new();
        let mut manifest = DatasetManifest::new(
            key.clone(),
            display_name,
            DatasetRecordEncoding::TokenIdsLeU32,
            tokenizer,
        )
        .with_context_window_tokens(max_tokens.max(1));

        for split in [
            TassadarSequenceSplit::Train,
            TassadarSequenceSplit::Validation,
            TassadarSequenceSplit::Test,
        ] {
            let Some(split_examples) = split_examples.get(&split) else {
                continue;
            };
            let split_name = split.as_str();
            let shard_key = format!("{split_name}-000");
            let payload = serialize_split_payload(split_examples.as_slice());
            let datastream_manifest = DatastreamManifest::from_bytes(
                format!("dataset://{}/{}", key.storage_key(), split_name),
                DatastreamSubjectKind::TokenizedCorpus,
                payload.as_slice(),
                payload.len().max(1),
                DatastreamEncoding::TokenIdsLeU32,
            )
            .with_dataset_binding(key.datastream_binding(split_name, shard_key.clone()))
            .with_provenance_digest(stable_digest(
                b"psionic_tassadar_sequence_split_payload|",
                &split_examples
                    .iter()
                    .map(|example| example.sequence_id.as_str())
                    .collect::<Vec<_>>(),
            ));
            let token_count = split_examples
                .iter()
                .map(|example| example.token_ids.len() as u64)
                .sum::<u64>();
            let min_tokens = split_examples
                .iter()
                .map(|example| example.token_ids.len() as u32)
                .min()
                .unwrap_or(1);
            let max_tokens = split_examples
                .iter()
                .map(|example| example.token_ids.len() as u32)
                .max()
                .unwrap_or(1);
            let shard = DatasetShardManifest::new(
                &key,
                split_name,
                shard_key,
                datastream_manifest.manifest_ref(),
                split_examples.len() as u64,
                token_count,
                min_tokens,
                max_tokens,
            )?;
            let declaration =
                DatasetSplitDeclaration::new(&key, split_name, split.dataset_kind(), vec![shard])?;
            split_declarations.push(declaration);

            manifest.metadata.insert(
                format!("tassadar.{}.sequence_ids", split_name),
                json!(
                    split_examples
                        .iter()
                        .map(|example| example.sequence_id.clone())
                        .collect::<Vec<_>>()
                ),
            );
        }

        manifest = manifest.with_splits(split_declarations);
        manifest.metadata.insert(
            String::from("tassadar.sequence_dataset_abi_version"),
            Value::String(String::from(TASSADAR_SEQUENCE_DATASET_ABI_VERSION)),
        );
        manifest.metadata.insert(
            String::from("tassadar.vocabulary_digest"),
            Value::String(vocabulary_digest.clone()),
        );
        manifest.metadata.insert(
            String::from("tassadar.example_count"),
            json!(examples.len()),
        );

        let contract = Self {
            abi_version: String::from(TASSADAR_SEQUENCE_DATASET_ABI_VERSION),
            manifest,
            vocabulary_digest,
            examples,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Returns the stable dataset storage key.
    #[must_use]
    pub fn storage_key(&self) -> String {
        self.manifest.storage_key()
    }

    /// Returns all examples for one split in stable dataset order.
    #[must_use]
    pub fn split_examples(&self, split: TassadarSequenceSplit) -> Vec<&TassadarSequenceExample> {
        self.examples
            .iter()
            .filter(|example| example.metadata.split == split)
            .collect()
    }

    /// Returns sequence descriptors for one split that can be fed into generic packing contracts.
    pub fn sequence_descriptors(
        &self,
        split: TassadarSequenceSplit,
    ) -> Vec<DatasetSequenceDescriptor> {
        let shard_key = format!("{}-000", split.as_str());
        self.split_examples(split)
            .into_iter()
            .enumerate()
            .map(|(index, example)| {
                DatasetSequenceDescriptor::new(
                    example.sequence_id.clone(),
                    shard_key.clone(),
                    index as u64,
                    example.token_ids.len() as u32,
                )
            })
            .collect()
    }

    /// Builds a generic packing plan for one split.
    pub fn packing_plan(
        &self,
        split: TassadarSequenceSplit,
        policy: &DatasetPackingPolicy,
    ) -> Result<DatasetPackingPlan, TassadarSequenceDatasetError> {
        let descriptors = self.sequence_descriptors(split);
        if descriptors.is_empty() {
            return Err(TassadarSequenceDatasetError::UnknownSplit {
                split_name: split.as_str().to_string(),
            });
        }
        Ok(policy.plan(descriptors.as_slice())?)
    }

    /// Validates the dataset contract and manifest coherence.
    pub fn validate(&self) -> Result<(), TassadarSequenceDatasetError> {
        if self.abi_version != TASSADAR_SEQUENCE_DATASET_ABI_VERSION {
            return Err(TassadarSequenceDatasetError::UnsupportedAbiVersion {
                abi_version: self.abi_version.clone(),
            });
        }
        if self.vocabulary_digest.trim().is_empty() {
            return Err(TassadarSequenceDatasetError::MissingVocabularyDigest);
        }
        self.manifest.validate()?;
        if self.examples.is_empty() {
            return Err(TassadarSequenceDatasetError::DatasetHasNoExamples);
        }

        let mut sequence_ids = BTreeSet::new();
        let mut expected_split_counts = BTreeMap::new();
        let mut expected_split_tokens = BTreeMap::new();
        for example in &self.examples {
            example.validate()?;
            if !sequence_ids.insert(example.sequence_id.clone()) {
                return Err(TassadarSequenceDatasetError::DuplicateSequenceId {
                    sequence_id: example.sequence_id.clone(),
                });
            }
            *expected_split_counts
                .entry(example.metadata.split.as_str().to_string())
                .or_insert(0_u64) += 1;
            *expected_split_tokens
                .entry(example.metadata.split.as_str().to_string())
                .or_insert(0_u64) += example.token_ids.len() as u64;
        }

        for split in &self.manifest.splits {
            let expected_count = expected_split_counts
                .get(split.split_name.as_str())
                .copied()
                .unwrap_or(0);
            let expected_tokens = expected_split_tokens
                .get(split.split_name.as_str())
                .copied()
                .unwrap_or(0);
            if split.sequence_count != expected_count {
                return Err(TassadarSequenceDatasetError::SplitSequenceCountMismatch {
                    split_name: split.split_name.clone(),
                    declared: split.sequence_count,
                    actual: expected_count,
                });
            }
            if split.token_count != expected_tokens {
                return Err(TassadarSequenceDatasetError::SplitTokenCountMismatch {
                    split_name: split.split_name.clone(),
                    declared: split.token_count,
                    actual: expected_tokens,
                });
            }
        }

        Ok(())
    }

    /// Returns a stable digest over the dataset contract.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(b"psionic_tassadar_sequence_dataset_contract|", self)
    }
}

/// Tassadar sequence dataset validation or packing failure.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TassadarSequenceDatasetError {
    /// Unsupported ABI version.
    #[error("unsupported Tassadar sequence dataset ABI version `{abi_version}`")]
    UnsupportedAbiVersion {
        /// Observed ABI version.
        abi_version: String,
    },
    /// Missing vocabulary digest.
    #[error("Tassadar sequence dataset is missing `vocabulary_digest`")]
    MissingVocabularyDigest,
    /// Missing sequence identifier.
    #[error("Tassadar sequence example is missing `sequence_id`")]
    MissingSequenceId,
    /// Empty dataset.
    #[error("Tassadar sequence dataset must contain at least one example")]
    DatasetHasNoExamples,
    /// Empty token sequence.
    #[error("Tassadar sequence `{sequence_id}` has no tokens")]
    SequenceHasNoTokens {
        /// Sequence identifier.
        sequence_id: String,
    },
    /// Duplicate sequence identifier.
    #[error("Tassadar sequence dataset repeated `sequence_id` `{sequence_id}`")]
    DuplicateSequenceId {
        /// Repeated sequence identifier.
        sequence_id: String,
    },
    /// Missing case identifier in metadata.
    #[error("Tassadar sequence `{sequence_id}` is missing `metadata.case_id`")]
    MissingCaseId {
        /// Sequence identifier.
        sequence_id: String,
    },
    /// One lineage digest was missing.
    #[error("Tassadar sequence `{sequence_id}` is missing one or more lineage digests")]
    MissingLineageDigest {
        /// Sequence identifier.
        sequence_id: String,
    },
    /// Total token count drifted from the visible token ids.
    #[error(
        "Tassadar sequence `{sequence_id}` declared total_token_count={declared} but carried {actual} tokens"
    )]
    TokenCountMismatch {
        /// Sequence identifier.
        sequence_id: String,
        /// Declared token count.
        declared: u32,
        /// Actual token count.
        actual: u32,
    },
    /// Prompt/target boundaries drifted from the total length.
    #[error(
        "Tassadar sequence `{sequence_id}` prompt/target boundary mismatch: prompt={prompt_tokens}, target={target_tokens}, total={total_tokens}"
    )]
    PromptTargetBoundaryMismatch {
        /// Sequence identifier.
        sequence_id: String,
        /// Prompt token count.
        prompt_tokens: u32,
        /// Target token count.
        target_tokens: u32,
        /// Total token count.
        total_tokens: u32,
    },
    /// Split requested for packing was not present.
    #[error("unknown Tassadar sequence split `{split_name}`")]
    UnknownSplit {
        /// Requested split name.
        split_name: String,
    },
    /// Split count drifted from the manifest.
    #[error(
        "Tassadar sequence split `{split_name}` declared sequence_count={declared} but examples derive {actual}"
    )]
    SplitSequenceCountMismatch {
        /// Split name.
        split_name: String,
        /// Manifest-declared count.
        declared: u64,
        /// Example-derived count.
        actual: u64,
    },
    /// Split token count drifted from the manifest.
    #[error(
        "Tassadar sequence split `{split_name}` declared token_count={declared} but examples derive {actual}"
    )]
    SplitTokenCountMismatch {
        /// Split name.
        split_name: String,
        /// Manifest-declared count.
        declared: u64,
        /// Example-derived count.
        actual: u64,
    },
    /// Generic dataset contract failure.
    #[error(transparent)]
    Dataset(#[from] DatasetContractError),
}

fn serialize_split_payload(examples: &[TassadarSequenceExample]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for example in examples {
        bytes.extend_from_slice(&(example.token_ids.len() as u32).to_le_bytes());
        for token in &example.token_ids {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
    }
    bytes
}

fn stable_digest<T>(prefix: &[u8], value: &T) -> String
where
    T: Serialize,
{
    let encoded =
        serde_json::to_vec(value).expect("Tassadar sequence dataset value should serialize");
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{
        TASSADAR_BENCHMARK_PACKAGE_SET_ABI_VERSION, TASSADAR_NUMERIC_OPCODE_LADDER_ABI_VERSION,
        TASSADAR_SEQUENCE_DATASET_ABI_VERSION, TASSADAR_TRACE_FAMILY_SET_ABI_VERSION,
        TassadarBenchmarkAxis, TassadarBenchmarkFamily, TassadarBenchmarkFamilyContract,
        TassadarBenchmarkPackageBinding, TassadarBenchmarkPackageSetContract,
        TassadarNumericOpcodeFamily, TassadarNumericOpcodeFamilyStatus,
        TassadarNumericOpcodeLadderContract, TassadarSequenceDatasetContract,
        TassadarSequenceExample, TassadarSequenceExampleMetadata, TassadarSequenceSplit,
        TassadarTraceFamilyAuthorityScope, TassadarTraceFamilyContract,
        TassadarTraceFamilySetContract, TassadarTraceFamilyTopology,
        TassadarTraceFamilyWorkloadBinding, tassadar_numeric_opcode_ladder_contract,
    };
    use crate::{DatasetKey, TokenizerDigest, TokenizerFamily};

    fn sample_example(split: TassadarSequenceSplit, suffix: &str) -> TassadarSequenceExample {
        TassadarSequenceExample {
            sequence_id: format!("seq-{suffix}"),
            token_ids: vec![1, 2, 3, 4],
            metadata: TassadarSequenceExampleMetadata {
                case_id: format!("case-{suffix}"),
                puzzle_digest: format!("puzzle-{suffix}"),
                program_id: format!("program-{suffix}"),
                program_digest: format!("program-digest-{suffix}"),
                program_artifact_digest: format!("artifact-digest-{suffix}"),
                trace_digest: format!("trace-digest-{suffix}"),
                behavior_digest: format!("behavior-digest-{suffix}"),
                split,
                given_count: 4,
                prompt_token_count: 2,
                target_token_count: 2,
                total_token_count: 4,
                trace_step_count: 1,
                backward_branch_count: 0,
                max_stack_depth: 1,
            },
        }
    }

    #[test]
    fn sequence_dataset_builds_manifest_and_split_contracts() {
        let dataset = TassadarSequenceDatasetContract::from_examples(
            DatasetKey::new("oa.tassadar.sudoku_v0.sequence", "train-v0"),
            "Tassadar Sudoku-v0 Sequence Dataset",
            TokenizerDigest::new(TokenizerFamily::Custom, "tokenizer-digest", 320),
            "vocab-digest",
            vec![
                sample_example(TassadarSequenceSplit::Train, "a"),
                sample_example(TassadarSequenceSplit::Validation, "b"),
                sample_example(TassadarSequenceSplit::Test, "c"),
            ],
        )
        .expect("dataset should build");

        assert_eq!(dataset.abi_version, TASSADAR_SEQUENCE_DATASET_ABI_VERSION);
        assert_eq!(
            dataset.manifest.record_encoding,
            crate::DatasetRecordEncoding::TokenIdsLeU32
        );
        assert_eq!(dataset.manifest.splits.len(), 3);
        assert_eq!(
            dataset.split_examples(TassadarSequenceSplit::Train).len(),
            1
        );
        assert_eq!(
            dataset
                .manifest
                .metadata
                .get("tassadar.vocabulary_digest")
                .expect("vocabulary digest metadata"),
            &Value::String(String::from("vocab-digest"))
        );
        assert!(!dataset.stable_digest().is_empty());
    }

    #[test]
    fn benchmark_package_set_contract_is_machine_legible() {
        let contract = TassadarBenchmarkPackageSetContract::new(
            "benchmark-set://openagents/tassadar/public",
            "2026.03.17",
            vec![
                TassadarBenchmarkFamilyContract {
                    family: TassadarBenchmarkFamily::Arithmetic,
                    summary: String::from("exact arithmetic kernels and validation microprograms"),
                    benchmark_packages: vec![
                        TassadarBenchmarkPackageBinding {
                            benchmark_ref: String::from(
                                "benchmark://openagents/tassadar/reference_fixture/validation_corpus",
                            ),
                            version: String::from("2026.03.17"),
                            environment_ref: String::from("env.openagents.tassadar.benchmark"),
                            dataset: DatasetKey::new(
                                "dataset://openagents/tassadar/validation_corpus",
                                "2026.03.17",
                            ),
                            split: Some(String::from("benchmark")),
                        },
                        TassadarBenchmarkPackageBinding {
                            benchmark_ref: String::from(
                                "benchmark://openagents/tassadar/compiled_kernel_suite/reference_fixture",
                            ),
                            version: String::from("v0"),
                            environment_ref: String::from(
                                "env.openagents.tassadar.compiled_kernel_suite.benchmark",
                            ),
                            dataset: DatasetKey::new(
                                "dataset://openagents/tassadar/compiled_kernel_suite",
                                "v0",
                            ),
                            split: Some(String::from("benchmark")),
                        },
                    ],
                    axis_coverage: vec![
                        TassadarBenchmarkAxis::Exactness,
                        TassadarBenchmarkAxis::LengthGeneralization,
                        TassadarBenchmarkAxis::PlannerUsefulness,
                    ],
                    case_ids: vec![String::from("locals_add")],
                },
                TassadarBenchmarkFamilyContract {
                    family: TassadarBenchmarkFamily::ClrsSubset,
                    summary: String::from("bounded CLRS-adjacent shortest-path witness"),
                    benchmark_packages: vec![TassadarBenchmarkPackageBinding {
                        benchmark_ref: String::from(
                            "benchmark://openagents/tassadar/reference_fixture/validation_corpus",
                        ),
                        version: String::from("2026.03.17"),
                        environment_ref: String::from("env.openagents.tassadar.benchmark"),
                        dataset: DatasetKey::new(
                            "dataset://openagents/tassadar/validation_corpus",
                            "2026.03.17",
                        ),
                        split: Some(String::from("benchmark")),
                    }],
                    axis_coverage: vec![
                        TassadarBenchmarkAxis::Exactness,
                        TassadarBenchmarkAxis::LengthGeneralization,
                        TassadarBenchmarkAxis::PlannerUsefulness,
                    ],
                    case_ids: vec![String::from("shortest_path_two_route")],
                },
            ],
        )
        .expect("package set contract should build");

        assert_eq!(
            contract.abi_version,
            TASSADAR_BENCHMARK_PACKAGE_SET_ABI_VERSION
        );
        assert_eq!(contract.families.len(), 2);
        assert!(!contract.stable_digest().is_empty());
    }

    #[test]
    fn numeric_opcode_ladder_contract_is_machine_legible() {
        let contract = tassadar_numeric_opcode_ladder_contract();
        assert_eq!(
            contract.abi_version,
            TASSADAR_NUMERIC_OPCODE_LADDER_ABI_VERSION
        );
        assert_eq!(contract.families.len(), 5);
        assert_eq!(
            contract.families[1].family,
            TassadarNumericOpcodeFamily::I32Comparisons
        );
        assert_eq!(
            contract.families[3].status,
            TassadarNumericOpcodeFamilyStatus::Refused
        );
        assert!(!contract.stable_digest().is_empty());
    }

    #[test]
    fn numeric_opcode_ladder_contract_rejects_duplicate_family() {
        let error = TassadarNumericOpcodeLadderContract::new(
            "tassadar://numeric_opcode_ladder/test",
            "v1",
            vec![
                tassadar_numeric_opcode_ladder_contract().families[0].clone(),
                tassadar_numeric_opcode_ladder_contract().families[0].clone(),
            ],
        )
        .expect_err("duplicate family should fail");
        assert!(format!("{error}").contains("repeated family"));
    }

    #[test]
    fn trace_family_set_contract_is_machine_legible() {
        let contract = TassadarTraceFamilySetContract::new(
            "trace-family-set://openagents/tassadar/sequence_variants",
            "trace-family-v1",
            vec![
                TassadarTraceFamilyContract {
                    family_label: String::from("sequential_cpu_reference"),
                    topology: TassadarTraceFamilyTopology::SequentialCpuReference,
                    summary: String::from(
                        "canonical CPU-style append-only authority trace for seeded Tassadar sequence workloads",
                    ),
                    dataset_suffix: None,
                    authority_scope: TassadarTraceFamilyAuthorityScope::FullCpuTrace,
                    workloads: vec![
                        TassadarTraceFamilyWorkloadBinding {
                            workload_ref: String::from("oa.tassadar.sudoku_v0.sequence"),
                            claim_boundary: String::from("learned_bounded"),
                        },
                        TassadarTraceFamilyWorkloadBinding {
                            workload_ref: String::from("oa.tassadar.sudoku_9x9.sequence"),
                            claim_boundary: String::from("learned_bounded"),
                        },
                    ],
                },
                TassadarTraceFamilyContract {
                    family_label: String::from("sudoku_diagonal_wavefront"),
                    topology: TassadarTraceFamilyTopology::ParallelWavefront,
                    summary: String::from(
                        "research-only anti-diagonal Sudoku wavefront target family",
                    ),
                    dataset_suffix: Some(String::from("sudoku_diagonal_wavefront")),
                    authority_scope: TassadarTraceFamilyAuthorityScope::FinalOutputsOnly,
                    workloads: vec![
                        TassadarTraceFamilyWorkloadBinding {
                            workload_ref: String::from("oa.tassadar.sudoku_v0.sequence"),
                            claim_boundary: String::from("research_only"),
                        },
                        TassadarTraceFamilyWorkloadBinding {
                            workload_ref: String::from("oa.tassadar.sudoku_9x9.sequence"),
                            claim_boundary: String::from("research_only"),
                        },
                    ],
                },
            ],
        )
        .expect("trace family set should build");

        assert_eq!(contract.abi_version, TASSADAR_TRACE_FAMILY_SET_ABI_VERSION);
        assert_eq!(contract.families.len(), 2);
        assert!(!contract.stable_digest().is_empty());
    }
}
