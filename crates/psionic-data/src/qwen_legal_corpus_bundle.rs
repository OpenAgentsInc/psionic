use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    LegalBenchmarkArtifactRef, LegalBenchmarkDatasetShardConfig, LegalBenchmarkDatasetShardError,
    LegalBenchmarkDatasetShardManifest, LegalDpoDatasetBuilderConfig, LegalDpoDatasetBuilderError,
    LegalDpoExcludedInput, LegalDpoPreferencePair, LegalSftDatasetBuilderConfig,
    LegalSftDatasetBuilderError, LegalSftDatasetExample, LegalSftExcludedInput,
    build_legal_benchmark_dataset_shards, build_legal_benchmark_dpo_dataset,
    build_legal_benchmark_sft_dataset,
};

pub const QWEN_LEGAL_CORPUS_BUNDLE_SCHEMA_VERSION: &str = "psionic.qwen_legal.corpus_bundle.v1";
pub const QWEN_LEGAL_CORPUS_RECEIPT_SCHEMA_VERSION: &str = "psionic.qwen_legal.corpus_receipt.v1";
pub const QWEN_LEGAL_CORPUS_SPLIT_RULE_VERSION: &str = "psionic.qwen_legal.corpus_split_rules.v1";
pub const QWEN_LEGAL_GRPO_ROLLOUT_SEED_SCHEMA_VERSION: &str =
    "psionic.qwen_legal.grpo_rollout_seed.v1";
pub const QWEN_LEGAL_EVAL_PACK_SCHEMA_VERSION: &str = "psionic.qwen_legal.eval_pack.v1";

const DEFAULT_CORPUS_ID: &str = "qwen-legal-corpus-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenLegalCorpusBundleConfig {
    pub corpus_id: String,
    pub runs_root: PathBuf,
    pub out_dir: PathBuf,
    pub sft_shard_count: u32,
}

impl QwenLegalCorpusBundleConfig {
    #[must_use]
    pub fn new(runs_root: PathBuf, out_dir: PathBuf) -> Self {
        Self {
            corpus_id: String::from(DEFAULT_CORPUS_ID),
            runs_root,
            out_dir,
            sft_shard_count: 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenLegalCorpusSplit {
    Train,
    Dev,
    PrivateEval,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalCorpusSplitRule {
    pub rule_version: String,
    pub train_rule: String,
    pub dev_rule: String,
    pub private_eval_rule: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalCorpusSourcePolicy {
    pub source_class: String,
    pub trainable_when: String,
    pub rejected_when: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalCorpusLeakageReport {
    pub forbidden_markers: Vec<String>,
    pub checked_source_artifacts: usize,
    pub checked_generated_artifacts: usize,
    pub rejected_trainable_inputs: Vec<QwenLegalRejectedCorpusInput>,
    pub generated_artifacts_passed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalRejectedCorpusInput {
    pub source_path: String,
    pub source_run_id: Option<String>,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalGrpoRolloutSeed {
    pub schema_version: String,
    pub seed_id: String,
    pub split: QwenLegalCorpusSplit,
    pub source_pair_id: String,
    pub prompt: Vec<Value>,
    pub preferred_response: String,
    pub rejected_response: String,
    pub failure_family: String,
    pub source_run_ids: Vec<String>,
    pub rollout_budget: u32,
    pub reward_tags: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalEvalPack {
    pub schema_version: String,
    pub corpus_id: String,
    pub split_rule_version: String,
    pub public_dev_sft_example_ids: Vec<String>,
    pub public_dev_dpo_pair_ids: Vec<String>,
    pub grpo_dev_seed_ids: Vec<String>,
    pub private_eval_boundary: String,
    pub private_eval_answers_materialized: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalCorpusPylonShardRef {
    pub corpus_shard_id: String,
    pub dataset_kind: String,
    pub shard_id: String,
    pub shard_index: u32,
    pub shard_count: u32,
    pub manifest_hash: String,
    pub dataset_global_hash: String,
    pub artifact: LegalBenchmarkArtifactRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalCorpusBundleManifest {
    pub schema_version: String,
    pub corpus_id: String,
    pub split_rule: QwenLegalCorpusSplitRule,
    pub source_policy: Vec<QwenLegalCorpusSourcePolicy>,
    pub source_inputs: Vec<LegalBenchmarkArtifactRef>,
    pub generated_outputs: Vec<LegalBenchmarkArtifactRef>,
    pub sft_all_count: usize,
    pub sft_train_count: usize,
    pub sft_dev_count: usize,
    pub dpo_all_count: usize,
    pub dpo_train_count: usize,
    pub dpo_dev_count: usize,
    pub grpo_seed_count: usize,
    pub eval_pack_ref: LegalBenchmarkArtifactRef,
    pub sft_shard_manifest_ref: LegalBenchmarkArtifactRef,
    pub pylon_shard_refs: Vec<QwenLegalCorpusPylonShardRef>,
    pub leakage_report: QwenLegalCorpusLeakageReport,
    pub manifest_hash: String,
}

impl QwenLegalCorpusBundleManifest {
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.manifest_hash.clear();
        stable_json_digest(b"psionic_qwen_legal_corpus_bundle|", &clone)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QwenLegalCorpusReceipt {
    pub schema_version: String,
    pub corpus_id: String,
    pub manifest_path: String,
    pub manifest_hash: String,
    pub generated_output_count: usize,
    pub source_input_count: usize,
    pub pylon_shard_ids: Vec<String>,
    pub receipt_hash: String,
}

impl QwenLegalCorpusReceipt {
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.receipt_hash.clear();
        stable_json_digest(b"psionic_qwen_legal_corpus_receipt|", &clone)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenLegalCorpusBundleBuildResult {
    pub manifest_path: PathBuf,
    pub receipt_path: PathBuf,
    pub manifest: QwenLegalCorpusBundleManifest,
    pub receipt: QwenLegalCorpusReceipt,
}

#[derive(Debug, Error)]
pub enum QwenLegalCorpusBundleError {
    #[error("Qwen legal corpus argument error: {0}")]
    InvalidArgument(String),
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("JSON error at {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("SFT dataset build failed: {0}")]
    Sft(#[from] LegalSftDatasetBuilderError),
    #[error("DPO dataset build failed: {0}")]
    Dpo(#[from] LegalDpoDatasetBuilderError),
    #[error("dataset sharding failed: {0}")]
    Sharding(#[from] LegalBenchmarkDatasetShardError),
    #[error("generated artifact `{path}` failed leakage check: {reason}")]
    Leakage { path: PathBuf, reason: String },
}

pub fn build_qwen_legal_corpus_bundle(
    config: &QwenLegalCorpusBundleConfig,
) -> Result<QwenLegalCorpusBundleBuildResult, QwenLegalCorpusBundleError> {
    validate_config(config)?;
    fs::create_dir_all(config.out_dir.as_path()).map_err(|source| {
        QwenLegalCorpusBundleError::Io {
            path: config.out_dir.clone(),
            source,
        }
    })?;

    let sft_dir = config.out_dir.join("sft");
    let dpo_dir = config.out_dir.join("dpo");
    let grpo_dir = config.out_dir.join("grpo");
    let eval_dir = config.out_dir.join("eval");
    let shard_dir = config.out_dir.join("shards").join("sft_train");

    let sft_all_jsonl = sft_dir.join("all.jsonl");
    let sft_manifest_json = sft_dir.join("manifest.json");
    let sft_result = build_legal_benchmark_sft_dataset(&LegalSftDatasetBuilderConfig {
        runs_root: config.runs_root.clone(),
        out_jsonl: sft_all_jsonl.clone(),
        manifest_json: sft_manifest_json.clone(),
        dataset_id: format!("{}.sft.all", config.corpus_id),
    })?;

    let dpo_all_jsonl = dpo_dir.join("all.jsonl");
    let dpo_manifest_json = dpo_dir.join("manifest.json");
    let dpo_result = build_legal_benchmark_dpo_dataset(&LegalDpoDatasetBuilderConfig {
        runs_root: config.runs_root.clone(),
        out_jsonl: dpo_all_jsonl.clone(),
        manifest_json: dpo_manifest_json.clone(),
        dataset_id: format!("{}.dpo.all", config.corpus_id),
    })?;

    let sft_splits = split_sft_examples(sft_result.examples.as_slice());
    let dpo_splits = split_dpo_pairs(dpo_result.pairs.as_slice());

    let sft_train_jsonl = sft_dir.join("train.jsonl");
    let sft_dev_jsonl = sft_dir.join("dev.jsonl");
    write_jsonl(sft_train_jsonl.as_path(), sft_splits.train.as_slice())?;
    write_jsonl(sft_dev_jsonl.as_path(), sft_splits.dev.as_slice())?;

    let dpo_train_jsonl = dpo_dir.join("train.jsonl");
    let dpo_dev_jsonl = dpo_dir.join("dev.jsonl");
    write_jsonl(dpo_train_jsonl.as_path(), dpo_splits.train.as_slice())?;
    write_jsonl(dpo_dev_jsonl.as_path(), dpo_splits.dev.as_slice())?;

    let grpo_seeds = build_grpo_rollout_seeds(dpo_result.pairs.as_slice());
    let grpo_splits = split_grpo_seeds(grpo_seeds.as_slice());
    let grpo_all_jsonl = grpo_dir.join("rollout_seeds.jsonl");
    let grpo_train_jsonl = grpo_dir.join("rollout_seeds.train.jsonl");
    let grpo_dev_jsonl = grpo_dir.join("rollout_seeds.dev.jsonl");
    write_jsonl(grpo_all_jsonl.as_path(), grpo_seeds.as_slice())?;
    write_jsonl(grpo_train_jsonl.as_path(), grpo_splits.train.as_slice())?;
    write_jsonl(grpo_dev_jsonl.as_path(), grpo_splits.dev.as_slice())?;

    let eval_pack = QwenLegalEvalPack {
        schema_version: String::from(QWEN_LEGAL_EVAL_PACK_SCHEMA_VERSION),
        corpus_id: config.corpus_id.clone(),
        split_rule_version: String::from(QWEN_LEGAL_CORPUS_SPLIT_RULE_VERSION),
        public_dev_sft_example_ids: sft_splits
            .dev
            .iter()
            .map(|example| example.example_id.clone())
            .collect(),
        public_dev_dpo_pair_ids: dpo_splits
            .dev
            .iter()
            .map(|pair| pair.pair_id.clone())
            .collect(),
        grpo_dev_seed_ids: grpo_splits
            .dev
            .iter()
            .map(|seed| seed.seed_id.clone())
            .collect(),
        private_eval_boundary: String::from(
            "private eval tasks and answers are not materialized in trainable corpus artifacts",
        ),
        private_eval_answers_materialized: false,
    };
    let eval_pack_json = eval_dir.join("eval_pack.json");
    write_json(eval_pack_json.as_path(), &eval_pack)?;

    let shard_result = build_legal_benchmark_dataset_shards(&LegalBenchmarkDatasetShardConfig {
        dataset_jsonl: sft_train_jsonl.clone(),
        shard_count: config.sft_shard_count,
        out_dir: shard_dir.clone(),
        dataset_id: format!("{}.sft_train", config.corpus_id),
    })?;

    let source_inputs = source_artifacts(config.runs_root.as_path())?;
    let mut generated_outputs = vec![
        artifact_ref("sft-all-jsonl", "sft_jsonl", sft_all_jsonl.as_path())?,
        artifact_ref("sft-train-jsonl", "sft_jsonl", sft_train_jsonl.as_path())?,
        artifact_ref("sft-dev-jsonl", "sft_jsonl", sft_dev_jsonl.as_path())?,
        artifact_ref("sft-manifest", "sft_manifest", sft_manifest_json.as_path())?,
        artifact_ref("dpo-all-jsonl", "dpo_jsonl", dpo_all_jsonl.as_path())?,
        artifact_ref("dpo-train-jsonl", "dpo_jsonl", dpo_train_jsonl.as_path())?,
        artifact_ref("dpo-dev-jsonl", "dpo_jsonl", dpo_dev_jsonl.as_path())?,
        artifact_ref("dpo-manifest", "dpo_manifest", dpo_manifest_json.as_path())?,
        artifact_ref(
            "grpo-rollout-seeds",
            "grpo_rollout_seed_jsonl",
            grpo_all_jsonl.as_path(),
        )?,
        artifact_ref(
            "grpo-rollout-seeds-train",
            "grpo_rollout_seed_jsonl",
            grpo_train_jsonl.as_path(),
        )?,
        artifact_ref(
            "grpo-rollout-seeds-dev",
            "grpo_rollout_seed_jsonl",
            grpo_dev_jsonl.as_path(),
        )?,
        artifact_ref("eval-pack", "eval_pack", eval_pack_json.as_path())?,
        artifact_ref(
            "sft-shard-manifest",
            "dataset_shard_manifest",
            shard_result.manifest_path.as_path(),
        )?,
    ];
    generated_outputs.extend(shard_artifacts(&shard_result.manifest));
    check_generated_artifacts(generated_outputs.as_slice())?;

    let eval_pack_ref = artifact_ref("eval-pack", "eval_pack", eval_pack_json.as_path())?;
    let sft_shard_manifest_ref = artifact_ref(
        "sft-shard-manifest",
        "dataset_shard_manifest",
        shard_result.manifest_path.as_path(),
    )?;
    let pylon_shard_refs = pylon_shard_refs(&config.corpus_id, &shard_result.manifest);
    let rejected_trainable_inputs = rejected_inputs(
        &sft_result.manifest.excluded_inputs,
        &dpo_result.manifest.excluded_inputs,
    );
    let leakage_report = QwenLegalCorpusLeakageReport {
        forbidden_markers: forbidden_markers().into_iter().map(String::from).collect(),
        checked_source_artifacts: source_inputs.len(),
        checked_generated_artifacts: generated_outputs.len(),
        rejected_trainable_inputs,
        generated_artifacts_passed: true,
    };

    let mut manifest = QwenLegalCorpusBundleManifest {
        schema_version: String::from(QWEN_LEGAL_CORPUS_BUNDLE_SCHEMA_VERSION),
        corpus_id: config.corpus_id.clone(),
        split_rule: split_rule(),
        source_policy: source_policy(),
        source_inputs,
        generated_outputs,
        sft_all_count: sft_result.examples.len(),
        sft_train_count: sft_splits.train.len(),
        sft_dev_count: sft_splits.dev.len(),
        dpo_all_count: dpo_result.pairs.len(),
        dpo_train_count: dpo_splits.train.len(),
        dpo_dev_count: dpo_splits.dev.len(),
        grpo_seed_count: grpo_seeds.len(),
        eval_pack_ref,
        sft_shard_manifest_ref,
        pylon_shard_refs,
        leakage_report,
        manifest_hash: String::new(),
    };
    manifest.manifest_hash = manifest.stable_digest();
    let manifest_path = config.out_dir.join("manifest.json");
    write_json(manifest_path.as_path(), &manifest)?;

    let mut receipt = QwenLegalCorpusReceipt {
        schema_version: String::from(QWEN_LEGAL_CORPUS_RECEIPT_SCHEMA_VERSION),
        corpus_id: config.corpus_id.clone(),
        manifest_path: manifest_path.display().to_string(),
        manifest_hash: manifest.manifest_hash.clone(),
        generated_output_count: manifest.generated_outputs.len(),
        source_input_count: manifest.source_inputs.len(),
        pylon_shard_ids: manifest
            .pylon_shard_refs
            .iter()
            .map(|shard| shard.corpus_shard_id.clone())
            .collect(),
        receipt_hash: String::new(),
    };
    receipt.receipt_hash = receipt.stable_digest();
    let receipt_path = config.out_dir.join("receipt.json");
    write_json(receipt_path.as_path(), &receipt)?;

    Ok(QwenLegalCorpusBundleBuildResult {
        manifest_path,
        receipt_path,
        manifest,
        receipt,
    })
}

fn validate_config(config: &QwenLegalCorpusBundleConfig) -> Result<(), QwenLegalCorpusBundleError> {
    if config.corpus_id.trim().is_empty() {
        return Err(QwenLegalCorpusBundleError::InvalidArgument(String::from(
            "corpus_id must be present",
        )));
    }
    if !config.runs_root.exists() {
        return Err(QwenLegalCorpusBundleError::InvalidArgument(format!(
            "runs_root does not exist: {}",
            config.runs_root.display()
        )));
    }
    if config.sft_shard_count == 0 {
        return Err(QwenLegalCorpusBundleError::InvalidArgument(String::from(
            "sft_shard_count must be greater than zero",
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Split<T> {
    train: Vec<T>,
    dev: Vec<T>,
}

fn split_sft_examples(examples: &[LegalSftDatasetExample]) -> Split<LegalSftDatasetExample> {
    split_by_id(examples, |example| example.example_id.as_str())
}

fn split_dpo_pairs(pairs: &[LegalDpoPreferencePair]) -> Split<LegalDpoPreferencePair> {
    split_by_id(pairs, |pair| pair.pair_id.as_str())
}

fn split_grpo_seeds(seeds: &[QwenLegalGrpoRolloutSeed]) -> Split<QwenLegalGrpoRolloutSeed> {
    split_by_id(seeds, |seed| seed.seed_id.as_str())
}

fn split_by_id<T: Clone>(items: &[T], id: impl Fn(&T) -> &str) -> Split<T> {
    let mut train = Vec::new();
    let mut dev = Vec::new();
    for item in items {
        if stable_bucket(id(item), 10) == 0 {
            dev.push(item.clone());
        } else {
            train.push(item.clone());
        }
    }
    Split { train, dev }
}

fn split_for_id(id: &str) -> QwenLegalCorpusSplit {
    if stable_bucket(id, 10) == 0 {
        QwenLegalCorpusSplit::Dev
    } else {
        QwenLegalCorpusSplit::Train
    }
}

fn build_grpo_rollout_seeds(pairs: &[LegalDpoPreferencePair]) -> Vec<QwenLegalGrpoRolloutSeed> {
    let mut seeds = pairs
        .iter()
        .map(|pair| QwenLegalGrpoRolloutSeed {
            schema_version: String::from(QWEN_LEGAL_GRPO_ROLLOUT_SEED_SCHEMA_VERSION),
            seed_id: format!("{}.grpo_seed", pair.pair_id),
            split: split_for_id(pair.pair_id.as_str()),
            source_pair_id: pair.pair_id.clone(),
            prompt: pair
                .prompt
                .iter()
                .map(|message| {
                    serde_json::json!({
                        "role": message.role.clone(),
                        "content": message.content.clone(),
                    })
                })
                .collect(),
            preferred_response: pair.chosen.clone(),
            rejected_response: pair.rejected.clone(),
            failure_family: pair.reason.clone(),
            source_run_ids: pair.source_run_ids.clone(),
            rollout_budget: 4,
            reward_tags: vec![
                String::from("write_required_file"),
                String::from("correct_path"),
                String::from("source_grounded_answer"),
                pair.reason.clone(),
            ],
        })
        .collect::<Vec<_>>();
    seeds.sort_by(|left, right| left.seed_id.cmp(&right.seed_id));
    seeds
}

fn split_rule() -> QwenLegalCorpusSplitRule {
    QwenLegalCorpusSplitRule {
        rule_version: String::from(QWEN_LEGAL_CORPUS_SPLIT_RULE_VERSION),
        train_rule: String::from("sha256(stable_id) mod 10 != 0"),
        dev_rule: String::from("sha256(stable_id) mod 10 == 0"),
        private_eval_rule: String::from(
            "private eval inputs and answers are never materialized in trainable artifacts",
        ),
    }
}

fn source_policy() -> Vec<QwenLegalCorpusSourcePolicy> {
    vec![
        QwenLegalCorpusSourcePolicy {
            source_class: String::from("public_harvey_training_slice"),
            trainable_when: String::from("benchmark_visibility is public or public_training"),
            rejected_when: vec![
                String::from("private, hidden, judge-only, scorer-only, or harness-added markers"),
                String::from("answer integrity is invalid"),
                String::from("answer file was not created and last modified by the model"),
            ],
        },
        QwenLegalCorpusSourcePolicy {
            source_class: String::from("autopilot_blueprint_accepted_trajectory"),
            trainable_when: String::from(
                "data classification explicitly allows model training and no private eval content appears",
            ),
            rejected_when: vec![
                String::from("private benchmark criteria or answers appear"),
                String::from("classification is missing or audit-only"),
            ],
        },
        QwenLegalCorpusSourcePolicy {
            source_class: String::from("rejected_trace"),
            trainable_when: String::from("training_eligible is true and failure_family is present"),
            rejected_when: vec![String::from(
                "hidden, private, scorer-only, harness-assisted, or integrity-invalid",
            )],
        },
    ]
}

fn rejected_inputs(
    sft: &[LegalSftExcludedInput],
    dpo: &[LegalDpoExcludedInput],
) -> Vec<QwenLegalRejectedCorpusInput> {
    let mut seen = BTreeSet::new();
    let mut rejected = Vec::new();
    for input in sft {
        let key = format!(
            "{}|{}|{}",
            input.source_path,
            input.source_run_id.clone().unwrap_or_default(),
            input.reason
        );
        if seen.insert(key) {
            rejected.push(QwenLegalRejectedCorpusInput {
                source_path: input.source_path.clone(),
                source_run_id: input.source_run_id.clone(),
                reason: input.reason.clone(),
            });
        }
    }
    for input in dpo {
        let key = format!(
            "{}|{}|{}",
            input.source_path,
            input.source_run_id.clone().unwrap_or_default(),
            input.reason
        );
        if seen.insert(key) {
            rejected.push(QwenLegalRejectedCorpusInput {
                source_path: input.source_path.clone(),
                source_run_id: input.source_run_id.clone(),
                reason: input.reason.clone(),
            });
        }
    }
    rejected.sort_by(|left, right| left.source_path.cmp(&right.source_path));
    rejected
}

fn pylon_shard_refs(
    corpus_id: &str,
    manifest: &LegalBenchmarkDatasetShardManifest,
) -> Vec<QwenLegalCorpusPylonShardRef> {
    manifest
        .shards
        .iter()
        .map(|shard| QwenLegalCorpusPylonShardRef {
            corpus_shard_id: format!("{corpus_id}.sft_train.shard.{:05}", shard.shard_index),
            dataset_kind: String::from("sft_train"),
            shard_id: shard.shard_id.clone(),
            shard_index: shard.shard_index,
            shard_count: shard.shard_count,
            manifest_hash: manifest.manifest_hash.clone(),
            dataset_global_hash: manifest.dataset_global_hash.clone(),
            artifact: shard.local_artifact.clone(),
        })
        .collect()
}

fn shard_artifacts(
    manifest: &LegalBenchmarkDatasetShardManifest,
) -> Vec<LegalBenchmarkArtifactRef> {
    manifest
        .shards
        .iter()
        .flat_map(|shard| {
            [
                shard.local_artifact.clone(),
                shard.uploaded_artifact.clone(),
            ]
        })
        .collect()
}

fn source_artifacts(
    root: &Path,
) -> Result<Vec<LegalBenchmarkArtifactRef>, QwenLegalCorpusBundleError> {
    let mut paths = Vec::new();
    collect_source_files(root, &mut paths)?;
    paths.sort();
    paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            artifact_ref(
                format!("source-input-{index:05}").as_str(),
                source_artifact_type(path),
                path.as_path(),
            )
        })
        .collect()
}

fn collect_source_files(
    root: &Path,
    paths: &mut Vec<PathBuf>,
) -> Result<(), QwenLegalCorpusBundleError> {
    for entry in fs::read_dir(root).map_err(|source| QwenLegalCorpusBundleError::Io {
        path: root.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| QwenLegalCorpusBundleError::Io {
            path: root.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|source| QwenLegalCorpusBundleError::Io {
                path: path.clone(),
                source,
            })?;
        if file_type.is_dir() {
            collect_source_files(path.as_path(), paths)?;
        } else if file_type.is_file() && source_file_extension_allowed(path.as_path()) {
            paths.push(path);
        }
    }
    Ok(())
}

fn source_file_extension_allowed(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("json" | "jsonl" | "md" | "txt")
    )
}

fn source_artifact_type(path: &Path) -> &str {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("json") => "source_json",
        Some("jsonl") => "source_jsonl",
        Some("md") => "source_markdown",
        Some("txt") => "source_text",
        _ => "source_file",
    }
}

fn check_generated_artifacts(
    artifacts: &[LegalBenchmarkArtifactRef],
) -> Result<(), QwenLegalCorpusBundleError> {
    for artifact in artifacts {
        let path = PathBuf::from(&artifact.path);
        let content = fs::read_to_string(path.as_path()).map_err(|source| {
            QwenLegalCorpusBundleError::Io {
                path: path.clone(),
                source,
            }
        })?;
        if let Some(marker) = forbidden_markers()
            .iter()
            .find(|marker| content.to_lowercase().contains(**marker))
        {
            return Err(QwenLegalCorpusBundleError::Leakage {
                path,
                reason: format!("forbidden marker `{marker}` found in generated artifact"),
            });
        }
    }
    Ok(())
}

fn forbidden_markers() -> Vec<&'static str> {
    vec![
        "hidden benchmark answer",
        "hidden scoring label",
        "judge-only label",
        "scorer-only target",
        "harness-injected",
        "private benchmark answer",
        "private eval answer",
        "hidden/private eval answer",
        "83 / 83",
    ]
}

fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> Result<(), QwenLegalCorpusBundleError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| QwenLegalCorpusBundleError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let mut file = fs::File::create(path).map_err(|source| QwenLegalCorpusBundleError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    for record in records {
        serde_json::to_writer(&mut file, record).map_err(|source| {
            QwenLegalCorpusBundleError::Json {
                path: path.to_path_buf(),
                source,
            }
        })?;
        writeln!(file).map_err(|source| QwenLegalCorpusBundleError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), QwenLegalCorpusBundleError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| QwenLegalCorpusBundleError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let bytes =
        serde_json::to_vec_pretty(value).map_err(|source| QwenLegalCorpusBundleError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    fs::write(path, bytes).map_err(|source| QwenLegalCorpusBundleError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn artifact_ref(
    artifact_id: &str,
    artifact_type: &str,
    path: &Path,
) -> Result<LegalBenchmarkArtifactRef, QwenLegalCorpusBundleError> {
    let bytes = fs::read(path).map_err(|source| QwenLegalCorpusBundleError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(LegalBenchmarkArtifactRef {
        artifact_id: String::from(artifact_id),
        artifact_type: String::from(artifact_type),
        path: path.display().to_string(),
        sha256: sha256_hex(bytes.as_slice()),
        byte_len: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
    })
}

fn stable_bucket(id: &str, modulo: u64) -> u64 {
    let digest = Sha256::digest(id.as_bytes());
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(bytes) % modulo
}

fn stable_json_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    if let Ok(bytes) = serde_json::to_vec(value) {
        hasher.update(bytes);
    }
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_bundle_rejects_private_material_from_trainable_outputs() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runs = temp.path().join("runs");
        write_good_run(
            &runs,
            "public-good",
            "run.public.good",
            "harvey.public.good",
            "public",
            "Public training answer.",
        );
        write_good_run(
            &runs,
            "private-leak",
            "run.private.leak",
            "harvey.private.leak",
            "private",
            "PRIVATE BENCHMARK ANSWER: do not train on this.",
        );
        write_bad_run(&runs, "bad-missing-file", "missing_file");

        let result = build_qwen_legal_corpus_bundle(&QwenLegalCorpusBundleConfig {
            corpus_id: String::from("qwen-legal-corpus-test"),
            runs_root: runs,
            out_dir: temp.path().join("bundle"),
            sft_shard_count: 2,
        })
        .expect("corpus bundle");

        assert!(result.manifest.sft_all_count > 0);
        assert!(result.manifest.dpo_all_count > 0);
        assert!(result.manifest.grpo_seed_count > 0);
        assert!(
            result
                .manifest
                .leakage_report
                .rejected_trainable_inputs
                .iter()
                .any(|input| input.reason.contains("private benchmark")),
        );
        let sft_all = fs::read_to_string(temp.path().join("bundle/sft/all.jsonl"))
            .expect("sft all should exist");
        assert!(!sft_all.contains("PRIVATE BENCHMARK ANSWER"));
        assert!(!sft_all.contains("harvey.private.leak"));
    }

    #[test]
    fn corpus_bundle_emits_pylon_shard_refs() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runs = temp.path().join("runs");
        write_good_run(
            &runs,
            "public-good",
            "run.public.good",
            "harvey.public.good",
            "public",
            "Public training answer.",
        );
        write_bad_run(&runs, "bad-missing-file", "missing_file");

        let result = build_qwen_legal_corpus_bundle(&QwenLegalCorpusBundleConfig {
            corpus_id: String::from("qwen-legal-corpus-shards"),
            runs_root: runs,
            out_dir: temp.path().join("bundle"),
            sft_shard_count: 2,
        })
        .expect("corpus bundle");

        assert_eq!(result.manifest.pylon_shard_refs.len(), 2);
        assert!(
            result
                .receipt
                .pylon_shard_ids
                .iter()
                .all(|id| { id.starts_with("qwen-legal-corpus-shards.sft_train.shard.") })
        );
        assert_eq!(result.manifest.manifest_hash, result.receipt.manifest_hash);
        assert_eq!(
            result.manifest.manifest_hash,
            result.manifest.stable_digest()
        );
        assert_eq!(result.receipt.receipt_hash, result.receipt.stable_digest());
    }

    fn write_good_run(
        root: &Path,
        folder: &str,
        run_id: &str,
        task_id: &str,
        visibility: &str,
        answer: &str,
    ) {
        let run_dir = root.join(folder);
        let output_dir = run_dir.join("output");
        fs::create_dir_all(output_dir.as_path()).expect("output dir");
        fs::write(output_dir.join("memo.md"), answer).expect("answer");
        let receipt = serde_json::json!({
            "run_spec": {
                "run_id": run_id,
                "task_id": task_id,
                "benchmark_visibility": visibility
            },
            "integrity": {
                "valid": true
            },
            "answer_files": [{
                "relative_path": "memo.md",
                "integrity_valid": true,
                "creation_actor": "model",
                "last_modifying_actor": "model"
            }],
            "tool_calls": []
        });
        fs::write(
            run_dir.join("run_receipt.json"),
            serde_json::to_vec_pretty(&receipt).expect("receipt json"),
        )
        .expect("receipt");
    }

    fn write_bad_run(root: &Path, folder: &str, failure_class: &str) {
        let run_dir = root.join(folder);
        fs::create_dir_all(run_dir.as_path()).expect("bad dir");
        let bad = serde_json::json!({
            "example_id": format!("bad_run.{folder}"),
            "base_task_id": "harvey.public.good",
            "benchmark_visibility": "public_training",
            "failure_class": failure_class,
            "full_prompt": "Draft the memo and write it to memo.md.",
            "required_file_paths": ["memo.md"],
            "full_model_response": "I answered in chat and did not write the file.",
            "suggested_correction": "Write memo.md through the output file tool, validate it exists, then submit.",
            "training_eligible": true
        });
        fs::write(
            run_dir.join("bad_run.json"),
            serde_json::to_vec_pretty(&bad).expect("bad json"),
        )
        .expect("bad run");
    }
}
