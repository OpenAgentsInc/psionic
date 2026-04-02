//! Local blob and catalog substrate for Psionic.

mod ollama;
mod registry;
mod tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate;
mod tassadar_post_article_plugin_manifest_identity_contract;
mod tassadar_post_article_starter_plugin_catalog_report;
mod benchmark_pack_manifest;
mod judge_pack_manifest;

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::UNIX_EPOCH,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub use ollama::*;
pub use registry::*;
pub use tassadar_post_article_plugin_authority_promotion_publication_and_trust_tier_gate::*;
pub use tassadar_post_article_plugin_manifest_identity_contract::*;
pub use tassadar_post_article_starter_plugin_catalog_report::*;
pub use benchmark_pack_manifest::*;
pub use judge_pack_manifest::*;

/// Human-readable crate ownership summary.
pub const CRATE_ROLE: &str = "local blob access and catalog substrate";

/// Local blob family known to the catalog layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalBlobKind {
    /// Standalone GGUF file discovered directly on disk.
    GgufFile,
    /// Ollama-managed blob resolved by digest inside the models directory.
    OllamaBlob,
}

/// Preferred local read strategy for a blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlobReadPreference {
    /// Prefer memory mapping but fall back to buffered reads when mmap fails.
    PreferMemoryMap,
    /// Require memory mapping and fail instead of falling back.
    RequireMemoryMap,
    /// Prefer a buffered in-memory read instead of mmap.
    PreferBuffered,
}

/// Actual local read path used after opening a blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlobReadPath {
    /// The blob bytes are exposed through a memory map.
    MemoryMapped,
    /// The blob bytes are exposed from a buffered in-memory copy.
    Buffered,
}

/// Integrity policy applied when opening a local blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlobIntegrityPolicy {
    /// Compute a stable SHA-256 over the full blob bytes.
    Sha256,
    /// Skip the full blob hash and emit a stable local-path metadata label instead.
    LocalUnverifiedLabel,
}

/// Open options for local blobs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalBlobOpenOptions {
    /// Preferred local read strategy.
    pub read_preference: BlobReadPreference,
    /// Logical page size to use for paged range views.
    pub page_size: usize,
    /// Integrity policy for the opened blob metadata.
    pub integrity_policy: BlobIntegrityPolicy,
    /// Optional digest expected by the caller.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_sha256: Option<String>,
}

impl Default for LocalBlobOpenOptions {
    fn default() -> Self {
        Self {
            read_preference: BlobReadPreference::PreferMemoryMap,
            page_size: 4096,
            integrity_policy: BlobIntegrityPolicy::Sha256,
            expected_sha256: None,
        }
    }
}

impl LocalBlobOpenOptions {
    /// Returns a copy with a different read preference.
    #[must_use]
    pub fn with_read_preference(mut self, read_preference: BlobReadPreference) -> Self {
        self.read_preference = read_preference;
        self
    }

    /// Returns a copy with a different logical page size.
    #[must_use]
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = page_size;
        self
    }

    /// Returns a copy with a different integrity policy.
    #[must_use]
    pub fn with_integrity_policy(mut self, integrity_policy: BlobIntegrityPolicy) -> Self {
        self.integrity_policy = integrity_policy;
        self
    }

    /// Returns a copy with an expected digest.
    #[must_use]
    pub fn with_expected_sha256(mut self, expected_sha256: impl Into<String>) -> Self {
        self.expected_sha256 = Some(expected_sha256.into());
        self
    }
}

/// Stable metadata for an opened local blob.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalBlobMetadata {
    /// Logical blob kind.
    pub kind: LocalBlobKind,
    /// Local file name.
    pub name: String,
    /// Absolute or caller-provided path used to open the blob.
    pub path: PathBuf,
    /// Blob length in bytes.
    pub byte_length: u64,
    /// Stable SHA-256 digest over the blob bytes.
    pub sha256: String,
    /// Actual local read path used for the bytes.
    pub read_path: BlobReadPath,
    /// Logical page size for paged range views.
    pub page_size: usize,
    /// Explicit fallback reason when a preferred mmap path was not used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

/// Blob access failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlobError {
    /// Opening or statting a blob file failed because it does not exist.
    #[error("blob `{path}` does not exist")]
    MissingFile {
        /// Path that could not be opened.
        path: String,
    },
    /// Opening or reading a blob failed for another reason.
    #[error("failed to read blob `{path}`: {message}")]
    Read {
        /// Path that failed.
        path: String,
        /// Failure summary.
        message: String,
    },
    /// Memory mapping was required but failed.
    #[error("failed to memory map blob `{path}`: {message}")]
    MemoryMap {
        /// Path that failed.
        path: String,
        /// Failure summary.
        message: String,
    },
    /// The requested page size is invalid.
    #[error("invalid blob page size `{page_size}`")]
    InvalidPageSize {
        /// Invalid page size requested by the caller.
        page_size: usize,
    },
}