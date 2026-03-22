use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Duration,
};

#[cfg(test)]
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;
use url::Url;

use crate::TASSADAR_POST_ARTICLE_PLUGIN_PACKET_ABI_VERSION;

pub const TASSADAR_POST_ARTICLE_PLUGIN_TEXT_URL_EXTRACT_RUNTIME_BUNDLE_REF: &str = "fixtures/tassadar/runs/tassadar_post_article_plugin_text_url_extract_v1/tassadar_post_article_plugin_text_url_extract_bundle.json";
pub const TASSADAR_POST_ARTICLE_PLUGIN_TEXT_URL_EXTRACT_RUN_ROOT_REF: &str =
    "fixtures/tassadar/runs/tassadar_post_article_plugin_text_url_extract_v1";
pub const TASSADAR_POST_ARTICLE_PLUGIN_HTTP_FETCH_TEXT_RUNTIME_BUNDLE_REF: &str =
    "fixtures/tassadar/runs/tassadar_post_article_plugin_http_fetch_text_v1/tassadar_post_article_plugin_http_fetch_text_bundle.json";
pub const TASSADAR_POST_ARTICLE_PLUGIN_HTTP_FETCH_TEXT_RUN_ROOT_REF: &str =
    "fixtures/tassadar/runs/tassadar_post_article_plugin_http_fetch_text_v1";
pub const TASSADAR_POST_ARTICLE_PLUGIN_HTML_EXTRACT_READABLE_RUNTIME_BUNDLE_REF: &str =
    "fixtures/tassadar/runs/tassadar_post_article_plugin_html_extract_readable_v1/tassadar_post_article_plugin_html_extract_readable_bundle.json";
pub const TASSADAR_POST_ARTICLE_PLUGIN_HTML_EXTRACT_READABLE_RUN_ROOT_REF: &str =
    "fixtures/tassadar/runs/tassadar_post_article_plugin_html_extract_readable_v1";

pub const STARTER_PLUGIN_VERSION: &str = "v1";
pub const STARTER_PLUGIN_TEXT_URL_EXTRACT_ID: &str = "plugin.text.url_extract";
pub const STARTER_PLUGIN_TEXT_URL_EXTRACT_TOOL_NAME: &str = "plugin_text_url_extract";
pub const STARTER_PLUGIN_TEXT_URL_EXTRACT_INPUT_SCHEMA_ID: &str =
    "plugin.text.url_extract.input.v1";
pub const STARTER_PLUGIN_TEXT_URL_EXTRACT_OUTPUT_SCHEMA_ID: &str =
    "plugin.text.url_extract.output.v1";
pub const STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID: &str = "plugin.refusal.schema_invalid.v1";
pub const STARTER_PLUGIN_REFUSAL_PACKET_TOO_LARGE_ID: &str = "plugin.refusal.packet_too_large.v1";
pub const STARTER_PLUGIN_REFUSAL_UNSUPPORTED_CODEC_ID: &str = "plugin.refusal.unsupported_codec.v1";
pub const STARTER_PLUGIN_REFUSAL_RUNTIME_RESOURCE_LIMIT_ID: &str =
    "plugin.refusal.runtime_resource_limit.v1";
pub const STARTER_PLUGIN_HTTP_FETCH_TEXT_ID: &str = "plugin.http.fetch_text";
pub const STARTER_PLUGIN_HTTP_FETCH_TEXT_TOOL_NAME: &str = "plugin_http_fetch_text";
pub const STARTER_PLUGIN_HTTP_FETCH_TEXT_INPUT_SCHEMA_ID: &str =
    "plugin.http.fetch_text.input.v1";
pub const STARTER_PLUGIN_HTTP_FETCH_TEXT_OUTPUT_SCHEMA_ID: &str =
    "plugin.http.fetch_text.output.v1";
pub const STARTER_PLUGIN_HTML_EXTRACT_READABLE_ID: &str = "plugin.html.extract_readable";
pub const STARTER_PLUGIN_HTML_EXTRACT_READABLE_TOOL_NAME: &str =
    "plugin_html_extract_readable";
pub const STARTER_PLUGIN_HTML_EXTRACT_READABLE_INPUT_SCHEMA_ID: &str =
    "plugin.html.extract_readable.input.v1";
pub const STARTER_PLUGIN_HTML_EXTRACT_READABLE_OUTPUT_SCHEMA_ID: &str =
    "plugin.html.extract_readable.output.v1";
pub const STARTER_PLUGIN_REFUSAL_NETWORK_DENIED_ID: &str = "plugin.refusal.network_denied.v1";
pub const STARTER_PLUGIN_REFUSAL_URL_NOT_PERMITTED_ID: &str =
    "plugin.refusal.url_not_permitted.v1";
pub const STARTER_PLUGIN_REFUSAL_TIMEOUT_ID: &str = "plugin.refusal.timeout.v1";
pub const STARTER_PLUGIN_REFUSAL_RESPONSE_TOO_LARGE_ID: &str =
    "plugin.refusal.response_too_large.v1";
pub const STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID: &str =
    "plugin.refusal.content_type_unsupported.v1";
pub const STARTER_PLUGIN_REFUSAL_DECODE_FAILED_ID: &str =
    "plugin.refusal.decode_failed.v1";
pub const STARTER_PLUGIN_REFUSAL_UPSTREAM_FAILURE_ID: &str =
    "plugin.refusal.upstream_failure.v1";
pub const STARTER_PLUGIN_REFUSAL_INPUT_TOO_LARGE_ID: &str =
    "plugin.refusal.input_too_large.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StarterPluginInvocationStatus {
    Success,
    Refusal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarterPluginToolProjection {
    pub plugin_id: String,
    pub tool_name: String,
    pub description: String,
    pub arguments_schema: Value,
    pub result_schema_id: String,
    pub refusal_schema_ids: Vec<String>,
    pub replay_class_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarterPluginInvocationReceipt {
    pub receipt_id: String,
    pub plugin_id: String,
    pub plugin_version: String,
    pub tool_name: String,
    pub packet_abi_version: String,
    pub mount_envelope_id: String,
    pub capability_namespace_ids: Vec<String>,
    pub replay_class_id: String,
    pub status: StarterPluginInvocationStatus,
    pub input_schema_id: String,
    pub input_packet_digest: String,
    pub output_or_refusal_schema_id: String,
    pub output_or_refusal_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal_class_id: Option<String>,
    pub detail: String,
    pub receipt_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarterPluginRefusal {
    pub schema_id: String,
    pub refusal_class_id: String,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UrlExtractRequest {
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UrlExtractResponse {
    pub urls: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UrlExtractConfig {
    pub packet_size_limit_bytes: usize,
    pub max_urls: usize,
}

impl Default for UrlExtractConfig {
    fn default() -> Self {
        Self {
            packet_size_limit_bytes: 16 * 1024,
            max_urls: 128,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UrlExtractInvocationOutcome {
    pub receipt: StarterPluginInvocationReceipt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<UrlExtractResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<StarterPluginRefusal>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UrlExtractRuntimeCaseStatus {
    ExactSuccess,
    TypedMalformedPacket,
    TypedRefusal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UrlExtractRuntimeCase {
    pub case_id: String,
    pub status: UrlExtractRuntimeCaseStatus,
    pub codec_id: String,
    pub request_packet_digest: String,
    pub response_or_refusal_schema_id: String,
    pub response_or_refusal_digest: String,
    pub receipt: StarterPluginInvocationReceipt,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UrlExtractRuntimeBundle {
    pub schema_version: u16,
    pub bundle_id: String,
    pub plugin_id: String,
    pub plugin_version: String,
    pub manifest_id: String,
    pub artifact_id: String,
    pub packet_abi_version: String,
    pub mount_envelope_id: String,
    pub tool_projection: StarterPluginToolProjection,
    pub negative_claim_ids: Vec<String>,
    pub case_rows: Vec<UrlExtractRuntimeCase>,
    pub claim_boundary: String,
    pub summary: String,
    pub bundle_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchTextRequest {
    pub url: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchTextResponse {
    pub final_url: String,
    pub status_code: u16,
    pub content_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub charset: Option<String>,
    pub body_text: String,
    pub truncated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchTextMountEnvelope {
    pub envelope_id: String,
    pub allowlisted_url_prefixes: Vec<String>,
    pub timeout_millis: u64,
    pub response_size_limit_bytes: usize,
    pub redirect_limit: usize,
    pub allowed_content_type_ids: Vec<String>,
    pub replay_class_id: String,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchTextSnapshotResponse {
    pub final_url: String,
    pub status_code: u16,
    pub content_type: String,
    pub charset: Option<String>,
    pub body_bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FetchTextSnapshotResult {
    Success(FetchTextSnapshotResponse),
    Timeout,
    NetworkDenied { detail: String },
    UpstreamFailure { detail: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FetchTextBackend {
    Snapshot(BTreeMap<String, FetchTextSnapshotResult>),
    Live,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchTextConfig {
    pub mount_envelope: FetchTextMountEnvelope,
    pub backend: FetchTextBackend,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchTextInvocationOutcome {
    pub receipt: StarterPluginInvocationReceipt,
    pub backend_id: String,
    pub logical_cpu_millis: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<FetchTextResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<StarterPluginRefusal>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FetchTextRuntimeCaseStatus {
    ExactSuccess,
    TypedMalformedPacket,
    TypedRefusal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchTextRuntimeCase {
    pub case_id: String,
    pub status: FetchTextRuntimeCaseStatus,
    pub codec_id: String,
    pub request_packet_digest: String,
    pub response_or_refusal_schema_id: String,
    pub response_or_refusal_digest: String,
    pub backend_id: String,
    pub logical_cpu_millis: u32,
    pub receipt: StarterPluginInvocationReceipt,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchTextRuntimeBundle {
    pub schema_version: u16,
    pub bundle_id: String,
    pub plugin_id: String,
    pub plugin_version: String,
    pub manifest_id: String,
    pub artifact_id: String,
    pub packet_abi_version: String,
    pub sample_mount_envelope: FetchTextMountEnvelope,
    pub tool_projection: StarterPluginToolProjection,
    pub supported_replay_class_ids: Vec<String>,
    pub negative_claim_ids: Vec<String>,
    pub case_rows: Vec<FetchTextRuntimeCase>,
    pub claim_boundary: String,
    pub summary: String,
    pub bundle_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractReadableRequest {
    pub source_url: String,
    pub content_type: String,
    pub body_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractReadableResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excerpt: Option<String>,
    pub readable_text: String,
    pub harvested_links: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_language: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractReadableConfig {
    pub input_size_limit_bytes: usize,
    pub max_links: usize,
}

impl Default for ExtractReadableConfig {
    fn default() -> Self {
        Self {
            input_size_limit_bytes: 64 * 1024,
            max_links: 128,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractReadableInvocationOutcome {
    pub receipt: StarterPluginInvocationReceipt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<ExtractReadableResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<StarterPluginRefusal>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractReadableRuntimeCaseStatus {
    ExactSuccess,
    TypedMalformedPacket,
    TypedRefusal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractReadableRuntimeCase {
    pub case_id: String,
    pub status: ExtractReadableRuntimeCaseStatus,
    pub codec_id: String,
    pub request_packet_digest: String,
    pub response_or_refusal_schema_id: String,
    pub response_or_refusal_digest: String,
    pub receipt: StarterPluginInvocationReceipt,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractReadableCompositionCase {
    pub case_id: String,
    pub step_plugin_ids: Vec<String>,
    pub step_receipt_ids: Vec<String>,
    pub schema_repair_allowed: bool,
    pub hidden_host_extraction_allowed: bool,
    pub green: bool,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractReadableRuntimeBundle {
    pub schema_version: u16,
    pub bundle_id: String,
    pub plugin_id: String,
    pub plugin_version: String,
    pub manifest_id: String,
    pub artifact_id: String,
    pub packet_abi_version: String,
    pub mount_envelope_id: String,
    pub tool_projection: StarterPluginToolProjection,
    pub negative_claim_ids: Vec<String>,
    pub case_rows: Vec<ExtractReadableRuntimeCase>,
    pub composition_case: ExtractReadableCompositionCase,
    pub claim_boundary: String,
    pub summary: String,
    pub bundle_digest: String,
}

#[derive(Debug, Error)]
pub enum StarterPluginRuntimeError {
    #[error("failed to create `{path}`: {error}")]
    CreateDir { path: String, error: std::io::Error },
    #[error("failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error("failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("failed to decode `{path}`: {error}")]
    Decode {
        path: String,
        error: serde_json::Error,
    },
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[must_use]
pub fn url_extract_tool_projection() -> StarterPluginToolProjection {
    StarterPluginToolProjection {
        plugin_id: String::from(STARTER_PLUGIN_TEXT_URL_EXTRACT_ID),
        tool_name: String::from(STARTER_PLUGIN_TEXT_URL_EXTRACT_TOOL_NAME),
        description: String::from(
            "extract literal http:// and https:// strings from packet-local text without URL validation, DNS, or network reachability claims.",
        ),
        arguments_schema: json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["text"],
            "properties": {
                "text": {
                    "type": "string",
                    "description": "packet-local input text scanned with the bounded https?://[^\\\\s]+ rule."
                }
            }
        }),
        result_schema_id: String::from(STARTER_PLUGIN_TEXT_URL_EXTRACT_OUTPUT_SCHEMA_ID),
        refusal_schema_ids: vec![
            String::from(STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID),
            String::from(STARTER_PLUGIN_REFUSAL_PACKET_TOO_LARGE_ID),
            String::from(STARTER_PLUGIN_REFUSAL_UNSUPPORTED_CODEC_ID),
            String::from(STARTER_PLUGIN_REFUSAL_RUNTIME_RESOURCE_LIMIT_ID),
        ],
        replay_class_id: String::from("deterministic_replayable"),
    }
}

#[must_use]
pub fn invoke_url_extract_json_packet(
    codec_id: &str,
    packet_bytes: &[u8],
    config: &UrlExtractConfig,
) -> UrlExtractInvocationOutcome {
    let input_packet_digest = sha256_digest(packet_bytes);
    if codec_id != "json" {
        return url_extract_refusal_outcome(
            &input_packet_digest,
            STARTER_PLUGIN_REFUSAL_UNSUPPORTED_CODEC_ID,
            "unsupported_codec",
            "url extract accepts only json packet input under packet.v1.",
        );
    }
    if packet_bytes.len() > config.packet_size_limit_bytes {
        return url_extract_refusal_outcome(
            &input_packet_digest,
            STARTER_PLUGIN_REFUSAL_PACKET_TOO_LARGE_ID,
            "packet_too_large",
            "url extract keeps packet size ceilings explicit instead of relying on ambient parser allocation behavior.",
        );
    }
    let request = match serde_json::from_slice::<UrlExtractRequest>(packet_bytes) {
        Ok(request) => request,
        Err(_) => {
            return url_extract_refusal_outcome(
                &input_packet_digest,
                STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID,
                "schema_invalid",
                "url extract refuses malformed packets without host-side schema repair.",
            );
        }
    };

    let urls = match extract_urls(&request.text, config.max_urls) {
        Ok(urls) => urls,
        Err(refusal) => {
            return url_extract_refusal_outcome(
                &input_packet_digest,
                STARTER_PLUGIN_REFUSAL_RUNTIME_RESOURCE_LIMIT_ID,
                "runtime_resource_limit",
                refusal,
            );
        }
    };
    let response = UrlExtractResponse { urls };
    let output_or_refusal_digest = stable_json_digest(b"url_extract_response|", &response);
    let mut receipt = StarterPluginInvocationReceipt {
        receipt_id: format!(
            "receipt.{}.{}.v1",
            STARTER_PLUGIN_TEXT_URL_EXTRACT_ID,
            &input_packet_digest[..16]
        ),
        plugin_id: String::from(STARTER_PLUGIN_TEXT_URL_EXTRACT_ID),
        plugin_version: String::from(STARTER_PLUGIN_VERSION),
        tool_name: String::from(STARTER_PLUGIN_TEXT_URL_EXTRACT_TOOL_NAME),
        packet_abi_version: String::from(TASSADAR_POST_ARTICLE_PLUGIN_PACKET_ABI_VERSION),
        mount_envelope_id: String::from("mount.plugin.text.url_extract.no_capabilities.v1"),
        capability_namespace_ids: Vec::new(),
        replay_class_id: String::from("deterministic_replayable"),
        status: StarterPluginInvocationStatus::Success,
        input_schema_id: String::from(STARTER_PLUGIN_TEXT_URL_EXTRACT_INPUT_SCHEMA_ID),
        input_packet_digest,
        output_or_refusal_schema_id: String::from(STARTER_PLUGIN_TEXT_URL_EXTRACT_OUTPUT_SCHEMA_ID),
        output_or_refusal_digest,
        refusal_class_id: None,
        detail: String::from(
            "url extract keeps the legacy left-to-right https?://[^\\s]+ rule with duplicate preservation and no network semantics.",
        ),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = stable_json_digest(b"url_extract_receipt|", &receipt);
    UrlExtractInvocationOutcome {
        receipt,
        response: Some(response),
        refusal: None,
    }
}

#[must_use]
pub fn build_url_extract_runtime_bundle() -> UrlExtractRuntimeBundle {
    let success_packet = br#"{"text":"Read https://alpha.example/a then http://beta.test/b and revisit https://alpha.example/a"}"#;
    let malformed_packet = br#"{"body":"missing text"}"#;
    let too_many_urls_packet = br#"{"text":"https://alpha.example/a https://beta.example/b"}"#;
    let oversized_packet = oversized_url_extract_packet(UrlExtractConfig::default());

    let success =
        invoke_url_extract_json_packet("json", success_packet, &UrlExtractConfig::default());
    let malformed =
        invoke_url_extract_json_packet("json", malformed_packet, &UrlExtractConfig::default());
    let packet_too_large =
        invoke_url_extract_json_packet("json", &oversized_packet, &UrlExtractConfig::default());
    let unsupported_codec =
        invoke_url_extract_json_packet("bytes", success_packet, &UrlExtractConfig::default());
    let runtime_resource_limit = invoke_url_extract_json_packet(
        "json",
        too_many_urls_packet,
        &UrlExtractConfig {
            packet_size_limit_bytes: UrlExtractConfig::default().packet_size_limit_bytes,
            max_urls: 1,
        },
    );

    let case_rows = vec![
        url_extract_case(
            "extract_urls_success",
            UrlExtractRuntimeCaseStatus::ExactSuccess,
            "json",
            sha256_digest(success_packet),
            STARTER_PLUGIN_TEXT_URL_EXTRACT_OUTPUT_SCHEMA_ID,
            success.receipt.output_or_refusal_digest.clone(),
            success.receipt.clone(),
            "the runtime preserves left-to-right output order and duplicate posture for literal http(s) strings.",
        ),
        url_extract_case(
            "schema_invalid_missing_text",
            UrlExtractRuntimeCaseStatus::TypedMalformedPacket,
            "json",
            sha256_digest(malformed_packet),
            STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID,
            malformed.receipt.output_or_refusal_digest.clone(),
            malformed.receipt.clone(),
            "missing `text` fails closed into a typed schema-invalid refusal.",
        ),
        url_extract_case(
            "packet_too_large_refusal",
            UrlExtractRuntimeCaseStatus::TypedRefusal,
            "json",
            sha256_digest(&oversized_packet),
            STARTER_PLUGIN_REFUSAL_PACKET_TOO_LARGE_ID,
            packet_too_large.receipt.output_or_refusal_digest.clone(),
            packet_too_large.receipt.clone(),
            "packet ceilings stay explicit rather than relying on ambient parser behavior.",
        ),
        url_extract_case(
            "unsupported_codec_refusal",
            UrlExtractRuntimeCaseStatus::TypedRefusal,
            "bytes",
            sha256_digest(success_packet),
            STARTER_PLUGIN_REFUSAL_UNSUPPORTED_CODEC_ID,
            unsupported_codec.receipt.output_or_refusal_digest.clone(),
            unsupported_codec.receipt.clone(),
            "unsupported codecs remain typed refusal truth instead of host-side best effort decode.",
        ),
        url_extract_case(
            "runtime_resource_limit_refusal",
            UrlExtractRuntimeCaseStatus::TypedRefusal,
            "json",
            sha256_digest(too_many_urls_packet),
            STARTER_PLUGIN_REFUSAL_RUNTIME_RESOURCE_LIMIT_ID,
            runtime_resource_limit
                .receipt
                .output_or_refusal_digest
                .clone(),
            runtime_resource_limit.receipt.clone(),
            "bounded output ceilings fail closed into one typed runtime-resource-limit refusal.",
        ),
    ];

    let mut bundle = UrlExtractRuntimeBundle {
        schema_version: 1,
        bundle_id: String::from("tassadar.post_article.plugin_text_url_extract.runtime_bundle.v1"),
        plugin_id: String::from(STARTER_PLUGIN_TEXT_URL_EXTRACT_ID),
        plugin_version: String::from(STARTER_PLUGIN_VERSION),
        manifest_id: String::from("manifest.plugin.text.url_extract.v1"),
        artifact_id: String::from("artifact.plugin.text.url_extract.v1"),
        packet_abi_version: String::from(TASSADAR_POST_ARTICLE_PLUGIN_PACKET_ABI_VERSION),
        mount_envelope_id: String::from("mount.plugin.text.url_extract.no_capabilities.v1"),
        tool_projection: url_extract_tool_projection(),
        negative_claim_ids: vec![
            String::from("url_validation_truth_not_claimed"),
            String::from("dns_resolution_not_claimed"),
            String::from("redirect_truth_not_claimed"),
            String::from("network_reachability_not_claimed"),
        ],
        case_rows,
        claim_boundary: String::from(
            "this runtime bundle closes one capability-free starter plugin that extracts literal http:// and https:// substrings from packet-local text under one deterministic regex rule. It does not claim URL validation, DNS truth, redirect truth, or any network semantics.",
        ),
        summary: String::new(),
        bundle_digest: String::new(),
    };
    bundle.summary = format!(
        "url extract runtime bundle covers {} cases across success={}, malformed={}, refusals={}.",
        bundle.case_rows.len(),
        bundle
            .case_rows
            .iter()
            .filter(|row| row.status == UrlExtractRuntimeCaseStatus::ExactSuccess)
            .count(),
        bundle
            .case_rows
            .iter()
            .filter(|row| row.status == UrlExtractRuntimeCaseStatus::TypedMalformedPacket)
            .count(),
        bundle
            .case_rows
            .iter()
            .filter(|row| row.status == UrlExtractRuntimeCaseStatus::TypedRefusal)
            .count(),
    );
    bundle.bundle_digest = stable_json_digest(b"url_extract_runtime_bundle|", &bundle);
    bundle
}

#[must_use]
pub fn tassadar_post_article_plugin_text_url_extract_runtime_bundle_path() -> PathBuf {
    repo_root().join(TASSADAR_POST_ARTICLE_PLUGIN_TEXT_URL_EXTRACT_RUNTIME_BUNDLE_REF)
}

pub fn write_url_extract_runtime_bundle(
    output_path: impl AsRef<Path>,
) -> Result<UrlExtractRuntimeBundle, StarterPluginRuntimeError> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| StarterPluginRuntimeError::CreateDir {
            path: parent.display().to_string(),
            error,
        })?;
    }
    let bundle = build_url_extract_runtime_bundle();
    let json = serde_json::to_string_pretty(&bundle)?;
    fs::write(output_path, format!("{json}\n")).map_err(|error| {
        StarterPluginRuntimeError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(bundle)
}

pub fn load_url_extract_runtime_bundle(
    path: impl AsRef<Path>,
) -> Result<UrlExtractRuntimeBundle, StarterPluginRuntimeError> {
    read_json(path)
}

impl FetchTextConfig {
    #[must_use]
    pub fn snapshot(entries: BTreeMap<String, FetchTextSnapshotResult>) -> Self {
        Self {
            mount_envelope: default_fetch_text_mount_envelope("replayable_with_snapshots"),
            backend: FetchTextBackend::Snapshot(entries),
        }
    }

    #[must_use]
    pub fn live(allowlisted_url_prefixes: Vec<String>) -> Self {
        let mut mount_envelope = default_fetch_text_mount_envelope("operator_replay_only");
        mount_envelope.allowlisted_url_prefixes = allowlisted_url_prefixes;
        Self {
            mount_envelope,
            backend: FetchTextBackend::Live,
        }
    }
}

#[must_use]
pub fn fetch_text_tool_projection() -> StarterPluginToolProjection {
    StarterPluginToolProjection {
        plugin_id: String::from(STARTER_PLUGIN_HTTP_FETCH_TEXT_ID),
        tool_name: String::from(STARTER_PLUGIN_HTTP_FETCH_TEXT_TOOL_NAME),
        description: String::from(
            "fetch allowlisted text content through a host-mediated read-only HTTP mount without browser execution, cookies, auth sessions, or unrestricted network access.",
        ),
        arguments_schema: json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["url"],
            "properties": {
                "url": {
                    "type": "string",
                    "description": "allowlisted absolute URL fetched with one bounded GET-only runtime path."
                }
            }
        }),
        result_schema_id: String::from(STARTER_PLUGIN_HTTP_FETCH_TEXT_OUTPUT_SCHEMA_ID),
        refusal_schema_ids: vec![
            String::from(STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID),
            String::from(STARTER_PLUGIN_REFUSAL_UNSUPPORTED_CODEC_ID),
            String::from(STARTER_PLUGIN_REFUSAL_NETWORK_DENIED_ID),
            String::from(STARTER_PLUGIN_REFUSAL_URL_NOT_PERMITTED_ID),
            String::from(STARTER_PLUGIN_REFUSAL_TIMEOUT_ID),
            String::from(STARTER_PLUGIN_REFUSAL_RESPONSE_TOO_LARGE_ID),
            String::from(STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID),
            String::from(STARTER_PLUGIN_REFUSAL_DECODE_FAILED_ID),
            String::from(STARTER_PLUGIN_REFUSAL_UPSTREAM_FAILURE_ID),
        ],
        replay_class_id: String::from("replayable_with_snapshots"),
    }
}

pub fn invoke_fetch_text_json_packet(
    codec_id: &str,
    packet_bytes: &[u8],
    config: &FetchTextConfig,
) -> FetchTextInvocationOutcome {
    let input_packet_digest = sha256_digest(packet_bytes);
    if codec_id != "json" {
        return fetch_text_refusal_outcome(
            &input_packet_digest,
            backend_id(config),
            replay_class_id(config),
            STARTER_PLUGIN_REFUSAL_UNSUPPORTED_CODEC_ID,
            "unsupported_codec",
            "fetch-text accepts only json packet input under packet.v1.",
        );
    }
    let request = match serde_json::from_slice::<FetchTextRequest>(packet_bytes) {
        Ok(request) => request,
        Err(_) => {
            return fetch_text_refusal_outcome(
                &input_packet_digest,
                backend_id(config),
                replay_class_id(config),
                STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID,
                "schema_invalid",
                "fetch-text refuses malformed packets without host-side schema repair.",
            );
        }
    };
    let url = match Url::parse(&request.url) {
        Ok(url) => url,
        Err(_) => {
            return fetch_text_refusal_outcome(
                &input_packet_digest,
                backend_id(config),
                replay_class_id(config),
                STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID,
                "schema_invalid",
                "fetch-text requires one absolute URL string.",
            );
        }
    };
    if !url_allowed(url.as_str(), &config.mount_envelope) {
        return fetch_text_refusal_outcome(
            &input_packet_digest,
            backend_id(config),
            replay_class_id(config),
            STARTER_PLUGIN_REFUSAL_URL_NOT_PERMITTED_ID,
            "url_not_permitted",
            "fetch-text keeps URL allowlists in the host-owned mount envelope instead of guest packets.",
        );
    }

    let result = match &config.backend {
        FetchTextBackend::Snapshot(entries) => invoke_snapshot_fetch(&url, entries),
        FetchTextBackend::Live => invoke_live_fetch(&url, &config.mount_envelope),
    };
    match result {
        Ok(response) => {
            let output_or_refusal_digest = stable_json_digest(b"fetch_text_response|", &response);
            let mut receipt = StarterPluginInvocationReceipt {
                receipt_id: format!(
                    "receipt.{}.{}.v1",
                    STARTER_PLUGIN_HTTP_FETCH_TEXT_ID,
                    &input_packet_digest[..16]
                ),
                plugin_id: String::from(STARTER_PLUGIN_HTTP_FETCH_TEXT_ID),
                plugin_version: String::from(STARTER_PLUGIN_VERSION),
                tool_name: String::from(STARTER_PLUGIN_HTTP_FETCH_TEXT_TOOL_NAME),
                packet_abi_version: String::from(TASSADAR_POST_ARTICLE_PLUGIN_PACKET_ABI_VERSION),
                mount_envelope_id: config.mount_envelope.envelope_id.clone(),
                capability_namespace_ids: vec![String::from("capability.http.read_only.v1")],
                replay_class_id: String::from(replay_class_id(config)),
                status: StarterPluginInvocationStatus::Success,
                input_schema_id: String::from(STARTER_PLUGIN_HTTP_FETCH_TEXT_INPUT_SCHEMA_ID),
                input_packet_digest,
                output_or_refusal_schema_id: String::from(STARTER_PLUGIN_HTTP_FETCH_TEXT_OUTPUT_SCHEMA_ID),
                output_or_refusal_digest,
                refusal_class_id: None,
                detail: String::from(
                    "fetch-text keeps GET-only host-mediated access explicit and returns structured text-fetch truth without browser semantics.",
                ),
                receipt_digest: String::new(),
            };
            receipt.receipt_digest = stable_json_digest(b"fetch_text_receipt|", &receipt);
            FetchTextInvocationOutcome {
                receipt,
                backend_id: String::from(backend_id(config)),
                logical_cpu_millis: 2,
                response: Some(response),
                refusal: None,
            }
        }
        Err((schema_id, refusal_class_id, detail)) => fetch_text_refusal_outcome(
            &input_packet_digest,
            backend_id(config),
            replay_class_id(config),
            schema_id,
            refusal_class_id,
            detail,
        ),
    }
}

#[must_use]
pub fn build_fetch_text_runtime_bundle() -> FetchTextRuntimeBundle {
    let snapshot_entries = BTreeMap::from([
        (
            String::from("https://snapshot.example/article"),
            FetchTextSnapshotResult::Success(FetchTextSnapshotResponse {
                final_url: String::from("https://snapshot.example/article"),
                status_code: 200,
                content_type: String::from("text/html"),
                charset: Some(String::from("utf-8")),
                body_bytes: b"<html><title>Snapshot Article</title><body><h1>Snapshot Article</h1><p>Bounded starter plugin content.</p></body></html>".to_vec(),
            }),
        ),
        (
            String::from("https://snapshot.example/slow"),
            FetchTextSnapshotResult::Timeout,
        ),
        (
            String::from("https://snapshot.example/binary"),
            FetchTextSnapshotResult::Success(FetchTextSnapshotResponse {
                final_url: String::from("https://snapshot.example/binary"),
                status_code: 200,
                content_type: String::from("image/png"),
                charset: None,
                body_bytes: vec![0x89, 0x50, 0x4e, 0x47],
            }),
        ),
        (
            String::from("https://snapshot.example/large"),
            FetchTextSnapshotResult::Success(FetchTextSnapshotResponse {
                final_url: String::from("https://snapshot.example/large"),
                status_code: 200,
                content_type: String::from("text/plain"),
                charset: Some(String::from("utf-8")),
                body_bytes: vec![b'x'; 16 * 1024 + 8],
            }),
        ),
        (
            String::from("https://snapshot.example/bad-utf8"),
            FetchTextSnapshotResult::Success(FetchTextSnapshotResponse {
                final_url: String::from("https://snapshot.example/bad-utf8"),
                status_code: 200,
                content_type: String::from("text/plain"),
                charset: Some(String::from("utf-8")),
                body_bytes: vec![0xff, 0xfe, 0xfd],
            }),
        ),
        (
            String::from("https://snapshot.example/broken"),
            FetchTextSnapshotResult::UpstreamFailure {
                detail: String::from("the mounted snapshot marks this URL as an upstream transport failure."),
            },
        ),
    ]);
    let snapshot_config = FetchTextConfig::snapshot(snapshot_entries);

    let success_packet = br#"{"url":"https://snapshot.example/article"}"#;
    let malformed_packet = br#"{"uri":"https://snapshot.example/article"}"#;
    let blocked_packet = br#"{"url":"https://blocked.example/article"}"#;
    let timeout_packet = br#"{"url":"https://snapshot.example/slow"}"#;
    let missing_packet = br#"{"url":"https://snapshot.example/missing"}"#;
    let large_packet = br#"{"url":"https://snapshot.example/large"}"#;
    let bad_type_packet = br#"{"url":"https://snapshot.example/binary"}"#;
    let bad_decode_packet = br#"{"url":"https://snapshot.example/bad-utf8"}"#;
    let upstream_failure_packet = br#"{"url":"https://snapshot.example/broken"}"#;

    let success = invoke_fetch_text_json_packet("json", success_packet, &snapshot_config);
    let malformed = invoke_fetch_text_json_packet("json", malformed_packet, &snapshot_config);
    let blocked = invoke_fetch_text_json_packet("json", blocked_packet, &snapshot_config);
    let timeout = invoke_fetch_text_json_packet("json", timeout_packet, &snapshot_config);
    let network_denied = invoke_fetch_text_json_packet("json", missing_packet, &snapshot_config);
    let response_too_large =
        invoke_fetch_text_json_packet("json", large_packet, &snapshot_config);
    let content_type_unsupported =
        invoke_fetch_text_json_packet("json", bad_type_packet, &snapshot_config);
    let decode_failed =
        invoke_fetch_text_json_packet("json", bad_decode_packet, &snapshot_config);
    let upstream_failure =
        invoke_fetch_text_json_packet("json", upstream_failure_packet, &snapshot_config);

    let case_rows = vec![
        fetch_text_case(
            "fetch_text_article_success",
            FetchTextRuntimeCaseStatus::ExactSuccess,
            "json",
            sha256_digest(success_packet),
            STARTER_PLUGIN_HTTP_FETCH_TEXT_OUTPUT_SCHEMA_ID,
            success.receipt.output_or_refusal_digest.clone(),
            success.backend_id.clone(),
            success.logical_cpu_millis,
            success.receipt.clone(),
            "snapshot-backed fetch-text returns structured host-mediated text-fetch truth.",
        ),
        fetch_text_case(
            "schema_invalid_missing_url",
            FetchTextRuntimeCaseStatus::TypedMalformedPacket,
            "json",
            sha256_digest(malformed_packet),
            STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID,
            malformed.receipt.output_or_refusal_digest.clone(),
            malformed.backend_id.clone(),
            malformed.logical_cpu_millis,
            malformed.receipt.clone(),
            "missing `url` fails closed into a typed schema-invalid refusal.",
        ),
        fetch_text_case(
            "url_not_permitted_refusal",
            FetchTextRuntimeCaseStatus::TypedRefusal,
            "json",
            sha256_digest(blocked_packet),
            STARTER_PLUGIN_REFUSAL_URL_NOT_PERMITTED_ID,
            blocked.receipt.output_or_refusal_digest.clone(),
            blocked.backend_id.clone(),
            blocked.logical_cpu_millis,
            blocked.receipt.clone(),
            "URL allowlists stay bound to the mount envelope instead of guest packets.",
        ),
        fetch_text_case(
            "timeout_refusal",
            FetchTextRuntimeCaseStatus::TypedRefusal,
            "json",
            sha256_digest(timeout_packet),
            STARTER_PLUGIN_REFUSAL_TIMEOUT_ID,
            timeout.receipt.output_or_refusal_digest.clone(),
            timeout.backend_id.clone(),
            timeout.logical_cpu_millis,
            timeout.receipt.clone(),
            "timeouts stay machine-readable and do not degrade into generic tool text.",
        ),
        fetch_text_case(
            "network_denied_refusal",
            FetchTextRuntimeCaseStatus::TypedRefusal,
            "json",
            sha256_digest(missing_packet),
            STARTER_PLUGIN_REFUSAL_NETWORK_DENIED_ID,
            network_denied.receipt.output_or_refusal_digest.clone(),
            network_denied.backend_id.clone(),
            network_denied.logical_cpu_millis,
            network_denied.receipt.clone(),
            "missing mounted snapshot entries remain explicit network-denied truth.",
        ),
        fetch_text_case(
            "response_too_large_refusal",
            FetchTextRuntimeCaseStatus::TypedRefusal,
            "json",
            sha256_digest(large_packet),
            STARTER_PLUGIN_REFUSAL_RESPONSE_TOO_LARGE_ID,
            response_too_large.receipt.output_or_refusal_digest.clone(),
            response_too_large.backend_id.clone(),
            response_too_large.logical_cpu_millis,
            response_too_large.receipt.clone(),
            "response-size ceilings remain explicit and fail closed.",
        ),
        fetch_text_case(
            "content_type_unsupported_refusal",
            FetchTextRuntimeCaseStatus::TypedRefusal,
            "json",
            sha256_digest(bad_type_packet),
            STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID,
            content_type_unsupported
                .receipt
                .output_or_refusal_digest
                .clone(),
            content_type_unsupported.backend_id.clone(),
            content_type_unsupported.logical_cpu_millis,
            content_type_unsupported.receipt.clone(),
            "binary content types remain outside the bounded text-fetch window.",
        ),
        fetch_text_case(
            "decode_failed_refusal",
            FetchTextRuntimeCaseStatus::TypedRefusal,
            "json",
            sha256_digest(bad_decode_packet),
            STARTER_PLUGIN_REFUSAL_DECODE_FAILED_ID,
            decode_failed.receipt.output_or_refusal_digest.clone(),
            decode_failed.backend_id.clone(),
            decode_failed.logical_cpu_millis,
            decode_failed.receipt.clone(),
            "decode failures remain explicit instead of silently replacing bytes.",
        ),
        fetch_text_case(
            "upstream_failure_refusal",
            FetchTextRuntimeCaseStatus::TypedRefusal,
            "json",
            sha256_digest(upstream_failure_packet),
            STARTER_PLUGIN_REFUSAL_UPSTREAM_FAILURE_ID,
            upstream_failure.receipt.output_or_refusal_digest.clone(),
            upstream_failure.backend_id.clone(),
            upstream_failure.logical_cpu_millis,
            upstream_failure.receipt.clone(),
            "snapshot-declared upstream failures stay typed and machine-readable.",
        ),
    ];

    let mut bundle = FetchTextRuntimeBundle {
        schema_version: 1,
        bundle_id: String::from("tassadar.post_article.plugin_http_fetch_text.runtime_bundle.v1"),
        plugin_id: String::from(STARTER_PLUGIN_HTTP_FETCH_TEXT_ID),
        plugin_version: String::from(STARTER_PLUGIN_VERSION),
        manifest_id: String::from("manifest.plugin.http.fetch_text.v1"),
        artifact_id: String::from("artifact.plugin.http.fetch_text.v1"),
        packet_abi_version: String::from(TASSADAR_POST_ARTICLE_PLUGIN_PACKET_ABI_VERSION),
        sample_mount_envelope: snapshot_config.mount_envelope.clone(),
        tool_projection: fetch_text_tool_projection(),
        supported_replay_class_ids: vec![
            String::from("replayable_with_snapshots"),
            String::from("operator_replay_only"),
        ],
        negative_claim_ids: vec![
            String::from("browser_execution_not_claimed"),
            String::from("javascript_execution_not_claimed"),
            String::from("cookie_or_auth_session_not_claimed"),
            String::from("arbitrary_headers_not_claimed"),
            String::from("general_unrestricted_web_access_not_claimed"),
        ],
        case_rows,
        claim_boundary: String::from(
            "this runtime bundle closes one read-only network starter plugin that fetches allowlisted text content through a host-mediated HTTP mount. It does not claim browser execution, JavaScript, cookies, auth sessions, arbitrary headers, or unrestricted network access.",
        ),
        summary: String::new(),
        bundle_digest: String::new(),
    };
    bundle.summary = format!(
        "fetch-text runtime bundle covers {} cases across success={}, malformed={}, refusals={} with snapshot-backed replay truth explicit.",
        bundle.case_rows.len(),
        bundle
            .case_rows
            .iter()
            .filter(|row| row.status == FetchTextRuntimeCaseStatus::ExactSuccess)
            .count(),
        bundle
            .case_rows
            .iter()
            .filter(|row| row.status == FetchTextRuntimeCaseStatus::TypedMalformedPacket)
            .count(),
        bundle
            .case_rows
            .iter()
            .filter(|row| row.status == FetchTextRuntimeCaseStatus::TypedRefusal)
            .count(),
    );
    bundle.bundle_digest = stable_json_digest(b"fetch_text_runtime_bundle|", &bundle);
    bundle
}

#[must_use]
pub fn tassadar_post_article_plugin_http_fetch_text_runtime_bundle_path() -> PathBuf {
    repo_root().join(TASSADAR_POST_ARTICLE_PLUGIN_HTTP_FETCH_TEXT_RUNTIME_BUNDLE_REF)
}

pub fn write_fetch_text_runtime_bundle(
    output_path: impl AsRef<Path>,
) -> Result<FetchTextRuntimeBundle, StarterPluginRuntimeError> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| StarterPluginRuntimeError::CreateDir {
            path: parent.display().to_string(),
            error,
        })?;
    }
    let bundle = build_fetch_text_runtime_bundle();
    let json = serde_json::to_string_pretty(&bundle)?;
    fs::write(output_path, format!("{json}\n")).map_err(|error| {
        StarterPluginRuntimeError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(bundle)
}

pub fn load_fetch_text_runtime_bundle(
    path: impl AsRef<Path>,
) -> Result<FetchTextRuntimeBundle, StarterPluginRuntimeError> {
    read_json(path)
}

#[must_use]
pub fn extract_readable_tool_projection() -> StarterPluginToolProjection {
    StarterPluginToolProjection {
        plugin_id: String::from(STARTER_PLUGIN_HTML_EXTRACT_READABLE_ID),
        tool_name: String::from(STARTER_PLUGIN_HTML_EXTRACT_READABLE_TOOL_NAME),
        description: String::from(
            "extract bounded readable text, metadata, and harvested links from already-fetched HTML without browser rendering or network access.",
        ),
        arguments_schema: json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["source_url", "content_type", "body_text"],
            "properties": {
                "source_url": { "type": "string" },
                "content_type": { "type": "string" },
                "body_text": { "type": "string" }
            }
        }),
        result_schema_id: String::from(STARTER_PLUGIN_HTML_EXTRACT_READABLE_OUTPUT_SCHEMA_ID),
        refusal_schema_ids: vec![
            String::from(STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID),
            String::from(STARTER_PLUGIN_REFUSAL_UNSUPPORTED_CODEC_ID),
            String::from(STARTER_PLUGIN_REFUSAL_INPUT_TOO_LARGE_ID),
            String::from(STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID),
        ],
        replay_class_id: String::from("deterministic_replayable"),
    }
}

#[must_use]
pub fn invoke_extract_readable_json_packet(
    codec_id: &str,
    packet_bytes: &[u8],
    config: &ExtractReadableConfig,
) -> ExtractReadableInvocationOutcome {
    let input_packet_digest = sha256_digest(packet_bytes);
    if codec_id != "json" {
        return extract_readable_refusal_outcome(
            &input_packet_digest,
            STARTER_PLUGIN_REFUSAL_UNSUPPORTED_CODEC_ID,
            "unsupported_codec",
            "extract-readable accepts only json packet input under packet.v1.",
        );
    }
    if packet_bytes.len() > config.input_size_limit_bytes {
        return extract_readable_refusal_outcome(
            &input_packet_digest,
            STARTER_PLUGIN_REFUSAL_INPUT_TOO_LARGE_ID,
            "input_too_large",
            "extract-readable keeps input size ceilings explicit instead of relying on parser allocation behavior.",
        );
    }
    let request = match serde_json::from_slice::<ExtractReadableRequest>(packet_bytes) {
        Ok(request) => request,
        Err(_) => {
            return extract_readable_refusal_outcome(
                &input_packet_digest,
                STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID,
                "schema_invalid",
                "extract-readable refuses malformed packets without host-side schema repair.",
            );
        }
    };
    let source_url = match Url::parse(&request.source_url) {
        Ok(url) => url,
        Err(_) => {
            return extract_readable_refusal_outcome(
                &input_packet_digest,
                STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID,
                "schema_invalid",
                "extract-readable requires one absolute `source_url` for canonical-link and link-resolution rules.",
            );
        }
    };
    let (content_type, _) = parse_content_type(&request.content_type);
    if content_type != "text/html" && content_type != "application/xhtml+xml" {
        return extract_readable_refusal_outcome(
            &input_packet_digest,
            STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID,
            "content_type_unsupported",
            "extract-readable stays bounded to html and xhtml input content types.",
        );
    }

    let document = scraper::Html::parse_document(&request.body_text);
    let readable_text = readable_text_from_document(&document);
    let title = first_text_for_selector(&document, "title")
        .or_else(|| meta_content(&document, "property", "og:title"));
    let canonical_url = first_attr_for_selector(&document, r#"link[rel="canonical"]"#, "href")
        .map(|href| resolve_link(&source_url, href.as_str()));
    let site_name = meta_content(&document, "property", "og:site_name")
        .or_else(|| source_url.host_str().map(String::from));
    let excerpt = meta_content(&document, "name", "description")
        .or_else(|| excerpt_from_text(&readable_text));
    let content_language = first_attr_for_selector(&document, "html", "lang")
        .or_else(|| meta_content(&document, "http-equiv", "content-language"));
    let harvested_links = harvested_links_from_document(&document, &source_url, config.max_links);
    let response = ExtractReadableResponse {
        title,
        canonical_url,
        site_name,
        excerpt,
        readable_text,
        harvested_links,
        content_language,
    };
    let output_or_refusal_digest = stable_json_digest(b"extract_readable_response|", &response);
    let mut receipt = StarterPluginInvocationReceipt {
        receipt_id: format!(
            "receipt.{}.{}.v1",
            STARTER_PLUGIN_HTML_EXTRACT_READABLE_ID,
            &input_packet_digest[..16]
        ),
        plugin_id: String::from(STARTER_PLUGIN_HTML_EXTRACT_READABLE_ID),
        plugin_version: String::from(STARTER_PLUGIN_VERSION),
        tool_name: String::from(STARTER_PLUGIN_HTML_EXTRACT_READABLE_TOOL_NAME),
        packet_abi_version: String::from(TASSADAR_POST_ARTICLE_PLUGIN_PACKET_ABI_VERSION),
        mount_envelope_id: String::from("mount.plugin.html.extract_readable.no_capabilities.v1"),
        capability_namespace_ids: Vec::new(),
        replay_class_id: String::from("deterministic_replayable"),
        status: StarterPluginInvocationStatus::Success,
        input_schema_id: String::from(STARTER_PLUGIN_HTML_EXTRACT_READABLE_INPUT_SCHEMA_ID),
        input_packet_digest,
        output_or_refusal_schema_id: String::from(
            STARTER_PLUGIN_HTML_EXTRACT_READABLE_OUTPUT_SCHEMA_ID,
        ),
        output_or_refusal_digest,
        refusal_class_id: None,
        detail: String::from(
            "extract-readable keeps bounded readability rules explicit and local instead of hiding them in host glue or browser state.",
        ),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = stable_json_digest(b"extract_readable_receipt|", &receipt);
    ExtractReadableInvocationOutcome {
        receipt,
        response: Some(response),
        refusal: None,
    }
}

#[must_use]
pub fn build_extract_readable_runtime_bundle() -> ExtractReadableRuntimeBundle {
    let article_html = "<html lang=\"en\"><head><title>Snapshot Article</title><meta name=\"description\" content=\"Bounded HTML extraction fixture.\" /><meta property=\"og:site_name\" content=\"Snapshot Example\" /><link rel=\"canonical\" href=\"/article\" /></head><body><main><h1>Snapshot Article</h1><p>Bounded starter plugin content.</p><a href=\"/alpha\">Alpha</a><a href=\"https://snapshot.example/beta\">Beta</a></main></body></html>";
    let malformed_html =
        "<html><head><title>Broken but readable<title></head><body><article><p>Recovered text<a href=\"/link\">Link";
    let success_packet = serde_json::to_vec(&json!({
        "source_url": "https://snapshot.example/article",
        "content_type": "text/html; charset=utf-8",
        "body_text": article_html
    }))
    .unwrap_or_else(|_| Vec::new());
    let malformed_packet = serde_json::to_vec(&json!({
        "source_url": "https://snapshot.example/article",
        "content_type": "text/html",
        "body_text": malformed_html
    }))
    .unwrap_or_else(|_| Vec::new());
    let missing_packet = br#"{"source_url":"https://snapshot.example/article","content_type":"text/html"}"#.to_vec();
    let wrong_type_packet = serde_json::to_vec(&json!({
        "source_url": "https://snapshot.example/article",
        "content_type": "application/json",
        "body_text": "{}"
    }))
    .unwrap_or_else(|_| Vec::new());
    let oversized_packet = serde_json::to_vec(&json!({
        "source_url": "https://snapshot.example/article",
        "content_type": "text/html",
        "body_text": "x".repeat(64 * 1024)
    }))
    .unwrap_or_else(|_| Vec::new());

    let success =
        invoke_extract_readable_json_packet("json", &success_packet, &ExtractReadableConfig::default());
    let malformed = invoke_extract_readable_json_packet(
        "json",
        &malformed_packet,
        &ExtractReadableConfig::default(),
    );
    let schema_invalid = invoke_extract_readable_json_packet(
        "json",
        &missing_packet,
        &ExtractReadableConfig::default(),
    );
    let wrong_type = invoke_extract_readable_json_packet(
        "json",
        &wrong_type_packet,
        &ExtractReadableConfig::default(),
    );
    let oversized = invoke_extract_readable_json_packet(
        "json",
        &oversized_packet,
        &ExtractReadableConfig {
            input_size_limit_bytes: 1024,
            max_links: 64,
        },
    );

    let fetch_packet = br#"{"url":"https://snapshot.example/article"}"#;
    let fetch_result = invoke_fetch_text_json_packet(
        "json",
        fetch_packet,
        &FetchTextConfig::snapshot(sample_fetch_text_snapshot_entries()),
    );

    let case_rows = vec![
        extract_readable_case(
            "extract_readable_success",
            ExtractReadableRuntimeCaseStatus::ExactSuccess,
            "json",
            sha256_digest(&success_packet),
            STARTER_PLUGIN_HTML_EXTRACT_READABLE_OUTPUT_SCHEMA_ID,
            success.receipt.output_or_refusal_digest.clone(),
            success.receipt.clone(),
            "deterministic readability extraction returns metadata, readable text, and harvested links from fetched HTML.",
        ),
        extract_readable_case(
            "extract_readable_malformed_but_recoverable_success",
            ExtractReadableRuntimeCaseStatus::ExactSuccess,
            "json",
            sha256_digest(&malformed_packet),
            STARTER_PLUGIN_HTML_EXTRACT_READABLE_OUTPUT_SCHEMA_ID,
            malformed.receipt.output_or_refusal_digest.clone(),
            malformed.receipt.clone(),
            "malformed-but-recoverable HTML still resolves under the bounded parser rules without browser semantics.",
        ),
        extract_readable_case(
            "schema_invalid_missing_body_text",
            ExtractReadableRuntimeCaseStatus::TypedMalformedPacket,
            "json",
            sha256_digest(&missing_packet),
            STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID,
            schema_invalid.receipt.output_or_refusal_digest.clone(),
            schema_invalid.receipt.clone(),
            "missing `body_text` fails closed into a typed schema-invalid refusal.",
        ),
        extract_readable_case(
            "content_type_unsupported_refusal",
            ExtractReadableRuntimeCaseStatus::TypedRefusal,
            "json",
            sha256_digest(&wrong_type_packet),
            STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID,
            wrong_type.receipt.output_or_refusal_digest.clone(),
            wrong_type.receipt.clone(),
            "non-html content types remain outside the bounded readability window.",
        ),
        extract_readable_case(
            "input_too_large_refusal",
            ExtractReadableRuntimeCaseStatus::TypedRefusal,
            "json",
            sha256_digest(&oversized_packet),
            STARTER_PLUGIN_REFUSAL_INPUT_TOO_LARGE_ID,
            oversized.receipt.output_or_refusal_digest.clone(),
            oversized.receipt.clone(),
            "input ceilings remain explicit instead of letting parser recovery define semantics.",
        ),
    ];
    let composition_case = ExtractReadableCompositionCase {
        case_id: String::from("fetch_then_extract_readable"),
        step_plugin_ids: vec![
            String::from(STARTER_PLUGIN_HTTP_FETCH_TEXT_ID),
            String::from(STARTER_PLUGIN_HTML_EXTRACT_READABLE_ID),
        ],
        step_receipt_ids: vec![
            fetch_result.receipt.receipt_id.clone(),
            success.receipt.receipt_id.clone(),
        ],
        schema_repair_allowed: false,
        hidden_host_extraction_allowed: false,
        green: fetch_result.response.is_some() && success.response.is_some(),
        detail: String::from(
            "the fetch-text response binds directly into extract-readable input fields without hidden host schema repair or host-side readability glue.",
        ),
    };

    let mut bundle = ExtractReadableRuntimeBundle {
        schema_version: 1,
        bundle_id: String::from(
            "tassadar.post_article.plugin_html_extract_readable.runtime_bundle.v1",
        ),
        plugin_id: String::from(STARTER_PLUGIN_HTML_EXTRACT_READABLE_ID),
        plugin_version: String::from(STARTER_PLUGIN_VERSION),
        manifest_id: String::from("manifest.plugin.html.extract_readable.v1"),
        artifact_id: String::from("artifact.plugin.html.extract_readable.v1"),
        packet_abi_version: String::from(TASSADAR_POST_ARTICLE_PLUGIN_PACKET_ABI_VERSION),
        mount_envelope_id: String::from("mount.plugin.html.extract_readable.no_capabilities.v1"),
        tool_projection: extract_readable_tool_projection(),
        negative_claim_ids: vec![
            String::from("browser_rendering_not_claimed"),
            String::from("javascript_evaluation_not_claimed"),
            String::from("css_layout_truth_not_claimed"),
            String::from("full_dom_semantics_not_claimed"),
        ],
        case_rows,
        composition_case,
        claim_boundary: String::from(
            "this runtime bundle closes one local deterministic readability extractor over already-fetched HTML. It does not claim browser rendering, JavaScript evaluation, CSS layout truth, or full DOM semantics.",
        ),
        summary: String::new(),
        bundle_digest: String::new(),
    };
    bundle.summary = format!(
        "extract-readable runtime bundle covers {} cases plus one green fetch-to-extract composition harness.",
        bundle.case_rows.len(),
    );
    bundle.bundle_digest = stable_json_digest(b"extract_readable_runtime_bundle|", &bundle);
    bundle
}

#[must_use]
pub fn tassadar_post_article_plugin_html_extract_readable_runtime_bundle_path() -> PathBuf {
    repo_root().join(TASSADAR_POST_ARTICLE_PLUGIN_HTML_EXTRACT_READABLE_RUNTIME_BUNDLE_REF)
}

pub fn write_extract_readable_runtime_bundle(
    output_path: impl AsRef<Path>,
) -> Result<ExtractReadableRuntimeBundle, StarterPluginRuntimeError> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| StarterPluginRuntimeError::CreateDir {
            path: parent.display().to_string(),
            error,
        })?;
    }
    let bundle = build_extract_readable_runtime_bundle();
    let json = serde_json::to_string_pretty(&bundle)?;
    fs::write(output_path, format!("{json}\n")).map_err(|error| {
        StarterPluginRuntimeError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(bundle)
}

pub fn load_extract_readable_runtime_bundle(
    path: impl AsRef<Path>,
) -> Result<ExtractReadableRuntimeBundle, StarterPluginRuntimeError> {
    read_json(path)
}

fn url_extract_refusal_outcome(
    input_packet_digest: &str,
    schema_id: &str,
    refusal_class_id: &str,
    detail: impl Into<String>,
) -> UrlExtractInvocationOutcome {
    let refusal = StarterPluginRefusal {
        schema_id: String::from(schema_id),
        refusal_class_id: String::from(refusal_class_id),
        detail: detail.into(),
    };
    let output_or_refusal_digest = stable_json_digest(b"url_extract_refusal|", &refusal);
    let mut receipt = StarterPluginInvocationReceipt {
        receipt_id: format!(
            "receipt.{}.{}.v1",
            STARTER_PLUGIN_TEXT_URL_EXTRACT_ID,
            &input_packet_digest[..16]
        ),
        plugin_id: String::from(STARTER_PLUGIN_TEXT_URL_EXTRACT_ID),
        plugin_version: String::from(STARTER_PLUGIN_VERSION),
        tool_name: String::from(STARTER_PLUGIN_TEXT_URL_EXTRACT_TOOL_NAME),
        packet_abi_version: String::from(TASSADAR_POST_ARTICLE_PLUGIN_PACKET_ABI_VERSION),
        mount_envelope_id: String::from("mount.plugin.text.url_extract.no_capabilities.v1"),
        capability_namespace_ids: Vec::new(),
        replay_class_id: String::from("deterministic_replayable"),
        status: StarterPluginInvocationStatus::Refusal,
        input_schema_id: String::from(STARTER_PLUGIN_TEXT_URL_EXTRACT_INPUT_SCHEMA_ID),
        input_packet_digest: String::from(input_packet_digest),
        output_or_refusal_schema_id: String::from(schema_id),
        output_or_refusal_digest,
        refusal_class_id: Some(String::from(refusal_class_id)),
        detail: refusal.detail.clone(),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = stable_json_digest(b"url_extract_receipt|", &receipt);
    UrlExtractInvocationOutcome {
        receipt,
        response: None,
        refusal: Some(refusal),
    }
}

fn extract_urls(text: &str, max_urls: usize) -> Result<Vec<String>, String> {
    let regex = regex::Regex::new(r"https?://[^\s]+").map_err(|error| error.to_string())?;
    let mut urls = Vec::new();
    for matched in regex.find_iter(text) {
        if urls.len() == max_urls {
            return Err(String::from(
                "url extract exceeded the configured output ceiling before completing the left-to-right scan.",
            ));
        }
        urls.push(matched.as_str().to_string());
    }
    Ok(urls)
}

fn url_extract_case(
    case_id: &str,
    status: UrlExtractRuntimeCaseStatus,
    codec_id: &str,
    request_packet_digest: String,
    response_or_refusal_schema_id: &str,
    response_or_refusal_digest: String,
    receipt: StarterPluginInvocationReceipt,
    detail: &str,
) -> UrlExtractRuntimeCase {
    UrlExtractRuntimeCase {
        case_id: String::from(case_id),
        status,
        codec_id: String::from(codec_id),
        request_packet_digest,
        response_or_refusal_schema_id: String::from(response_or_refusal_schema_id),
        response_or_refusal_digest,
        receipt,
        detail: String::from(detail),
    }
}

fn oversized_url_extract_packet(config: UrlExtractConfig) -> Vec<u8> {
    let oversized = "x".repeat(config.packet_size_limit_bytes.saturating_add(1));
    serde_json::to_vec(&json!({ "text": oversized }))
        .unwrap_or_else(|error| format!("{{\"error\":\"{error}\"}}").into_bytes())
}

fn default_fetch_text_mount_envelope(replay_class_id: &str) -> FetchTextMountEnvelope {
    FetchTextMountEnvelope {
        envelope_id: String::from("mount.plugin.http.fetch_text.read_only_http_allowlist.v1"),
        allowlisted_url_prefixes: vec![String::from("https://snapshot.example/")],
        timeout_millis: 500,
        response_size_limit_bytes: 16 * 1024,
        redirect_limit: 3,
        allowed_content_type_ids: vec![
            String::from("text/"),
            String::from("application/xhtml+xml"),
            String::from("application/xml"),
            String::from("text/xml"),
            String::from("application/rss+xml"),
            String::from("application/atom+xml"),
        ],
        replay_class_id: String::from(replay_class_id),
        detail: String::from(
            "the sample fetch-text mount envelope binds allowlist, timeout, redirect, response-size, and content-type policy to one host-mediated read-only HTTP capability.",
        ),
    }
}

fn backend_id(config: &FetchTextConfig) -> &'static str {
    match &config.backend {
        FetchTextBackend::Snapshot(_) => "snapshot_http_mount",
        FetchTextBackend::Live => "host_http_client",
    }
}

fn replay_class_id(config: &FetchTextConfig) -> &'static str {
    match &config.backend {
        FetchTextBackend::Snapshot(_) => "replayable_with_snapshots",
        FetchTextBackend::Live => "operator_replay_only",
    }
}

fn url_allowed(url: &str, mount_envelope: &FetchTextMountEnvelope) -> bool {
    mount_envelope
        .allowlisted_url_prefixes
        .iter()
        .any(|prefix| url.starts_with(prefix))
}

fn invoke_snapshot_fetch(
    url: &Url,
    entries: &BTreeMap<String, FetchTextSnapshotResult>,
) -> Result<FetchTextResponse, (&'static str, &'static str, String)> {
    let Some(result) = entries.get(url.as_str()) else {
        return Err((
            STARTER_PLUGIN_REFUSAL_NETWORK_DENIED_ID,
            "network_denied",
            String::from("the mounted snapshot did not admit this allowlisted URL."),
        ));
    };
    match result {
        FetchTextSnapshotResult::Success(response) => materialize_fetch_text_response(response),
        FetchTextSnapshotResult::Timeout => Err((
            STARTER_PLUGIN_REFUSAL_TIMEOUT_ID,
            "timeout",
            String::from("the mounted snapshot records this fetch as a timeout."),
        )),
        FetchTextSnapshotResult::NetworkDenied { detail } => Err((
            STARTER_PLUGIN_REFUSAL_NETWORK_DENIED_ID,
            "network_denied",
            detail.clone(),
        )),
        FetchTextSnapshotResult::UpstreamFailure { detail } => Err((
            STARTER_PLUGIN_REFUSAL_UPSTREAM_FAILURE_ID,
            "upstream_failure",
            detail.clone(),
        )),
    }
}

fn invoke_live_fetch(
    url: &Url,
    mount_envelope: &FetchTextMountEnvelope,
) -> Result<FetchTextResponse, (&'static str, &'static str, String)> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(mount_envelope.timeout_millis))
        .redirect(reqwest::redirect::Policy::limited(
            mount_envelope.redirect_limit,
        ))
        .build()
        .map_err(|error| {
            (
                STARTER_PLUGIN_REFUSAL_NETWORK_DENIED_ID,
                "network_denied",
                format!("failed to build host HTTP client: {error}"),
            )
        })?;
    let response = client.get(url.clone()).send().map_err(|error| {
        if error.is_timeout() {
            (
                STARTER_PLUGIN_REFUSAL_TIMEOUT_ID,
                "timeout",
                String::from("the host HTTP client hit the mounted timeout ceiling."),
            )
        } else {
            (
                STARTER_PLUGIN_REFUSAL_UPSTREAM_FAILURE_ID,
                "upstream_failure",
                format!("host HTTP fetch failed: {error}"),
            )
        }
    })?;
    let header_value = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/octet-stream");
    let (content_type, charset) = parse_content_type(header_value);
    if !content_type_supported(&content_type, mount_envelope) {
        return Err((
            STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID,
            "content_type_unsupported",
            format!("content type `{content_type}` is outside the bounded text-fetch window."),
        ));
    }
    if !response.status().is_success() {
        return Err((
            STARTER_PLUGIN_REFUSAL_UPSTREAM_FAILURE_ID,
            "upstream_failure",
            format!("upstream returned HTTP status {}.", response.status().as_u16()),
        ));
    }
    let final_url = response.url().to_string();
    let status_code = response.status().as_u16();
    let mut limited = response
        .take((mount_envelope.response_size_limit_bytes + 1) as u64);
    let mut body_bytes = Vec::new();
    limited.read_to_end(&mut body_bytes).map_err(|error| {
        (
            STARTER_PLUGIN_REFUSAL_UPSTREAM_FAILURE_ID,
            "upstream_failure",
            format!("failed to read upstream response: {error}"),
        )
    })?;
    if body_bytes.len() > mount_envelope.response_size_limit_bytes {
        return Err((
            STARTER_PLUGIN_REFUSAL_RESPONSE_TOO_LARGE_ID,
            "response_too_large",
            String::from("the response exceeded the mounted response-size ceiling."),
        ));
    }
    let body_text = decode_text_body(&body_bytes, charset.as_deref()).map_err(|detail| {
        (
            STARTER_PLUGIN_REFUSAL_DECODE_FAILED_ID,
            "decode_failed",
            detail,
        )
    })?;
    Ok(FetchTextResponse {
        final_url,
        status_code,
        content_type,
        charset,
        body_text,
        truncated: false,
    })
}

fn materialize_fetch_text_response(
    response: &FetchTextSnapshotResponse,
) -> Result<FetchTextResponse, (&'static str, &'static str, String)> {
    let mount_envelope = default_fetch_text_mount_envelope("replayable_with_snapshots");
    if !content_type_supported(&response.content_type, &mount_envelope) {
        return Err((
            STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID,
            "content_type_unsupported",
            format!(
                "content type `{}` is outside the bounded text-fetch window.",
                response.content_type
            ),
        ));
    }
    if response.body_bytes.len() > mount_envelope.response_size_limit_bytes {
        return Err((
            STARTER_PLUGIN_REFUSAL_RESPONSE_TOO_LARGE_ID,
            "response_too_large",
            String::from("the response exceeded the mounted response-size ceiling."),
        ));
    }
    let body_text =
        decode_text_body(&response.body_bytes, response.charset.as_deref()).map_err(|detail| {
            (
                STARTER_PLUGIN_REFUSAL_DECODE_FAILED_ID,
                "decode_failed",
                detail,
            )
        })?;
    Ok(FetchTextResponse {
        final_url: response.final_url.clone(),
        status_code: response.status_code,
        content_type: response.content_type.clone(),
        charset: response.charset.clone(),
        body_text,
        truncated: false,
    })
}

fn fetch_text_refusal_outcome(
    input_packet_digest: &str,
    backend_id: &str,
    replay_class_id: &str,
    schema_id: &str,
    refusal_class_id: &str,
    detail: impl Into<String>,
) -> FetchTextInvocationOutcome {
    let refusal = StarterPluginRefusal {
        schema_id: String::from(schema_id),
        refusal_class_id: String::from(refusal_class_id),
        detail: detail.into(),
    };
    let output_or_refusal_digest = stable_json_digest(b"fetch_text_refusal|", &refusal);
    let mut receipt = StarterPluginInvocationReceipt {
        receipt_id: format!(
            "receipt.{}.{}.v1",
            STARTER_PLUGIN_HTTP_FETCH_TEXT_ID,
            &input_packet_digest[..16]
        ),
        plugin_id: String::from(STARTER_PLUGIN_HTTP_FETCH_TEXT_ID),
        plugin_version: String::from(STARTER_PLUGIN_VERSION),
        tool_name: String::from(STARTER_PLUGIN_HTTP_FETCH_TEXT_TOOL_NAME),
        packet_abi_version: String::from(TASSADAR_POST_ARTICLE_PLUGIN_PACKET_ABI_VERSION),
        mount_envelope_id: String::from("mount.plugin.http.fetch_text.read_only_http_allowlist.v1"),
        capability_namespace_ids: vec![String::from("capability.http.read_only.v1")],
        replay_class_id: String::from(replay_class_id),
        status: StarterPluginInvocationStatus::Refusal,
        input_schema_id: String::from(STARTER_PLUGIN_HTTP_FETCH_TEXT_INPUT_SCHEMA_ID),
        input_packet_digest: String::from(input_packet_digest),
        output_or_refusal_schema_id: String::from(schema_id),
        output_or_refusal_digest,
        refusal_class_id: Some(String::from(refusal_class_id)),
        detail: refusal.detail.clone(),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = stable_json_digest(b"fetch_text_receipt|", &receipt);
    FetchTextInvocationOutcome {
        receipt,
        backend_id: String::from(backend_id),
        logical_cpu_millis: 1,
        response: None,
        refusal: Some(refusal),
    }
}

fn fetch_text_case(
    case_id: &str,
    status: FetchTextRuntimeCaseStatus,
    codec_id: &str,
    request_packet_digest: String,
    response_or_refusal_schema_id: &str,
    response_or_refusal_digest: String,
    backend_id: String,
    logical_cpu_millis: u32,
    receipt: StarterPluginInvocationReceipt,
    detail: &str,
) -> FetchTextRuntimeCase {
    FetchTextRuntimeCase {
        case_id: String::from(case_id),
        status,
        codec_id: String::from(codec_id),
        request_packet_digest,
        response_or_refusal_schema_id: String::from(response_or_refusal_schema_id),
        response_or_refusal_digest,
        backend_id,
        logical_cpu_millis,
        receipt,
        detail: String::from(detail),
    }
}

fn parse_content_type(header_value: &str) -> (String, Option<String>) {
    let mut parts = header_value.split(';');
    let content_type = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("application/octet-stream")
        .to_ascii_lowercase();
    let charset = parts.find_map(|part| {
        let part = part.trim();
        part.strip_prefix("charset=")
            .map(|value| value.trim_matches('"').to_ascii_lowercase())
    });
    (content_type, charset)
}

fn content_type_supported(content_type: &str, mount_envelope: &FetchTextMountEnvelope) -> bool {
    mount_envelope.allowed_content_type_ids.iter().any(|allowed| {
        if allowed.ends_with('/') {
            content_type.starts_with(allowed)
        } else {
            content_type == allowed
        }
    })
}

fn decode_text_body(bytes: &[u8], charset: Option<&str>) -> Result<String, String> {
    match charset.unwrap_or("utf-8") {
        "utf-8" | "utf8" => String::from_utf8(bytes.to_vec())
            .map_err(|_| String::from("the response body failed utf-8 decoding under the declared charset.")),
        "us-ascii" | "ascii" => {
            if bytes.iter().all(|byte| byte.is_ascii()) {
                String::from_utf8(bytes.to_vec()).map_err(|_| {
                    String::from("the response body failed ascii decoding under the declared charset.")
                })
            } else {
                Err(String::from(
                    "the response body contained non-ascii bytes under an ascii charset declaration.",
                ))
            }
        }
        unsupported => Err(format!(
            "the declared charset `{unsupported}` is outside the bounded fetch-text decoder window."
        )),
    }
}

fn sample_fetch_text_snapshot_entries() -> BTreeMap<String, FetchTextSnapshotResult> {
    BTreeMap::from([
        (
            String::from("https://snapshot.example/article"),
            FetchTextSnapshotResult::Success(FetchTextSnapshotResponse {
                final_url: String::from("https://snapshot.example/article"),
                status_code: 200,
                content_type: String::from("text/html"),
                charset: Some(String::from("utf-8")),
                body_bytes: b"<html lang=\"en\"><head><title>Snapshot Article</title><meta name=\"description\" content=\"Bounded HTML extraction fixture.\" /><meta property=\"og:site_name\" content=\"Snapshot Example\" /><link rel=\"canonical\" href=\"/article\" /></head><body><main><h1>Snapshot Article</h1><p>Bounded starter plugin content.</p><a href=\"/alpha\">Alpha</a><a href=\"https://snapshot.example/beta\">Beta</a></main></body></html>".to_vec(),
            }),
        ),
        (
            String::from("https://snapshot.example/slow"),
            FetchTextSnapshotResult::Timeout,
        ),
        (
            String::from("https://snapshot.example/binary"),
            FetchTextSnapshotResult::Success(FetchTextSnapshotResponse {
                final_url: String::from("https://snapshot.example/binary"),
                status_code: 200,
                content_type: String::from("image/png"),
                charset: None,
                body_bytes: vec![0x89, 0x50, 0x4e, 0x47],
            }),
        ),
        (
            String::from("https://snapshot.example/large"),
            FetchTextSnapshotResult::Success(FetchTextSnapshotResponse {
                final_url: String::from("https://snapshot.example/large"),
                status_code: 200,
                content_type: String::from("text/plain"),
                charset: Some(String::from("utf-8")),
                body_bytes: vec![b'x'; 16 * 1024 + 8],
            }),
        ),
        (
            String::from("https://snapshot.example/bad-utf8"),
            FetchTextSnapshotResult::Success(FetchTextSnapshotResponse {
                final_url: String::from("https://snapshot.example/bad-utf8"),
                status_code: 200,
                content_type: String::from("text/plain"),
                charset: Some(String::from("utf-8")),
                body_bytes: vec![0xff, 0xfe, 0xfd],
            }),
        ),
        (
            String::from("https://snapshot.example/broken"),
            FetchTextSnapshotResult::UpstreamFailure {
                detail: String::from(
                    "the mounted snapshot marks this URL as an upstream transport failure.",
                ),
            },
        ),
    ])
}

fn extract_readable_refusal_outcome(
    input_packet_digest: &str,
    schema_id: &str,
    refusal_class_id: &str,
    detail: impl Into<String>,
) -> ExtractReadableInvocationOutcome {
    let refusal = StarterPluginRefusal {
        schema_id: String::from(schema_id),
        refusal_class_id: String::from(refusal_class_id),
        detail: detail.into(),
    };
    let output_or_refusal_digest = stable_json_digest(b"extract_readable_refusal|", &refusal);
    let mut receipt = StarterPluginInvocationReceipt {
        receipt_id: format!(
            "receipt.{}.{}.v1",
            STARTER_PLUGIN_HTML_EXTRACT_READABLE_ID,
            &input_packet_digest[..16]
        ),
        plugin_id: String::from(STARTER_PLUGIN_HTML_EXTRACT_READABLE_ID),
        plugin_version: String::from(STARTER_PLUGIN_VERSION),
        tool_name: String::from(STARTER_PLUGIN_HTML_EXTRACT_READABLE_TOOL_NAME),
        packet_abi_version: String::from(TASSADAR_POST_ARTICLE_PLUGIN_PACKET_ABI_VERSION),
        mount_envelope_id: String::from("mount.plugin.html.extract_readable.no_capabilities.v1"),
        capability_namespace_ids: Vec::new(),
        replay_class_id: String::from("deterministic_replayable"),
        status: StarterPluginInvocationStatus::Refusal,
        input_schema_id: String::from(STARTER_PLUGIN_HTML_EXTRACT_READABLE_INPUT_SCHEMA_ID),
        input_packet_digest: String::from(input_packet_digest),
        output_or_refusal_schema_id: String::from(schema_id),
        output_or_refusal_digest,
        refusal_class_id: Some(String::from(refusal_class_id)),
        detail: refusal.detail.clone(),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = stable_json_digest(b"extract_readable_receipt|", &receipt);
    ExtractReadableInvocationOutcome {
        receipt,
        response: None,
        refusal: Some(refusal),
    }
}

fn extract_readable_case(
    case_id: &str,
    status: ExtractReadableRuntimeCaseStatus,
    codec_id: &str,
    request_packet_digest: String,
    response_or_refusal_schema_id: &str,
    response_or_refusal_digest: String,
    receipt: StarterPluginInvocationReceipt,
    detail: &str,
) -> ExtractReadableRuntimeCase {
    ExtractReadableRuntimeCase {
        case_id: String::from(case_id),
        status,
        codec_id: String::from(codec_id),
        request_packet_digest,
        response_or_refusal_schema_id: String::from(response_or_refusal_schema_id),
        response_or_refusal_digest,
        receipt,
        detail: String::from(detail),
    }
}

fn readable_text_from_document(document: &scraper::Html) -> String {
    let candidate = text_for_selector(document, "article")
        .filter(|text| !text.is_empty())
        .or_else(|| text_for_selector(document, "main").filter(|text| !text.is_empty()))
        .or_else(|| text_for_selector(document, "body").filter(|text| !text.is_empty()))
        .unwrap_or_default();
    normalize_text(&candidate)
}

fn harvested_links_from_document(
    document: &scraper::Html,
    base_url: &Url,
    max_links: usize,
) -> Vec<String> {
    let Some(selector) = selector("a") else {
        return Vec::new();
    };
    document
        .select(&selector)
        .filter_map(|node| node.value().attr("href"))
        .take(max_links)
        .map(|href| resolve_link(base_url, href))
        .collect()
}

fn excerpt_from_text(text: &str) -> Option<String> {
    let excerpt = text.chars().take(240).collect::<String>();
    (!excerpt.is_empty()).then_some(excerpt)
}

fn resolve_link(base_url: &Url, href: &str) -> String {
    match base_url.join(href) {
        Ok(url) => url.to_string(),
        Err(_) => String::from(href),
    }
}

fn meta_content(document: &scraper::Html, attr_name: &str, attr_value: &str) -> Option<String> {
    let selector = selector(&format!(r#"meta[{attr_name}="{attr_value}"]"#))?;
    document
        .select(&selector)
        .find_map(|node| node.value().attr("content").map(normalize_text))
        .filter(|value| !value.is_empty())
}

fn first_attr_for_selector(
    document: &scraper::Html,
    selector_text: &str,
    attr_name: &str,
) -> Option<String> {
    let selector = selector(selector_text)?;
    document
        .select(&selector)
        .find_map(|node| node.value().attr(attr_name).map(String::from))
}

fn first_text_for_selector(document: &scraper::Html, selector_text: &str) -> Option<String> {
    text_for_selector(document, selector_text).filter(|value| !value.is_empty())
}

fn text_for_selector(document: &scraper::Html, selector_text: &str) -> Option<String> {
    let selector = selector(selector_text)?;
    document
        .select(&selector)
        .next()
        .map(|node| normalize_text(&node.text().collect::<Vec<_>>().join(" ")))
}

fn selector(selector_text: &str) -> Option<scraper::Selector> {
    scraper::Selector::parse(selector_text).ok()
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn stable_json_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let encoded = serde_json::to_vec(value)
        .unwrap_or_else(|error| format!("serialization_error:{error}").into_bytes());
    stable_digest(prefix, &encoded)
}

fn sha256_digest(bytes: &[u8]) -> String {
    stable_digest(b"sha256|", bytes)
}

fn stable_digest(prefix: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, StarterPluginRuntimeError> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|error| StarterPluginRuntimeError::Read {
        path: path.display().to_string(),
        error,
    })?;
    serde_json::from_slice(&bytes).map_err(|error| StarterPluginRuntimeError::Decode {
        path: path.display().to_string(),
        error,
    })
}

#[cfg(not(test))]
fn read_json<T: for<'de> Deserialize<'de>>(
    path: impl AsRef<Path>,
) -> Result<T, StarterPluginRuntimeError> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|error| StarterPluginRuntimeError::Read {
        path: path.display().to_string(),
        error,
    })?;
    serde_json::from_slice(&bytes).map_err(|error| StarterPluginRuntimeError::Decode {
        path: path.display().to_string(),
        error,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ExtractReadableConfig, ExtractReadableRuntimeCaseStatus, FetchTextConfig,
        FetchTextRuntimeCaseStatus, FetchTextSnapshotResponse, FetchTextSnapshotResult,
        STARTER_PLUGIN_HTML_EXTRACT_READABLE_OUTPUT_SCHEMA_ID,
        STARTER_PLUGIN_HTTP_FETCH_TEXT_OUTPUT_SCHEMA_ID,
        STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID,
        STARTER_PLUGIN_REFUSAL_DECODE_FAILED_ID, STARTER_PLUGIN_REFUSAL_NETWORK_DENIED_ID,
        STARTER_PLUGIN_REFUSAL_INPUT_TOO_LARGE_ID,
        STARTER_PLUGIN_REFUSAL_PACKET_TOO_LARGE_ID,
        STARTER_PLUGIN_REFUSAL_RESPONSE_TOO_LARGE_ID,
        STARTER_PLUGIN_REFUSAL_RUNTIME_RESOURCE_LIMIT_ID, STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID,
        STARTER_PLUGIN_REFUSAL_TIMEOUT_ID, STARTER_PLUGIN_REFUSAL_UNSUPPORTED_CODEC_ID,
        STARTER_PLUGIN_REFUSAL_UPSTREAM_FAILURE_ID, STARTER_PLUGIN_REFUSAL_URL_NOT_PERMITTED_ID,
        STARTER_PLUGIN_TEXT_URL_EXTRACT_OUTPUT_SCHEMA_ID, UrlExtractConfig,
        UrlExtractRuntimeCaseStatus, build_extract_readable_runtime_bundle,
        build_fetch_text_runtime_bundle, build_url_extract_runtime_bundle,
        invoke_extract_readable_json_packet, invoke_fetch_text_json_packet,
        invoke_url_extract_json_packet,
        tassadar_post_article_plugin_html_extract_readable_runtime_bundle_path,
        tassadar_post_article_plugin_http_fetch_text_runtime_bundle_path,
        tassadar_post_article_plugin_text_url_extract_runtime_bundle_path,
        write_extract_readable_runtime_bundle, write_fetch_text_runtime_bundle,
        write_url_extract_runtime_bundle,
    };
    use tempfile::tempdir;

    #[test]
    fn url_extract_success_preserves_order_and_duplicates() {
        let packet =
            br#"{"text":"https://alpha.example/a http://beta.test/b https://alpha.example/a"}"#;
        let outcome = invoke_url_extract_json_packet("json", packet, &UrlExtractConfig::default());

        assert_eq!(
            outcome.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_TEXT_URL_EXTRACT_OUTPUT_SCHEMA_ID
        );
        assert_eq!(
            outcome.response.expect("response").urls,
            vec![
                String::from("https://alpha.example/a"),
                String::from("http://beta.test/b"),
                String::from("https://alpha.example/a"),
            ]
        );
    }

    #[test]
    fn url_extract_refuses_schema_invalid_packets() {
        let outcome = invoke_url_extract_json_packet(
            "json",
            br#"{"body":"missing"}"#,
            &UrlExtractConfig::default(),
        );
        assert_eq!(
            outcome.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID
        );
    }

    #[test]
    fn url_extract_refuses_oversized_packets() {
        let outcome = invoke_url_extract_json_packet(
            "json",
            br#"{"text":"0123456789"}"#,
            &UrlExtractConfig {
                packet_size_limit_bytes: 8,
                max_urls: 8,
            },
        );
        assert_eq!(
            outcome.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_PACKET_TOO_LARGE_ID
        );
    }

    #[test]
    fn url_extract_refuses_unsupported_codecs() {
        let outcome = invoke_url_extract_json_packet(
            "bytes",
            br#"{"text":"https://alpha.example/a"}"#,
            &UrlExtractConfig::default(),
        );
        assert_eq!(
            outcome.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_UNSUPPORTED_CODEC_ID
        );
    }

    #[test]
    fn url_extract_refuses_runtime_resource_limit_overflow() {
        let outcome = invoke_url_extract_json_packet(
            "json",
            br#"{"text":"https://alpha.example/a https://beta.example/b"}"#,
            &UrlExtractConfig {
                packet_size_limit_bytes: 1024,
                max_urls: 1,
            },
        );
        assert_eq!(
            outcome.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_RUNTIME_RESOURCE_LIMIT_ID
        );
    }

    #[test]
    fn url_extract_runtime_bundle_covers_declared_cases() {
        let bundle = build_url_extract_runtime_bundle();

        assert_eq!(bundle.case_rows.len(), 5);
        assert!(
            bundle
                .case_rows
                .iter()
                .any(|row| row.status == UrlExtractRuntimeCaseStatus::ExactSuccess)
        );
        assert!(bundle.case_rows.iter().any(|row| {
            row.status == UrlExtractRuntimeCaseStatus::TypedMalformedPacket
                && row.response_or_refusal_schema_id == STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID
        }));
        assert!(bundle.case_rows.iter().any(|row| {
            row.status == UrlExtractRuntimeCaseStatus::TypedRefusal
                && row.response_or_refusal_schema_id == STARTER_PLUGIN_REFUSAL_PACKET_TOO_LARGE_ID
        }));
        assert!(bundle.case_rows.iter().any(|row| {
            row.status == UrlExtractRuntimeCaseStatus::TypedRefusal
                && row.response_or_refusal_schema_id == STARTER_PLUGIN_REFUSAL_UNSUPPORTED_CODEC_ID
        }));
        assert!(bundle.case_rows.iter().any(|row| {
            row.status == UrlExtractRuntimeCaseStatus::TypedRefusal
                && row.response_or_refusal_schema_id
                    == STARTER_PLUGIN_REFUSAL_RUNTIME_RESOURCE_LIMIT_ID
        }));
    }

    #[test]
    fn url_extract_runtime_bundle_writes_and_loads() {
        let tempdir = tempdir().expect("tempdir");
        let output_path = tempdir.path().join("url_extract_bundle.json");
        let written = write_url_extract_runtime_bundle(&output_path).expect("write bundle");
        let loaded: super::UrlExtractRuntimeBundle =
            super::load_url_extract_runtime_bundle(&output_path).expect("load bundle");

        assert_eq!(written, loaded);
        assert!(output_path.exists());
    }

    #[test]
    fn url_extract_runtime_bundle_repo_path_is_under_fixtures() {
        let path = tassadar_post_article_plugin_text_url_extract_runtime_bundle_path();
        assert!(path.ends_with(
            "fixtures/tassadar/runs/tassadar_post_article_plugin_text_url_extract_v1/tassadar_post_article_plugin_text_url_extract_bundle.json"
        ));
    }

    #[test]
    fn fetch_text_snapshot_success_returns_structured_fields() {
        let config = FetchTextConfig::snapshot(std::collections::BTreeMap::from([(
            String::from("https://snapshot.example/article"),
            FetchTextSnapshotResult::Success(FetchTextSnapshotResponse {
                final_url: String::from("https://snapshot.example/article"),
                status_code: 200,
                content_type: String::from("text/plain"),
                charset: Some(String::from("utf-8")),
                body_bytes: b"snapshot body".to_vec(),
            }),
        )]));
        let outcome = invoke_fetch_text_json_packet(
            "json",
            br#"{"url":"https://snapshot.example/article"}"#,
            &config,
        );

        assert_eq!(
            outcome.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_HTTP_FETCH_TEXT_OUTPUT_SCHEMA_ID
        );
        let response = outcome.response.expect("response");
        assert_eq!(response.status_code, 200);
        assert_eq!(response.body_text, "snapshot body");
    }

    #[test]
    fn fetch_text_refuses_blocked_urls() {
        let config = FetchTextConfig::snapshot(std::collections::BTreeMap::new());
        let outcome = invoke_fetch_text_json_packet(
            "json",
            br#"{"url":"https://blocked.example/article"}"#,
            &config,
        );
        assert_eq!(
            outcome.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_URL_NOT_PERMITTED_ID
        );
    }

    #[test]
    fn fetch_text_refuses_timeout_and_missing_snapshot_rows() {
        let config = FetchTextConfig::snapshot(std::collections::BTreeMap::from([(
            String::from("https://snapshot.example/slow"),
            FetchTextSnapshotResult::Timeout,
        )]));
        let timeout = invoke_fetch_text_json_packet(
            "json",
            br#"{"url":"https://snapshot.example/slow"}"#,
            &config,
        );
        assert_eq!(
            timeout.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_TIMEOUT_ID
        );

        let missing = invoke_fetch_text_json_packet(
            "json",
            br#"{"url":"https://snapshot.example/missing"}"#,
            &config,
        );
        assert_eq!(
            missing.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_NETWORK_DENIED_ID
        );
    }

    #[test]
    fn fetch_text_refuses_large_binary_decode_and_upstream_cases() {
        let config = FetchTextConfig::snapshot(std::collections::BTreeMap::from([
            (
                String::from("https://snapshot.example/large"),
                FetchTextSnapshotResult::Success(FetchTextSnapshotResponse {
                    final_url: String::from("https://snapshot.example/large"),
                    status_code: 200,
                    content_type: String::from("text/plain"),
                    charset: Some(String::from("utf-8")),
                    body_bytes: vec![b'x'; 16 * 1024 + 1],
                }),
            ),
            (
                String::from("https://snapshot.example/binary"),
                FetchTextSnapshotResult::Success(FetchTextSnapshotResponse {
                    final_url: String::from("https://snapshot.example/binary"),
                    status_code: 200,
                    content_type: String::from("image/png"),
                    charset: None,
                    body_bytes: vec![0x89, 0x50, 0x4e, 0x47],
                }),
            ),
            (
                String::from("https://snapshot.example/bad-utf8"),
                FetchTextSnapshotResult::Success(FetchTextSnapshotResponse {
                    final_url: String::from("https://snapshot.example/bad-utf8"),
                    status_code: 200,
                    content_type: String::from("text/plain"),
                    charset: Some(String::from("utf-8")),
                    body_bytes: vec![0xff, 0xfe],
                }),
            ),
            (
                String::from("https://snapshot.example/broken"),
                FetchTextSnapshotResult::UpstreamFailure {
                    detail: String::from("broken"),
                },
            ),
        ]));

        let large = invoke_fetch_text_json_packet(
            "json",
            br#"{"url":"https://snapshot.example/large"}"#,
            &config,
        );
        assert_eq!(
            large.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_RESPONSE_TOO_LARGE_ID
        );

        let bad_type = invoke_fetch_text_json_packet(
            "json",
            br#"{"url":"https://snapshot.example/binary"}"#,
            &config,
        );
        assert_eq!(
            bad_type.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID
        );

        let bad_decode = invoke_fetch_text_json_packet(
            "json",
            br#"{"url":"https://snapshot.example/bad-utf8"}"#,
            &config,
        );
        assert_eq!(
            bad_decode.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_DECODE_FAILED_ID
        );

        let upstream = invoke_fetch_text_json_packet(
            "json",
            br#"{"url":"https://snapshot.example/broken"}"#,
            &config,
        );
        assert_eq!(
            upstream.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_UPSTREAM_FAILURE_ID
        );
    }

    #[test]
    fn fetch_text_runtime_bundle_covers_declared_cases() {
        let bundle = build_fetch_text_runtime_bundle();

        assert_eq!(bundle.case_rows.len(), 9);
        assert!(
            bundle
                .case_rows
                .iter()
                .any(|row| row.status == FetchTextRuntimeCaseStatus::ExactSuccess)
        );
        assert!(bundle.case_rows.iter().any(|row| {
            row.status == FetchTextRuntimeCaseStatus::TypedMalformedPacket
                && row.response_or_refusal_schema_id == STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID
        }));
        assert!(bundle.case_rows.iter().any(|row| {
            row.response_or_refusal_schema_id == STARTER_PLUGIN_REFUSAL_URL_NOT_PERMITTED_ID
        }));
        assert!(bundle.case_rows.iter().any(|row| {
            row.response_or_refusal_schema_id == STARTER_PLUGIN_REFUSAL_TIMEOUT_ID
        }));
        assert!(bundle.case_rows.iter().any(|row| {
            row.response_or_refusal_schema_id == STARTER_PLUGIN_REFUSAL_NETWORK_DENIED_ID
        }));
        assert!(bundle.case_rows.iter().any(|row| {
            row.response_or_refusal_schema_id == STARTER_PLUGIN_REFUSAL_RESPONSE_TOO_LARGE_ID
        }));
        assert!(bundle.case_rows.iter().any(|row| {
            row.response_or_refusal_schema_id
                == STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID
        }));
        assert!(bundle.case_rows.iter().any(|row| {
            row.response_or_refusal_schema_id == STARTER_PLUGIN_REFUSAL_DECODE_FAILED_ID
        }));
        assert!(bundle.case_rows.iter().any(|row| {
            row.response_or_refusal_schema_id == STARTER_PLUGIN_REFUSAL_UPSTREAM_FAILURE_ID
        }));
    }

    #[test]
    fn fetch_text_runtime_bundle_writes_and_loads() {
        let tempdir = tempdir().expect("tempdir");
        let output_path = tempdir.path().join("fetch_text_bundle.json");
        let written = write_fetch_text_runtime_bundle(&output_path).expect("write bundle");
        let loaded: super::FetchTextRuntimeBundle =
            super::load_fetch_text_runtime_bundle(&output_path).expect("load bundle");

        assert_eq!(written, loaded);
        assert!(output_path.exists());
    }

    #[test]
    fn fetch_text_runtime_bundle_repo_path_is_under_fixtures() {
        let path = tassadar_post_article_plugin_http_fetch_text_runtime_bundle_path();
        assert!(path.ends_with(
            "fixtures/tassadar/runs/tassadar_post_article_plugin_http_fetch_text_v1/tassadar_post_article_plugin_http_fetch_text_bundle.json"
        ));
    }

    #[test]
    fn extract_readable_success_returns_metadata_and_links() {
        let packet = serde_json::to_vec(&serde_json::json!({
            "source_url": "https://snapshot.example/article",
            "content_type": "text/html",
            "body_text": "<html lang=\"en\"><head><title>Example</title><meta name=\"description\" content=\"summary\" /><link rel=\"canonical\" href=\"/article\" /></head><body><main><h1>Example</h1><p>Readable body.</p><a href=\"/alpha\">Alpha</a></main></body></html>"
        }))
        .expect("packet");
        let outcome = invoke_extract_readable_json_packet(
            "json",
            &packet,
            &ExtractReadableConfig::default(),
        );

        assert_eq!(
            outcome.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_HTML_EXTRACT_READABLE_OUTPUT_SCHEMA_ID
        );
        let response = outcome.response.expect("response");
        assert_eq!(response.title.as_deref(), Some("Example"));
        assert_eq!(response.canonical_url.as_deref(), Some("https://snapshot.example/article"));
        assert_eq!(response.content_language.as_deref(), Some("en"));
        assert_eq!(
            response.harvested_links,
            vec![String::from("https://snapshot.example/alpha")]
        );
    }

    #[test]
    fn extract_readable_refuses_schema_invalid_content_type_and_size() {
        let schema_invalid = invoke_extract_readable_json_packet(
            "json",
            br#"{"source_url":"https://snapshot.example/article","content_type":"text/html"}"#,
            &ExtractReadableConfig::default(),
        );
        assert_eq!(
            schema_invalid.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID
        );

        let wrong_type = invoke_extract_readable_json_packet(
            "json",
            br#"{"source_url":"https://snapshot.example/article","content_type":"application/json","body_text":"{}"}"#,
            &ExtractReadableConfig::default(),
        );
        assert_eq!(
            wrong_type.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID
        );

        let oversized = invoke_extract_readable_json_packet(
            "json",
            br#"{"source_url":"https://snapshot.example/article","content_type":"text/html","body_text":"xxxxxxxxxxxxxxxx"}"#,
            &ExtractReadableConfig {
                input_size_limit_bytes: 8,
                max_links: 8,
            },
        );
        assert_eq!(
            oversized.receipt.output_or_refusal_schema_id,
            STARTER_PLUGIN_REFUSAL_INPUT_TOO_LARGE_ID
        );
    }

    #[test]
    fn extract_readable_runtime_bundle_covers_declared_cases() {
        let bundle = build_extract_readable_runtime_bundle();

        assert_eq!(bundle.case_rows.len(), 5);
        assert!(bundle
            .case_rows
            .iter()
            .any(|row| row.status == ExtractReadableRuntimeCaseStatus::ExactSuccess));
        assert!(bundle.case_rows.iter().any(|row| {
            row.status == ExtractReadableRuntimeCaseStatus::TypedMalformedPacket
                && row.response_or_refusal_schema_id == STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID
        }));
        assert!(bundle.case_rows.iter().any(|row| {
            row.response_or_refusal_schema_id
                == STARTER_PLUGIN_REFUSAL_CONTENT_TYPE_UNSUPPORTED_ID
        }));
        assert!(bundle.case_rows.iter().any(|row| {
            row.response_or_refusal_schema_id == STARTER_PLUGIN_REFUSAL_INPUT_TOO_LARGE_ID
        }));
        assert!(bundle.composition_case.green);
        assert_eq!(
            bundle.composition_case.step_plugin_ids,
            vec![
                String::from("plugin.http.fetch_text"),
                String::from("plugin.html.extract_readable"),
            ]
        );
    }

    #[test]
    fn extract_readable_runtime_bundle_writes_and_loads() {
        let tempdir = tempdir().expect("tempdir");
        let output_path = tempdir.path().join("extract_readable_bundle.json");
        let written = write_extract_readable_runtime_bundle(&output_path).expect("write bundle");
        let loaded: super::ExtractReadableRuntimeBundle =
            super::load_extract_readable_runtime_bundle(&output_path).expect("load bundle");

        assert_eq!(written, loaded);
        assert!(output_path.exists());
    }

    #[test]
    fn extract_readable_runtime_bundle_repo_path_is_under_fixtures() {
        let path = tassadar_post_article_plugin_html_extract_readable_runtime_bundle_path();
        assert!(path.ends_with(
            "fixtures/tassadar/runs/tassadar_post_article_plugin_html_extract_readable_v1/tassadar_post_article_plugin_html_extract_readable_bundle.json"
        ));
    }
}
