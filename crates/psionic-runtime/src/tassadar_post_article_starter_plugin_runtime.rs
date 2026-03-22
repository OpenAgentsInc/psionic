use std::{
    fs,
    path::{Path, PathBuf},
};

#[cfg(test)]
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::TASSADAR_POST_ARTICLE_PLUGIN_PACKET_ABI_VERSION;

pub const TASSADAR_POST_ARTICLE_PLUGIN_TEXT_URL_EXTRACT_RUNTIME_BUNDLE_REF: &str = "fixtures/tassadar/runs/tassadar_post_article_plugin_text_url_extract_v1/tassadar_post_article_plugin_text_url_extract_bundle.json";
pub const TASSADAR_POST_ARTICLE_PLUGIN_TEXT_URL_EXTRACT_RUN_ROOT_REF: &str =
    "fixtures/tassadar/runs/tassadar_post_article_plugin_text_url_extract_v1";

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
        STARTER_PLUGIN_REFUSAL_PACKET_TOO_LARGE_ID,
        STARTER_PLUGIN_REFUSAL_RUNTIME_RESOURCE_LIMIT_ID, STARTER_PLUGIN_REFUSAL_SCHEMA_INVALID_ID,
        STARTER_PLUGIN_REFUSAL_UNSUPPORTED_CODEC_ID,
        STARTER_PLUGIN_TEXT_URL_EXTRACT_OUTPUT_SCHEMA_ID, UrlExtractConfig,
        UrlExtractRuntimeCaseStatus, build_url_extract_runtime_bundle,
        invoke_url_extract_json_packet,
        tassadar_post_article_plugin_text_url_extract_runtime_bundle_path,
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
}
