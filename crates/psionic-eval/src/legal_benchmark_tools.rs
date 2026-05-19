//! Closed legal benchmark tool surface.
//!
//! These tools intentionally stay small and replayable: shell, read, write,
//! edit, glob, and grep. Shell execution is modeled as sandbox-owned; callers
//! must attach a sandbox runner before commands execute.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    ExtractedArtifact, Metadata, ToolCallRecord, TranscriptEvent, TranscriptEventKind,
    stable_json_digest,
};

pub const LEGAL_BENCHMARK_TOOL_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegalBenchmarkToolName {
    Shell,
    Read,
    Write,
    Edit,
    Glob,
    Grep,
}

impl LegalBenchmarkToolName {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Shell => "shell",
            Self::Read => "read",
            Self::Write => "write",
            Self::Edit => "edit",
            Self::Glob => "glob",
            Self::Grep => "grep",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegalBenchmarkPathRoot {
    Documents,
    Workspace,
    Output,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tool", content = "input", rename_all = "snake_case")]
pub enum LegalBenchmarkToolInput {
    Shell {
        command: Vec<String>,
        timeout_ms: Option<u64>,
    },
    Read {
        root: LegalBenchmarkPathRoot,
        relative_path: String,
        prefer_extracted: bool,
    },
    Write {
        root: LegalBenchmarkPathRoot,
        relative_path: String,
        content: String,
        overwrite: bool,
    },
    Edit {
        root: LegalBenchmarkPathRoot,
        relative_path: String,
        find: String,
        replace: String,
        expected_replacements: Option<u32>,
    },
    Glob {
        root: LegalBenchmarkPathRoot,
        pattern: String,
        max_results: usize,
        include_hidden: bool,
    },
    Grep {
        root: LegalBenchmarkPathRoot,
        pattern: String,
        case_sensitive: bool,
        max_results: usize,
        include_hidden: bool,
    },
}

impl LegalBenchmarkToolInput {
    pub const fn tool_name(&self) -> LegalBenchmarkToolName {
        match self {
            Self::Shell { .. } => LegalBenchmarkToolName::Shell,
            Self::Read { .. } => LegalBenchmarkToolName::Read,
            Self::Write { .. } => LegalBenchmarkToolName::Write,
            Self::Edit { .. } => LegalBenchmarkToolName::Edit,
            Self::Glob { .. } => LegalBenchmarkToolName::Glob,
            Self::Grep { .. } => LegalBenchmarkToolName::Grep,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegalBenchmarkToolFailureKind {
    SandboxUnavailable,
    SandboxFailed,
    InvalidPath,
    ReadForbidden,
    WriteForbidden,
    FileMissing,
    FileExists,
    InputTooLarge,
    BinaryFile,
    EditConflict,
    PatternInvalid,
    LimitExceeded,
    IoError,
    InternalError,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LegalBenchmarkPathTouch {
    pub root: LegalBenchmarkPathRoot,
    pub relative_path: String,
    pub access: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_hash: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LegalBenchmarkGrepMatch {
    pub root: LegalBenchmarkPathRoot,
    pub relative_path: String,
    pub line_number: u64,
    pub line: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tool", content = "output", rename_all = "snake_case")]
pub enum LegalBenchmarkToolOutput {
    Shell {
        stdout: String,
        stderr: String,
        exit_code: Option<i32>,
        sandbox_receipt_ref: Option<String>,
    },
    Read {
        content: String,
        source: String,
        bytes_read: u64,
    },
    Write {
        relative_path: String,
        bytes_written: u64,
        after_hash: String,
    },
    Edit {
        relative_path: String,
        replacements: u32,
        before_hash: String,
        after_hash: String,
        bytes_written: u64,
    },
    Glob {
        matches: Vec<String>,
        truncated: bool,
    },
    Grep {
        matches: Vec<LegalBenchmarkGrepMatch>,
        binary_files_skipped: u32,
        truncated: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegalBenchmarkToolReceipt {
    pub schema_version: u16,
    pub tool_call_id: String,
    pub tool_name: LegalBenchmarkToolName,
    pub input_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_kind: Option<LegalBenchmarkToolFailureKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_detail: Option<String>,
    pub elapsed_ms: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    pub touched_paths: Vec<LegalBenchmarkPathTouch>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sandbox_receipt_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegalBenchmarkToolExecution {
    pub receipt: LegalBenchmarkToolReceipt,
    pub input: LegalBenchmarkToolInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<LegalBenchmarkToolOutput>,
    pub transcript_events: Vec<TranscriptEvent>,
    pub tool_call_record: ToolCallRecord,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegalBenchmarkToolWorkspace {
    pub documents_root: PathBuf,
    pub workspace_root: PathBuf,
    pub output_root: PathBuf,
    pub extracted_text_by_path: BTreeMap<String, String>,
    pub max_read_bytes: usize,
    pub max_write_bytes: usize,
}

impl LegalBenchmarkToolWorkspace {
    pub fn new(
        documents_root: impl Into<PathBuf>,
        workspace_root: impl Into<PathBuf>,
        output_root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            documents_root: documents_root.into(),
            workspace_root: workspace_root.into(),
            output_root: output_root.into(),
            extracted_text_by_path: BTreeMap::new(),
            max_read_bytes: 4 * 1024 * 1024,
            max_write_bytes: 4 * 1024 * 1024,
        }
    }

    pub fn with_extracted_artifacts(mut self, artifacts: &[ExtractedArtifact]) -> Self {
        for artifact in artifacts {
            self.extracted_text_by_path.insert(
                artifact.artifact.relative_path.clone(),
                artifact.text.clone(),
            );
        }
        self
    }
}

pub fn execute_legal_benchmark_tool(
    workspace: &LegalBenchmarkToolWorkspace,
    input: LegalBenchmarkToolInput,
) -> LegalBenchmarkToolExecution {
    let started = Instant::now();
    match &input {
        LegalBenchmarkToolInput::Shell { .. } => failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::SandboxUnavailable,
            "shell tool requires an attached sandbox backend",
            Vec::new(),
            0,
            0,
            None,
            None,
        ),
        LegalBenchmarkToolInput::Read {
            root,
            relative_path,
            prefer_extracted,
        } => execute_read(
            workspace,
            input.clone(),
            *root,
            relative_path,
            *prefer_extracted,
            started,
        ),
        LegalBenchmarkToolInput::Write {
            root,
            relative_path,
            content,
            overwrite,
        } => execute_write(
            workspace,
            input.clone(),
            *root,
            relative_path,
            content,
            *overwrite,
            started,
        ),
        LegalBenchmarkToolInput::Edit {
            root,
            relative_path,
            find,
            replace,
            expected_replacements,
        } => execute_edit(
            workspace,
            input.clone(),
            *root,
            relative_path,
            find,
            replace,
            *expected_replacements,
            started,
        ),
        LegalBenchmarkToolInput::Glob {
            root,
            pattern,
            max_results,
            include_hidden,
        } => execute_glob(
            workspace,
            input.clone(),
            *root,
            pattern,
            *max_results,
            *include_hidden,
            started,
        ),
        LegalBenchmarkToolInput::Grep {
            root,
            pattern,
            case_sensitive,
            max_results,
            include_hidden,
        } => execute_grep(
            workspace,
            input.clone(),
            *root,
            pattern,
            *case_sensitive,
            *max_results,
            *include_hidden,
            started,
        ),
    }
}

#[cfg(feature = "full")]
pub fn execute_shell_with_podman(
    input: LegalBenchmarkToolInput,
    podman: &psionic_sandbox::PodmanSandboxBackend,
    config: &psionic_sandbox::PodmanSandboxConfig,
) -> LegalBenchmarkToolExecution {
    use psionic_sandbox::SandboxCommandBackend;

    let started = Instant::now();
    let LegalBenchmarkToolInput::Shell { command, .. } = &input else {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::InternalError,
            "execute_shell_with_podman only accepts shell input",
            Vec::new(),
            0,
            0,
            None,
            None,
        );
    };
    match podman.run_sandbox_command(config, command.as_slice()) {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(output.stdout.as_slice()).to_string();
            let stderr = String::from_utf8_lossy(output.stderr.as_slice()).to_string();
            let sandbox_receipt_ref = output.receipt.command_digest.clone();
            success_execution(
                input,
                started,
                LegalBenchmarkToolOutput::Shell {
                    stdout,
                    stderr,
                    exit_code: output.receipt.exit_code,
                    sandbox_receipt_ref: Some(sandbox_receipt_ref.clone()),
                },
                Vec::new(),
                output.receipt.stdout_bytes,
                0,
                output.receipt.exit_code,
                Some(sandbox_receipt_ref),
            )
        }
        Err(error) => failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::SandboxFailed,
            format!("sandbox command failed: {error}"),
            Vec::new(),
            0,
            0,
            None,
            None,
        ),
    }
}

pub fn legal_benchmark_tool_receipt_digest(
    receipt: &LegalBenchmarkToolReceipt,
) -> Result<String, serde_json::Error> {
    stable_json_digest("psionic.legal_benchmark.tool_receipt.v1", receipt)
}

fn execute_read(
    workspace: &LegalBenchmarkToolWorkspace,
    input: LegalBenchmarkToolInput,
    root: LegalBenchmarkPathRoot,
    relative_path: &str,
    prefer_extracted: bool,
    started: Instant,
) -> LegalBenchmarkToolExecution {
    if prefer_extracted {
        if let Some(text) = workspace.extracted_text_by_path.get(relative_path) {
            return success_execution(
                input,
                started,
                LegalBenchmarkToolOutput::Read {
                    content: text.clone(),
                    source: "extracted_text".to_string(),
                    bytes_read: u64::try_from(text.len()).unwrap_or(u64::MAX),
                },
                vec![LegalBenchmarkPathTouch {
                    root,
                    relative_path: relative_path.to_string(),
                    access: "read_extracted".to_string(),
                    before_hash: None,
                    after_hash: None,
                }],
                u64::try_from(text.len()).unwrap_or(u64::MAX),
                0,
                None,
                None,
            );
        }
    }
    let resolved = match resolve_existing_path(workspace, root, relative_path) {
        Ok(path) => path,
        Err(error) => return path_failure(input, started, root, relative_path, error),
    };
    let bytes = match fs::read(resolved.as_path()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return failure_execution(
                input,
                started,
                LegalBenchmarkToolFailureKind::IoError,
                format!("failed to read file: {error}"),
                Vec::new(),
                0,
                0,
                None,
                None,
            );
        }
    };
    if bytes.len() > workspace.max_read_bytes {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::InputTooLarge,
            "read file exceeds workspace max_read_bytes",
            Vec::new(),
            0,
            0,
            None,
            None,
        );
    }
    let Ok(content) = String::from_utf8(bytes.clone()) else {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::BinaryFile,
            "read refuses binary or non-UTF-8 files",
            Vec::new(),
            0,
            0,
            None,
            None,
        );
    };
    success_execution(
        input,
        started,
        LegalBenchmarkToolOutput::Read {
            content,
            source: "raw_file".to_string(),
            bytes_read: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        },
        vec![LegalBenchmarkPathTouch {
            root,
            relative_path: relative_path.to_string(),
            access: "read".to_string(),
            before_hash: Some(sha256_hex(bytes.as_slice())),
            after_hash: None,
        }],
        u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        0,
        None,
        None,
    )
}

fn execute_write(
    workspace: &LegalBenchmarkToolWorkspace,
    input: LegalBenchmarkToolInput,
    root: LegalBenchmarkPathRoot,
    relative_path: &str,
    content: &str,
    overwrite: bool,
    started: Instant,
) -> LegalBenchmarkToolExecution {
    if !matches!(
        root,
        LegalBenchmarkPathRoot::Workspace | LegalBenchmarkPathRoot::Output
    ) {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::WriteForbidden,
            "write is allowed only in workspace or output roots",
            Vec::new(),
            0,
            0,
            None,
            None,
        );
    }
    if content.len() > workspace.max_write_bytes {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::InputTooLarge,
            "write content exceeds workspace max_write_bytes",
            Vec::new(),
            0,
            0,
            None,
            None,
        );
    }
    let resolved = match resolve_new_or_existing_path(workspace, root, relative_path) {
        Ok(path) => path,
        Err(error) => return path_failure(input, started, root, relative_path, error),
    };
    if resolved.exists() && !overwrite {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::FileExists,
            "write refused to overwrite existing file",
            Vec::new(),
            0,
            0,
            None,
            None,
        );
    }
    if let Some(parent) = resolved.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            return failure_execution(
                input,
                started,
                LegalBenchmarkToolFailureKind::IoError,
                format!("failed to create parent directory: {error}"),
                Vec::new(),
                0,
                0,
                None,
                None,
            );
        }
    }
    if let Err(error) = fs::write(resolved.as_path(), content.as_bytes()) {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::IoError,
            format!("failed to write file: {error}"),
            Vec::new(),
            0,
            0,
            None,
            None,
        );
    }
    let after_hash = sha256_hex(content.as_bytes());
    success_execution(
        input,
        started,
        LegalBenchmarkToolOutput::Write {
            relative_path: relative_path.to_string(),
            bytes_written: u64::try_from(content.len()).unwrap_or(u64::MAX),
            after_hash: after_hash.clone(),
        },
        vec![LegalBenchmarkPathTouch {
            root,
            relative_path: relative_path.to_string(),
            access: "write".to_string(),
            before_hash: None,
            after_hash: Some(after_hash),
        }],
        0,
        u64::try_from(content.len()).unwrap_or(u64::MAX),
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_edit(
    workspace: &LegalBenchmarkToolWorkspace,
    input: LegalBenchmarkToolInput,
    root: LegalBenchmarkPathRoot,
    relative_path: &str,
    find: &str,
    replace: &str,
    expected_replacements: Option<u32>,
    started: Instant,
) -> LegalBenchmarkToolExecution {
    if find.is_empty() {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::EditConflict,
            "edit find string cannot be empty",
            Vec::new(),
            0,
            0,
            None,
            None,
        );
    }
    if !matches!(
        root,
        LegalBenchmarkPathRoot::Workspace | LegalBenchmarkPathRoot::Output
    ) {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::WriteForbidden,
            "edit is allowed only in workspace or output roots",
            Vec::new(),
            0,
            0,
            None,
            None,
        );
    }
    let resolved = match resolve_existing_path(workspace, root, relative_path) {
        Ok(path) => path,
        Err(error) => return path_failure(input, started, root, relative_path, error),
    };
    let bytes = match fs::read(resolved.as_path()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return failure_execution(
                input,
                started,
                LegalBenchmarkToolFailureKind::IoError,
                format!("failed to read file for edit: {error}"),
                Vec::new(),
                0,
                0,
                None,
                None,
            );
        }
    };
    let Ok(content) = String::from_utf8(bytes.clone()) else {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::BinaryFile,
            "edit refuses binary or non-UTF-8 files",
            Vec::new(),
            0,
            0,
            None,
            None,
        );
    };
    let replacements = u32::try_from(content.matches(find).count()).unwrap_or(u32::MAX);
    if replacements == 0 || expected_replacements.is_some_and(|expected| expected != replacements) {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::EditConflict,
            "edit replacement count did not match expectation",
            Vec::new(),
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            0,
            None,
            None,
        );
    }
    let edited = content.replace(find, replace);
    if edited.len() > workspace.max_write_bytes {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::InputTooLarge,
            "edited content exceeds workspace max_write_bytes",
            Vec::new(),
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            0,
            None,
            None,
        );
    }
    if let Err(error) = fs::write(resolved.as_path(), edited.as_bytes()) {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::IoError,
            format!("failed to write edited file: {error}"),
            Vec::new(),
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            0,
            None,
            None,
        );
    }
    let before_hash = sha256_hex(bytes.as_slice());
    let after_hash = sha256_hex(edited.as_bytes());
    success_execution(
        input,
        started,
        LegalBenchmarkToolOutput::Edit {
            relative_path: relative_path.to_string(),
            replacements,
            before_hash: before_hash.clone(),
            after_hash: after_hash.clone(),
            bytes_written: u64::try_from(edited.len()).unwrap_or(u64::MAX),
        },
        vec![LegalBenchmarkPathTouch {
            root,
            relative_path: relative_path.to_string(),
            access: "edit".to_string(),
            before_hash: Some(before_hash),
            after_hash: Some(after_hash),
        }],
        u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        u64::try_from(edited.len()).unwrap_or(u64::MAX),
        None,
        None,
    )
}

fn execute_glob(
    workspace: &LegalBenchmarkToolWorkspace,
    input: LegalBenchmarkToolInput,
    root: LegalBenchmarkPathRoot,
    pattern: &str,
    max_results: usize,
    include_hidden: bool,
    started: Instant,
) -> LegalBenchmarkToolExecution {
    if pattern.trim().is_empty() {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::PatternInvalid,
            "glob pattern cannot be empty",
            Vec::new(),
            0,
            0,
            None,
            None,
        );
    }
    let root_path = root_path(workspace, root);
    let files = match list_files(root_path, include_hidden) {
        Ok(files) => files,
        Err(error) => {
            return failure_execution(
                input,
                started,
                LegalBenchmarkToolFailureKind::IoError,
                format!("failed to list files: {error}"),
                Vec::new(),
                0,
                0,
                None,
                None,
            );
        }
    };
    let mut matches = files
        .into_iter()
        .filter(|path| wildcard_matches(pattern, path.as_str()))
        .collect::<Vec<_>>();
    matches.sort();
    let truncated = matches.len() > max_results;
    matches.truncate(max_results);
    success_execution(
        input,
        started,
        LegalBenchmarkToolOutput::Glob { matches, truncated },
        Vec::new(),
        0,
        0,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_grep(
    workspace: &LegalBenchmarkToolWorkspace,
    input: LegalBenchmarkToolInput,
    root: LegalBenchmarkPathRoot,
    pattern: &str,
    case_sensitive: bool,
    max_results: usize,
    include_hidden: bool,
    started: Instant,
) -> LegalBenchmarkToolExecution {
    if pattern.is_empty() {
        return failure_execution(
            input,
            started,
            LegalBenchmarkToolFailureKind::PatternInvalid,
            "grep pattern cannot be empty",
            Vec::new(),
            0,
            0,
            None,
            None,
        );
    }
    let root_path = root_path(workspace, root);
    let files = match list_files(root_path, include_hidden) {
        Ok(files) => files,
        Err(error) => {
            return failure_execution(
                input,
                started,
                LegalBenchmarkToolFailureKind::IoError,
                format!("failed to list files: {error}"),
                Vec::new(),
                0,
                0,
                None,
                None,
            );
        }
    };
    let needle = if case_sensitive {
        pattern.to_string()
    } else {
        pattern.to_ascii_lowercase()
    };
    let mut matches = Vec::new();
    let mut binary_files_skipped = 0u32;
    let mut bytes_read = 0u64;
    'files: for relative_path in files {
        let path = root_path.join(relative_path.as_str());
        let bytes = match fs::read(path.as_path()) {
            Ok(bytes) => bytes,
            Err(_) => continue,
        };
        bytes_read = bytes_read.saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        let Ok(content) = String::from_utf8(bytes) else {
            binary_files_skipped = binary_files_skipped.saturating_add(1);
            continue;
        };
        for (line_index, line) in content.lines().enumerate() {
            let haystack = if case_sensitive {
                line.to_string()
            } else {
                line.to_ascii_lowercase()
            };
            if haystack.contains(needle.as_str()) {
                matches.push(LegalBenchmarkGrepMatch {
                    root,
                    relative_path: relative_path.clone(),
                    line_number: u64::try_from(line_index + 1).unwrap_or(u64::MAX),
                    line: line.to_string(),
                });
                if matches.len() >= max_results {
                    break 'files;
                }
            }
        }
    }
    let truncated = matches.len() >= max_results;
    success_execution(
        input,
        started,
        LegalBenchmarkToolOutput::Grep {
            matches,
            binary_files_skipped,
            truncated,
        },
        Vec::new(),
        bytes_read,
        0,
        None,
        None,
    )
}

fn success_execution(
    input: LegalBenchmarkToolInput,
    started: Instant,
    output: LegalBenchmarkToolOutput,
    touched_paths: Vec<LegalBenchmarkPathTouch>,
    bytes_read: u64,
    bytes_written: u64,
    exit_code: Option<i32>,
    sandbox_receipt_ref: Option<String>,
) -> LegalBenchmarkToolExecution {
    let output_hash = stable_json_digest("psionic.legal_benchmark.tool_output.v1", &output).ok();
    build_execution(
        input,
        Some(output),
        None,
        None,
        started,
        touched_paths,
        bytes_read,
        bytes_written,
        exit_code,
        sandbox_receipt_ref,
        output_hash,
    )
}

#[allow(clippy::too_many_arguments)]
fn failure_execution(
    input: LegalBenchmarkToolInput,
    started: Instant,
    failure_kind: LegalBenchmarkToolFailureKind,
    failure_detail: impl Into<String>,
    touched_paths: Vec<LegalBenchmarkPathTouch>,
    bytes_read: u64,
    bytes_written: u64,
    exit_code: Option<i32>,
    sandbox_receipt_ref: Option<String>,
) -> LegalBenchmarkToolExecution {
    build_execution(
        input,
        None,
        Some(failure_kind),
        Some(failure_detail.into()),
        started,
        touched_paths,
        bytes_read,
        bytes_written,
        exit_code,
        sandbox_receipt_ref,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_execution(
    input: LegalBenchmarkToolInput,
    output: Option<LegalBenchmarkToolOutput>,
    failure_kind: Option<LegalBenchmarkToolFailureKind>,
    failure_detail: Option<String>,
    started: Instant,
    touched_paths: Vec<LegalBenchmarkPathTouch>,
    bytes_read: u64,
    bytes_written: u64,
    exit_code: Option<i32>,
    sandbox_receipt_ref: Option<String>,
    output_hash: Option<String>,
) -> LegalBenchmarkToolExecution {
    let input_hash =
        stable_json_digest("psionic.legal_benchmark.tool_input.v1", &input).unwrap_or_default();
    let tool_call_id = deterministic_tool_call_id(input.tool_name(), input_hash.as_str());
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let receipt = LegalBenchmarkToolReceipt {
        schema_version: LEGAL_BENCHMARK_TOOL_SCHEMA_VERSION,
        tool_call_id: tool_call_id.clone(),
        tool_name: input.tool_name(),
        input_hash,
        output_hash,
        failure_kind,
        failure_detail,
        elapsed_ms,
        bytes_read,
        bytes_written,
        exit_code,
        touched_paths,
        sandbox_receipt_ref,
        metadata: BTreeMap::new(),
    };
    let output_value = output
        .as_ref()
        .and_then(|output| serde_json::to_value(output).ok());
    let error_kind = receipt.failure_kind.map(|kind| format!("{kind:?}"));
    let transcript_events =
        transcript_events(&tool_call_id, &input, output_value.clone(), &receipt);
    let tool_call_record = ToolCallRecord {
        tool_call_id,
        tool_name: receipt.tool_name.as_str().to_string(),
        call_event_index: 0,
        result_event_index: Some(1),
        input: serde_json::to_value(&input).unwrap_or(Value::Null),
        output: output_value,
        error_kind,
        elapsed_ms,
    };
    LegalBenchmarkToolExecution {
        receipt,
        input,
        output,
        transcript_events,
        tool_call_record,
    }
}

fn transcript_events(
    tool_call_id: &str,
    input: &LegalBenchmarkToolInput,
    output: Option<Value>,
    receipt: &LegalBenchmarkToolReceipt,
) -> Vec<TranscriptEvent> {
    let timestamp = now_ms();
    vec![
        TranscriptEvent {
            event_index: 0,
            event_kind: TranscriptEventKind::ToolCall,
            role: Some("assistant".to_string()),
            content: None,
            payload: Some(json!({
                "tool_call_id": tool_call_id,
                "tool_name": input.tool_name().as_str(),
                "input": input
            })),
            timestamp_ms: timestamp,
        },
        TranscriptEvent {
            event_index: 1,
            event_kind: TranscriptEventKind::ToolResult,
            role: Some("tool".to_string()),
            content: None,
            payload: Some(json!({
                "tool_call_id": tool_call_id,
                "output": output,
                "receipt": receipt
            })),
            timestamp_ms: timestamp,
        },
    ]
}

fn path_failure(
    input: LegalBenchmarkToolInput,
    started: Instant,
    root: LegalBenchmarkPathRoot,
    relative_path: &str,
    detail: String,
) -> LegalBenchmarkToolExecution {
    failure_execution(
        input,
        started,
        LegalBenchmarkToolFailureKind::InvalidPath,
        detail,
        vec![LegalBenchmarkPathTouch {
            root,
            relative_path: relative_path.to_string(),
            access: "rejected".to_string(),
            before_hash: None,
            after_hash: None,
        }],
        0,
        0,
        None,
        None,
    )
}

fn resolve_existing_path(
    workspace: &LegalBenchmarkToolWorkspace,
    root: LegalBenchmarkPathRoot,
    relative_path: &str,
) -> Result<PathBuf, String> {
    let path = resolve_new_or_existing_path(workspace, root, relative_path)?;
    if !path.exists() {
        return Err("path does not exist".to_string());
    }
    Ok(path)
}

fn resolve_new_or_existing_path(
    workspace: &LegalBenchmarkToolWorkspace,
    root: LegalBenchmarkPathRoot,
    relative_path: &str,
) -> Result<PathBuf, String> {
    validate_relative_path(relative_path)?;
    let root_path = root_path(workspace, root);
    let candidate = root_path.join(relative_path);
    let canonical_root = root_path
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize root: {error}"))?;
    let parent = candidate.parent().unwrap_or(root_path);
    let canonical_parent = if parent.exists() {
        parent
            .canonicalize()
            .map_err(|error| format!("failed to canonicalize parent: {error}"))?
    } else {
        let mut ancestor = parent;
        while !ancestor.exists() {
            ancestor = ancestor.parent().ok_or_else(|| {
                "path parent does not have an existing ancestor under root".to_string()
            })?;
        }
        ancestor
            .canonicalize()
            .map_err(|error| format!("failed to canonicalize ancestor: {error}"))?
    };
    if !canonical_parent.starts_with(canonical_root.as_path()) {
        return Err("path escapes selected root".to_string());
    }
    Ok(candidate)
}

fn validate_relative_path(relative_path: &str) -> Result<(), String> {
    if relative_path.trim().is_empty() {
        return Err("relative path cannot be empty".to_string());
    }
    let path = Path::new(relative_path);
    if path.is_absolute() {
        return Err("relative path cannot be absolute".to_string());
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir => return Err("relative path cannot contain .".to_string()),
            Component::ParentDir => return Err("relative path cannot contain ..".to_string()),
            Component::RootDir | Component::Prefix(_) => {
                return Err("relative path cannot contain root or prefix".to_string());
            }
        }
    }
    Ok(())
}

fn root_path(workspace: &LegalBenchmarkToolWorkspace, root: LegalBenchmarkPathRoot) -> &Path {
    match root {
        LegalBenchmarkPathRoot::Documents => workspace.documents_root.as_path(),
        LegalBenchmarkPathRoot::Workspace => workspace.workspace_root.as_path(),
        LegalBenchmarkPathRoot::Output => workspace.output_root.as_path(),
    }
}

fn list_files(root: &Path, include_hidden: bool) -> Result<Vec<String>, std::io::Error> {
    let mut files = Vec::new();
    list_files_inner(root, root, include_hidden, &mut files)?;
    files.sort();
    Ok(files)
}

fn list_files_inner(
    root: &Path,
    current: &Path,
    include_hidden: bool,
    files: &mut Vec<String>,
) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let hidden = name.to_string_lossy().starts_with('.');
        if hidden && !include_hidden {
            continue;
        }
        if path.is_dir() {
            list_files_inner(root, path.as_path(), include_hidden, files)?;
        } else if path.is_file() {
            let relative = path
                .strip_prefix(root)
                .map(|path| path.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string_lossy().to_string());
            files.push(relative);
        }
    }
    Ok(())
}

fn wildcard_matches(pattern: &str, value: &str) -> bool {
    wildcard_matches_inner(pattern.as_bytes(), value.as_bytes())
}

fn wildcard_matches_inner(pattern: &[u8], value: &[u8]) -> bool {
    if pattern.is_empty() {
        return value.is_empty();
    }
    match pattern[0] {
        b'*' => {
            wildcard_matches_inner(&pattern[1..], value)
                || (!value.is_empty() && wildcard_matches_inner(pattern, &value[1..]))
        }
        b'?' => !value.is_empty() && wildcard_matches_inner(&pattern[1..], &value[1..]),
        literal => {
            !value.is_empty()
                && literal == value[0]
                && wildcard_matches_inner(&pattern[1..], &value[1..])
        }
    }
}

fn deterministic_tool_call_id(tool_name: LegalBenchmarkToolName, input_hash: &str) -> String {
    let digest = sha256_hex(format!("{}|{input_hash}", tool_name.as_str()).as_bytes());
    format!("tool_call.{}.{}", tool_name.as_str(), digest)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArtifactKind, DataClassification, SourceArtifact};

    fn workspace() -> LegalBenchmarkToolWorkspace {
        let root = tempfile::tempdir().expect("tempdir");
        let root_path = root.keep();
        let documents = root_path.join("documents");
        let workspace = root_path.join("workspace");
        let output = root_path.join("output");
        fs::create_dir_all(documents.join("nested")).expect("documents");
        fs::create_dir_all(workspace.join("nested")).expect("workspace");
        fs::create_dir_all(output.as_path()).expect("output");
        fs::write(documents.join("case.txt"), "Alpha clause\nBeta clause\n").expect("case");
        fs::write(workspace.join("notes.md"), "Alpha draft\nAlpha clause\n").expect("notes");
        fs::write(workspace.join("nested/summary.txt"), "Beta summary\n").expect("summary");
        fs::write(workspace.join(".hidden.txt"), "hidden Alpha\n").expect("hidden");
        fs::write(workspace.join("binary.bin"), [0, 159, 146, 150]).expect("binary");

        let extracted = ExtractedArtifact {
            artifact: SourceArtifact {
                artifact_id: "artifact.source.case.extracted".to_string(),
                artifact_kind: ArtifactKind::ExtractedText,
                relative_path: "case.txt".to_string(),
                original_filename: "case.txt".to_string(),
                media_type: "text/plain".to_string(),
                byte_size: 12,
                sha256: "hash".to_string(),
                data_classification: DataClassification::PublicReference,
                provenance: Some("receipt".to_string()),
            },
            text: "Extracted Alpha clause".to_string(),
        };
        LegalBenchmarkToolWorkspace::new(documents, workspace, output)
            .with_extracted_artifacts(&[extracted])
    }

    #[test]
    fn read_prefers_extracted_text_when_available() {
        let workspace = workspace();
        let execution = execute_legal_benchmark_tool(
            &workspace,
            LegalBenchmarkToolInput::Read {
                root: LegalBenchmarkPathRoot::Documents,
                relative_path: "case.txt".to_string(),
                prefer_extracted: true,
            },
        );
        assert!(execution.receipt.failure_kind.is_none());
        assert_eq!(execution.transcript_events.len(), 2);
        match execution.output.expect("read output") {
            LegalBenchmarkToolOutput::Read {
                content, source, ..
            } => {
                assert_eq!(source, "extracted_text");
                assert!(content.contains("Extracted Alpha"));
            }
            _ => panic!("unexpected output"),
        }
    }

    #[test]
    fn write_and_edit_are_limited_to_output_or_workspace_roots() {
        let workspace = workspace();
        let write = execute_legal_benchmark_tool(
            &workspace,
            LegalBenchmarkToolInput::Write {
                root: LegalBenchmarkPathRoot::Output,
                relative_path: "answer.md".to_string(),
                content: "Initial answer".to_string(),
                overwrite: false,
            },
        );
        assert!(write.receipt.failure_kind.is_none());

        let edit = execute_legal_benchmark_tool(
            &workspace,
            LegalBenchmarkToolInput::Edit {
                root: LegalBenchmarkPathRoot::Output,
                relative_path: "answer.md".to_string(),
                find: "Initial".to_string(),
                replace: "Final".to_string(),
                expected_replacements: Some(1),
            },
        );
        assert!(edit.receipt.failure_kind.is_none());
        assert_eq!(edit.receipt.touched_paths.len(), 1);

        let forbidden = execute_legal_benchmark_tool(
            &workspace,
            LegalBenchmarkToolInput::Write {
                root: LegalBenchmarkPathRoot::Documents,
                relative_path: "case.txt".to_string(),
                content: "bad".to_string(),
                overwrite: true,
            },
        );
        assert_eq!(
            forbidden.receipt.failure_kind,
            Some(LegalBenchmarkToolFailureKind::WriteForbidden)
        );
    }

    #[test]
    fn edit_conflict_is_structured() {
        let workspace = workspace();
        let execution = execute_legal_benchmark_tool(
            &workspace,
            LegalBenchmarkToolInput::Edit {
                root: LegalBenchmarkPathRoot::Workspace,
                relative_path: "notes.md".to_string(),
                find: "Alpha".to_string(),
                replace: "Gamma".to_string(),
                expected_replacements: Some(1),
            },
        );
        assert_eq!(
            execution.receipt.failure_kind,
            Some(LegalBenchmarkToolFailureKind::EditConflict)
        );
    }

    #[test]
    fn glob_orders_results_and_respects_hidden_policy() {
        let workspace = workspace();
        let execution = execute_legal_benchmark_tool(
            &workspace,
            LegalBenchmarkToolInput::Glob {
                root: LegalBenchmarkPathRoot::Workspace,
                pattern: "*.txt".to_string(),
                max_results: 10,
                include_hidden: false,
            },
        );
        match execution.output.expect("glob output") {
            LegalBenchmarkToolOutput::Glob { matches, truncated } => {
                assert!(!truncated);
                assert_eq!(matches, vec!["nested/summary.txt"]);
            }
            _ => panic!("unexpected output"),
        }
    }

    #[test]
    fn grep_skips_binary_files_and_records_matches() {
        let workspace = workspace();
        let execution = execute_legal_benchmark_tool(
            &workspace,
            LegalBenchmarkToolInput::Grep {
                root: LegalBenchmarkPathRoot::Workspace,
                pattern: "alpha".to_string(),
                case_sensitive: false,
                max_results: 10,
                include_hidden: false,
            },
        );
        match execution.output.expect("grep output") {
            LegalBenchmarkToolOutput::Grep {
                matches,
                binary_files_skipped,
                ..
            } => {
                assert_eq!(matches.len(), 2);
                assert_eq!(binary_files_skipped, 1);
            }
            _ => panic!("unexpected output"),
        }
    }

    #[test]
    fn shell_without_sandbox_is_structured_failure() {
        let workspace = workspace();
        let execution = execute_legal_benchmark_tool(
            &workspace,
            LegalBenchmarkToolInput::Shell {
                command: vec!["echo".to_string(), "hello".to_string()],
                timeout_ms: Some(1000),
            },
        );
        assert_eq!(
            execution.receipt.failure_kind,
            Some(LegalBenchmarkToolFailureKind::SandboxUnavailable)
        );
        assert_eq!(
            execution.tool_call_record.error_kind.as_deref(),
            Some("SandboxUnavailable")
        );
    }

    #[test]
    fn traversal_paths_are_rejected() {
        let workspace = workspace();
        let execution = execute_legal_benchmark_tool(
            &workspace,
            LegalBenchmarkToolInput::Read {
                root: LegalBenchmarkPathRoot::Documents,
                relative_path: "../case.txt".to_string(),
                prefer_extracted: false,
            },
        );
        assert_eq!(
            execution.receipt.failure_kind,
            Some(LegalBenchmarkToolFailureKind::InvalidPath)
        );
    }
}
