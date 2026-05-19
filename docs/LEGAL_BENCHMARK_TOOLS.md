# Legal Benchmark Tools

> Status: implemented_early for typed Rust schemas and deterministic local
> filesystem behavior.

The legal benchmark tool surface lives in
`crates/psionic-eval/src/legal_benchmark_tools.rs`.

It defines the closed Harvey-compatible tool set:

- `shell`
- `read`
- `write`
- `edit`
- `glob`
- `grep`

Every call returns a `LegalBenchmarkToolExecution` with:

- a typed input and output
- a `LegalBenchmarkToolReceipt`
- structured failure kind and detail when the tool fails
- touched paths
- bytes read and written
- exit status for shell-capable results
- transcript call/result events
- a `ToolCallRecord`

## Root Model

Paths resolve against one of three roots:

- `documents`
- `workspace`
- `output`

Relative paths must stay relative, may not contain `.` or `..`, and are checked
against the selected root before filesystem access. `write` and `edit` are
allowed only in `workspace` or `output`. Source documents remain read-only.

## Tool Behavior

`read` prefers extracted text when the caller asks for it and the extracted
artifact is available. Otherwise it reads UTF-8 files from the selected root and
returns a structured binary-file error for non-text inputs.

`write` creates or overwrites files only in writable roots and records the
after hash.

`edit` performs exact string replacement, requires a non-empty find string, and
can enforce an expected replacement count. Conflicts return `edit_conflict`
instead of partially editing a file.

`glob` walks files deterministically, applies a small `*` and `?` wildcard
matcher, respects hidden-file policy, and truncates at the caller limit.

`grep` performs deterministic substring matching, supports case-insensitive
matching, skips binary files, and records how many binary files were skipped.

`shell` is sandbox-owned. The generic dispatcher returns `sandbox_unavailable`
unless a sandbox runner is attached. With the full feature set,
`execute_shell_with_podman` routes shell commands through the `psionic-sandbox`
Podman backend and records the sandbox command digest as the sandbox receipt
reference.

## Prompt Surface

Agents should receive the six tools as a closed set. Prompts should state that:

- document inputs are read-only
- outputs belong in the output root
- workspace files are scratch files
- shell may be unavailable unless the run profile grants a sandbox
- binary files and path traversal are structured tool errors
- every tool call is transcripted and receipt-backed
