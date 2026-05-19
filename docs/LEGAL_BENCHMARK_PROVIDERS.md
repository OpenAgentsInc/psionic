# Legal Benchmark Provider Adapters

> Status: implemented_early.

The legal benchmark provider surface lives in
`crates/psionic-eval/src/legal_benchmark_provider.rs`. It defines the model
contract that the Rust benchmark agent loop will consume without binding the
runner to one hosted API, one local server, or one credential source.

## Contract

The adapter layer defines:

- `ModelProviderRoute`
- `ModelRequest`
- `ModelMessage`
- `ModelToolSpec`
- `ModelToolCall`
- `ToolResultMessage`
- `ModelResponse`
- `ModelUsage`
- `ModelAdapter`
- `ProviderHttpTransport`
- `OpenAiCompatibleAdapter`
- `AnthropicAdapter`
- `MockModelAdapter`

Routes record provider family, base URL, model id, endpoint path, route id,
model config hash, and secret reference id. They do not carry raw API keys.
HTTP request builders use redacted credential placeholders such as
`<secret_ref:secret.openai.local>` so serialized run artifacts can identify the
credential route without leaking the credential value.

## Tool Calling

`legal_benchmark_model_tool_specs()` exposes the closed benchmark tool set:

- shell
- read
- write
- edit
- glob
- grep

OpenAI-compatible responses normalize `tool_calls[].function` payloads into
`ModelToolCall`. Anthropic responses normalize `tool_use` content blocks into
the same structure. Tool results flow back through `ToolResultMessage`, which
can be rendered as provider-specific tool-result messages by the adapters.

## Failure Classification

Provider failures are explicit:

- `timeout`
- `rate_limited`
- `context_overflow`
- `safety_refusal`
- `provider_error`
- `transport_error`
- `parse_error`
- `invalid_request`
- `internal_error`

Retry behavior is configured by `ModelRetryPolicy`. Rate limits, timeouts, and
server errors can retry without sleeping in the adapter itself; the outer runner
owns wall-clock budgeting.

## CI And Live Smoke

CI should use `MockModelAdapter` or `MockHttpTransport` and must not require
live provider keys. A live smoke can be added by an operator-controlled wrapper
only when a secret reference resolves outside the run artifact writer.

The adapter records:

- provider route
- provider family
- model id
- model config hash
- secret reference id
- input/output/cached token usage
- elapsed time
- retry count
- raw response hash

Those fields are the data the agent loop needs for run receipts, cost analysis,
provider comparison, and later Autopilot4 import.
