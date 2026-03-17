# Psionic MLX-LM Package

> Status: canonical `PMLX-701` / `#14` reference record, updated 2026-03-17
> after landing `crates/psionic-mlx-lm`.

`psionic-mlx-lm` is the first bounded `mlx-lm`-style text package in this
repo.

It does not create a second runtime. It packages the already-landed Psionic
GGUF loader, prompt renderer, generation runtime, streaming surface, continuous
batch scheduler, and shared-prefix cache provenance into one local library and
CLI boundary.

## Current Scope

- load one local GGUF text model through `psionic-catalog` and `psionic-serve`
- expose a package-owned load report with blob identity, decoder descriptor,
  runtime support, and chat-template metadata
- render chat prompts through the GGUF template metadata before generation
- execute generate, stream, and continuous-batch workflows
- persist prompt-cache artifacts from generation responses as JSON
- keep context-overflow and shared-prefix cache policy explicit at the request
  boundary

## Current Boundaries

- local GGUF paths only
- package-owned CLI only
- no remote registry, Hugging Face cache, or architecture-registry workflow yet
- no OpenAI-compatible HTTP surface in this crate
- no notebook/UI shell

Those later surfaces belong to `PMLX-702` through `PMLX-709`.

## Library Surface

The crate now exposes:

- `MlxLmTextRuntime`
- `MlxLmTextRequest`
- `MlxLmLoadReport`
- `MlxLmPromptCacheArtifact`
- `MlxLmBatchReport`

The package keeps public request semantics honest by reusing the native Psionic
types underneath:

- `GenerationOptions`
- `ContextOverflowPolicy`
- `PrefixCacheControl`
- `PromptMessage`
- `RenderedPrompt`

## CLI

Run from the repo root:

```bash
cargo run -p psionic-mlx-lm --bin psionic-mlx-lm -- load --model /path/to/model.gguf
cargo run -p psionic-mlx-lm --bin psionic-mlx-lm -- generate --model /path/to/model.gguf --prompt "hello"
cargo run -p psionic-mlx-lm --bin psionic-mlx-lm -- stream --model /path/to/model.gguf --prompt "hello"
cargo run -p psionic-mlx-lm --bin psionic-mlx-lm -- batch --model /path/to/model.gguf --prompt "hello" --prompt "hello again"
```

Render one chat prompt from GGUF template metadata:

```bash
cargo run -p psionic-mlx-lm --bin psionic-mlx-lm -- render-chat \
  --model /path/to/model.gguf \
  --messages-json /tmp/messages.json
```

Persist one prompt-cache artifact from a generate or stream run:

```bash
cargo run -p psionic-mlx-lm --bin psionic-mlx-lm -- generate \
  --model /path/to/model.gguf \
  --prompt "hello" \
  --prompt-cache-artifact /tmp/prompt-cache.json
```

## Why This Exists

`PMLX-701` is not about claiming the whole MLX ecosystem is done. It is the
first package layer above the native framework so local text-model workflows no
longer require callers to stitch together raw GGUF loading, prompt rendering,
generation, streaming, batch scheduling, and prefix-cache receipts by hand.
