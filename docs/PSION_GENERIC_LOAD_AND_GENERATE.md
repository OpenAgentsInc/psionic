# Psion Generic Load And Generate

> Status: canonical `PSION-0901` / `#785` generic learned-lane load and
> generate packet, written 2026-03-30 after landing the first retained
> operator-facing runtime packet on top of the existing `psionic-serve`
> substrate.

This document records the first honest generic `Psion` load to serve closure.

It does not introduce a second serve stack. It binds the generic learned lane
to the already-shipped `psionic-serve` runtime seams:

- load one artifact-backed decoder model
- expose that model through the normal local runtime catalog
- encode a prompt
- generate a response
- attach explicit served-evidence and served-output-claim posture
- retain one packet with stable digests for review

That is the correct integrity closeout for the generic family: a real
operator-facing load-and-generate seam on the current artifact-backed decoder
runtime, not a prose claim that the lane is "serve-ready."

## Canonical Artifacts

- `docs/PSION_GENERIC_LOAD_AND_GENERATE.md` is the canonical human-readable
  packet contract.
- `crates/psionic-serve/src/psion_generic_load_and_generate.rs` owns the
  retained packet builder and fixture-matching test.
- `crates/psionic-serve/examples/psion_generic_load_and_generate_fixtures.rs`
  writes the canonical retained packet fixture.
- `fixtures/psion/serve/psion_generic_load_and_generate_v1.json` is the
  retained operator-facing packet.

The stable schema version is `psion.generic_load_and_generate.v1`.

## What This Packet Proves

The retained packet proves four narrow things:

- the generic learned `Psion` family can load one typed artifact through the
  current local runtime catalog
- the same runtime can encode a prompt and generate a response through the
  normal `CpuModelTextGenerationService`
- cold and warm request states are visible in retained provenance
- the served response can carry explicit learned-lane evidence and claim
  posture instead of implying exact execution, benchmark proof, or hidden tool
  usage

This is enough to make the generic family operationally real.

It is not enough to claim:

- full compact-decoder pilot-anchor checkpoint parity
- executor-lane closure
- source-grounded citation truth
- verification-backed output

Those remain separate lane-specific contracts.

## Retained Packet Truth

The retained packet fixture is
`fixtures/psion/serve/psion_generic_load_and_generate_v1.json`.

Current retained facts:

- packet digest:
  `591b281a0ad9e944af8d0962d60736b4e8433b960460adbc2ea3d8379b6a3033`
- runtime path: `psionic_local_runtime.cpu_model_text_generation_service`
- served artifact digest:
  `92620238f04e8846f90fff146948a9a8d973148510528f3fa2f8e80bd72b5218`
- execution-plan digest:
  `d661f0ba2e2941f5111482c29037d09f10efc9ab23e4d47d35da96e6c83ca034`
- prompt text: `open agents`
- encoded prompt tokens: `[1, 5, 6]`
- cold response load state: `cold`
- warm response load state: `warm`
- evidence-bundle digest:
  `bae27fc7b20a34b38d698aab96822ed6b42aed24e74fc1e3d9b94a393bed298c`
- claim-posture digest:
  `08418ba4e47184f4c4b365cf0f8f8f98ebc39662ab2350dfe4e242302b029ed7`

The retained response is intentionally minimal. The point is to prove honest
load-to-serve closure on the current generic substrate, not to maximize output
quality with a toy decoder.

## Claim Boundary

The retained packet uses the same public claim discipline already defined by:

- `docs/PSION_SERVED_EVIDENCE.md`
- `docs/PSION_SERVED_OUTPUT_CLAIMS.md`

The packet is direct learned-lane output only.

It explicitly carries:

- learned judgment visibility: `true`
- source grounding visibility: `false`
- executor backing visibility: `false`
- benchmark backing visibility: `false`
- verification visibility: `false`

It also makes two assumptions explicit:

- the packet proves the current artifact-backed generic decoder runtime seam,
  not full compact-decoder-anchor parity
- no hidden tool, executor, or fresh external state was invoked

## Why This Matters For The Umbrella Program

The executor-capable `Psion` / `Tassadar` lane already has bounded replacement
and promotion packets.

The generic family needed one corresponding integrity closeout: evidence that
the repo can load a generic learned artifact and surface it through the normal
serve seams with explicit claim posture.

This packet is that closeout. It keeps the umbrella family honest without
pretending that the generic lane already carries the executor-lane exactness or
replacement bar.
