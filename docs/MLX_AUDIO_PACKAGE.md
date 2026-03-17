# Psionic MLX Audio Package

This document defines the bounded `psionic-mlx-audio` package that closes
`PMLX-705`.

## Scope

`psionic-mlx-audio` is the MLX-facing reusable audio package above the shared
Psionic runtime and artifact layers.

It owns:

- audio-family registry and quantized-checkpoint metadata
- WAV IO and bounded codec helpers
- text-to-speech and speech-to-speech request contracts
- streaming audio chunk contracts
- server-facing speech request/response shapes
- package-facing CLI entrypoints for synthesize, speech-to-speech, and WAV
  inspection

It does not own:

- product voice UX
- a claim of human-quality TTS or speech translation
- a browser/mobile audio player shell

## Current Truth

The current package closes the MLX audio ecosystem gap with one honest
CPU-reference runtime.

That means:

- text-to-speech requests render deterministic waveform clips
- speech-to-speech requests apply a bounded reference transform to the input
  waveform
- codec mode normalizes clips and owns the WAV/container contract
- streaming output is surfaced as explicit chunk lists
- server-facing speech requests can be handled through the same reference lane

This is a contract and packaging closure, not a claim that Psionic already
ships a production neural speech stack.

## Builtin Families

The builtin registry currently covers:

- `kokoro` for bounded text-to-speech
- `xtts` / `xtts_v2` for bounded text-to-speech plus speech-to-speech
- `encodec` / `codec` for bounded codec helpers

Each family exposes explicit supported tasks, conditioning modes, and
quantized-checkpoint descriptors such as `q4_k`, `q6_k`, and `q8_0`.

## Conditioning

The package keeps conditioning posture explicit:

- `none`
- `voice_label`
- `reference_audio`

If a family does not support one conditioning mode, the request must fail
explicitly.

## Server Surface

`MlxAudioSpeechRequest` and `MlxAudioSpeechResponse` define the server-facing
speech contract for this package.

The current reference lane can answer those requests directly and surface:

- output content type
- clip digest
- output clip metadata
- optional stream chunks

## CLI

The package CLI is:

- `psionic-mlx-audio synthesize`
- `psionic-mlx-audio speech-to-speech`
- `psionic-mlx-audio inspect-wav`
- `psionic-mlx-audio speech-request`
