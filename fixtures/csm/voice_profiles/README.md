# CSM Voice Profiles

This directory contains the machine-readable governance layer for CSM voice
profiles admitted by the Psionic Rust speech server.

- `lyra_voice_profiles.schema.json` defines the manifest shape.
- `lyra_voice_profiles.v1.json` is the current committed manifest.

The first profile is `lyra/default_female_v1`. It is an internal Lyra
placeholder mapped to the committed CSM parity prompt `conversational_a`; it is
not arbitrary user reference audio, and it is not production voice-cloning
permission.

Watermarking is currently `unsupported_fail_closed`. Public demo keys are not
production safety controls. CSM output remains blocked for production Lyra
cutover until a private watermark or equivalent voice-safety control exists.
