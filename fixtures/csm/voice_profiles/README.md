# CSM Voice Profiles

This directory contains the machine-readable governance layer for CSM voice
profiles admitted by the Psionic Rust speech server.

- `lyra_voice_profiles.schema.json` defines the manifest shape. The filename is
  historical; the served profile id is now OpenAgents/Autopilot-owned.
- `lyra_voice_profiles.v1.json` is the current committed manifest.

The first profile is `openagents/default_female_v1`. It is an
OpenAgents-operated Autopilot dogfood profile mapped to the committed CSM
parity prompt `conversational_b`; it is not arbitrary user reference audio,
and it is not production voice-cloning permission.

Watermarking is currently `unsupported_operator_accepted_limited_dogfood`.
Public demo keys are not production safety controls. CSM output is admitted
only for bounded OpenAgents-operated Autopilot dogfood until a private
watermark or equivalent voice-safety control exists.
