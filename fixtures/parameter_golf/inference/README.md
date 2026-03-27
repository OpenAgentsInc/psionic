# Promoted PGOLF Golden Prompt Suite

`parameter_golf_promoted_golden_prompts.json` is the checked-in conformance suite for the first promoted PGOLF-shaped Psion small-decoder family.

Claim boundary:
- prompts are evaluated against the repo-owned promoted general profile, not the strict challenge overlay
- `expected_text` is the declared local inference output for the emitted promoted bundle under the listed decode mode, token budget, and optional seed
- the suite is used to check parity across training-side restore, the public runtime bundle loader, the local generation path, and `psionic-serve`

Operator note:
- `cargo run -p psionic-train --example parameter_golf_promoted_prompt -- <bundle_dir>` now fails closed if any case drifts from the checked-in expected output
