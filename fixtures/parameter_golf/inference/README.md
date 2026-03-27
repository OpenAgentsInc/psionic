# Promoted PGOLF Golden Prompt Suite

`parameter_golf_promoted_golden_prompts.json` is the checked-in conformance suite for the first promoted PGOLF-shaped Psion small-decoder family.

Claim boundary:
- prompts are evaluated against the repo-owned promoted general profile, not the strict challenge overlay
- `expected_text` is the declared local inference output for the emitted promoted bundle under the listed decode mode, token budget, and optional seed
- the suite is used to check parity across training-side restore, the public runtime bundle loader, the local generation path, and `psionic-serve`

Operator note:
- `cargo run -p psionic-train --example parameter_golf_promoted_prompt -- <bundle_dir>` now fails closed if any case drifts from the checked-in expected output
- `cargo run -p psionic-serve --example parameter_golf_promoted_operator -- inspect <bundle_dir>` reports the current manifest and artifact inventory
- `cargo run -p psionic-serve --example parameter_golf_promoted_operator -- validate <bundle_dir> --assume bundle` emits the typed inference-promotion receipt and exits non-zero if the bundle is refused
- `cargo run -p psionic-serve --example parameter_golf_promoted_operator -- prompt <bundle_dir> --prompt "abcd"` runs local prompt inference only after the promotion receipt is green
- `cargo run -p psionic-serve --example parameter_golf_promoted_operator -- warm <bundle_dir>` loads the same promoted bundle through `psionic-serve`, creates one session, and runs one tiny warm/smoke generation
