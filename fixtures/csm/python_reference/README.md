# CSM Python Reference Fixture

This directory freezes the first compact CSM parity corpus for the Psionic CSM
runtime lane.

Source reference:

- local repo: `/Users/christopherdavid/code/csm`
- command that proved the demo: `NO_TORCH_COMPILE=1 .venv/bin/python run_csm.py`
- root audit:
  `/Users/christopherdavid/work/docs/2026-05-06-csm-rust-lyra-psionic-audit.md`

The fixture records small, reviewable values derived from the Python reference:

- prompt WAV hashes and metadata for `conversational_a` and
  `conversational_b`
- Llama tokenizer outputs for two short speaker-prefixed utterances
- 33-lane text frame and mask examples
- full precomputed Mimi prompt codebooks for the two governed source prompts
- a three-frame greedy CSM generation prefix for one short utterance

The fixture does not contain Hugging Face tokens, provider keys, full prompt
audio, or full model weights. It now includes full prompt codebooks so the Rust
served path can use `prompt_profile_only` context and keep the governed
OpenAgents voice profile stable without calling Python at request time.
