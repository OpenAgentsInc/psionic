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
- compact Mimi codebook prefixes plus full prompt-codebook digests
- a three-frame greedy CSM generation prefix for one short utterance

The fixture does not contain Hugging Face tokens, provider keys, full prompt
audio, full model weights, or large codebook tensors. It is a parity target for
Rust implementation work, not a production Python runtime.

