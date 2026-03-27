# HOMEGOLF Fresh Rerun Results Audit

This audit records one fresh HOMEGOLF rerun on 2026-03-27 and compares it
against the currently committed HOMEGOLF and public Parameter Golf surfaces.

Historical note after `HOMEGOLF-8`: this rerun remains a valid exact-family
bundle proof, but it is no longer the canonical runnable HOMEGOLF contest-lane
surface. The canonical runnable lane is now the strict preflight surface at:

- `crates/psionic-train/src/bin/parameter_golf_homegolf_strict_challenge_lane.rs`
- `fixtures/parameter_golf/reports/parameter_golf_homegolf_strict_challenge_lane.json`

## What Ran

Command:

```bash
cargo run -q -p psionic-serve --example parameter_golf_homegolf_dense_bundle_proof -- \
  /tmp/psionic_homegolf_run_1774647888
```

This is the current honest runnable HOMEGOLF path:

- exact `parameter-golf-sp1024-9x512` family contract
- real bounded Psionic training run
- emitted promoted runtime bundle
- direct inference from the emitted bundle
- served inference through `psionic-serve`

It is not yet the full public contest scorepath. The emitted training config
still uses:

- `evaluation_identity = local_reference_validation`
- `contest_bits_per_byte_accounting_required = false`
- `exact_compressed_artifact_cap_required = false`

So this is a real HOMEGOLF rerun, but still a bounded exact-family proof lane,
not a public-leaderboard-equivalent submission.

## Fresh Run Result

The fresh rerun emitted:

- report:
  `/tmp/psionic_homegolf_run_1774647888/parameter_golf_homegolf_dense_bundle_proof.json`
- bundle dir:
  `/tmp/psionic_homegolf_run_1774647888/bundle`

Key results:

- descriptor digest:
  `8a111f908acf02174554a75e83e13092852ee7caa534f65c43be897bd4c606ee`
- tokenizer digest:
  `49b44264442058c20b2b95a947f3aac60e8729fd57d63db8b8754de8edb98a6d`
- model artifact bytes:
  `68,248,296`
- initial validation mean loss:
  `8.611438751220703`
- final validation mean loss:
  `8.60598874092102`
- final validation bits per byte:
  `9.93265382277841`
- int8-zlib roundtrip validation mean loss:
  `8.598987579345703`
- prompt:
  `abcd`
- direct output:
  `gbgb`
- served output:
  `gbgb`
- direct/served parity:
  `true`

Training-shape details from the emitted bundle:

- `max_steps = 2`
- `train_batch_tokens = 32`
- `validation_batch_tokens = 32`
- `train_sequence_length = 4`
- `grad_accum_steps = 8`
- `max_wallclock_seconds = 600`

## Reproducibility Result

The fresh rerun matched the committed canonical HOMEGOLF proof fixture exactly:

```bash
cmp -s \
  /tmp/psionic_homegolf_run_1774647888/parameter_golf_homegolf_dense_bundle_proof.json \
  fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_bundle_proof.json
```

Result:

- exact byte-for-byte match

That means today's rerun did not drift from the currently committed HOMEGOLF
proof. The canonical tracked result remains:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_bundle_proof.json`

## Comparison To Public Parameter Golf

Using the frozen HOMEGOLF public-comparison surface:

- HOMEGOLF `val_bpb`:
  `9.93265382277841`
- public naive baseline `val_bpb`:
  `1.2243657`
- current public best `val_bpb`:
  `1.1194`

Current deltas:

- versus public naive baseline:
  `+8.70828812277841 val_bpb`
- versus current public best:
  `+8.813253822778409 val_bpb`

Artifact posture:

- fresh HOMEGOLF scored model bytes:
  `68,248,296`
- HOMEGOLF counted code bytes:
  `7,188,700`
- HOMEGOLF total counted bytes:
  `75,436,996`
- contest cap:
  `16,000,000`
- current cap delta:
  `+59,436,996`

So the current HOMEGOLF rerun is still:

- far worse than the public baseline on `val_bpb`
- far over the artifact cap
- not a public-leaderboard-equivalent run

## What This Run Actually Proves

This fresh rerun proves that Psionic can currently do all of these together for
the exact HOMEGOLF family:

- train a real bounded `9x512` model
- emit a promoted runtime bundle
- reload that bundle for inference
- serve that same bundle through `psionic-serve`
- reproduce the same retained result deterministically

That is real progress, but it is still the bounded proof lane, not the final
HOMEGOLF destination.

## What It Still Does Not Prove

This rerun does not prove:

- exact FineWeb `SP1024` public scorepath closure
- contest-required BPB accounting on the live rerun lane
- contest-cap artifact compliance
- one live mixed-device dense HOMEGOLF run producing the scored bundle
- one H100-backed HOMEGOLF run
- one competitive public-baseline result

## Current Read

The honest current status is:

- reproducibility is good
- train-to-infer closure is real
- serve parity is real
- competitiveness is still poor

The system is now reliable enough to iterate on, but the next work must improve
three things at once:

- quality
- artifact size
- exact contest-path fidelity

## Most Important Next Moves

1. Make the live HOMEGOLF lane use the exact contest evaluation/accounting
   policy instead of `local_reference_validation`.
2. Get one real mixed-device dense HOMEGOLF run to produce the scored bundle,
   instead of composing device receipts with a separate exact-family proof.
3. Reduce artifact size aggressively so the HOMEGOLF counted-byte report can
   move from refusal toward a real pass.
4. Improve model quality on the exact-family lane before spending time on public
   leaderboard rhetoric.

## Exact Competitive Closure Bar

For HOMEGOLF to count as a genuinely competitive contest-style run, all of the
following must be true at the same time:

- exact public model family or one explicit allowed public-competitive variant
- exact FineWeb `sp1024` data lane
- exact challenge tokenizer identity
- exact `600s` training budget
- exact challenge-style evaluation contract, including sliding-window scoring
  and legal score-first TTT when the profile requests it
- exact `16,000,000`-byte artifact accounting
- one produced model bundle from the live dense run itself
- one score that is at least baseline-competitive

Today HOMEGOLF misses every competitive bar except:

- the family shape
- the wallclock budget declaration
- reproducible train-to-infer closure

## Exact Gaps To Close

### Gap 1: HOMEGOLF is still running the wrong profile

The fresh rerun is still on the general local-reference profile, not the strict
challenge overlay.

Current emitted profile facts:

- `evaluation_identity = local_reference_validation`
- `contest_bits_per_byte_accounting_required = false`
- `exact_compressed_artifact_cap_required = false`

Psionic already has the strict contest profile contract in code, but the
current HOMEGOLF runnable lane does not use it.

Necessary fix:

- make the runnable HOMEGOLF lane execute the strict challenge overlay instead
  of refusing it
- bind the live run to exact public tokenizer, dataset, evaluation, and
  artifact policy

### Gap 2: HOMEGOLF is training on a tiny bounded proof, not contest volume

The fresh rerun trained with:

- `max_steps = 2`
- `train_batch_tokens = 32`
- `train_sequence_length = 4`

The public naive baseline used:

- `TRAIN_BATCH_TOKENS = 524288`
- `TRAIN_SEQ_LEN = 1024`
- about `13,780` steps in `600s`
- about `7,224,688,640` train tokens seen

So the current HOMEGOLF lane is not “slower but same contest.” It is a tiny
proof lane that happens to share the model family name.

Necessary fix:

- replace the bounded sparse proof updater with a true dense trainer
- keep model and optimizer state resident across the run
- feed contest-scale batches and sequence lengths
- consume the real FineWeb `sp1024` lane

### Gap 3: the scored HOMEGOLF bundle is still composed, not produced by the live home cluster

The current clustered HOMEGOLF score surface is honest, but it is still a
composed surface:

- home-device contribution receipts come from a real mixed-device run
- the exact-family score bundle comes from a separate bounded proof lane

That is useful for proof and comparison. It is not a competitive contest run.

Necessary fix:

- one live dense mixed-device HOMEGOLF run must directly emit the scored bundle
- the public-comparison and artifact-accounting reports must derive from that
  exact run, not a composed surrogate

### Gap 4: quality is nowhere near baseline, much less leaderboard relevance

Current HOMEGOLF rerun:

- `val_bpb = 9.93265382277841`

Public comparison points:

- naive baseline: `1.2243657`
- current public best: `1.1194`

So HOMEGOLF is currently:

- `+8.70828812277841` BPB worse than the public naive baseline
- `+8.813253822778409` BPB worse than the current public best

Necessary fix:

- first beat the naive baseline honestly
- then drive toward the `~1.12` band before claiming contest competitiveness

### Gap 5: artifact size is currently disqualifying

Current HOMEGOLF accounting:

- counted code bytes: `7,188,700`
- scored model bytes: `68,248,296`
- total counted bytes: `75,436,996`
- cap: `16,000,000`
- over cap by: `59,436,996`

This means total counted bytes must drop by about `78.8%`.

If code bytes stayed unchanged, the model would have to shrink from
`68,248,296` bytes down to `8,811,300` bytes, which is about an `87.1%`
reduction.

Necessary fix:

- shrink the counted runtime/code surface
- shrink the compressed model artifact much more aggressively
- make the export pipeline produce a real under-cap answer instead of an honest
  refusal

### Gap 6: the public winning tricks are only partially wired into HOMEGOLF

The public leaderboard is not winning only on “same model but better luck.”
The top records layer in techniques like:

- 10-11 layer competitive variants
- `BigramHash`
- partial RoPE
- XSA on deep layers
- `LeakyReLU(0.5)^2`
- EMA and tight SWA
- VE late-layer embeddings
- GPTQ-lite or QAT-backed export
- legal score-first TTT
- parameter banking plus parallel Muon

Important nuance:

- several of these surfaces already exist somewhere in Psionic
- they are not yet assembled into one exact competitive HOMEGOLF lane

Necessary fix:

- reuse existing Psionic competitive PGOLF surfaces where they already exist
- port the missing ones
- run ablations on the exact HOMEGOLF lane instead of leaving them as dormant
  capability flags or separate H100-only evidence

### Gap 7: throughput on the admitted home cluster still has to become score-relevant

HOMEGOLF is supposed to be “same contest, different cluster.”

That means the home-device lane needs:

- one real dense mixed-device execution topology
- measured per-device contribution
- enough throughput to make architecture work matter

Today the admitted-device results are useful for short-run adapter comparison,
but not yet for dense competitive HOMEGOLF training.

Necessary fix:

- make the dense train path run across M5 plus home CUDA plus optional M2/H100
- keep the wallclock under `600s`
- publish per-device throughput and contribution receipts from the dense lane

## What Is Already Available And Should Be Reused

The fastest route is not to invent another Parameter Golf stack.

Psionic already has useful pieces:

- the exact `parameter-golf-sp1024-9x512` family contract
- the strict challenge overlay contract
- the single-H100 trainer with challenge-data and score-first-TTT surfaces
- the distributed `8xH100` score-path work, including persistent-worker and
  Parallel-Muon runtime surfaces
- model-family support for `BigramHash`, XSA, `LeakyReLU(0.5)^2`, VE, and
  parameter banking
- EMA/SWA surfaces in the single-H100 trainer lane
- quantization, PTQ/QAT, and export-aware capability surfaces

So the real task is:

- stop proving that these pieces exist in isolation
- wire them together into one competitive HOMEGOLF lane

## Ordered Work Required

The exact order that makes sense is:

1. switch HOMEGOLF from the general local-reference profile to the strict
   challenge overlay
2. make one live dense mixed-device HOMEGOLF run produce the scored bundle
3. raise dense-lane throughput enough that `600s` runs are meaningful
4. port and ablate competitive public techniques on that exact lane
5. close the artifact cap
6. publish multi-seed public-comparison reports from the real lane

Doing this out of order will waste time.

For example:

- shrinking bytes before the real dense lane exists risks optimizing the wrong
  artifact
- adding more architecture tricks before the strict challenge path is live
  risks improving the wrong benchmark
- producing more public comparisons before baseline parity exists only creates
  clearer evidence of non-competitiveness

## Tracked Optimization Spine

The current GitHub issue spine for this work is:

- `#629` `HOMEGOLF-7: Track competitive HOMEGOLF closure`
- `#622` `HOMEGOLF-8: Switch the runnable HOMEGOLF lane onto the strict PGOLF challenge overlay`
- `#623` `HOMEGOLF-9: Make one live dense mixed-device HOMEGOLF run produce the scored bundle`
- `#624` `HOMEGOLF-10: Raise dense HOMEGOLF throughput on the admitted home cluster`
- `#625` `HOMEGOLF-11: Port and ablate competitive PGOLF techniques on the exact HOMEGOLF lane`
- `#626` `HOMEGOLF-12: Close the HOMEGOLF artifact-cap gap`
- `#627` `HOMEGOLF-13: Move optional H100 support into the live dense HOMEGOLF lane`
- `#628` `HOMEGOLF-14: Publish a multi-seed leaderboard-comparable HOMEGOLF package`
