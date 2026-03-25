# Parameter Golf 8xH100 Score-Path Audit

Date: 2026-03-25

## Contract

The upstream Parameter Golf contract is explicit:

- record-track training must complete in under `600` seconds on `8xH100`
- record-track evaluation has a separate `600` second budget
- the public score is FineWeb validation `bits_per_byte`

The current public scoreboard in [`~/code/parameter-golf/records/track_10min_16mb/`](../../../parameter-golf/records/track_10min_16mb) is already below `1.13 val_bpb`, with the best local record surface at the time of this audit showing `1.119400` for `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`.

Psionic is not close to that score path yet. The current gap is not only model quality. The current `8xH100` execution topology is too slow to produce a valid scoreboard-grade run.

## Current Evidence

### 1. Operator and input binding are now real

Retained RunPod proof root:

- `/workspace/parameter-golf-runpod/parameter-golf-runpod-scoreproof-20260325T091348Z`

What this run proves:

- `pre_training` completed successfully
- immutable PGOLF inputs were bound correctly from the retained materialization report
- the exported-folder `distributed_8xh100_train` mode crossed the old missing-env/operator failures
- the shipped runtime entered the real distributed benchmark root at:
  - `.../parameter-golf-distributed-8xh100-run/benchmark/`

This closes the old operator/input gap. That work now belongs in the closed issue `#544`.

### 2. The current distributed proof topology is still one-step and still slow

Retained rank train-step receipts from the same proof root:

- `.../benchmark/runtime_train_step_receipts/rank_0.json`
- `.../benchmark/runtime_train_step_receipts/rank_1.json`
- `.../benchmark/runtime_train_step_receipts/rank_2.json`
- `.../benchmark/runtime_train_step_receipts/rank_3.json`
- `.../benchmark/runtime_train_step_receipts/rank_4.json`
- `.../benchmark/runtime_train_step_receipts/rank_5.json`
- `.../benchmark/runtime_train_step_receipts/rank_6.json`
- `.../benchmark/runtime_train_step_receipts/rank_7.json`

Observed one-step rank-local wallclock:

- rank `0`: `229019 ms`
- rank `1`: `217156 ms`
- rank `2`: `220134 ms`
- rank `3`: `220256 ms`
- rank `4`: `227089 ms`
- rank `5`: `220245 ms`
- rank `6`: `226251 ms`
- rank `7`: `222638 ms`

Observed rank-local phase timings are dominated by forward and backward:

- forward: about `93s` to `99s`
- backward: about `118s` to `128s`
- token materialization: about `0.19s` to `0.25s`
- host gradient materialization: about `0.42s` to `0.74s`

The retained aggregated gradient artifact is:

- `.../benchmark/runtime_train_step_gradients/aggregated_step_1.safetensors`
- size: about `66 MB`

This is still a one-step proof topology. It is not a real 600-second training run.

### 3. The current proof topology still uses expensive orchestration

The current `8xH100` runtime path still does all of the following in the hot path:

- parent process spawns `8` runtime child processes for the train step
- parent writes per-rank window JSON before the step
- each child exports one per-rank gradient `safetensors` artifact
- parent waits for all children, then reopens every gradient artifact on disk
- parent aggregates gradients on host
- parent applies the optimizer step on host
- parent later spawns `8` more runtime child processes for validation

This is the wrong topology for scoreboard-grade throughput.

It is useful as a retained proof lane. It is not a viable steady-state record lane.

### 4. Prior retained validation evidence was already too slow

The earlier real distributed validation proof documented on `#510` had only reached roughly `30..36 / 119` validation batches per rank after about `27..28` minutes.

That earlier evidence matters because it shows the validation path is also not just “missing a few receipts.” The old distributed validation path was materially too slow for the `600` second scoreboard budget.

## Structural Gaps

### Gap A: spawn-per-step worker lifecycle

The current distributed runtime launches fresh child processes per step and again per validation pass.

That forces:

- process startup overhead
- fresh CUDA runtime setup
- fresh model reconstruction in every child
- repeated benchmark-root file IO in the hot path

The scoreboard lane needs persistent rank workers, not repeated process fanout.

### Gap B: file-artifact gradient synchronization

The current runtime still synchronizes one step through per-rank gradient files plus one parent-side aggregation pass.

That means the hot path still depends on:

- writing eight gradient artifacts
- reopening eight gradient artifacts
- host-side aggregation
- only then moving to the next step

This is not comparable to the upstream `torch.distributed` all-reduce posture.

### Gap C: no real `600` second repeated training loop

The current `8xH100` runtime still executes one explicit step and then heads toward validation.

That does not match the upstream loop in `train_gpt.py`, which:

- keeps model and optimizer state resident
- advances steps until the wallclock cap is reached
- then runs final roundtrip validation

Without that repeated loop, Psionic does not own a valid scoreboard-grade training run.

### Gap D: final artifact semantics are still wrong for real training

The exported-folder completion path still binds the static shipped `final_model.int8.ptz` from the package manifest.

That is not sufficient for a real training run. A real score path must:

- export the trained final artifact from the live runtime
- bind the completion receipt to that exact trained artifact digest and size
- evaluate the trained roundtrip artifact, not the stale packaged fixture artifact

### Gap E: validation still needs the same persistent-runtime treatment

Even after sliding-window scoring landed, the distributed validation path still uses spawned children and full per-child model reconstruction.

That is not viable for the `600` second eval budget if it remains the final design.

## What This Means

Two conclusions are now defensible:

1. The current operator lane is no longer the main blocker.
2. A simple `while elapsed < 600s` wrapper around the current one-step proof will not produce a solid score.

The next score-path work must change the runtime topology, not just the receipt surface.

## Required Path Forward

### Priority 1: persistent distributed worker mesh

Replace the spawn-per-step proof lane with one persistent `WORLD_SIZE=8` runtime mesh that:

- launches the rank workers once
- keeps model and optimizer state resident across steps
- keeps step-local graph state warm
- exposes one coordinator path for stop conditions and finalization

### Priority 2: in-memory gradient synchronization

Replace per-rank gradient artifact export in the hot path with a real distributed synchronization path:

- collective or equivalent in-memory reduction
- no per-step gradient `safetensors` roundtrip in the score lane
- no parent-side file reopen and host reduction in the score lane

### Priority 3: real wallclock-capped repeated training loop

Once the persistent mesh exists, implement the actual public challenge loop:

- repeated steps under `MAX_WALLCLOCK_SECONDS=600`
- learning-rate / warmdown behavior tied to elapsed wallclock
- final stop when the cap is reached
- measured repeated-step receipt surface instead of one-step-only proof receipts

### Priority 4: trained final artifact closure

The distributed runtime must export the real trained final artifact from the live run and bind:

- final artifact path
- final artifact digest
- final artifact size
- final distributed validation metrics

to one honest completion receipt.

### Priority 5: scoreboard-grade validation and later TTT

Only after the runtime topology is fast enough should Psionic spend serious effort on:

- sliding-window scoreboard tuning
- legal score-first TTT
- model-architecture and schedule improvements aimed at matching the public top entries

Those improvements matter for score quality. They do not remove the current execution bottleneck.

## Issue Mapping

The current open issue stack only partially captures this.

Existing relevant issues:

- `#510`: real distributed validation and aggregation
- `#512`: exported-folder completion and receipt closure
- `#541`: scoreboard-grade sliding-window evaluation
- `#542`: legal score-first TTT
- `#543`: real `600` second multi-step `8xH100` loop

A new issue is required for the missing structural gap:

- persistent distributed worker mesh plus in-memory gradient synchronization replacing the current spawn-per-step proof topology

## Bottom Line

Psionic now has a real RunPod `8xH100` operator lane and a real one-step distributed proof path.

Psionic does not yet have a scoreboard-grade `8xH100` execution topology.

The next serious work is not another operator fix. It is replacing the proof topology with a persistent distributed trainer that can actually use the `600` second training budget efficiently.
