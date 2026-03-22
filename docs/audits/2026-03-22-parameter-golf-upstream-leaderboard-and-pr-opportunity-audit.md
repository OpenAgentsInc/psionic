# Parameter Golf Upstream Leaderboard And PR Opportunity Audit

> Status: comprehensive audit of the upstream `openai/parameter-golf` README,
> leaderboard, selected merged and open pull requests, and the current local
> Psionic Parameter Golf posture, written 2026-03-22 after reviewing the
> upstream repo state visible on 2026-03-22 and the local Psionic PGOLF docs.

## Why This Audit Exists

On 2026-03-22, we decided to treat Parameter Golf as an actively useful
discipline again for Psionic and `Psion`, rather than only as a paused
historical lane.

This audit exists to answer four questions in one place:

1. what the upstream repo currently says the accepted leaderboard is
2. what the open pull-request frontier appears to be already claiming
3. which technique clusters look real versus rule-risky
4. what that means for an honest Psionic re-entry plan

## Sources Reviewed

### Upstream `openai/parameter-golf`

- repo README on `main` as rendered on GitHub on 2026-03-22
- record README linked from the merged leaderboard entry
  `2026-03-19_SlidingWindowEval`
- PR `#124` `Fix: score final partial window in sliding window eval`
- PR `#153` `Add strong-submission eval pipeline and ablation tooling`
- PR `#180` `Record: 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04`
- PR `#398` `Record: 11L EMA + TTT(20ep,freeze=0)`
- PR `#414` `Record: 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15`
- PR `#441` `Add BigramHash: hashed bigram embeddings with optional dim projection`
- PR `#442` `Record: 11L EMA + AdamW TTT 10ep`
- PR `#447` `Bigram-Aware Context Modeling with Mixed-Precision Quantization`
- issue `#402` `Invalid submissions due to information leakage during TTT`

### Local Psionic docs

- `docs/ROADMAP_PARAMETERGOLF.md`
- `docs/PARAMETER_GOLF_ACCOUNTING.md`
- `docs/PARAMETER_GOLF_ACCEPTANCE_MATRIX.md`
- `docs/PARAMETER_GOLF_AFTER_ACTION.md`
- local `~/code/parameter-golf/README.md`

## Executive Summary

The upstream accepted leaderboard has moved very fast since the 2026-03-18
baseline, and the accepted merged top score visible in the README on
2026-03-22 is now `1.1428 val_bpb`, not `1.2244`.

However, the merged leaderboard is no longer the whole story. The upstream
README explicitly says the leaderboard may lag review and verification, and
the current state of the open pull requests shows a much lower claimed
frontier already:

- open PR `#442` claims `1.1027` mean `val_bpb`
- open PR `#414` claims `1.1233`
- open PR `#398` claims `1.1213` best seed and `1.1221` mean
- open PR `#447` claims `1.1431`

The most important qualitative read is:

- the accepted merged leaderboard is currently dominated by mixed quantization,
  more layers, bigger MLPs, hashed context features, SmearGate-style control,
  and sliding-window evaluation
- the open-PR frontier appears to be dominated by aggressive
  test-time-training-style evaluation, EMA, and further quantization tricks
- the single biggest rules risk in the current frontier is that upstream issue
  `#402` argues several of the strongest TTT submissions are invalid because
  they adapt on evaluation tokens before those tokens are scored

That leads to one practical conclusion for Psionic:

- the fastest honest path back into this competition is not "copy the most
  aggressive open TTT PR immediately"
- the fastest honest path is to re-enter with the strongest merged-safe ideas
  first, while building a stricter receipt-backed causal TTT evaluation lane
  only as a later tranche

## Current Upstream README Facts

As rendered on GitHub on 2026-03-22, the upstream README says:

- the challenge target is still a `16,000,000` byte artifact trained in under
  `10` minutes on `8xH100` and scored by FineWeb validation `val_bpb`
- evaluation may use aggressive evaluation methods, any sequence length, and
  additional evaluation time, but may not train on validation tokens before
  those tokens have been evaluated
- leaderboard updates may lag because submissions are public and accepted
  chronologically under review

The merged leaderboard visible on 2026-03-22 shows this top accepted band:

| Rank | Score | Entry | Date | Main visible ingredients |
| --- | --- | --- | --- | --- |
| 1 | `1.1428` | `10L Int5-MLP + BigramHash(10240)` | 2026-03-20 | 10 layers, mixed int5/int6 quantization, `BigramHash(10240)`, SWA, weight decay |
| 2 | `1.1458` | `Int6 MLP3x + SmearGate + BigramHash` | 2026-03-20 | 3x MLP, SmearGate, BigramHash, OrthoInit, Muon WD, SWA |
| 3 | `1.1502` | `11L MLP3x + Int6 QAT` | 2026-03-20 | 11 layers, wider MLP, int6 QAT, sliding eval |
| 4 | `1.1556` | `SmearGate + OrthoInit + Muon WD` | 2026-03-19 | SmearGate, BigramHash, 3x MLP, int6 STE QAT, sliding eval |
| 5 | `1.1586` | `10L Int6 QAT + Zstd MLP2.6x` | 2026-03-19 | 10 layers, int6 QAT, zstd-22, bigger MLP, Muon |
| 6 | `1.1630` | `Mixed Quant + Sliding Window Eval` | 2026-03-19 | mixed quantization plus sliding-window eval |
| 7 | `1.1748` | `Muon WD + 10 layer` | 2026-03-19 | 10 layers plus optimizer and residual-control tuning |
| 8 | `1.1925` | `Sliding Window Eval` | 2026-03-19 | evaluation-only context improvement |
| 9 | `1.1928` | `Lora TTT` | 2026-03-19 | test-time training with LoRAs |
| 10 | `1.2014` | `4k seq length` | 2026-03-19 | longer train and eval sequence length |

Two upstream README details matter a lot for strategy:

1. sliding-window or longer-context evaluation is explicitly inside the allowed
   envelope
2. validation leakage is not, and the README now says that test-time training
   is only allowed on validation tokens that have already been graded

## What The PR Frontier Is Actually Doing

The open PR frontier has moved past the visible merged leaderboard.

### Frontier Pattern 1: TTT is driving the lowest claimed scores

The strongest visible open claim I reviewed is PR `#442`, which reports:

- `1.1027` mean `val_bpb`
- `11` layers
- EMA
- AdamW
- TTT for `10` epochs

PR `#398` also claims a strong TTT result:

- `1.1213` best seed
- `1.1221` three-seed mean
- `11` layers
- EMA
- aggressive TTT with all blocks unfrozen

This looks like a real frontier direction in practice, but it is also the most
contested one. Issue `#402` explicitly calls out PRs `#398`, `#442`, and
several related submissions as potentially invalid if they adapt on all eval
tokens before the scoring pass.

This is not a minor paperwork issue. If the critique is right, then the
current numerically strongest frontier is at least partly inflated by invalid
evaluation procedure.

### Frontier Pattern 2: the safest non-TTT wins combine compression plus more model

The accepted leaderboard and the safer open PRs repeatedly converge on the
same recipe:

- mixed precision or mixed-bit quantization
- stronger post-train compression
- more layers, often `10` or `11`
- bigger MLPs, commonly around `2.6x` to `3x`
- hashed context features such as `BigramHash`
- SmearGate or similar residual/control tricks
- sliding-window evaluation
- Muon weight decay, EMA, SWA, and warmdown tuning

PR `#414` is a good example of the safer high-end non-TTT or lower-risk band.
It claims `1.1233` with:

- `11` layers, `512d`, `8H/4KV`
- `MLP 3x`
- `SmearGate`
- `BigramHash(2048)`
- mixed quantization via `GPTQ-lite`
- EMA and warmdown tuning
- sliding-window evaluation

PR `#447` pushes in a similar direction with a direct bigram-aware context
modeling story and a claimed `1.1431`, which is close to the current merged
top without relying on the same level of contested TTT behavior.

### Frontier Pattern 3: evaluation engineering is a first-class lever

The leaderboard improvement from `1.2244` baseline to the `1.19x` band was not
initially architecture-only. PR `#124` fixed a real bug in sliding-window
evaluation where the final short window could be dropped, and the
`SlidingWindowEval` record README explains the core idea clearly:

- score each token exactly once
- give almost every token near-maximal left context
- get a real metric gain from evaluation procedure alone

This is not a gimmick. The upstream README explicitly allows evaluation at any
sequence length and encourages aggressive evaluation methods, which means
evaluation engineering is now part of the competition.

### Frontier Pattern 4: the community is copying strong public deltas extremely quickly

The conversation around PR `#398` shows immediate derivative work by other
participants. Multiple follow-on commits in outside forks reference it
directly, use it as a scaffold, or claim small deltas on top of it.

That means:

- the upstream frontier is not slow and isolated
- once a strong idea becomes public, it diffuses almost immediately
- the value of a new idea is highest before it is fully normalized into the
  public stack

Psionic should assume any effective public trick has a very short novelty half
life.

### Frontier Pattern 5: maintainers are open to core-code quality-of-life changes

PR `#153` proposes:

- `FINAL_EVAL_MODE=standard|sliding|ttt`
- configurable sliding eval
- decoupled Muon weight decay
- export-control flags
- ablation tooling

The upstream README also explicitly says PRs on `train_gpt.py` and
`train_gpt_mlx.py` are acceptable if they improve the starter scripts without
significantly increasing complexity.

This matters because the upstream competition is not purely "submit a folder
and never touch core." There is a secondary opportunity to upstream small,
legible improvements to the common baseline surface, though that is not the
primary scoreboard path.

## Opportunity Map For Psionic

### High-confidence opportunities

These look worth pursuing quickly because they align with accepted or clearly
allowed upstream practice and with the current local Psionic substrate.

#### 1. Resume with the best merged-safe stack before chasing contested TTT

The highest-confidence competitive recipe today appears to be:

- `10` to `11` layers
- larger MLP (`2.6x` to `3x`)
- mixed int5 or int6 plus stronger post-train compression
- sliding-window evaluation
- `BigramHash`
- SmearGate-style residual/control features
- Muon weight decay
- EMA or SWA
- warmdown tuning

This is the right first tranche because every major ingredient in that list is
already represented in either the merged leaderboard or clearly allowed eval
rules.

#### 2. Treat `BigramHash` and similar cheap context features as first-tier work

The leaderboard now repeatedly mentions `BigramHash`, and PR `#441` proposes a
core reusable implementation of hashed bigram embeddings with optional
projection.

That suggests a real pattern:

- compact models are getting useful extra local-context signal without paying
  the full dense-parameter cost
- this is exactly the kind of trick a Psionic compact decoder should be able
  to explore cleanly

For Psionic, this is especially attractive because it is a model-surface
change, not a full runtime philosophy shift.

#### 3. Mixed quantization and better clip selection are now central, not optional

The accepted and open frontier both lean heavily on:

- mixed int5/int6 or int6/int8 allocations
- better clip calibration
- zstd or similar stronger post-roundtrip compression

PR `#414`'s `GPTQ-lite` per-row clip search is especially noteworthy because it
claims a measurable gain with no extra training cost.

This is a strong fit for Psionic because our local PGOLF lane already invested
in artifact accounting, roundtrip validation, and quantized export posture.

#### 4. Evaluation infrastructure should be treated as competitive infrastructure

Sliding-window evaluation is now upstream truth, not just an experiment, and
the README makes it explicit that evaluation engineering is allowed if it does
not violate leakage rules.

Psionic should therefore treat the evaluation path as a first-class system:

- exact upstream-compatible sliding evaluation
- exact causal scoring semantics
- explicit receipts for which tokens were scored and when
- explicit proof that no future validation tokens were consumed before scoring

That last point is what lets us later explore TTT without entering the same
rule ambiguity as the upstream contested PRs.

### Medium-confidence opportunities

These look promising but need more evidence or more implementation work.

#### 5. Moderate architecture widening remains alive

The accepted leaderboard repeatedly improves by paying compression gains back
into slightly larger dense models:

- add a layer
- widen the MLP
- add cheap local-context augmentation
- keep the artifact under the cap anyway

This is strategically important. Parameter Golf is not behaving like a pure
"make the tiniest possible model" contest. It is behaving more like:

- compress hard enough to buy back extra useful structure

That maps well onto a Psionic-owned lane because we already built the
artifact-accounting contract instead of pretending runtime or export bytes are
free.

#### 6. Strictly causal TTT may still be worth it later

The open frontier is too strong to ignore forever. If the rules settle in a
way that still allows properly causal TTT, then it may eventually be required
to compete for the true frontier.

But there is one strict condition:

- Psionic should only enter that lane with a receipt-backed per-token
  score-then-adapt evaluation loop that can survive adversarial review

Anything weaker risks building a fast local win on top of semantics that later
get disallowed.

### Low-confidence or clearly risky opportunities

#### 7. Copying the strongest open TTT PRs as-is is not an honest first move

Issue `#402` is too direct to ignore. The upstream language now explicitly says
participants may only test-time train on validation tokens they have already
evaluated.

That means a naive "port PR `#442` or `#398` into Psionic" plan is not the
right opening move, even if those numbers are currently the most exciting.

#### 8. Sparse or restricted-attention leaps should stay evidence-gated

The upstream PR stream includes block-sparse and exotic attention proposals,
but the safer current accepted wins are not dominated by those ideas.

This matches the local Psionic historical evidence too: our own PGOLF lane
already recorded negative evidence for one fixed restricted-attention window.

That does not mean "never explore locality." It means:

- do not let locality become the restart story before stronger merged-safe
  deltas have been harvested

## What This Means For The Local Psionic PGOLF Lane

The local Psionic Parameter Golf posture before this restart decision was:

- `challenge-oracle-parity`: implemented
- `single-device-trainer-parity`: implemented_early
- `distributed-throughput-closure`: partial
- `packaging-readiness`: implemented
- `record-track-readiness`: partial_outside_psionic

The lane stopped on 2026-03-19, but the key substrate that remains useful is:

- exact challenge-oracle parity
- compact decoder graph work
- non-record package and compatibility verification
- bounded CUDA and runtime evidence
- H100 bring-up seams

The local after-action also said the next concrete runtime move was:

1. restore bounded H100 trace workflow
2. reduce host-materialized view-op cost
3. finish the actual Rust-only single-H100 baseline trainer path
4. only then spend on real exported-folder `8xH100` evidence

That restart ordering still looks correct today.

The only thing that changes after reviewing the upstream repo is priority:

- we now have better visibility into which model and eval deltas are most worth
  implementing once the trainer path is alive again

For naming discipline, "we want our `Psion` model to compete" should not be
interpreted as "take the current broad `Psion` learned-model lane and submit it
unchanged." In practice it should mean:

- use the Psionic and `Psion` training substrate
- keep the honest PGOLF-specific compact decoder and export lane
- compete with a compact challenge-shaped model family rather than with the
  broader general-purpose served `Psion` posture

## Recommended Psionic Re-Entry Plan

### Phase A: reopen the lane on the strongest merged-safe recipe

Reopen PGOLF with a rules-safe target stack:

- exact upstream-correct sliding-window evaluation
- `10` to `11` layer compact decoder variants
- larger MLP band around `2.6x` to `3x`
- `BigramHash`-style context augmentation
- SmearGate-like residual or control experiments
- mixed int5/int6 quantization plus stronger post-train compression
- Muon weight decay, EMA or SWA, and warmdown sweeps

This phase should aim first at:

- an honest `non_record_submission` upgrade over the old local baseline
- then a credible non-TTT record-candidate posture

### Phase B: resume the systems path that the local after-action already identified

Use the existing Psionic H100 evidence as the restart anchor:

- remove view-op overhead in the CUDA path
- finish the real Rust-native single-H100 trainer loop
- then lift that into `8xH100`

This is the prerequisite for any serious scoreboard attempt.

### Phase C: build a strict-causal TTT lane only after the rules are machine-checkable

If we pursue TTT, the lane should require:

- per-token score receipts
- per-token adapt receipts
- proof that token `t` was scored before any adaptation step that used token
  `t`
- replay verification over the evaluation trace

That would turn the current upstream rules ambiguity into a Psionic advantage,
because we could defend the semantics instead of arguing them informally in PR
threads.

## Concrete Immediate Opportunities

If we restart today, the best immediate bets appear to be:

1. implement upstream-compatible sliding-window eval, including the corrected
   tail-window behavior from PR `#124`
2. add `BigramHash` or closely related hashed local-context features to the
   compact decoder family
3. add mixed int5/int6 export plus better clip-search calibration
4. sweep `10L` and `11L` models with `MLP 2.6x` to `3x`
5. add EMA and decoupled Muon weight decay support if our local lane does not
   already expose the equivalent knobs
6. re-run the single-H100 path only after view-op overhead is reduced
7. hold TTT behind an explicit semantics gate instead of mixing it into the
   first restart tranche

## Final Assessment

The upstream contest has moved from "port the baseline into Rust" to a much
sharper game:

- compress aggressively
- spend those bytes on slightly bigger or better-structured small models
- exploit evaluation-time context legally
- use public ablations to ratchet tiny deltas very quickly

Psionic is not starting from zero. The paused local lane already owns the hard
contracts around oracle parity, packaging, and honest accounting. The biggest
gap is no longer conceptual. It is:

- getting the real Rust-native trainer path back onto competitive hardware
- then harvesting the strongest merged-safe architecture and quantization
  deltas before the public frontier moves again

If we want an honest competitive return, the right stance is:

- copy the merged-safe frontier now
- instrument the evaluation semantics more rigorously than upstream
- treat TTT as a later audited weapon, not the first story we tell ourselves

That would let Psionic compete on both discipline and score instead of only on
one of them.
