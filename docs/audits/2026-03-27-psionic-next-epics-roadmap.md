# 2026-03-27 Psionic Next Epics Roadmap

This document lays out the next major epics in the order they should be
implemented if the goal is to turn `psionic` from a strong bounded prototype
into a system that can:

- train models that are actually useful
- run inference on those models in a real product path
- scale training across serious mixed Apple and NVIDIA compute
- let outside devices join training runs over the internet
- automatically promote successful runs into served production models

This is not a “random list of good ideas.” It is an ordered dependency plan.
The main question is:

> what has to happen first so later work is real instead of fake

## Current Truth

As of 2026-03-27, the repo already has several important things working:

- a bounded XTRAIN system with machine-readable contracts and retained evidence
- a bounded PGOLF-shaped train-to-infer path
- local inference on trained PGOLF-shaped models
- major tok/s improvements over the earlier inference path
- one real trusted-LAN mixed-hardware swarm run across Mac MLX and Linux CUDA
- one separate truthful local snapshot publication proof for the first swarm
  publish target

The best current references for that state are:

- `docs/audits/2026-03-26-xtrain-system-state-and-operations-audit.md`
- `docs/audits/2026-03-26-psion-trained-model-inference-gap-audit.md`
- `docs/audits/2026-03-26-promoted-pgolf-inference-proof.md`
- `docs/audits/2026-03-27-xtrain-pgolf-train-infer-iteration-audit.md`
- `docs/audits/2026-03-27-first-swarm-trusted-lan-real-run-audit.md`
- `docs/audits/2026-03-27-first-swarm-local-snapshot-publication-proof.md`

The honest gap is that the system is now real enough to prove plumbing, but not
yet real enough to claim strong model quality, broad decentralized training, or
automatic product promotion.

## The Big Missing Things

These are the missing outcomes we still do not have:

- the models are still tiny and not useful enough yet
- quality is improving, but not yet at “good product model” level
- the swarm system is still mostly a careful bounded proof, not a full open
  internet network
- any device cannot yet join and train honestly over the internet
- serious mixed Apple plus NVIDIA training at scale is not yet proven
- a successful live run does not yet automatically become a served production
  model

## Epic Order

The correct order is:

1. ship one genuinely useful small model family
2. harden the train-to-infer-to-serve promotion spine for that family
3. scale the training substrate from bounded local proof to serious mixed-backend training
4. turn the swarm substrate from trusted-LAN proof into elastic WAN training
5. open the network so outside devices can join honestly
6. add automatic run promotion into production serving

Some side work can happen in parallel, but this is the main dependency order.

## Epic 1: Useful Small Model Baseline

### Goal

Move from the current tiny proof models to one small language model family that
is actually useful for real text tasks.

### Why this comes first

If the model family itself is still toy-grade, then scaling, decentralization,
and automatic promotion mostly just make the wrong thing bigger and harder to
operate.

### Main work

- freeze one first serious small-model family
- choose the long-term tokenizer and bundle shape
- define the target eval set for “useful enough to matter”
- improve data quality, curriculum, and held-out evaluation
- improve optimization, checkpoint selection, and overfitting control
- keep the model family compatible with the PGOLF-shaped inference lane where
  that still helps

### Done when

- the repo has one named small-model family intended for real use, not just
  proof runs
- that model clearly beats the current toy/reference quality floor
- inference uses the same honest train artifact, not a separate fake runtime

## Epic 2: Product-Grade Train To Infer To Serve Spine

### Goal

Make the useful small model family move cleanly through the whole lifecycle:

- train
- checkpoint
- evaluate
- package
- publish
- serve

### Why this is second

Once we have one model family worth caring about, the next job is to make sure
it can actually become a real product asset without hand-wavy steps in the
middle.

### Main work

- finish the model bundle and tokenizer asset contract
- add stronger inference parity checks between direct runtime and serve runtime
- define promotion gates that combine quality, safety, and runtime checks
- connect published bundle artifacts to the serving stack
- add rollback and “do not promote” rules when a run regresses
- make “best checkpoint” and “servable candidate” machine-decidable

### Done when

- one training run can honestly produce one candidate bundle that the product
  serving path can load
- promotion into serving is no longer mostly human judgment and shell work
- every promoted model has retained evidence for why it was promoted

## Epic 3: Serious Mixed-Backend Training Substrate

### Goal

Prove that `psionic` can train serious models across mixed Apple and NVIDIA
hardware, not just bounded open-adapter proofs.

### Why this is third

Before opening the network, the repo needs a strong local and controlled
multi-backend training core. Otherwise the internet work just exposes weak math
and weak runtime assumptions.

### Main work

- expand from adapter-only proofs toward full-model or larger-shard training
- implement stronger distributed checkpointing and resume
- add real multi-node gradient exchange and synchronization strategy
- make Apple and NVIDIA paths share one truthful math/evidence story
- measure throughput, utilization, and recovery under serious workloads
- prove larger mixed-backend training jobs than the current two-node bounded
  swarm

### Done when

- the repo can run a serious mixed Apple plus NVIDIA training job with retained
  checkpoints and evidence
- training scale is no longer limited to tiny proof workloads
- quality gains from more compute are visible and repeatable

## Epic 4: Elastic WAN Swarm Runtime

### Goal

Turn the current trusted-LAN / configured-peer swarm proofs into a real
wide-area training runtime that can survive delay, churn, and partial failure.

### Why this is fourth

The current swarm path proves the idea. It does not yet prove that the system
can survive the messy conditions of real decentralized training.

### Main work

- resilient node membership and lease management
- checkpoint catch-up for late joiners and recovering nodes
- WAN-aware routing, relay, and bandwidth adaptation
- artifact exchange that works across unreliable networks
- better replay and validator behavior under partial failure
- elastic run resumption when nodes leave and rejoin

### Done when

- a decentralized run can continue through realistic WAN failures
- nodes can join late, leave, recover, and still contribute honestly
- the run does not depend on one trusted-LAN shape or one static pair of peers

## Epic 5: Open Internet Decentralized Training Network

### Goal

Make it possible for outside devices to join training runs over the internet in
a way that is discoverable, controlled, and hard to fake.

### Why this is fifth

This should not be attempted before the WAN runtime is stable. Public admission
on top of a fragile runtime just creates chaos.

### Main work

- public node identity and signing
- discovery and rendezvous
- NAT traversal and relay fallback
- admission policy and capability matching
- trust classes, fraud resistance, and validator selection
- public explorer surfaces so operators can see what the network is doing
- incentive, settlement, and slashing rules for public participation

### Done when

- an outside device can discover a run, request admission, receive work, submit
  artifacts, and be validated under public network rules
- the network has machine-readable trust and fraud boundaries
- decentralized training no longer means “devices we already fully control”

## Epic 6: Automatic Promotion To Production Models

### Goal

Make successful live runs automatically flow into production model serving when
they pass the right gates.

### Why this is last

Automatic promotion is dangerous until:

- the model is worth promoting
- the train-to-infer bundle is stable
- the runtime is trustworthy
- the network cannot be trivially gamed

### Main work

- define final promotion policy over quality, runtime, safety, and provenance
- connect validator outcomes to candidate publication
- connect candidate publication to serve deployment
- add canary, rollback, and “hold promotion” rules
- retain one deployment receipt chain from run evidence to serving state

### Done when

- a live decentralized training run can honestly produce a deployable model
- the deployment decision is machine-backed and auditable
- bad or ambiguous runs fail closed instead of sneaking into production

## Recommended Sequencing Inside The Epics

The practical order inside the big plan should be:

1. get one useful small model over the line
2. make that model promotable into the current serve path
3. scale training for that exact model family on controlled mixed hardware
4. extend that exact run shape into elastic WAN operation
5. only then widen admission to outside internet devices
6. only then allow automatic production promotion

That sequencing matters because each step gives the next one a real object to
operate on:

- Epic 1 gives us a model worth caring about
- Epic 2 gives us a product lifecycle for that model
- Epic 3 gives us compute scale
- Epic 4 gives us robust decentralized runtime behavior
- Epic 5 gives us public network participation
- Epic 6 gives us safe product automation

## What Should Not Happen

These are the main ways to waste time:

- trying to build public internet swarm admission before the WAN runtime is
  stable
- trying to automate production promotion before the model quality is good
  enough
- trying to scale training broadly before one strong model family exists
- building separate fake artifact paths for training and serving
- treating the current trusted-LAN swarm proof as if it already means internet
  decentralized training

## Bottom Line

`psionic` is no longer at the “nothing works” stage.

The repo now has enough real training, inference, swarm, and publication proof
that the next phase should not be about inventing more bounded demos. The next
phase should be about turning the existing truthful substrate into:

- one useful model family
- one real promotion spine
- one serious mixed-backend trainer
- one elastic WAN runtime
- one honest public decentralized network
- one automatic production promotion path

That is the shortest honest path from today’s state to the system we actually
want.
