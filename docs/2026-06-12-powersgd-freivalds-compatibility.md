# Does PowerSGD-Class Low-Rank Compression Compose With Freivalds Committed-Matrix Verification?

> Status: canonical `#1128` record, created 2026-06-12. Answers the
> named research question from the Pluralis adaptation roadmap item
> P3.2 (openagents
> `docs/training/2026-06-12-pluralis-to-pylon-adaptation-roadmap.md`,
> commit `463b0d76c`). This is a research answer, not a capability
> claim; nothing here ships behavior.

## Short Answer

Compression composes with the algebra but not with the provenance.
Low-rank factors admit Freivalds-style random-projection consistency
checks more cheaply than dense matrices do, but lossy compression
severs the exact algebraic identity that the `freivalds_merkle` class
depends on, so a validator cannot verify that submitted factors came
from the true gradient without recomputing it. Compressed gradient
contributions from strangers therefore ride `seeded_replication` or
stay inside the trust boundary. The bandwidth win of PowerSGD is
peer-to-peer, not verification-side.

## The Two Mechanisms

**Freivalds' check.** Given committed matrices `A`, `B` and a claimed
product `C`, a validator samples a random vector `r` (e.g. uniform
over `{0,1}^n`) and accepts only if `C·r = A·(B·r)`. Cost is `O(n^2)`
per probe instead of `O(n^3)` for recomputation. Soundness: if
`C ≠ A·B`, then `D = C − A·B` has a nonzero row `d`; fixing any index
`k` with `d_k ≠ 0` and conditioning on the other coordinates of `r`,
at most one of the two values of `r_k` makes `d·r = 0`, so a wrong
claim survives each probe with probability at most 1/2 and `k` probes
with probability at most `2^-k`. The property that matters: Freivalds
verifies an **exact algebraic identity among committed operands**. It
verifies computation, not approximation.

**PowerSGD** (Vogels, Karimireddy, Jaggi 2019, arXiv:1905.13727; the
implementation Pluralis ships is
`projects/pluralis/repos/node0/src/node0/server/power_sgd_averager.py`
with `averager_rank: 64`). For a gradient matrix `M ∈ R^{m×n}`, one
warm-started power-iteration step computes `P = M·Q`, orthonormalizes
`P`, then `Q ← Mᵀ·P`. Only the factors `P` (`m×r`) and `Q` (`n×r`)
are communicated; the decoded approximation is `M̂ = P·Qᵀ`. The
residual `M − M̂` is kept locally as error feedback and added to the
next round's gradient. The compression is **lossy**: `M̂ ≠ M` except
in the degenerate case where `M` has rank at most `r` and the power
iteration has converged.

## Part (a): The Algebra Composes

Random-projection checks work on low-rank products, and cheaply.
Suppose a worker commits a dense claimed matrix `M̂` (say, a Merkle
root over its entries) together with factors `(P, Q)`, and claims
`M̂ = P·Qᵀ`. A validator can:

- **Projection probe.** Sample `r`, compute `u = Qᵀ·r` in `O(nr)`,
  then `P·u` in `O(mr)`, and compare with `M̂·r`. The factor side of
  each probe costs `O((m+n)·r)` — the rank structure makes the probe
  cheaper, not harder. Soundness is the same Freivalds bound: a false
  factorization claim survives each probe with probability ≤ 1/2.
- **Coordinate probe.** Open `M̂[i][j]` under the Merkle root and
  check it equals `⟨P_i, Q_j⟩` in `O(r)` per coordinate.

The power-iteration procedure itself is also a chain of matrix
products — `P = M·Q` and `Q' = Mᵀ·P` are each Freivalds-checkable,
and orthonormalization can be checked via `PᵀP = I_r` with the same
projection trick — **provided `M` is committed**. That proviso is the
whole problem, and it is Part (b).

## Part (b): The Provenance Does Not

Freivalds verifies identities among committed operands. PowerSGD's
defining relation to the true gradient `G` is `M̂ ≈ G` — an
approximation with an unbounded-by-identity residual, not an
identity. There is no exact equation `f(P, Q, data) = 0` binding the
shipped factors to the gradient of the loss on the committed data
unless `G` itself, or the forward/backward computation that produced
it, is committed. The options, exhaustively:

1. **Commit full-rank `G`.** Then `P = G·Q` and `Q' = Gᵀ·P` are exact
   identities and Freivalds applies. But computing `G·r` for the
   probe requires touching all of `G` (row probes of `P = G·Q` cost
   `n` Merkle openings per row), and committing `G` says nothing
   about whether `G` is the true gradient — that still requires the
   full `freivalds_merkle` chain over the forward/backward
   computation. The verification artifact is full-rank; compression
   saves the verifier nothing. The peer-to-peer averaging traffic
   still shrinks, which is the saving Pluralis actually uses.
2. **Make `(P, Q)` the committed deliverable.** Redefine the work
   class so the factors are the artifact and verification checks the
   power-iteration procedure from a committed seed and warm-start
   `Q_0`. The procedure is deterministic given `Q_0` and `G` — and
   there is the same `G` again. Circular.
3. **Seeded replication.** A validator with the same data shard,
   seed, and code recomputes `G`, runs the same power iteration, and
   compares digests of `(P, Q)`. Given bitwise-deterministic
   execution, this works. It is the `seeded_replication` class at
   full recompute cost; nothing about it is Freivalds.

## Worked Example: Internal Consistency Is Not Provenance

Take `m = n = 2`, rank `r = 1`. Suppose the true gradient (what
seeded replication would reproduce) is

```
G = [ 2  1 ]
    [ 1  2 ]
```

An honest rank-1 PowerSGD step from warm start `Q_0 = (1,1)ᵀ/√2`
converges immediately (that is the top eigenvector, eigenvalue 3) and
ships factors decoding to

```
M̂_honest = [ 1.5  1.5 ]
           [ 1.5  1.5 ]
```

A malicious worker instead ships `P* = (10, −10)ᵀ`,
`Q* = (10, 10)ᵀ`, decoding to

```
M* = P*·Q*ᵀ = [  100   100 ]
              [ −100  −100 ]
```

Every internal-consistency probe on `(M*, P*, Q*)` passes with
probability 1: for any probe vector `r`, `M*·r = P*(Q*ᵀ·r)` holds
identically, because `M*` was constructed as exactly that product.
Coordinate probes pass too: `M*[i][j] = P*_i · Q*_j` by construction.
Yet `‖M* − G‖_F ≈ 200` against `‖G‖_F ≈ 3.16`; the submission is not
near `G`, nor near the best rank-1 approximation of `G`. No probe
that sees only the factors and their product can detect this, because
such probes test the factorization identity, which the adversary
satisfies by construction. Detection requires reference to `G` —
provenance, not consistency.

The degenerate form of the same point: in PowerSGD as deployed, `M̂`
is never transmitted at all — only the factors are. Then there is no
claimed identity to check. "`M̂ = P·Qᵀ`" is the definition of
decoding, true for arbitrary `(P, Q)`.

## Consequence For The Verification Map

The `freivalds_merkle` class's core property is verifying the
gradient computation itself without recompute, by chaining exact
matrix identities through a committed computation. Lossy compression
severs that chain: the shipped artifact is no longer connected to the
computation by any identity. Therefore:

- Compressed gradient contributions from strangers ride
  `seeded_replication` (full recompute by a same-class validator
  against committed shard, seed, and code) or do not enter at all.
- Inside the trust boundary (operator devices), PowerSGD-class
  compression is unconstrained by this result; trusted peers may
  exchange factors freely.
- The bandwidth saving is real but lives between peers, not between
  worker and validator. A compressed work class open to strangers at
  WAN-friendly **verification** bandwidth does not follow from this
  analysis.

This is the "no" branch of roadmap P3.2, with the boundary stated:
compression stays inside the trust boundary for the main optimizer
path, and the verification map records that compressed contributions
carry the `seeded_replication` grade.

## Open Refinement (a follow-up question, not a result)

A hybrid scheme might buy probabilistic detection of gross dishonesty
at sub-recompute cost: the worker additionally commits Merkle roots of
row-space sketches of `G` (e.g. `S = G·Ω` for a committed random
`Ω`), and the validator probes `(P, Q)` against the sketch within an
error-feedback tolerance, spot-opening sketch rows. Whether this is
sound against an adversary who chooses the sketch and the factors
together, and how the approximation tolerance bounds the adversary's
slack, is unanalyzed. Flagged here as a possible follow-up only.

## Ledger Update

Per this answer, entry 2 of `docs/PSION_DERISKING_LEDGER.md`
(PowerSGD rank-compressed gradient averaging) moves from `blocked` to
`answered (2026-06-12)` with the one-line answer: compression
composes with the algebra but not the provenance; compressed
contributions ride seeded_replication or stay inside the trust
boundary.

Authored by Fable (claude-fable-5) for psionic#1124 / #1128.
