# Psionic Philox 4x32-10 Counter-Based RNG

> Status: canonical `psionic#1116` record, 2026-06-12, after landing the
> owned Philox 4x32-10 implementation as the standard RNG for seeded work
> classes.

Authored by Fable (claude-fable-5) for psionic#1116.

## What This Is

Psionic owns a Philox 4x32-10 counter-based RNG and declares it the standard
RNG for all seeded work classes: ablation cells, scaling-sweep cells, rollout
generation, and any future seeded generation kinds.

- `crates/psionic-core/src/philox.rs` — the implementation
- `fixtures/rng/philox4x32_reference_vectors.json` — published reference
  vectors and the committed determinism receipt
- `scripts/check-philox-rng.sh` — fixture parity via an independent Python
  reimplementation, plus the pinned Rust tests
- contract id: `psionic.core.philox_rng.v1`

## Why Counter-Based

Seeded work classes verify by `seeded_replication`: a same-class validator
recomputes the work from the same seed and compares (see
`docs/2026-06-12-powersgd-freivalds-compatibility.md` for how compressed
contributions ride this class). That verification class only holds if the RNG
produces bit-identical output on every contributor device and backend. An RNG
that differs across backends breaks the class.

Philox 4x32-10 (Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy
as 1, 2, 3", SC'11, doi:10.1145/2063384.2063405) is a counter-based block
function: `(seed, stream, counter) -> 128 random bits` through pure 32-bit
integer arithmetic. There is no mutable global state, no call-order
dependence, no ambient entropy, and no backend-specific library code in the
path. Cross-device reproducibility is a property of the construction, not a
property we test for and hope holds.

PyTorch's CUDA generator (`at::Philox4_32`) uses the same algorithm and the
same `(seed, subsequence, offset)` parameterization, so the stream layout in
this module matches what GPU training stacks already standardize on.

## Reference Sources

Read-only references (never vendored):

- `projects/tether/repos/qvac-ext-stable-diffusion.cpp/src/rng_philox.hpp` —
  the QVAC/Tether port absorbed via the 2026-06-10 QVAC edge-stack analysis
  (openagents `docs/training/2026-06-10-qvac-edge-stack-analysis.md`, Tier 1);
  itself a port of AUTOMATIC1111 `modules/rng_philox.py`, which replicates
  `torch.randn` on CUDA.
- `DEShawResearch/random123` `tests/kat_vectors` — the published known-answer
  vectors for `philox4x32 10`, copied verbatim into the fixture and into
  `PHILOX4X32_10_REFERENCE_VECTORS`.

## API Shape

Three layers, matching how existing seeded call sites already work:

1. `philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4]` — the bare
   block function, pinned to the published vectors.
2. `PhiloxStream { seed, stream }` — random-access draws by counter/index:
   `block`, `u32_at`, `u64_at`, `unit_f64_at`. Layout: `key = [seed_lo,
   seed_hi]`, counter words `[counter_lo, counter_hi, stream_lo, stream_hi]`
   (PyTorch-compatible; the QVAC layout is the 32-bit special case, pinned by
   test).
3. `philox_counter_rng(seed, stream, counter) -> u64` and
   `philox_counter_unit(seed, stream, counter) -> f64` — stateless draws with
   the same shape as the house splitmix64 mix (`sparta_counter_rng` /
   `sparta_counter_unit` in `psionic-train`), so call sites migrate without
   restructuring.

Floating-point scope: the uniform draw uses only exact IEEE-754 operations
(53-bit mantissa fill, division by a power of two) and is bit-identical on
every conforming platform. Normal draws are intentionally not provided:
Box–Muller depends on platform `libm` `log`/`sin`, which is exactly the
backend-dependent behavior this standard exists to exclude. A deterministic
transcendental policy is a recorded follow-up below.

## Determinism Receipt

The committed workload: `philox_counter_rng(seed, stream, counter)` for
`seed = 0x11165EEDC0DE2026`, streams `0..4`, counters `0..1024`, stream-major
order; SHA-256 over the little-endian bytes of each `u64` draw.

```
sha256 = 0950714a1c358c8223476960e20170bd2046c114c69da1c507ad59e361f61d40
```

What is claimed: the digest was produced by the Rust implementation on an
arm64 macOS (Metal-attached) host and reproduced exactly by an independent
Python reimplementation validated against the same published vectors. The
workload is host-side scalar integer arithmetic, so any conforming host —
CPU-only, Metal-attached, or CUDA-attached — must reproduce it;
`philox_determinism_receipt_digest()` recomputes it anywhere, and
`scripts/check-philox-rng.sh` checks it.

What is not claimed: recorded receipt runs from a CUDA host and from
additional contributor hardware have not been executed yet (recorded
follow-up). No claim is made about parity with live `torch.randn` output
streams; the compatibility claim is at the algorithm-and-parameterization
level, pinned by the published vectors and the layout tests, not by a
captured PyTorch output vector.

## Standard and Migration Posture

- New seeded work classes (ablation cells, scaling-sweep cells, rollout
  generation, future seeded generation kinds) must take their randomness from
  this module via caller-passed `(seed, stream, counter)` triples — never
  ambient entropy, wall clocks, or stateful sequential generators.
- Existing splitmix64-based seeded sites are deliberately not migrated in
  this change: `cs336_a4_data_refinery.rs` (psionic-data),
  `tassadar_alm_bounded_check.rs` (psionic-compiler), and `sparta_canary.rs`
  (psionic-train) have outputs pinned in committed fixtures; swapping their
  RNG would churn working fixtures without adding verification value. They
  migrate only when their lane is next reworked for its own reasons.

## Recorded Follow-Ups

1. Wire `PhiloxStream` as the default seeded-work RNG when the ablation
   harness (`training.ablation_system.v1`) lands in psionic.
2. Wire `PhiloxStream` into the A5 rollout lanes when rollout generation
   gains a seeded sampling path (`cs336_a5_alignment_reference` currently has
   no RNG call site).
3. Run the determinism-receipt workload on a CUDA host and on at least one
   additional contributor machine, and record the runs as evidence.
4. Decide a deterministic transcendental policy (owned polynomial `log`/`sin`
   or integer-only alternatives such as ziggurat-with-fixed-tables) before
   offering normal draws under this contract.
5. Optionally capture a live `torch.randn` CUDA output vector and pin it as a
   torch-parity fixture if exact torch stream parity ever becomes a
   requirement (it is not one today).

## Validation

```
./scripts/check-philox-rng.sh
cargo test -p psionic-core philox
```
