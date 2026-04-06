# MoE Flash Reference Prefetch Audit

This audit records what was actually useful in
`competition/repos/qwen3.5-gemma4-moe-flash-mlx-turbo-quant` and what Psionic
ported from it into the admitted sparse `Gemma 4 26B A4B` lane.

The main lesson from that repo is not a secret kernel. It is execution shape.
The repo keeps resident weights separate from routed expert weights, brings
expert pages resident before compute touches them, and avoids discovering every
expert row through one cold page fault at a time. That matches the current weak
point in Psionic's sparse Apple-Silicon path much more closely than the dense
Qwen 27B path.

Before this pass, Psionic's admitted sparse `Gemma 4 26B A4B` lane on Metal
already worked, but its host fallback did too much reactive I/O during expert
math. The sparse host path would choose routed experts and then walk gate, up,
and down rows through the paged storage one row at a time. That made the lane
honest but slow. The earlier retained receipt on the same benchmark shape was
`43.0130 s` for `16` output tokens, about `0.372 tok/s` end to end.

The change landed here was narrow and deliberate. `psionic-catalog` now exposes
range-prefetch on paged blob slices. On macOS that prefault path uses
`madvise(..., MADV_WILLNEED)` on the aligned mapped range and then touches one
byte per page so the subsequent row walkers do not fault the range in lazily.
`psionic-models` forwards that prefetch capability through `PagedTensorStorage`.
`psionic-serve` now prefaults the selected gate, up, and down expert ranges
immediately after routing picks the top experts for the token. After prefault,
the same sparse host path fans selected experts out in parallel with Rayon
instead of evaluating them strictly serially.

That is the direct ported idea from the reference repo: do not wait until the
inner expert matvec loop to discover expert pages, and do not serialize the
selected-expert work if the host already knows which experts this token will
use.

On the same local benchmark lane after this change, Psionic now records:

- model = `gemma-4-26B-A4B-it-Q4_K_M.gguf`
- mode = `distributed-sparse`
- backend = `metal/experts=128x8`
- load = `30.7238 s`
- total = `12.4447 s`
- output tokens = `16`
- throughput = about `1.286 tok/s`

That is about a `3.46x` end-to-end improvement over the earlier retained local
Metal sparse receipt.

This does not close the whole gap. The lane is still fallback-heavy. Unsupported
quantized rows still route through host projection instead of a native Metal
expert kernel, and this path still lacks the stronger next-layer speculative
expert prefetch that the reference repo experiments with. But the basic lesson
was right: sparse MoE performance on Apple Silicon is not only a kernel
question. It is also a page-residency and expert-scheduling question, and
Psionic's sparse path moved materially once it stopped treating expert pages as
something to discover lazily inside the hottest loop.
