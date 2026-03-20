# TAS-172 Fast-Route Architecture Selection

`TAS-172` closes the architecture-selection tranche for the article-equivalence
program.

The repo now makes one machine-readable decision:

- `HullCache` is the canonical fast article route.

That selection is not based only on raw speed. The report keeps three
requirements separate and requires all of them:

- article-matrix exactness stays explicit
- fallback or refusal posture stays explicit
- the fast family fits the canonical planner route contract

The current committed evidence says:

- `HullCache` is exact on the current article-class matrix, already promoted on
  the runtime-facing article lane, and already fits the canonical decode-mode
  contract with explicit direct-versus-reference-linear-fallback boundaries
- the recurrent runtime baseline is fast and exact on its bounded workload set,
  but it remains research-only and has no canonical decode-mode contract
- the hierarchical-hull candidate is also strong research evidence, but it
  remains research-only and has no canonical decode-mode contract
- the 2D-head hard-max lane remains a bounded research comparison surface with
  hull fallback, not an article-class fast-route contract

So `TAS-172` chooses `HullCache` as the fast route without pretending the owned
Transformer-backed model already runs on it.

That stronger claim remains blocked on later tranches:

- `TAS-173` integrates the chosen fast path into the canonical
  Transformer-backed model
- `TAS-174` closes full fast-route exactness and no-fallback posture
- `TAS-175` closes the fast-route throughput floor
