# Tassadar Post-Article Continuation Non-Computationality Contract

`TAS-210` closes the continuation non-computationality boundary for the
canonical post-article machine.

The canonical machine-readable artifacts are:

- `fixtures/tassadar/reports/tassadar_post_article_continuation_non_computationality_contract_report.json`
- `fixtures/tassadar/reports/tassadar_post_article_continuation_non_computationality_contract_summary.json`
- `crates/psionic-transformer/src/tassadar_post_article_continuation_non_computationality_contract.rs`
- `crates/psionic-provider/src/tassadar_post_article_continuation_non_computationality_contract.rs`
- `scripts/check-tassadar-post-article-continuation-non-computationality-contract.sh`

This tranche makes four things explicit and machine-readable:

1. Checkpoint, spill, tape, session, process-object, and installed-process
   surfaces are continuation transport surfaces only; they do not become a
   second planner, scheduler, or workflow machine.
2. Continuation now stays bound to the same canonical machine tuple, the same
   canonical computational-model statement, and the same proof-transport
   boundary instead of inheriting that identity by adjacency.
3. Resume, retry, and stop semantics may extend the declared continuation
   carrier only while the host remains non-planner, and weighted plugin control
   may not repurpose continuation as hidden compute.
4. The contract closes here without collapsing later fast-route legitimacy,
   broader anti-drift closure, served/public universality, or the final
   claim-bearing canonical machine closure bundle into this issue.

This audit closes continuation non-computationality, moves the next anti-drift
stability frontier to `TAS-211`, keeps the final claim-bearing canonical
machine closure bundle separate for `TAS-215`, and still does not turn plugin
publication, served/public universality, or arbitrary software capability
green.
