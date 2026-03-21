# TAS-185 Audit

The canonical fast article route now has one explicit cross-machine
reproducibility matrix.

That matrix is intentionally narrower than a generic portability claim. It is
bound to the selected deterministic `HullCache` route, the declared
`host_cpu_x86_64` and `host_cpu_aarch64` machine classes, the canonical
Hungarian and Sudoku demo evidence, the deterministic single-run long-horizon
closure, and the already-declared throughput-floor policy.

The public repo now says, machine-readably:

- the selected fast route is stable as a direct deterministic `HullCache` lane
- the declared demo surface remains exact on that lane
- the declared million-step and multi-million-step horizons remain exact on
  that lane
- the declared supported CPU machine classes stay inside the same throughput
  drift policy
- stochastic execution is not silently admitted as part of the article route

This is stronger than one-host optimism, and it is stronger than treating the
existing throughput-floor report as if it already implied full reproducibility.

It is still not the final article-equivalence verdict.

The matrix does not yet say:

- that the route is minimal
- that stochastic execution is part of the admitted article claim
- that the final publication checker is green
- that the broader article argument is now complete without the remaining
  `TAS-185A` and `TAS-186` tranches

Canonical artifacts for this tranche:

- `fixtures/tassadar/reports/tassadar_article_cross_machine_reproducibility_matrix_report.json`
- `fixtures/tassadar/reports/tassadar_article_cross_machine_reproducibility_matrix_summary.json`
- `fixtures/tassadar/reports/tassadar_article_cross_machine_reproducibility_publication.json`
- `scripts/check-tassadar-article-cross-machine-reproducibility-matrix.sh`
