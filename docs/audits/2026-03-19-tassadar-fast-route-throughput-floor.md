# TAS-175 Fast-Route Throughput Floor

`TAS-175` closes the bounded throughput tranche for the selected fast article
route.

What is now true:

- the canonical fast route is still `HullCache`
- the route stays exact and direct on the committed Hungarian article run and
  the committed `sudoku_9x9_test_a` hard-Sudoku stand-in run
- the route now carries explicit measured CPU throughput floors for those demo
  rows
- the route now also carries direct measured HullCache throughput floors for
  the bounded million-step and multi-million-step kernel set
- the eval surface now binds those floors to the TAS-175 acceptance-gate row,
  the TAS-174 exactness prerequisite, and a zero-drift policy across the
  declared `host_cpu_aarch64` and `host_cpu_x86_64` classes

What is still not true:

- this is not final Hungarian demo parity on the canonical article route
- this is not Arto Inkala closure
- this is not benchmark-wide hard-Sudoku closure
- this is not single-run no-spill million-step closure
- this is not final article-equivalence green status

Canonical artifacts:

- runtime bundle:
  `fixtures/tassadar/runs/article_fast_route_throughput_v1/article_fast_route_throughput_bundle.json`
- eval report:
  `fixtures/tassadar/reports/tassadar_article_fast_route_throughput_floor_report.json`
- research summary:
  `fixtures/tassadar/reports/tassadar_article_fast_route_throughput_floor_summary.json`
- checker:
  `scripts/check-tassadar-article-fast-route-throughput-floor.sh`
