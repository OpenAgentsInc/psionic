# Psion CS336 A4 Data-Refinery Reference Lane

> Status: `implemented_early` bounded reference lane, landed 2026-06-10
> under issue #1102. Companion to the A1, A2, and A5 reference lanes.

This document records the owned `psionic` surfaces for the bounded
Stanford CS336 Assignment 4 (data) port program. It is the psionic-side
answer to the OpenAgents homework epic's external ask (openagents issue
#4680).

## Identity

- lane id: `psion_cs336_a4_data_refinery_reference_v1`
- owned surface: `crates/psionic-data/src/cs336_a4_data_refinery.rs`
- claim boundary: bounded deterministic reference implementations of the
  model-free Stanford CS336 A4 surface (PII masking, Gopher rules, exact
  line dedup, MinHash dedup) over in-memory documents only; heuristic
  scanners are unverified against Stanford fixtures, and HTML extraction,
  language identification, and model-backed quality/NSFW/toxicity
  classification are not implemented or claimed.

## Adapter Matrix

| Stanford adapter | Owned surface | Status | Notes |
| --- | --- | --- | --- |
| `run_extract_text_from_html_bytes` | — | `planned` | resiliparse-class extraction; out of scope for the deterministic lane |
| `run_identify_language` | — | `planned` | model-backed (fastText-class); refusal until a bounded model lane exists |
| `run_mask_emails` | `cs336_a4_mask_emails` | `partial` | `\|\|\|EMAIL_ADDRESS\|\|\|` token + count; heuristic pattern, fixture conformance unverified |
| `run_mask_phone_numbers` | `cs336_a4_mask_phone_numbers` | `partial` | US formats (parenthesized, dashed, dotted, spaced, +1) with digit-boundary guards |
| `run_mask_ips` | `cs336_a4_mask_ips` | `partial` | dotted-quad IPv4 with 0–255 validation; invalid quads left unmasked |
| `run_classify_nsfw` / `run_classify_toxic_speech` / `run_classify_quality` | — | `planned` | model-backed classification; never claimed by this lane |
| `run_gopher_quality_filter` | `cs336_a4_gopher_quality_filter` | `implemented_early` | word count 50–100k, mean word length 3–10, <30% ellipsis lines, ≥80% alphabetic words; per-rule verdict report |
| `run_exact_line_deduplication` | `cs336_a4_exact_line_deduplication` | `implemented_early` | corpus-frequency rule over in-memory documents; file orchestration belongs to the dispatch layer |
| `run_minhash_deduplication` | `cs336_a4_minhash_deduplication` | `implemented_early` | normalized word n-gram shingles, seeded splitmix64 hash family, LSH banding, exact-Jaccard verification, union-find clustering keeping the lowest-index representative; deterministic |

## Landed Tests

8 unit tests: email/phone/IP masking with counts and negative cases
(order numbers unmasked, out-of-range quads preserved), Gopher rules on
ordinary prose plus three degenerate-document failures, corpus-level
exact line dedup, MinHash near-duplicate clustering with distinct-document
retention, invalid-parameter refusals, and determinism.

## Relation To The Homework Epic

OpenAgents #4680 dispatches these stages as deterministic-recompute CPU
homework. This lane is the reference the validators recompute against.
Before any payout depends on the heuristic scanners (PII masks), they
must be conformance-tested against the Stanford fixtures per the epic's
verification rules; the `partial` statuses above are the tracking marker
for exactly that gap.
