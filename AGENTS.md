# Psionic Agent Contract

## Writing style

Write directly. Prefer blunt declarative sentences over rhetorical scene-setting.

Do not use contrastive metadiscourse or rhetorical throat-clearing such as:

- "the interesting question is not X, it is Y"
- "the real question is..."
- "the point is not..., the point is..."
- "X lives in a different neighborhood"
- "X sits in a different orbit / layer / space"

Do not use vague conceptual metaphors when a direct architectural statement will
do.

Bad:

- "The HyperAgents paper lives in a different neighborhood."
- "The interesting question is not whether this is impressive in the abstract."

Good:

- "HyperAgents improves worker, validator, and routing quality inside OpenAgents."
- "This matters because it lowers cost per accepted outcome."

Default to:

- direct claims
- explicit subsystem names
- concrete causal language
- plain statements of what is true, what changes, and why it matters

## Scope

- This repository is the standalone `psionic` workspace extracted from
  `openagents`.
- Primary scope is `crates/psionic-*`, `docs/`, `fixtures/`, and the small set
  of repo-local `scripts/`.
- Psionic owns the machine-facing execution substrate: tensor and compiler
  contracts, runtime and backend truth, serving interfaces, cluster and sandbox
  execution, adapters, data, eval, research, and the early training substrate.
- Keep app UX, wallet or payout flows, market orchestration, and kernel or
  settlement authority out of this repo unless the user explicitly asks for
  cross-repo work.

## Canonical Docs

- `README.md` is the Psionic entrypoint and map.
- `docs/ARCHITECTURE.md` is the canonical Psionic-wide system spec.
- `docs/FRAMEWORK_CORE_ACCEPTANCE_MATRIX.md` is the canonical framework-core
  completion bar.
- `docs/INFERENCE_ENGINE.md` is the canonical inference-engine completion doc.
- `docs/TRAIN_SYSTEM.md` is the canonical training subsystem spec.
- Domain-specific docs in `docs/` define the contract for their subsystem.
- `docs/audits/` explains rationale and follow-on direction, but audits are not
  the canonical current-state spec.
- For Tassadar paper-backed roadmap or issue work, the local paper corpus lives
  in `~/code/alpha/tassadar/tassadar-research/papers/` and
  `~/code/alpha/tassadar/can-llms-be-computers/papers/`.
- The corresponding local reading notes live in
  `~/code/alpha/tassadar/tassadar-research/notes.md` and
  `~/code/alpha/tassadar/can-llms-be-computers/notes.md`.

## Workspace Map

- Core execution path lives in crates such as `psionic-core`, `psionic-array`,
  `psionic-ir`, `psionic-compiler`, `psionic-runtime`, and the backend crates.
- Distributed and execution plumbing lives in crates such as
  `psionic-cluster`, `psionic-collectives`, `psionic-distributed`,
  `psionic-datastream`, `psionic-sandbox`, and `psionic-net`.
- Serving and provider-facing compute surfaces live in crates such as
  `psionic-serve`, `psionic-provider`, `psionic-router`, and
  `psionic-catalog`.
- Data, model, training, eval, environment, and research lanes live in crates
  such as `psionic-data`, `psionic-models`, `psionic-train`,
  `psionic-environments`, `psionic-eval`, `psionic-research`,
  `psionic-adapters`, and `psionic-apple-fm`.
- `fixtures/` contains committed evidence, run bundles, and compatibility
  artifacts. Treat them as versioned substrate truth, not disposable samples.

## Execution Rules

- Read the relevant canonical doc before editing a subsystem.
- Keep machine-legible truth explicit: manifests, receipts, proofs, capability
  reports, refusal reasons, runtime identity, artifact identity, and lineage
  should remain accurate and deterministic.
- Preserve replay-safe and deterministic behavior. Do not hide fallbacks,
  bounded support, or refusal posture behind optimistic defaults.
- Prefer extending existing crate boundaries over adding cross-cutting
  shortcuts or hidden control planes.
- If behavior or architecture changes, update the relevant doc in `docs/` in
  the same change.
- Use the repo status vocabulary consistently:
  `implemented`, `implemented_early`, `partial`,
  `partial_outside_psionic`, and `planned`.
- If a host, credential, external machine, or network path is blocked, do not
  sit idle waiting for the user by default.
- Route around the obstacle when possible: keep working on accessible devices,
  reduce the active scope to the honest reachable set, and update issues or
  docs so the recorded plan matches reality.
- Only stop to ask the user when the blocked dependency is mandatory for the
  next honest step and there is no meaningful non-blocked work left.

## Worktree Hygiene

- Start every task by checking `git status --short --branch`.
- If the checkout already contains unrelated changes, do not run broad
  formatters, artifact-generating examples, or repo-wide update commands in
  that checkout.
- Commits and pushes land on `main` unless the user explicitly instructs
  otherwise.
- Prefer working directly in the main checkout and cleaning up task-owned dirt
  there rather than creating extra worktrees or issue branches.
- Never clean a dirty tree by resetting, discarding, or reverting changes you
  did not make.
- If unrelated user-owned dirt in the main checkout makes safe progress
  impossible, use a temporary `git worktree` or temporary branch only as a last
  resort.
- Temporary worktrees and temporary branches are not backlog. Delete them after
  the change lands on `main`, and do not leave abandoned issue branches or old
  worktrees behind.
- Before commit or push, the checkout used for the change must contain only the
  intended task files.

## Formatting Discipline

- Formatting changes are allowed and expected, but they must stay scoped to the
  files intentionally edited for the task unless the user explicitly asks for a
  broader formatting sweep.
- Treat `cargo fmt`, `rustfmt`, and similar formatters as potentially broader
  than they look. In Rust crates, formatting one file or one module entrypoint
  can rewrite sibling modules through the module tree.
- Do not run workspace-wide `cargo fmt` from a dirty or shared checkout.
- Prefer formatting only the Rust files you edited.
- When doing targeted work, prefer the narrowest formatter invocation that
  preserves the intended file set, and assume it may still spill into adjacent
  modules until `git diff --stat` proves otherwise.
- After running formatting, inspect `git diff --stat` or `git status --short`.
  If unrelated files changed, revert those incidental formatting changes before
  continuing.
- Do not leave incidental formatter spill in a targeted change just because the
  files are “only formatting.” Either scope it back down or move the work to a
  clean dedicated formatting pass.
- If a task genuinely requires a broad formatting pass, land that as a
  dedicated clean change, not mixed into unrelated code or artifact updates.
- When the checkout is clean and there is a good opportunity, agents should
  occasionally run a broader `cargo fmt` pass, review the result, and commit
  those formatting-only changes separately from behavioral work.

## Artifact-Generating Commands

- Treat `cargo run`, example binaries, report writers, fixture generators, and
  repo-local scripts that write under `fixtures/`, `docs/`, or report paths as
  artifact-generating commands.
- Before running one of those commands, assume it may rewrite more files than
  the immediate code edit.
- Prefer running artifact-generating commands from a clean main checkout.
- If a temporary worktree is truly necessary for an artifact-generating
  command, remove it after the change lands on `main`.
- If generated artifacts are part of the task, review them and commit them in
  the same change as the code or doc update that requires them.
- If generated artifacts are not part of the task, do not leave them behind as
  worktree dirt.

## Push Gate

- Before finalizing a task, run `git status --short --branch`.
- Do not commit or push from a checkout that still contains unrelated dirt.
- Push to `origin/main` unless the user explicitly asks for another branch or
  workflow.
- Never open a pull request, issue, or other outbound contribution against an
  external repository unless the user explicitly instructs you to do so.
- This ban is especially strict for OpenAI-owned repositories such as
  `openai/parameter-golf`: never open or update a PR there unless the user
  explicitly directs you to.
- If temporary isolation was required, land the change on `main` and then
  remove the temporary worktree or branch before considering the task done.

## Extraction Notes

- Some Psionic docs still mention historical `openagents` files or scripts that
  are not present in this standalone repo.
- Treat those references as historical context, not live dependencies for work
  in this repository.
- Do not pull code or docs back from `openagents` by default. Restore or copy
  material only when the user explicitly asks for it.

## Disclosure-Safe Release Flow

- When private or alpha-only material informs public `psionic` work, decompose
  it before landing anything here: strip private naming, private product
  framing, speculative capability language, and any unannounced roadmap nouns.
- Public docs, issue bodies, report summaries, and benchmark notes must use
  bounded public claim language only. Do not copy private roadmap wording or
  internal shorthand forward into public repo surfaces.
- Preserve dependency markers honestly. If a capability still depends on
  `kernel-policy`, `world-mounts`, `nexus`, or `openagents`, say so explicitly
  instead of backfilling placeholder behavior into `psionic`.
- If private-only framing survives review, refuse publication in this repo and
  make that refusal explicit in the review artifact instead of silently
  weakening the boundary.
- The canonical public checklist and checker live at
  `docs/TASSADAR_PUBLIC_DISCLOSURE_FLOW.md`,
  `fixtures/tassadar/reports/tassadar_public_disclosure_release_review.json`,
  and `scripts/check-tassadar-public-disclosure.sh`.

## Validation

- Prefer targeted validation first: `cargo test -p <crate>` or the specific
  test, example, or fixture path that covers the touched subsystem.
- Use `cargo test --workspace` when the change is broad enough and the local
  environment makes that realistic.
- If you change fixtures, reports, or schema-like contracts, verify the code
  that produces and consumes them.
