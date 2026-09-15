# Kairyu

Kairyu is a vLLM-compatible LLM inference framework with native orchestration,
layered as L3 Interface / L2 Orchestration / L1 Engines. Package: `kairyu/`.

## Session start

ALWAYS read `PROGRESS.md` first — it is the cross-session memory of design changes,
milestone status, and blockers.

## Progress log rules

@.claude/rules/progress-log.md

## Framework and example ownership

@.claude/rules/framework-boundary.md

## Test policy

Apply these rules whenever a change adds, removes, moves, or reclassifies product
behavior or tests, and whenever CI is repaired after such a change.

1. Keep only tests that are necessary to protect current product behavior from a
   concrete regression. The suite must be necessary and sufficient, not maximal.
2. Test externally observable behavior, cross-component contracts, and previously
   demonstrated regression risks. Do not add a test merely to mirror constants,
   enumerate static registry/configuration contents, or restate an invariant that
   can be enforced at construction or load time in product code. In particular,
   content that is sufficiently protected by a code assertion or validation does
   not also get a pytest case without a concrete product-level failure mode.
3. For a linear `A -> B -> C` flow, test the accepted input at A and the observable
   result at C. Protect intermediate B invariants with assertions or validation in
   the implementation, not a separate test. Test B directly only when it is an
   independent public/reused boundary or owns branching, trust, concurrency, or a
   failure mode that an A-to-C test cannot expose.
4. When a feature is deleted, delete its feature-specific tests, fixtures, and test
   helpers. Do not migrate a stale test only to make CI green. Do not retain a
   negative test or runtime assertion that only proves the removed feature remains
   absent; use a one-time repository search during the change instead. Update a
   test only when it still protects a surviving product contract.
5. For deletion work, compare base and head collection counts with the same pytest
   command, markers, and environment. The test count must decrease. Any retained or
   new test must be justified by a concrete surviving behavior and regression risk,
   not by framework structure or coverage percentage alone.
6. Treat file moves as ownership changes, not new coverage. Before adding a test,
   check for an existing test of the same behavior and remove duplication.
7. Report the base/head collection counts and the rationale for every retained or
   newly added test area before declaring deletion work complete.

## Where things live

- Design decisions and rationale (D-IDs, review amendments): `docs/design/m1..m4-*.md`
- Archived progress history (old Change Log entries, status snapshots): `docs/progress/archive/`
- GPU-day execution plan: `docs/gpu-runbook.md`
- Implementation plans: `docs/superpowers/plans/`
- Dev commands: `uv sync --group dev`, `uv run pytest`, `uv run ruff check .`
