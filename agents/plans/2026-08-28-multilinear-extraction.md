# Task Plan

## Mission

Improve Flock's code structure and reduce duplication without changing protocol behavior.
Complete Phase 0, then create `flock-multilinear` and migrate proven shared primitives.

## Sources Of Truth

- User instructions in the active Codex task.
- Workspace CI commands in `.github/workflows/test.yml` and `.github/workflows/lint.yml`.
- Current behavior and tests in `flock-core` and `flock-prover`.

## Scope

- Record repository, line-count, build, test, lint, and format baselines.
- Compare all equality-table and multilinear-evaluation conventions.
- Use parameters for meaningful conventions such as variable order.
- Keep one implementation when parameter use has no measured performance cost.
- Create a dependency-safe `flock-multilinear` crate.
- Migrate only callers covered by parity tests in this slice.

## Current State

- Phase: Universal import cleanup complete.
- Last completed: Applied the universal import style to all Rust targets, tests, benches, examples, and binaries.
- Current result: All imports use explicit names. Standard, external, and crate imports use separate ordered groups. Repeated cfg import gates use one grouped import.
- Next action: Commit, push, and check PR 42 CI again.
- Open risk: The Blackwell GPU runner is offline. The user approved completion without this job.
- Files changed: Phase 0 and Phase 1 are on PR 42.
- Commands run: Format, native workspace check, release Clippy, release workspace tests, AVX-512 workspace check, and AVX-512 Clippy passed. Source audits found no glob or relative imports.

## Checklist

- [x] Record all Phase 0 baseline results.
- [x] Classify each current equality-table and MLE implementation.
- [x] Define explicit variable-order parameters and shape checks.
- [x] Add `flock-multilinear` to the workspace.
- [x] Add parity and convention tests.
- [x] Migrate shared production callers.
- [x] Remove duplicate covered implementations.
- [x] Run focused tests and all local CI gates.
- [x] Search for stale paths and old helper names.
- [x] Rename protocol errors to specific names.
- [x] Consolidate production imports at module scope.
- [x] Reduce internal field visibility with compiler checks.
- [x] Remove obsolete dead helpers and review remaining target oracles.
- [x] Remove historical design and milestone comments.
- [x] Run pinned proof-byte tests through the release workspace suite.
- [x] Extract binary field types into `flock-field`.
- [x] Extract challengers and transcript recording into `flock-transcript`.
- [x] Forward core features and preserve the `flock-core` API through re-exports.
- [x] Extract hash selection and compression primitives into `flock-hash`.
- [x] Extract Merkle commitments into `flock-merkle`.
- [x] Make Merkle construction and verification generic over `MerkleHash`.
- [x] Keep runtime hash selection and optimized hash kernels as compatibility paths.
- [x] Require exact equality-table iterator lengths.
- [x] Move F128 slice kernels into `flock-field`.
- [x] Give Merkle hashing traits to `flock-merkle`.
- [x] Share one all-core rayon pool through `flock-parallel`.
- [x] Use direct owner-crate imports from `flock-prover`.
- [x] Repair stale crate paths in repository documentation.
- [x] Remove all function-local imports from Rust code.
- [x] Replace inline paths for structs, enums, functions, traits, and constants.
- [x] Keep test-only imports at the top of each test module.
- [x] Run the syntax-tree audit again and run all local CI gates.
- [x] Remove all wildcard imports, including test preludes and public re-exports.
- [x] Group imports as standard library, external crates, then crate modules.
- [x] Sort external crate imports alphabetically.
- [x] Consolidate imports that use the same cfg gate.
- [x] Add a workspace Clippy rule against wildcard imports.

## Recovery

1. Reread this file.
2. Inspect current repository state.
3. Emit `CHECKPOINT RESTORED`.
4. Continue from Current State.
