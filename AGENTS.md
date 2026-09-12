# Project Contract

Stay conformant to `SPEC.md`. If an implementation choice conflicts with the spec, change the implementation or update
the spec in the same change with a clear reason.

Keep `docs/tools.md` up to date whenever built-in tool behavior changes. This includes tool names, arguments, filesystem
safety policy, output limits, truncation behavior, and user-visible tool errors.

# Comments and docstrings

Write inline comments and docstrings (in Rust: `///` doc comments on items and `//!` module docs) didactically. The
reader to optimize for is someone who does not know this codebase in detail: a contributor arriving cold, or the author
months later. They should be able to skim a file and recover what each piece is for, what it promises its callers, and
why it is shaped the way it is, without reading every line of the implementation. This applies to private helpers as
much as to public API; visibility is not what decides whether a symbol needs explaining.

That means a docstring leads with the contract and the reason the symbol exists, not with a restatement of its name.
Document invariants, edge cases, failure modes, and the assumptions callers must not make, since those are exactly the
details that are invisible in the code and expensive to rediscover. When a simpler-looking approach would have been
wrong, say so and say why; that is the kind of context that stops a future reader from "fixing" it.

Inline comments follow the same rule. They explain why, orient the reader to a block or a decision, and flag surprising
constraints. They do not translate syntax into English or label the obvious. A comment that adds nothing the code does
not already say should not exist.

Never invent a rationale you do not actually know. A plausible but wrong "why" is worse than none, because readers will
trust it. The same goes for a rationale that used to be true: when a change alters a contract, invariant, or failure
mode, update the nearby docstring in the same change.

NOTE: This is deliberately more documentation than this codebase currently has. Sparse existing comments in a file are
not a reason to keep new code under-documented; the target is readability by someone without detailed code knowledge,
and that target does not move because a neighboring function is terse.

Tests get the same treatment: a test docstring says why the behavior matters and states what is being checked in the
manner of a specification, so a reader can judge whether the test is right without reverse-engineering the assertions.

# Rust

Run these checks before handing off code changes when the toolchain is available:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets`
- `cargo test --workspace`
- `dprint check`

Use `tracing` for diagnostics. Do not add the `log` ecosystem unless there is a specific compatibility bridge and the
reason is documented.

# Conventional Commits

All commit messages and PR titles must use Conventional Commit format: `<type>: <short summary>`

Allowed types: `feat`, `fix`, `docs`, `perf`, `refactor`, `style`, `test`, `chore`, `ci`, `revert`.

Append `!` after the type for breaking changes (e.g. `feat!: remove legacy endpoint`). Scope is optional.

Rules:

- Type reflects the user-visible effect, not the implementation activity. A bug fix that requires heavy refactoring is
  `fix`, not `refactor`. A new CLI flag is `feat`, not `chore`.
- The summary after the colon is lowercase, imperative mood, no trailing period.
- Keep the first line under 72 characters.
