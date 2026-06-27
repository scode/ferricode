# Project Contract

Stay conformant to `SPEC.md`. If an implementation choice conflicts with the spec, change the implementation or update
the spec in the same change with a clear reason.

Keep `docs/tools.md` up to date whenever built-in tool behavior changes. This includes tool names, arguments, filesystem
safety policy, output limits, truncation behavior, and user-visible tool errors.

# Rust

Run these checks before handing off code changes when the toolchain is available:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets`
- `cargo test --workspace`
- `dprint check`

Use `tracing` for diagnostics. Do not add the `log` ecosystem unless there is a specific compatibility bridge and the
reason is documented.

# Documentation While Writing Code

When writing code, err on the side of adding useful docstrings and comments even if the surrounding code is sparse. This
repo's existing style is not a reason to omit documentation for new or changed behavior. Follow more specific local
instructions if they explicitly ask for less documentation in a particular file or module, but do not infer that from
silence or from a historically under-documented file.

Document symbols and helpers with real behavior, invariants, lifecycle constraints, async/race behavior, security or
trust-boundary assumptions, persistence guarantees, test harness assumptions, or non-obvious failure modes. Prefer a
strong first sentence that states the contract or reason the symbol exists. Avoid filler comments that restate the code.

During review, do not mechanically demand docstrings everywhere. Still flag missing documentation when the change adds a
contract, invariant, portability assumption, or piece of context that a future reader would otherwise have to
rediscover.

# Conventional Commits

All commit messages and PR titles must use Conventional Commit format: `<type>: <short summary>`

Allowed types: `feat`, `fix`, `docs`, `perf`, `refactor`, `style`, `test`, `chore`, `ci`, `revert`.

Append `!` after the type for breaking changes (e.g. `feat!: remove legacy endpoint`). Scope is optional.

Rules:

- Type reflects the user-visible effect, not the implementation activity. A bug fix that requires heavy refactoring is
  `fix`, not `refactor`. A new CLI flag is `feat`, not `chore`.
- The summary after the colon is lowercase, imperative mood, no trailing period.
- Keep the first line under 72 characters.
