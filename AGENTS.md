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

# Conventional Commits

All commit messages and PR titles must use Conventional Commit format: `<type>: <short summary>`

Allowed types: `feat`, `fix`, `docs`, `perf`, `refactor`, `style`, `test`, `chore`, `ci`, `revert`.

Append `!` after the type for breaking changes (e.g. `feat!: remove legacy endpoint`). Scope is optional.

Rules:

- Type reflects the user-visible effect, not the implementation activity. A bug fix that requires heavy refactoring is
  `fix`, not `refactor`. A new CLI flag is `feat`, not `chore`.
- The summary after the colon is lowercase, imperative mood, no trailing period.
- Keep the first line under 72 characters.
