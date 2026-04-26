# Ferricode

Ferricode is a Rust coding harness. The CLI binary is `ferric`.

This is only the project bootstrap. The current commands exercise the crate boundaries and tracing/Clap wiring; they do
not run an agent loop yet.

# Usage

Send a prompt through the core harness:

```sh
cargo run -p ferric -- run "inspect repository"
```

Both commands accept an explicit working directory context with `--cwd`:

```sh
cargo run -p ferric -- run "inspect repository" --cwd /path/to/repo
```

The `tui` subcommand currently uses the TUI crate boundary but still returns the same bootstrap response instead of
drawing a full terminal interface:

```sh
cargo run -p ferric -- tui "inspect repository" --cwd /path/to/repo
```

# Architecture

`SPEC.md` is the architectural contract. The important rule is that `ferricode-core` owns UI-neutral harness policy,
`ferricode-tui` owns terminal presentation concerns, and `ferric` owns process setup such as argument parsing and
tracing subscriber initialization.
