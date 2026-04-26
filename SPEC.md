# Ferricode SPEC

Ferricode is a Rust coding harness. The command line tool is `ferric`.

This file is intentionally small right now. Treat it as the architectural contract for the repository, not as a product
roadmap.

# Architecture

The harness core and the terminal UI must stay separate.

`ferricode-core` owns the UI-neutral harness model: request types, planning and execution policy, tool orchestration,
state transitions, and any future model/provider integration. It must not depend on terminal UI crates, command-line
parsing, process-global logging setup, or direct user input.

`ferricode-tui` owns terminal presentation and interaction. It may depend on `ferricode-core`; `ferricode-core` must not
depend on it. The TUI should render core-owned state and send user intent back through core-owned APIs instead of
duplicating harness policy.

During the bootstrap phase, `ferricode-tui` may return a minimal displayable response instead of taking over the
terminal. That exception exists only so the crate boundary can be tested before a real TUI framework is introduced; once
the TUI starts rendering screens or handling input, that responsibility belongs inside `ferricode-tui`.

`ferric` is the CLI binary. It owns argument parsing, process setup, logging subscriber initialization, exit behavior,
and choosing whether a request runs through the direct CLI path or the TUI path.

# Dependency Rules

Core dependencies should be chosen as if `ferricode-core` may later be embedded in tests, other binaries, or an RPC
service. Avoid process-global side effects in the core crate.

The binary may depend on both core and TUI crates. The TUI may depend on core. Core must not depend on either.

# Logging

Use `tracing` for diagnostics. Libraries should emit tracing events when useful, but subscriber initialization belongs
at the process boundary.

# Command Line

Use `clap` for command-line parsing. CLI types should stay in the `ferric` crate unless there is a strong reason to
share a UI-neutral concept through `ferricode-core`.

# Tests

Tests should enforce the crate boundary where practical. Do not make tests mutate process environment variables; use
dependency injection or explicit configuration instead.
