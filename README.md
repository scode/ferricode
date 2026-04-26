# Ferricode

Ferricode is a Rust coding harness. The CLI binary is `ferric`.

This is still early bootstrap code. It has a minimal provider boundary and an OpenAI Codex-compatible auth path, but it
does not have a real agent loop, tools, streaming, or a useful TUI yet.

NOTE: The `openai-codex` provider is not the OpenAI Platform API-key flow. It uses Codex-compatible ChatGPT OAuth in the
browser and stores the returned account tokens in `~/.ferric/auth.toml`.

# Usage

Sign in with OpenAI Codex auth:

```sh
cargo run -p ferric -- auth openai-codex
```

The command starts a short-lived callback listener on `127.0.0.1:1455`, tries to open the browser, and also prints the
auth URL so you can open it manually.

After auth succeeds, send a prompt through the OpenAI Codex provider:

```sh
cargo run -p ferric -- run "inspect repository"
```

`run` sends the prompt and working-directory context to the remote Codex backend. Both `run` and `tui` accept an
explicit working directory context with `--cwd`:

```sh
cargo run -p ferric -- run "inspect repository" --cwd /path/to/repo
```

The `tui` subcommand currently uses the TUI crate boundary and prints the same provider-backed remote response instead
of drawing a full terminal interface:

```sh
cargo run -p ferric -- tui "inspect repository" --cwd /path/to/repo
```

# Architecture

`SPEC.md` is the architectural contract. The important rule is that `ferricode-core` owns UI-neutral harness policy,
`ferricode-tui` owns terminal presentation concerns, and `ferric` owns process setup such as argument parsing and
tracing subscriber initialization.
