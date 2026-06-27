# Ferricode

NOTE: This is a playground and learning project. I am sharing it publicly in case it is interesting to someone else, but
it is not meant for actual use at this time.

Ferricode is a Rust coding harness. The CLI binary is `ferric`.

This is still early bootstrap code. It has an OpenAI Codex-compatible auth path and a small read-only built-in tool
loop, but it does not have MCP, mutation tools, public streaming, or a useful TUI yet.

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
cargo run -p ferric -- run "summarize this repository"
```

`run` sends the prompt and working-directory context to the remote Codex backend. The model can ask Ferricode to run
built-in read-only tools when it needs local repository context. Both `run` and `tui` accept an explicit working
directory context with `--cwd`:

```sh
cargo run -p ferric -- run "summarize this repository" --cwd /path/to/repo
```

See `docs/tools.md` for the current built-in tool behavior and filesystem limits.

The `tui` subcommand currently uses the TUI crate boundary and prints the same harness response instead of drawing a
full terminal interface:

```sh
cargo run -p ferric -- tui "summarize this repository" --cwd /path/to/repo
```

# Architecture

`SPEC.md` is the architectural contract. The important rule is that `ferricode-core` owns UI-neutral harness policy,
`ferricode-tui` owns terminal presentation concerns, and `ferric` owns process setup such as argument parsing and
tracing subscriber initialization.
