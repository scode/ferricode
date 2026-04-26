# Ferricode SPEC

Ferricode is a Rust coding harness. The command line tool is `ferric`.

This file is intentionally small right now. Treat it as the architectural contract for the repository, not as a product
roadmap.

# Architecture

The harness core and the terminal UI must stay separate.

`ferricode-core` owns the UI-neutral harness model: request types, planning and execution policy, provider traits, tool
orchestration, and state transitions. It must not depend on terminal UI crates, command-line parsing, process-global
logging setup, provider implementations, or direct user input.

`ferricode-tui` owns terminal presentation and interaction. It may depend on `ferricode-core`; `ferricode-core` must not
depend on it. The TUI should render core-owned state and send user intent back through core-owned APIs instead of
duplicating harness policy.

During the bootstrap phase, `ferricode-tui` may return a minimal displayable response instead of taking over the
terminal. That exception exists only so the crate boundary can be tested before a real TUI framework is introduced; once
the TUI starts rendering screens or handling input, that responsibility belongs inside `ferricode-tui`.

`ferric` is the CLI binary. It owns argument parsing, process setup, logging subscriber initialization, exit behavior,
and choosing whether a request runs through the direct CLI path or the TUI path.

# Model Providers

Provider implementations must live outside `ferricode-core`. The core crate may define a minimal provider trait and the
model-facing request type that trait consumes, but concrete provider code belongs in provider-specific crates.

The harness request type is not the provider request type. `HarnessRequest` is the user-facing harness input contract;
providers receive the narrower model-facing request that the harness chooses to build from it. Those types may look the
same during bootstrap, but they must not be treated as interchangeable.

The bootstrap provider interface is intentionally small: one prompt and working-directory context in, one assistant text
response out. Do not add registries, model selection, public streaming APIs, tool calls, multi-turn state, or provider
fallback until the harness actually needs them.

The first provider is `ferricode-openai-codex`. Its public provider name is `openai-codex`. It uses Codex-compatible
ChatGPT OAuth, not the OpenAI Platform API key path. For now it hardcodes `gpt-5.4` and medium reasoning effort.

# Authentication

The bootstrap credential store is `~/.ferric/auth.toml`. This is an intentional UX choice, not an XDG path.

`ferric auth openai-codex` starts browser PKCE OAuth against `https://auth.openai.com`. It uses the Codex public client
ID (`app_EMoamEEZ73f0CkXaXp7hrann`).

The browser callback is fixed to `http://localhost:1455/auth/callback`. The listener binds `127.0.0.1:1455`, while the
OAuth redirect URI remains `localhost` to match the Codex flow. If the port is unavailable, the command should fail
clearly.

The authorization request must include PKCE (`code_challenge_method=S256`), a generated state value, scopes
`openid profile email offline_access api.connectors.read api.connectors.invoke`, `id_token_add_organizations=true`,
`codex_cli_simplified_flow=true`, and `originator=codex_cli_rs`. The token exchange and token refresh requests use
`client_id` with the authorization code verifier or refresh token. They must not send a client secret.

Successful auth stores returned tokens and account metadata in `~/.ferric/auth.toml` under `openai_codex`. Auth file
writes require Unix-style private file permissions for now; platforms where the crate cannot create private token files
must fail rather than store long-lived credentials with ambient default permissions.

# Dependency Rules

Core dependencies should be chosen as if `ferricode-core` may later be embedded in tests, other binaries, or an RPC
service. Avoid process-global side effects in the core crate.

The binary may depend on both core and TUI crates. The TUI may depend on core. Core must not depend on either.

The binary may depend on provider implementation crates. Provider implementation crates may depend on core. Core must
not depend on provider implementation crates.

# Logging

Use `tracing` for diagnostics. Libraries should emit tracing events when useful, but subscriber initialization belongs
at the process boundary.

# Command Line

Use `clap` for command-line parsing. CLI types should stay in the `ferric` crate unless there is a strong reason to
share a UI-neutral concept through `ferricode-core`.

# Tests

Tests should enforce the crate boundary where practical. Do not make tests mutate process environment variables; use
dependency injection or explicit configuration instead.

# Current Non-Goals

Do not add OpenAI Platform API-key support, model flags, reasoning-effort flags, public streaming APIs, tool calls,
multi-turn state, keyring storage, or real TUI rendering in the bootstrap provider change. Provider internals may parse
buffered SSE when the upstream transport requires it, but that must not leak into the core provider trait yet.
