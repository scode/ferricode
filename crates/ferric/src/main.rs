use clap::{Parser, Subcommand};
use ferricode_core::{
    Harness, HarnessEvent, HarnessEventSink, HarnessRequest, ModelProvider, NoopEventSink,
};
use ferricode_openai_codex::{OpenAiCodexProvider, authenticate_openai_codex, default_auth_path};
use std::future::Future;
use std::io::Write;
use std::path::Path;
use std::pin::Pin;
use std::sync::Mutex;
use tracing::info;
use tracing_subscriber::EnvFilter;

/// Ferricode command line entry point.
///
/// The binary owns process concerns: argument parsing, logging setup, exit
/// behavior, and choosing which front end should handle a request. Harness
/// decisions stay in `ferricode-core`.
#[derive(Debug, Parser)]
#[command(name = "ferric", version, about = "Run the Ferricode coding harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Configure or refresh credentials stored on this machine.
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },

    /// Send a prompt through the core harness.
    Run {
        /// User intent to pass into the harness.
        prompt: String,

        /// Working directory context for the harness.
        #[arg(long, default_value = ".")]
        cwd: String,
    },

    /// Send a prompt through the bootstrap TUI boundary.
    Tui {
        /// User intent to pass into the harness.
        prompt: String,

        /// Working directory context for the harness.
        #[arg(long, default_value = ".")]
        cwd: String,
    },
}

#[derive(Debug, Subcommand)]
enum AuthCommand {
    /// Sign in with Codex-compatible OpenAI OAuth.
    OpenaiCodex,
}

/// Process entry point: owns the exit policy and nothing else.
///
/// Every failure prints its `Display` text behind an `error:` prefix, followed
/// by the `source()` chain, and exits 1. Returning a `Result` from `main` would
/// instead print the error's `Debug` form, which for a missing-auth run showed
/// the struct wrapper around the message.
#[tokio::main]
async fn main() {
    init_tracing();

    if let Err(error) = try_main().await {
        eprintln!("error: {error}");
        let mut source = error.source();
        while let Some(cause) = source {
            eprintln!("  caused by: {cause}");
            source = cause.source();
        }
        std::process::exit(1);
    }
}

/// Everything `main` does that can fail, so `?` stays usable.
async fn try_main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let provider = OpenAiCodexProvider::from_default_auth_path()?;
    let mut stdout = std::io::stdout();
    if let Some(output) = run_with_tui_output(cli, &provider, &mut stdout).await? {
        println!("{output}");
    }
    Ok(())
}

/// Runs a parsed CLI request and returns the text the process should print.
///
/// Keeping this separate from `main` lets tests cover non-auth command behavior
/// without spawning the binary or installing a tracing subscriber. The `Tui`
/// command returns `None` on purpose: its output has already been streamed to
/// `tui_output` as it arrived, and returning the summary too would print the
/// same text twice.
async fn run_with_tui_output(
    cli: Cli,
    provider: &dyn ModelProvider,
    tui_output: &mut (dyn Write + Send),
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    match cli.command {
        Command::Auth { command } => Ok(Some(run_auth(command).await?)),
        Command::Run { prompt, cwd } => {
            let request = HarnessRequest::new(prompt, cwd)?;
            let response = Harness::new()
                .handle(&request, provider, &NoopEventSink)
                .await?;
            info!(summary = response.summary(), "handled harness request");
            Ok(Some(response.summary().to_owned()))
        }
        Command::Tui { prompt, cwd } => {
            let request = HarnessRequest::new(prompt, cwd)?;
            let sink = TuiOutputSink::new(tui_output);
            let response = ferricode_tui::launch(request, provider, &sink).await?;
            info!(summary = response.summary(), "handled tui harness request");
            Ok(None)
        }
    }
}

/// Runs a CLI command without retaining TUI event output for tests that only
/// inspect the returned string.
///
/// The process entry point supplies stdout to `run_with_tui_output`; this
/// wrapper exists so command tests that only inspect final CLI results do not
/// need to own an irrelevant writer.
#[cfg(test)]
async fn run(cli: Cli, provider: &dyn ModelProvider) -> Result<String, Box<dyn std::error::Error>> {
    let mut output = std::io::sink();
    Ok(run_with_tui_output(cli, provider, &mut output)
        .await?
        .unwrap_or_default())
}

/// Writes the subset of harness events the bootstrap TUI can render today.
///
/// The mutex makes a borrowed writer compatible with the event sink's `Sync`
/// contract. Write failures are intentionally ignored: an event sink is
/// informational, so terminal output must not alter harness control flow.
///
/// Text is rendered from deltas as they arrive. The completion event's text is
/// used only as a fallback: a provider that returns a turn without streaming
/// any delta (a future non-streaming provider, or a backend variant that emits
/// no text deltas) would otherwise leave the user staring at a blank line
/// while `ferric run` printed the summary.
struct TuiOutputSink<W> {
    state: Mutex<TuiOutputState<W>>,
}

/// Writer plus the one bit of per-turn memory the fallback needs.
struct TuiOutputState<W> {
    writer: W,
    saw_delta: bool,
}

impl<W> TuiOutputSink<W> {
    /// Wraps a caller-owned writer so tests can observe TUI output safely.
    fn new(writer: W) -> Self {
        Self {
            state: Mutex::new(TuiOutputState {
                writer,
                saw_delta: false,
            }),
        }
    }
}

impl<W: Write + Send> HarnessEventSink for TuiOutputSink<W> {
    fn on_event(&self, event: HarnessEvent) {
        // A poisoned lock must not turn an informational sink into a panic through the harness.
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match event {
            HarnessEvent::AssistantTextDelta { text } => {
                state.saw_delta = true;
                let _ = state.writer.write_all(text.as_bytes());
                let _ = state.writer.flush();
            }
            HarnessEvent::TurnFinished { text } => {
                if !text.is_empty() {
                    if !state.saw_delta {
                        let _ = state.writer.write_all(text.as_bytes());
                    }
                    let _ = writeln!(state.writer);
                    let _ = state.writer.flush();
                }
                state.saw_delta = false;
            }
            HarnessEvent::ToolCallStarted { .. } | HarnessEvent::ToolCallFinished { .. } => {}
        }
    }
}

async fn run_auth(command: AuthCommand) -> Result<String, Box<dyn std::error::Error>> {
    let mut stdout = std::io::stdout();
    run_auth_with(command, &mut stdout, authenticate_openai_codex_boxed).await
}

type AuthFuture<'a> = Pin<Box<dyn Future<Output = Result<(), Box<dyn std::error::Error>>> + 'a>>;
type Authenticator = for<'a> fn(&'a Path, &'a mut dyn std::io::Write) -> AuthFuture<'a>;

async fn run_auth_with(
    command: AuthCommand,
    output: &mut dyn std::io::Write,
    authenticate: Authenticator,
) -> Result<String, Box<dyn std::error::Error>> {
    match command {
        AuthCommand::OpenaiCodex => {
            let path = default_auth_path()?;
            authenticate(&path, output).await?;
            Ok(format!("Updated OpenAI Codex tokens in {}", path.display()))
        }
    }
}

fn authenticate_openai_codex_boxed<'a>(
    path: &'a Path,
    output: &'a mut dyn std::io::Write,
) -> AuthFuture<'a> {
    Box::pin(async move {
        authenticate_openai_codex(path, output).await?;
        Ok(())
    })
}

/// Initializes tracing from `RUST_LOG` without making libraries process-aware.
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
}

#[cfg(test)]
mod tests {
    use super::{Cli, Parser};
    use ferricode_core::{
        HarnessEvent, HarnessEventSink, ModelProvider, ProviderError, ProviderErrorKind,
        ProviderFuture, ProviderRequest, ProviderTurn, Transcript, TranscriptItem,
    };

    /// Supplies deterministic final text and one matching delta, allowing CLI
    /// tests to exercise the same event contract a streaming provider uses.
    struct StaticProvider;

    impl ModelProvider for StaticProvider {
        fn complete<'a>(
            &'a self,
            request: &'a ProviderRequest,
            _: &'a Transcript,
            sink: &'a dyn HarnessEventSink,
        ) -> ProviderFuture<'a> {
            Box::pin(async move {
                let text = format!(
                    "provider response from {}: {}",
                    request.working_directory().display(),
                    request.prompt()
                );
                sink.on_event(HarnessEvent::AssistantTextDelta { text: text.clone() });
                Ok(ProviderTurn::Final {
                    items: vec![TranscriptItem::AssistantMessage { text: text.clone() }],
                    text,
                })
            })
        }
    }

    /// A provider that must never be reached; used to prove that request
    /// validation happens before any model call.
    struct UnreachableProvider;

    impl ModelProvider for UnreachableProvider {
        fn complete<'a>(
            &'a self,
            _: &'a ProviderRequest,
            _: &'a Transcript,
            _: &'a dyn HarnessEventSink,
        ) -> ProviderFuture<'a> {
            Box::pin(async { panic!("provider must not be called for an invalid request") })
        }
    }

    /// Returns a classified auth failure so this test exercises the CLI's
    /// message boundary without reading credentials or contacting a backend.
    struct AuthRequiredProvider;

    impl ModelProvider for AuthRequiredProvider {
        fn complete<'a>(
            &'a self,
            _: &'a ProviderRequest,
            _: &'a Transcript,
            _: &'a dyn HarnessEventSink,
        ) -> ProviderFuture<'a> {
            Box::pin(async {
                Err(ProviderError::new(
                    ProviderErrorKind::AuthRequired,
                    "authentication required",
                ))
            })
        }
    }

    /// The default `--cwd` is `.`, which the harness resolves to the process's
    /// canonical current directory before the provider sees it.
    #[tokio::test]
    async fn run_uses_default_cwd() {
        let cli = Cli::try_parse_from(["ferric", "run", "summarize task"]).unwrap();
        let expected = std::env::current_dir().unwrap().canonicalize().unwrap();

        let output = super::run(cli, &StaticProvider).await.unwrap();

        assert_eq!(
            output,
            format!(
                "provider response from {}: summarize task",
                expected.display()
            )
        );
    }

    /// An explicit `--cwd` reaches the provider in canonical form, not as typed.
    #[tokio::test]
    async fn run_uses_explicit_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().to_str().unwrap();
        let cli = Cli::try_parse_from(["ferric", "run", "summarize task", "--cwd", cwd]).unwrap();

        let output = super::run(cli, &StaticProvider).await.unwrap();

        assert_eq!(
            output,
            format!(
                "provider response from {}: summarize task",
                dir.path().canonicalize().unwrap().display()
            )
        );
    }

    /// The `tui` subcommand passes canonical `--cwd` through core and prints
    /// streamed deltas without repeating the completed response summary.
    #[tokio::test]
    async fn tui_uses_core_request_contract() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().to_str().unwrap();
        let cli = Cli::try_parse_from(["ferric", "tui", "open screen", "--cwd", cwd]).unwrap();

        let mut output = Vec::new();
        let returned = super::run_with_tui_output(cli, &StaticProvider, &mut output)
            .await
            .unwrap();

        assert_eq!(
            output,
            format!(
                "provider response from {}: open screen\n",
                dir.path().canonicalize().unwrap().display()
            )
            .into_bytes()
        );
        assert_eq!(returned, None);
    }

    /// The bootstrap TUI renders streamed text exactly once, making its writer
    /// injectable instead of relying on process-global stdout in tests.
    #[test]
    fn tui_sink_writes_deltas_to_an_injected_writer() {
        let mut output = Vec::new();
        {
            let sink = super::TuiOutputSink::new(&mut output);
            sink.on_event(HarnessEvent::AssistantTextDelta {
                text: "hel".to_string(),
            });
            sink.on_event(HarnessEvent::AssistantTextDelta {
                text: "lo".to_string(),
            });
            sink.on_event(HarnessEvent::TurnFinished {
                text: "hello".to_string(),
            });
        }

        assert_eq!(output, b"hello\n");
    }

    /// A `--cwd` that does not exist fails before the provider is contacted,
    /// with a message that names the path the user typed.
    #[tokio::test]
    async fn missing_cwd_fails_before_provider_call() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope");
        let cwd = missing.to_str().unwrap();
        let cli = Cli::try_parse_from(["ferric", "run", "summarize task", "--cwd", cwd]).unwrap();

        let error = super::run(cli, &UnreachableProvider).await.unwrap_err();

        assert!(
            error
                .to_string()
                .starts_with(&format!("working directory `{cwd}` is not usable"))
        );
    }

    /// `run` boxes provider errors into `Box<dyn Error>`; this guards that the
    /// boxing never wraps or decorates the message, so what `main` prints is
    /// the provider's own text. The `error:` prefix and exit code live in
    /// `main` and are not exercised here.
    #[tokio::test]
    async fn provider_error_display_is_message_only() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().to_str().unwrap();
        let cli = Cli::try_parse_from(["ferric", "run", "summarize task", "--cwd", cwd]).unwrap();

        let error = super::run(cli, &AuthRequiredProvider).await.unwrap_err();

        assert_eq!(error.to_string(), "authentication required");
    }

    #[tokio::test]
    async fn empty_prompt_is_rejected_after_parsing() {
        let cli = Cli::try_parse_from(["ferric", "run", "   "]).unwrap();

        let error = super::run(cli, &StaticProvider).await.unwrap_err();

        assert_eq!(error.to_string(), "prompt must not be empty");
    }

    #[test]
    fn auth_commands_parse() {
        Cli::try_parse_from(["ferric", "auth", "openai-codex"]).unwrap();
    }

    #[tokio::test]
    async fn auth_command_dispatches_openai_codex_auth() {
        fn authenticate<'a>(
            path: &'a std::path::Path,
            output: &'a mut dyn std::io::Write,
        ) -> super::AuthFuture<'a> {
            Box::pin(async move {
                assert!(path.ends_with(".ferric/auth.toml"));
                writeln!(output, "auth called")?;
                Ok(())
            })
        }

        let mut output = Vec::new();
        let message =
            super::run_auth_with(super::AuthCommand::OpenaiCodex, &mut output, authenticate)
                .await
                .unwrap();

        assert!(message.contains("Updated OpenAI Codex tokens in"));
        assert_eq!(String::from_utf8(output).unwrap(), "auth called\n");
    }
}
