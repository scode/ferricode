use clap::{Parser, Subcommand};
use ferricode_core::{Harness, HarnessRequest, ModelProvider};
use ferricode_openai_codex::{OpenAiCodexProvider, authenticate_openai_codex, default_auth_path};
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();

    let cli = Cli::parse();
    let provider = OpenAiCodexProvider::from_default_auth_path()?;
    let output = run(cli, &provider).await?;
    println!("{output}");

    Ok(())
}

/// Runs a parsed CLI request and returns the text the process should print.
///
/// Keeping this separate from `main` lets tests cover non-auth command behavior
/// without spawning the binary or installing a tracing subscriber.
async fn run(
    cli: Cli,
    provider: &impl ModelProvider,
) -> Result<String, Box<dyn std::error::Error>> {
    match cli.command {
        Command::Auth { command } => run_auth(command).await,
        Command::Run { prompt, cwd } => {
            let request = HarnessRequest::new(prompt, cwd)?;
            let response = Harness::new().handle(&request, provider).await?;
            info!(summary = response.summary(), "handled harness request");
            Ok(response.summary().to_owned())
        }
        Command::Tui { prompt, cwd } => {
            let request = HarnessRequest::new(prompt, cwd)?;
            let response = ferricode_tui::launch(request, provider).await?;
            info!(summary = response.summary(), "handled tui harness request");
            Ok(response.summary().to_owned())
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
    use ferricode_core::{ModelProvider, ProviderError, ProviderRequest, ProviderTurn, ToolOutput};

    struct StaticProvider;

    impl ModelProvider for StaticProvider {
        type State = ();

        async fn start<'a>(
            &'a self,
            request: &'a ProviderRequest,
        ) -> Result<ProviderTurn<Self::State>, ProviderError> {
            Ok(ProviderTurn::Final(format!(
                "provider response from {}: {}",
                request.working_directory(),
                request.prompt()
            )))
        }

        async fn resume<'a>(
            &'a self,
            _state: Self::State,
            _tool_outputs: &'a [ToolOutput],
        ) -> Result<ProviderTurn<Self::State>, ProviderError> {
            unreachable!("static test provider never requests tools")
        }
    }

    #[tokio::test]
    async fn run_uses_default_cwd() {
        let cli = Cli::try_parse_from(["ferric", "run", "summarize task"]).unwrap();

        let output = super::run(cli, &StaticProvider).await.unwrap();

        assert_eq!(output, "provider response from .: summarize task");
    }

    #[tokio::test]
    async fn run_uses_explicit_cwd() {
        let cli =
            Cli::try_parse_from(["ferric", "run", "summarize task", "--cwd", "/work"]).unwrap();

        let output = super::run(cli, &StaticProvider).await.unwrap();

        assert_eq!(output, "provider response from /work: summarize task");
    }

    #[tokio::test]
    async fn tui_uses_core_request_contract() {
        let cli = Cli::try_parse_from(["ferric", "tui", "open screen", "--cwd", "/work"]).unwrap();

        let output = super::run(cli, &StaticProvider).await.unwrap();

        assert_eq!(output, "provider response from /work: open screen");
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
