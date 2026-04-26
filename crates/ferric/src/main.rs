use clap::{Parser, Subcommand};
use ferricode_core::{Harness, HarnessRequest};
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();

    let cli = Cli::parse();
    let output = run(cli)?;
    println!("{output}");

    Ok(())
}

/// Runs a parsed CLI request and returns the text the process should print.
///
/// Keeping this separate from `main` lets tests cover the command contract
/// without spawning the binary, installing a tracing subscriber, or depending
/// on process-global state.
fn run(cli: Cli) -> Result<String, Box<dyn std::error::Error>> {
    match cli.command {
        Command::Run { prompt, cwd } => {
            let request = HarnessRequest::new(prompt, cwd)?;
            let response = Harness::new().handle(&request);
            info!(summary = response.summary(), "handled harness request");
            Ok(response.summary().to_owned())
        }
        Command::Tui { prompt, cwd } => {
            let request = HarnessRequest::new(prompt, cwd)?;
            let response = ferricode_tui::launch(request);
            info!(summary = response.summary(), "handled tui harness request");
            Ok(response.summary().to_owned())
        }
    }
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

    #[test]
    fn run_uses_default_cwd() {
        let cli = Cli::try_parse_from(["ferric", "run", "inspect repository"]).unwrap();

        let output = super::run(cli).unwrap();

        assert_eq!(output, "Received coding task from .: inspect repository");
    }

    #[test]
    fn run_uses_explicit_cwd() {
        let cli =
            Cli::try_parse_from(["ferric", "run", "inspect repository", "--cwd", "/work"]).unwrap();

        let output = super::run(cli).unwrap();

        assert_eq!(
            output,
            "Received coding task from /work: inspect repository"
        );
    }

    #[test]
    fn tui_uses_core_request_contract() {
        let cli = Cli::try_parse_from(["ferric", "tui", "open screen", "--cwd", "/work"]).unwrap();

        let output = super::run(cli).unwrap();

        assert_eq!(output, "Received coding task from /work: open screen");
    }

    #[test]
    fn empty_prompt_is_rejected_after_parsing() {
        let cli = Cli::try_parse_from(["ferric", "run", "   "]).unwrap();

        let error = super::run(cli).unwrap_err();

        assert_eq!(error.to_string(), "prompt must not be empty");
    }
}
