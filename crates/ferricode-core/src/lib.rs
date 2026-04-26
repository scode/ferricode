//! Harness primitives that do not know how the user interface is rendered.
//!
//! The core crate is deliberately small at bootstrap time. Its job is to hold
//! the contracts that the CLI, TUI, and future automation surfaces will share
//! without letting terminal rendering, command-line parsing, or process-global
//! logging leak into the harness model.

/// The user request and working directory context supplied to the harness.
///
/// This type is intentionally UI-neutral. Callers may collect the prompt from a
/// CLI argument, a TUI input widget, or a future RPC boundary, but the harness
/// should see the same semantic request either way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessRequest {
    prompt: String,
    working_directory: String,
}

impl HarnessRequest {
    /// Builds a request from caller-owned text.
    ///
    /// Empty prompts are rejected here rather than in the CLI or TUI so every
    /// front end gets the same contract. The working directory is currently a
    /// string because the harness does not yet perform filesystem operations;
    /// once it does, this should become a path-oriented type at the boundary
    /// where that behavior is introduced.
    pub fn new(
        prompt: impl Into<String>,
        working_directory: impl Into<String>,
    ) -> Result<Self, HarnessError> {
        let prompt = prompt.into();
        if prompt.trim().is_empty() {
            return Err(HarnessError::EmptyPrompt);
        }

        Ok(Self {
            prompt,
            working_directory: working_directory.into(),
        })
    }

    /// Returns the exact prompt text supplied by the caller.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Returns the working directory context the harness should treat as root.
    pub fn working_directory(&self) -> &str {
        &self.working_directory
    }
}

/// A harness response that can be rendered by any user interface.
///
/// The response is deliberately small during bootstrap. Keeping it in the core
/// crate from the start makes the separation explicit: front ends render
/// core-owned output, but they do not decide what the harness intends to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessResponse {
    summary: String,
}

impl HarnessResponse {
    /// Creates a response summary meant for display, logging, or later execution.
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
        }
    }

    /// Returns a stable human-readable summary of the harness response.
    pub fn summary(&self) -> &str {
        &self.summary
    }
}

/// The UI-independent coding harness coordinator.
///
/// The harness owns policy and task orchestration. It does not parse command
/// lines, initialize logging subscribers, draw terminal widgets, or read from
/// stdin directly. Those decisions belong at the shell-specific edges.
#[derive(Debug, Default)]
pub struct Harness;

impl Harness {
    /// Constructs a harness with the default bootstrap configuration.
    pub fn new() -> Self {
        Self
    }

    /// Handles a user request and returns UI-neutral output.
    ///
    /// This is intentionally deterministic for now. Future model calls, tool
    /// execution, or repository inspection should preserve the same boundary:
    /// caller-provided context enters through `HarnessRequest`, and UI-neutral
    /// results come back out.
    pub fn handle(&self, request: &HarnessRequest) -> HarnessResponse {
        HarnessResponse::new(format!(
            "Received coding task from {}: {}",
            request.working_directory(),
            request.prompt()
        ))
    }
}

/// Errors that can be reported before the harness starts doing work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessError {
    /// The harness cannot reason about a request without user intent.
    EmptyPrompt,
}

impl std::fmt::Display for HarnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPrompt => f.write_str("prompt must not be empty"),
        }
    }
}

impl std::error::Error for HarnessError {}

#[cfg(test)]
mod tests {
    use super::{Harness, HarnessError, HarnessRequest};

    #[test]
    fn rejects_empty_prompts() {
        assert_eq!(
            HarnessRequest::new("   ", ".").unwrap_err(),
            HarnessError::EmptyPrompt
        );
    }

    #[test]
    fn handles_request_context() {
        let harness = Harness::new();
        let request = HarnessRequest::new("inspect failures", "/work").unwrap();

        let response = harness.handle(&request);

        assert_eq!(
            response.summary(),
            "Received coding task from /work: inspect failures"
        );
    }
}
