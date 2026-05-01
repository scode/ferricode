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

/// The model-facing request produced by the harness.
///
/// This deliberately is not a type alias for `HarnessRequest`. The harness
/// request is the public input contract for Ferricode, while this is the
/// smaller contract providers receive after the harness has decided what text
/// and context should be sent to a model. The fields are equivalent today, but
/// callers should not assume they will stay that way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRequest {
    prompt: String,
    working_directory: String,
}

impl ProviderRequest {
    /// Builds the narrow request a provider needs to produce assistant text.
    pub fn new(prompt: impl Into<String>, working_directory: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            working_directory: working_directory.into(),
        }
    }

    /// Returns the prompt text selected by the harness for the provider.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Returns the working directory context selected by the harness.
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

/// A provider that can turn a model-facing request into assistant text.
///
/// This is intentionally narrower than a real agent backend. It does not model
/// streaming, tool calls, model choice, or conversation state. Those concepts
/// should enter the trait only when the harness has real behavior that needs
/// them.
pub trait ModelProvider {
    /// Produces assistant text for a single request.
    fn respond<'a>(
        &'a self,
        request: &'a ProviderRequest,
    ) -> impl std::future::Future<Output = Result<String, ProviderError>> + Send + 'a;
}

/// Provider failures surfaced through the harness boundary.
///
/// The core crate keeps provider errors as user-facing text for now because the
/// bootstrap harness has no recovery policy beyond reporting the failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderError {
    message: String,
}

impl ProviderError {
    /// Creates a provider error with an actionable message for the caller.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProviderError {}

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

    /// Handles a user request through the supplied provider.
    ///
    /// The harness stays responsible for orchestration, while the provider owns
    /// how assistant text is produced. This keeps concrete model backends out
    /// of the core crate.
    pub async fn handle(
        &self,
        request: &HarnessRequest,
        provider: &impl ModelProvider,
    ) -> Result<HarnessResponse, ProviderError> {
        let provider_request = ProviderRequest::new(request.prompt(), request.working_directory());
        let summary = provider.respond(&provider_request).await?;
        Ok(HarnessResponse::new(summary))
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
    use super::{
        Harness, HarnessError, HarnessRequest, ModelProvider, ProviderError, ProviderRequest,
    };

    struct EchoProvider;

    impl ModelProvider for EchoProvider {
        async fn respond<'a>(
            &'a self,
            request: &'a ProviderRequest,
        ) -> Result<String, ProviderError> {
            Ok(format!(
                "provider saw {} from {}",
                request.prompt(),
                request.working_directory()
            ))
        }
    }

    struct FailingProvider;

    impl ModelProvider for FailingProvider {
        async fn respond<'a>(
            &'a self,
            _request: &'a ProviderRequest,
        ) -> Result<String, ProviderError> {
            Err(ProviderError::new("provider failed"))
        }
    }

    #[test]
    fn rejects_empty_prompts() {
        assert_eq!(
            HarnessRequest::new("   ", ".").unwrap_err(),
            HarnessError::EmptyPrompt
        );
    }

    #[tokio::test]
    async fn handles_request_context_through_provider() {
        let harness = Harness::new();
        let request = HarnessRequest::new("inspect failures", "/work").unwrap();

        let response = harness.handle(&request, &EchoProvider).await.unwrap();

        assert_eq!(
            response.summary(),
            "provider saw inspect failures from /work"
        );
    }

    #[tokio::test]
    async fn provider_errors_cross_the_harness_boundary() {
        let harness = Harness::new();
        let request = HarnessRequest::new("inspect failures", "/work").unwrap();

        let error = harness
            .handle(&request, &FailingProvider)
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "provider failed");
    }
}
