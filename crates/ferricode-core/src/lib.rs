//! Harness primitives that do not know how the user interface is rendered.
//!
//! The core crate holds the contracts and policy shared by the CLI, TUI, and
//! future automation surfaces. Provider crates adapt those contracts to a
//! concrete model backend, but tool orchestration and local filesystem policy
//! stay here so every front end gets the same behavior.

mod events;
mod tools;
mod transcript;

pub use events::{HarnessEvent, HarnessEventSink, NoopEventSink};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use tools::execute_tool_calls;
pub use tools::{ToolCall, ToolDefinition, ToolOutput, built_in_tools};
pub use transcript::{Transcript, TranscriptItem};

const MAX_TOOL_TURNS: usize = 32;

/// The system prompt sent with every model request.
///
/// It says what the agent is and how it should use the built-in tools, so it
/// is harness policy rather than provider configuration: core owns the
/// wording and providers relay it verbatim (`ProviderRequest::instructions`).
/// Changing this text changes model behavior on every request.
pub const DEFAULT_INSTRUCTIONS: &str = "You are Ferricode, a coding harness. Use the built-in filesystem tools when the user's request requires repository context. Start with a directory listing when you need to understand the working directory, then read specific relevant text files. Do not ask for clarification when the request can be handled by inspecting files.";

/// The user request and working directory context supplied to the harness.
///
/// This type is intentionally UI-neutral. Callers may collect the prompt from a
/// CLI argument, a TUI input widget, or a future RPC boundary, but the harness
/// should see the same semantic request either way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessRequest {
    prompt: String,
    working_directory: PathBuf,
}

impl HarnessRequest {
    /// Builds a request from caller-owned text and a directory that must exist.
    ///
    /// Empty prompts are rejected here rather than in the CLI or TUI so every
    /// front end gets the same contract. The working directory is resolved to
    /// its canonical absolute form here too, and a path that does not exist or
    /// is not a directory is an error. Validating up front matters because the
    /// alternative, which this code used to do, was to send the prompt to the
    /// model and only discover the bad directory when a tool call tried to use
    /// it, surfacing as a tool error the model then reasoned about.
    ///
    /// Canonicalizing once establishes the precondition `resolve_tool_path`
    /// relies on (a canonical root) in one place, so tools no longer repeat the
    /// resolution on every call. It is done synchronously on purpose: this runs
    /// once, before any model call, and is not on the tool execution path where
    /// blocking I/O is forbidden.
    pub fn new(
        prompt: impl Into<String>,
        working_directory: impl AsRef<Path>,
    ) -> Result<Self, HarnessError> {
        let prompt = prompt.into();
        if prompt.trim().is_empty() {
            return Err(HarnessError::EmptyPrompt);
        }

        let requested = working_directory.as_ref();
        let invalid = |reason: String| HarnessError::InvalidWorkingDirectory {
            path: requested.display().to_string(),
            reason,
        };
        let working_directory =
            std::fs::canonicalize(requested).map_err(|error| invalid(error.to_string()))?;
        // `Path::is_dir` swallows metadata errors as `false`, which would report a
        // permission problem as "not a directory"; ask for the metadata directly.
        let metadata =
            std::fs::metadata(&working_directory).map_err(|error| invalid(error.to_string()))?;
        if !metadata.is_dir() {
            return Err(invalid("not a directory".to_string()));
        }

        Ok(Self {
            prompt,
            working_directory,
        })
    }

    /// Returns the exact prompt text supplied by the caller.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Returns the canonical, existing directory the harness treats as root.
    pub fn working_directory(&self) -> &Path {
        &self.working_directory
    }
}

/// The model-facing request produced by the harness.
///
/// This deliberately is not a type alias for `HarnessRequest`. The harness
/// request is the public input contract for Ferricode, while this is the
/// smaller contract providers receive after the harness has decided what text
/// and context should be sent to a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRequest {
    prompt: String,
    working_directory: PathBuf,
    instructions: &'static str,
}

impl ProviderRequest {
    /// Builds the narrow request a provider needs to produce assistant text.
    ///
    /// The system prompt is always `DEFAULT_INSTRUCTIONS`; there is no way for a
    /// caller to substitute one, because the prompt is harness policy rather
    /// than a per-request choice. (`DEFAULT_` anticipates a later per-run
    /// override, for example from a TUI setting; none exists today.) This
    /// constructor does not validate the
    /// directory; the harness has already done that when it built the
    /// `HarnessRequest`. Tests and other direct callers may pass any path, but
    /// tools fail containment on any root that is not the canonical absolute
    /// form (relative, symlinked, or containing `..`), not only on one that
    /// does not exist.
    pub fn new(prompt: impl Into<String>, working_directory: impl Into<PathBuf>) -> Self {
        Self {
            prompt: prompt.into(),
            working_directory: working_directory.into(),
            instructions: DEFAULT_INSTRUCTIONS,
        }
    }

    /// Returns the prompt text selected by the harness for the provider.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Returns the system prompt the provider must send verbatim with the
    /// request. Providers relay it; they do not compose their own.
    pub fn instructions(&self) -> &'static str {
        self.instructions
    }

    /// Returns the working directory the harness resolved, which tools treat as
    /// the containment root and providers may show to the model.
    pub fn working_directory(&self) -> &Path {
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
    transcript: Transcript,
}

impl HarnessResponse {
    /// Creates the final response and preserves the full conversation for UIs.
    ///
    /// This constructor is private because only the harness can promise that
    /// the transcript contains the user request, provider turns, and tool
    /// results in execution order.
    fn new(summary: impl Into<String>, transcript: Transcript) -> Self {
        Self {
            summary: summary.into(),
            transcript,
        }
    }

    /// Returns a stable human-readable summary of the harness response.
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// Returns the ordered conversation that produced this summary.
    pub fn transcript(&self) -> &Transcript {
        &self.transcript
    }
}

/// One model-request turn produced by a provider.
///
/// Providers return the transcript entries they produced alongside either
/// final text or tool calls. Core appends `items` exactly as supplied before it
/// returns or runs calls; a provider must therefore put opaque reasoning before
/// the calls it explains and retain backend output order within the vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderTurn {
    /// The provider completed the request with user-facing assistant text.
    ///
    /// Core rejects turns whose `items` lack the corresponding
    /// `AssistantMessage`. Items may also contain provider-only state that has
    /// to survive a future turn.
    Final {
        text: String,
        items: Vec<TranscriptItem>,
    },
    /// The provider needs core-owned tools before it can continue.
    ToolCalls {
        /// Entries to append before core executes the derived calls.
        ///
        /// Providers include every entry, opaque state included, in backend
        /// output order. Core derives the calls from the `ToolCall` entries.
        items: Vec<TranscriptItem>,
    },
}

impl ProviderTurn {
    /// Returns the transcript entries this turn produced, whichever kind it is.
    pub fn items(&self) -> &[TranscriptItem] {
        match self {
            Self::Final { items, .. } | Self::ToolCalls { items } => items,
        }
    }

    /// Returns the assistant text of this turn: every `AssistantMessage` entry
    /// concatenated in order, or an empty string when the turn produced none.
    ///
    /// This is the single definition of "what the model said this turn". Core
    /// uses it for the completion event and providers use it to forward
    /// buffered text, so the two never disagree.
    pub fn assistant_text(&self) -> String {
        self.items()
            .iter()
            .filter_map(|item| match item {
                TranscriptItem::AssistantMessage { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }
}

/// The boxed future used by the object-safe provider boundary.
pub type ProviderFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ProviderTurn, ProviderError>> + Send + 'a>>;

/// A provider that can drive one model interaction through core-owned tools.
///
/// The core trait intentionally remains narrow. It exposes progress only
/// through the core-owned informational sink, rather than a provider streaming
/// API, and otherwise gives the harness enough structure to run local built-in
/// tools and hand their outputs back to the same provider.
pub trait ModelProvider: Send + Sync {
    /// Renders the complete transcript and returns the next model turn.
    ///
    /// Providers are stateless between calls: they must derive all backend
    /// input from `request` and `transcript`, skipping opaque entries owned by
    /// other providers. They must not retain a hidden continuation token or
    /// assume this call follows an earlier call on the same instance.
    fn complete<'a>(
        &'a self,
        request: &'a ProviderRequest,
        transcript: &'a Transcript,
        sink: &'a dyn HarnessEventSink,
    ) -> ProviderFuture<'a>;
}

/// Coarse classification of a provider failure, so front ends can decide what to do
/// without parsing message text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorKind {
    /// No usable credentials; the front end should prompt the user to re-authenticate.
    AuthRequired,
    /// The request never got a response; the front end can retry the request.
    Network,
    /// The backend answered with a non-success status; the front end should report it.
    BackendStatus,
    /// The provider could not make sense of what it was given: a backend
    /// response in an unexpected shape, or a stored file it cannot interpret.
    /// Not retryable as-is; the front end should report it.
    Protocol,
    /// Any other failure, including local I/O and configuration errors; the front end should report it.
    Other,
}

/// Provider failures surfaced through the harness boundary.
///
/// The kind gives front ends recovery information without making them parse
/// the user-facing message. `Display` intentionally remains message-only so
/// errors are readable at the CLI boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderError {
    kind: ProviderErrorKind,
    message: String,
}

impl ProviderError {
    /// Creates a classified provider error with a message suitable for display.
    pub fn new(kind: ProviderErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Returns the coarse failure class that front ends can use for recovery.
    pub fn kind(&self) -> ProviderErrorKind {
        self.kind
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

    /// Handles a user request through the supplied provider and progress sink.
    ///
    /// The harness stays responsible for orchestration, including built-in tool
    /// execution. Provider crates own only the backend-specific request and
    /// response format needed to ask a model what to do next. Every successful
    /// provider turn receives a completion event before core validates,
    /// records, or executes that turn, so a sink sees the model-turn boundary
    /// independently of later harness policy.
    pub async fn handle(
        &self,
        request: &HarnessRequest,
        provider: &dyn ModelProvider,
        sink: &dyn HarnessEventSink,
    ) -> Result<HarnessResponse, ProviderError> {
        let provider_request = ProviderRequest::new(request.prompt(), request.working_directory());
        let mut transcript = Transcript::for_request(&provider_request);
        let mut tool_turns = 0;

        loop {
            let turn = provider
                .complete(&provider_request, &transcript, sink)
                .await?;
            sink.on_event(HarnessEvent::TurnFinished {
                text: turn.assistant_text(),
            });
            match turn {
                ProviderTurn::Final { text, items } => {
                    if !items
                        .iter()
                        .any(|item| matches!(item, TranscriptItem::AssistantMessage { .. }))
                    {
                        return Err(ProviderError::new(
                            ProviderErrorKind::Protocol,
                            "provider returned a final turn with no assistant message",
                        ));
                    }
                    transcript.extend(items);
                    return Ok(HarnessResponse::new(text, transcript));
                }
                ProviderTurn::ToolCalls { items } => {
                    if tool_turns == MAX_TOOL_TURNS {
                        // A harness policy limit, not a wire-shape problem, so
                        // `Other` rather than `Protocol`; nothing about the
                        // transport is broken.
                        return Err(ProviderError::new(
                            ProviderErrorKind::Other,
                            format!(
                                "model exceeded the built-in tool turn limit of {MAX_TOOL_TURNS}"
                            ),
                        ));
                    }
                    let calls = items
                        .iter()
                        .filter_map(|item| match item {
                            TranscriptItem::ToolCall(call) => Some(call.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    if calls.is_empty() {
                        return Err(ProviderError::new(
                            ProviderErrorKind::Protocol,
                            "provider returned a tool-call turn with no tool calls",
                        ));
                    }
                    tool_turns += 1;
                    transcript.extend(items);
                    let outputs = execute_tool_calls(&provider_request, calls, sink).await;
                    transcript.extend(outputs.into_iter().map(TranscriptItem::ToolResult));
                }
            }
        }
    }
}

/// Errors that can be reported before the harness starts doing work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessError {
    /// The harness cannot reason about a request without user intent.
    EmptyPrompt,
    /// The working directory does not exist, cannot be resolved, or is not a
    /// directory. `path` is what the caller passed, not a canonical form, so the
    /// user recognizes it in the message.
    InvalidWorkingDirectory { path: String, reason: String },
}

impl std::fmt::Display for HarnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPrompt => f.write_str("prompt must not be empty"),
            Self::InvalidWorkingDirectory { path, reason } => {
                write!(f, "working directory `{path}` is not usable: {reason}")
            }
        }
    }
}

impl std::error::Error for HarnessError {}

#[cfg(test)]
mod tests {
    use super::{
        Harness, HarnessError, HarnessEvent, HarnessEventSink, HarnessRequest, ModelProvider,
        NoopEventSink, ProviderError, ProviderErrorKind, ProviderFuture, ProviderRequest,
        ProviderTurn, ToolCall, ToolOutput, Transcript, TranscriptItem,
    };
    use serde_json::Value;
    use std::fs;
    use std::sync::Mutex;
    use tempfile::tempdir;

    struct EchoProvider;

    impl ModelProvider for EchoProvider {
        fn complete<'a>(
            &'a self,
            request: &'a ProviderRequest,
            _transcript: &'a Transcript,
            _: &'a dyn HarnessEventSink,
        ) -> ProviderFuture<'a> {
            Box::pin(async move {
                let text = format!(
                    "provider saw {} from {}",
                    request.prompt(),
                    request.working_directory().display()
                );
                Ok(ProviderTurn::Final {
                    items: vec![TranscriptItem::AssistantMessage { text: text.clone() }],
                    text,
                })
            })
        }
    }

    struct FailingProvider;

    impl ModelProvider for FailingProvider {
        fn complete<'a>(
            &'a self,
            _request: &'a ProviderRequest,
            _transcript: &'a Transcript,
            _: &'a dyn HarnessEventSink,
        ) -> ProviderFuture<'a> {
            Box::pin(async {
                Err(ProviderError::new(
                    ProviderErrorKind::Other,
                    "provider failed",
                ))
            })
        }
    }

    struct ScriptedProvider {
        calls: Vec<Vec<ToolCall>>,
        outputs: Mutex<Vec<Vec<ToolOutput>>>,
    }

    impl ScriptedProvider {
        fn new(calls: impl IntoIterator<Item = Vec<ToolCall>>) -> Self {
            Self {
                calls: calls.into_iter().collect::<Vec<_>>(),
                outputs: Mutex::new(Vec::new()),
            }
        }
    }

    /// Records informational events so event ordering can be asserted without
    /// coupling the harness tests to a terminal implementation.
    #[derive(Default)]
    struct RecordingSink {
        events: Mutex<Vec<HarnessEvent>>,
    }

    impl HarnessEventSink for RecordingSink {
        fn on_event(&self, event: HarnessEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    impl ModelProvider for ScriptedProvider {
        fn complete<'a>(
            &'a self,
            _request: &'a ProviderRequest,
            transcript: &'a Transcript,
            sink: &'a dyn HarnessEventSink,
        ) -> ProviderFuture<'a> {
            Box::pin(async move {
                let last_outputs = transcript
                    .items()
                    .rsplit(|item| !matches!(item, TranscriptItem::ToolResult(_)))
                    .next()
                    .unwrap();
                if !last_outputs.is_empty() {
                    self.outputs.lock().unwrap().push(
                        last_outputs
                            .iter()
                            .filter_map(|item| match item {
                                TranscriptItem::ToolResult(output) => Some(output.clone()),
                                _ => None,
                            })
                            .collect(),
                    );
                }
                if let Some(calls) = self.calls.get(self.outputs.lock().unwrap().len()) {
                    let calls = calls.clone();
                    let items = calls
                        .iter()
                        .cloned()
                        .map(TranscriptItem::ToolCall)
                        .collect();
                    Ok(ProviderTurn::ToolCalls { items })
                } else {
                    sink.on_event(HarnessEvent::AssistantTextDelta {
                        text: "done".to_string(),
                    });
                    Ok(ProviderTurn::Final {
                        text: "done".to_string(),
                        items: vec![TranscriptItem::AssistantMessage {
                            text: "done".to_string(),
                        }],
                    })
                }
            })
        }
    }

    #[test]
    fn rejects_empty_prompts() {
        assert_eq!(
            HarnessRequest::new("   ", ".").unwrap_err(),
            HarnessError::EmptyPrompt
        );
    }

    /// A bad `--cwd` must fail before any provider call, not surface later as
    /// a tool error the model has to reason about. Both a missing path and a
    /// path that exists but is a file are rejected, and the message names the
    /// path as the caller typed it.
    #[test]
    fn rejects_missing_or_non_directory_working_directory() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");
        let file = dir.path().join("file.txt");
        fs::write(&file, "x").unwrap();

        let missing_error = HarnessRequest::new("prompt", &missing).unwrap_err();
        let file_error = HarnessRequest::new("prompt", &file).unwrap_err();

        assert!(matches!(
            &missing_error,
            HarnessError::InvalidWorkingDirectory { path, .. } if path == &missing.display().to_string()
        ));
        assert!(missing_error.to_string().starts_with(&format!(
            "working directory `{}` is not usable",
            missing.display()
        )));
        assert!(matches!(
            file_error,
            HarnessError::InvalidWorkingDirectory { ref reason, .. } if reason == "not a directory"
        ));
    }

    /// The working directory handed to providers and tools is the canonical
    /// form: a `..` segment is resolved away and, on Unix, a symlinked `--cwd`
    /// resolves to its target. The symlink case is the one containment depends
    /// on, since `resolve_tool_path` compares canonical paths against this root.
    #[test]
    fn working_directory_is_canonicalized() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("a").join("b");
        fs::create_dir_all(&nested).unwrap();
        let expected = nested.canonicalize().unwrap();
        let dotted = dir.path().join("a").join("b").join("..").join("b");

        let request = HarnessRequest::new("prompt", &dotted).unwrap();

        assert_eq!(request.working_directory(), expected);

        #[cfg(unix)]
        {
            let link = dir.path().join("link");
            std::os::unix::fs::symlink(&nested, &link).unwrap();

            let request = HarnessRequest::new("prompt", &link).unwrap();

            assert_eq!(request.working_directory(), expected);
        }
    }

    #[tokio::test]
    async fn handles_request_context_through_provider() {
        let dir = tempdir().unwrap();
        let harness = Harness::new();
        let request = HarnessRequest::new("inspect failures", dir.path()).unwrap();

        let response = harness
            .handle(&request, &EchoProvider, &NoopEventSink)
            .await
            .unwrap();

        assert_eq!(
            response.summary(),
            format!(
                "provider saw inspect failures from {}",
                request.working_directory().display()
            )
        );
    }

    /// The public harness entry point must accept trait objects so front ends
    /// can choose providers at runtime without carrying provider generics.
    #[tokio::test]
    async fn handles_boxed_provider_trait_object() {
        let dir = tempdir().unwrap();
        let provider: Box<dyn ModelProvider> = Box::new(EchoProvider);
        let request = HarnessRequest::new("inspect failures", dir.path()).unwrap();

        let response = Harness::new()
            .handle(&request, provider.as_ref(), &NoopEventSink)
            .await
            .unwrap();

        assert!(response.summary().contains("inspect failures"));
    }

    #[tokio::test]
    async fn repository_prompts_are_not_special_cased() {
        let dir = tempdir().unwrap();
        let harness = Harness::new();
        let request = HarnessRequest::new("summarize this repository", dir.path()).unwrap();

        let response = harness
            .handle(&request, &EchoProvider, &NoopEventSink)
            .await
            .unwrap();

        assert_eq!(
            response.summary(),
            format!(
                "provider saw summarize this repository from {}",
                request.working_directory().display()
            )
        );
    }

    #[tokio::test]
    async fn provider_errors_cross_the_harness_boundary() {
        let dir = tempdir().unwrap();
        let harness = Harness::new();
        let request = HarnessRequest::new("inspect failures", dir.path()).unwrap();

        let sink = RecordingSink::default();
        let error = harness
            .handle(&request, &FailingProvider, &sink)
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "provider failed");
        // A provider that never produced a turn produced no completion event
        // either; a sink must not see "turn finished" for a request that failed.
        assert!(sink.events.lock().unwrap().is_empty());
    }

    /// The regression the event design exists to prevent: a tool-call turn that
    /// also carries assistant text must report that text in its completion
    /// event, not an empty string, so a front end that renders per-turn text
    /// sees what the model said before it called the tool.
    #[tokio::test]
    async fn turn_finished_carries_text_of_a_tool_call_turn() {
        struct TextThenCallProvider;

        impl ModelProvider for TextThenCallProvider {
            fn complete<'a>(
                &'a self,
                _: &'a ProviderRequest,
                transcript: &'a Transcript,
                sink: &'a dyn HarnessEventSink,
            ) -> ProviderFuture<'a> {
                Box::pin(async move {
                    let already_called = transcript
                        .items()
                        .iter()
                        .any(|item| matches!(item, TranscriptItem::ToolResult(_)));
                    if already_called {
                        return Ok(ProviderTurn::Final {
                            text: "done".to_string(),
                            items: vec![TranscriptItem::AssistantMessage {
                                text: "done".to_string(),
                            }],
                        });
                    }
                    sink.on_event(HarnessEvent::AssistantTextDelta {
                        text: "looking".to_string(),
                    });
                    Ok(ProviderTurn::ToolCalls {
                        items: vec![
                            TranscriptItem::AssistantMessage {
                                text: "looking".to_string(),
                            },
                            TranscriptItem::ToolCall(ToolCall::new(
                                "list",
                                "ferricode_list_directory",
                                r#"{"path":"."}"#,
                            )),
                        ],
                    })
                })
            }
        }

        let dir = tempdir().unwrap();
        let request = HarnessRequest::new("inspect", dir.path()).unwrap();
        let sink = RecordingSink::default();

        Harness::new()
            .handle(&request, &TextThenCallProvider, &sink)
            .await
            .unwrap();

        let events = sink.events.lock().unwrap();
        assert_eq!(
            events[0],
            HarnessEvent::AssistantTextDelta {
                text: "looking".to_string()
            }
        );
        assert_eq!(
            events[1],
            HarnessEvent::TurnFinished {
                text: "looking".to_string()
            }
        );
        assert!(matches!(events[2], HarnessEvent::ToolCallStarted { .. }));
        assert_eq!(
            *events.last().unwrap(),
            HarnessEvent::TurnFinished {
                text: "done".to_string()
            }
        );
    }

    /// A tool-call turn without a core tool call is malformed provider output,
    /// so the harness must reject it before it can enter an empty tool cycle.
    #[tokio::test]
    async fn tool_call_turn_without_tool_calls_is_a_protocol_error() {
        struct EmptyToolCallProvider;

        impl ModelProvider for EmptyToolCallProvider {
            fn complete<'a>(
                &'a self,
                _: &'a ProviderRequest,
                _: &'a Transcript,
                _: &'a dyn HarnessEventSink,
            ) -> ProviderFuture<'a> {
                Box::pin(async {
                    Ok(ProviderTurn::ToolCalls {
                        items: vec![TranscriptItem::ProviderOpaque {
                            provider: "test",
                            payload: serde_json::json!({ "type": "reasoning" }),
                        }],
                    })
                })
            }
        }

        let dir = tempdir().unwrap();
        let request = HarnessRequest::new("inspect", dir.path()).unwrap();

        let error = Harness::new()
            .handle(&request, &EmptyToolCallProvider, &NoopEventSink)
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ProviderErrorKind::Protocol);
        assert_eq!(
            error.to_string(),
            "provider returned a tool-call turn with no tool calls"
        );
    }

    #[tokio::test]
    async fn executes_list_then_read_across_tool_turns() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("README.md"), "hello").unwrap();
        let provider = ScriptedProvider::new([
            vec![ToolCall::new(
                "list",
                "ferricode_list_directory",
                r#"{"path":"."}"#,
            )],
            vec![ToolCall::new(
                "read",
                "ferricode_read_file",
                r#"{"path":"README.md"}"#,
            )],
        ]);
        let request = HarnessRequest::new("read project files", dir.path()).unwrap();

        let response = Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        assert_eq!(response.summary(), "done");
        let outputs = provider.outputs.lock().unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0][0].call_id(), "list");
        let listed = parse_output(&outputs[0][0]);
        assert_eq!(listed["ok"], true);
        assert_eq!(listed["entries"][0]["name"], "README.md");
        let read = parse_output(&outputs[1][0]);
        assert_eq!(read["ok"], true);
        assert_eq!(read["content"], "hello");
        assert!(matches!(
            response.transcript().items(),
            [
                TranscriptItem::UserMessage { .. },
                TranscriptItem::ToolCall(list),
                TranscriptItem::ToolResult(list_result),
                TranscriptItem::ToolCall(read),
                TranscriptItem::ToolResult(read_result),
                TranscriptItem::AssistantMessage { text },
            ] if list.id() == "list"
                && list_result.call_id() == "list"
                && read.id() == "read"
                && read_result.call_id() == "read"
                && text == "done"
        ));
    }

    /// Core completes every returned model turn before it runs that turn's
    /// tools, while provider text remains in the model turn that produced it.
    #[tokio::test]
    async fn emits_tool_and_provider_events_in_request_order() {
        let dir = tempdir().unwrap();
        let call = ToolCall::new("unknown", "ferricode_unknown", "{}");
        let provider = ScriptedProvider::new([vec![call.clone()]]);
        let sink = RecordingSink::default();
        let request = HarnessRequest::new("inspect", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &sink)
            .await
            .unwrap();

        assert_eq!(
            *sink.events.lock().unwrap(),
            vec![
                HarnessEvent::TurnFinished {
                    text: String::new(),
                },
                HarnessEvent::ToolCallStarted { call },
                HarnessEvent::ToolCallFinished {
                    output: ToolOutput::new(
                        "unknown",
                        r#"{"error":"unknown built-in tool `ferricode_unknown`","ok":false}"#,
                    ),
                },
                HarnessEvent::AssistantTextDelta {
                    text: "done".to_string(),
                },
                HarnessEvent::TurnFinished {
                    text: "done".to_string(),
                },
            ]
        );
    }

    /// Foreign opaque entries are conversation data, not core policy. This
    /// proves core preserves them for a later provider to decide whether it can
    /// consume them instead of silently discarding state it cannot understand.
    #[tokio::test]
    async fn replays_foreign_opaque_items_untouched() {
        struct OpaqueProvider;
        impl ModelProvider for OpaqueProvider {
            fn complete<'a>(
                &'a self,
                _: &'a ProviderRequest,
                transcript: &'a Transcript,
                _: &'a dyn HarnessEventSink,
            ) -> ProviderFuture<'a> {
                Box::pin(async move {
                    if transcript
                        .items()
                        .iter()
                        .any(|item| matches!(item, TranscriptItem::ToolResult(_)))
                    {
                        let opaque = transcript.items().iter().find(|item| {
                            matches!(
                                item,
                                TranscriptItem::ProviderOpaque {
                                    provider: "other-provider",
                                    ..
                                }
                            )
                        });
                        match opaque {
                            Some(TranscriptItem::ProviderOpaque { payload, .. })
                                if payload == &serde_json::json!({"reasoning": "opaque"}) =>
                            {
                                Ok(ProviderTurn::Final {
                                    text: "done".to_string(),
                                    items: vec![TranscriptItem::AssistantMessage {
                                        text: "done".to_string(),
                                    }],
                                })
                            }
                            _ => Err(ProviderError::new(
                                ProviderErrorKind::Protocol,
                                "core changed a foreign opaque transcript item",
                            )),
                        }
                    } else {
                        let call =
                            ToolCall::new("list", "ferricode_list_directory", r#"{"path":"."}"#);
                        Ok(ProviderTurn::ToolCalls {
                            items: vec![
                                TranscriptItem::ProviderOpaque {
                                    provider: "other-provider",
                                    payload: serde_json::json!({"reasoning": "opaque"}),
                                },
                                TranscriptItem::ToolCall(call),
                            ],
                        })
                    }
                })
            }
        }

        let dir = tempdir().unwrap();
        let request = HarnessRequest::new("inspect", dir.path()).unwrap();
        let response = Harness::new()
            .handle(&request, &OpaqueProvider, &NoopEventSink)
            .await
            .unwrap();

        assert!(
            matches!(response.transcript().items()[1], TranscriptItem::ProviderOpaque { provider: "other-provider", ref payload } if payload == &serde_json::json!({"reasoning": "opaque"}))
        );
    }

    /// A final turn needs an assistant message in its replayable transcript,
    /// even when the provider also returns opaque backend state. Otherwise a
    /// later provider request loses the assistant response that completed it.
    #[tokio::test]
    async fn final_turn_without_assistant_message_is_a_protocol_error() {
        struct OpaqueFinalProvider;

        impl ModelProvider for OpaqueFinalProvider {
            fn complete<'a>(
                &'a self,
                _: &'a ProviderRequest,
                _: &'a Transcript,
                _: &'a dyn HarnessEventSink,
            ) -> ProviderFuture<'a> {
                Box::pin(async {
                    Ok(ProviderTurn::Final {
                        text: "done".to_string(),
                        items: vec![TranscriptItem::ProviderOpaque {
                            provider: "test",
                            payload: serde_json::json!({ "type": "reasoning" }),
                        }],
                    })
                })
            }
        }

        let dir = tempdir().unwrap();
        let request = HarnessRequest::new("inspect", dir.path()).unwrap();

        let sink = RecordingSink::default();
        let error = Harness::new()
            .handle(&request, &OpaqueFinalProvider, &sink)
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ProviderErrorKind::Protocol);
        assert_eq!(
            error.to_string(),
            "provider returned a final turn with no assistant message"
        );
        // The model did finish a turn, so its completion event still fires (with
        // no text) even though core then rejects the turn as malformed.
        assert_eq!(
            *sink.events.lock().unwrap(),
            vec![HarnessEvent::TurnFinished {
                text: String::new()
            }]
        );
    }

    /// Exactly 32 tool-call turns must succeed before the final response.
    /// Together with the 33-turn rejection test, this pins both sides of the
    /// rewritten loop counter: an off-by-one in either direction could pass a
    /// single boundary test.
    #[tokio::test]
    async fn tool_loop_limit_allows_exactly_32_turns() {
        let dir = tempdir().unwrap();
        let calls = (0..32).map(|index| {
            vec![ToolCall::new(
                format!("call-{index}"),
                "ferricode_list_directory",
                r#"{"path":"."}"#,
            )]
        });
        let provider = ScriptedProvider::new(calls);
        let request = HarnessRequest::new("loop", dir.path()).unwrap();

        let response = Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        assert_eq!(response.summary(), "done");
        assert_eq!(provider.outputs.lock().unwrap().len(), 32);
    }

    /// The 33rd tool-call turn must fail before execution. Together with the
    /// 32-turn success test, this pins both sides of the rewritten loop counter:
    /// an off-by-one in either direction could pass a single boundary test.
    #[tokio::test]
    async fn tool_loop_limit_fails_clearly() {
        let dir = tempdir().unwrap();
        let calls = (0..33).map(|index| {
            vec![ToolCall::new(
                format!("call-{index}"),
                "ferricode_list_directory",
                r#"{"path":"."}"#,
            )]
        });
        let provider = ScriptedProvider::new(calls);
        let request = HarnessRequest::new("loop", dir.path()).unwrap();

        let error = Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "model exceeded the built-in tool turn limit of 32"
        );
    }

    #[tokio::test]
    async fn directory_listing_is_sorted_and_truncated_with_metadata() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("z.txt"), "z").unwrap();
        fs::write(dir.path().join(".hidden"), "h").unwrap();
        fs::create_dir(dir.path().join("subdir")).unwrap();
        for index in 0..205 {
            fs::write(dir.path().join(format!("entry-{index:03}.txt")), "x").unwrap();
        }
        let provider = ScriptedProvider::new([vec![ToolCall::new(
            "list",
            "ferricode_list_directory",
            r#"{"path":"."}"#,
        )]]);
        let request = HarnessRequest::new("list", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        let outputs = provider.outputs.lock().unwrap();
        let listed = parse_output(&outputs[0][0]);
        assert_eq!(listed["ok"], true);
        assert_eq!(listed["truncated"], true);
        assert_eq!(listed["entries"].as_array().unwrap().len(), 200);
        assert_eq!(listed["entries"][0]["name"], ".hidden");
        assert_eq!(listed["entries"][0]["type"], "file");
        assert_eq!(listed["entries"][0]["size"], 1);
        assert_eq!(listed["entries"][199]["name"], "entry-198.txt");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn directory_listing_does_not_expose_symlink_target_size() {
        let dir = tempdir().unwrap();
        let outside = tempdir().unwrap();
        fs::write(outside.path().join("outside.txt"), "outside").unwrap();
        std::os::unix::fs::symlink(outside.path().join("outside.txt"), dir.path().join("link"))
            .unwrap();
        let provider = ScriptedProvider::new([vec![ToolCall::new(
            "list",
            "ferricode_list_directory",
            r#"{"path":"."}"#,
        )]]);
        let request = HarnessRequest::new("list", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        let outputs = provider.outputs.lock().unwrap();
        let listed = parse_output(&outputs[0][0]);
        assert_eq!(listed["ok"], true);
        assert_eq!(listed["entries"][0]["name"], "link");
        assert_eq!(listed["entries"][0]["type"], "symlink");
        assert!(listed["entries"][0].get("size").is_none());
    }

    #[tokio::test]
    async fn file_read_truncates_at_64_kib() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("large.txt"), "a".repeat(70 * 1024)).unwrap();
        let provider = ScriptedProvider::new([vec![ToolCall::new(
            "read",
            "ferricode_read_file",
            r#"{"path":"large.txt"}"#,
        )]]);
        let request = HarnessRequest::new("read", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        let outputs = provider.outputs.lock().unwrap();
        let read = parse_output(&outputs[0][0]);
        assert_eq!(read["ok"], true);
        assert_eq!(read["truncated"], true);
        assert_eq!(read["content"].as_str().unwrap().len(), 64 * 1024);
    }

    #[tokio::test]
    async fn file_read_truncates_before_partial_utf8_sequence() {
        let dir = tempdir().unwrap();
        let mut content = "a".repeat((64 * 1024) - 1);
        content.push('é');
        content.push_str(&"b".repeat(1024));
        fs::write(dir.path().join("large.txt"), content).unwrap();
        let provider = ScriptedProvider::new([vec![ToolCall::new(
            "read",
            "ferricode_read_file",
            r#"{"path":"large.txt"}"#,
        )]]);
        let request = HarnessRequest::new("read", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        let outputs = provider.outputs.lock().unwrap();
        let read = parse_output(&outputs[0][0]);
        assert_eq!(read["ok"], true);
        assert_eq!(read["truncated"], true);
        assert_eq!(
            read["content"].as_str().unwrap(),
            "a".repeat((64 * 1024) - 1)
        );
    }

    #[tokio::test]
    async fn binary_file_read_returns_tool_error() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("data.bin"), [0, 159, 146, 150]).unwrap();
        let provider = ScriptedProvider::new([vec![ToolCall::new(
            "read",
            "ferricode_read_file",
            r#"{"path":"data.bin"}"#,
        )]]);
        let request = HarnessRequest::new("read", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        let outputs = provider.outputs.lock().unwrap();
        let read = parse_output(&outputs[0][0]);
        assert_eq!(read["ok"], false);
        assert!(read["error"].as_str().unwrap().contains("binary"));
    }

    #[tokio::test]
    async fn file_read_ignores_nul_outside_returned_window() {
        let dir = tempdir().unwrap();
        let mut content = vec![b'a'; 64 * 1024];
        content.push(0);
        fs::write(dir.path().join("large.txt"), content).unwrap();
        let provider = ScriptedProvider::new([vec![ToolCall::new(
            "read",
            "ferricode_read_file",
            r#"{"path":"large.txt"}"#,
        )]]);
        let request = HarnessRequest::new("read", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        let outputs = provider.outputs.lock().unwrap();
        let read = parse_output(&outputs[0][0]);
        assert_eq!(read["ok"], true);
        assert_eq!(read["truncated"], true);
        assert_eq!(read["content"].as_str().unwrap().len(), 64 * 1024);
    }

    #[tokio::test]
    async fn invalid_utf8_file_read_returns_tool_error() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("invalid.txt"), [0xff, b'a']).unwrap();
        let provider = ScriptedProvider::new([vec![ToolCall::new(
            "read",
            "ferricode_read_file",
            r#"{"path":"invalid.txt"}"#,
        )]]);
        let request = HarnessRequest::new("read", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        let outputs = provider.outputs.lock().unwrap();
        let read = parse_output(&outputs[0][0]);
        assert_eq!(read["ok"], false);
        assert!(read["error"].as_str().unwrap().contains("valid UTF-8"));
    }

    #[tokio::test]
    async fn path_policy_rejects_escapes() {
        let dir = tempdir().unwrap();
        let outside = tempdir().unwrap();
        fs::write(outside.path().join("secret.txt"), "secret").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.path().join("secret.txt"), dir.path().join("link"))
            .unwrap();

        let mut calls = vec![
            ToolCall::new(
                "absolute",
                "ferricode_read_file",
                r#"{"path":"/etc/passwd"}"#,
            ),
            ToolCall::new("parent", "ferricode_read_file", r#"{"path":"../secret"}"#),
        ];
        #[cfg(unix)]
        calls.push(ToolCall::new(
            "symlink",
            "ferricode_read_file",
            r#"{"path":"link"}"#,
        ));
        let provider = ScriptedProvider::new([calls]);
        let request = HarnessRequest::new("read", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        let outputs = provider.outputs.lock().unwrap();
        for output in &outputs[0] {
            assert_eq!(parse_output(output)["ok"], false);
        }
        assert!(
            parse_output(&outputs[0][0])["error"]
                .as_str()
                .unwrap()
                .contains("relative")
        );
        assert!(
            parse_output(&outputs[0][1])["error"]
                .as_str()
                .unwrap()
                .contains("traverse")
        );
        #[cfg(unix)]
        assert!(
            parse_output(&outputs[0][2])["error"]
                .as_str()
                .unwrap()
                .contains("outside")
        );
    }

    #[tokio::test]
    async fn directory_listing_path_policy_rejects_escapes() {
        let dir = tempdir().unwrap();
        let outside = tempdir().unwrap();
        fs::write(outside.path().join("secret.txt"), "secret").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();

        let mut calls = vec![
            ToolCall::new("absolute", "ferricode_list_directory", r#"{"path":"/etc"}"#),
            ToolCall::new("parent", "ferricode_list_directory", r#"{"path":".."}"#),
        ];
        #[cfg(unix)]
        calls.push(ToolCall::new(
            "symlink",
            "ferricode_list_directory",
            r#"{"path":"link"}"#,
        ));
        let provider = ScriptedProvider::new([calls]);
        let request = HarnessRequest::new("list", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        let outputs = provider.outputs.lock().unwrap();
        for output in &outputs[0] {
            assert_eq!(parse_output(output)["ok"], false);
        }
        assert!(
            parse_output(&outputs[0][0])["error"]
                .as_str()
                .unwrap()
                .contains("relative")
        );
        assert!(
            parse_output(&outputs[0][1])["error"]
                .as_str()
                .unwrap()
                .contains("traverse")
        );
        #[cfg(unix)]
        assert!(
            parse_output(&outputs[0][2])["error"]
                .as_str()
                .unwrap()
                .contains("outside")
        );
    }

    #[tokio::test]
    async fn tool_call_batch_limit_returns_structured_errors() {
        let dir = tempdir().unwrap();
        let calls = (0..17)
            .map(|index| {
                ToolCall::new(
                    format!("call-{index}"),
                    "ferricode_list_directory",
                    r#"{"path":"."}"#,
                )
            })
            .collect::<Vec<_>>();
        let provider = ScriptedProvider::new([calls]);
        let request = HarnessRequest::new("list", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        let outputs = provider.outputs.lock().unwrap();
        assert_eq!(outputs[0].len(), 17);
        assert!(outputs[0].iter().all(|output| {
            let value = parse_output(output);
            value["ok"] == false
                && value["error"]
                    .as_str()
                    .unwrap()
                    .contains("too many built-in tool calls")
        }));
    }

    /// Tool-size policy violations are recoverable model inputs, not harness failures.
    /// Each oversized field must produce a structured error so the model can adjust
    /// its next call while the provider interaction remains alive.
    #[tokio::test]
    async fn oversized_tool_arguments_return_structured_error() {
        let dir = tempdir().unwrap();
        let provider = ScriptedProvider::new([vec![
            ToolCall::new("large", "ferricode_read_file", "x".repeat((16 * 1024) + 1)),
            ToolCall::new("x".repeat(257), "ferricode_read_file", r#"{"path":"."}"#),
            ToolCall::new("name", "x".repeat(257), r#"{"path":"."}"#),
        ]]);
        let request = HarnessRequest::new("read", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        let outputs = provider.outputs.lock().unwrap();
        assert_eq!(outputs[0].len(), 3);
        for output in &outputs[0] {
            let output = parse_output(output);
            assert_eq!(output["ok"], false);
            assert!(output["error"].as_str().is_some());
        }
        assert!(
            parse_output(&outputs[0][0])["error"]
                .as_str()
                .unwrap()
                .contains("tool arguments exceeded")
        );
        assert!(
            parse_output(&outputs[0][1])["error"]
                .as_str()
                .unwrap()
                .contains("tool call id exceeded")
        );
        assert!(
            parse_output(&outputs[0][2])["error"]
                .as_str()
                .unwrap()
                .contains("tool name exceeded")
        );
    }

    #[tokio::test]
    async fn malformed_tool_calls_return_structured_errors() {
        let dir = tempdir().unwrap();
        let provider = ScriptedProvider::new([vec![
            ToolCall::new("unknown", "ferricode_unknown", r#"{"path":"."}"#),
            ToolCall::new("json", "ferricode_read_file", "{"),
            ToolCall::new("missing", "ferricode_read_file", r#"{}"#),
            ToolCall::new("empty", "ferricode_read_file", r#"{"path":""}"#),
        ]]);
        let request = HarnessRequest::new("read", dir.path()).unwrap();

        Harness::new()
            .handle(&request, &provider, &NoopEventSink)
            .await
            .unwrap();

        let outputs = provider.outputs.lock().unwrap();
        let errors = outputs[0].iter().map(parse_output).collect::<Vec<_>>();
        assert!(errors.iter().all(|output| output["ok"] == false));
        assert!(errors[0]["error"].as_str().unwrap().contains("unknown"));
        assert!(errors[1]["error"].as_str().unwrap().contains("valid JSON"));
        assert!(
            errors[2]["error"]
                .as_str()
                .unwrap()
                .contains("string `path`")
        );
        assert!(
            errors[3]["error"]
                .as_str()
                .unwrap()
                .contains("must not be empty")
        );
    }

    fn parse_output(output: &ToolOutput) -> Value {
        serde_json::from_str(output.output()).unwrap()
    }
}
