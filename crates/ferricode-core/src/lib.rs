//! Harness primitives that do not know how the user interface is rendered.
//!
//! The core crate holds the contracts and policy shared by the CLI, TUI, and
//! future automation surfaces. Provider crates adapt those contracts to a
//! concrete model backend, but tool orchestration and local filesystem policy
//! stay here so every front end gets the same behavior.

mod tools;

use tools::execute_tool_calls;
pub use tools::{ToolCall, ToolOutput};

const MAX_TOOL_TURNS: usize = 32;

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
    /// front end gets the same contract. The working directory remains a string
    /// at this boundary because callers may only be passing context; local tool
    /// execution resolves and validates it later, when filesystem access is
    /// actually required.
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
/// and context should be sent to a model.
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

/// One model-request turn produced by a provider.
///
/// Providers either return final assistant text or ask the harness to execute
/// local tools and resume the same model interaction. The state value is opaque
/// to the core crate; it lets provider crates preserve backend-specific
/// transcript items that must be sent back with tool outputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderTurn<State> {
    /// The provider has completed the request with user-facing assistant text.
    Final(String),
    /// The provider needs core-owned tools before it can continue.
    ToolCalls {
        /// Opaque provider state to pass back on the continuation request.
        state: State,
        /// Tool calls requested by the model.
        calls: Vec<ToolCall>,
    },
}

/// A provider that can drive one model interaction through core-owned tools.
///
/// The core trait intentionally remains narrow. It does not expose streaming,
/// model selection, provider fallback, or MCP. It only gives the harness enough
/// structure to run local built-in tools and hand their outputs back to the
/// same provider.
pub trait ModelProvider {
    /// Opaque provider transcript state preserved across tool turns.
    type State: Send;

    /// Starts a model interaction from the harness-selected request.
    fn start<'a>(
        &'a self,
        request: &'a ProviderRequest,
    ) -> impl std::future::Future<Output = Result<ProviderTurn<Self::State>, ProviderError>> + Send + 'a;

    /// Continues a model interaction after the harness has run requested tools.
    fn resume<'a>(
        &'a self,
        state: Self::State,
        tool_outputs: &'a [ToolOutput],
    ) -> impl std::future::Future<Output = Result<ProviderTurn<Self::State>, ProviderError>> + Send + 'a;
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
    /// The harness stays responsible for orchestration, including built-in tool
    /// execution. Provider crates own only the backend-specific request and
    /// response format needed to ask a model what to do next.
    pub async fn handle(
        &self,
        request: &HarnessRequest,
        provider: &impl ModelProvider,
    ) -> Result<HarnessResponse, ProviderError> {
        let provider_request = ProviderRequest::new(request.prompt(), request.working_directory());
        let mut turn = provider.start(&provider_request).await?;

        for _ in 0..MAX_TOOL_TURNS {
            match turn {
                ProviderTurn::Final(summary) => return Ok(HarnessResponse::new(summary)),
                ProviderTurn::ToolCalls { state, calls } => {
                    let outputs = execute_tool_calls(&provider_request, calls);
                    turn = provider.resume(state, &outputs).await?;
                }
            }
        }

        match turn {
            ProviderTurn::Final(summary) => Ok(HarnessResponse::new(summary)),
            ProviderTurn::ToolCalls { .. } => Err(ProviderError::new(format!(
                "model exceeded the built-in tool turn limit of {MAX_TOOL_TURNS}"
            ))),
        }
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
        ProviderTurn, ToolCall, ToolOutput,
    };
    use serde_json::Value;
    use std::fs;
    use std::sync::Mutex;
    use tempfile::tempdir;

    struct EchoProvider;

    impl ModelProvider for EchoProvider {
        type State = ();

        async fn start<'a>(
            &'a self,
            request: &'a ProviderRequest,
        ) -> Result<ProviderTurn<Self::State>, ProviderError> {
            Ok(ProviderTurn::Final(format!(
                "provider saw {} from {}",
                request.prompt(),
                request.working_directory()
            )))
        }

        async fn resume<'a>(
            &'a self,
            _state: Self::State,
            _tool_outputs: &'a [ToolOutput],
        ) -> Result<ProviderTurn<Self::State>, ProviderError> {
            unreachable!("echo provider never requests tools")
        }
    }

    struct FailingProvider;

    impl ModelProvider for FailingProvider {
        type State = ();

        async fn start<'a>(
            &'a self,
            _request: &'a ProviderRequest,
        ) -> Result<ProviderTurn<Self::State>, ProviderError> {
            Err(ProviderError::new("provider failed"))
        }

        async fn resume<'a>(
            &'a self,
            _state: Self::State,
            _tool_outputs: &'a [ToolOutput],
        ) -> Result<ProviderTurn<Self::State>, ProviderError> {
            unreachable!("failing provider never requests tools")
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

    impl ModelProvider for ScriptedProvider {
        type State = usize;

        async fn start<'a>(
            &'a self,
            _request: &'a ProviderRequest,
        ) -> Result<ProviderTurn<Self::State>, ProviderError> {
            Ok(ProviderTurn::ToolCalls {
                state: 0,
                calls: self.calls[0].clone(),
            })
        }

        async fn resume<'a>(
            &'a self,
            state: Self::State,
            tool_outputs: &'a [ToolOutput],
        ) -> Result<ProviderTurn<Self::State>, ProviderError> {
            self.outputs.lock().unwrap().push(tool_outputs.to_vec());
            let next_state = state + 1;
            if let Some(calls) = self.calls.get(next_state) {
                Ok(ProviderTurn::ToolCalls {
                    state: next_state,
                    calls: calls.clone(),
                })
            } else {
                Ok(ProviderTurn::Final("done".to_string()))
            }
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
    async fn repository_prompts_are_not_special_cased() {
        let harness = Harness::new();
        let request = HarnessRequest::new("summarize this repository", "/work").unwrap();

        let response = harness.handle(&request, &EchoProvider).await.unwrap();

        assert_eq!(
            response.summary(),
            "provider saw summarize this repository from /work"
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
        let request =
            HarnessRequest::new("read project files", dir.path().to_string_lossy()).unwrap();

        let response = Harness::new().handle(&request, &provider).await.unwrap();

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
    }

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
        let request = HarnessRequest::new("loop", dir.path().to_string_lossy()).unwrap();

        let error = Harness::new()
            .handle(&request, &provider)
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
        let request = HarnessRequest::new("list", dir.path().to_string_lossy()).unwrap();

        Harness::new().handle(&request, &provider).await.unwrap();

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
        let request = HarnessRequest::new("list", dir.path().to_string_lossy()).unwrap();

        Harness::new().handle(&request, &provider).await.unwrap();

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
        let request = HarnessRequest::new("read", dir.path().to_string_lossy()).unwrap();

        Harness::new().handle(&request, &provider).await.unwrap();

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
        let request = HarnessRequest::new("read", dir.path().to_string_lossy()).unwrap();

        Harness::new().handle(&request, &provider).await.unwrap();

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
        let request = HarnessRequest::new("read", dir.path().to_string_lossy()).unwrap();

        Harness::new().handle(&request, &provider).await.unwrap();

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
        let request = HarnessRequest::new("read", dir.path().to_string_lossy()).unwrap();

        Harness::new().handle(&request, &provider).await.unwrap();

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
        let request = HarnessRequest::new("read", dir.path().to_string_lossy()).unwrap();

        Harness::new().handle(&request, &provider).await.unwrap();

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
        let request = HarnessRequest::new("read", dir.path().to_string_lossy()).unwrap();

        Harness::new().handle(&request, &provider).await.unwrap();

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
        let request = HarnessRequest::new("list", dir.path().to_string_lossy()).unwrap();

        Harness::new().handle(&request, &provider).await.unwrap();

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
        let request = HarnessRequest::new("list", dir.path().to_string_lossy()).unwrap();

        Harness::new().handle(&request, &provider).await.unwrap();

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

    #[tokio::test]
    async fn oversized_tool_arguments_return_structured_error() {
        let dir = tempdir().unwrap();
        let provider = ScriptedProvider::new([vec![ToolCall::new(
            "large",
            "ferricode_read_file",
            "x".repeat((16 * 1024) + 1),
        )]]);
        let request = HarnessRequest::new("read", dir.path().to_string_lossy()).unwrap();

        Harness::new().handle(&request, &provider).await.unwrap();

        let outputs = provider.outputs.lock().unwrap();
        let output = parse_output(&outputs[0][0]);
        assert_eq!(output["ok"], false);
        assert!(
            output["error"]
                .as_str()
                .unwrap()
                .contains("tool arguments exceeded")
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
        let request = HarnessRequest::new("read", dir.path().to_string_lossy()).unwrap();

        Harness::new().handle(&request, &provider).await.unwrap();

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
