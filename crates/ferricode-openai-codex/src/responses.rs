//! Implements Codex Responses API requests and provider state transitions.
//!
//! This module is the boundary between Ferricode's provider contract and Codex's
//! backend protocol. It builds fixed request shapes and decides when stored
//! tokens need refreshing and persists the result, but leaves OAuth callback
//! validation and the token endpoint itself to `auth` and parses backend output
//! through `sse`. The tool schemas it sends are translated from the
//! definitions `ferricode-core` publishes; this module does not decide tool
//! policy.

use crate::{
    OpenAiCodexError, TokenSet,
    auth::{
        CODEX_ORIGINATOR, DEFAULT_ISSUER, now_unix_ms, refresh_access_token, token_needs_refresh,
        tokens_from_response,
    },
    sse::{parse_assistant_turn, parse_sse_assistant_stream},
    store::{default_auth_path, read_auth_file, write_auth_file},
};
use ferricode_core::{
    ModelProvider, ProviderError, ProviderErrorKind, ProviderRequest, ProviderTurn, ToolOutput,
    built_in_tools,
};
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::{Value, json};
use std::path::PathBuf;

const CODEX_BACKEND_RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";
/// Model sent on every Codex responses request. `gpt-5.4` was rejected by the
/// backend on 2026-09-12 with "not supported when using Codex with a ChatGPT
/// account"; `gpt-6-astra` was verified live through a tool turn the same day.
/// SPEC.md names this model too and must stay in sync.
const MODEL: &str = "gpt-6-astra";
const REASONING_EFFORT: &str = "medium";

#[derive(Debug, Clone)]
pub struct OpenAiCodexProvider {
    auth_path: PathBuf,
    issuer: String,
    backend_url: String,
    client: reqwest::Client,
}

impl OpenAiCodexProvider {
    /// Creates a provider that reads credentials from the default auth path.
    pub fn from_default_auth_path() -> Result<Self, OpenAiCodexError> {
        Ok(Self::new(default_auth_path()?))
    }

    /// Creates a provider that reads credentials from an explicit path.
    pub fn new(auth_path: impl Into<PathBuf>) -> Self {
        Self {
            auth_path: auth_path.into(),
            issuer: DEFAULT_ISSUER.to_string(),
            backend_url: CODEX_BACKEND_RESPONSES_URL.to_string(),
            client: reqwest::Client::new(),
        }
    }

    #[cfg(test)]
    fn with_urls(
        auth_path: impl Into<PathBuf>,
        issuer: impl Into<String>,
        backend_url: impl Into<String>,
    ) -> Self {
        Self {
            auth_path: auth_path.into(),
            issuer: issuer.into(),
            backend_url: backend_url.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Runs one provider interaction and expects final assistant text.
    ///
    /// This helper is for tests and simple callers that deliberately bypass the
    /// core harness. Production requests should go through `ferricode-core` so
    /// built-in tool calls can be executed.
    pub async fn respond(&self, request: &ProviderRequest) -> Result<String, ProviderError> {
        match self.start(request).await? {
            ProviderTurn::Final(text) => Ok(text),
            ProviderTurn::ToolCalls { .. } => Err(ProviderError::new(
                ProviderErrorKind::Protocol,
                "model requested built-in tools outside the core harness",
            )),
        }
    }

    async fn authenticated_tokens(&self) -> Result<TokenSet, OpenAiCodexError> {
        let mut tokens = read_auth_file(&self.auth_path)?
            .openai_codex
            .and_then(|auth| auth.tokens)
            .ok_or(OpenAiCodexError::MissingTokens)?;

        if token_needs_refresh(&tokens, now_unix_ms()?) {
            let refresh_token = tokens.refresh_token.clone();
            let refreshed =
                refresh_access_token(&self.client, &self.issuer, &refresh_token).await?;
            let refreshed_tokens = tokens_from_response(refreshed, Some(&refresh_token))?;
            let mut latest_auth = read_auth_file(&self.auth_path)?;

            if let Some(latest_tokens) = latest_auth.openai_codex.as_ref().and_then(|auth| {
                auth.tokens
                    .as_ref()
                    .filter(|tokens| tokens.refresh_token != refresh_token)
                    .cloned()
            }) {
                tokens = latest_tokens;
            } else {
                tokens = refreshed_tokens;
                latest_auth
                    .openai_codex
                    .get_or_insert_with(Default::default)
                    .tokens = Some(tokens.clone());
                write_auth_file(&self.auth_path, &latest_auth)?;
            }
        }

        Ok(tokens)
    }
}

impl ModelProvider for OpenAiCodexProvider {
    type State = OpenAiCodexState;

    async fn start<'a>(
        &'a self,
        request: &'a ProviderRequest,
    ) -> Result<ProviderTurn<Self::State>, ProviderError> {
        let tokens = self.authenticated_tokens().await?;
        let body = build_responses_body(request);
        let turn = self
            .send_responses_request(&tokens, &body)
            .await
            .map_err(ProviderError::from)?;
        Ok(attach_request_context(turn, &body))
    }

    async fn resume<'a>(
        &'a self,
        state: Self::State,
        tool_outputs: &'a [ToolOutput],
    ) -> Result<ProviderTurn<Self::State>, ProviderError> {
        let tokens = self.authenticated_tokens().await?;
        let body = build_tool_outputs_body(state, tool_outputs);
        let turn = self
            .send_responses_request(&tokens, &body)
            .await
            .map_err(ProviderError::from)?;
        Ok(attach_request_context(turn, &body))
    }
}

impl OpenAiCodexProvider {
    async fn send_responses_request(
        &self,
        tokens: &TokenSet,
        body: &Value,
    ) -> Result<ProviderTurn<OpenAiCodexState>, OpenAiCodexError> {
        let response = self
            .client
            .post(&self.backend_url)
            .headers(build_codex_headers(tokens)?)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await?;
            return Err(OpenAiCodexError::BackendStatus { status, body: text });
        }

        read_assistant_response(response).await
    }
}

/// OpenAI response output items needed to resume after tool execution.
///
/// `instructions` rides along because `resume` receives no request and each
/// continuation is a fresh stateless request (`"store": false`) that must
/// restate the system prompt itself; the harness owns that prompt
/// (`ProviderRequest::instructions`) and this provider only relays it.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiCodexState {
    pub(crate) input_items: Vec<Value>,
    pub(crate) output_items: Vec<Value>,
    pub(crate) instructions: String,
}

/// Builds the hardcoded bootstrap Responses body.
pub fn build_responses_body(request: &ProviderRequest) -> Value {
    json!({
        "model": MODEL,
        "instructions": request.instructions(),
        "stream": true,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": format!("Working directory: {}\n\n{}", request.working_directory().display(), request.prompt())
                    }
                ]
            }
        ],
        "tools": built_in_tool_schemas(),
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "reasoning": {
            "effort": REASONING_EFFORT
        },
        "store": false
    })
}

fn build_tool_outputs_body(state: OpenAiCodexState, tool_outputs: &[ToolOutput]) -> Value {
    let mut input = state.input_items;
    input.extend(state.output_items.into_iter().map(strip_provider_item_ids));
    input.extend(tool_outputs.iter().map(|output| {
        json!({
            "type": "function_call_output",
            "call_id": output.call_id(),
            "output": output.output(),
        })
    }));

    json!({
        "model": MODEL,
        "instructions": state.instructions,
        "stream": true,
        "input": input,
        "tools": built_in_tool_schemas(),
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "reasoning": {
            "effort": REASONING_EFFORT
        },
        "store": false
    })
}

fn response_input_items(body: &Value) -> Vec<Value> {
    body.get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// Copies what the next request must repeat (the input items sent so far and
/// the system prompt) from the body just sent into the resume state.
///
/// The parser that produced `turn` only sees the response, so it cannot know
/// what was asked; reading both values back from the request body keeps a
/// single source of truth for what the backend has been told.
fn attach_request_context(
    turn: ProviderTurn<OpenAiCodexState>,
    body: &Value,
) -> ProviderTurn<OpenAiCodexState> {
    match turn {
        ProviderTurn::ToolCalls { mut state, calls } => {
            state.input_items = response_input_items(body);
            // Every body is built by this module, so a missing key is a bug;
            // falling back to "" would send a request with no system prompt
            // and nothing would notice, which is the failure this field exists
            // to rule out.
            state.instructions = body["instructions"]
                .as_str()
                .expect("request bodies built by this module always carry instructions")
                .to_string();
            ProviderTurn::ToolCalls { state, calls }
        }
        ProviderTurn::Final(text) => ProviderTurn::Final(text),
    }
}

fn strip_provider_item_ids(mut item: Value) -> Value {
    if let Some(map) = item.as_object_mut() {
        map.remove("id");
    }
    item
}

fn built_in_tool_schemas() -> Value {
    Value::Array(
        built_in_tools()
            .iter()
            .map(|definition| {
                json!({
                    "type": "function",
                    "name": definition.name(),
                    "description": definition.description(),
                    "parameters": definition.parameters_schema(),
                    "strict": true
                })
            })
            .collect(),
    )
}

async fn read_assistant_response(
    mut response: reqwest::Response,
) -> Result<ProviderTurn<OpenAiCodexState>, OpenAiCodexError> {
    if !response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"))
    {
        let text = response.text().await?;
        return parse_assistant_turn(&text);
    }

    parse_sse_assistant_stream(&mut response).await
}

fn build_codex_headers(tokens: &TokenSet) -> Result<HeaderMap, OpenAiCodexError> {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        "OpenAI-Beta",
        HeaderValue::from_static("responses=experimental"),
    );
    headers.insert("originator", HeaderValue::from_static(CODEX_ORIGINATOR));
    headers.insert(
        "chatgpt-account-id",
        HeaderValue::from_str(&tokens.chatgpt_account_id)
            .map_err(|_| OpenAiCodexError::MissingAccountId)?,
    );
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", tokens.access_token))
            .map_err(|_| OpenAiCodexError::MissingTokens)?,
    );
    Ok(headers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;
    use ferricode_core::ProviderRequest;
    use serde_json::json;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;
    use tokio::time::timeout;

    #[test]
    fn response_body_uses_hardcoded_model_and_effort() {
        let request = ProviderRequest::new("summarize this repository", "/repo");

        let body = build_responses_body(&request);

        assert_eq!(body["model"], MODEL);
        assert_eq!(body["instructions"], ferricode_core::DEFAULT_INSTRUCTIONS);
        assert!(body["instructions"].as_str().unwrap().contains("inspect"));
        assert!(
            body["instructions"]
                .as_str()
                .unwrap()
                .contains("filesystem tools")
        );
        assert_eq!(body["stream"], true);
        assert_eq!(body["tools"].as_array().unwrap().len(), 2);
        assert_eq!(body["tools"][0]["name"], "ferricode_list_directory");
        assert_eq!(body["tools"][0]["strict"], true);
        assert_eq!(
            body["tools"][0]["parameters"]["additionalProperties"],
            false
        );
        assert_eq!(body["tools"][1]["name"], "ferricode_read_file");
        assert_eq!(body["tools"][1]["strict"], true);
        assert_eq!(body["tools"][1]["parameters"]["required"], json!(["path"]));
        assert_eq!(
            body["tools"][1]["parameters"]["additionalProperties"],
            false
        );
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["reasoning"]["effort"], REASONING_EFFORT);
        assert_eq!(body["store"], false);
        assert_eq!(
            body["input"][0]["content"][0]["text"],
            "Working directory: /repo\n\nsummarize this repository"
        );
    }

    #[test]
    fn tool_outputs_body_preserves_prior_items() {
        let state = OpenAiCodexState {
            instructions: ferricode_core::DEFAULT_INSTRUCTIONS.to_string(),
            input_items: vec![json!({
                "role": "user",
                "content": [{"type": "input_text", "text": "Working directory: /repo\n\nread it"}]
            })],
            output_items: vec![json!({
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_1",
                "name": "ferricode_read_file",
                "arguments": "{\"path\":\"README.md\"}"
            })],
        };
        let outputs = vec![ToolOutput::new(
            "call_1",
            r#"{"ok":true,"path":"README.md","content":"hi","truncated":false}"#,
        )];

        let body = build_tool_outputs_body(state, &outputs);

        assert_eq!(body["input"].as_array().unwrap().len(), 3);
        assert_eq!(body["input"][0]["role"], "user");
        assert_eq!(body["input"][1]["type"], "function_call");
        assert!(body["input"][1].get("id").is_none());
        assert_eq!(body["input"][2]["type"], "function_call_output");
        assert_eq!(body["input"][2]["call_id"], "call_1");
        assert_eq!(body["input"][2]["output"], outputs[0].output());
        assert_eq!(body["tools"].as_array().unwrap().len(), 2);
        assert_eq!(body["instructions"], ferricode_core::DEFAULT_INSTRUCTIONS);
    }

    #[tokio::test]
    async fn provider_resume_posts_tool_outputs_and_returns_next_turn() {
        let (base_url, requests) = spawn_test_server(vec![TestResponse::json(
            r#"{"output":[{"content":[{"type":"output_text","text":"done"}]}]}"#,
        )])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        write_auth_file(
            &path,
            &auth_with_tokens("access", "refresh", 9_999_999_999_999),
        )
        .unwrap();
        let provider =
            OpenAiCodexProvider::with_urls(&path, &base_url, format!("{base_url}/codex/responses"));
        let state = OpenAiCodexState {
            instructions: ferricode_core::DEFAULT_INSTRUCTIONS.to_string(),
            input_items: vec![json!({
                "role": "user",
                "content": [{"type": "input_text", "text": "Working directory: /repo\n\nread it"}]
            })],
            output_items: vec![json!({
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_1",
                "name": "ferricode_read_file",
                "arguments": "{\"path\":\"README.md\"}"
            })],
        };
        let outputs = vec![ToolOutput::new(
            "call_1",
            r#"{"ok":true,"path":"README.md","content":"hi","truncated":false}"#,
        )];

        let turn = provider.resume(state, &outputs).await.unwrap();

        assert_eq!(turn, ProviderTurn::Final("done".to_string()));
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].contains("Working directory: /repo"));
        assert!(requests[0].contains(r#""type":"function_call_output""#));
        assert!(requests[0].contains(r#""call_id":"call_1""#));
        assert!(requests[0].contains(r#""output":"{\"ok\":true,"#));
        assert!(!requests[0].contains(r#""id":"fc_123""#));
    }

    #[tokio::test]
    async fn provider_start_state_preserves_input_for_resume() {
        let (base_url, requests) = spawn_test_server(vec![
            TestResponse::json(
                r#"{"output":[{"type":"function_call","id":"fc_123","call_id":"call_1","name":"ferricode_read_file","arguments":"{\"path\":\"README.md\"}"}]}"#,
            ),
            TestResponse::json(r#"{"output":[{"content":[{"type":"output_text","text":"done"}]}]}"#),
        ])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        write_auth_file(
            &path,
            &auth_with_tokens("access", "refresh", 9_999_999_999_999),
        )
        .unwrap();
        let provider =
            OpenAiCodexProvider::with_urls(&path, &base_url, format!("{base_url}/codex/responses"));
        let request = ProviderRequest::new("read it", "/repo");

        let ProviderTurn::ToolCalls { state, calls } = provider.start(&request).await.unwrap()
        else {
            panic!("expected tool call turn");
        };
        assert_eq!(calls.len(), 1);
        let outputs = vec![ToolOutput::new(
            "call_1",
            r#"{"ok":true,"path":"README.md","content":"hi","truncated":false}"#,
        )];
        let turn = provider.resume(state, &outputs).await.unwrap();

        assert_eq!(turn, ProviderTurn::Final("done".to_string()));
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[1].contains("Working directory: /repo"));
        assert!(requests[1].contains("read it"));
        assert!(requests[1].contains(r#""type":"function_call_output""#));
        // The resume request must repeat the system prompt the first turn sent,
        // not a placeholder; only `attach_request_context` carries it across.
        assert!(requests[1].contains(ferricode_core::DEFAULT_INSTRUCTIONS));
    }

    #[tokio::test]
    async fn provider_reports_missing_tokens_before_network() {
        let dir = tempdir().unwrap();
        let provider = OpenAiCodexProvider::new(dir.path().join("auth.toml"));
        let request = ProviderRequest::new("hello", ".");

        let error = provider.respond(&request).await.unwrap_err();

        assert_eq!(
            error.to_string(),
            "OpenAI Codex auth is missing; run `ferric auth openai-codex` first"
        );
    }

    #[tokio::test]
    async fn provider_success_sends_expected_request() {
        let (base_url, requests) = spawn_test_server(vec![TestResponse::json(
            r#"{"output":[{"content":[{"type":"output_text","text":"assistant"}]}]}"#,
        )])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        write_auth_file(
            &path,
            &auth_with_tokens("access", "refresh", 9_999_999_999_999),
        )
        .unwrap();
        let provider =
            OpenAiCodexProvider::with_urls(&path, &base_url, format!("{base_url}/codex/responses"));
        let request = ProviderRequest::new("hello", "/repo");

        let text = provider.respond(&request).await.unwrap();

        assert_eq!(text, "assistant");
        let requests = requests.lock().unwrap();
        assert!(requests[0].starts_with("POST /codex/responses "));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer access")
        );
        assert!(requests[0].contains(r#""model":"gpt-6-astra""#));
        assert!(requests[0].contains(r#""effort":"medium""#));
    }

    #[tokio::test]
    async fn provider_returns_when_sse_completion_arrives_before_eof() {
        use std::sync::{Arc, Mutex};

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_request(&mut stream).await;
            captured
                .lock()
                .unwrap()
                .push(String::from_utf8_lossy(&request).to_string());
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n\
                      data: {\"type\":\"response.output_text.delta\",\"delta\":\"assistant\"}\n\n\
                      data: {\"type\":\"response.completed\"}\n\n",
                )
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_secs(5)).await;
        });

        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        write_auth_file(
            &path,
            &auth_with_tokens("access", "refresh", 9_999_999_999_999),
        )
        .unwrap();
        let base_url = format!("http://{addr}");
        let provider =
            OpenAiCodexProvider::with_urls(&path, &base_url, format!("{base_url}/codex/responses"));
        let request = ProviderRequest::new("hello", "/repo");

        let text = timeout(Duration::from_millis(500), provider.respond(&request))
            .await
            .expect("provider should return on the SSE completion event")
            .unwrap();

        assert_eq!(text, "assistant");
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn provider_backend_failure_uses_backend_error() {
        let (base_url, _requests) = spawn_test_server(vec![TestResponse::new(
            "503 Service Unavailable",
            "unavailable",
        )])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        write_auth_file(
            &path,
            &auth_with_tokens("access", "refresh", 9_999_999_999_999),
        )
        .unwrap();
        let provider =
            OpenAiCodexProvider::with_urls(&path, &base_url, format!("{base_url}/codex/responses"));
        let request = ProviderRequest::new("hello", "/repo");

        let error = provider.respond(&request).await.unwrap_err();

        assert!(error.to_string().contains("Codex backend failed"));
        assert!(!error.to_string().contains("token exchange failed"));
    }

    #[tokio::test]
    async fn refresh_does_not_overwrite_newer_stored_refresh_token() {
        use std::sync::{Arc, Mutex};

        let refreshed_id = id_token("acct_refreshed", "pro");
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        write_auth_file(&path, &auth_with_tokens("expired-access", "refresh", 1)).unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let auth_path = path.clone();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_request(&mut stream).await;
            captured
                .lock()
                .unwrap()
                .push(String::from_utf8_lossy(&request).to_string());
            write_auth_file(
                &auth_path,
                &auth_with_tokens("already-access", "already-refresh", 9_999_999_999_999),
            )
            .unwrap();
            write_response(
                &mut stream,
                TestResponse::json(&token_json(
                    "new-access",
                    "new-refresh",
                    &refreshed_id,
                    3600,
                )),
            )
            .await;

            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_request(&mut stream).await;
            captured
                .lock()
                .unwrap()
                .push(String::from_utf8_lossy(&request).to_string());
            write_response(&mut stream, TestResponse::json(r#"{"output_text":"done"}"#)).await;
        });
        let base_url = format!("http://{addr}");
        let provider =
            OpenAiCodexProvider::with_urls(&path, &base_url, format!("{base_url}/codex/responses"));
        let request = ProviderRequest::new("hello", "/repo");

        let text = provider.respond(&request).await.unwrap();

        assert_eq!(text, "done");
        let tokens = read_auth_file(&path)
            .unwrap()
            .openai_codex
            .unwrap()
            .tokens
            .unwrap();
        assert_eq!(tokens.access_token, "already-access");
        assert_eq!(tokens.refresh_token, "already-refresh");
        let requests = requests.lock().unwrap();
        assert!(
            requests[1]
                .to_ascii_lowercase()
                .contains("authorization: bearer already-access")
        );
    }

    #[tokio::test]
    async fn refresh_response_can_reuse_existing_refresh_token() {
        let refreshed_id = id_token("acct_refreshed", "pro");
        let (base_url, requests) = spawn_test_server(vec![
            TestResponse::json(&token_json_without_refresh(
                "new-access",
                &refreshed_id,
                3600,
            )),
            TestResponse::json(r#"{"output_text":"done"}"#),
        ])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        write_auth_file(&path, &auth_with_tokens("expired-access", "refresh", 1)).unwrap();
        let provider =
            OpenAiCodexProvider::with_urls(&path, &base_url, format!("{base_url}/codex/responses"));
        let request = ProviderRequest::new("hello", "/repo");

        let text = provider.respond(&request).await.unwrap();

        assert_eq!(text, "done");
        let auth = read_auth_file(&path).unwrap();
        let tokens = auth.openai_codex.unwrap().tokens.unwrap();
        assert_eq!(tokens.access_token, "new-access");
        assert_eq!(tokens.refresh_token, "refresh");
        assert_eq!(requests.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn provider_refresh_request_uses_public_client_fields() {
        let refreshed_id = id_token("acct_refreshed", "pro");
        let (base_url, requests) = spawn_test_server(vec![
            TestResponse::json(&token_json(
                "new-access",
                "new-refresh",
                &refreshed_id,
                3600,
            )),
            TestResponse::json(r#"{"output_text":"done"}"#),
        ])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        write_auth_file(&path, &auth_with_tokens("expired-access", "refresh", 1)).unwrap();
        let provider =
            OpenAiCodexProvider::with_urls(&path, &base_url, format!("{base_url}/codex/responses"));
        let request = ProviderRequest::new("hello", "/repo");

        let text = provider.respond(&request).await.unwrap();

        assert_eq!(text, "done");
        let auth = read_auth_file(&path).unwrap();
        let tokens = auth.openai_codex.unwrap().tokens.unwrap();
        assert_eq!(tokens.access_token, "new-access");
        assert_eq!(tokens.refresh_token, "new-refresh");
        assert_eq!(tokens.chatgpt_account_id, "acct_refreshed");
        let requests = requests.lock().unwrap();
        assert!(requests[0].starts_with("POST /oauth/token "));
        assert!(requests[0].contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
        assert!(requests[0].contains("refresh_token=refresh"));
        assert!(!requests[0].contains("client_secret"));
        assert!(
            requests[1]
                .to_ascii_lowercase()
                .contains("authorization: bearer new-access")
        );
    }
}
