//! Implements Codex Responses API requests from core-owned transcripts.
//!
//! This module is the boundary between Ferricode's provider contract and Codex's
//! backend protocol. It builds fixed request shapes and decides when stored
//! tokens need refreshing and persists the result, but leaves OAuth callback
//! validation and the token endpoint itself to `auth` and parses backend output
//! through `sse`. The tool schemas it sends are translated from the
//! definitions `ferricode-core` publishes; this module does not decide tool
//! policy.

use crate::{
    OpenAiCodexError, PROVIDER_NAME, TokenSet,
    auth::{
        CODEX_ORIGINATOR, DEFAULT_ISSUER, now_unix_ms, refresh_access_token, token_needs_refresh,
        tokens_from_response,
    },
    sse::{parse_assistant_turn_with_sink, parse_sse_assistant_stream},
    store::{default_auth_path, read_auth_file, write_auth_file},
};
use ferricode_core::{
    HarnessEventSink, ModelProvider, NoopEventSink, ProviderError, ProviderErrorKind,
    ProviderFuture, ProviderRequest, ProviderTurn, Transcript, TranscriptItem, built_in_tools,
};
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::{Value, json};
use std::path::PathBuf;

const CODEX_BACKEND_RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";
/// Model sent on every Codex responses request.
///
/// The backend only accepts some model names for ChatGPT-account logins:
/// `gpt-5.4` was rejected on 2026-09-12 with "not supported when using Codex
/// with a ChatGPT account". `gpt-5.6-luna` at high effort was verified live
/// through two tool turns on 2026-09-13. SPEC.md names this model too and must
/// stay in sync.
const MODEL: &str = "gpt-5.6-luna";
/// Reasoning effort sent alongside `MODEL`. SPEC.md names this value too and
/// must stay in sync.
const REASONING_EFFORT: &str = "high";

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
        let transcript = Transcript::for_request(request);
        match self.complete(request, &transcript, &NoopEventSink).await? {
            ProviderTurn::Final { text, .. } => Ok(text),
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
    fn complete<'a>(
        &'a self,
        request: &'a ProviderRequest,
        transcript: &'a Transcript,
        sink: &'a dyn HarnessEventSink,
    ) -> ProviderFuture<'a> {
        Box::pin(async move {
            let tokens = self.authenticated_tokens().await?;
            let body = build_responses_body(request, transcript);
            log_request_item_counts(transcript);
            let turn = self
                .send_responses_request(&tokens, &body, sink)
                .await
                .map_err(ProviderError::from)?;
            tracing::debug!(output_item_types = ?turn_item_types(&turn), "received OpenAI Codex response items");
            Ok(turn)
        })
    }
}

impl OpenAiCodexProvider {
    async fn send_responses_request(
        &self,
        tokens: &TokenSet,
        body: &Value,
        sink: &dyn HarnessEventSink,
    ) -> Result<ProviderTurn, OpenAiCodexError> {
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

        read_assistant_response(response, sink).await
    }
}

/// Builds the fixed Responses request by translating the complete transcript.
///
/// Every request is stateless (`"store": false`), including continuation
/// requests. Opaque reasoning is therefore replayed here for this provider
/// only, with its top-level response id removed because it is not valid input.
pub fn build_responses_body(request: &ProviderRequest, transcript: &Transcript) -> Value {
    json!({
        "model": MODEL,
        "instructions": request.instructions(),
        "stream": true,
        "input": transcript.items().iter().filter_map(render_transcript_item).collect::<Vec<_>>(),
        "tools": built_in_tool_schemas(),
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "reasoning": {
            "effort": REASONING_EFFORT
        },
        "store": false
    })
}

/// Maps each renderable transcript item one-to-one into request input order.
///
/// The only elision is an opaque item tagged for another provider, because
/// core preserves all opaque state while each provider alone decides what it
/// can replay.
fn render_transcript_item(item: &TranscriptItem) -> Option<Value> {
    match item {
        TranscriptItem::UserMessage { text } => Some(json!({
            "role": "user", "content": [{ "type": "input_text", "text": text }],
        })),
        TranscriptItem::AssistantMessage { text } => Some(json!({
            "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": text }],
        })),
        TranscriptItem::ToolCall(call) => Some(json!({
            "type": "function_call", "call_id": call.id(), "name": call.name(), "arguments": call.arguments(),
        })),
        TranscriptItem::ToolResult(output) => Some(json!({
            "type": "function_call_output", "call_id": output.call_id(), "output": output.output(),
        })),
        TranscriptItem::ProviderOpaque { provider, payload } if *provider == PROVIDER_NAME => {
            Some(strip_provider_item_ids(payload.clone()))
        }
        TranscriptItem::ProviderOpaque { .. } => None,
    }
}

/// Emits a compact breakdown of the core vocabulary before it reaches the wire.
///
/// Keeping this separate from rendering makes diagnostics describe the full
/// transcript, including foreign opaque entries that this provider will skip.
fn log_request_item_counts(transcript: &Transcript) {
    let mut counts = [0; 5];
    for item in transcript.items() {
        counts[match item {
            TranscriptItem::UserMessage { .. } => 0,
            TranscriptItem::AssistantMessage { .. } => 1,
            TranscriptItem::ToolCall(_) => 2,
            TranscriptItem::ToolResult(_) => 3,
            TranscriptItem::ProviderOpaque { .. } => 4,
        }] += 1;
    }
    tracing::debug!(
        user_messages = counts[0],
        assistant_messages = counts[1],
        tool_calls = counts[2],
        tool_results = counts[3],
        provider_opaque = counts[4],
        "sending OpenAI Codex transcript items"
    );
}

/// Names transcript item kinds for response diagnostics without logging payloads.
///
/// Opaque items report their provider wire `type` when available so unknown
/// backend state can be identified without exposing its contents.
fn turn_item_types(turn: &ProviderTurn) -> Vec<String> {
    let items = match turn {
        ProviderTurn::Final { items, .. } | ProviderTurn::ToolCalls { items } => items,
    };
    items
        .iter()
        .map(|item| match item {
            TranscriptItem::UserMessage { .. } => "user_message".to_string(),
            TranscriptItem::AssistantMessage { .. } => "message".to_string(),
            TranscriptItem::ToolCall(_) => "function_call".to_string(),
            TranscriptItem::ToolResult(_) => "function_call_output".to_string(),
            TranscriptItem::ProviderOpaque { payload, .. } => payload
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("<missing-type>")
                .to_string(),
        })
        .collect()
}

/// Removes a top-level output item `id` before replaying it as Responses input.
///
/// The API rejects that item id as input. This behavior comes from the earlier
/// state-based implementation. On 2026-09-13 `gpt-5.6-luna` at high effort
/// returned a `reasoning` item alongside a function call, and the backend
/// accepted it replayed this way on both continuation requests. Nothing checks
/// whether the model actually uses the replayed reasoning; the request carries
/// no `include` for encrypted reasoning content.
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
    sink: &dyn HarnessEventSink,
) -> Result<ProviderTurn, OpenAiCodexError> {
    if !response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"))
    {
        let text = response.text().await?;
        return parse_assistant_turn_with_sink(&text, sink);
    }

    parse_sse_assistant_stream(&mut response, sink).await
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
    use ferricode_core::{ProviderRequest, ToolCall, ToolOutput, Transcript, TranscriptItem};
    use serde_json::json;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;
    use tokio::time::timeout;

    #[test]
    fn response_body_uses_hardcoded_model_and_effort() {
        let request = ProviderRequest::new("summarize this repository", "/repo");
        let mut transcript = Transcript::new();
        transcript.push(TranscriptItem::UserMessage {
            text: "Working directory: /repo\n\nsummarize this repository".to_string(),
        });

        let body = build_responses_body(&request, &transcript);

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
    fn transcript_body_preserves_prior_items() {
        let request = ProviderRequest::new("read it", "/repo");
        let output = ToolOutput::new(
            "call_1",
            r#"{"ok":true,"path":"README.md","content":"hi","truncated":false}"#,
        );
        let mut transcript = Transcript::new();
        transcript.push(TranscriptItem::UserMessage {
            text: "Working directory: /repo\n\nread it".to_string(),
        });
        transcript.push(TranscriptItem::ToolCall(ToolCall::new(
            "call_1",
            "ferricode_read_file",
            r#"{"path":"README.md"}"#,
        )));
        transcript.push(TranscriptItem::ToolResult(output.clone()));
        let body = build_responses_body(&request, &transcript);

        assert_eq!(body["input"].as_array().unwrap().len(), 3);
        assert_eq!(body["input"][0]["role"], "user");
        assert_eq!(body["input"][1]["type"], "function_call");
        assert!(body["input"][1].get("id").is_none());
        assert_eq!(body["input"][2]["type"], "function_call_output");
        assert_eq!(body["input"][2]["call_id"], "call_1");
        assert_eq!(body["input"][2]["output"], output.output());
        assert_eq!(body["tools"].as_array().unwrap().len(), 2);
        assert_eq!(body["instructions"], ferricode_core::DEFAULT_INSTRUCTIONS);
    }

    /// Foreign opaque state belongs to its producer, so this provider must not
    /// leak it into an OpenAI request even though core preserved it in order.
    #[test]
    fn transcript_body_skips_foreign_opaque_items() {
        let request = ProviderRequest::new("read it", "/repo");
        let mut transcript = Transcript::new();
        transcript.push(TranscriptItem::UserMessage {
            text: "Working directory: /repo\n\nread it".to_string(),
        });
        transcript.push(TranscriptItem::ProviderOpaque {
            provider: "other-provider",
            payload: json!({ "type": "reasoning", "secret_state": "not ours" }),
        });

        let body = build_responses_body(&request, &transcript);

        assert_eq!(body["input"].as_array().unwrap().len(), 1);
        assert!(body.to_string().contains("Working directory: /repo"));
        assert!(!body.to_string().contains("secret_state"));
    }

    #[tokio::test]
    async fn provider_complete_posts_transcript_tool_output_and_returns_next_turn() {
        let (base_url, requests) = spawn_test_server(vec![TestResponse::json(
            r#"{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}"#,
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
        let mut transcript = Transcript::new();
        transcript.push(TranscriptItem::UserMessage {
            text: "Working directory: /repo\n\nread it".to_string(),
        });
        transcript.push(TranscriptItem::ProviderOpaque {
            provider: PROVIDER_NAME,
            payload: json!({ "type": "reasoning", "id": "rs_123" }),
        });
        transcript.push(TranscriptItem::ToolCall(ToolCall::new(
            "call_1",
            "ferricode_read_file",
            r#"{"path":"README.md"}"#,
        )));
        transcript.push(TranscriptItem::ToolResult(ToolOutput::new(
            "call_1",
            r#"{"ok":true,"path":"README.md","content":"hi","truncated":false}"#,
        )));

        let turn = provider
            .complete(
                &ProviderRequest::new("read it", "/repo"),
                &transcript,
                &NoopEventSink,
            )
            .await
            .unwrap();

        assert!(matches!(turn, ProviderTurn::Final { text, .. } if text == "done"));
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].contains("Working directory: /repo"));
        assert!(requests[0].contains(r#""type":"function_call_output""#));
        assert!(requests[0].contains(r#""call_id":"call_1""#));
        assert!(requests[0].contains(r#""output":"{\"ok\":true,"#));
        assert!(requests[0].contains(r#""type":"reasoning""#));
        assert!(!requests[0].contains(r#""id":"rs_123""#));
    }

    #[tokio::test]
    async fn provider_transcript_preserves_input_for_next_request() {
        let (base_url, requests) = spawn_test_server(vec![
            TestResponse::json(
                r#"{"output":[{"type":"function_call","id":"fc_123","call_id":"call_1","name":"ferricode_read_file","arguments":"{\"path\":\"README.md\"}"}]}"#,
            ),
            TestResponse::json(r#"{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}"#),
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

        let mut transcript = Transcript::new();
        transcript.push(TranscriptItem::UserMessage {
            text: "Working directory: /repo\n\nread it".to_string(),
        });
        let ProviderTurn::ToolCalls { items } = provider
            .complete(&request, &transcript, &NoopEventSink)
            .await
            .unwrap()
        else {
            panic!("expected tool call turn");
        };
        assert!(
            matches!(items.as_slice(), [TranscriptItem::ToolCall(call)] if call.id() == "call_1")
        );
        transcript.extend(items);
        transcript.push(TranscriptItem::ToolResult(ToolOutput::new(
            "call_1",
            r#"{"ok":true,"path":"README.md","content":"hi","truncated":false}"#,
        )));
        let turn = provider
            .complete(&request, &transcript, &NoopEventSink)
            .await
            .unwrap();

        assert!(matches!(turn, ProviderTurn::Final { text, .. } if text == "done"));
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[1].contains("Working directory: /repo"));
        assert!(requests[1].contains("read it"));
        assert!(requests[1].contains(r#""type":"function_call_output""#));
        // A stateless provider must rebuild the system prompt from the request,
        // rather than relying on the first request having left server state.
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
            r#"{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"assistant"}]}]}"#,
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
        assert!(requests[0].contains(r#""model":"gpt-5.6-luna""#));
        assert!(requests[0].contains(r#""effort":"high""#));
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
