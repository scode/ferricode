//! OpenAI Codex-compatible provider support for Ferricode.
//!
//! This crate intentionally does not implement the OpenAI Platform API-key
//! flow. It uses the browser PKCE OAuth shape used by Codex-compatible CLIs and
//! stores Codex OAuth token state and account metadata.

use base64::Engine;
use ferricode_core::{
    ModelProvider, ProviderError, ProviderRequest, ProviderTurn, ToolCall, ToolOutput,
};
use rand::RngCore;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::future::Future;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::str;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

const DEFAULT_ISSUER: &str = "https://auth.openai.com";
const CODEX_BACKEND_RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";
const OAUTH_AUTHORIZE_PATH: &str = "/oauth/authorize";
const OAUTH_TOKEN_PATH: &str = "/oauth/token";
const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_ORIGINATOR: &str = "codex_cli_rs";
const CALLBACK_HOST: &str = "127.0.0.1";
const CALLBACK_PORT: u16 = 1455;
const CALLBACK_PATH: &str = "/auth/callback";
const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const CODEX_SCOPES: &str =
    "openid profile email offline_access api.connectors.read api.connectors.invoke";
const MODEL: &str = "gpt-5.4";
const REASONING_EFFORT: &str = "medium";
const INSTRUCTIONS: &str = "You are Ferricode, a coding harness. Use the built-in filesystem tools when the user's request requires repository context. Start with a directory listing when you need to understand the working directory, then read specific relevant text files. Do not ask for clarification when the request can be handled by inspecting files.";
const REFRESH_SKEW: Duration = Duration::from_secs(60);
const CALLBACK_READ_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_STREAMING_OUTPUT_INDEX: usize = 1024;
const MAX_FUNCTION_CALL_ID_BYTES: usize = 256;
const MAX_FUNCTION_CALL_NAME_BYTES: usize = 256;
const MAX_FUNCTION_CALL_ARGUMENT_BYTES: usize = 16 * 1024;

/// Errors produced by the OpenAI Codex provider and auth flow.
#[derive(Debug, Error)]
pub enum OpenAiCodexError {
    #[error("OpenAI Codex auth is missing; run `ferric auth openai-codex` first")]
    MissingTokens,
    #[error("could not find a home directory for ~/.ferric/auth.toml")]
    MissingHome,
    #[error("auth file did not contain a ChatGPT account id")]
    MissingAccountId,
    #[error("OpenAI Codex response did not contain assistant text")]
    MissingAssistantText,
    #[error("OpenAI Codex auth callback port 1455 is already in use")]
    CallbackPortInUse,
    #[error("OpenAI Codex auth callback was not a valid HTTP request")]
    InvalidCallbackRequest,
    #[error("pasted OpenAI Codex auth callback URL was not http://localhost:1455/auth/callback")]
    InvalidPastedCallbackUrl,
    #[error("OpenAI Codex auth callback state did not match")]
    StateMismatch,
    #[error("OpenAI Codex auth callback returned {error}: {description}")]
    OAuthCallbackError { error: String, description: String },
    #[error("OpenAI Codex auth callback did not include an authorization code")]
    MissingAuthorizationCode,
    #[error("token exchange failed with status {status}: {body}")]
    TokenStatus {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("OpenAI Codex backend failed with status {status}: {body}")]
    BackendStatus {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("system clock is before the Unix epoch")]
    Clock,
    #[error("{0}")]
    Protocol(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error(transparent)]
    TomlDe(#[from] toml::de::Error),
    #[error(transparent)]
    TomlSer(#[from] toml::ser::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Url(#[from] url::ParseError),
}

impl From<OpenAiCodexError> for ProviderError {
    fn from(value: OpenAiCodexError) -> Self {
        Self::new(value.to_string())
    }
}

/// Top-level auth file persisted at `~/.ferric/auth.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AuthFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_codex: Option<OpenAiCodexAuth>,
}

/// OpenAI Codex token state.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiCodexAuth {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<TokenSet>,
}

/// Token state returned by the OpenAI Codex auth flow.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenSet {
    pub access_token: String,
    pub refresh_token: String,
    pub id_token: String,
    pub expires_at_unix_ms: u64,
    pub chatgpt_account_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chatgpt_plan_type: Option<String>,
}

/// Generated PKCE values for one browser OAuth attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PkceCodes {
    pub code_verifier: String,
    pub code_challenge: String,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    id_token: String,
    expires_in: u64,
}

/// Returns the default Ferricode auth path.
pub fn default_auth_path() -> Result<PathBuf, OpenAiCodexError> {
    let home = std::env::var_os("HOME").ok_or(OpenAiCodexError::MissingHome)?;
    Ok(PathBuf::from(home).join(".ferric").join("auth.toml"))
}

/// Loads the auth file, returning an empty configuration when the file is absent.
pub fn read_auth_file(path: &Path) -> Result<AuthFile, OpenAiCodexError> {
    let buffer = match fs::read_to_string(path) {
        Ok(buffer) => buffer,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(AuthFile::default()),
        Err(error) => return Err(error.into()),
    };
    if buffer.trim().is_empty() {
        return Ok(AuthFile::default());
    }
    Ok(toml::from_str(&buffer)?)
}

/// Writes auth state without falling back to ambient file permissions.
pub fn write_auth_file(path: &Path, auth: &AuthFile) -> Result<(), OpenAiCodexError> {
    if let Some(parent) = path.parent() {
        let parent_exists = parent.exists();
        fs::create_dir_all(parent)?;
        if !parent_exists {
            set_dir_permissions(parent)?;
        }
    }

    let content = toml::to_string_pretty(auth)?;
    reject_symlink(path)?;
    let temp_path = auth_temp_path(path);
    let mut file = create_secret_file(&temp_path)?;
    file.write_all(content.as_bytes())?;
    file.sync_all()?;
    drop(file);
    if let Err(error) = fs::rename(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(error.into());
    }
    Ok(())
}

fn auth_temp_path(path: &Path) -> PathBuf {
    let suffix = random_urlsafe(16);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("auth.toml");
    path.with_file_name(format!(".{file_name}.{suffix}.tmp"))
}

fn reject_symlink(path: &Path) -> Result<(), OpenAiCodexError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(OpenAiCodexError::Protocol(
            "auth file path must not be a symlink".to_string(),
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// Performs browser PKCE authentication and stores the returned Codex tokens.
pub async fn authenticate_openai_codex(
    path: &Path,
    output: &mut (impl Write + ?Sized),
) -> Result<(), OpenAiCodexError> {
    let listener = callback_listener_or_paste_only(bind_callback_listener().await, output)?;
    authenticate_openai_codex_with_inputs(
        path,
        output,
        DEFAULT_ISSUER,
        true,
        listener,
        read_pasted_callback_from_stdin(),
    )
    .await
}

/// Treats the loopback listener as a convenience path, not a hard auth requirement.
///
/// Port conflicts are common when a previous auth run is still around or another
/// tool uses the same Codex callback port. In that case auth can still complete
/// from a pasted localhost callback URL, so only non-port-conflict bind failures
/// abort the command.
fn callback_listener_or_paste_only(
    listener: Result<TcpListener, OpenAiCodexError>,
    output: &mut (impl Write + ?Sized),
) -> Result<Option<TcpListener>, OpenAiCodexError> {
    match listener {
        Ok(listener) => Ok(Some(listener)),
        Err(OpenAiCodexError::CallbackPortInUse) => {
            writeln!(
                output,
                "OpenAI Codex auth callback port 1455 is already in use; continuing with pasted redirect URL only."
            )?;
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// Binds the fixed Codex callback port and preserves port conflicts as a user-facing auth mode.
async fn bind_callback_listener() -> Result<TcpListener, OpenAiCodexError> {
    TcpListener::bind((CALLBACK_HOST, CALLBACK_PORT))
        .await
        .map_err(|error| {
            if error.kind() == ErrorKind::AddrInUse {
                OpenAiCodexError::CallbackPortInUse
            } else {
                OpenAiCodexError::Io(error)
            }
        })
}

/// One-shot source for a pasted browser callback URL.
///
/// `None` means stdin reached EOF before a URL was entered. That is fatal in
/// paste-only mode, but with a live HTTP listener it just disables the paste path
/// and lets the browser callback keep working.
type PastedCallbackFuture =
    Pin<Box<dyn Future<Output = Result<Option<String>, OpenAiCodexError>> + Send>>;

/// Runs the Codex PKCE browser flow through whichever callback path is available.
///
/// The HTTP listener and pasted URL path intentionally converge before state
/// validation and token exchange. While both paths are pending, the code only
/// races pasted input against accepting a connection; once a browser connection
/// is accepted, that callback is processed to completion so a late stdin EOF
/// cannot cancel an in-flight OAuth callback.
async fn authenticate_openai_codex_with_inputs(
    path: &Path,
    output: &mut (impl Write + ?Sized),
    issuer: &str,
    open_browser: bool,
    listener: Option<TcpListener>,
    pasted_callback: PastedCallbackFuture,
) -> Result<(), OpenAiCodexError> {
    let pkce = generate_pkce();
    let state = generate_state();
    let authorize_url = build_authorize_url(issuer, CODEX_CLIENT_ID, REDIRECT_URI, &pkce, &state);
    let client = reqwest::Client::new();

    if open_browser {
        let _ = webbrowser::open(&authorize_url);
    }
    writeln!(
        output,
        "Open this URL to sign in with OpenAI Codex auth:\n\n{}\n",
        authorize_url
    )?;
    writeln!(
        output,
        "If your browser ends at a localhost error, paste the full broken URL here:"
    )?;
    output.flush()?;

    let Some(listener) = listener.as_ref() else {
        let pasted = pasted_callback.await?.ok_or_else(|| {
            OpenAiCodexError::Protocol(
                "failed to read pasted OpenAI Codex auth callback URL".to_string(),
            )
        })?;
        process_pasted_callback_url(path, issuer, &state, &pkce, &pasted, &client).await?;
        writeln!(output, "OpenAI Codex authentication complete.")?;
        return Ok(());
    };

    let mut pasted_callback = Some(pasted_callback);
    loop {
        match pasted_callback.as_mut() {
            Some(callback) => {
                tokio::select! {
                    accepted = listener.accept() => {
                        let (stream, _) = accepted?;
                        if handle_http_callback_stream(stream, path, issuer, &state, &pkce, &client).await? {
                            writeln!(output, "OpenAI Codex authentication complete.")?;
                            return Ok(());
                        }
                    }
                    result = callback => {
                        match result? {
                            Some(pasted) => {
                                process_pasted_callback_url(path, issuer, &state, &pkce, &pasted, &client).await?;
                                writeln!(output, "OpenAI Codex authentication complete.")?;
                                return Ok(());
                            }
                            None => {
                                pasted_callback = None;
                            }
                        }
                    }
                }
            }
            None => {
                let (stream, _) = listener.accept().await?;
                if handle_http_callback_stream(stream, path, issuer, &state, &pkce, &client).await?
                {
                    writeln!(output, "OpenAI Codex authentication complete.")?;
                    return Ok(());
                }
            }
        }
    }
}

/// Reads at most one terminal line without tying the async runtime to blocking stdin.
///
/// A detached OS thread is deliberate here. The HTTP callback may finish first,
/// and a Tokio blocking task stuck in `stdin().read_line()` would still need to be
/// joined before the CLI could exit. The returned future resolves to `None` on
/// EOF so the caller can distinguish "no pasted URL is coming" from malformed
/// pasted input.
fn read_pasted_callback_from_stdin() -> PastedCallbackFuture {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let result = std::io::stdin()
            .read_line(&mut line)
            .map(|bytes| (bytes > 0).then_some(line))
            .map_err(OpenAiCodexError::Io);
        let _ = sender.send(result);
    });

    Box::pin(async move {
        receiver.await.map_err(|_| {
            OpenAiCodexError::Protocol(
                "failed to read pasted OpenAI Codex auth callback URL".to_string(),
            )
        })?
    })
}

/// Processes an already-accepted callback connection to a terminal response.
///
/// This function owns the accepted stream until it has either completed auth or
/// sent a recoverable error response. Keeping this separate from `accept()` lets
/// the auth loop race pasted input only while no browser connection is in hand.
async fn handle_http_callback_stream(
    mut stream: TcpStream,
    auth_path: &Path,
    issuer: &str,
    expected_state: &str,
    pkce: &PkceCodes,
    client: &reqwest::Client,
) -> Result<bool, OpenAiCodexError> {
    let result = match timeout(CALLBACK_READ_TIMEOUT, read_http_target(&mut stream)).await {
        Ok(Ok(target)) => {
            process_callback_target(auth_path, issuer, expected_state, pkce, &target, client).await
        }
        Ok(Err(error)) => Err(error),
        Err(_) => Err(OpenAiCodexError::InvalidCallbackRequest),
    };

    match result {
        Ok(CallbackAction::Continue) => {
            write_http_response(&mut stream, 404, "Not Found").await?;
            Ok(false)
        }
        Ok(CallbackAction::Complete) => {
            write_http_response(&mut stream, 200, "OpenAI Codex authentication complete.").await?;
            Ok(true)
        }
        Err(error) => {
            write_http_response(&mut stream, 400, &error.to_string()).await?;
            if callback_error_is_recoverable(&error) {
                Ok(false)
            } else {
                Err(error)
            }
        }
    }
}

/// OpenAI Codex-backed model provider.
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
        let input_items = response_input_items(&body);
        let turn = self
            .send_responses_request(&tokens, &body)
            .await
            .map_err(ProviderError::from)?;
        Ok(attach_input_items(turn, input_items))
    }

    async fn resume<'a>(
        &'a self,
        state: Self::State,
        tool_outputs: &'a [ToolOutput],
    ) -> Result<ProviderTurn<Self::State>, ProviderError> {
        let tokens = self.authenticated_tokens().await?;
        let body = build_tool_outputs_body(state, tool_outputs);
        let input_items = response_input_items(&body);
        let turn = self
            .send_responses_request(&tokens, &body)
            .await
            .map_err(ProviderError::from)?;
        Ok(attach_input_items(turn, input_items))
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
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiCodexState {
    input_items: Vec<Value>,
    output_items: Vec<Value>,
}

/// Builds the hardcoded bootstrap Responses body.
pub fn build_responses_body(request: &ProviderRequest) -> Value {
    json!({
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "stream": true,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": format!("Working directory: {}\n\n{}", request.working_directory(), request.prompt())
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
        "instructions": INSTRUCTIONS,
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

fn attach_input_items(
    turn: ProviderTurn<OpenAiCodexState>,
    input_items: Vec<Value>,
) -> ProviderTurn<OpenAiCodexState> {
    match turn {
        ProviderTurn::ToolCalls { mut state, calls } => {
            state.input_items = input_items;
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
    json!([
        {
            "type": "function",
            "name": "ferricode_list_directory",
            "description": "List one directory under the request working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "A relative path under the request working directory."
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            },
            "strict": true
        },
        {
            "type": "function",
            "name": "ferricode_read_file",
            "description": "Read one UTF-8 text file under the request working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "A relative path under the request working directory."
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            },
            "strict": true
        }
    ])
}

/// Parses either JSON or minimal SSE `data:` events into assistant text.
pub fn parse_assistant_text(text: &str) -> Result<String, OpenAiCodexError> {
    match parse_assistant_turn(text)? {
        ProviderTurn::Final(text) => Ok(text),
        ProviderTurn::ToolCalls { .. } => Err(OpenAiCodexError::MissingAssistantText),
    }
}

fn parse_assistant_turn(text: &str) -> Result<ProviderTurn<OpenAiCodexState>, OpenAiCodexError> {
    let trimmed = text.trim();
    if trimmed.lines().any(|line| line.starts_with("data:")) {
        return parse_sse_assistant_text(trimmed);
    }

    let value: Value = serde_json::from_str(trimmed)?;
    parse_response_turn(&value)
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

/// Builds the Codex-compatible browser authorization URL.
pub fn build_authorize_url(
    issuer: &str,
    client_id: &str,
    redirect_uri: &str,
    pkce: &PkceCodes,
    state: &str,
) -> String {
    let query = [
        ("response_type", "code"),
        ("client_id", client_id),
        ("redirect_uri", redirect_uri),
        ("scope", CODEX_SCOPES),
        ("code_challenge", pkce.code_challenge.as_str()),
        ("code_challenge_method", "S256"),
        ("id_token_add_organizations", "true"),
        ("codex_cli_simplified_flow", "true"),
        ("state", state),
        ("originator", CODEX_ORIGINATOR),
    ];
    let qs = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(query)
        .finish();
    format!(
        "{}{}?{}",
        issuer.trim_end_matches('/'),
        OAUTH_AUTHORIZE_PATH,
        qs
    )
}

fn generate_pkce() -> PkceCodes {
    let code_verifier = random_urlsafe(32);
    let code_challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(Sha256::digest(code_verifier.as_bytes()));
    PkceCodes {
        code_verifier,
        code_challenge,
    }
}

fn generate_state() -> String {
    random_urlsafe(32)
}

fn random_urlsafe(len: usize) -> String {
    let mut bytes = vec![0; len];
    rand::rng().fill_bytes(&mut bytes);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallbackAction {
    Continue,
    Complete,
}

/// Converts an HTTP request target into the shared callback URL path.
///
/// Request targets are accepted only in origin-form (`/path?...`). Absolute URLs
/// are not needed for the local callback listener and are treated as malformed
/// callback requests.
async fn process_callback_target(
    auth_path: &Path,
    issuer: &str,
    expected_state: &str,
    pkce: &PkceCodes,
    target: &str,
    client: &reqwest::Client,
) -> Result<CallbackAction, OpenAiCodexError> {
    let parsed = callback_url_from_request_target(target)?;
    process_callback_url(auth_path, issuer, expected_state, pkce, parsed, client).await
}

/// Validates a pasted browser URL before using the shared callback processor.
///
/// Pasted URLs are untrusted terminal input. They must be the exact localhost
/// callback origin/path used by the Codex OAuth flow before the code is allowed
/// to reach state validation or token exchange.
async fn process_pasted_callback_url(
    auth_path: &Path,
    issuer: &str,
    expected_state: &str,
    pkce: &PkceCodes,
    pasted_url: &str,
    client: &reqwest::Client,
) -> Result<CallbackAction, OpenAiCodexError> {
    let parsed = callback_url_from_pasted_url(pasted_url)?;
    process_callback_url(auth_path, issuer, expected_state, pkce, parsed, client).await
}

/// Applies the OAuth callback contract shared by HTTP and pasted callbacks.
///
/// Non-callback paths return `Continue` so the HTTP listener can ignore browser
/// noise such as `/favicon.ico`. Once the path is the callback path, state
/// validation happens before any token exchange or auth file write.
async fn process_callback_url(
    auth_path: &Path,
    issuer: &str,
    expected_state: &str,
    pkce: &PkceCodes,
    parsed: url::Url,
    client: &reqwest::Client,
) -> Result<CallbackAction, OpenAiCodexError> {
    if parsed.path() != CALLBACK_PATH {
        return Ok(CallbackAction::Continue);
    }

    let params = parsed.query_pairs().into_owned().collect::<HashMap<_, _>>();
    if params.get("state").map(String::as_str) != Some(expected_state) {
        return Err(OpenAiCodexError::StateMismatch);
    }
    if let Some(error) = params.get("error") {
        return Err(OpenAiCodexError::OAuthCallbackError {
            error: error.clone(),
            description: params
                .get("error_description")
                .cloned()
                .unwrap_or_else(|| "no error description returned".to_string()),
        });
    }
    let code = params
        .get("code")
        .filter(|code| !code.is_empty())
        .ok_or(OpenAiCodexError::MissingAuthorizationCode)?;
    let response = exchange_authorization_code(client, issuer, code, &pkce.code_verifier).await?;
    let tokens = tokens_from_response(response, None)?;
    let mut auth = read_auth_file(auth_path)?;
    auth.openai_codex
        .get_or_insert_with(Default::default)
        .tokens = Some(tokens);
    write_auth_file(auth_path, &auth)?;
    Ok(CallbackAction::Complete)
}

/// Reconstructs a localhost callback URL from an HTTP request target.
fn callback_url_from_request_target(target: &str) -> Result<url::Url, OpenAiCodexError> {
    if !target.starts_with('/') {
        return Err(OpenAiCodexError::InvalidCallbackRequest);
    }
    Ok(url::Url::parse(&format!("http://localhost{target}"))?)
}

/// Parses pasted callback input and rejects URLs outside the Codex localhost callback.
///
/// `read_line` keeps the terminal newline, so this parser trims surrounding
/// whitespace before validation. It still rejects alternate schemes, hosts,
/// ports, and paths before state or authorization code handling.
fn callback_url_from_pasted_url(pasted_url: &str) -> Result<url::Url, OpenAiCodexError> {
    let parsed = url::Url::parse(pasted_url.trim())?;
    if parsed.scheme() != "http"
        || parsed.host_str() != Some("localhost")
        || parsed.port_or_known_default() != Some(CALLBACK_PORT)
        || parsed.path() != CALLBACK_PATH
    {
        return Err(OpenAiCodexError::InvalidPastedCallbackUrl);
    }
    Ok(parsed)
}

/// Classifies callback failures that should keep the HTTP listener alive.
///
/// Browser noise and user retryable callback mistakes get an HTTP error response
/// but do not end the auth command. OAuth provider errors are not recoverable
/// here: they represent the actual authorization result.
fn callback_error_is_recoverable(error: &OpenAiCodexError) -> bool {
    matches!(
        error,
        OpenAiCodexError::InvalidCallbackRequest
            | OpenAiCodexError::StateMismatch
            | OpenAiCodexError::MissingAuthorizationCode
            | OpenAiCodexError::Url(_)
    )
}

/// Reads the request target from the small HTTP subset needed for OAuth callbacks.
///
/// This is intentionally not a general HTTP parser. It accepts a single GET
/// request line, stops after headers, and caps request bytes so callback noise
/// cannot grow memory without bound.
async fn read_http_target(stream: &mut TcpStream) -> Result<String, OpenAiCodexError> {
    let mut request = Vec::new();
    let mut buffer = [0; 1024];
    loop {
        let bytes = stream.read(&mut buffer).await?;
        if bytes == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..bytes]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
        if request.len() > 16 * 1024 {
            return Err(OpenAiCodexError::InvalidCallbackRequest);
        }
    }

    let request = String::from_utf8_lossy(&request);
    let first_line = request
        .lines()
        .next()
        .ok_or(OpenAiCodexError::InvalidCallbackRequest)?;
    let mut pieces = first_line.split_whitespace();
    match (pieces.next(), pieces.next(), pieces.next()) {
        (Some("GET"), Some(target), Some(_version)) => Ok(target.to_string()),
        _ => Err(OpenAiCodexError::InvalidCallbackRequest),
    }
}

async fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    body: &str,
) -> Result<(), OpenAiCodexError> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: text/plain; charset=utf-8\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

async fn exchange_authorization_code(
    client: &reqwest::Client,
    issuer: &str,
    code: &str,
    code_verifier: &str,
) -> Result<TokenResponse, OpenAiCodexError> {
    post_token_form(
        client,
        issuer,
        &[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", REDIRECT_URI),
            ("code_verifier", code_verifier),
        ],
    )
    .await
}

async fn refresh_access_token(
    client: &reqwest::Client,
    issuer: &str,
    refresh_token: &str,
) -> Result<TokenResponse, OpenAiCodexError> {
    post_token_form(
        client,
        issuer,
        &[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
        ],
    )
    .await
}

async fn post_token_form(
    client: &reqwest::Client,
    issuer: &str,
    fields: &[(&str, &str)],
) -> Result<TokenResponse, OpenAiCodexError> {
    let mut body = vec![("client_id", CODEX_CLIENT_ID)];
    body.extend_from_slice(fields);

    let response = client
        .post(format!(
            "{}{}",
            issuer.trim_end_matches('/'),
            OAUTH_TOKEN_PATH
        ))
        .form(&body)
        .send()
        .await?;

    let status = response.status();
    let text = response.text().await?;
    if !status.is_success() {
        return Err(OpenAiCodexError::TokenStatus { status, body: text });
    }
    Ok(serde_json::from_str(&text)?)
}

fn parse_sse_assistant_text(
    text: &str,
) -> Result<ProviderTurn<OpenAiCodexState>, OpenAiCodexError> {
    let mut accumulator = SseAccumulator::default();
    for line in text.lines() {
        if accumulator.process_line(line.as_bytes())? == SseDataAction::Complete {
            break;
        }
    }

    accumulator.into_provider_turn()
}

async fn parse_sse_assistant_stream(
    response: &mut reqwest::Response,
) -> Result<ProviderTurn<OpenAiCodexState>, OpenAiCodexError> {
    let mut pending = Vec::new();
    let mut accumulator = SseAccumulator::default();

    while let Some(chunk) = response.chunk().await? {
        pending.extend_from_slice(&chunk);
        while let Some(line_end) = pending.iter().position(|byte| *byte == b'\n') {
            let line = pending.drain(..=line_end).collect::<Vec<_>>();
            if accumulator.process_line(&line)? == SseDataAction::Complete {
                return accumulator.into_provider_turn();
            }
        }
    }

    if !pending.is_empty() {
        accumulator.process_line(&pending)?;
    }
    accumulator.into_provider_turn()
}

fn parse_response_turn(value: &Value) -> Result<ProviderTurn<OpenAiCodexState>, OpenAiCodexError> {
    let output_items = value
        .get("output")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let calls = collect_function_calls(output_items.iter())?;
    if !calls.is_empty() {
        return Ok(ProviderTurn::ToolCalls {
            state: OpenAiCodexState {
                input_items: Vec::new(),
                output_items,
            },
            calls,
        });
    }

    let text = extract_text_from_response(value)
        .filter(|value| !value.trim().is_empty())
        .ok_or(OpenAiCodexError::MissingAssistantText)?;
    Ok(ProviderTurn::Final(text))
}

#[derive(Default)]
struct SseAccumulator {
    text: String,
    output_items: Vec<Value>,
    function_calls: BTreeMap<usize, StreamingFunctionCall>,
}

impl SseAccumulator {
    fn process_line(&mut self, line: &[u8]) -> Result<SseDataAction, OpenAiCodexError> {
        let line = parse_sse_line(line)?;
        let Some(data) = line.strip_prefix("data:") else {
            return Ok(SseDataAction::Continue);
        };
        self.process_data_line(data)
    }

    fn process_data_line(&mut self, data: &str) -> Result<SseDataAction, OpenAiCodexError> {
        let data = data.trim();
        if data.is_empty() {
            return Ok(SseDataAction::Continue);
        }
        if data == "[DONE]" {
            return Ok(SseDataAction::Complete);
        }

        let value = serde_json::from_str::<Value>(data)?;
        if let Some(text) = extract_text_from_event(&value) {
            self.text.push_str(&text);
        }

        match value.get("type").and_then(Value::as_str) {
            Some("response.output_item.added") => {
                let index = required_event_output_index(&value)?;
                let item = required_event_value(&value, "item")?;
                if is_function_call_item(item) {
                    self.function_calls
                        .entry(index)
                        .or_default()
                        .merge_item(item)?;
                }
            }
            Some("response.function_call_arguments.delta") => {
                let index = required_event_output_index(&value)?;
                let delta = required_event_string(&value, "delta")?;
                let call = self.function_calls.entry(index).or_default();
                if call.arguments.len() + delta.len() > MAX_FUNCTION_CALL_ARGUMENT_BYTES {
                    return Err(OpenAiCodexError::Protocol(format!(
                        "streamed function call arguments exceeded the limit of {MAX_FUNCTION_CALL_ARGUMENT_BYTES} bytes"
                    )));
                }
                call.arguments.push_str(delta);
            }
            Some("response.function_call_arguments.done") => {
                let index = required_event_output_index(&value)?;
                let arguments = required_event_string(&value, "arguments")?;
                validate_function_call_field("arguments", arguments)?;
                self.function_calls.entry(index).or_default().arguments = arguments.to_string();
            }
            Some("response.output_item.done") => {
                let item = required_event_value(&value, "item")?;
                if is_function_call_item(item) {
                    let index = required_event_output_index(&value)?;
                    self.function_calls
                        .entry(index)
                        .or_default()
                        .merge_item(item)?;
                }
                self.output_items.push(item.clone());
            }
            Some("response.completed") => return Ok(SseDataAction::Complete),
            Some("response.failed" | "response.incomplete") => {
                return Err(OpenAiCodexError::Protocol(
                    "OpenAI Codex backend ended the response without completing it".to_string(),
                ));
            }
            _ => {}
        }

        Ok(SseDataAction::Continue)
    }

    fn into_provider_turn(mut self) -> Result<ProviderTurn<OpenAiCodexState>, OpenAiCodexError> {
        if !self.function_calls.is_empty() {
            self.merge_streaming_function_items()?;
            let calls = collect_streaming_function_calls(&self.function_calls);
            return Ok(ProviderTurn::ToolCalls {
                state: OpenAiCodexState {
                    input_items: Vec::new(),
                    output_items: self.output_items,
                },
                calls,
            });
        }

        completed_sse_text(self.text).map(ProviderTurn::Final)
    }

    fn merge_streaming_function_items(&mut self) -> Result<(), OpenAiCodexError> {
        for (index, call) in &self.function_calls {
            if *index >= MAX_STREAMING_OUTPUT_INDEX {
                return Err(OpenAiCodexError::Protocol(format!(
                    "streamed function call output_index {index} exceeded the limit of {MAX_STREAMING_OUTPUT_INDEX}"
                )));
            }
            let item = call.to_item()?;
            if self.output_items.len() <= *index {
                self.output_items.resize(*index + 1, Value::Null);
            }
            self.output_items[*index] = item;
        }
        self.output_items.retain(|item| !item.is_null());
        Ok(())
    }
}

fn parse_sse_line(line: &[u8]) -> Result<&str, OpenAiCodexError> {
    let line = str::from_utf8(line)
        .map_err(|_| OpenAiCodexError::Protocol("SSE response was not valid UTF-8".to_string()))?
        .trim_end_matches(['\r', '\n']);
    Ok(line)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SseDataAction {
    Continue,
    Complete,
}

fn completed_sse_text(joined: String) -> Result<String, OpenAiCodexError> {
    if joined.trim().is_empty() {
        Err(OpenAiCodexError::MissingAssistantText)
    } else {
        Ok(joined)
    }
}

fn extract_text_from_response(value: &Value) -> Option<String> {
    if let Some(text) = value.get("output_text").and_then(Value::as_str) {
        return Some(text.to_string());
    }

    let output = value.get("output")?.as_array()?;
    collect_text(output.iter().filter_map(extract_text_from_output_item))
}

fn extract_text_from_output_item(value: &Value) -> Option<String> {
    let content = value.get("content")?.as_array()?;
    collect_text(content.iter().filter_map(extract_text_from_content_item))
}

fn extract_text_from_content_item(value: &Value) -> Option<String> {
    if !matches!(
        value.get("type").and_then(Value::as_str),
        Some("output_text")
    ) {
        return None;
    }

    value
        .get("text")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn extract_text_from_event(value: &Value) -> Option<String> {
    let map = value.as_object()?;
    if let Some(Value::String(text)) = map.get("output_text") {
        return Some(text.clone());
    }
    if matches!(
        map.get("type").and_then(Value::as_str),
        Some("output_text" | "response.output_text.delta")
    ) && let Some(Value::String(text)) = map.get("text").or_else(|| map.get("delta"))
    {
        return Some(text.clone());
    }
    None
}

fn collect_function_calls<'a>(
    items: impl Iterator<Item = &'a Value>,
) -> Result<Vec<ToolCall>, OpenAiCodexError> {
    items
        .filter_map(function_call_from_item)
        .collect::<Result<Vec<_>, _>>()
}

fn function_call_from_item(item: &Value) -> Option<Result<ToolCall, OpenAiCodexError>> {
    if !is_function_call_item(item) {
        return None;
    }
    Some((|| {
        Ok(ToolCall::new(
            required_function_call_string(item, "call_id")?,
            required_function_call_string(item, "name")?,
            required_function_call_string(item, "arguments")?,
        ))
    })())
}

fn required_function_call_string<'a>(
    item: &'a Value,
    field: &str,
) -> Result<&'a str, OpenAiCodexError> {
    let value = item.get(field).and_then(Value::as_str).ok_or_else(|| {
        OpenAiCodexError::Protocol(format!(
            "function_call item did not include string `{field}`"
        ))
    })?;
    validate_function_call_field(field, value)?;
    Ok(value)
}

fn validate_function_call_field(field: &str, value: &str) -> Result<(), OpenAiCodexError> {
    let Some(limit) = function_call_field_limit(field) else {
        return Ok(());
    };
    if value.len() > limit {
        return Err(OpenAiCodexError::Protocol(format!(
            "function_call `{field}` exceeded the limit of {limit} bytes"
        )));
    }
    Ok(())
}

fn function_call_field_limit(field: &str) -> Option<usize> {
    match field {
        "call_id" => Some(MAX_FUNCTION_CALL_ID_BYTES),
        "name" => Some(MAX_FUNCTION_CALL_NAME_BYTES),
        "arguments" => Some(MAX_FUNCTION_CALL_ARGUMENT_BYTES),
        _ => None,
    }
}

fn is_function_call_item(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some("function_call")
    )
}

fn event_output_index(value: &Value) -> Option<usize> {
    value
        .get("output_index")
        .and_then(Value::as_u64)
        .and_then(|value| value.try_into().ok())
}

fn required_event_output_index(value: &Value) -> Result<usize, OpenAiCodexError> {
    event_output_index(value).ok_or_else(|| {
        OpenAiCodexError::Protocol(
            "function call stream event did not include integer `output_index`".to_string(),
        )
    })
}

fn required_event_value<'a>(value: &'a Value, field: &str) -> Result<&'a Value, OpenAiCodexError> {
    value.get(field).ok_or_else(|| {
        OpenAiCodexError::Protocol(format!(
            "function call stream event did not include `{field}`"
        ))
    })
}

fn required_event_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, OpenAiCodexError> {
    value.get(field).and_then(Value::as_str).ok_or_else(|| {
        OpenAiCodexError::Protocol(format!(
            "function call stream event did not include string `{field}`"
        ))
    })
}

#[derive(Debug, Clone, Default)]
struct StreamingFunctionCall {
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl StreamingFunctionCall {
    fn merge_item(&mut self, item: &Value) -> Result<(), OpenAiCodexError> {
        if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
            validate_function_call_field("call_id", call_id)?;
            self.call_id = Some(call_id.to_string());
        }
        if let Some(name) = item.get("name").and_then(Value::as_str) {
            validate_function_call_field("name", name)?;
            self.name = Some(name.to_string());
        }
        if let Some(arguments) = item.get("arguments").and_then(Value::as_str)
            && !arguments.is_empty()
        {
            validate_function_call_field("arguments", arguments)?;
            self.arguments = arguments.to_string();
        }
        Ok(())
    }

    fn to_item(&self) -> Result<Value, OpenAiCodexError> {
        let call_id = self.call_id.as_deref().ok_or_else(|| {
            OpenAiCodexError::Protocol("streamed function call did not include call_id".to_string())
        })?;
        let name = self.name.as_deref().ok_or_else(|| {
            OpenAiCodexError::Protocol("streamed function call did not include name".to_string())
        })?;
        Ok(json!({
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": self.arguments,
        }))
    }
}

fn collect_streaming_function_calls(
    calls: &BTreeMap<usize, StreamingFunctionCall>,
) -> Vec<ToolCall> {
    calls
        .values()
        .filter_map(|call| {
            Some(ToolCall::new(
                call.call_id.as_deref()?,
                call.name.as_deref()?,
                call.arguments.as_str(),
            ))
        })
        .collect()
}

fn collect_text(pieces: impl Iterator<Item = String>) -> Option<String> {
    let joined = pieces.collect::<String>();
    (!joined.is_empty()).then_some(joined)
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

fn token_needs_refresh(tokens: &TokenSet, now_unix_ms: u64) -> bool {
    now_unix_ms.saturating_add(REFRESH_SKEW.as_millis() as u64) >= tokens.expires_at_unix_ms
}

fn tokens_from_response(
    response: TokenResponse,
    fallback_refresh_token: Option<&str>,
) -> Result<TokenSet, OpenAiCodexError> {
    let claims = parse_chatgpt_claims(&response.id_token)?;
    let account_id = claims
        .chatgpt_account_id
        .ok_or(OpenAiCodexError::MissingAccountId)?;
    let refresh_token = response
        .refresh_token
        .or_else(|| fallback_refresh_token.map(str::to_string))
        .ok_or_else(|| {
            OpenAiCodexError::Protocol("token response did not contain a refresh token".to_string())
        })?;
    let expires_in_ms = response
        .expires_in
        .checked_mul(1000)
        .ok_or_else(|| OpenAiCodexError::Protocol("token expiry overflowed".to_string()))?;
    let expires_at_unix_ms = now_unix_ms()?
        .checked_add(expires_in_ms)
        .ok_or_else(|| OpenAiCodexError::Protocol("token expiry overflowed".to_string()))?;

    Ok(TokenSet {
        access_token: response.access_token,
        refresh_token,
        id_token: response.id_token,
        expires_at_unix_ms,
        chatgpt_account_id: account_id,
        chatgpt_plan_type: claims.chatgpt_plan_type,
    })
}

#[derive(Debug, Deserialize)]
struct JwtClaims {
    #[serde(rename = "https://api.openai.com/auth")]
    auth: Option<ChatGptClaims>,
}

#[derive(Debug, Deserialize)]
struct ChatGptClaims {
    chatgpt_account_id: Option<String>,
    chatgpt_plan_type: Option<String>,
}

fn parse_chatgpt_claims(jwt: &str) -> Result<ChatGptClaims, OpenAiCodexError> {
    let payload = jwt
        .split('.')
        .nth(1)
        .ok_or(OpenAiCodexError::MissingAccountId)?;
    let claims: JwtClaims = serde_json::from_slice(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(payload)
            .map_err(|_| OpenAiCodexError::MissingAccountId)?,
    )?;
    Ok(claims.auth.unwrap_or(ChatGptClaims {
        chatgpt_account_id: None,
        chatgpt_plan_type: None,
    }))
}

#[cfg(unix)]
fn create_secret_file(path: &Path) -> Result<fs::File, OpenAiCodexError> {
    use std::os::unix::fs::OpenOptionsExt;

    Ok(fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)?)
}

#[cfg(not(unix))]
fn create_secret_file(path: &Path) -> Result<fs::File, OpenAiCodexError> {
    let _ = path;
    Err(OpenAiCodexError::Protocol(
        "OpenAI Codex auth storage requires private Unix file permissions".to_string(),
    ))
}

fn now_unix_ms() -> Result<u64, OpenAiCodexError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| OpenAiCodexError::Clock)?
        .as_millis() as u64)
}

#[cfg(unix)]
fn set_dir_permissions(path: &Path) -> Result<(), OpenAiCodexError> {
    use std::os::unix::fs::PermissionsExt;

    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_dir_permissions(_path: &Path) -> Result<(), OpenAiCodexError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use ferricode_core::ProviderRequest;
    use tempfile::tempdir;

    #[test]
    fn auth_write_creates_parent_directory_and_auth_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(".ferric").join("auth.toml");

        write_auth_file(
            &path,
            &auth_with_tokens("access", "refresh", 9_999_999_999_999),
        )
        .unwrap();

        assert!(path.exists());
        let auth = read_auth_file(&path).unwrap();
        assert_eq!(
            auth.openai_codex.unwrap().tokens.unwrap().access_token,
            "access"
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            assert_eq!(
                fs::metadata(path.parent().unwrap())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn writing_auth_rejects_symlink_path() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let target = dir.path().join("target.toml");
        let path = dir.path().join("auth.toml");
        fs::write(&target, "unchanged").unwrap();
        symlink(&target, &path).unwrap();

        let error = write_auth_file(&path, &AuthFile::default()).unwrap_err();

        assert!(matches!(error, OpenAiCodexError::Protocol(_)));
        assert_eq!(fs::read_to_string(&target).unwrap(), "unchanged");
    }

    #[cfg(unix)]
    #[test]
    fn writing_auth_does_not_repermission_existing_parent_directory() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let parent = dir.path().join("existing-config");
        fs::create_dir(&parent).unwrap();
        fs::set_permissions(&parent, fs::Permissions::from_mode(0o755)).unwrap();

        write_auth_file(&parent.join("auth.toml"), &AuthFile::default()).unwrap();

        assert_eq!(
            fs::metadata(parent).unwrap().permissions().mode() & 0o777,
            0o755
        );
    }

    #[test]
    fn authorize_url_uses_codex_pkce_contract() {
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        let url = build_authorize_url(
            "https://auth.openai.com",
            CODEX_CLIENT_ID,
            REDIRECT_URI,
            &pkce,
            "state",
        );
        let parsed = url::Url::parse(&url).unwrap();
        let params = parsed.query_pairs().into_owned().collect::<HashMap<_, _>>();

        assert_eq!(
            parsed.as_str().split('?').next().unwrap(),
            "https://auth.openai.com/oauth/authorize"
        );
        assert_eq!(params.get("response_type").unwrap(), "code");
        assert_eq!(params.get("client_id").unwrap(), CODEX_CLIENT_ID);
        assert_eq!(params.get("redirect_uri").unwrap(), REDIRECT_URI);
        assert_eq!(params.get("scope").unwrap(), CODEX_SCOPES);
        assert_eq!(params.get("code_challenge").unwrap(), "challenge");
        assert_eq!(params.get("code_challenge_method").unwrap(), "S256");
        assert_eq!(params.get("id_token_add_organizations").unwrap(), "true");
        assert_eq!(params.get("codex_cli_simplified_flow").unwrap(), "true");
        assert_eq!(params.get("state").unwrap(), "state");
        assert_eq!(params.get("originator").unwrap(), CODEX_ORIGINATOR);
        assert!(!params.contains_key("client_secret"));
    }

    #[test]
    fn generated_pkce_challenge_is_sha256_s256() {
        let pkce = generate_pkce();
        let expected = URL_SAFE_NO_PAD.encode(Sha256::digest(pkce.code_verifier.as_bytes()));

        assert_eq!(pkce.code_challenge, expected);
    }

    #[test]
    fn response_body_uses_hardcoded_model_and_effort() {
        let request = ProviderRequest::new("summarize this repository", "/repo");

        let body = build_responses_body(&request);

        assert_eq!(body["model"], MODEL);
        assert_eq!(body["instructions"], INSTRUCTIONS);
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
    fn parses_json_response_text() {
        let text = r#"{"output":[{"content":[{"type":"output_text","text":"hello"}]}]}"#;

        assert_eq!(parse_assistant_text(text).unwrap(), "hello");
    }

    #[test]
    fn parses_sse_response_text() {
        let text = r#"data: {"type":"response.output_text.delta","delta":"hel"}
data: {"type":"response.output_text.delta","delta":"lo"}
data: [DONE]"#;

        assert_eq!(parse_assistant_text(text).unwrap(), "hello");
    }

    #[test]
    fn parses_sse_response_with_event_fields() {
        let text = r#"event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"hello"}
data: [DONE]"#;

        assert_eq!(parse_assistant_text(text).unwrap(), "hello");
    }

    #[test]
    fn parses_json_function_call_turn() {
        let text = r#"{"output":[{"type":"function_call","call_id":"call_1","name":"ferricode_list_directory","arguments":"{\"path\":\".\"}"}]}"#;

        let turn = parse_assistant_turn(text).unwrap();

        let ProviderTurn::ToolCalls { state, calls } = turn else {
            panic!("expected function call turn");
        };
        assert_eq!(state.output_items.len(), 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id(), "call_1");
        assert_eq!(calls[0].name(), "ferricode_list_directory");
        assert_eq!(calls[0].arguments(), r#"{"path":"."}"#);
    }

    #[test]
    fn json_function_call_requires_provider_fields() {
        let text = r#"{"output":[{"type":"function_call","call_id":"call_1","name":"ferricode_list_directory"}]}"#;

        let error = parse_assistant_turn(text).unwrap_err();

        assert!(error.to_string().contains("arguments"));
    }

    #[test]
    fn json_function_call_rejects_oversized_arguments() {
        let body = json!({
            "output": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "ferricode_read_file",
                "arguments": "x".repeat((16 * 1024) + 1),
            }]
        })
        .to_string();

        let error = parse_assistant_turn(&body).unwrap_err();

        assert!(error.to_string().contains("arguments"));
        assert!(error.to_string().contains("limit"));
    }

    #[test]
    fn parses_streamed_function_call_arguments() {
        let text = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":""}}
data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"path\":\"READ"}
data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"ME.md\"}"}
data: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"path\":\"README.md\"}"}
data: {"type":"response.completed"}"#;

        let turn = parse_assistant_turn(text).unwrap();

        let ProviderTurn::ToolCalls { state, calls } = turn else {
            panic!("expected function call turn");
        };
        assert_eq!(state.output_items.len(), 1);
        assert_eq!(
            state.output_items[0]["arguments"],
            r#"{"path":"README.md"}"#
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id(), "call_1");
        assert_eq!(calls[0].name(), "ferricode_read_file");
        assert_eq!(calls[0].arguments(), r#"{"path":"README.md"}"#);
    }

    #[test]
    fn parses_streamed_function_call_from_done_item() {
        let text = r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":"{\"path\":\"README.md\"}"}}
data: {"type":"response.completed"}"#;

        let turn = parse_assistant_turn(text).unwrap();

        let ProviderTurn::ToolCalls { state, calls } = turn else {
            panic!("expected function call turn");
        };
        assert_eq!(state.output_items.len(), 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id(), "call_1");
        assert_eq!(calls[0].name(), "ferricode_read_file");
        assert_eq!(calls[0].arguments(), r#"{"path":"README.md"}"#);
    }

    #[test]
    fn streamed_function_call_requires_provider_identifiers() {
        let text = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","arguments":"{}"}}
data: {"type":"response.completed"}"#;

        let error = parse_assistant_turn(text).unwrap_err();

        assert!(error.to_string().contains("call_id"));
    }

    #[test]
    fn streamed_function_call_argument_delta_requires_delta() {
        let text = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":""}}
data: {"type":"response.function_call_arguments.delta","output_index":0}
data: {"type":"response.completed"}"#;

        let error = parse_assistant_turn(text).unwrap_err();

        assert!(error.to_string().contains("delta"));
    }

    #[test]
    fn streamed_function_call_argument_delta_requires_output_index() {
        let text = r#"data: {"type":"response.function_call_arguments.delta","delta":"{}"}
data: {"type":"response.completed"}"#;

        let error = parse_assistant_turn(text).unwrap_err();

        assert!(error.to_string().contains("output_index"));
    }

    #[test]
    fn streamed_function_call_rejects_oversized_argument_delta() {
        let body = format!(
            "data: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"ferricode_read_file\",\"arguments\":\"\"}}}}\n\
data: {{\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{}\"}}\n\
data: {{\"type\":\"response.completed\"}}",
            "x".repeat((16 * 1024) + 1)
        );

        let error = parse_assistant_turn(&body).unwrap_err();

        assert!(error.to_string().contains("arguments"));
        assert!(error.to_string().contains("limit"));
    }

    #[test]
    fn streamed_function_call_rejects_huge_sparse_index() {
        let text = r#"data: {"type":"response.output_item.added","output_index":999999,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":"{}"}}
data: {"type":"response.completed"}"#;

        let error = parse_assistant_turn(text).unwrap_err();

        assert!(error.to_string().contains("output_index"));
    }

    #[test]
    fn tool_outputs_body_preserves_prior_items() {
        let state = OpenAiCodexState {
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
    }

    #[test]
    fn parser_ignores_unrelated_metadata_text() {
        let text = r#"{"metadata":{"text":"not assistant output"},"output":[]}"#;

        assert!(matches!(
            parse_assistant_text(text),
            Err(OpenAiCodexError::MissingAssistantText)
        ));
    }

    #[test]
    fn parser_reports_missing_assistant_text() {
        assert!(matches!(
            parse_assistant_text(r#"{"output":[]}"#),
            Err(OpenAiCodexError::MissingAssistantText)
        ));
        assert!(matches!(
            parse_assistant_text("data: [DONE]"),
            Err(OpenAiCodexError::MissingAssistantText)
        ));
    }

    #[test]
    fn malformed_sse_json_is_an_error() {
        assert!(matches!(
            parse_assistant_text("data: not-json"),
            Err(OpenAiCodexError::Json(_))
        ));
    }

    #[test]
    fn jwt_claim_parser_extracts_account_and_plan() {
        let claims = json!({
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_123",
                "chatgpt_plan_type": "plus"
            }
        });
        let jwt = format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD.encode(claims.to_string())
        );

        let parsed = parse_chatgpt_claims(&jwt).unwrap();

        assert_eq!(parsed.chatgpt_account_id.unwrap(), "acct_123");
        assert_eq!(parsed.chatgpt_plan_type.unwrap(), "plus");
    }

    #[test]
    fn refresh_decision_uses_injected_time() {
        let tokens = TokenSet {
            access_token: "access".to_string(),
            refresh_token: "refresh".to_string(),
            id_token: "id".to_string(),
            expires_at_unix_ms: 120_000,
            chatgpt_account_id: "acct".to_string(),
            chatgpt_plan_type: None,
        };

        assert!(token_needs_refresh(&tokens, 61_000));
        assert!(!token_needs_refresh(&tokens, 100));
    }

    #[test]
    fn token_expiry_overflow_is_an_error() {
        let response = TokenResponse {
            access_token: "access".to_string(),
            refresh_token: Some("refresh".to_string()),
            id_token: id_token("acct", "plus"),
            expires_in: u64::MAX,
        };

        assert!(matches!(
            tokens_from_response(response, None),
            Err(OpenAiCodexError::Protocol(_))
        ));
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
    async fn auth_loop_rejects_unrelated_request_then_completes_callback() {
        let id_token = id_token("acct_auth", "plus");
        let (base_url, requests) = spawn_test_server(vec![TestResponse::json(&token_json(
            "access", "refresh", &id_token, 3600,
        ))])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let output = SharedOutput::default();
        let captured_output = output.clone();

        let auth_task = tokio::spawn(async move {
            let mut output = output;
            authenticate_openai_codex_with_inputs(
                &path,
                &mut output,
                &base_url,
                false,
                Some(listener),
                Box::pin(std::future::pending()),
            )
            .await
        });
        let state = wait_for_state(&captured_output).await;

        let not_found = send_get(addr, "/favicon.ico").await;
        assert!(not_found.starts_with("HTTP/1.1 404 Not Found"));

        let ok = send_get(
            addr,
            &format!("/auth/callback?state={state}&code=auth-code"),
        )
        .await;
        assert!(ok.starts_with("HTTP/1.1 200 OK"));
        auth_task.await.unwrap().unwrap();

        assert_eq!(requests.lock().unwrap().len(), 1);
        assert!(
            String::from_utf8(captured_output.bytes())
                .unwrap()
                .contains("authentication complete")
        );
    }

    #[test]
    fn callback_port_conflict_falls_back_to_pasted_callback_url() {
        let mut output = Vec::new();

        let listener =
            callback_listener_or_paste_only(Err(OpenAiCodexError::CallbackPortInUse), &mut output)
                .unwrap();

        assert!(listener.is_none());
        assert!(
            String::from_utf8(output)
                .unwrap()
                .contains("continuing with pasted redirect URL only")
        );
    }

    #[tokio::test]
    async fn auth_without_listener_completes_from_pasted_callback_url() {
        let id_token = id_token("acct_auth", "plus");
        let (base_url, requests) = spawn_test_server(vec![TestResponse::json(&token_json(
            "access", "refresh", &id_token, 3600,
        ))])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let mut output = SharedOutput::default();
        let captured_output = output.clone();
        let pasted_callback = Box::pin(async move {
            let state = wait_for_state(&captured_output).await;
            Ok(Some(format!(
                "http://localhost:1455/auth/callback?state={state}&code=auth-code"
            )))
        });

        authenticate_openai_codex_with_inputs(
            &path,
            &mut output,
            &base_url,
            false,
            None,
            pasted_callback,
        )
        .await
        .unwrap();

        let auth = read_auth_file(&path).unwrap();
        let tokens = auth.openai_codex.unwrap().tokens.unwrap();
        assert_eq!(tokens.access_token, "access");
        assert_eq!(tokens.chatgpt_account_id, "acct_auth");
        assert_eq!(requests.lock().unwrap().len(), 1);
        assert!(
            String::from_utf8(output.bytes())
                .unwrap()
                .contains("paste the full broken URL here")
        );
    }

    #[tokio::test]
    async fn auth_without_listener_fails_when_pasted_callback_is_eof() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let mut output = SharedOutput::default();
        let pasted_callback = Box::pin(async { Ok(None) });

        let error = authenticate_openai_codex_with_inputs(
            &path,
            &mut output,
            "http://127.0.0.1:9",
            false,
            None,
            pasted_callback,
        )
        .await
        .unwrap_err();

        assert!(matches!(error, OpenAiCodexError::Protocol(_)));
        assert_eq!(
            error.to_string(),
            "failed to read pasted OpenAI Codex auth callback URL"
        );
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn auth_with_listener_completes_when_pasted_callback_wins() {
        let id_token = id_token("acct_auth", "plus");
        let (base_url, requests) = spawn_test_server(vec![TestResponse::json(&token_json(
            "access", "refresh", &id_token, 3600,
        ))])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut output = SharedOutput::default();
        let captured_output = output.clone();
        let pasted_callback = Box::pin(async move {
            let state = wait_for_state(&captured_output).await;
            Ok(Some(format!(
                "http://localhost:1455/auth/callback?state={state}&code=auth-code"
            )))
        });

        authenticate_openai_codex_with_inputs(
            &path,
            &mut output,
            &base_url,
            false,
            Some(listener),
            pasted_callback,
        )
        .await
        .unwrap();

        let auth = read_auth_file(&path).unwrap();
        let tokens = auth.openai_codex.unwrap().tokens.unwrap();
        assert_eq!(tokens.access_token, "access");
        assert_eq!(tokens.chatgpt_account_id, "acct_auth");
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn auth_with_listener_ignores_pasted_callback_eof() {
        let id_token = id_token("acct_auth", "plus");
        let (base_url, requests) = spawn_test_server(vec![TestResponse::json(&token_json(
            "access", "refresh", &id_token, 3600,
        ))])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let output = SharedOutput::default();
        let captured_output = output.clone();
        let pasted_callback = Box::pin(async { Ok(None) });

        let auth_task = tokio::spawn(async move {
            let mut output = output;
            authenticate_openai_codex_with_inputs(
                &path,
                &mut output,
                &base_url,
                false,
                Some(listener),
                pasted_callback,
            )
            .await
        });
        let state = wait_for_state(&captured_output).await;

        let ok = send_get(
            addr,
            &format!("/auth/callback?state={state}&code=auth-code"),
        )
        .await;
        assert!(ok.starts_with("HTTP/1.1 200 OK"));
        auth_task.await.unwrap().unwrap();

        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn auth_with_listener_finishes_accepted_callback_when_paste_finishes() {
        let id_token = id_token("acct_auth", "plus");
        let (base_url, requests) = spawn_test_server(vec![TestResponse::json(&token_json(
            "access", "refresh", &id_token, 3600,
        ))])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let output = SharedOutput::default();
        let captured_output = output.clone();
        let pasted_callback = Box::pin(async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Ok(None)
        });

        let auth_task = tokio::spawn(async move {
            let mut output = output;
            authenticate_openai_codex_with_inputs(
                &path,
                &mut output,
                &base_url,
                false,
                Some(listener),
                pasted_callback,
            )
            .await
        });
        let state = wait_for_state(&captured_output).await;
        let mut stream = TcpStream::connect(addr).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        let raw = format!(
            "GET /auth/callback?state={state}&code=auth-code HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\n\r\n"
        );
        stream.write_all(raw.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        assert!(
            String::from_utf8(response)
                .unwrap()
                .starts_with("HTTP/1.1 200 OK")
        );
        auth_task.await.unwrap().unwrap();

        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn valid_callback_exchanges_and_persists_openai_codex_tokens() {
        let id_token = id_token("acct_auth", "plus");
        let (base_url, requests) = spawn_test_server(vec![TestResponse::json(&token_json(
            "access", "refresh", &id_token, 3600,
        ))])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        let result = process_callback_target(
            &path,
            &base_url,
            "expected-state",
            &pkce,
            "/auth/callback?state=expected-state&code=auth-code",
            &reqwest::Client::new(),
        )
        .await
        .unwrap();

        assert_eq!(result, CallbackAction::Complete);
        let auth = read_auth_file(&path).unwrap();
        let tokens = auth.openai_codex.unwrap().tokens.unwrap();
        assert_eq!(tokens.access_token, "access");
        assert_eq!(tokens.refresh_token, "refresh");
        assert_eq!(tokens.chatgpt_account_id, "acct_auth");
        let requests = requests.lock().unwrap();
        assert!(requests[0].starts_with("POST /oauth/token "));
        assert!(requests[0].contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
        assert!(requests[0].contains("code=auth-code"));
        assert!(requests[0].contains("code_verifier=verifier"));
        assert!(
            requests[0].contains("redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback")
        );
        assert!(!requests[0].contains("client_secret"));
    }

    #[tokio::test]
    async fn pasted_callback_url_exchanges_and_persists_openai_codex_tokens() {
        let id_token = id_token("acct_auth", "plus");
        let (base_url, requests) = spawn_test_server(vec![TestResponse::json(&token_json(
            "access", "refresh", &id_token, 3600,
        ))])
        .await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        process_pasted_callback_url(
            &path,
            &base_url,
            "expected-state",
            &pkce,
            "  http://localhost:1455/auth/callback?state=expected-state&code=auth-code\n",
            &reqwest::Client::new(),
        )
        .await
        .unwrap();

        let auth = read_auth_file(&path).unwrap();
        let tokens = auth.openai_codex.unwrap().tokens.unwrap();
        assert_eq!(tokens.access_token, "access");
        assert_eq!(tokens.refresh_token, "refresh");
        assert_eq!(tokens.chatgpt_account_id, "acct_auth");
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
        assert!(requests[0].contains("code=auth-code"));
        assert!(requests[0].contains("code_verifier=verifier"));
        assert!(
            requests[0].contains("redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback")
        );
        assert!(!requests[0].contains("client_secret"));
    }

    #[tokio::test]
    async fn non_callback_request_is_ignored_without_network() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        let result = process_callback_target(
            &path,
            "http://127.0.0.1:9",
            "expected-state",
            &pkce,
            "/favicon.ico",
            &reqwest::Client::new(),
        )
        .await
        .unwrap();

        assert_eq!(result, CallbackAction::Continue);
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn non_callback_pasted_url_is_rejected_without_network() {
        let (base_url, requests) = spawn_test_server(Vec::new()).await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        let error = process_pasted_callback_url(
            &path,
            &base_url,
            "expected-state",
            &pkce,
            "http://localhost:1455/favicon.ico",
            &reqwest::Client::new(),
        )
        .await
        .unwrap_err();

        assert!(matches!(error, OpenAiCodexError::InvalidPastedCallbackUrl));
        assert!(!path.exists());
        assert!(requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn pasted_callback_rejects_wrong_origin_without_network() {
        let (base_url, requests) = spawn_test_server(Vec::new()).await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        for pasted_url in [
            "https://localhost:1455/auth/callback?state=expected-state&code=auth-code",
            "http://127.0.0.1:1455/auth/callback?state=expected-state&code=auth-code",
            "http://localhost:1456/auth/callback?state=expected-state&code=auth-code",
        ] {
            let error = process_pasted_callback_url(
                &path,
                &base_url,
                "expected-state",
                &pkce,
                pasted_url,
                &reqwest::Client::new(),
            )
            .await
            .unwrap_err();

            assert!(matches!(error, OpenAiCodexError::InvalidPastedCallbackUrl));
        }

        assert!(!path.exists());
        assert!(requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn state_mismatch_fails_without_persisting_tokens() {
        let (base_url, requests) = spawn_test_server(Vec::new()).await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        let error = process_callback_target(
            &path,
            &base_url,
            "expected-state",
            &pkce,
            "/auth/callback?state=wrong-state&code=auth-code",
            &reqwest::Client::new(),
        )
        .await
        .unwrap_err();

        assert!(matches!(error, OpenAiCodexError::StateMismatch));
        assert!(!path.exists());
        assert!(requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn pasted_callback_state_mismatch_fails_without_persisting_tokens() {
        let (base_url, requests) = spawn_test_server(Vec::new()).await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        let error = process_pasted_callback_url(
            &path,
            &base_url,
            "expected-state",
            &pkce,
            "http://localhost:1455/auth/callback?state=wrong-state&code=auth-code",
            &reqwest::Client::new(),
        )
        .await
        .unwrap_err();

        assert!(matches!(error, OpenAiCodexError::StateMismatch));
        assert!(!path.exists());
        assert!(requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn missing_authorization_code_fails_without_persisting_tokens() {
        let (base_url, requests) = spawn_test_server(Vec::new()).await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        let error = process_callback_target(
            &path,
            &base_url,
            "expected-state",
            &pkce,
            "/auth/callback?state=expected-state",
            &reqwest::Client::new(),
        )
        .await
        .unwrap_err();

        assert!(matches!(error, OpenAiCodexError::MissingAuthorizationCode));
        assert!(callback_error_is_recoverable(&error));
        assert!(!path.exists());
        assert!(requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn pasted_callback_missing_authorization_code_fails_without_persisting_tokens() {
        let (base_url, requests) = spawn_test_server(Vec::new()).await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        let error = process_pasted_callback_url(
            &path,
            &base_url,
            "expected-state",
            &pkce,
            "http://localhost:1455/auth/callback?state=expected-state",
            &reqwest::Client::new(),
        )
        .await
        .unwrap_err();

        assert!(matches!(error, OpenAiCodexError::MissingAuthorizationCode));
        assert!(callback_error_is_recoverable(&error));
        assert!(!path.exists());
        assert!(requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn oauth_error_callback_surfaces_error_without_persisting_tokens() {
        let (base_url, requests) = spawn_test_server(Vec::new()).await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        let error = process_callback_target(
            &path,
            &base_url,
            "expected-state",
            &pkce,
            "/auth/callback?state=expected-state&error=access_denied&error_description=nope",
            &reqwest::Client::new(),
        )
        .await
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "OpenAI Codex auth callback returned access_denied: nope"
        );
        assert!(!path.exists());
        assert!(requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn pasted_oauth_error_callback_surfaces_error_without_persisting_tokens() {
        let (base_url, requests) = spawn_test_server(Vec::new()).await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        let error = process_pasted_callback_url(
            &path,
            &base_url,
            "expected-state",
            &pkce,
            "http://localhost:1455/auth/callback?state=expected-state&error=access_denied&error_description=nope",
            &reqwest::Client::new(),
        )
        .await
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "OpenAI Codex auth callback returned access_denied: nope"
        );
        assert!(!path.exists());
        assert!(requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn token_exchange_failure_surfaces_token_status() {
        let (base_url, requests) =
            spawn_test_server(vec![TestResponse::new("400 Bad Request", "bad token")]).await;
        let dir = tempdir().unwrap();
        let path = dir.path().join("auth.toml");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_string(),
            code_challenge: "challenge".to_string(),
        };

        let error = process_callback_target(
            &path,
            &base_url,
            "expected-state",
            &pkce,
            "/auth/callback?state=expected-state&code=auth-code",
            &reqwest::Client::new(),
        )
        .await
        .unwrap_err();

        assert!(matches!(error, OpenAiCodexError::TokenStatus { .. }));
        assert!(!callback_error_is_recoverable(&error));
        assert!(!path.exists());
        assert_eq!(requests.lock().unwrap().len(), 1);
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
        assert!(requests[0].contains(r#""model":"gpt-5.4""#));
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

    #[derive(Clone)]
    struct TestResponse {
        status: &'static str,
        content_type: &'static str,
        body: String,
    }

    impl TestResponse {
        fn new(status: &'static str, body: &str) -> Self {
            Self {
                status,
                content_type: "text/plain",
                body: body.to_string(),
            }
        }

        fn json(body: &str) -> Self {
            Self {
                status: "200 OK",
                content_type: "application/json",
                body: body.to_string(),
            }
        }
    }

    #[derive(Clone, Default)]
    struct SharedOutput(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl SharedOutput {
        fn bytes(&self) -> Vec<u8> {
            self.0.lock().unwrap().clone()
        }
    }

    impl Write for SharedOutput {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    async fn wait_for_state(output: &SharedOutput) -> String {
        for _ in 0..100 {
            let text = String::from_utf8(output.bytes()).unwrap();
            if let Some(url) = text.lines().find(|line| line.starts_with("http")) {
                let parsed = url::Url::parse(url).unwrap();
                return parsed
                    .query_pairs()
                    .find_map(|(key, value)| (key == "state").then(|| value.into_owned()))
                    .unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("authorize URL was not written");
    }

    async fn spawn_test_server(
        responses: Vec<TestResponse>,
    ) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        use std::sync::{Arc, Mutex};

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);

        tokio::spawn(async move {
            for response in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let request = read_request(&mut stream).await;
                captured
                    .lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&request).to_string());
                write_response(&mut stream, response).await;
            }
        });

        (format!("http://{addr}"), requests)
    }

    async fn send_get(addr: std::net::SocketAddr, target: &str) -> String {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let raw = format!("GET {target} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\n\r\n");
        stream.write_all(raw.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    }

    async fn write_response(stream: &mut TcpStream, response: TestResponse) {
        let raw = format!(
            "HTTP/1.1 {}\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            response.status,
            response.content_type,
            response.body.len(),
            response.body
        );
        stream.write_all(raw.as_bytes()).await.unwrap();
    }

    async fn read_request(stream: &mut TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0; 1024];
        loop {
            let bytes = stream.read(&mut buffer).await.unwrap();
            if bytes == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..bytes]);
            if request_is_complete(&request) {
                break;
            }
        }
        request
    }

    fn request_is_complete(request: &[u8]) -> bool {
        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            return false;
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);

        request.len() >= header_end + 4 + content_length
    }

    fn auth_with_tokens(
        access_token: &str,
        refresh_token: &str,
        expires_at_unix_ms: u64,
    ) -> AuthFile {
        AuthFile {
            openai_codex: Some(OpenAiCodexAuth {
                tokens: Some(TokenSet {
                    access_token: access_token.to_string(),
                    refresh_token: refresh_token.to_string(),
                    id_token: id_token("acct", "plus"),
                    expires_at_unix_ms,
                    chatgpt_account_id: "acct".to_string(),
                    chatgpt_plan_type: Some("plus".to_string()),
                }),
            }),
        }
    }

    fn token_json(access: &str, refresh: &str, id_token: &str, expires_in: u64) -> String {
        json!({
            "access_token": access,
            "refresh_token": refresh,
            "id_token": id_token,
            "expires_in": expires_in
        })
        .to_string()
    }

    fn token_json_without_refresh(access: &str, id_token: &str, expires_in: u64) -> String {
        json!({
            "access_token": access,
            "id_token": id_token,
            "expires_in": expires_in
        })
        .to_string()
    }

    fn id_token(account_id: &str, plan: &str) -> String {
        let claims = json!({
            "https://api.openai.com/auth": {
                "chatgpt_account_id": account_id,
                "chatgpt_plan_type": plan
            }
        });
        format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD.encode(claims.to_string())
        )
    }
}
