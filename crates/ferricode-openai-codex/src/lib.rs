//! OpenAI Codex-compatible provider support for Ferricode.
//!
//! This crate intentionally does not implement the OpenAI Platform API-key
//! flow. It uses the browser PKCE OAuth shape used by Codex-compatible CLIs and
//! stores Codex OAuth token state and account metadata.

use base64::Engine;
use ferricode_core::{ModelProvider, ProviderError, ProviderRequest};
use rand::RngCore;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
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
const MODEL: &str = "gpt-5.5";
const REASONING_EFFORT: &str = "medium";
const INSTRUCTIONS: &str =
    "You are Ferricode, a coding harness. Respond directly to the user's request.";
const REFRESH_SKEW: Duration = Duration::from_secs(60);
const CALLBACK_READ_TIMEOUT: Duration = Duration::from_secs(10);

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

/// Writes auth state while keeping newly created auth paths private where supported.
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
    output: &mut impl Write,
) -> Result<(), OpenAiCodexError> {
    let listener = bind_callback_listener().await?;
    authenticate_openai_codex_with_listener(path, output, DEFAULT_ISSUER, true, listener).await
}

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

async fn authenticate_openai_codex_with_listener(
    path: &Path,
    output: &mut impl Write,
    issuer: &str,
    open_browser: bool,
    listener: TcpListener,
) -> Result<(), OpenAiCodexError> {
    let pkce = generate_pkce();
    let state = generate_state();
    let authorize_url = build_authorize_url(issuer, CODEX_CLIENT_ID, REDIRECT_URI, &pkce, &state);

    if open_browser {
        let _ = webbrowser::open(&authorize_url);
    }
    writeln!(
        output,
        "Open this URL to sign in with OpenAI Codex auth:\n\n{}\n",
        authorize_url
    )?;

    loop {
        let (mut stream, _) = listener.accept().await?;
        let target = timeout(CALLBACK_READ_TIMEOUT, read_http_target(&mut stream))
            .await
            .unwrap_or(Err(OpenAiCodexError::InvalidCallbackRequest));
        let result = match target {
            Ok(target) => {
                process_callback_target(
                    path,
                    issuer,
                    &state,
                    &pkce,
                    &target,
                    &reqwest::Client::new(),
                )
                .await
            }
            Err(error) => Err(error),
        };

        match result {
            Ok(CallbackAction::Continue) => {
                write_http_response(&mut stream, 404, "Not Found").await?;
            }
            Ok(CallbackAction::Complete) => {
                write_http_response(&mut stream, 200, "OpenAI Codex authentication complete.")
                    .await?;
                writeln!(output, "OpenAI Codex authentication complete.")?;
                return Ok(());
            }
            Err(error) => {
                write_http_response(&mut stream, 400, &error.to_string()).await?;
                if callback_error_is_recoverable(&error) {
                    continue;
                }
                return Err(error);
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

    async fn authenticated_tokens(&self) -> Result<TokenSet, OpenAiCodexError> {
        let auth = read_auth_file(&self.auth_path)?;
        let mut tokens = auth
            .openai_codex
            .as_ref()
            .and_then(|auth| auth.tokens.clone())
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
    async fn respond<'a>(&'a self, request: &'a ProviderRequest) -> Result<String, ProviderError> {
        let tokens = self.authenticated_tokens().await?;
        let body = build_responses_body(request);
        let response = self
            .client
            .post(&self.backend_url)
            .headers(build_codex_headers(&tokens)?)
            .json(&body)
            .send()
            .await
            .map_err(OpenAiCodexError::from)?;

        let status = response.status();
        let text = response.text().await.map_err(OpenAiCodexError::from)?;
        if !status.is_success() {
            return Err(OpenAiCodexError::BackendStatus { status, body: text }.into());
        }

        parse_assistant_text(&text).map_err(ProviderError::from)
    }
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
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "reasoning": {
            "effort": REASONING_EFFORT
        },
        "store": false
    })
}

/// Parses either JSON or minimal SSE `data:` events into assistant text.
pub fn parse_assistant_text(text: &str) -> Result<String, OpenAiCodexError> {
    let trimmed = text.trim();
    if trimmed.lines().any(|line| line.starts_with("data:")) {
        return parse_sse_assistant_text(trimmed);
    }

    let value: Value = serde_json::from_str(trimmed)?;
    extract_text_from_response(&value)
        .filter(|value| !value.trim().is_empty())
        .ok_or(OpenAiCodexError::MissingAssistantText)
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

async fn process_callback_target(
    auth_path: &Path,
    issuer: &str,
    expected_state: &str,
    pkce: &PkceCodes,
    target: &str,
    client: &reqwest::Client,
) -> Result<CallbackAction, OpenAiCodexError> {
    let parsed = url::Url::parse(&format!("http://localhost{target}"))?;
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

fn callback_error_is_recoverable(error: &OpenAiCodexError) -> bool {
    matches!(
        error,
        OpenAiCodexError::InvalidCallbackRequest
            | OpenAiCodexError::StateMismatch
            | OpenAiCodexError::MissingAuthorizationCode
            | OpenAiCodexError::Url(_)
    )
}

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

fn parse_sse_assistant_text(text: &str) -> Result<String, OpenAiCodexError> {
    let mut joined = String::new();
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" || data.is_empty() {
            continue;
        }
        let value = serde_json::from_str::<Value>(data)?;
        if let Some(text) = extract_text_from_event(&value) {
            joined.push_str(&text);
        }
    }

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
    match value {
        Value::Object(map) => {
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
        _ => None,
    }
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
    Ok(fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?)
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
    use ferricode_core::{ModelProvider, ProviderRequest};
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
        let request = ProviderRequest::new("fix it", "/repo");

        let body = build_responses_body(&request);

        assert_eq!(body["model"], MODEL);
        assert_eq!(body["instructions"], INSTRUCTIONS);
        assert_eq!(body["stream"], true);
        assert_eq!(body["tools"], json!([]));
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["reasoning"]["effort"], REASONING_EFFORT);
        assert_eq!(body["store"], false);
        assert!(body.to_string().contains("/repo"));
        assert!(body.to_string().contains("fix it"));
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
            authenticate_openai_codex_with_listener(&path, &mut output, &base_url, false, listener)
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
        assert!(requests[0].contains(r#""model":"gpt-5.5""#));
        assert!(requests[0].contains(r#""effort":"medium""#));
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
