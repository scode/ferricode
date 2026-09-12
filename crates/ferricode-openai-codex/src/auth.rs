//! Runs the browser PKCE flow and validates the fixed loopback callback contract.
//!
//! This module crosses the OAuth and local-callback trust boundaries: it creates
//! state and verifier values, accepts only the expected callback shape, and turns
//! token responses into persisted credentials. Both the loopback HTTP callback
//! and a pasted redirect URL converge here before state checking and code
//! exchange. It is also the only place that talks to the token endpoint, for
//! the initial exchange and for refresh (`responses` decides when to refresh,
//! not how). It does not decide model requests or interpret Responses API
//! output.

use crate::{
    OpenAiCodexError, TokenSet,
    store::{random_urlsafe, read_auth_file, write_auth_file},
};
use base64::Engine;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::future::Future;
use std::io::{ErrorKind, Write};
use std::path::Path;
use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::timeout;

pub(crate) const DEFAULT_ISSUER: &str = "https://auth.openai.com";
const OAUTH_AUTHORIZE_PATH: &str = "/oauth/authorize";
const OAUTH_TOKEN_PATH: &str = "/oauth/token";
const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub(crate) const CODEX_ORIGINATOR: &str = "codex_cli_rs";
const CALLBACK_HOST: &str = "127.0.0.1";
const CALLBACK_PORT: u16 = 1455;
const CALLBACK_PATH: &str = "/auth/callback";
const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const CODEX_SCOPES: &str =
    "openid profile email offline_access api.connectors.read api.connectors.invoke";
const REFRESH_SKEW: Duration = Duration::from_secs(60);
const CALLBACK_READ_TIMEOUT: Duration = Duration::from_secs(10);

/// Generated PKCE values for one browser OAuth attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PkceCodes {
    pub code_verifier: String,
    pub code_challenge: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    id_token: String,
    expires_in: u64,
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

pub(crate) async fn refresh_access_token(
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

pub(crate) fn token_needs_refresh(tokens: &TokenSet, now_unix_ms: u64) -> bool {
    now_unix_ms.saturating_add(REFRESH_SKEW.as_millis() as u64) >= tokens.expires_at_unix_ms
}

pub(crate) fn tokens_from_response(
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

pub(crate) fn now_unix_ms() -> Result<u64, OpenAiCodexError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| OpenAiCodexError::Clock)?
        .as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use serde_json::json;
    use tempfile::tempdir;

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
}
