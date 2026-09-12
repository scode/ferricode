//! OpenAI Codex-compatible provider support for Ferricode.
//!
//! This crate intentionally does not implement the OpenAI Platform API-key
//! flow. It uses the browser PKCE OAuth shape used by Codex-compatible CLIs and
//! stores Codex OAuth token state and account metadata.
//!
//! Module map: `auth` obtains tokens, `store` persists them, `responses` spends
//! them against the Codex backend, and `sse` parses what comes back. The error
//! enum below is shared by all four and is the only thing defined in this file.

mod auth;
mod responses;
mod sse;
mod store;

#[cfg(test)]
pub(crate) mod test_support;

pub use auth::{PkceCodes, authenticate_openai_codex, build_authorize_url};
pub use responses::{OpenAiCodexProvider, OpenAiCodexState, build_responses_body};
pub use sse::parse_assistant_text;
pub use store::{
    AuthFile, OpenAiCodexAuth, TokenSet, default_auth_path, read_auth_file, write_auth_file,
};

use ferricode_core::ProviderError;
use thiserror::Error;

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
