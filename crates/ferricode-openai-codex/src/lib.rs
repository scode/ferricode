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

use ferricode_core::{ProviderError, ProviderErrorKind};
use reqwest::StatusCode;
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
    /// Maps provider failures onto the coarse classes front ends act on.
    ///
    /// Two arms are not obvious from the variant names. A token-endpoint 400
    /// or 401 is `AuthRequired`, not `BackendStatus`: OAuth returns
    /// `invalid_grant` for an expired or revoked refresh token as HTTP 400
    /// (RFC 6749 section 5.2), and in either case only a fresh sign-in can
    /// help. This fires for the initial code exchange too, which is harmless.
    /// `reqwest` errors are split on whether the request ever completed:
    /// connect, timeout, and request-building failures are `Network`
    /// (retryable); anything after a response arrived (decode, body) is
    /// `Protocol`.
    fn from(value: OpenAiCodexError) -> Self {
        let kind = match &value {
            OpenAiCodexError::MissingTokens | OpenAiCodexError::MissingAccountId => {
                ProviderErrorKind::AuthRequired
            }
            OpenAiCodexError::Http(error)
                if error.is_connect() || error.is_timeout() || error.is_request() =>
            {
                ProviderErrorKind::Network
            }
            OpenAiCodexError::Http(_) => ProviderErrorKind::Protocol,
            OpenAiCodexError::TokenStatus { status, .. }
                if *status == StatusCode::UNAUTHORIZED || *status == StatusCode::BAD_REQUEST =>
            {
                ProviderErrorKind::AuthRequired
            }
            OpenAiCodexError::BackendStatus { .. } | OpenAiCodexError::TokenStatus { .. } => {
                ProviderErrorKind::BackendStatus
            }
            OpenAiCodexError::MissingAssistantText
            | OpenAiCodexError::Protocol(_)
            | OpenAiCodexError::Json(_) => ProviderErrorKind::Protocol,
            OpenAiCodexError::MissingHome
            | OpenAiCodexError::CallbackPortInUse
            | OpenAiCodexError::InvalidCallbackRequest
            | OpenAiCodexError::InvalidPastedCallbackUrl
            | OpenAiCodexError::StateMismatch
            | OpenAiCodexError::OAuthCallbackError { .. }
            | OpenAiCodexError::MissingAuthorizationCode
            | OpenAiCodexError::Clock
            | OpenAiCodexError::Io(_)
            | OpenAiCodexError::TomlDe(_)
            | OpenAiCodexError::TomlSer(_)
            | OpenAiCodexError::Url(_) => ProviderErrorKind::Other,
        };

        Self::new(kind, value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{OpenAiCodexError, ProviderErrorKind};
    use reqwest::StatusCode;

    /// Missing stored tokens must tell front ends to start the authentication
    /// flow instead of treating the failure as a retryable backend problem.
    #[test]
    fn missing_tokens_require_authentication() {
        let error = ferricode_core::ProviderError::from(OpenAiCodexError::MissingTokens);

        assert_eq!(error.kind(), ProviderErrorKind::AuthRequired);
    }

    /// The token endpoint reports an expired or revoked refresh token as HTTP
    /// 400 (`invalid_grant`) and a bad credential as 401; both mean only a
    /// fresh sign-in can help, so both are `AuthRequired` rather than
    /// `BackendStatus`.
    #[test]
    fn token_endpoint_400_and_401_require_authentication() {
        for status in [StatusCode::BAD_REQUEST, StatusCode::UNAUTHORIZED] {
            let error = ferricode_core::ProviderError::from(OpenAiCodexError::TokenStatus {
                status,
                body: "rejected".to_string(),
            });

            assert_eq!(error.kind(), ProviderErrorKind::AuthRequired, "{status}");
        }
    }

    /// Any other token-endpoint status is a backend problem, not a credential
    /// problem, so it must not send the user back through sign-in.
    #[test]
    fn token_endpoint_500_is_backend_status() {
        let error = ferricode_core::ProviderError::from(OpenAiCodexError::TokenStatus {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: "down".to_string(),
        });

        assert_eq!(error.kind(), ProviderErrorKind::BackendStatus);
    }

    /// A 401 from the responses backend stays `BackendStatus`. This asymmetry
    /// with the token endpoint is deliberate: the credential was accepted at
    /// sign-in, so the failure is about this request or this account's access,
    /// and re-authenticating is not known to fix it.
    #[test]
    fn backend_401_is_backend_status_not_auth() {
        let error = ferricode_core::ProviderError::from(OpenAiCodexError::BackendStatus {
            status: StatusCode::UNAUTHORIZED,
            body: "no access".to_string(),
        });

        assert_eq!(error.kind(), ProviderErrorKind::BackendStatus);
    }
}
