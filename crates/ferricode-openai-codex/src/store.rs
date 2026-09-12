//! Persists Codex OAuth state without relying on the process umask: new files
//! are 0600 and new directories 0700, and non-Unix targets are refused.
//!
//! This module is the filesystem trust boundary for bearer and refresh tokens. It
//! accepts an explicit path from its caller, refuses to write through a symlinked
//! auth path, and creates new Unix paths with restrictive permissions; it does
//! not discover or repair credentials from elsewhere. Reads follow symlinks like
//! any other file read.

use crate::OpenAiCodexError;
use base64::Engine;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

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

/// Produces `len` random bytes as unpadded base64url text.
///
/// Lives here rather than in `auth` because the token store needs it for temp
/// file names and must not depend on the OAuth module; `auth` uses it for PKCE
/// verifiers and state values.
pub(crate) fn random_urlsafe(len: usize) -> String {
    let mut bytes = vec![0; len];
    rand::rng().fill_bytes(&mut bytes);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
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
    use crate::test_support::*;
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
}
