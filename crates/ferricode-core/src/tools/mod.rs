//! Built-in tool interfaces and execution policy.
//!
//! Providers expose these tools to a model, but they do not implement them.
//! Keeping execution here gives every front end and provider the same local
//! filesystem policy, output shape, and failure behavior.

mod list_directory;
mod read_file;

use crate::ProviderRequest;
use serde_json::{Value, json};
use std::fs;
use std::path::{Component, Path, PathBuf};

const MAX_TOOL_CALLS_PER_TURN: usize = 16;
const MAX_TOOL_CALL_ID_BYTES: usize = 256;
const MAX_TOOL_NAME_BYTES: usize = 256;
const MAX_TOOL_ARGUMENT_BYTES: usize = 16 * 1024;

/// A provider-neutral request to run one built-in tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub(super) id: String,
    name: String,
    arguments: String,
}

impl ToolCall {
    /// Builds a tool call from provider-owned wire data.
    ///
    /// The harness treats arguments as JSON text so providers do not need to
    /// expose backend-specific argument-delta mechanics. Unknown tool names and
    /// invalid argument JSON are returned to the model as tool errors.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// Returns the provider's stable identifier for this call.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the provider-neutral tool name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the JSON argument object supplied by the model.
    pub fn arguments(&self) -> &str {
        &self.arguments
    }
}

/// The result of running one tool call.
///
/// The output is a JSON string because OpenAI Responses accepts function output
/// as text. The schema inside that string is still owned by the core tool
/// implementation so providers can pass it through without interpreting it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutput {
    call_id: String,
    output: String,
}

impl ToolOutput {
    /// Builds a tool output that can be matched to the provider's call id.
    pub fn new(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            output: output.into(),
        }
    }

    /// Returns the provider's stable identifier for the original call.
    pub fn call_id(&self) -> &str {
        &self.call_id
    }

    /// Returns the JSON string produced by the tool implementation.
    pub fn output(&self) -> &str {
        &self.output
    }
}

pub(crate) fn execute_tool_calls(
    request: &ProviderRequest,
    calls: Vec<ToolCall>,
) -> Vec<ToolOutput> {
    if calls.len() > MAX_TOOL_CALLS_PER_TURN {
        return calls
            .into_iter()
            .map(|call| {
                ToolOutput::new(
                    call.id,
                    tool_error_json(format!(
                        "model requested too many built-in tool calls in one turn; limit is {MAX_TOOL_CALLS_PER_TURN}"
                    )),
                )
            })
            .collect();
    }

    calls
        .into_iter()
        .map(|call| {
            let output = execute_tool_call(request, &call);
            ToolOutput::new(call.id, output)
        })
        .collect()
}

fn execute_tool_call(request: &ProviderRequest, call: &ToolCall) -> String {
    if call.id.len() > MAX_TOOL_CALL_ID_BYTES {
        return tool_error_json(format!(
            "tool call id exceeded the limit of {MAX_TOOL_CALL_ID_BYTES} bytes"
        ));
    }
    if call.name.len() > MAX_TOOL_NAME_BYTES {
        return tool_error_json(format!(
            "tool name exceeded the limit of {MAX_TOOL_NAME_BYTES} bytes"
        ));
    }
    if call.arguments.len() > MAX_TOOL_ARGUMENT_BYTES {
        return tool_error_json(format!(
            "tool arguments exceeded the limit of {MAX_TOOL_ARGUMENT_BYTES} bytes"
        ));
    }

    let output = match call.name.as_str() {
        "ferricode_list_directory" => list_directory::run(request, &call.arguments),
        "ferricode_read_file" => read_file::run(request, &call.arguments),
        name => Err(ToolError::new(format!("unknown built-in tool `{name}`"))),
    };

    match output {
        Ok(value) => value.to_string(),
        Err(error) => tool_error_json(error.message),
    }
}

fn tool_error_json(message: impl Into<String>) -> String {
    json!({
        "ok": false,
        "error": message.into(),
    })
    .to_string()
}

pub(super) fn parse_tool_path(arguments: &str) -> Result<PathBuf, ToolError> {
    let value: Value = serde_json::from_str(arguments)
        .map_err(|error| ToolError::new(format!("tool arguments must be valid JSON: {error}")))?;
    let path = value
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::new("tool arguments must include a string `path`"))?;
    let path = PathBuf::from(path);
    if path.as_os_str().is_empty() {
        return Err(ToolError::new("tool path must not be empty"));
    }
    if path.is_absolute() {
        return Err(ToolError::new(
            "tool path must be relative to the working directory",
        ));
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(ToolError::new(
            "tool path must not traverse outside the working directory",
        ));
    }
    Ok(path)
}

pub(super) fn resolve_tool_path(
    working_directory: &str,
    relative_path: &Path,
) -> Result<PathBuf, ToolError> {
    let root = fs::canonicalize(working_directory).map_err(|error| {
        ToolError::new(format!(
            "could not resolve working directory `{working_directory}`: {error}"
        ))
    })?;
    let resolved = fs::canonicalize(root.join(relative_path)).map_err(|error| {
        ToolError::new(format!(
            "could not resolve `{}`: {error}",
            relative_path.display()
        ))
    })?;
    if !resolved.starts_with(&root) {
        return Err(ToolError::new(
            "tool path resolved outside the working directory",
        ));
    }
    Ok(resolved)
}

pub(super) fn normalize_display_path(path: &Path) -> String {
    if path.as_os_str().is_empty() || path == Path::new(".") {
        ".".to_string()
    } else {
        path.display().to_string()
    }
}

#[derive(Debug)]
pub(super) struct ToolError {
    message: String,
}

impl ToolError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
