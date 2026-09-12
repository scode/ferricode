//! Built-in tool definitions, interfaces, and execution policy.
//!
//! Providers expose these tools to a model, but they do not implement them.
//! Keeping execution here gives every front end and provider the same local
//! filesystem policy, output shape, and failure behavior.
//! This module is the single source of truth for the definitions that providers
//! translate into their wire formats.
//!
//! Tool execution is asynchronous so slow or blocking local work never stalls
//! the executor. Calls within one turn run sequentially, in input order, and
//! implementations must not perform blocking work such as synchronous
//! standard-library filesystem access or `std::thread::sleep`. One
//! consequence future tool authors need to know: `tokio::fs` runs each
//! operation on the blocking pool, and dropping a tool future (the harness
//! call being cancelled) does not cancel an in-flight operation. Harmless for
//! the read-only tools here; a write or shell tool must not assume that
//! dropping its future stops the side effect.

mod list_directory;
mod read_file;

use crate::ProviderRequest;
use serde_json::{Value, json};
use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::pin::Pin;

/// Argument schema shared by the filesystem tools: one relative `path` string
/// and nothing else.
///
/// `additionalProperties: false` and listing every property under `required`
/// are not stylistic: the OpenAI Codex provider sends these schemas with
/// `strict: true`, and OpenAI strict function calling rejects schemas that
/// leave either out. Any new tool schema has to keep both.
pub(super) const PATH_ARGUMENTS_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "A relative path under the request working directory."
    }
  },
  "required": ["path"],
  "additionalProperties": false
}"#;

/// The in-flight execution of one tool call.
///
/// Boxed because the registry stores plain function pointers, and a function
/// pointer cannot name the anonymous future type an `async fn` returns. `Send`
/// so the harness future stays `Send` for multi-threaded executors.
pub(super) type ToolFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Value, ToolError>> + Send + 'a>>;

/// The entry point of one built-in tool: JSON argument text in, a future
/// resolving to tool JSON or a model-facing error out. Each tool module wraps
/// its `async fn run` in a small adapter with this signature.
type RunFn = for<'a> fn(&'a ProviderRequest, &'a str) -> ToolFuture<'a>;

/// Provider-neutral description of one built-in tool: what it is called, what
/// the model should read to decide when to use it, the JSON Schema of its
/// arguments, and the function that runs it.
///
/// This struct is the whole registry. `built_in_tools()` is the only list of
/// tools in the codebase, and dispatch in `execute_tool_call` walks it, so a
/// tool that is published is always runnable and a tool that is runnable is
/// always published. The `run` field is private so providers see only the
/// wire-facing half.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    name: &'static str,
    description: &'static str,
    parameters_schema: &'static str,
    run: RunFn,
}

impl ToolDefinition {
    /// Returns the stable provider-facing name of this built-in tool.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the model-facing guidance for when this tool should be used.
    pub fn description(&self) -> &'static str {
        self.description
    }

    /// Parses this built-in tool's JSON Schema object.
    ///
    /// The schema is a core constant, so malformed text indicates a programming
    /// error rather than caller input. Tests cover the constants to keep this
    /// provider-facing contract intact.
    pub fn parameters_schema(&self) -> Value {
        serde_json::from_str(self.parameters_schema)
            .expect("built-in tool parameter schema must be valid JSON")
    }
}

/// Returns the built-in tools core will execute, in stable provider-facing order.
pub fn built_in_tools() -> &'static [ToolDefinition] {
    &[list_directory::DEFINITION, read_file::DEFINITION]
}

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

/// Executes one turn's calls one at a time, in input order.
///
/// Sequential execution is a deliberate choice, not a constraint of the
/// provider protocol: outputs are matched to calls by id, so running calls
/// concurrently would be protocol-safe. It is kept sequential so the async
/// conversion only moves I/O off the executor thread and leaves observable
/// behavior, including the order in which side effects of future tools
/// happen, as it was. Revisit if a provider ever issues several independent
/// calls per turn.
pub(crate) async fn execute_tool_calls(
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

    let mut outputs = Vec::with_capacity(calls.len());
    for call in calls {
        let output = execute_tool_call(request, &call).await;
        outputs.push(ToolOutput::new(call.id, output));
    }
    outputs
}

/// Validates and executes one call, converting every tool failure to model JSON.
///
/// The limit checks run before dispatch, so an oversized or unknown call never
/// reaches a tool, and every failure path returns the same
/// `{"ok": false, "error": ...}` shape so the model sees one vocabulary.
async fn execute_tool_call(request: &ProviderRequest, call: &ToolCall) -> String {
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

    let output = match built_in_tools()
        .iter()
        .find(|definition| definition.name == call.name)
    {
        Some(definition) => (definition.run)(request, &call.arguments).await,
        None => Err(ToolError::new(format!(
            "unknown built-in tool `{}`",
            call.name
        ))),
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

/// Resolves a relative tool path and rejects targets outside the working directory.
///
/// `parse_tool_path` only checks the lexical form. A symlink under the root
/// can still point anywhere, so both the root and the joined path are
/// canonicalized (following symlinks) and the result must still sit under the
/// root. The path must exist: canonicalization of a missing target fails, and
/// that failure is reported to the model as a tool error.
pub(super) async fn resolve_tool_path(
    working_directory: &str,
    relative_path: &Path,
) -> Result<PathBuf, ToolError> {
    let root = tokio::fs::canonicalize(working_directory)
        .await
        .map_err(|error| {
            ToolError::new(format!(
                "could not resolve working directory `{working_directory}`: {error}"
            ))
        })?;
    let resolved = tokio::fs::canonicalize(root.join(relative_path))
        .await
        .map_err(|error| {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every published schema must be an object with `additionalProperties`
    /// false and every property listed as required, because the OpenAI Codex
    /// provider sends them under `strict: true` and the backend rejects
    /// anything looser. This is the table-wide invariant; the per-tool shape
    /// is checked separately below.
    #[test]
    fn built_in_schemas_satisfy_strict_function_calling() {
        for definition in built_in_tools() {
            let schema = definition.parameters_schema();
            let properties = schema["properties"].as_object().unwrap();
            let required = schema["required"].as_array().unwrap();

            assert_eq!(schema["type"], "object", "{}", definition.name());
            assert_eq!(
                schema["additionalProperties"],
                false,
                "{}",
                definition.name()
            );
            assert_eq!(required.len(), properties.len(), "{}", definition.name());
            for property in properties.keys() {
                assert!(
                    required.iter().any(|value| value == property),
                    "{}: `{property}` is not required",
                    definition.name()
                );
            }
        }
    }

    /// The two filesystem tools take exactly one relative `path` string; this
    /// pins that shape per tool so a later tool with different arguments does
    /// not have to weaken a table-wide test.
    #[test]
    fn filesystem_tools_take_a_single_path_argument() {
        for definition in [&list_directory::DEFINITION, &read_file::DEFINITION] {
            let schema = definition.parameters_schema();

            assert_eq!(schema["required"], json!(["path"]), "{}", definition.name());
            assert_eq!(
                schema["properties"]["path"]["type"],
                "string",
                "{}",
                definition.name()
            );
        }
    }
}
