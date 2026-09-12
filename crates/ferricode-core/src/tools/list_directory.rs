use super::{ToolError, normalize_display_path, parse_tool_path, resolve_tool_path};
use crate::ProviderRequest;
use serde_json::{Value, json};

const MAX_DIRECTORY_ENTRIES: usize = 200;

/// Registry entry for the directory-listing tool.
///
/// `name` is the wire-facing identifier documented in `docs/tools.md`;
/// renaming it is a model-integration change, and `run` below is reachable
/// only through it. The schema is what the model is told about the arguments,
/// not what is enforced: `parse_tool_path` still validates the path itself.
pub(super) const DEFINITION: super::ToolDefinition = super::ToolDefinition {
    name: "ferricode_list_directory",
    description: "List one directory under the request working directory.",
    parameters_schema: super::PATH_ARGUMENTS_SCHEMA,
    run: run_boxed,
};

/// Adapts `run` to the registry's function-pointer signature by boxing its
/// future; see `RunFn` in the parent module for why the box is needed.
fn run_boxed<'a>(request: &'a ProviderRequest, arguments: &'a str) -> super::ToolFuture<'a> {
    Box::pin(run(request, arguments))
}

/// Lists one validated directory without blocking the async executor.
pub(super) async fn run(request: &ProviderRequest, arguments: &str) -> Result<Value, ToolError> {
    let path = parse_tool_path(arguments)?;
    let resolved = resolve_tool_path(request.working_directory(), &path).await?;
    let metadata = tokio::fs::metadata(&resolved).await.map_err(|error| {
        ToolError::new(format!("could not inspect `{}`: {error}", path.display()))
    })?;
    if !metadata.is_dir() {
        return Err(ToolError::new(format!(
            "`{}` is not a directory",
            path.display()
        )));
    }

    let mut entries = Vec::new();
    let mut directory = tokio::fs::read_dir(&resolved)
        .await
        .map_err(|error| ToolError::new(format!("could not list `{}`: {error}", path.display())))?;
    while let Some(entry) = directory.next_entry().await.map_err(|error| {
        ToolError::new(format!(
            "could not read an entry in `{}`: {error}",
            path.display()
        ))
    })? {
        let metadata = tokio::fs::symlink_metadata(entry.path())
            .await
            .map_err(|error| {
                ToolError::new(format!(
                    "could not inspect `{}`: {error}",
                    entry.file_name().to_string_lossy()
                ))
            })?;
        entries.push(DirectoryEntry {
            name: entry.file_name().to_string_lossy().into_owned(),
            kind: file_kind(&metadata.file_type()),
            size: metadata.is_file().then_some(metadata.len()),
        });
    }

    entries.sort_by(|left, right| left.name.cmp(&right.name));
    let truncated = entries.len() > MAX_DIRECTORY_ENTRIES;
    entries.truncate(MAX_DIRECTORY_ENTRIES);

    Ok(json!({
        "ok": true,
        "path": normalize_display_path(&path),
        "entries": entries.into_iter().map(|entry| entry.to_json()).collect::<Vec<_>>(),
        "truncated": truncated,
    }))
}

/// Maps a file type to the fixed vocabulary `docs/tools.md` promises the model.
///
/// Tokio's metadata APIs hand back the std `FileType`, which is why a std type
/// appears in an otherwise Tokio-only file. It is a plain value, so this does
/// no I/O; the symlink metadata call that produced it is the only read.
fn file_kind(file_type: &std::fs::FileType) -> &'static str {
    if file_type.is_dir() {
        "directory"
    } else if file_type.is_file() {
        "file"
    } else if file_type.is_symlink() {
        "symlink"
    } else {
        "other"
    }
}

#[derive(Debug)]
struct DirectoryEntry {
    name: String,
    kind: &'static str,
    size: Option<u64>,
}

impl DirectoryEntry {
    fn to_json(&self) -> Value {
        let mut value = json!({
            "name": self.name,
            "type": self.kind,
        });
        if let Some(size) = self.size {
            value["size"] = json!(size);
        }
        value
    }
}
