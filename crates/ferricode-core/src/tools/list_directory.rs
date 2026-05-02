use super::{ToolError, normalize_display_path, parse_tool_path, resolve_tool_path};
use crate::ProviderRequest;
use serde_json::{Value, json};
use std::fs;

const MAX_DIRECTORY_ENTRIES: usize = 200;

pub(super) fn run(request: &ProviderRequest, arguments: &str) -> Result<Value, ToolError> {
    let path = parse_tool_path(arguments)?;
    let resolved = resolve_tool_path(request.working_directory(), &path)?;
    let metadata = fs::metadata(&resolved).map_err(|error| {
        ToolError::new(format!("could not inspect `{}`: {error}", path.display()))
    })?;
    if !metadata.is_dir() {
        return Err(ToolError::new(format!(
            "`{}` is not a directory",
            path.display()
        )));
    }

    let mut entries = Vec::new();
    for entry in fs::read_dir(&resolved)
        .map_err(|error| ToolError::new(format!("could not list `{}`: {error}", path.display())))?
    {
        let entry = entry.map_err(|error| {
            ToolError::new(format!(
                "could not read an entry in `{}`: {error}",
                path.display()
            ))
        })?;
        let metadata = fs::symlink_metadata(entry.path()).map_err(|error| {
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

fn file_kind(file_type: &fs::FileType) -> &'static str {
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
