use super::{ToolError, normalize_display_path, parse_tool_path, resolve_tool_path};
use crate::ProviderRequest;
use serde_json::{Value, json};
use std::fs;
use std::io::Read;

const MAX_FILE_READ_BYTES: usize = 64 * 1024;

/// Registry entry for the file-reading tool.
///
/// `name` is the wire-facing identifier documented in `docs/tools.md`;
/// renaming it is a model-integration change, and `run` below is reachable
/// only through it. The schema is what the model is told about the arguments,
/// not what is enforced: `parse_tool_path` still validates the path itself.
pub(super) const DEFINITION: super::ToolDefinition = super::ToolDefinition {
    name: "ferricode_read_file",
    description: "Read one UTF-8 text file under the request working directory.",
    parameters_schema: super::PATH_ARGUMENTS_SCHEMA,
    run,
};

pub(super) fn run(request: &ProviderRequest, arguments: &str) -> Result<Value, ToolError> {
    let path = parse_tool_path(arguments)?;
    let resolved = resolve_tool_path(request.working_directory(), &path)?;
    let metadata = fs::metadata(&resolved).map_err(|error| {
        ToolError::new(format!("could not inspect `{}`: {error}", path.display()))
    })?;
    if !metadata.is_file() {
        return Err(ToolError::new(format!(
            "`{}` is not a regular file",
            path.display()
        )));
    }

    let mut bytes = Vec::with_capacity(MAX_FILE_READ_BYTES + 4);
    fs::File::open(&resolved)
        .map_err(|error| ToolError::new(format!("could not read `{}`: {error}", path.display())))?
        .take((MAX_FILE_READ_BYTES + 4) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| ToolError::new(format!("could not read `{}`: {error}", path.display())))?;

    let truncated = bytes.len() > MAX_FILE_READ_BYTES;
    let content_len = if truncated {
        utf8_boundary_at_or_before(&bytes, MAX_FILE_READ_BYTES)
    } else {
        bytes.len()
    };
    if bytes[..content_len].contains(&0) {
        return Err(ToolError::new(format!(
            "`{}` appears to be binary data",
            path.display()
        )));
    }
    let content = std::str::from_utf8(&bytes[..content_len])
        .map_err(|_| ToolError::new(format!("`{}` is not valid UTF-8 text", path.display())))?;

    Ok(json!({
        "ok": true,
        "path": normalize_display_path(&path),
        "content": content,
        "truncated": truncated,
    }))
}

fn utf8_boundary_at_or_before(bytes: &[u8], limit: usize) -> usize {
    let mut index = limit.min(bytes.len());
    while index > 0 && index < bytes.len() && (bytes[index] & 0b1100_0000) == 0b1000_0000 {
        index -= 1;
    }
    index
}
