//! Parses Responses API JSON and SSE output into assistant turns.
//!
//! This module treats backend event data as untrusted protocol input. While
//! reassembling function calls it caps the streamed output index and the
//! function-call id, name, and argument sizes (see the `MAX_*` constants) so a
//! hostile stream cannot allocate unboundedly. It parses transport output only:
//! it never executes calls, selects tools, or decides provider policy.

use crate::{OpenAiCodexError, OpenAiCodexState};
use ferricode_core::{ProviderTurn, ToolCall};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::str;

const MAX_STREAMING_OUTPUT_INDEX: usize = 1024;
const MAX_FUNCTION_CALL_ID_BYTES: usize = 256;
const MAX_FUNCTION_CALL_NAME_BYTES: usize = 256;
const MAX_FUNCTION_CALL_ARGUMENT_BYTES: usize = 16 * 1024;

/// Parses either JSON or minimal SSE `data:` events into assistant text.
pub fn parse_assistant_text(text: &str) -> Result<String, OpenAiCodexError> {
    match parse_assistant_turn(text)? {
        ProviderTurn::Final(text) => Ok(text),
        ProviderTurn::ToolCalls { .. } => Err(OpenAiCodexError::MissingAssistantText),
    }
}

pub(crate) fn parse_assistant_turn(
    text: &str,
) -> Result<ProviderTurn<OpenAiCodexState>, OpenAiCodexError> {
    let trimmed = text.trim();
    if trimmed.lines().any(|line| line.starts_with("data:")) {
        return parse_sse_assistant_text(trimmed);
    }

    let value: Value = serde_json::from_str(trimmed)?;
    parse_response_turn(&value)
}

fn parse_sse_assistant_text(
    text: &str,
) -> Result<ProviderTurn<OpenAiCodexState>, OpenAiCodexError> {
    let mut accumulator = SseAccumulator::default();
    for line in text.lines() {
        if accumulator.process_line(line.as_bytes())? == SseDataAction::Complete {
            break;
        }
    }

    accumulator.into_provider_turn()
}

pub(crate) async fn parse_sse_assistant_stream(
    response: &mut reqwest::Response,
) -> Result<ProviderTurn<OpenAiCodexState>, OpenAiCodexError> {
    let mut pending = Vec::new();
    let mut accumulator = SseAccumulator::default();

    while let Some(chunk) = response.chunk().await? {
        pending.extend_from_slice(&chunk);
        while let Some(line_end) = pending.iter().position(|byte| *byte == b'\n') {
            let line = pending.drain(..=line_end).collect::<Vec<_>>();
            if accumulator.process_line(&line)? == SseDataAction::Complete {
                return accumulator.into_provider_turn();
            }
        }
    }

    if !pending.is_empty() {
        accumulator.process_line(&pending)?;
    }
    accumulator.into_provider_turn()
}

fn parse_response_turn(value: &Value) -> Result<ProviderTurn<OpenAiCodexState>, OpenAiCodexError> {
    let output_items = value
        .get("output")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let calls = collect_function_calls(output_items.iter())?;
    if !calls.is_empty() {
        return Ok(ProviderTurn::ToolCalls {
            state: OpenAiCodexState {
                input_items: Vec::new(),
                output_items,
            },
            calls,
        });
    }

    let text = extract_text_from_response(value)
        .filter(|value| !value.trim().is_empty())
        .ok_or(OpenAiCodexError::MissingAssistantText)?;
    Ok(ProviderTurn::Final(text))
}

#[derive(Default)]
struct SseAccumulator {
    text: String,
    output_items: Vec<Value>,
    function_calls: BTreeMap<usize, StreamingFunctionCall>,
}

impl SseAccumulator {
    fn process_line(&mut self, line: &[u8]) -> Result<SseDataAction, OpenAiCodexError> {
        let line = parse_sse_line(line)?;
        let Some(data) = line.strip_prefix("data:") else {
            return Ok(SseDataAction::Continue);
        };
        self.process_data_line(data)
    }

    fn process_data_line(&mut self, data: &str) -> Result<SseDataAction, OpenAiCodexError> {
        let data = data.trim();
        if data.is_empty() {
            return Ok(SseDataAction::Continue);
        }
        if data == "[DONE]" {
            return Ok(SseDataAction::Complete);
        }

        let value = serde_json::from_str::<Value>(data)?;
        if let Some(text) = extract_text_from_event(&value) {
            self.text.push_str(&text);
        }

        match value.get("type").and_then(Value::as_str) {
            Some("response.output_item.added") => {
                let index = required_event_output_index(&value)?;
                let item = required_event_value(&value, "item")?;
                if is_function_call_item(item) {
                    self.function_calls
                        .entry(index)
                        .or_default()
                        .merge_item(item)?;
                }
            }
            Some("response.function_call_arguments.delta") => {
                let index = required_event_output_index(&value)?;
                let delta = required_event_string(&value, "delta")?;
                let call = self.function_calls.entry(index).or_default();
                if call.arguments.len() + delta.len() > MAX_FUNCTION_CALL_ARGUMENT_BYTES {
                    return Err(OpenAiCodexError::Protocol(format!(
                        "streamed function call arguments exceeded the limit of {MAX_FUNCTION_CALL_ARGUMENT_BYTES} bytes"
                    )));
                }
                call.arguments.push_str(delta);
            }
            Some("response.function_call_arguments.done") => {
                let index = required_event_output_index(&value)?;
                let arguments = required_event_string(&value, "arguments")?;
                validate_function_call_field("arguments", arguments)?;
                self.function_calls.entry(index).or_default().arguments = arguments.to_string();
            }
            Some("response.output_item.done") => {
                let item = required_event_value(&value, "item")?;
                if is_function_call_item(item) {
                    let index = required_event_output_index(&value)?;
                    self.function_calls
                        .entry(index)
                        .or_default()
                        .merge_item(item)?;
                }
                self.output_items.push(item.clone());
            }
            Some("response.completed") => return Ok(SseDataAction::Complete),
            Some("response.failed" | "response.incomplete") => {
                return Err(OpenAiCodexError::Protocol(
                    "OpenAI Codex backend ended the response without completing it".to_string(),
                ));
            }
            _ => {}
        }

        Ok(SseDataAction::Continue)
    }

    fn into_provider_turn(mut self) -> Result<ProviderTurn<OpenAiCodexState>, OpenAiCodexError> {
        if !self.function_calls.is_empty() {
            self.merge_streaming_function_items()?;
            let calls = collect_streaming_function_calls(&self.function_calls);
            return Ok(ProviderTurn::ToolCalls {
                state: OpenAiCodexState {
                    input_items: Vec::new(),
                    output_items: self.output_items,
                },
                calls,
            });
        }

        completed_sse_text(self.text).map(ProviderTurn::Final)
    }

    fn merge_streaming_function_items(&mut self) -> Result<(), OpenAiCodexError> {
        for (index, call) in &self.function_calls {
            if *index >= MAX_STREAMING_OUTPUT_INDEX {
                return Err(OpenAiCodexError::Protocol(format!(
                    "streamed function call output_index {index} exceeded the limit of {MAX_STREAMING_OUTPUT_INDEX}"
                )));
            }
            let item = call.to_item()?;
            if self.output_items.len() <= *index {
                self.output_items.resize(*index + 1, Value::Null);
            }
            self.output_items[*index] = item;
        }
        self.output_items.retain(|item| !item.is_null());
        Ok(())
    }
}

fn parse_sse_line(line: &[u8]) -> Result<&str, OpenAiCodexError> {
    let line = str::from_utf8(line)
        .map_err(|_| OpenAiCodexError::Protocol("SSE response was not valid UTF-8".to_string()))?
        .trim_end_matches(['\r', '\n']);
    Ok(line)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SseDataAction {
    Continue,
    Complete,
}

fn completed_sse_text(joined: String) -> Result<String, OpenAiCodexError> {
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
    let map = value.as_object()?;
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

fn collect_function_calls<'a>(
    items: impl Iterator<Item = &'a Value>,
) -> Result<Vec<ToolCall>, OpenAiCodexError> {
    items
        .filter_map(function_call_from_item)
        .collect::<Result<Vec<_>, _>>()
}

fn function_call_from_item(item: &Value) -> Option<Result<ToolCall, OpenAiCodexError>> {
    if !is_function_call_item(item) {
        return None;
    }
    Some((|| {
        Ok(ToolCall::new(
            required_function_call_string(item, "call_id")?,
            required_function_call_string(item, "name")?,
            required_function_call_string(item, "arguments")?,
        ))
    })())
}

fn required_function_call_string<'a>(
    item: &'a Value,
    field: &str,
) -> Result<&'a str, OpenAiCodexError> {
    let value = item.get(field).and_then(Value::as_str).ok_or_else(|| {
        OpenAiCodexError::Protocol(format!(
            "function_call item did not include string `{field}`"
        ))
    })?;
    validate_function_call_field(field, value)?;
    Ok(value)
}

fn validate_function_call_field(field: &str, value: &str) -> Result<(), OpenAiCodexError> {
    let Some(limit) = function_call_field_limit(field) else {
        return Ok(());
    };
    if value.len() > limit {
        return Err(OpenAiCodexError::Protocol(format!(
            "function_call `{field}` exceeded the limit of {limit} bytes"
        )));
    }
    Ok(())
}

fn function_call_field_limit(field: &str) -> Option<usize> {
    match field {
        "call_id" => Some(MAX_FUNCTION_CALL_ID_BYTES),
        "name" => Some(MAX_FUNCTION_CALL_NAME_BYTES),
        "arguments" => Some(MAX_FUNCTION_CALL_ARGUMENT_BYTES),
        _ => None,
    }
}

fn is_function_call_item(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some("function_call")
    )
}

fn event_output_index(value: &Value) -> Option<usize> {
    value
        .get("output_index")
        .and_then(Value::as_u64)
        .and_then(|value| value.try_into().ok())
}

fn required_event_output_index(value: &Value) -> Result<usize, OpenAiCodexError> {
    event_output_index(value).ok_or_else(|| {
        OpenAiCodexError::Protocol(
            "function call stream event did not include integer `output_index`".to_string(),
        )
    })
}

fn required_event_value<'a>(value: &'a Value, field: &str) -> Result<&'a Value, OpenAiCodexError> {
    value.get(field).ok_or_else(|| {
        OpenAiCodexError::Protocol(format!(
            "function call stream event did not include `{field}`"
        ))
    })
}

fn required_event_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, OpenAiCodexError> {
    value.get(field).and_then(Value::as_str).ok_or_else(|| {
        OpenAiCodexError::Protocol(format!(
            "function call stream event did not include string `{field}`"
        ))
    })
}

#[derive(Debug, Clone, Default)]
struct StreamingFunctionCall {
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl StreamingFunctionCall {
    fn merge_item(&mut self, item: &Value) -> Result<(), OpenAiCodexError> {
        if let Some(call_id) = item.get("call_id").and_then(Value::as_str) {
            validate_function_call_field("call_id", call_id)?;
            self.call_id = Some(call_id.to_string());
        }
        if let Some(name) = item.get("name").and_then(Value::as_str) {
            validate_function_call_field("name", name)?;
            self.name = Some(name.to_string());
        }
        if let Some(arguments) = item.get("arguments").and_then(Value::as_str)
            && !arguments.is_empty()
        {
            validate_function_call_field("arguments", arguments)?;
            self.arguments = arguments.to_string();
        }
        Ok(())
    }

    fn to_item(&self) -> Result<Value, OpenAiCodexError> {
        let call_id = self.call_id.as_deref().ok_or_else(|| {
            OpenAiCodexError::Protocol("streamed function call did not include call_id".to_string())
        })?;
        let name = self.name.as_deref().ok_or_else(|| {
            OpenAiCodexError::Protocol("streamed function call did not include name".to_string())
        })?;
        Ok(json!({
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": self.arguments,
        }))
    }
}

fn collect_streaming_function_calls(
    calls: &BTreeMap<usize, StreamingFunctionCall>,
) -> Vec<ToolCall> {
    calls
        .values()
        .filter_map(|call| {
            Some(ToolCall::new(
                call.call_id.as_deref()?,
                call.name.as_deref()?,
                call.arguments.as_str(),
            ))
        })
        .collect()
}

fn collect_text(pieces: impl Iterator<Item = String>) -> Option<String> {
    let joined = pieces.collect::<String>();
    (!joined.is_empty()).then_some(joined)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn parses_json_function_call_turn() {
        let text = r#"{"output":[{"type":"function_call","call_id":"call_1","name":"ferricode_list_directory","arguments":"{\"path\":\".\"}"}]}"#;

        let turn = parse_assistant_turn(text).unwrap();

        let ProviderTurn::ToolCalls { state, calls } = turn else {
            panic!("expected function call turn");
        };
        assert_eq!(state.output_items.len(), 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id(), "call_1");
        assert_eq!(calls[0].name(), "ferricode_list_directory");
        assert_eq!(calls[0].arguments(), r#"{"path":"."}"#);
    }

    #[test]
    fn json_function_call_requires_provider_fields() {
        let text = r#"{"output":[{"type":"function_call","call_id":"call_1","name":"ferricode_list_directory"}]}"#;

        let error = parse_assistant_turn(text).unwrap_err();

        assert!(error.to_string().contains("arguments"));
    }

    #[test]
    fn json_function_call_rejects_oversized_arguments() {
        let body = json!({
            "output": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "ferricode_read_file",
                "arguments": "x".repeat((16 * 1024) + 1),
            }]
        })
        .to_string();

        let error = parse_assistant_turn(&body).unwrap_err();

        assert!(error.to_string().contains("arguments"));
        assert!(error.to_string().contains("limit"));
    }

    #[test]
    fn parses_streamed_function_call_arguments() {
        let text = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":""}}
data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"path\":\"READ"}
data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"ME.md\"}"}
data: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"path\":\"README.md\"}"}
data: {"type":"response.completed"}"#;

        let turn = parse_assistant_turn(text).unwrap();

        let ProviderTurn::ToolCalls { state, calls } = turn else {
            panic!("expected function call turn");
        };
        assert_eq!(state.output_items.len(), 1);
        assert_eq!(
            state.output_items[0]["arguments"],
            r#"{"path":"README.md"}"#
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id(), "call_1");
        assert_eq!(calls[0].name(), "ferricode_read_file");
        assert_eq!(calls[0].arguments(), r#"{"path":"README.md"}"#);
    }

    #[test]
    fn parses_streamed_function_call_from_done_item() {
        let text = r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":"{\"path\":\"README.md\"}"}}
data: {"type":"response.completed"}"#;

        let turn = parse_assistant_turn(text).unwrap();

        let ProviderTurn::ToolCalls { state, calls } = turn else {
            panic!("expected function call turn");
        };
        assert_eq!(state.output_items.len(), 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id(), "call_1");
        assert_eq!(calls[0].name(), "ferricode_read_file");
        assert_eq!(calls[0].arguments(), r#"{"path":"README.md"}"#);
    }

    #[test]
    fn streamed_function_call_requires_provider_identifiers() {
        let text = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","arguments":"{}"}}
data: {"type":"response.completed"}"#;

        let error = parse_assistant_turn(text).unwrap_err();

        assert!(error.to_string().contains("call_id"));
    }

    #[test]
    fn streamed_function_call_argument_delta_requires_delta() {
        let text = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":""}}
data: {"type":"response.function_call_arguments.delta","output_index":0}
data: {"type":"response.completed"}"#;

        let error = parse_assistant_turn(text).unwrap_err();

        assert!(error.to_string().contains("delta"));
    }

    #[test]
    fn streamed_function_call_argument_delta_requires_output_index() {
        let text = r#"data: {"type":"response.function_call_arguments.delta","delta":"{}"}
data: {"type":"response.completed"}"#;

        let error = parse_assistant_turn(text).unwrap_err();

        assert!(error.to_string().contains("output_index"));
    }

    #[test]
    fn streamed_function_call_rejects_oversized_argument_delta() {
        let body = format!(
            "data: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"ferricode_read_file\",\"arguments\":\"\"}}}}\n\
data: {{\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{}\"}}\n\
data: {{\"type\":\"response.completed\"}}",
            "x".repeat((16 * 1024) + 1)
        );

        let error = parse_assistant_turn(&body).unwrap_err();

        assert!(error.to_string().contains("arguments"));
        assert!(error.to_string().contains("limit"));
    }

    #[test]
    fn streamed_function_call_rejects_huge_sparse_index() {
        let text = r#"data: {"type":"response.output_item.added","output_index":999999,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":"{}"}}
data: {"type":"response.completed"}"#;

        let error = parse_assistant_turn(text).unwrap_err();

        assert!(error.to_string().contains("output_index"));
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
}
