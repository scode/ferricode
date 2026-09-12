//! Parses Responses API JSON and SSE output into assistant turns.
//!
//! This module treats backend event data as untrusted protocol input. While
//! reassembling function calls it caps the streamed output index and bounds
//! the size of each buffered function-call field so a single runaway field
//! cannot exhaust memory; it does not bound the number of output items or the
//! assistant text. It parses transport output only: it never executes calls,
//! selects tools, or decides provider policy.

use crate::{OpenAiCodexError, PROVIDER_NAME};
use ferricode_core::{
    HarnessEvent, HarnessEventSink, NoopEventSink, ProviderTurn, ToolCall, TranscriptItem,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::str;

const MAX_STREAMING_OUTPUT_INDEX: usize = 1024;

/// Maximum bytes retained for one streamed function-call field while it is
/// being reassembled. This is a transport guard, not tool policy: it must stay
/// well above core's per-field tool-call limits (see `ferricode_core::tools`),
/// so anything core would reject reaches core and becomes a recoverable tool
/// error rather than a transport failure that aborts the run. Exceeding it is
/// a `Protocol` error because a real backend never emits fields this large.
const MAX_STREAMED_FUNCTION_CALL_BYTES: usize = 256 * 1024;

/// Parses either JSON or minimal SSE `data:` events into assistant text.
pub fn parse_assistant_text(text: &str) -> Result<String, OpenAiCodexError> {
    match parse_assistant_turn(text)? {
        ProviderTurn::Final { text, .. } => Ok(text),
        ProviderTurn::ToolCalls { .. } => Err(OpenAiCodexError::MissingAssistantText),
    }
}

pub(crate) fn parse_assistant_turn(text: &str) -> Result<ProviderTurn, OpenAiCodexError> {
    parse_assistant_turn_with_sink(text, &NoopEventSink)
}

/// Parses one buffered backend response while forwarding its visible text.
///
/// This keeps the sink-less parser free of event-observation requirements,
/// while the provider uses the same parser with its caller's sink.
pub(crate) fn parse_assistant_turn_with_sink(
    text: &str,
    sink: &dyn HarnessEventSink,
) -> Result<ProviderTurn, OpenAiCodexError> {
    let trimmed = text.trim();
    if trimmed.lines().any(|line| line.starts_with("data:")) {
        return parse_sse_assistant_text(trimmed, sink);
    }

    let value: Value = serde_json::from_str(trimmed)?;
    let turn = parse_response_turn(&value)?;
    forward_buffered_text(sink, &turn);
    Ok(turn)
}

fn parse_sse_assistant_text(
    text: &str,
    sink: &dyn HarnessEventSink,
) -> Result<ProviderTurn, OpenAiCodexError> {
    let mut accumulator = SseAccumulator::new(sink);
    for line in text.lines() {
        if accumulator.process_line(line.as_bytes())? == SseDataAction::Complete {
            break;
        }
    }

    accumulator.into_provider_turn()
}

pub(crate) async fn parse_sse_assistant_stream(
    response: &mut reqwest::Response,
    sink: &dyn HarnessEventSink,
) -> Result<ProviderTurn, OpenAiCodexError> {
    let mut pending = Vec::new();
    let mut accumulator = SseAccumulator::new(sink);

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

fn parse_response_turn(value: &Value) -> Result<ProviderTurn, OpenAiCodexError> {
    let output_items = value
        .get("output")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let items = transcript_items_from_output(output_items)?;
    if items
        .iter()
        .any(|item| matches!(item, TranscriptItem::ToolCall(_)))
    {
        return Ok(ProviderTurn::ToolCalls { items });
    }

    let text = extract_text_from_response(value)
        .filter(|value| !value.trim().is_empty())
        .ok_or(OpenAiCodexError::MissingAssistantText)?;
    Ok(final_turn(text, items))
}

/// Retains complete streamed text for the returned turn while forwarding each
/// delta immediately to the caller's sink.
struct SseAccumulator<'a> {
    text: String,
    output_items: BTreeMap<usize, Value>,
    function_calls: BTreeMap<usize, StreamingFunctionCall>,
    sink: &'a dyn HarnessEventSink,
}

impl<'a> SseAccumulator<'a> {
    fn new(sink: &'a dyn HarnessEventSink) -> Self {
        Self {
            text: String::new(),
            output_items: BTreeMap::new(),
            function_calls: BTreeMap::new(),
            sink,
        }
    }

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
            self.sink
                .on_event(HarnessEvent::AssistantTextDelta { text });
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
                check_streamed_function_call_size("arguments", call.arguments.len() + delta.len())?;
                call.arguments.push_str(delta);
            }
            Some("response.function_call_arguments.done") => {
                let index = required_event_output_index(&value)?;
                let arguments = required_event_string(&value, "arguments")?;
                check_streamed_function_call_size("arguments", arguments.len())?;
                self.function_calls.entry(index).or_default().arguments = arguments.to_string();
            }
            Some("response.output_item.done") => {
                let index = required_event_output_index(&value)?;
                let item = required_event_value(&value, "item")?;
                if is_function_call_item(item) {
                    self.function_calls
                        .entry(index)
                        .or_default()
                        .merge_item(item)?;
                }
                self.output_items.insert(index, item.clone());
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

    fn into_provider_turn(mut self) -> Result<ProviderTurn, OpenAiCodexError> {
        if !self.function_calls.is_empty() {
            self.merge_streaming_function_items()?;
            let items = transcript_items_from_output(self.output_items.into_values().collect())?;
            return Ok(ProviderTurn::ToolCalls { items });
        }

        let text = completed_sse_text(self.text)?;
        let items = transcript_items_from_output(self.output_items.into_values().collect())?;
        Ok(final_turn(text, items))
    }

    fn merge_streaming_function_items(&mut self) -> Result<(), OpenAiCodexError> {
        for (index, call) in &self.function_calls {
            if *index >= MAX_STREAMING_OUTPUT_INDEX {
                return Err(OpenAiCodexError::Protocol(format!(
                    "streamed function call output_index {index} exceeded the limit of {MAX_STREAMING_OUTPUT_INDEX}"
                )));
            }
            let item = call.to_item()?;
            self.output_items.insert(*index, item);
        }
        Ok(())
    }
}

/// Forwards buffered response text as one delta so JSON and SSE expose the
/// same visible assistant text; core emits the turn completion after parsing.
fn forward_buffered_text(sink: &dyn HarnessEventSink, turn: &ProviderTurn) {
    let text = turn.assistant_text();
    if !text.is_empty() {
        sink.on_event(HarnessEvent::AssistantTextDelta { text });
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

/// Converts raw Responses output into core's ordered transcript vocabulary.
///
/// The provider keeps every item the backend produced. Known calls and
/// assistant messages become entries core can render and execute; reasoning
/// and future backend-specific item types stay opaque so the next request can
/// echo them without teaching core this wire format.
fn transcript_items_from_output(
    output_items: Vec<Value>,
) -> Result<Vec<TranscriptItem>, OpenAiCodexError> {
    let mut call_ids = BTreeSet::new();
    let mut items = Vec::with_capacity(output_items.len());
    for item in output_items {
        if let Some(call) = function_call_from_item(&item) {
            let call = call?;
            if !call_ids.insert(call.id().to_string()) {
                return Err(OpenAiCodexError::Protocol(format!(
                    "duplicate function call id `{}` in one response",
                    call.id()
                )));
            }
            items.push(TranscriptItem::ToolCall(call));
        } else if is_assistant_message_item(&item) {
            if let Some(text) = assistant_message_text(&item) {
                items.push(TranscriptItem::AssistantMessage { text });
            } else {
                items.push(TranscriptItem::ProviderOpaque {
                    provider: PROVIDER_NAME,
                    payload: item,
                });
            }
        } else {
            items.push(TranscriptItem::ProviderOpaque {
                provider: PROVIDER_NAME,
                payload: item,
            });
        }
    }
    Ok(items)
}

/// Requires both `type: message` and `role: assistant`.
///
/// A bare `content` array is not accepted because the role is what makes it
/// replayable as an assistant message; test fixtures must carry the full wire
/// shape rather than relying on a lenient parser.
fn is_assistant_message_item(item: &Value) -> bool {
    matches!(item.get("type").and_then(Value::as_str), Some("message"))
        && matches!(item.get("role").and_then(Value::as_str), Some("assistant"))
}

/// Returns renderable assistant text and leaves non-text messages opaque.
///
/// Refusals and empty content are meaningful provider state but have no core
/// assistant-text representation, so callers preserve their original item.
fn assistant_message_text(item: &Value) -> Option<String> {
    extract_text_from_output_item(item).filter(|text| !text.trim().is_empty())
}

/// Builds a final turn, supplying a message when items lack one.
///
/// Top-level `output_text` shorthand and SSE deltas without a `done` item can
/// provide final text without a completed message, so this preserves the
/// transcript invariant that final text is replayable.
fn final_turn(text: String, mut items: Vec<TranscriptItem>) -> ProviderTurn {
    if !items
        .iter()
        .any(|item| matches!(item, TranscriptItem::AssistantMessage { .. }))
    {
        items.push(TranscriptItem::AssistantMessage { text: text.clone() });
    }
    ProviderTurn::Final { text, items }
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
    Ok(value)
}

fn check_streamed_function_call_size(field: &str, size: usize) -> Result<(), OpenAiCodexError> {
    if size > MAX_STREAMED_FUNCTION_CALL_BYTES {
        return Err(OpenAiCodexError::Protocol(format!(
            "streamed function call `{field}` exceeded the buffer limit of {MAX_STREAMED_FUNCTION_CALL_BYTES} bytes"
        )));
    }
    Ok(())
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
            check_streamed_function_call_size("call_id", call_id.len())?;
            self.call_id = Some(call_id.to_string());
        }
        if let Some(name) = item.get("name").and_then(Value::as_str) {
            check_streamed_function_call_size("name", name.len())?;
            self.name = Some(name.to_string());
        }
        if let Some(arguments) = item.get("arguments").and_then(Value::as_str)
            && !arguments.is_empty()
        {
            check_streamed_function_call_size("arguments", arguments.len())?;
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

fn collect_text(pieces: impl Iterator<Item = String>) -> Option<String> {
    let joined = pieces.collect::<String>();
    (!joined.is_empty()).then_some(joined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferricode_core::{HarnessEvent, HarnessEventSink};
    use std::sync::Mutex;

    /// Captures parser events without involving a network response or UI.
    #[derive(Default)]
    struct RecordingSink(Mutex<Vec<HarnessEvent>>);

    impl HarnessEventSink for RecordingSink {
        fn on_event(&self, event: HarnessEvent) {
            self.0.lock().unwrap().push(event);
        }
    }

    #[test]
    fn parses_json_response_text() {
        let text = r#"{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}]}"#;

        assert_eq!(parse_assistant_text(text).unwrap(), "hello");
    }

    #[test]
    fn parses_sse_response_text() {
        let text = r#"data: {"type":"response.output_text.delta","delta":"hel"}
data: {"type":"response.output_text.delta","delta":"lo"}
data: [DONE]"#;

        assert_eq!(parse_assistant_text(text).unwrap(), "hello");
    }

    /// Streaming text must be forwarded in data-line order; core, rather than
    /// this provider parser, announces that the final turn completed.
    #[test]
    fn forwards_sse_text_deltas_in_order() {
        let text = r#"data: {"type":"response.output_text.delta","delta":"hel"}
data: {"type":"response.output_text.delta","delta":"lo"}
data: [DONE]"#;
        let sink = RecordingSink::default();

        let turn = parse_assistant_turn_with_sink(text, &sink).unwrap();

        assert!(matches!(turn, ProviderTurn::Final { ref text, .. } if text == "hello"));
        assert_eq!(
            *sink.0.lock().unwrap(),
            vec![
                HarnessEvent::AssistantTextDelta {
                    text: "hel".to_string(),
                },
                HarnessEvent::AssistantTextDelta {
                    text: "lo".to_string(),
                },
            ]
        );
    }

    /// Buffered JSON exposes one complete-text delta; core adds the matching
    /// turn completion after the provider returns the parsed turn.
    #[test]
    fn forwards_buffered_json_as_one_delta() {
        let sink = RecordingSink::default();

        parse_assistant_turn_with_sink(r#"{"output_text":"hello"}"#, &sink).unwrap();

        assert_eq!(
            *sink.0.lock().unwrap(),
            vec![HarnessEvent::AssistantTextDelta {
                text: "hello".to_string(),
            }]
        );
    }

    /// Buffered tool-call responses can include a visible assistant message,
    /// which must be forwarded before core executes the returned call.
    #[test]
    fn forwards_text_from_a_buffered_json_tool_call_turn() {
        let sink = RecordingSink::default();
        let text = r#"{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I will inspect it."}]},{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":"{\"path\":\"README.md\"}"}]}"#;

        parse_assistant_turn_with_sink(text, &sink).unwrap();

        assert_eq!(
            *sink.0.lock().unwrap(),
            vec![HarnessEvent::AssistantTextDelta {
                text: "I will inspect it.".to_string(),
            }]
        );
    }

    /// A streamed message can accompany function calls, so it must survive in
    /// transcript order and reach the sink even though core owns completion.
    #[test]
    fn forwards_text_from_an_sse_tool_call_turn_without_completing_it() {
        let text = r#"data: {"type":"response.output_text.delta","delta":"I will inspect it."}
data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I will inspect it."}]}}
data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":"{\"path\":\"README.md\"}"}}
data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":"{\"path\":\"README.md\"}"}}
data: {"type":"response.completed"}"#;
        let sink = RecordingSink::default();

        let turn = parse_assistant_turn_with_sink(text, &sink).unwrap();

        assert!(matches!(
            turn,
            ProviderTurn::ToolCalls { items }
                if matches!(
                    items.as_slice(),
                    [
                        TranscriptItem::AssistantMessage { text },
                        TranscriptItem::ToolCall(call),
                    ] if text == "I will inspect it." && call.id() == "call_1"
                )
        ));
        assert_eq!(
            *sink.0.lock().unwrap(),
            vec![HarnessEvent::AssistantTextDelta {
                text: "I will inspect it.".to_string(),
            }]
        );
    }

    /// A failed stream may already have shown text, but it must not fabricate
    /// another event or a completed turn after the backend reports failure.
    #[test]
    fn preserves_prior_delta_when_an_sse_response_fails() {
        let text = r#"data: {"type":"response.output_text.delta","delta":"partial"}
data: {"type":"response.failed"}"#;
        let sink = RecordingSink::default();

        assert!(parse_assistant_turn_with_sink(text, &sink).is_err());
        assert_eq!(
            *sink.0.lock().unwrap(),
            vec![HarnessEvent::AssistantTextDelta {
                text: "partial".to_string(),
            }]
        );
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

        let ProviderTurn::ToolCalls { items } = turn else {
            panic!("expected function call turn");
        };
        assert!(
            matches!(items.as_slice(), [TranscriptItem::ToolCall(call)] if call.id() == "call_1")
        );
    }

    /// Reasoning is not a core concept, but it must stay immediately before
    /// the call it informed so the next OpenAI request can retain that state.
    #[test]
    fn preserves_reasoning_before_a_function_call() {
        let text = r#"{"output":[{"type":"reasoning","id":"rs_1","summary":[]},{"type":"function_call","call_id":"call_1","name":"ferricode_list_directory","arguments":"{\"path\":\".\"}"}]}"#;

        let ProviderTurn::ToolCalls { items } = parse_assistant_turn(text).unwrap() else {
            panic!("expected function call turn");
        };

        assert!(matches!(
            items.as_slice(),
            [TranscriptItem::ProviderOpaque { provider: PROVIDER_NAME, payload }, TranscriptItem::ToolCall(call)]
                if payload["type"] == "reasoning" && call.id() == "call_1"
        ));
    }

    /// Final assistant text has to become a transcript message, not merely the
    /// response summary, or a later provider cannot render the conversation.
    #[test]
    fn converts_output_message_to_assistant_transcript_item() {
        let text = r#"{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}]}"#;

        let ProviderTurn::Final { text, items } = parse_assistant_turn(text).unwrap() else {
            panic!("expected final turn");
        };

        assert_eq!(text, "hello");
        assert!(
            matches!(items.as_slice(), [TranscriptItem::AssistantMessage { text }] if text == "hello")
        );
    }

    #[test]
    fn json_function_call_requires_provider_fields() {
        let text = r#"{"output":[{"type":"function_call","call_id":"call_1","name":"ferricode_list_directory"}]}"#;

        let error = parse_assistant_turn(text).unwrap_err();

        assert!(error.to_string().contains("arguments"));
    }

    /// A JSON response just over core's argument limit must reach core intact,
    /// where the harness can turn the policy violation into a tool error.
    #[test]
    fn json_function_call_passes_through_arguments_over_core_limit() {
        let body = json!({
            "output": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "ferricode_read_file",
                "arguments": "x".repeat((16 * 1024) + 1),
            }]
        })
        .to_string();

        let turn = parse_assistant_turn(&body).unwrap();

        let ProviderTurn::ToolCalls { items } = turn else {
            panic!("expected function call turn");
        };
        assert!(
            matches!(items.as_slice(), [TranscriptItem::ToolCall(call)] if call.arguments() == "x".repeat((16 * 1024) + 1))
        );
    }

    #[test]
    fn parses_streamed_function_call_arguments() {
        let text = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":""}}
data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"path\":\"READ"}
data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"ME.md\"}"}
data: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"path\":\"README.md\"}"}
data: {"type":"response.completed"}"#;

        let turn = parse_assistant_turn(text).unwrap();

        let ProviderTurn::ToolCalls { items } = turn else {
            panic!("expected function call turn");
        };
        assert!(
            matches!(items.as_slice(), [TranscriptItem::ToolCall(call)] if call.arguments() == r#"{"path":"README.md"}"#)
        );
    }

    #[test]
    fn parses_streamed_function_call_from_done_item() {
        let text = r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":"{\"path\":\"README.md\"}"}}
data: {"type":"response.completed"}"#;

        let turn = parse_assistant_turn(text).unwrap();

        let ProviderTurn::ToolCalls { items } = turn else {
            panic!("expected function call turn");
        };
        assert!(
            matches!(items.as_slice(), [TranscriptItem::ToolCall(call)] if call.id() == "call_1")
        );
    }

    /// A function call at output index one still represents one backend item.
    /// The added and done events describe the same call, and sparse indexes
    /// must not manufacture a second call or a placeholder at index zero.
    #[test]
    fn streamed_function_call_at_sparse_output_index_is_not_duplicated() {
        let text = r#"data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":""}}
data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":"{\"path\":\"README.md\"}"}}
data: {"type":"response.completed"}"#;

        let turn = parse_assistant_turn(text).unwrap();

        assert!(
            matches!(turn, ProviderTurn::ToolCalls { items } if matches!(items.as_slice(), [TranscriptItem::ToolCall(call)] if call.id() == "call_1"))
        );
    }

    /// Desynchronized done-event indexes must not make core execute the same
    /// backend call twice, even though both output items otherwise parse.
    #[test]
    fn streamed_duplicate_function_call_id_is_a_protocol_error() {
        let text = r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":"{\"path\":\"README.md\"}"}}
data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":"{\"path\":\"README.md\"}"}}
data: {"type":"response.completed"}"#;

        let error = parse_assistant_turn(text).unwrap_err();

        assert_eq!(
            error.to_string(),
            "duplicate function call id `call_1` in one response"
        );
    }

    /// A textless assistant message is backend state rather than a malformed
    /// turn, so a sibling tool call remains executable and the message survives replay.
    #[test]
    fn tool_call_turn_preserves_empty_assistant_message_as_opaque() {
        let text = r#"{"output":[{"type":"function_call","call_id":"call_1","name":"ferricode_read_file","arguments":"{\"path\":\"README.md\"}"},{"type":"message","role":"assistant","content":[]}]}"#;

        let ProviderTurn::ToolCalls { items } = parse_assistant_turn(text).unwrap() else {
            panic!("expected function call turn");
        };

        assert!(matches!(
            items.as_slice(),
            [TranscriptItem::ToolCall(call), TranscriptItem::ProviderOpaque { provider: PROVIDER_NAME, payload }]
                if call.id() == "call_1" && payload["type"] == "message"
        ));
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

    /// A streamed argument just over core's limit is still below the transport
    /// buffer guard and must be passed through without truncation.
    #[test]
    fn streamed_function_call_passes_through_delta_over_core_limit() {
        let body = format!(
            "data: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"ferricode_read_file\",\"arguments\":\"\"}}}}\n\
data: {{\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{}\"}}\n\
data: {{\"type\":\"response.completed\"}}",
            "x".repeat((16 * 1024) + 1)
        );

        let turn = parse_assistant_turn(&body).unwrap();

        let ProviderTurn::ToolCalls { items } = turn else {
            panic!("expected function call turn");
        };
        assert!(
            matches!(items.as_slice(), [TranscriptItem::ToolCall(call)] if call.arguments() == "x".repeat((16 * 1024) + 1))
        );
    }

    /// The buffer guard is cumulative across deltas: two deltas that are each
    /// under the ceiling but together exceed it are a broken transport input,
    /// so parsing must fail before producing a tool call. A guard that only
    /// looked at each delta on its own would pass this stream.
    #[test]
    fn streamed_function_call_rejects_transport_buffer_overflow() {
        let body = format!(
            "data: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"ferricode_read_file\",\"arguments\":\"\"}}}}\n\
data: {{\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{}\"}}\n\
data: {{\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{}\"}}\n\
data: {{\"type\":\"response.completed\"}}",
            "x".repeat(128 * 1024),
            "x".repeat((128 * 1024) + 1)
        );

        let error = parse_assistant_turn(&body).unwrap_err();

        assert!(
            matches!(error, OpenAiCodexError::Protocol(message) if message.contains("buffer limit"))
        );
    }

    /// The same ceiling applies to the identifier fields carried on an item
    /// event, not only to streamed argument text, so an absurd `call_id` fails
    /// the same way.
    #[test]
    fn streamed_function_call_rejects_oversized_call_id() {
        let body = format!(
            "data: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"type\":\"function_call\",\"call_id\":\"{}\",\"name\":\"ferricode_read_file\",\"arguments\":\"\"}}}}\n\
data: {{\"type\":\"response.completed\"}}",
            "c".repeat((256 * 1024) + 1)
        );

        let error = parse_assistant_turn(&body).unwrap_err();

        assert!(
            matches!(error, OpenAiCodexError::Protocol(message) if message.contains("`call_id`"))
        );
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
