//! Informational progress events that give terminal front ends a TUI seam now.
//!
//! The harness emits tool and completion events while the active provider
//! forwards assistant text deltas, letting a future TUI render progress without
//! changing the core/provider boundary. They never affect control flow:
//! dropping an event or ignoring a sink must leave the harness result
//! unchanged. Within one model turn, deltas arrive in provider order and are
//! followed by exactly one completion event whose text is the assistant text
//! core recorded for that turn (providers are expected to have streamed the
//! same text as deltas, but core cannot enforce that equality). A provider
//! failure ends the turn after any deltas already sent, with no completion
//! event. Tool events bracket each executed call.

use crate::{ToolCall, ToolOutput};

/// Something a front end may want to show while a request is in progress.
/// Events are informational: dropping them never changes harness behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessEvent {
    /// The harness is about to execute this tool call.
    ToolCallStarted { call: ToolCall },
    /// The harness finished executing the call with this output.
    ToolCallFinished { output: ToolOutput },
    /// A fragment of assistant text arrived from the active provider, in order.
    AssistantTextDelta { text: String },
    /// The harness finished one model turn; `text` is the assistant text of
    /// that turn, empty when the turn produced none.
    TurnFinished { text: String },
}

/// Receives lightweight request-progress events on the harness request path.
///
/// Implementations must return promptly and must not make event delivery part
/// of harness control flow, because a front end may always opt out by using
/// [`NoopEventSink`].
pub trait HarnessEventSink: Send + Sync {
    /// Accepts one informational event from the harness or active provider.
    fn on_event(&self, event: HarnessEvent);
}

/// Ignores every event for callers that only need the final harness response.
pub struct NoopEventSink;

impl HarnessEventSink for NoopEventSink {
    fn on_event(&self, _: HarnessEvent) {}
}
