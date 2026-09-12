//! Provider-neutral conversation history owned by the harness.
//!
//! A provider cannot safely own the conversation: core needs the same history
//! to show an agent's work, persist a session, or hand a later request to a
//! different provider. Most entries have a core-defined meaning. Some backends
//! also emit state that only they understand, such as OpenAI reasoning items
//! that the Responses API expects to see again on the next request. Core
//! never inspects its payload, preserves its order, and tags it with the
//! provider allowed to replay it.

use crate::{ProviderRequest, ToolCall, ToolOutput};

/// One entry in a provider-neutral conversation transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptItem {
    /// Text the user sent. The harness prepends working-directory context when
    /// it builds this item, so providers must not add the context again.
    UserMessage { text: String },
    /// Final or intermediate assistant text.
    AssistantMessage { text: String },
    /// A tool call the model requested, exactly as core received it.
    ToolCall(ToolCall),
    /// The output core produced for a tool call, keyed by call id.
    ToolResult(ToolOutput),
    /// Provider-only state that must be replayed by the named provider.
    ///
    /// Core does not validate or read `payload`, including when it hands the
    /// transcript to another provider. Consumers must skip entries tagged for
    /// a different provider rather than asking core to filter them.
    ///
    /// A persisted transcript would need an owned tag (`Cow<'static, str>`);
    /// that is deferred until persistence exists.
    ProviderOpaque {
        provider: &'static str,
        payload: serde_json::Value,
    },
}

/// Ordered conversation history owned by the harness.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Transcript {
    items: Vec<TranscriptItem>,
}

impl Transcript {
    /// Creates an empty transcript for one harness request.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates the initial history for a model request with its directory context.
    ///
    /// This is the only place that composes the `Working directory:` prefix,
    /// so the harness and provider convenience path cannot drift apart.
    pub fn for_request(request: &ProviderRequest) -> Self {
        Self {
            items: vec![TranscriptItem::UserMessage {
                text: format!(
                    "Working directory: {}\n\n{}",
                    request.working_directory().display(),
                    request.prompt()
                ),
            }],
        }
    }

    /// Appends one event without interpreting, deduplicating, or reordering it.
    pub fn push(&mut self, item: TranscriptItem) {
        self.items.push(item);
    }

    /// Returns the complete ordered history providers render into wire input.
    pub fn items(&self) -> &[TranscriptItem] {
        &self.items
    }
}

impl Extend<TranscriptItem> for Transcript {
    /// Appends a sequence without interpreting, deduplicating, or reordering it.
    ///
    /// Provider turns are already ordered protocol history, so callers use this
    /// instead of rebuilding the same append loop at each orchestration point.
    fn extend<T: IntoIterator<Item = TranscriptItem>>(&mut self, iter: T) {
        self.items.extend(iter);
    }
}
