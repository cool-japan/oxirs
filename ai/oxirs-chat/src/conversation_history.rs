//! Conversation history manager with token tracking and summarisation support.
//!
//! Maintains an ordered queue of [`Message`]s, tracks cumulative token usage,
//! supports soft summarisation (compress old messages into a
//! [`ConversationSummary`]), and provides helpers for preparing context
//! windows for LLM calls.

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Sender/receiver role of a conversation message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    /// System prompt / instruction.
    System,
    /// Human user turn.
    User,
    /// Language-model assistant turn.
    Assistant,
    /// Tool / function call result.
    Tool,
}

impl MessageRole {
    /// Return the lowercase string representation used in prompt serialisation.
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        }
    }
}

/// A single conversation message.
#[derive(Debug, Clone)]
pub struct Message {
    /// Who sent the message.
    pub role: MessageRole,
    /// Text content.
    pub content: String,
    /// Timestamp in arbitrary units (e.g. Unix ms).
    pub timestamp_ms: u64,
    /// Token count for this message, if known.
    pub token_count: Option<usize>,
}

impl Message {
    /// Create a new message.
    pub fn new(role: MessageRole, content: impl Into<String>, timestamp_ms: u64) -> Self {
        Self {
            role,
            content: content.into(),
            timestamp_ms,
            token_count: None,
        }
    }

    /// Create a new message with a token count.
    pub fn with_tokens(
        role: MessageRole,
        content: impl Into<String>,
        timestamp_ms: u64,
        token_count: usize,
    ) -> Self {
        Self {
            role,
            content: content.into(),
            timestamp_ms,
            token_count: Some(token_count),
        }
    }
}

/// Configuration for a `ConversationHistory`.
#[derive(Debug, Clone)]
pub struct HistoryConfig {
    /// Maximum number of messages to keep in the queue.
    pub max_messages: usize,
    /// Soft token budget: triggers `needs_summarization()` when exceeded.
    pub max_tokens: usize,
    /// Fraction of `max_tokens` at which summarisation is recommended (0–1).
    pub summary_trigger_ratio: f64,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            max_messages: 50,
            max_tokens: 4096,
            summary_trigger_ratio: 0.8,
        }
    }
}

/// A compressed representation of previous conversation turns.
#[derive(Debug, Clone)]
pub struct ConversationSummary {
    /// Natural-language summary text.
    pub content: String,
    /// Number of messages that were compressed.
    pub messages_summarized: usize,
    /// Total token count of the messages before summarisation.
    pub token_count_before: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// ConversationHistory
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe-friendly in-process conversation history manager.
pub struct ConversationHistory {
    /// Ordered message queue (front = oldest).
    messages: VecDeque<Message>,
    /// Latest summary, if any.
    summary: Option<ConversationSummary>,
    /// Configuration.
    config: HistoryConfig,
    /// Running total of token counts (only messages with `Some(token_count)`).
    total_tokens: usize,
}

impl ConversationHistory {
    /// Create an empty conversation history with the given configuration.
    pub fn new(config: HistoryConfig) -> Self {
        Self {
            messages: VecDeque::new(),
            summary: None,
            config,
            total_tokens: 0,
        }
    }

    // ── mutation ──────────────────────────────────────────────────────────────

    /// Append a message to the history.
    ///
    /// Updates the running token total; if the message carries no explicit
    /// token count, a rough estimate of `content.len() / 4` is used.
    pub fn push(&mut self, message: Message) {
        let tokens = message
            .token_count
            .unwrap_or_else(|| estimate_tokens(&message.content));
        self.total_tokens += tokens;
        self.messages.push_back(message);
    }

    /// Clear all messages and the current summary, resetting token count.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.summary = None;
        self.total_tokens = 0;
    }

    // ── read-only accessors ───────────────────────────────────────────────────

    /// Immutable reference to the message queue.
    pub fn messages(&self) -> &VecDeque<Message> {
        &self.messages
    }

    /// Most-recently added message.
    pub fn latest(&self) -> Option<&Message> {
        self.messages.back()
    }

    /// Number of messages in the queue.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Total tracked token count.
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Current summary, if any.
    pub fn summary(&self) -> Option<&ConversationSummary> {
        self.summary.as_ref()
    }

    // ── summarisation ─────────────────────────────────────────────────────────

    /// Returns `true` when the running token total exceeds
    /// `max_tokens * summary_trigger_ratio`.
    pub fn needs_summarization(&self) -> bool {
        let threshold =
            (self.config.max_tokens as f64 * self.config.summary_trigger_ratio) as usize;
        self.total_tokens > threshold
    }

    /// Replace all non-System messages with a single summary message and store
    /// the compressed summary.
    ///
    /// Returns the created [`ConversationSummary`].
    pub fn summarize(&mut self, summary_content: String) -> ConversationSummary {
        // Preserve System messages.
        let system_messages: Vec<Message> = self
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .cloned()
            .collect();

        let non_system_count = self.messages.len() - system_messages.len();
        let tokens_before = self.total_tokens;

        // Rebuild queue from scratch.
        self.messages.clear();
        self.total_tokens = 0;
        for msg in system_messages {
            self.push(msg);
        }

        // Inject summary as an assistant turn.
        let summary_msg = Message::with_tokens(
            MessageRole::Assistant,
            format!("[Summary] {summary_content}"),
            0,
            estimate_tokens(&summary_content),
        );
        self.push(summary_msg);

        let summary = ConversationSummary {
            content: summary_content,
            messages_summarized: non_system_count,
            token_count_before: tokens_before,
        };
        self.summary = Some(summary.clone());
        summary
    }

    // ── LLM prompt helpers ────────────────────────────────────────────────────

    /// Return `(role_str, content_str)` pairs suitable for passing to an LLM
    /// API, preserving message order.
    pub fn to_prompt_messages(&self) -> Vec<(&str, &str)> {
        self.messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str()))
            .collect()
    }

    /// Most recent message with role `User`.
    pub fn last_user_message(&self) -> Option<&Message> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
    }

    /// Most recent message with role `Assistant`.
    pub fn last_assistant_message(&self) -> Option<&Message> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::Assistant)
    }

    // ── truncation ────────────────────────────────────────────────────────────

    /// Remove the oldest non-System messages until `total_tokens` is at or
    /// below `max_tokens`.
    ///
    /// Returns the number of messages removed.
    pub fn truncate_to_limit(&mut self) -> usize {
        let max = self.config.max_tokens;
        let mut removed = 0;
        while self.total_tokens > max {
            // Find the first (oldest) non-System message and remove it.
            let pos = self
                .messages
                .iter()
                .position(|m| m.role != MessageRole::System);
            match pos {
                Some(idx) => {
                    if let Some(msg) = self.messages.remove(idx) {
                        let tokens = msg
                            .token_count
                            .unwrap_or_else(|| estimate_tokens(&msg.content));
                        self.total_tokens = self.total_tokens.saturating_sub(tokens);
                        removed += 1;
                    }
                }
                None => break, // only System messages remain; stop
            }
        }
        removed
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Rough token count estimate: 1 token ≈ 4 characters.
fn estimate_tokens(content: &str) -> usize {
    (content.len() + 3) / 4
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_history() -> ConversationHistory {
        ConversationHistory::new(HistoryConfig::default())
    }

    fn user_msg(content: &str, tokens: usize) -> Message {
        Message::with_tokens(MessageRole::User, content, 0, tokens)
    }

    fn assistant_msg(content: &str, tokens: usize) -> Message {
        Message::with_tokens(MessageRole::Assistant, content, 0, tokens)
    }

    fn system_msg(content: &str) -> Message {
        Message::with_tokens(MessageRole::System, content, 0, 10)
    }

    // ── push / messages ───────────────────────────────────────────────────────

    #[test]
    fn test_push_increases_count() {
        let mut h = default_history();
        h.push(user_msg("Hello", 5));
        assert_eq!(h.message_count(), 1);
    }

    #[test]
    fn test_push_multiple_messages() {
        let mut h = default_history();
        h.push(user_msg("A", 1));
        h.push(assistant_msg("B", 2));
        h.push(user_msg("C", 3));
        assert_eq!(h.message_count(), 3);
    }

    #[test]
    fn test_messages_preserves_order() {
        let mut h = default_history();
        h.push(user_msg("first", 1));
        h.push(user_msg("second", 1));
        let msgs: Vec<&str> = h.messages().iter().map(|m| m.content.as_str()).collect();
        assert_eq!(msgs, vec!["first", "second"]);
    }

    // ── latest ────────────────────────────────────────────────────────────────

    #[test]
    fn test_latest_returns_last_pushed() {
        let mut h = default_history();
        h.push(user_msg("first", 1));
        h.push(assistant_msg("last", 1));
        assert_eq!(h.latest().expect("should succeed").content, "last");
    }

    #[test]
    fn test_latest_empty_none() {
        let h = default_history();
        assert!(h.latest().is_none());
    }

    // ── message_count ─────────────────────────────────────────────────────────

    #[test]
    fn test_message_count_zero_initially() {
        assert_eq!(default_history().message_count(), 0);
    }

    // ── total_tokens ──────────────────────────────────────────────────────────

    #[test]
    fn test_total_tokens_sums_correctly() {
        let mut h = default_history();
        h.push(user_msg("A", 10));
        h.push(assistant_msg("B", 20));
        assert_eq!(h.total_tokens(), 30);
    }

    #[test]
    fn test_total_tokens_estimated_when_no_count() {
        let mut h = default_history();
        // "ABCD" = 4 chars → 1 token by estimate
        h.push(Message::new(MessageRole::User, "ABCD", 0));
        assert_eq!(h.total_tokens(), 1);
    }

    #[test]
    fn test_total_tokens_zero_initially() {
        assert_eq!(default_history().total_tokens(), 0);
    }

    // ── needs_summarization ───────────────────────────────────────────────────

    #[test]
    fn test_needs_summarization_false_when_below_threshold() {
        let mut h = ConversationHistory::new(HistoryConfig {
            max_tokens: 100,
            summary_trigger_ratio: 0.8,
            ..HistoryConfig::default()
        });
        // 60 tokens = 60 %, below 80 % threshold
        h.push(user_msg("msg", 60));
        assert!(!h.needs_summarization());
    }

    #[test]
    fn test_needs_summarization_true_when_above_threshold() {
        let mut h = ConversationHistory::new(HistoryConfig {
            max_tokens: 100,
            summary_trigger_ratio: 0.8,
            ..HistoryConfig::default()
        });
        // 90 tokens = 90 % > 80 %
        h.push(user_msg("msg", 90));
        assert!(h.needs_summarization());
    }

    // ── summarize ─────────────────────────────────────────────────────────────

    #[test]
    fn test_summarize_replaces_non_system_messages() {
        let mut h = default_history();
        h.push(system_msg("You are a helpful assistant."));
        h.push(user_msg("Hello", 5));
        h.push(assistant_msg("Hi there", 5));
        let summary = h.summarize("User said hello.".into());
        assert_eq!(summary.messages_summarized, 2);
        // Only system + one summary assistant message should remain.
        assert_eq!(h.message_count(), 2);
    }

    #[test]
    fn test_summarize_preserves_system_message() {
        let mut h = default_history();
        h.push(system_msg("System prompt."));
        h.push(user_msg("Q", 5));
        h.summarize("Summary.".into());
        let system_count = h
            .messages()
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .count();
        assert_eq!(system_count, 1);
    }

    #[test]
    fn test_summarize_returns_summary_struct() {
        let mut h = default_history();
        h.push(user_msg("msg", 100));
        let summary = h.summarize("Compact.".into());
        assert_eq!(summary.content, "Compact.");
        assert_eq!(summary.messages_summarized, 1);
        assert!(summary.token_count_before > 0);
    }

    #[test]
    fn test_summarize_stores_summary() {
        let mut h = default_history();
        h.push(user_msg("msg", 10));
        h.summarize("Summary text".into());
        assert!(h.summary().is_some());
        assert_eq!(h.summary().expect("should succeed").content, "Summary text");
    }

    // ── clear ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_empties_messages() {
        let mut h = default_history();
        h.push(user_msg("A", 5));
        h.push(user_msg("B", 5));
        h.clear();
        assert_eq!(h.message_count(), 0);
    }

    #[test]
    fn test_clear_resets_tokens() {
        let mut h = default_history();
        h.push(user_msg("A", 50));
        h.clear();
        assert_eq!(h.total_tokens(), 0);
    }

    #[test]
    fn test_clear_removes_summary() {
        let mut h = default_history();
        h.push(user_msg("A", 5));
        h.summarize("S".into());
        h.clear();
        assert!(h.summary().is_none());
    }

    // ── to_prompt_messages ────────────────────────────────────────────────────

    #[test]
    fn test_to_prompt_messages_format() {
        let mut h = default_history();
        h.push(system_msg("You are helpful."));
        h.push(user_msg("Hi", 1));
        let pairs = h.to_prompt_messages();
        assert_eq!(pairs[0].0, "system");
        assert_eq!(pairs[1].0, "user");
    }

    #[test]
    fn test_to_prompt_messages_content() {
        let mut h = default_history();
        h.push(assistant_msg("Hello!", 1));
        let pairs = h.to_prompt_messages();
        assert_eq!(pairs[0].1, "Hello!");
    }

    #[test]
    fn test_to_prompt_messages_empty() {
        let h = default_history();
        assert!(h.to_prompt_messages().is_empty());
    }

    // ── last_user_message / last_assistant_message ────────────────────────────

    #[test]
    fn test_last_user_message() {
        let mut h = default_history();
        h.push(user_msg("first user", 1));
        h.push(assistant_msg("response", 1));
        h.push(user_msg("second user", 1));
        assert_eq!(
            h.last_user_message().expect("should succeed").content,
            "second user"
        );
    }

    #[test]
    fn test_last_user_message_none_when_absent() {
        let mut h = default_history();
        h.push(assistant_msg("reply", 1));
        assert!(h.last_user_message().is_none());
    }

    #[test]
    fn test_last_assistant_message() {
        let mut h = default_history();
        h.push(assistant_msg("first reply", 1));
        h.push(user_msg("follow-up", 1));
        h.push(assistant_msg("second reply", 1));
        assert_eq!(
            h.last_assistant_message().expect("should succeed").content,
            "second reply"
        );
    }

    #[test]
    fn test_last_assistant_message_none_when_absent() {
        let mut h = default_history();
        h.push(user_msg("hello", 1));
        assert!(h.last_assistant_message().is_none());
    }

    // ── truncate_to_limit ─────────────────────────────────────────────────────

    #[test]
    fn test_truncate_removes_oldest_non_system() {
        let mut h = ConversationHistory::new(HistoryConfig {
            max_tokens: 10,
            summary_trigger_ratio: 0.8,
            max_messages: 100,
        });
        h.push(system_msg("sys")); // 10 tokens
        h.push(user_msg("u1", 5));
        h.push(user_msg("u2", 5));
        // total = 20; limit = 10; need to remove non-system msgs
        let removed = h.truncate_to_limit();
        assert!(removed > 0);
    }

    #[test]
    fn test_truncate_preserves_system_message() {
        let mut h = ConversationHistory::new(HistoryConfig {
            max_tokens: 5,
            summary_trigger_ratio: 0.8,
            max_messages: 100,
        });
        h.push(system_msg("sys")); // 10 tokens > limit but cannot be removed
        h.push(user_msg("big user message that fills tokens", 100));
        h.truncate_to_limit();
        let system_count = h
            .messages()
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .count();
        assert_eq!(system_count, 1);
    }

    #[test]
    fn test_truncate_returns_count_removed() {
        let mut h = ConversationHistory::new(HistoryConfig {
            max_tokens: 5,
            summary_trigger_ratio: 1.0,
            max_messages: 100,
        });
        h.push(user_msg("a", 3));
        h.push(user_msg("b", 3)); // total = 6 > 5
        let removed = h.truncate_to_limit();
        assert_eq!(removed, 1);
    }

    #[test]
    fn test_truncate_no_op_when_within_limit() {
        let mut h = ConversationHistory::new(HistoryConfig {
            max_tokens: 100,
            summary_trigger_ratio: 0.8,
            max_messages: 50,
        });
        h.push(user_msg("small", 5));
        let removed = h.truncate_to_limit();
        assert_eq!(removed, 0);
    }

    // ── MessageRole as_str ────────────────────────────────────────────────────

    #[test]
    fn test_role_as_str_system() {
        assert_eq!(MessageRole::System.as_str(), "system");
    }

    #[test]
    fn test_role_as_str_tool() {
        assert_eq!(MessageRole::Tool.as_str(), "tool");
    }

    #[test]
    fn test_role_as_str_user() {
        assert_eq!(MessageRole::User.as_str(), "user");
    }

    #[test]
    fn test_role_as_str_assistant() {
        assert_eq!(MessageRole::Assistant.as_str(), "assistant");
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_message_new_no_token_count() {
        let m = Message::new(MessageRole::User, "hello", 100);
        assert!(m.token_count.is_none());
    }

    #[test]
    fn test_message_with_tokens_stores_count() {
        let m = Message::with_tokens(MessageRole::User, "hello", 100, 50);
        assert_eq!(m.token_count, Some(50));
    }

    #[test]
    fn test_message_timestamp_stored() {
        let m = Message::new(MessageRole::User, "x", 99999);
        assert_eq!(m.timestamp_ms, 99999);
    }

    #[test]
    fn test_default_history_config() {
        let cfg = HistoryConfig::default();
        assert_eq!(cfg.max_messages, 50);
        assert_eq!(cfg.max_tokens, 4096);
        assert!((cfg.summary_trigger_ratio - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_summary_messages_summarized_count() {
        let mut h = default_history();
        h.push(user_msg("u1", 5));
        h.push(assistant_msg("a1", 5));
        h.push(user_msg("u2", 5));
        let summary = h.summarize("Summary".into());
        assert_eq!(summary.messages_summarized, 3);
    }

    #[test]
    fn test_summary_token_count_before_nonzero() {
        let mut h = default_history();
        h.push(user_msg("msg", 100));
        let summary = h.summarize("S".into());
        assert_eq!(summary.token_count_before, 100);
    }

    #[test]
    fn test_push_tool_message() {
        let mut h = default_history();
        h.push(Message::with_tokens(MessageRole::Tool, "result", 0, 3));
        assert_eq!(h.message_count(), 1);
        assert_eq!(h.latest().expect("should succeed").role, MessageRole::Tool);
    }

    #[test]
    fn test_to_prompt_messages_all_roles() {
        let mut h = default_history();
        h.push(system_msg("sys"));
        h.push(user_msg("hi", 1));
        h.push(assistant_msg("hello", 1));
        let pairs = h.to_prompt_messages();
        assert_eq!(pairs[0].0, "system");
        assert_eq!(pairs[1].0, "user");
        assert_eq!(pairs[2].0, "assistant");
    }

    #[test]
    fn test_last_user_message_empty() {
        assert!(default_history().last_user_message().is_none());
    }

    #[test]
    fn test_last_assistant_message_before_user() {
        let mut h = default_history();
        h.push(assistant_msg("first", 1));
        h.push(user_msg("second", 1));
        // last assistant is still "first"
        assert_eq!(
            h.last_assistant_message().expect("should succeed").content,
            "first"
        );
    }

    #[test]
    fn test_truncate_reduces_total_tokens() {
        let mut h = ConversationHistory::new(HistoryConfig {
            max_tokens: 5,
            summary_trigger_ratio: 1.0,
            max_messages: 100,
        });
        h.push(user_msg("a", 3));
        h.push(user_msg("b", 3));
        h.truncate_to_limit();
        assert!(h.total_tokens() <= 5);
    }

    #[test]
    fn test_estimate_tokens_roughly_quarter_char_count() {
        // "AAAA" = 4 chars → 1 token
        let mut h = default_history();
        h.push(Message::new(MessageRole::User, "AAAA", 0));
        assert_eq!(h.total_tokens(), 1);
    }

    #[test]
    fn test_push_zero_token_message() {
        let mut h = default_history();
        h.push(Message::with_tokens(MessageRole::User, "", 0, 0));
        assert_eq!(h.total_tokens(), 0);
    }

    #[test]
    fn test_summarize_multiple_systems_all_preserved() {
        let mut h = default_history();
        h.push(system_msg("s1"));
        h.push(system_msg("s2"));
        h.push(user_msg("q", 1));
        h.summarize("S".into());
        let sys_count = h
            .messages()
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .count();
        assert_eq!(sys_count, 2);
    }
}
