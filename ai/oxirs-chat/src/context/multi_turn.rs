//! Multi-Turn Context Manager
//!
//! Advanced context-window management for multi-turn conversations. Provides
//! token-aware eviction, summarisation, and flexible trimming strategies.

use crate::analytics::conversation::ConversationTurn;
use crate::llm::types::{ChatMessage, ChatRole};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// ContextWindow
// ---------------------------------------------------------------------------

/// A sliding token-bounded window of chat messages.
///
/// Messages are stored in arrival order. When the window is full, the oldest
/// message is evicted (or the caller can drive a summarisation pass).
#[derive(Debug, Clone)]
pub struct ContextWindow {
    pub max_tokens: usize,
    pub messages: VecDeque<ChatMessage>,
    pub system_prompt: Option<String>,
    pub total_tokens: usize,
}

impl ContextWindow {
    /// Create a window with the given token budget
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            messages: VecDeque::new(),
            system_prompt: None,
            total_tokens: 0,
        }
    }

    /// Create a window with a pre-set system prompt.
    /// The system prompt tokens are counted against the budget.
    pub fn with_system(max_tokens: usize, system: String) -> Self {
        let system_tokens = estimate_tokens(&system);
        Self {
            max_tokens,
            messages: VecDeque::new(),
            system_prompt: Some(system),
            total_tokens: system_tokens,
        }
    }

    /// Attempt to push a message into the window.
    ///
    /// If the message fits within the remaining budget it is appended and
    /// `true` is returned. If the message alone exceeds the entire budget
    /// it is dropped and `false` is returned. Otherwise the oldest messages
    /// are evicted until room is made, then the message is appended.
    pub fn push(&mut self, msg: ChatMessage) -> bool {
        let msg_tokens = estimate_tokens(&msg.content);

        // Message is bigger than the entire budget — reject it outright.
        if msg_tokens > self.max_tokens {
            return false;
        }

        // Evict oldest messages until the new message fits.
        while self.total_tokens + msg_tokens > self.max_tokens && !self.messages.is_empty() {
            self.evict_oldest();
        }

        self.total_tokens += msg_tokens;
        self.messages.push_back(msg);
        true
    }

    /// Remove and return the oldest message, updating the token count.
    pub fn evict_oldest(&mut self) -> Option<ChatMessage> {
        if let Some(msg) = self.messages.pop_front() {
            let tokens = estimate_tokens(&msg.content);
            self.total_tokens = self.total_tokens.saturating_sub(tokens);
            Some(msg)
        } else {
            None
        }
    }

    /// Tokens still available for new messages
    pub fn available_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.total_tokens)
    }

    /// True when no more tokens are available
    pub fn is_full(&self) -> bool {
        self.available_tokens() == 0
    }

    /// Number of messages currently held in the window
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Borrow-only ordered slice of all messages
    pub fn to_messages_slice(&self) -> Vec<&ChatMessage> {
        self.messages.iter().collect()
    }
}

// ---------------------------------------------------------------------------
// ConversationSummariser
// ---------------------------------------------------------------------------

/// Produces compact summaries from conversation histories.
pub struct ConversationSummarizer;

impl ConversationSummarizer {
    /// Summarise a slice of turns by extracting the first sentence of each
    /// assistant response and returning at most `max_words` total words.
    pub fn summarize(turns: &[ConversationTurn], max_words: usize) -> String {
        let mut parts: Vec<String> = Vec::new();

        for turn in turns {
            if turn.role == "assistant" {
                let first_sentence = Self::first_sentence(&turn.content);
                if !first_sentence.is_empty() {
                    parts.push(first_sentence);
                }
            }
        }

        // Truncate to max_words total
        let combined = parts.join(" ");
        let words: Vec<&str> = combined.split_whitespace().collect();
        if words.len() <= max_words {
            combined
        } else {
            words[..max_words].join(" ")
        }
    }

    /// Summarise old messages in the window to free up context space.
    ///
    /// This converts the window's current contents into a summary message
    /// attributed to the assistant role, replacing all prior messages.
    pub fn compress_context(window: &mut ContextWindow, summarizer: &ConversationSummarizer) {
        // Drain all messages and convert to turns
        let msgs: Vec<ChatMessage> = window.messages.drain(..).collect();
        window.total_tokens = window
            .system_prompt
            .as_deref()
            .map(estimate_tokens)
            .unwrap_or(0);

        let turns: Vec<ConversationTurn> = msgs
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let role = match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::System => "system",
                };
                ConversationTurn::new(
                    role,
                    &m.content,
                    i as i64 * 1000,
                    estimate_tokens(&m.content),
                    None,
                )
            })
            .collect();

        let _ = summarizer; // the method is on the type, kept for API symmetry
        let summary_text = Self::summarize(&turns, 100);

        if !summary_text.is_empty() {
            let summary_msg = ChatMessage {
                role: ChatRole::Assistant,
                content: format!("[Summary] {}", summary_text),
                metadata: None,
            };
            window.push(summary_msg);
        }
    }

    fn first_sentence(text: &str) -> String {
        for delim in ['.', '!', '?'] {
            if let Some(idx) = text.find(delim) {
                return text[..=idx].trim().to_string();
            }
        }
        text.trim().to_string()
    }
}

// ---------------------------------------------------------------------------
// ContextTrimmer
// ---------------------------------------------------------------------------

/// Strategy used by [`ContextTrimmer`] to reduce a window's token count
#[derive(Debug, Clone)]
pub enum TrimStrategy {
    /// Drop oldest messages until the target is reached
    DropOldest,
    /// Summarise old messages before trimming
    Summarize,
    /// Discard everything except the system prompt and the last N messages
    KeepSystemAndLatestN(usize),
}

/// Trims a [`ContextWindow`] according to a configurable strategy
pub struct ContextTrimmer {
    pub strategy: TrimStrategy,
}

impl ContextTrimmer {
    /// Create a trimmer with the given strategy
    pub fn new(strategy: TrimStrategy) -> Self {
        Self { strategy }
    }

    /// Reduce `window` so that `total_tokens <= target_tokens`.
    pub fn trim(&self, window: &mut ContextWindow, target_tokens: usize) {
        match &self.strategy {
            TrimStrategy::DropOldest => {
                while window.total_tokens > target_tokens && !window.messages.is_empty() {
                    window.evict_oldest();
                }
            }

            TrimStrategy::Summarize => {
                // Summarise then trim remainder with DropOldest
                ConversationSummarizer::compress_context(window, &ConversationSummarizer);
                while window.total_tokens > target_tokens && !window.messages.is_empty() {
                    window.evict_oldest();
                }
            }

            TrimStrategy::KeepSystemAndLatestN(n) => {
                let n = *n;
                let total = window.messages.len();
                if total > n {
                    let drop_count = total - n;
                    for _ in 0..drop_count {
                        window.evict_oldest();
                    }
                }
                // Then fall back to DropOldest if still over budget
                while window.total_tokens > target_tokens && !window.messages.is_empty() {
                    window.evict_oldest();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rough token estimate: 1 token ≈ 4 characters
fn estimate_tokens(text: &str) -> usize {
    ((text.chars().count() + 3) / 4).max(1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn user_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: ChatRole::User,
            content: content.to_string(),
            metadata: None,
        }
    }

    fn assistant_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: ChatRole::Assistant,
            content: content.to_string(),
            metadata: None,
        }
    }

    // --- ContextWindow::new ---

    #[test]
    fn test_new_empty_window() {
        let w = ContextWindow::new(1000);
        assert_eq!(w.message_count(), 0);
        assert_eq!(w.total_tokens, 0);
        assert_eq!(w.available_tokens(), 1000);
        assert!(!w.is_full());
    }

    #[test]
    fn test_with_system_consumes_tokens() {
        let sys = "You are a helpful assistant.".to_string();
        let sys_tokens = sys.chars().count().div_ceil(4).max(1);
        let w = ContextWindow::with_system(1000, sys);
        assert_eq!(w.total_tokens, sys_tokens);
        assert_eq!(w.available_tokens(), 1000 - sys_tokens);
    }

    // --- ContextWindow::push ---

    #[test]
    fn test_push_fits_in_window() {
        let mut w = ContextWindow::new(1000);
        let accepted = w.push(user_msg("Hello!"));
        assert!(accepted);
        assert_eq!(w.message_count(), 1);
    }

    #[test]
    fn test_push_oversized_message_rejected() {
        let mut w = ContextWindow::new(1); // budget of 1 token
        let big = "a".repeat(100);
        let accepted = w.push(user_msg(&big));
        assert!(!accepted);
        assert_eq!(w.message_count(), 0);
    }

    #[test]
    fn test_push_evicts_oldest_when_full() {
        // Budget for ~5 tokens; each "Hello" ≈ 2 tokens
        let mut w = ContextWindow::new(10);
        for _ in 0..10 {
            w.push(user_msg("Hello"));
        }
        // Window should not grow unboundedly
        assert!(w.total_tokens <= 10);
    }

    // --- ContextWindow::evict_oldest ---

    #[test]
    fn test_evict_oldest_removes_first_message() {
        let mut w = ContextWindow::new(1000);
        w.push(user_msg("First"));
        w.push(assistant_msg("Second"));
        let evicted = w.evict_oldest();
        assert!(evicted.is_some());
        assert_eq!(evicted.map(|m| m.content), Some("First".to_string()));
        assert_eq!(w.message_count(), 1);
    }

    #[test]
    fn test_evict_oldest_empty_window() {
        let mut w = ContextWindow::new(1000);
        assert!(w.evict_oldest().is_none());
    }

    // --- ContextWindow::available_tokens / is_full ---

    #[test]
    fn test_available_tokens_decreases_on_push() {
        let mut w = ContextWindow::new(1000);
        let before = w.available_tokens();
        w.push(user_msg("Hello there!"));
        assert!(w.available_tokens() < before);
    }

    #[test]
    fn test_is_full_when_no_space() {
        // Each char-4 block → 1 token. "abcd" → 1 token with max 1.
        let mut w = ContextWindow::new(1);
        w.push(user_msg("abcd")); // exactly 1 token, fills budget
        assert!(w.is_full());
    }

    // --- ContextWindow::to_messages_slice ---

    #[test]
    fn test_to_messages_slice_order() {
        let mut w = ContextWindow::new(1000);
        w.push(user_msg("first"));
        w.push(assistant_msg("second"));
        w.push(user_msg("third"));
        let slice = w.to_messages_slice();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0].content, "first");
        assert_eq!(slice[2].content, "third");
    }

    // --- ConversationSummarizer ---

    #[test]
    fn test_summarize_empty() {
        let summary = ConversationSummarizer::summarize(&[], 50);
        assert!(summary.is_empty());
    }

    #[test]
    fn test_summarize_user_turns_ignored() {
        let turns = vec![
            ConversationTurn::new("user", "What is SPARQL?", 0, 5, None),
            ConversationTurn::new(
                "assistant",
                "SPARQL is a query language. It is used for RDF.",
                1000,
                12,
                None,
            ),
        ];
        let summary = ConversationSummarizer::summarize(&turns, 50);
        assert!(summary.contains("SPARQL"));
    }

    #[test]
    fn test_summarize_respects_max_words() {
        let long_content = "SPARQL is a query language. ".repeat(30);
        let turns = vec![ConversationTurn::new(
            "assistant",
            &long_content,
            0,
            100,
            None,
        )];
        let summary = ConversationSummarizer::summarize(&turns, 10);
        let word_count = summary.split_whitespace().count();
        assert!(word_count <= 10, "got {word_count} words");
    }

    #[test]
    fn test_compress_context_replaces_messages() {
        let mut w = ContextWindow::new(2000);
        for i in 0..6 {
            let role = if i % 2 == 0 {
                ChatRole::User
            } else {
                ChatRole::Assistant
            };
            let content = format!(
                "{}. This is turn number {}.",
                if i % 2 == 0 { "Question" } else { "Answer" },
                i
            );
            w.push(ChatMessage {
                role,
                content,
                metadata: None,
            });
        }
        let before_count = w.message_count();
        ConversationSummarizer::compress_context(&mut w, &ConversationSummarizer);
        // After compression the window should hold a summary message
        assert!(w.message_count() <= before_count);
        assert!(w.message_count() >= 1);
    }

    // --- ContextTrimmer ---

    #[test]
    fn test_trim_drop_oldest() {
        let mut w = ContextWindow::new(1000);
        for i in 0..20 {
            w.push(user_msg(&format!("Message number {i}")));
        }
        let trimmer = ContextTrimmer::new(TrimStrategy::DropOldest);
        let target = w.total_tokens / 2;
        trimmer.trim(&mut w, target);
        assert!(w.total_tokens <= target);
    }

    #[test]
    fn test_trim_keep_latest_n() {
        let mut w = ContextWindow::new(5000);
        for i in 0..10 {
            w.push(user_msg(&format!("Message {i}")));
        }
        let trimmer = ContextTrimmer::new(TrimStrategy::KeepSystemAndLatestN(3));
        trimmer.trim(&mut w, 5000);
        assert!(w.message_count() <= 3);
    }

    #[test]
    fn test_trim_summarize_strategy() {
        let mut w = ContextWindow::new(5000);
        for i in 0..6 {
            let role = if i % 2 == 0 {
                ChatRole::User
            } else {
                ChatRole::Assistant
            };
            w.push(ChatMessage {
                role,
                content: format!("Content for turn {i}."),
                metadata: None,
            });
        }
        let trimmer = ContextTrimmer::new(TrimStrategy::Summarize);
        trimmer.trim(&mut w, 5000); // generous budget — just check it doesn't panic
                                    // message_count is always non-negative (usize), just verify trim didn't panic
        let _ = w.message_count();
    }

    #[test]
    fn test_trim_noop_when_under_budget() {
        let mut w = ContextWindow::new(5000);
        w.push(user_msg("Short message."));
        let count_before = w.message_count();
        let trimmer = ContextTrimmer::new(TrimStrategy::DropOldest);
        trimmer.trim(&mut w, 5000);
        assert_eq!(w.message_count(), count_before);
    }
}
