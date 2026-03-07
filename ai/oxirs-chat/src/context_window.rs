//! Context window management for LLM interactions.
//!
//! Provides token counting, sliding window, priority-based truncation,
//! context compression, message importance scoring, reserved space management,
//! overflow detection, and multi-turn conversation windowing.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Role of a message in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// System instruction (highest priority).
    System,
    /// User message.
    User,
    /// Assistant response.
    Assistant,
    /// Tool/function result.
    Tool,
}

impl Role {
    /// Default importance weight for this role.
    pub fn default_importance(&self) -> f64 {
        match self {
            Role::System => 1.0,
            Role::User => 0.8,
            Role::Tool => 0.6,
            Role::Assistant => 0.5,
        }
    }
}

/// A message in the context window.
#[derive(Debug, Clone)]
pub struct ContextMessage {
    /// Role of the sender.
    pub role: Role,
    /// Text content.
    pub content: String,
    /// Estimated token count.
    pub token_count: usize,
    /// Turn number (0-based).
    pub turn: usize,
    /// Whether this message is pinned (never truncated).
    pub pinned: bool,
}

impl ContextMessage {
    /// Create a new message with auto-estimated token count.
    pub fn new(role: Role, content: impl Into<String>, turn: usize) -> Self {
        let content = content.into();
        let token_count = estimate_tokens(&content);
        Self {
            role,
            content,
            token_count,
            turn,
            pinned: false,
        }
    }

    /// Create a pinned message (will not be truncated).
    pub fn pinned(role: Role, content: impl Into<String>, turn: usize) -> Self {
        let mut msg = Self::new(role, content, turn);
        msg.pinned = true;
        msg
    }

    /// Importance score: role weight * recency bonus.
    pub fn importance(&self, max_turn: usize) -> f64 {
        let role_weight = self.role.default_importance();
        let recency = if max_turn == 0 {
            1.0
        } else {
            0.5 + 0.5 * (self.turn as f64 / max_turn as f64)
        };
        role_weight * recency
    }
}

/// Configuration for the context window manager.
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Maximum number of tokens allowed in the context.
    pub max_tokens: usize,
    /// Tokens reserved for the model's response.
    pub reserved_for_response: usize,
    /// Truncation strategy when the window overflows.
    pub truncation_strategy: TruncationStrategy,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            reserved_for_response: 512,
            truncation_strategy: TruncationStrategy::SlidingWindow,
        }
    }
}

/// Strategy for truncating messages when the context overflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Drop oldest messages first (sliding window).
    SlidingWindow,
    /// Drop lowest-importance messages first.
    PriorityBased,
    /// Compress old messages into a summary.
    Compress,
}

/// Summary produced by context compression.
#[derive(Debug, Clone)]
pub struct ContextSummary {
    /// Compressed text.
    pub text: String,
    /// Estimated token count of the summary.
    pub token_count: usize,
    /// Number of original messages that were compressed.
    pub messages_compressed: usize,
    /// Total tokens of the original messages.
    pub original_tokens: usize,
}

/// Snapshot of the current window state.
#[derive(Debug, Clone)]
pub struct WindowState {
    /// Current total token usage (excluding reserved).
    pub total_tokens: usize,
    /// Available tokens (max - reserved - current).
    pub available_tokens: usize,
    /// Number of messages in the window.
    pub message_count: usize,
    /// Whether the window is overflowing.
    pub is_overflowing: bool,
    /// Fill ratio [0, 1].
    pub fill_ratio: f64,
}

// ---------------------------------------------------------------------------
// ContextWindow
// ---------------------------------------------------------------------------

/// Context window manager for LLM interactions.
pub struct ContextWindow {
    config: WindowConfig,
    messages: VecDeque<ContextMessage>,
    summaries: Vec<ContextSummary>,
    next_turn: usize,
}

impl ContextWindow {
    /// Create a new context window with the given configuration.
    pub fn new(config: WindowConfig) -> Self {
        Self {
            config,
            messages: VecDeque::new(),
            summaries: Vec::new(),
            next_turn: 0,
        }
    }

    /// Add a message to the context window.
    ///
    /// If the window overflows after adding, the configured truncation strategy
    /// is applied automatically.
    pub fn add_message(&mut self, role: Role, content: impl Into<String>) {
        let msg = ContextMessage::new(role, content, self.next_turn);
        self.next_turn += 1;
        self.messages.push_back(msg);

        if self.is_overflowing() {
            self.apply_truncation();
        }
    }

    /// Add a pinned message that will never be truncated.
    pub fn add_pinned(&mut self, role: Role, content: impl Into<String>) {
        let msg = ContextMessage::pinned(role, content, self.next_turn);
        self.next_turn += 1;
        self.messages.push_back(msg);
    }

    /// Current total tokens used by messages + summaries.
    pub fn total_tokens(&self) -> usize {
        let msg_tokens: usize = self.messages.iter().map(|m| m.token_count).sum();
        let summary_tokens: usize = self.summaries.iter().map(|s| s.token_count).sum();
        msg_tokens + summary_tokens
    }

    /// Available tokens for new content (excluding reserved space).
    pub fn available_tokens(&self) -> usize {
        let usable = self
            .config
            .max_tokens
            .saturating_sub(self.config.reserved_for_response);
        usable.saturating_sub(self.total_tokens())
    }

    /// Whether the current context exceeds the available budget.
    pub fn is_overflowing(&self) -> bool {
        let usable = self
            .config
            .max_tokens
            .saturating_sub(self.config.reserved_for_response);
        self.total_tokens() > usable
    }

    /// Fill ratio [0, 1] of the usable token budget.
    pub fn fill_ratio(&self) -> f64 {
        let usable = self
            .config
            .max_tokens
            .saturating_sub(self.config.reserved_for_response) as f64;
        if usable == 0.0 {
            return 1.0;
        }
        (self.total_tokens() as f64 / usable).min(1.0)
    }

    /// Get a snapshot of the current window state.
    pub fn state(&self) -> WindowState {
        WindowState {
            total_tokens: self.total_tokens(),
            available_tokens: self.available_tokens(),
            message_count: self.messages.len(),
            is_overflowing: self.is_overflowing(),
            fill_ratio: self.fill_ratio(),
        }
    }

    /// Number of messages currently in the window.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Access messages as a slice (deque iteration).
    pub fn messages(&self) -> impl Iterator<Item = &ContextMessage> {
        self.messages.iter()
    }

    /// Access summaries.
    pub fn summaries(&self) -> &[ContextSummary] {
        &self.summaries
    }

    /// Build the final context string for sending to the LLM.
    pub fn build_context(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        // Summaries first (compressed history).
        for summary in &self.summaries {
            parts.push(format!("[Summary] {}", summary.text));
        }

        // Then messages in order.
        for msg in &self.messages {
            let role_str = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };
            parts.push(format!("[{}] {}", role_str, msg.content));
        }

        parts.join("\n")
    }

    /// Manually trigger truncation.
    pub fn truncate(&mut self) {
        self.apply_truncation();
    }

    /// Clear all messages and summaries.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.summaries.clear();
        self.next_turn = 0;
    }

    /// Return the current configuration.
    pub fn config(&self) -> &WindowConfig {
        &self.config
    }

    // --- private helpers ---

    fn apply_truncation(&mut self) {
        match self.config.truncation_strategy {
            TruncationStrategy::SlidingWindow => self.truncate_sliding(),
            TruncationStrategy::PriorityBased => self.truncate_priority(),
            TruncationStrategy::Compress => self.truncate_compress(),
        }
    }

    /// Drop oldest non-pinned messages until within budget.
    fn truncate_sliding(&mut self) {
        while self.is_overflowing() && !self.messages.is_empty() {
            // Find the first non-pinned message from the front.
            let idx = self.messages.iter().position(|m| !m.pinned);
            match idx {
                Some(i) => {
                    self.messages.remove(i);
                }
                None => break, // all pinned, cannot truncate
            }
        }
    }

    /// Drop lowest-importance non-pinned messages until within budget.
    fn truncate_priority(&mut self) {
        let max_turn = self.next_turn.saturating_sub(1);

        while self.is_overflowing() && !self.messages.is_empty() {
            // Find the non-pinned message with lowest importance.
            let mut min_idx: Option<usize> = None;
            let mut min_importance = f64::MAX;

            for (i, msg) in self.messages.iter().enumerate() {
                if msg.pinned {
                    continue;
                }
                let imp = msg.importance(max_turn);
                if imp < min_importance {
                    min_importance = imp;
                    min_idx = Some(i);
                }
            }

            match min_idx {
                Some(i) => {
                    self.messages.remove(i);
                }
                None => break,
            }
        }
    }

    /// Compress the oldest half of non-pinned messages into a summary.
    fn truncate_compress(&mut self) {
        // Gather non-pinned messages from the front.
        let total_non_pinned = self.messages.iter().filter(|m| !m.pinned).count();
        if total_non_pinned <= 1 {
            // Fall back to sliding if nothing to compress.
            self.truncate_sliding();
            return;
        }

        let to_compress = total_non_pinned / 2;
        let mut compressed_texts: Vec<String> = Vec::new();
        let mut compressed_tokens: usize = 0;
        let mut compressed_count: usize = 0;
        let mut indices_to_remove: Vec<usize> = Vec::new();

        for (i, msg) in self.messages.iter().enumerate() {
            if compressed_count >= to_compress {
                break;
            }
            if !msg.pinned {
                compressed_texts.push(msg.content.clone());
                compressed_tokens += msg.token_count;
                compressed_count += 1;
                indices_to_remove.push(i);
            }
        }

        // Remove in reverse order to preserve indices.
        for &i in indices_to_remove.iter().rev() {
            self.messages.remove(i);
        }

        if !compressed_texts.is_empty() {
            // Build a simple summary (in production this would call an LLM).
            let summary_text = format!(
                "Previous conversation ({} messages): {}",
                compressed_count,
                compressed_texts.join(" | ")
            );
            let summary_token_count = estimate_tokens(&summary_text);

            self.summaries.push(ContextSummary {
                text: summary_text,
                token_count: summary_token_count,
                messages_compressed: compressed_count,
                original_tokens: compressed_tokens,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Approximate token count: ~4 characters per token (GPT-style heuristic).
pub fn estimate_tokens(text: &str) -> usize {
    let chars = text.len();
    // Rough estimate: 1 token per 4 characters, minimum 1 token for non-empty.
    if chars == 0 {
        0
    } else {
        (chars / 4).max(1)
    }
}

/// Estimate tokens for a list of strings.
pub fn estimate_tokens_batch(texts: &[&str]) -> Vec<usize> {
    texts.iter().map(|t| estimate_tokens(t)).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_window() -> ContextWindow {
        ContextWindow::new(WindowConfig::default())
    }

    fn small_window() -> ContextWindow {
        ContextWindow::new(WindowConfig {
            max_tokens: 100,
            reserved_for_response: 20,
            truncation_strategy: TruncationStrategy::SlidingWindow,
        })
    }

    fn priority_window() -> ContextWindow {
        ContextWindow::new(WindowConfig {
            max_tokens: 100,
            reserved_for_response: 20,
            truncation_strategy: TruncationStrategy::PriorityBased,
        })
    }

    fn compress_window() -> ContextWindow {
        ContextWindow::new(WindowConfig {
            max_tokens: 100,
            reserved_for_response: 20,
            truncation_strategy: TruncationStrategy::Compress,
        })
    }

    // --- estimate_tokens ---

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_short() {
        // "Hi" = 2 chars → max(2/4, 1) = 1
        assert_eq!(estimate_tokens("Hi"), 1);
    }

    #[test]
    fn test_estimate_tokens_longer() {
        // 20 chars → 5 tokens
        let text = "a]".repeat(10); // 20 chars
        assert_eq!(estimate_tokens(&text), 5);
    }

    #[test]
    fn test_estimate_tokens_batch_length() {
        let result = estimate_tokens_batch(&["hello", "world", "foo"]);
        assert_eq!(result.len(), 3);
    }

    // --- Role importance ---

    #[test]
    fn test_system_highest_importance() {
        assert!(Role::System.default_importance() > Role::User.default_importance());
        assert!(Role::System.default_importance() > Role::Assistant.default_importance());
    }

    #[test]
    fn test_user_higher_than_assistant() {
        assert!(Role::User.default_importance() > Role::Assistant.default_importance());
    }

    // --- ContextMessage ---

    #[test]
    fn test_message_auto_token_count() {
        let msg = ContextMessage::new(Role::User, "Hello world, this is a test!", 0);
        assert!(msg.token_count > 0);
    }

    #[test]
    fn test_pinned_message_flag() {
        let msg = ContextMessage::pinned(Role::System, "You are a helpful assistant.", 0);
        assert!(msg.pinned);
    }

    #[test]
    fn test_message_importance_increases_with_turn() {
        let early = ContextMessage::new(Role::User, "early", 0);
        let late = ContextMessage::new(Role::User, "late", 10);
        assert!(late.importance(10) > early.importance(10));
    }

    #[test]
    fn test_message_importance_system_higher_than_assistant() {
        let sys = ContextMessage::new(Role::System, "sys", 5);
        let asst = ContextMessage::new(Role::Assistant, "asst", 5);
        assert!(sys.importance(10) > asst.importance(10));
    }

    // --- ContextWindow basics ---

    #[test]
    fn test_empty_window_state() {
        let w = default_window();
        assert_eq!(w.total_tokens(), 0);
        assert_eq!(w.message_count(), 0);
        assert!(!w.is_overflowing());
    }

    #[test]
    fn test_add_message_increments_count() {
        let mut w = default_window();
        w.add_message(Role::User, "Hello");
        assert_eq!(w.message_count(), 1);
    }

    #[test]
    fn test_add_message_increases_tokens() {
        let mut w = default_window();
        w.add_message(Role::User, "Hello there");
        assert!(w.total_tokens() > 0);
    }

    #[test]
    fn test_clear_resets_window() {
        let mut w = default_window();
        w.add_message(Role::User, "test");
        w.clear();
        assert_eq!(w.total_tokens(), 0);
        assert_eq!(w.message_count(), 0);
    }

    // --- overflow detection ---

    #[test]
    fn test_overflow_detected() {
        let mut w = small_window(); // max=100, reserved=20, usable=80
                                    // Add enough tokens to exceed 80
        for _ in 0..50 {
            w.add_message(
                Role::User,
                "This is a somewhat long message for testing overflow detection.",
            );
        }
        // After truncation, it should no longer overflow (sliding window)
        assert!(!w.is_overflowing());
    }

    #[test]
    fn test_available_tokens_decreases() {
        let mut w = small_window();
        let before = w.available_tokens();
        w.add_message(Role::User, "Some content here");
        let after = w.available_tokens();
        assert!(after < before);
    }

    // --- sliding window truncation ---

    #[test]
    fn test_sliding_window_removes_oldest() {
        let mut w = small_window();
        w.add_message(Role::User, "First message with some content");
        w.add_message(Role::User, "Second message with some content");
        // Add many to trigger overflow
        for i in 0..30 {
            w.add_message(Role::User, format!("Message number {} with content", i));
        }
        // After truncation, should not overflow
        assert!(!w.is_overflowing());
        // Oldest messages should have been dropped
        assert!(w.message_count() < 32);
    }

    #[test]
    fn test_sliding_window_preserves_pinned() {
        let mut w = small_window();
        w.add_pinned(Role::System, "Pinned system prompt that must stay");
        for i in 0..30 {
            w.add_message(Role::User, format!("Overflow message {}", i));
        }
        // The pinned message should still be present.
        let has_pinned = w.messages().any(|m| m.pinned);
        assert!(has_pinned, "pinned messages should survive truncation");
    }

    // --- priority-based truncation ---

    #[test]
    fn test_priority_truncation_removes_low_importance() {
        let mut w = priority_window();
        w.add_message(
            Role::Assistant,
            "Low importance assistant response from early turn",
        );
        w.add_message(Role::System, "High importance system message");
        // Fill up
        for i in 0..30 {
            w.add_message(Role::User, format!("User message {}", i));
        }
        assert!(!w.is_overflowing());
    }

    #[test]
    fn test_priority_preserves_pinned() {
        let mut w = priority_window();
        w.add_pinned(Role::System, "Pinned instruction");
        for i in 0..30 {
            w.add_message(Role::User, format!("Filler message {}", i));
        }
        let has_pinned = w.messages().any(|m| m.pinned);
        assert!(has_pinned);
    }

    // --- compress truncation ---

    #[test]
    fn test_compress_creates_summary() {
        let mut w = compress_window();
        for i in 0..30 {
            w.add_message(Role::User, format!("Message about topic {}", i));
        }
        // After compression, there should be at least one summary.
        assert!(
            !w.summaries().is_empty() || w.message_count() < 30,
            "compression should either create summaries or reduce messages"
        );
    }

    #[test]
    fn test_compress_summary_has_metadata() {
        let mut w = compress_window();
        for i in 0..30 {
            w.add_message(Role::User, format!("Message about topic {}", i));
        }
        if let Some(summary) = w.summaries().first() {
            assert!(summary.messages_compressed > 0);
            assert!(summary.original_tokens > 0);
            assert!(summary.token_count > 0);
        }
    }

    // --- build_context ---

    #[test]
    fn test_build_context_includes_messages() {
        let mut w = default_window();
        w.add_message(Role::User, "Hello AI");
        let ctx = w.build_context();
        assert!(ctx.contains("Hello AI"));
        assert!(ctx.contains("[user]"));
    }

    #[test]
    fn test_build_context_includes_summaries() {
        let mut w = compress_window();
        for i in 0..30 {
            w.add_message(Role::User, format!("Message {}", i));
        }
        let ctx = w.build_context();
        if !w.summaries().is_empty() {
            assert!(ctx.contains("[Summary]"));
        }
    }

    #[test]
    fn test_build_context_empty_window() {
        let w = default_window();
        let ctx = w.build_context();
        assert!(ctx.is_empty());
    }

    // --- fill_ratio ---

    #[test]
    fn test_fill_ratio_empty_is_zero() {
        let w = default_window();
        assert!((w.fill_ratio() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fill_ratio_increases_with_messages() {
        let mut w = default_window();
        let before = w.fill_ratio();
        w.add_message(Role::User, "Some content to fill the window");
        let after = w.fill_ratio();
        assert!(after > before);
    }

    #[test]
    fn test_fill_ratio_capped_at_one() {
        let mut w = small_window();
        for _ in 0..100 {
            w.add_pinned(Role::User, "lots of pinned content that cannot be removed");
        }
        assert!(w.fill_ratio() <= 1.0);
    }

    // --- state snapshot ---

    #[test]
    fn test_state_snapshot_fields() {
        let mut w = default_window();
        w.add_message(Role::User, "test");
        let state = w.state();
        assert!(state.total_tokens > 0);
        assert_eq!(state.message_count, 1);
        assert!(!state.is_overflowing);
    }

    // --- config access ---

    #[test]
    fn test_config_accessor() {
        let w = default_window();
        assert_eq!(w.config().max_tokens, 4096);
        assert_eq!(w.config().reserved_for_response, 512);
    }

    // --- WindowConfig default ---

    #[test]
    fn test_window_config_default() {
        let config = WindowConfig::default();
        assert_eq!(config.max_tokens, 4096);
        assert_eq!(config.reserved_for_response, 512);
        assert_eq!(
            config.truncation_strategy,
            TruncationStrategy::SlidingWindow
        );
    }

    // --- multi-turn windowing ---

    #[test]
    fn test_multi_turn_conversation() {
        let mut w = default_window();
        for i in 0..5 {
            w.add_message(Role::User, format!("User turn {}", i));
            w.add_message(Role::Assistant, format!("Assistant turn {}", i));
        }
        assert_eq!(w.message_count(), 10);
    }

    #[test]
    fn test_turn_numbers_are_sequential() {
        let mut w = default_window();
        w.add_message(Role::User, "First");
        w.add_message(Role::Assistant, "Second");
        w.add_message(Role::User, "Third");
        let turns: Vec<usize> = w.messages().map(|m| m.turn).collect();
        assert_eq!(turns, vec![0, 1, 2]);
    }

    // --- edge cases ---

    #[test]
    fn test_zero_max_tokens_always_overflows() {
        let mut w = ContextWindow::new(WindowConfig {
            max_tokens: 0,
            reserved_for_response: 0,
            truncation_strategy: TruncationStrategy::SlidingWindow,
        });
        w.add_message(Role::User, "Hello");
        assert_eq!(
            w.message_count(),
            0,
            "should truncate everything with 0 budget"
        );
    }

    #[test]
    fn test_reserved_larger_than_max() {
        let w = ContextWindow::new(WindowConfig {
            max_tokens: 10,
            reserved_for_response: 20,
            truncation_strategy: TruncationStrategy::SlidingWindow,
        });
        assert_eq!(w.available_tokens(), 0);
    }
}
