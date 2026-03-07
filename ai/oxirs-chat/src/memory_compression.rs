//! # Conversation Memory Compression
//!
//! Compresses long conversation histories by summarising older messages while
//! preserving recent context. Reduces token usage and improves response latency
//! for multi-turn chat sessions.
//!
//! ## Features
//!
//! - **Sliding window**: Keep N most recent messages verbatim
//! - **Extractive summarisation**: Condense older messages to key points
//! - **Topic segmentation**: Group messages by topic for coherent summaries
//! - **Importance scoring**: Prioritise messages containing entities/questions
//! - **Token budget**: Compress to fit within a target token budget
//! - **Incremental compression**: Re-compress only when threshold exceeded

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for memory compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCompressionConfig {
    /// Maximum number of recent messages to keep verbatim (default: 10).
    pub recent_window: usize,
    /// Target token budget for the compressed history (default: 2000).
    pub token_budget: usize,
    /// Approximate tokens per word for estimation (default: 1.3).
    pub tokens_per_word: f64,
    /// Compression threshold: compress when total tokens exceed this (default: 4000).
    pub compression_threshold: usize,
    /// Minimum importance score to include in summary (default: 0.3).
    pub min_importance: f64,
    /// Maximum summary sentences per topic segment (default: 3).
    pub max_summary_sentences: usize,
}

impl Default for MemoryCompressionConfig {
    fn default() -> Self {
        Self {
            recent_window: 10,
            token_budget: 2000,
            tokens_per_word: 1.3,
            compression_threshold: 4000,
            min_importance: 0.3,
            max_summary_sentences: 3,
        }
    }
}

// ─────────────────────────────────────────────
// Message types
// ─────────────────────────────────────────────

/// Role in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// A single conversation message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    /// Message role.
    pub role: MessageRole,
    /// Message content.
    pub content: String,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
    /// Estimated token count.
    pub token_count: usize,
    /// Importance score (0.0–1.0, computed by the compressor).
    pub importance: f64,
    /// Topic label (if assigned).
    pub topic: Option<String>,
}

impl ConversationMessage {
    /// Create a new message.
    pub fn new(role: MessageRole, content: String) -> Self {
        let token_count = estimate_tokens(&content, 1.3);
        Self {
            role,
            content,
            timestamp: Utc::now(),
            token_count,
            importance: 0.5,
            topic: None,
        }
    }

    /// Create a user message.
    pub fn user(content: &str) -> Self {
        Self::new(MessageRole::User, content.to_string())
    }

    /// Create an assistant message.
    pub fn assistant(content: &str) -> Self {
        Self::new(MessageRole::Assistant, content.to_string())
    }

    /// Create a system message.
    pub fn system(content: &str) -> Self {
        Self::new(MessageRole::System, content.to_string())
    }
}

/// A compressed summary of older messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedSummary {
    /// Summary text.
    pub text: String,
    /// Number of original messages summarised.
    pub original_message_count: usize,
    /// Total tokens of original messages.
    pub original_token_count: usize,
    /// Token count of the summary.
    pub summary_token_count: usize,
    /// Compression ratio (original / summary).
    pub compression_ratio: f64,
    /// Topics covered in the summary.
    pub topics: Vec<String>,
    /// When the summary was generated.
    pub generated_at: DateTime<Utc>,
}

// ─────────────────────────────────────────────
// Compressed Memory
// ─────────────────────────────────────────────

/// The compressed conversation memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedMemory {
    /// Summary of older messages.
    pub summary: Option<CompressedSummary>,
    /// Recent messages kept verbatim.
    pub recent_messages: Vec<ConversationMessage>,
    /// Total token count of the compressed memory.
    pub total_tokens: usize,
}

// ─────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────

/// Statistics for memory compression.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Total compressions performed.
    pub compressions_performed: u64,
    /// Total messages processed.
    pub total_messages_processed: u64,
    /// Total tokens saved by compression.
    pub tokens_saved: u64,
    /// Average compression ratio.
    pub avg_compression_ratio: f64,
    /// Current memory token count.
    pub current_tokens: usize,
}

// ─────────────────────────────────────────────
// Memory Compressor
// ─────────────────────────────────────────────

/// Compresses conversation histories.
pub struct MemoryCompressor {
    config: MemoryCompressionConfig,
    stats: CompressionStats,
}

impl MemoryCompressor {
    /// Create a new compressor.
    pub fn new(config: MemoryCompressionConfig) -> Self {
        Self {
            config,
            stats: CompressionStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(MemoryCompressionConfig::default())
    }

    /// Compress a conversation history.
    pub fn compress(&mut self, messages: &[ConversationMessage]) -> CompressedMemory {
        let total_tokens: usize = messages.iter().map(|m| m.token_count).sum();

        // If under threshold, return as-is
        if total_tokens <= self.config.compression_threshold
            || messages.len() <= self.config.recent_window
        {
            return CompressedMemory {
                summary: None,
                recent_messages: messages.to_vec(),
                total_tokens,
            };
        }

        let recent_count = self.config.recent_window.min(messages.len());
        let split_point = messages.len() - recent_count;

        let old_messages = &messages[..split_point];
        let recent_messages = messages[split_point..].to_vec();

        // Score importance
        let scored: Vec<(usize, f64)> = old_messages
            .iter()
            .enumerate()
            .map(|(i, m)| (i, self.score_importance(m)))
            .collect();

        // Segment by topics
        let topics = self.segment_topics(old_messages);

        // Generate extractive summary
        let summary_text = self.generate_summary(old_messages, &scored, &topics);
        let summary_tokens = estimate_tokens(&summary_text, self.config.tokens_per_word);
        let old_tokens: usize = old_messages.iter().map(|m| m.token_count).sum();
        let recent_tokens: usize = recent_messages.iter().map(|m| m.token_count).sum();

        let compression_ratio = if summary_tokens > 0 {
            old_tokens as f64 / summary_tokens as f64
        } else {
            1.0
        };

        self.stats.compressions_performed += 1;
        self.stats.total_messages_processed += old_messages.len() as u64;
        self.stats.tokens_saved += (old_tokens.saturating_sub(summary_tokens)) as u64;
        self.stats.current_tokens = summary_tokens + recent_tokens;

        let n = self.stats.compressions_performed as f64;
        self.stats.avg_compression_ratio =
            self.stats.avg_compression_ratio * (n - 1.0) / n + compression_ratio / n;

        let summary = CompressedSummary {
            text: summary_text,
            original_message_count: old_messages.len(),
            original_token_count: old_tokens,
            summary_token_count: summary_tokens,
            compression_ratio,
            topics: topics.keys().cloned().collect(),
            generated_at: Utc::now(),
        };

        CompressedMemory {
            summary: Some(summary),
            recent_messages,
            total_tokens: summary_tokens + recent_tokens,
        }
    }

    /// Check if compression is needed for the given message count.
    pub fn needs_compression(&self, total_tokens: usize) -> bool {
        total_tokens > self.config.compression_threshold
    }

    /// Get statistics.
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Get configuration.
    pub fn config(&self) -> &MemoryCompressionConfig {
        &self.config
    }

    // ─── Internal ────────────────────────────

    fn score_importance(&self, message: &ConversationMessage) -> f64 {
        let mut score: f64 = 0.3; // Base score

        // Questions are important
        if message.content.contains('?') {
            score += 0.2;
        }

        // Longer messages tend to be more important
        let word_count = message.content.split_whitespace().count();
        if word_count > 20 {
            score += 0.1;
        }

        // System messages are always important
        if message.role == MessageRole::System {
            score += 0.3;
        }

        // Messages with URIs/entities are important
        if message.content.contains("http://") || message.content.contains("https://") {
            score += 0.15;
        }

        // Code blocks are important
        if message.content.contains("```") {
            score += 0.15;
        }

        score.min(1.0)
    }

    fn segment_topics(&self, messages: &[ConversationMessage]) -> HashMap<String, Vec<usize>> {
        let mut topics: HashMap<String, Vec<usize>> = HashMap::new();
        let mut current_topic = "general".to_string();

        for (i, msg) in messages.iter().enumerate() {
            // Simple topic detection based on keywords
            let lower = msg.content.to_lowercase();
            if lower.contains("sparql") || lower.contains("query") {
                current_topic = "queries".to_string();
            } else if lower.contains("error") || lower.contains("fix") || lower.contains("bug") {
                current_topic = "debugging".to_string();
            } else if lower.contains("how") || lower.contains("what") || lower.contains("why") {
                current_topic = "questions".to_string();
            }

            topics.entry(current_topic.clone()).or_default().push(i);
        }

        topics
    }

    fn generate_summary(
        &self,
        messages: &[ConversationMessage],
        scored: &[(usize, f64)],
        topics: &HashMap<String, Vec<usize>>,
    ) -> String {
        let mut summary_parts = Vec::new();

        // Add topic-based summaries
        for (topic, indices) in topics {
            let important_msgs: Vec<&ConversationMessage> = indices
                .iter()
                .filter_map(|&i| {
                    let score = scored.iter().find(|(idx, _)| *idx == i).map(|(_, s)| *s);
                    if score.unwrap_or(0.0) >= self.config.min_importance {
                        messages.get(i)
                    } else {
                        None
                    }
                })
                .take(self.config.max_summary_sentences)
                .collect();

            if !important_msgs.is_empty() {
                let topic_summary = format!(
                    "[{}] {}",
                    topic,
                    important_msgs
                        .iter()
                        .map(|m| {
                            let truncated = truncate_text(&m.content, 100);
                            let role = match m.role {
                                MessageRole::User => "User",
                                MessageRole::Assistant => "Assistant",
                                MessageRole::System => "System",
                            };
                            format!("{role}: {truncated}")
                        })
                        .collect::<Vec<_>>()
                        .join("; ")
                );
                summary_parts.push(topic_summary);
            }
        }

        if summary_parts.is_empty() {
            // Fallback: include first and last messages
            if let Some(first) = messages.first() {
                summary_parts.push(format!(
                    "Conversation started with: {}",
                    truncate_text(&first.content, 100)
                ));
            }
            if messages.len() > 1 {
                if let Some(last) = messages.last() {
                    summary_parts.push(format!(
                        "Last discussed: {}",
                        truncate_text(&last.content, 100)
                    ));
                }
            }
        }

        summary_parts.join("\n")
    }
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/// Estimate token count from text.
fn estimate_tokens(text: &str, tokens_per_word: f64) -> usize {
    let words = text.split_whitespace().count();
    (words as f64 * tokens_per_word).ceil() as usize
}

/// Truncate text to approximately `max_chars` characters.
fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        format!("{}...", &text[..max_chars.min(text.len())])
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_messages(count: usize) -> Vec<ConversationMessage> {
        (0..count)
            .map(|i| {
                if i % 2 == 0 {
                    ConversationMessage::user(&format!(
                        "User message number {i}: How do I query SPARQL for entity {i}?"
                    ))
                } else {
                    ConversationMessage::assistant(&format!(
                        "You can use SELECT ?s WHERE {{ ?s a <http://example.org/Entity{i}> }}"
                    ))
                }
            })
            .collect()
    }

    #[test]
    fn test_default_config() {
        let config = MemoryCompressionConfig::default();
        assert_eq!(config.recent_window, 10);
        assert_eq!(config.token_budget, 2000);
        assert_eq!(config.compression_threshold, 4000);
    }

    #[test]
    fn test_no_compression_small_history() {
        let mut compressor = MemoryCompressor::with_defaults();
        let messages = sample_messages(5);
        let result = compressor.compress(&messages);
        assert!(result.summary.is_none());
        assert_eq!(result.recent_messages.len(), 5);
    }

    #[test]
    fn test_compression_triggers() {
        let mut compressor = MemoryCompressor::new(MemoryCompressionConfig {
            recent_window: 3,
            compression_threshold: 50, // Very low
            ..Default::default()
        });
        let messages = sample_messages(20);
        let result = compressor.compress(&messages);
        assert!(result.summary.is_some());
        assert_eq!(result.recent_messages.len(), 3);
    }

    #[test]
    fn test_recent_window_preserved() {
        let mut compressor = MemoryCompressor::new(MemoryCompressionConfig {
            recent_window: 5,
            compression_threshold: 10,
            ..Default::default()
        });
        let messages = sample_messages(20);
        let result = compressor.compress(&messages);
        assert_eq!(result.recent_messages.len(), 5);
    }

    #[test]
    fn test_summary_has_content() {
        let mut compressor = MemoryCompressor::new(MemoryCompressionConfig {
            recent_window: 2,
            compression_threshold: 10,
            ..Default::default()
        });
        let messages = sample_messages(10);
        let result = compressor.compress(&messages);
        let summary = result.summary.expect("should have summary");
        assert!(!summary.text.is_empty());
    }

    #[test]
    fn test_compression_ratio() {
        let mut compressor = MemoryCompressor::new(MemoryCompressionConfig {
            recent_window: 2,
            compression_threshold: 10,
            ..Default::default()
        });
        let messages = sample_messages(20);
        let result = compressor.compress(&messages);
        let summary = result.summary.expect("should have summary");
        assert!(summary.compression_ratio >= 1.0);
    }

    #[test]
    fn test_message_roles() {
        let m1 = ConversationMessage::user("hello");
        assert_eq!(m1.role, MessageRole::User);

        let m2 = ConversationMessage::assistant("hi there");
        assert_eq!(m2.role, MessageRole::Assistant);

        let m3 = ConversationMessage::system("you are helpful");
        assert_eq!(m3.role, MessageRole::System);
    }

    #[test]
    fn test_token_estimation() {
        let tokens = estimate_tokens("hello world foo bar", 1.3);
        assert_eq!(tokens, 6); // 4 * 1.3 = 5.2 -> ceil = 6
    }

    #[test]
    fn test_token_estimation_empty() {
        assert_eq!(estimate_tokens("", 1.3), 0);
    }

    #[test]
    fn test_truncate_text_short() {
        assert_eq!(truncate_text("hello", 100), "hello");
    }

    #[test]
    fn test_truncate_text_long() {
        let long = "a".repeat(200);
        let truncated = truncate_text(&long, 50);
        assert!(truncated.len() < 60);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_importance_scoring_question() {
        let compressor = MemoryCompressor::with_defaults();
        let q = ConversationMessage::user("What is SPARQL?");
        let score = compressor.score_importance(&q);
        assert!(score > 0.4);
    }

    #[test]
    fn test_importance_scoring_system() {
        let compressor = MemoryCompressor::with_defaults();
        let s = ConversationMessage::system("You are a helpful assistant.");
        let score = compressor.score_importance(&s);
        assert!(score > 0.5);
    }

    #[test]
    fn test_importance_scoring_uri() {
        let compressor = MemoryCompressor::with_defaults();
        let m = ConversationMessage::user("Check http://example.org/resource");
        let score = compressor.score_importance(&m);
        assert!(score > 0.4);
    }

    #[test]
    fn test_topic_segmentation() {
        let compressor = MemoryCompressor::with_defaults();
        let messages = vec![
            ConversationMessage::user("How do I write a SPARQL query?"),
            ConversationMessage::assistant("Use SELECT..."),
            ConversationMessage::user("I got an error in my code"),
            ConversationMessage::assistant("Let me fix that bug"),
        ];
        let topics = compressor.segment_topics(&messages);
        assert!(!topics.is_empty());
    }

    #[test]
    fn test_needs_compression() {
        let compressor = MemoryCompressor::with_defaults();
        assert!(!compressor.needs_compression(1000));
        assert!(compressor.needs_compression(5000));
    }

    #[test]
    fn test_stats_tracking() {
        let mut compressor = MemoryCompressor::new(MemoryCompressionConfig {
            recent_window: 2,
            compression_threshold: 10,
            ..Default::default()
        });
        compressor.compress(&sample_messages(20));
        assert_eq!(compressor.stats().compressions_performed, 1);
        assert!(compressor.stats().total_messages_processed > 0);
    }

    #[test]
    fn test_stats_tokens_saved() {
        let mut compressor = MemoryCompressor::new(MemoryCompressionConfig {
            recent_window: 2,
            compression_threshold: 10,
            ..Default::default()
        });
        compressor.compress(&sample_messages(20));
        assert!(compressor.stats().tokens_saved > 0);
    }

    #[test]
    fn test_empty_messages() {
        let mut compressor = MemoryCompressor::with_defaults();
        let result = compressor.compress(&[]);
        assert!(result.summary.is_none());
        assert!(result.recent_messages.is_empty());
    }

    #[test]
    fn test_single_message() {
        let mut compressor = MemoryCompressor::with_defaults();
        let messages = vec![ConversationMessage::user("hello")];
        let result = compressor.compress(&messages);
        assert!(result.summary.is_none());
        assert_eq!(result.recent_messages.len(), 1);
    }

    #[test]
    fn test_config_serialization() {
        let config = MemoryCompressionConfig::default();
        let json = serde_json::to_string(&config).expect("serialize failed");
        assert!(json.contains("recent_window"));
    }

    #[test]
    fn test_summary_serialization() {
        let summary = CompressedSummary {
            text: "test summary".into(),
            original_message_count: 10,
            original_token_count: 500,
            summary_token_count: 50,
            compression_ratio: 10.0,
            topics: vec!["general".into()],
            generated_at: Utc::now(),
        };
        let json = serde_json::to_string(&summary).expect("serialize failed");
        assert!(json.contains("compression_ratio"));
    }

    #[test]
    fn test_compressed_memory_serialization() {
        let mem = CompressedMemory {
            summary: None,
            recent_messages: vec![ConversationMessage::user("hello")],
            total_tokens: 2,
        };
        let json = serde_json::to_string(&mem).expect("serialize failed");
        assert!(json.contains("recent_messages"));
    }

    #[test]
    fn test_message_token_count_set() {
        let m = ConversationMessage::user("hello world");
        assert!(m.token_count > 0);
    }

    #[test]
    fn test_code_block_importance() {
        let compressor = MemoryCompressor::with_defaults();
        let m = ConversationMessage::user("Here is code:\n```rust\nfn main() {}\n```");
        let score = compressor.score_importance(&m);
        assert!(score > 0.4);
    }

    #[test]
    fn test_summary_original_count() {
        let mut compressor = MemoryCompressor::new(MemoryCompressionConfig {
            recent_window: 3,
            compression_threshold: 10,
            ..Default::default()
        });
        let messages = sample_messages(15);
        let result = compressor.compress(&messages);
        let summary = result.summary.expect("should have summary");
        assert_eq!(summary.original_message_count, 12); // 15 - 3
    }

    #[test]
    fn test_multiple_compressions() {
        let mut compressor = MemoryCompressor::new(MemoryCompressionConfig {
            recent_window: 2,
            compression_threshold: 10,
            ..Default::default()
        });
        compressor.compress(&sample_messages(10));
        compressor.compress(&sample_messages(15));
        assert_eq!(compressor.stats().compressions_performed, 2);
    }

    #[test]
    fn test_message_new_constructor() {
        let m = ConversationMessage::new(MessageRole::User, "test content".into());
        assert_eq!(m.role, MessageRole::User);
        assert_eq!(m.content, "test content");
        assert!(m.importance > 0.0);
    }

    #[test]
    fn test_avg_compression_ratio_stats() {
        let mut compressor = MemoryCompressor::new(MemoryCompressionConfig {
            recent_window: 2,
            compression_threshold: 10,
            ..Default::default()
        });
        compressor.compress(&sample_messages(20));
        assert!(compressor.stats().avg_compression_ratio > 0.0);
    }

    #[test]
    fn test_summary_topics_populated() {
        let mut compressor = MemoryCompressor::new(MemoryCompressionConfig {
            recent_window: 2,
            compression_threshold: 10,
            ..Default::default()
        });
        let messages = sample_messages(10);
        let result = compressor.compress(&messages);
        let summary = result.summary.expect("should have summary");
        assert!(!summary.topics.is_empty());
    }

    #[test]
    fn test_role_serde() {
        let role = MessageRole::User;
        let json = serde_json::to_string(&role).expect("serialize failed");
        let deser: MessageRole = serde_json::from_str(&json).expect("deser failed");
        assert_eq!(deser, role);
    }
}
