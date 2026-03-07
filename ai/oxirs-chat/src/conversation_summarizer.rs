//! # Conversation Summarizer
//!
//! Provides extractive and abstractive summarization of chat conversation histories
//! with key topic extraction and configurable summarization strategies.
//!
//! ## Features
//!
//! - **Extractive summarization**: Select the most important messages by scoring
//! - **Abstractive summarization**: Generate compressed summaries from message content
//! - **Topic extraction**: Identify key topics using TF-IDF-like scoring
//! - **Sliding window summaries**: Summarize fixed-size conversation windows
//! - **Incremental updates**: Update summaries as new messages arrive

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the conversation summarizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizerConfig {
    /// Maximum number of sentences to include in extractive summary (default: 5).
    pub max_extractive_sentences: usize,
    /// Maximum length (chars) for abstractive summary (default: 500).
    pub max_abstractive_length: usize,
    /// Maximum number of topics to extract (default: 10).
    pub max_topics: usize,
    /// Minimum word frequency to be considered a topic (default: 2).
    pub min_topic_frequency: usize,
    /// Sliding window size for incremental summaries (default: 20 messages).
    pub window_size: usize,
    /// Stop words to exclude from topic extraction.
    pub stop_words: Vec<String>,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self {
            max_extractive_sentences: 5,
            max_abstractive_length: 500,
            max_topics: 10,
            min_topic_frequency: 2,
            window_size: 20,
            stop_words: default_stop_words(),
        }
    }
}

fn default_stop_words() -> Vec<String> {
    [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "it", "its", "this", "that", "these", "those", "i", "you", "he", "she", "we",
        "they", "me", "him", "her", "us", "them", "my", "your", "his", "our", "their", "and",
        "but", "or", "not", "no", "if", "then", "so", "what", "which", "who", "when", "where",
        "how", "all", "each", "every", "both", "few", "more", "most", "other", "some", "such",
        "than", "too", "very", "just", "about",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

// ─────────────────────────────────────────────
// Message types
// ─────────────────────────────────────────────

/// Role of the message sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
    System,
}

/// A single conversation message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    /// Unique message ID.
    pub id: String,
    /// Sender role.
    pub role: Role,
    /// Message content.
    pub content: String,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
}

impl ConversationMessage {
    pub fn new(id: impl Into<String>, role: Role, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role,
            content: content.into(),
            timestamp: Utc::now(),
        }
    }

    /// Word count of this message.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

// ─────────────────────────────────────────────
// Summary types
// ─────────────────────────────────────────────

/// An extracted topic with its score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    /// The topic term.
    pub term: String,
    /// Relevance score (higher = more relevant).
    pub score: f64,
    /// Number of occurrences.
    pub frequency: usize,
}

/// Result of summarization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    /// Extractive summary: selected key messages.
    pub extractive: Vec<ScoredMessage>,
    /// Abstractive summary: compressed text.
    pub abstractive: String,
    /// Extracted topics.
    pub topics: Vec<Topic>,
    /// Number of messages summarized.
    pub message_count: usize,
    /// Total word count of the conversation.
    pub total_words: usize,
    /// Compression ratio (summary length / original length).
    pub compression_ratio: f64,
    /// When the summary was generated.
    pub generated_at: DateTime<Utc>,
}

/// A message with its importance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredMessage {
    /// Original message ID.
    pub message_id: String,
    /// The message content.
    pub content: String,
    /// Role.
    pub role: Role,
    /// Importance score (0.0 - 1.0).
    pub score: f64,
}

/// Statistics about the summarization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SummarizerStats {
    /// Number of conversations summarized.
    pub summaries_generated: u64,
    /// Total messages processed.
    pub messages_processed: u64,
    /// Total topics extracted.
    pub topics_extracted: u64,
}

// ─────────────────────────────────────────────
// ConversationSummarizer
// ─────────────────────────────────────────────

/// Summarizes chat conversation histories.
pub struct ConversationSummarizer {
    config: SummarizerConfig,
    stats: SummarizerStats,
}

impl ConversationSummarizer {
    /// Create a new summarizer with default configuration.
    pub fn new() -> Self {
        Self {
            config: SummarizerConfig::default(),
            stats: SummarizerStats::default(),
        }
    }

    /// Create a new summarizer with the given configuration.
    pub fn with_config(config: SummarizerConfig) -> Self {
        Self {
            config,
            stats: SummarizerStats::default(),
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> &SummarizerStats {
        &self.stats
    }

    /// Summarize a conversation.
    pub fn summarize(&mut self, messages: &[ConversationMessage]) -> ConversationSummary {
        self.stats.summaries_generated += 1;
        self.stats.messages_processed += messages.len() as u64;

        let total_words: usize = messages.iter().map(|m| m.word_count()).sum();

        // Extract topics
        let topics = self.extract_topics(messages);
        self.stats.topics_extracted += topics.len() as u64;

        // Score messages for extractive summary
        let scored = self.score_messages(messages, &topics);
        let extractive: Vec<ScoredMessage> = scored
            .into_iter()
            .take(self.config.max_extractive_sentences)
            .collect();

        // Generate abstractive summary
        let abstractive = self.generate_abstractive(messages, &topics);

        let summary_words = abstractive.split_whitespace().count()
            + extractive
                .iter()
                .map(|m| m.content.split_whitespace().count())
                .sum::<usize>();
        let compression_ratio = if total_words > 0 {
            summary_words as f64 / total_words as f64
        } else {
            0.0
        };

        ConversationSummary {
            extractive,
            abstractive,
            topics,
            message_count: messages.len(),
            total_words,
            compression_ratio,
            generated_at: Utc::now(),
        }
    }

    /// Summarize only the last N messages (sliding window).
    pub fn summarize_window(&mut self, messages: &[ConversationMessage]) -> ConversationSummary {
        let window_size = self.config.window_size;
        let start = messages.len().saturating_sub(window_size);
        self.summarize(&messages[start..])
    }

    /// Extract topics from conversation messages.
    pub fn extract_topics(&self, messages: &[ConversationMessage]) -> Vec<Topic> {
        let stop_words: std::collections::HashSet<&str> =
            self.config.stop_words.iter().map(|s| s.as_str()).collect();

        // Count word frequencies
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        let mut doc_freq: HashMap<String, usize> = HashMap::new();

        for msg in messages {
            let words: Vec<String> = msg
                .content
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .filter(|w| w.len() > 2 && !stop_words.contains(w.as_str()))
                .collect();

            let unique_words: std::collections::HashSet<&str> =
                words.iter().map(|s| s.as_str()).collect();

            for word in &words {
                *word_freq.entry(word.clone()).or_insert(0) += 1;
            }
            for word in unique_words {
                *doc_freq.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        let num_docs = messages.len().max(1) as f64;

        // Compute TF-IDF-like scores
        let mut topics: Vec<Topic> = word_freq
            .iter()
            .filter(|(_, &freq)| freq >= self.config.min_topic_frequency)
            .map(|(term, &freq)| {
                let df = doc_freq.get(term).copied().unwrap_or(1) as f64;
                let idf = (num_docs / df).ln() + 1.0;
                let tf = freq as f64;
                let score = tf * idf;
                Topic {
                    term: term.clone(),
                    score,
                    frequency: freq,
                }
            })
            .collect();

        topics.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        topics.truncate(self.config.max_topics);
        topics
    }

    /// Score messages by importance.
    fn score_messages(
        &self,
        messages: &[ConversationMessage],
        topics: &[Topic],
    ) -> Vec<ScoredMessage> {
        let topic_terms: HashMap<&str, f64> =
            topics.iter().map(|t| (t.term.as_str(), t.score)).collect();

        let mut scored: Vec<ScoredMessage> = messages
            .iter()
            .map(|msg| {
                let mut score = 0.0;

                // Score based on topic term overlap
                let words: Vec<String> = msg
                    .content
                    .split_whitespace()
                    .map(|w| w.to_lowercase())
                    .collect();

                for word in &words {
                    if let Some(&topic_score) = topic_terms.get(word.as_str()) {
                        score += topic_score;
                    }
                }

                // Normalize by message length
                let word_count = words.len().max(1) as f64;
                score /= word_count;

                // Boost questions (user messages containing '?')
                if msg.role == Role::User && msg.content.contains('?') {
                    score *= 1.5;
                }

                // Boost longer messages (more informative)
                if word_count > 10.0 {
                    score *= 1.2;
                }

                ScoredMessage {
                    message_id: msg.id.clone(),
                    content: msg.content.clone(),
                    role: msg.role,
                    score,
                }
            })
            .collect();

        // Normalize scores to [0, 1]
        let max_score = scored.iter().map(|s| s.score).fold(0.0f64, f64::max);
        if max_score > 0.0 {
            for s in &mut scored {
                s.score /= max_score;
            }
        }

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored
    }

    /// Generate an abstractive summary.
    fn generate_abstractive(&self, messages: &[ConversationMessage], topics: &[Topic]) -> String {
        if messages.is_empty() {
            return String::new();
        }

        let mut summary_parts = Vec::new();

        // Overview
        let user_msgs = messages.iter().filter(|m| m.role == Role::User).count();
        let asst_msgs = messages
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .count();
        summary_parts.push(format!(
            "Conversation with {} messages ({} user, {} assistant).",
            messages.len(),
            user_msgs,
            asst_msgs,
        ));

        // Topic summary
        if !topics.is_empty() {
            let top_topics: Vec<&str> = topics.iter().take(5).map(|t| t.term.as_str()).collect();
            summary_parts.push(format!("Key topics: {}.", top_topics.join(", ")));
        }

        // Key questions asked
        let questions: Vec<&str> = messages
            .iter()
            .filter(|m| m.role == Role::User && m.content.contains('?'))
            .map(|m| m.content.as_str())
            .take(3)
            .collect();
        if !questions.is_empty() {
            summary_parts.push("Key questions discussed:".to_string());
            for q in questions {
                let truncated = if q.len() > 100 { &q[..100] } else { q };
                summary_parts.push(format!("- {truncated}"));
            }
        }

        let mut result = summary_parts.join(" ");
        if result.len() > self.config.max_abstractive_length {
            result.truncate(self.config.max_abstractive_length);
            // Find last complete word
            if let Some(last_space) = result.rfind(' ') {
                result.truncate(last_space);
            }
            result.push_str("...");
        }

        result
    }
}

impl Default for ConversationSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_conversation() -> Vec<ConversationMessage> {
        vec![
            ConversationMessage::new("1", Role::User, "What is SPARQL and how does it work?"),
            ConversationMessage::new(
                "2",
                Role::Assistant,
                "SPARQL is a query language for RDF data. It allows you to query knowledge graphs using triple patterns. SPARQL supports SELECT, CONSTRUCT, ASK, and DESCRIBE query forms.",
            ),
            ConversationMessage::new(
                "3",
                Role::User,
                "Can you explain how SPARQL triple patterns match against RDF triples in a graph?",
            ),
            ConversationMessage::new(
                "4",
                Role::Assistant,
                "Triple patterns in SPARQL consist of subject, predicate, and object positions. Each position can be a variable (prefixed with ?) or a concrete value. The SPARQL engine matches these patterns against the RDF graph.",
            ),
            ConversationMessage::new(
                "5",
                Role::User,
                "What about SPARQL federation with SERVICE keyword?",
            ),
            ConversationMessage::new(
                "6",
                Role::Assistant,
                "SPARQL federation uses the SERVICE keyword to query remote endpoints. This allows distributed queries across multiple SPARQL endpoints. The federated query engine sends subqueries to remote services and combines the results.",
            ),
            ConversationMessage::new(
                "7",
                Role::User,
                "How does OxiRS handle query optimization?",
            ),
            ConversationMessage::new(
                "8",
                Role::Assistant,
                "OxiRS uses a cost-based query optimizer that considers join ordering, index selection, and cardinality estimation. It supports adaptive query processing to handle changing data distributions.",
            ),
        ]
    }

    // ═══ Config tests ════════════════════════════════════

    #[test]
    fn test_default_config() {
        let config = SummarizerConfig::default();
        assert_eq!(config.max_extractive_sentences, 5);
        assert_eq!(config.max_topics, 10);
        assert!(!config.stop_words.is_empty());
    }

    #[test]
    fn test_custom_config() {
        let config = SummarizerConfig {
            max_extractive_sentences: 3,
            max_topics: 5,
            ..Default::default()
        };
        assert_eq!(config.max_extractive_sentences, 3);
    }

    // ═══ Message tests ═══════════════════════════════════

    #[test]
    fn test_message_creation() {
        let msg = ConversationMessage::new("1", Role::User, "Hello world");
        assert_eq!(msg.id, "1");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.word_count(), 2);
    }

    #[test]
    fn test_message_word_count() {
        let msg = ConversationMessage::new("1", Role::User, "one two three four five");
        assert_eq!(msg.word_count(), 5);
    }

    #[test]
    fn test_message_empty_content() {
        let msg = ConversationMessage::new("1", Role::User, "");
        assert_eq!(msg.word_count(), 0);
    }

    // ═══ Topic extraction tests ══════════════════════════

    #[test]
    fn test_extract_topics() {
        let summarizer = ConversationSummarizer::new();
        let messages = sample_conversation();
        let topics = summarizer.extract_topics(&messages);
        assert!(!topics.is_empty());
    }

    #[test]
    fn test_topics_contain_sparql() {
        let summarizer = ConversationSummarizer::new();
        let messages = sample_conversation();
        let topics = summarizer.extract_topics(&messages);
        let has_sparql = topics.iter().any(|t| t.term.contains("sparql"));
        assert!(has_sparql);
    }

    #[test]
    fn test_topics_bounded() {
        let config = SummarizerConfig {
            max_topics: 3,
            ..Default::default()
        };
        let summarizer = ConversationSummarizer::with_config(config);
        let messages = sample_conversation();
        let topics = summarizer.extract_topics(&messages);
        assert!(topics.len() <= 3);
    }

    #[test]
    fn test_topics_sorted_by_score() {
        let summarizer = ConversationSummarizer::new();
        let messages = sample_conversation();
        let topics = summarizer.extract_topics(&messages);
        for window in topics.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }

    #[test]
    fn test_extract_topics_empty() {
        let summarizer = ConversationSummarizer::new();
        let topics = summarizer.extract_topics(&[]);
        assert!(topics.is_empty());
    }

    // ═══ Summarization tests ═════════════════════════════

    #[test]
    fn test_summarize_basic() {
        let mut summarizer = ConversationSummarizer::new();
        let messages = sample_conversation();
        let summary = summarizer.summarize(&messages);
        assert_eq!(summary.message_count, 8);
        assert!(summary.total_words > 0);
        assert!(!summary.abstractive.is_empty());
    }

    #[test]
    fn test_summarize_extractive() {
        let mut summarizer = ConversationSummarizer::new();
        let messages = sample_conversation();
        let summary = summarizer.summarize(&messages);
        assert!(!summary.extractive.is_empty());
        assert!(summary.extractive.len() <= 5);
    }

    #[test]
    fn test_extractive_scores_normalized() {
        let mut summarizer = ConversationSummarizer::new();
        let messages = sample_conversation();
        let summary = summarizer.summarize(&messages);
        for msg in &summary.extractive {
            assert!(msg.score >= 0.0 && msg.score <= 1.0);
        }
    }

    #[test]
    fn test_extractive_sorted_by_score() {
        let mut summarizer = ConversationSummarizer::new();
        let messages = sample_conversation();
        let summary = summarizer.summarize(&messages);
        for window in summary.extractive.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }

    #[test]
    fn test_abstractive_length_bounded() {
        let config = SummarizerConfig {
            max_abstractive_length: 100,
            ..Default::default()
        };
        let mut summarizer = ConversationSummarizer::with_config(config);
        let messages = sample_conversation();
        let summary = summarizer.summarize(&messages);
        // Allow for "..." suffix
        assert!(summary.abstractive.len() <= 110);
    }

    #[test]
    fn test_compression_ratio() {
        let mut summarizer = ConversationSummarizer::new();
        let messages = sample_conversation();
        let summary = summarizer.summarize(&messages);
        assert!(summary.compression_ratio >= 0.0);
        assert!(summary.compression_ratio <= 2.0);
    }

    // ═══ Window summary tests ════════════════════════════

    #[test]
    fn test_summarize_window() {
        let config = SummarizerConfig {
            window_size: 4,
            ..Default::default()
        };
        let mut summarizer = ConversationSummarizer::with_config(config);
        let messages = sample_conversation();
        let summary = summarizer.summarize_window(&messages);
        // Should only summarize last 4 messages
        assert_eq!(summary.message_count, 4);
    }

    #[test]
    fn test_summarize_window_smaller_than_messages() {
        let config = SummarizerConfig {
            window_size: 2,
            ..Default::default()
        };
        let mut summarizer = ConversationSummarizer::with_config(config);
        let messages = sample_conversation();
        let summary = summarizer.summarize_window(&messages);
        assert_eq!(summary.message_count, 2);
    }

    // ═══ Empty conversation tests ════════════════════════

    #[test]
    fn test_summarize_empty() {
        let mut summarizer = ConversationSummarizer::new();
        let summary = summarizer.summarize(&[]);
        assert_eq!(summary.message_count, 0);
        assert_eq!(summary.total_words, 0);
        assert!(summary.abstractive.is_empty());
    }

    // ═══ Statistics tests ════════════════════════════════

    #[test]
    fn test_stats_updated() {
        let mut summarizer = ConversationSummarizer::new();
        let messages = sample_conversation();
        summarizer.summarize(&messages);
        assert_eq!(summarizer.stats().summaries_generated, 1);
        assert_eq!(summarizer.stats().messages_processed, 8);
    }

    #[test]
    fn test_stats_cumulative() {
        let mut summarizer = ConversationSummarizer::new();
        let messages = sample_conversation();
        summarizer.summarize(&messages);
        summarizer.summarize(&messages);
        assert_eq!(summarizer.stats().summaries_generated, 2);
        assert_eq!(summarizer.stats().messages_processed, 16);
    }

    // ═══ Role tests ══════════════════════════════════════

    #[test]
    fn test_role_equality() {
        assert_eq!(Role::User, Role::User);
        assert_ne!(Role::User, Role::Assistant);
        assert_ne!(Role::System, Role::User);
    }

    // ═══ Topic frequency filter test ═════════════════════

    #[test]
    fn test_topic_min_frequency() {
        let config = SummarizerConfig {
            min_topic_frequency: 5,
            ..Default::default()
        };
        let summarizer = ConversationSummarizer::with_config(config);
        let messages = sample_conversation();
        let topics = summarizer.extract_topics(&messages);
        for topic in &topics {
            assert!(topic.frequency >= 5);
        }
    }

    // ═══ Default impl test ═══════════════════════════════

    #[test]
    fn test_default_impl() {
        let mut summarizer = ConversationSummarizer::default();
        let summary = summarizer.summarize(&[]);
        assert_eq!(summary.message_count, 0);
    }

    // ═══ Single message test ═════════════════════════════

    #[test]
    fn test_single_message_summary() {
        let mut summarizer = ConversationSummarizer::new();
        let messages = vec![ConversationMessage::new(
            "1",
            Role::User,
            "Tell me about knowledge graphs.",
        )];
        let summary = summarizer.summarize(&messages);
        assert_eq!(summary.message_count, 1);
    }

    // ═══ Stop word filtering test ════════════════════════

    #[test]
    fn test_stop_words_filtered() {
        let summarizer = ConversationSummarizer::new();
        let messages = vec![ConversationMessage::new(
            "1",
            Role::User,
            "the the the the the",
        )];
        let topics = summarizer.extract_topics(&messages);
        // "the" should be filtered
        assert!(!topics.iter().any(|t| t.term == "the"));
    }
}
