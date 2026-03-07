//! Conversation Analytics
//!
//! Analyses conversation patterns, quality, and engagement for multi-turn
//! interactions. All computations are lightweight and purely in-memory.

use std::collections::HashMap;

/// Stop-words excluded from keyword counting
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not", "no", "nor", "so", "yet",
    "both", "either", "neither", "each", "than", "such", "too", "very", "just", "that", "this",
    "it", "its", "i", "you", "he", "she", "we", "they", "what", "which", "who", "whom", "how",
    "when", "where", "why", "if", "then", "as", "up", "out", "about", "from", "into",
];

/// A single turn in a conversation
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    /// Role identifier: `"user"` or `"assistant"`
    pub role: String,
    pub content: String,
    /// Wall-clock timestamp in milliseconds since Unix epoch
    pub timestamp_ms: i64,
    pub tokens: usize,
    /// Server-side latency for generating this turn (only meaningful for assistant turns)
    pub latency_ms: Option<u64>,
}

impl ConversationTurn {
    /// Convenience constructor
    pub fn new(
        role: impl Into<String>,
        content: impl Into<String>,
        timestamp_ms: i64,
        tokens: usize,
        latency_ms: Option<u64>,
    ) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            timestamp_ms,
            tokens,
            latency_ms,
        }
    }

    /// True when this turn belongs to the user
    pub fn is_user(&self) -> bool {
        self.role == "user"
    }

    /// True when this turn belongs to the assistant
    pub fn is_assistant(&self) -> bool {
        self.role == "assistant"
    }
}

/// Accumulates and analyses turns in a conversation
#[derive(Debug, Clone, Default)]
pub struct ConversationAnalytics {
    turns: Vec<ConversationTurn>,
}

impl ConversationAnalytics {
    /// Create an empty analytics collector
    pub fn new() -> Self {
        Self { turns: Vec::new() }
    }

    /// Append a new turn
    pub fn add_turn(&mut self, turn: ConversationTurn) {
        self.turns.push(turn);
    }

    /// Total number of turns (user + assistant)
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// Sum of tokens across all turns
    pub fn total_tokens(&self) -> usize {
        self.turns.iter().map(|t| t.tokens).sum()
    }

    /// Average latency in milliseconds across all assistant turns that have a recorded latency.
    /// Returns 0.0 if no latency data is available.
    pub fn avg_latency_ms(&self) -> f64 {
        let values: Vec<u64> = self
            .assistant_turns()
            .iter()
            .filter_map(|t| t.latency_ms)
            .collect();
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<u64>() as f64 / values.len() as f64
    }

    /// All user turns (borrows)
    pub fn user_turns(&self) -> Vec<&ConversationTurn> {
        self.turns.iter().filter(|t| t.is_user()).collect()
    }

    /// All assistant turns (borrows)
    pub fn assistant_turns(&self) -> Vec<&ConversationTurn> {
        self.turns.iter().filter(|t| t.is_assistant()).collect()
    }

    /// Average character length of user messages; 0.0 if no user turns
    pub fn avg_user_message_length(&self) -> f64 {
        let user_turns = self.user_turns();
        if user_turns.is_empty() {
            return 0.0;
        }
        let total: usize = user_turns.iter().map(|t| t.content.len()).sum();
        total as f64 / user_turns.len() as f64
    }

    /// Average character length of assistant messages; 0.0 if no assistant turns
    pub fn avg_assistant_message_length(&self) -> f64 {
        let assistant_turns = self.assistant_turns();
        if assistant_turns.is_empty() {
            return 0.0;
        }
        let total: usize = assistant_turns.iter().map(|t| t.content.len()).sum();
        total as f64 / assistant_turns.len() as f64
    }

    /// Return the top 10 non-stop words by frequency across all turns
    pub fn topic_keywords(&self) -> Vec<(String, usize)> {
        let mut freq: HashMap<String, usize> = HashMap::new();

        for turn in &self.turns {
            for word in turn.content.split(|c: char| !c.is_alphanumeric()) {
                let word = word.to_lowercase();
                if word.len() >= 3 && !STOP_WORDS.contains(&word.as_str()) {
                    *freq.entry(word).or_insert(0) += 1;
                }
            }
        }

        let mut pairs: Vec<(String, usize)> = freq.into_iter().collect();
        // Sort by frequency descending, then alphabetically for determinism
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs.truncate(10);
        pairs
    }

    /// Milliseconds between the first and last turn; 0 for fewer than 2 turns
    pub fn conversation_duration_ms(&self) -> i64 {
        if self.turns.len() < 2 {
            return 0;
        }
        let first = self.turns.first().map(|t| t.timestamp_ms).unwrap_or(0);
        let last = self.turns.last().map(|t| t.timestamp_ms).unwrap_or(0);
        (last - first).max(0)
    }

    /// Tokens consumed per minute; 0.0 if duration is zero
    pub fn tokens_per_minute(&self) -> f64 {
        let duration_ms = self.conversation_duration_ms();
        if duration_ms == 0 {
            return 0.0;
        }
        let minutes = duration_ms as f64 / 60_000.0;
        self.total_tokens() as f64 / minutes
    }

    /// Count the number of `'?'` characters in user turns as a question proxy
    pub fn question_count(&self) -> usize {
        self.user_turns()
            .iter()
            .map(|t| t.content.chars().filter(|&c| c == '?').count())
            .sum()
    }
}

/// Produces a quality score for a conversation
pub struct ConversationQualityScorer;

/// Scored dimensions of conversation quality
#[derive(Debug, Clone)]
pub struct ConversationQualityScore {
    /// Engagement: 0-1, derived from turn count and message length
    pub engagement: f64,
    /// Coherence: 0-1, approximated from topic keyword consistency
    pub coherence: f64,
    /// Responsiveness: 0-1, derived from average latency (lower latency → higher score)
    pub responsiveness: f64,
    /// Arithmetic mean of the three dimensions
    pub overall: f64,
}

impl ConversationQualityScorer {
    /// Compute a quality score for the given analytics snapshot
    pub fn score(analytics: &ConversationAnalytics) -> ConversationQualityScore {
        let engagement = Self::compute_engagement(analytics);
        let coherence = Self::compute_coherence(analytics);
        let responsiveness = Self::compute_responsiveness(analytics);
        let overall = (engagement + coherence + responsiveness) / 3.0;

        ConversationQualityScore {
            engagement,
            coherence,
            responsiveness,
            overall,
        }
    }

    fn compute_engagement(analytics: &ConversationAnalytics) -> f64 {
        // Score based on turn count and average user message length
        let turn_score = (analytics.turn_count() as f64 / 20.0).min(1.0);
        let len_score = (analytics.avg_user_message_length() / 200.0).min(1.0);
        (turn_score + len_score) / 2.0
    }

    fn compute_coherence(analytics: &ConversationAnalytics) -> f64 {
        // Proxy: ratio of unique top-10 keyword coverage to total words
        let keywords = analytics.topic_keywords();
        if keywords.is_empty() {
            return 0.5; // Neutral when we have no data
        }
        // Higher keyword concentration → more coherent
        let top_count = keywords.first().map(|(_, c)| *c).unwrap_or(0) as f64;
        let total_words: usize = analytics
            .turns
            .iter()
            .map(|t| t.content.split_whitespace().count())
            .sum();
        if total_words == 0 {
            return 0.5;
        }
        (top_count / total_words as f64 * 10.0).min(1.0)
    }

    fn compute_responsiveness(analytics: &ConversationAnalytics) -> f64 {
        let avg_latency = analytics.avg_latency_ms();
        if avg_latency == 0.0 {
            return 1.0; // No latency data — assume instant
        }
        // 500 ms → 1.0, 5000 ms → 0.0, linear interpolation
        let score = 1.0 - (avg_latency - 500.0).max(0.0) / 4500.0;
        score.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(offset_s: i64) -> i64 {
        1_700_000_000_000 + offset_s * 1000
    }

    fn user_turn(content: &str, offset_s: i64) -> ConversationTurn {
        ConversationTurn::new("user", content, ts(offset_s), content.len() / 4 + 1, None)
    }

    fn assistant_turn(content: &str, offset_s: i64, latency: u64) -> ConversationTurn {
        ConversationTurn::new(
            "assistant",
            content,
            ts(offset_s),
            content.len() / 4 + 1,
            Some(latency),
        )
    }

    fn sample_analytics() -> ConversationAnalytics {
        let mut a = ConversationAnalytics::new();
        a.add_turn(user_turn("What is SPARQL and how does it work?", 0));
        a.add_turn(assistant_turn(
            "SPARQL is a query language for RDF data stores.",
            2,
            800,
        ));
        a.add_turn(user_turn("Can you show me an example query?", 10));
        a.add_turn(assistant_turn(
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10",
            12,
            600,
        ));
        a.add_turn(user_turn("How do I filter results?", 20));
        a.add_turn(assistant_turn(
            "Use the FILTER keyword: FILTER(?o > 5)",
            22,
            700,
        ));
        a
    }

    // --- ConversationTurn tests ---

    #[test]
    fn test_turn_role_helpers() {
        let u = user_turn("hello?", 0);
        assert!(u.is_user());
        assert!(!u.is_assistant());

        let a = assistant_turn("Hi!", 1, 200);
        assert!(a.is_assistant());
        assert!(!a.is_user());
    }

    // --- ConversationAnalytics tests ---

    #[test]
    fn test_empty_analytics() {
        let a = ConversationAnalytics::new();
        assert_eq!(a.turn_count(), 0);
        assert_eq!(a.total_tokens(), 0);
        assert_eq!(a.avg_latency_ms(), 0.0);
        assert_eq!(a.avg_user_message_length(), 0.0);
        assert_eq!(a.avg_assistant_message_length(), 0.0);
        assert!(a.topic_keywords().is_empty());
        assert_eq!(a.conversation_duration_ms(), 0);
        assert_eq!(a.tokens_per_minute(), 0.0);
        assert_eq!(a.question_count(), 0);
    }

    #[test]
    fn test_turn_count() {
        let a = sample_analytics();
        assert_eq!(a.turn_count(), 6);
    }

    #[test]
    fn test_user_and_assistant_turns() {
        let a = sample_analytics();
        assert_eq!(a.user_turns().len(), 3);
        assert_eq!(a.assistant_turns().len(), 3);
    }

    #[test]
    fn test_total_tokens() {
        let a = sample_analytics();
        assert!(a.total_tokens() > 0);
    }

    #[test]
    fn test_avg_latency() {
        let a = sample_analytics();
        // Average of 800, 600, 700 = 700
        let avg = a.avg_latency_ms();
        assert!((avg - 700.0).abs() < 1.0, "expected ~700 ms, got {avg}");
    }

    #[test]
    fn test_avg_user_message_length() {
        let a = sample_analytics();
        let avg = a.avg_user_message_length();
        assert!(avg > 0.0);
    }

    #[test]
    fn test_avg_assistant_message_length() {
        let a = sample_analytics();
        let avg = a.avg_assistant_message_length();
        assert!(avg > 0.0);
    }

    #[test]
    fn test_topic_keywords_returns_at_most_10() {
        let a = sample_analytics();
        let kw = a.topic_keywords();
        assert!(kw.len() <= 10);
    }

    #[test]
    fn test_topic_keywords_excludes_stop_words() {
        let mut a = ConversationAnalytics::new();
        a.add_turn(user_turn("the a an and is are was were be", 0));
        let kw = a.topic_keywords();
        // All words above are stop-words — no keywords should remain
        assert!(kw.is_empty());
    }

    #[test]
    fn test_topic_keywords_sorted_by_frequency() {
        let mut a = ConversationAnalytics::new();
        a.add_turn(user_turn("rdf rdf rdf sparql sparql turtle", 0));
        let kw = a.topic_keywords();
        // "rdf" should rank first
        assert_eq!(kw[0].0, "rdf");
        assert_eq!(kw[0].1, 3);
    }

    #[test]
    fn test_conversation_duration() {
        let a = sample_analytics();
        let duration = a.conversation_duration_ms();
        // Turns span 0s to 22s
        assert_eq!(duration, 22_000);
    }

    #[test]
    fn test_tokens_per_minute() {
        let a = sample_analytics();
        let tpm = a.tokens_per_minute();
        assert!(tpm > 0.0);
    }

    #[test]
    fn test_question_count() {
        let a = sample_analytics();
        // "What is SPARQL and how does it work?" → 1
        // "Can you show me an example query?" → 1
        // "How do I filter results?" → 1
        assert_eq!(a.question_count(), 3);
    }

    #[test]
    fn test_single_turn_duration_zero() {
        let mut a = ConversationAnalytics::new();
        a.add_turn(user_turn("Hello?", 0));
        assert_eq!(a.conversation_duration_ms(), 0);
    }

    // --- ConversationQualityScorer tests ---

    #[test]
    fn test_quality_score_range() {
        let a = sample_analytics();
        let score = ConversationQualityScorer::score(&a);
        assert!((0.0..=1.0).contains(&score.engagement));
        assert!((0.0..=1.0).contains(&score.coherence));
        assert!((0.0..=1.0).contains(&score.responsiveness));
        assert!((0.0..=1.0).contains(&score.overall));
    }

    #[test]
    fn test_quality_score_empty() {
        let a = ConversationAnalytics::new();
        let score = ConversationQualityScorer::score(&a);
        // Empty conversation should still produce valid (low) scores
        assert!((0.0..=1.0).contains(&score.overall));
    }

    #[test]
    fn test_responsiveness_low_latency_scores_high() {
        let mut a = ConversationAnalytics::new();
        a.add_turn(user_turn("Hello?", 0));
        a.add_turn(assistant_turn("Hi!", 1, 100)); // 100 ms → near 1.0
        let score = ConversationQualityScorer::score(&a);
        assert!(score.responsiveness > 0.9);
    }

    #[test]
    fn test_responsiveness_high_latency_scores_low() {
        let mut a = ConversationAnalytics::new();
        a.add_turn(user_turn("Hello?", 0));
        a.add_turn(assistant_turn("Hi!", 10, 10_000)); // 10 s → near 0.0
        let score = ConversationQualityScorer::score(&a);
        assert!(score.responsiveness < 0.1);
    }
}
