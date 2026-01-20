//! Fuzzy search for query history
//!
//! Provides intelligent fuzzy matching for interactive query history search
//! with scoring, ranking, and context-aware suggestions.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use strsim;

/// Type alias for history entry with metadata (query, timestamp, execution_time, result_count)
pub type HistoryEntryWithMetadata = (String, Option<String>, Option<u64>, Option<usize>);

/// A scored history match from fuzzy search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyMatch {
    /// The matched query
    pub query: String,
    /// Fuzzy match score (0.0 = no match, 1.0 = perfect match)
    pub score: f64,
    /// Position in the original history (most recent = 0)
    pub position: usize,
    /// Execution timestamp (if available)
    pub timestamp: Option<String>,
    /// Execution time in milliseconds (if available)
    pub execution_time_ms: Option<u64>,
    /// Number of results returned (if available)
    pub result_count: Option<usize>,
}

impl FuzzyMatch {
    /// Create a new fuzzy match
    pub fn new(query: String, score: f64, position: usize) -> Self {
        Self {
            query,
            score,
            position,
            timestamp: None,
            execution_time_ms: None,
            result_count: None,
        }
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: String) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Set execution time
    pub fn with_execution_time(mut self, ms: u64) -> Self {
        self.execution_time_ms = Some(ms);
        self
    }

    /// Set result count
    pub fn with_result_count(mut self, count: usize) -> Self {
        self.result_count = Some(count);
        self
    }
}

impl PartialEq for FuzzyMatch {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.position == other.position
    }
}

impl Eq for FuzzyMatch {}

impl PartialOrd for FuzzyMatch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FuzzyMatch {
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort by score (descending), then by recency (ascending position)
        match other.score.partial_cmp(&self.score) {
            Some(Ordering::Equal) => self.position.cmp(&other.position),
            Some(ord) => ord,
            None => Ordering::Equal,
        }
    }
}

/// Configuration for fuzzy search behavior
#[derive(Debug, Clone)]
pub struct FuzzyConfig {
    /// Minimum score threshold (0.0-1.0)
    pub min_score: f64,
    /// Maximum number of results to return
    pub max_results: usize,
    /// Boost recent queries (recency factor 0.0-1.0)
    pub recency_boost: f64,
    /// Case sensitivity
    pub case_sensitive: bool,
    /// Use substring matching as fallback
    pub allow_substring: bool,
}

impl Default for FuzzyConfig {
    fn default() -> Self {
        Self {
            min_score: 0.3,        // 30% minimum match
            max_results: 20,       // Top 20 results
            recency_boost: 0.1,    // 10% boost for recent queries
            case_sensitive: false, // Case-insensitive by default
            allow_substring: true, // Allow substring matches
        }
    }
}

impl FuzzyConfig {
    /// Create a strict configuration (higher threshold, fewer results)
    pub fn strict() -> Self {
        Self {
            min_score: 0.6,
            max_results: 10,
            recency_boost: 0.05,
            case_sensitive: false,
            allow_substring: false,
        }
    }

    /// Create a lenient configuration (lower threshold, more results)
    pub fn lenient() -> Self {
        Self {
            min_score: 0.2,
            max_results: 30,
            recency_boost: 0.15,
            case_sensitive: false,
            allow_substring: true,
        }
    }
}

/// Fuzzy history searcher with configurable behavior
pub struct FuzzyHistorySearch {
    config: FuzzyConfig,
}

impl FuzzyHistorySearch {
    /// Create a new fuzzy search with default configuration
    pub fn new() -> Self {
        Self {
            config: FuzzyConfig::default(),
        }
    }

    /// Create a fuzzy search with custom configuration
    pub fn with_config(config: FuzzyConfig) -> Self {
        Self { config }
    }

    /// Search query history with fuzzy matching
    ///
    /// # Arguments
    ///
    /// * `pattern` - The search pattern
    /// * `history` - The query history (most recent first)
    ///
    /// # Returns
    ///
    /// A vector of fuzzy matches sorted by relevance score
    pub fn search(&self, pattern: &str, history: &[String]) -> Vec<FuzzyMatch> {
        if pattern.is_empty() {
            return Vec::new();
        }

        let mut matches = Vec::new();
        let total_items = history.len();

        for (pos, query) in history.iter().enumerate() {
            let score = self.calculate_score(pattern, query, pos, total_items);

            if score >= self.config.min_score {
                matches.push(FuzzyMatch::new(query.clone(), score, pos));
            }
        }

        // Sort by score (descending) and position (ascending for ties)
        matches.sort();

        // Limit results
        matches.truncate(self.config.max_results);

        matches
    }

    /// Calculate fuzzy match score with multiple algorithms
    fn calculate_score(&self, pattern: &str, query: &str, position: usize, total: usize) -> f64 {
        let (pattern_cmp, query_cmp) = if self.config.case_sensitive {
            (pattern.to_string(), query.to_string())
        } else {
            (pattern.to_lowercase(), query.to_lowercase())
        };

        // 1. Exact match (score = 1.0)
        if pattern_cmp == query_cmp {
            return 1.0;
        }

        // 2. Jaro-Winkler similarity (good for typos and partial matches)
        let jaro_winkler_score = strsim::jaro_winkler(&pattern_cmp, &query_cmp);

        // 3. Normalized Levenshtein distance (good for character edits)
        let levenshtein_score = strsim::normalized_levenshtein(&pattern_cmp, &query_cmp);

        // 4. Sorensen-Dice coefficient (good for token-based similarity)
        let sorensen_dice_score = strsim::sorensen_dice(&pattern_cmp, &query_cmp);

        // 5. Substring bonus (if pattern is substring of query)
        let substring_bonus = if self.config.allow_substring && query_cmp.contains(&pattern_cmp) {
            let ratio = pattern_cmp.len() as f64 / query_cmp.len() as f64;
            ratio * 0.3 // Up to 30% bonus for substring match
        } else {
            0.0
        };

        // 6. Token match bonus (if all pattern words appear in query)
        let token_bonus = self.calculate_token_bonus(&pattern_cmp, &query_cmp);

        // Weighted combination of all scores
        let base_score = (jaro_winkler_score * 0.35
            + levenshtein_score * 0.25
            + sorensen_dice_score * 0.20
            + substring_bonus
            + token_bonus)
            .min(1.0);

        // Apply recency boost (recent queries get a small bonus)
        if total > 0 && self.config.recency_boost > 0.0 {
            let recency_factor = 1.0 - (position as f64 / total as f64);
            let boost = recency_factor * self.config.recency_boost;
            (base_score + boost).min(1.0)
        } else {
            base_score
        }
    }

    /// Calculate token-based matching bonus
    fn calculate_token_bonus(&self, pattern: &str, query: &str) -> f64 {
        let pattern_tokens: Vec<&str> = pattern.split_whitespace().collect();
        let query_lower = query.to_lowercase();

        if pattern_tokens.is_empty() {
            return 0.0;
        }

        let matched_tokens = pattern_tokens
            .iter()
            .filter(|token| query_lower.contains(*token))
            .count();

        let match_ratio = matched_tokens as f64 / pattern_tokens.len() as f64;
        match_ratio * 0.2 // Up to 20% bonus for token matches
    }

    /// Search with metadata (timestamps, execution times, result counts)
    pub fn search_with_metadata(
        &self,
        pattern: &str,
        history_entries: &[HistoryEntryWithMetadata],
    ) -> Vec<FuzzyMatch> {
        if pattern.is_empty() {
            return Vec::new();
        }

        let total_items = history_entries.len();
        let mut matches = Vec::new();

        for (pos, (query, timestamp, exec_time, result_count)) in history_entries.iter().enumerate()
        {
            let score = self.calculate_score(pattern, query, pos, total_items);

            if score >= self.config.min_score {
                let mut fuzzy_match = FuzzyMatch::new(query.clone(), score, pos);

                if let Some(ts) = timestamp {
                    fuzzy_match = fuzzy_match.with_timestamp(ts.clone());
                }
                if let Some(time) = exec_time {
                    fuzzy_match = fuzzy_match.with_execution_time(*time);
                }
                if let Some(count) = result_count {
                    fuzzy_match = fuzzy_match.with_result_count(*count);
                }

                matches.push(fuzzy_match);
            }
        }

        matches.sort();
        matches.truncate(self.config.max_results);
        matches
    }

    /// Filter history by execution time (fast queries, slow queries)
    pub fn filter_by_execution_time(
        matches: Vec<FuzzyMatch>,
        min_ms: Option<u64>,
        max_ms: Option<u64>,
    ) -> Vec<FuzzyMatch> {
        matches
            .into_iter()
            .filter(|m| {
                if let Some(exec_time) = m.execution_time_ms {
                    let passes_min = min_ms.map_or(true, |min| exec_time >= min);
                    let passes_max = max_ms.map_or(true, |max| exec_time <= max);
                    passes_min && passes_max
                } else {
                    true // Keep matches without metadata
                }
            })
            .collect()
    }

    /// Filter history by result count
    pub fn filter_by_result_count(
        matches: Vec<FuzzyMatch>,
        min_count: Option<usize>,
        max_count: Option<usize>,
    ) -> Vec<FuzzyMatch> {
        matches
            .into_iter()
            .filter(|m| {
                if let Some(count) = m.result_count {
                    let passes_min = min_count.map_or(true, |min| count >= min);
                    let passes_max = max_count.map_or(true, |max| count <= max);
                    passes_min && passes_max
                } else {
                    true
                }
            })
            .collect()
    }
}

impl Default for FuzzyHistorySearch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let searcher = FuzzyHistorySearch::new();
        let history = vec!["SELECT * WHERE { ?s ?p ?o }".to_string()];

        let matches = searcher.search("SELECT * WHERE { ?s ?p ?o }", &history);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].score, 1.0); // Exact match
    }

    #[test]
    fn test_fuzzy_match_typo() {
        let searcher = FuzzyHistorySearch::new();
        let history = vec!["SELECT * WHERE { ?s ?p ?o }".to_string()];

        let matches = searcher.search("SELEKT * WHERE { ?s ?p ?o }", &history);

        assert!(!matches.is_empty());
        assert!(matches[0].score > 0.8); // High score despite typo
    }

    #[test]
    fn test_case_insensitive() {
        let config = FuzzyConfig {
            case_sensitive: false,
            ..Default::default()
        };
        let searcher = FuzzyHistorySearch::with_config(config);
        let history = vec!["SELECT * WHERE { ?s ?p ?o }".to_string()];

        let matches = searcher.search("select * where", &history);

        assert!(!matches.is_empty());
        assert!(matches[0].score > 0.7);
    }

    #[test]
    fn test_substring_match() {
        let searcher = FuzzyHistorySearch::new();
        let history = vec![
            "SELECT * WHERE { ?s rdf:type foaf:Person }".to_string(),
            "SELECT ?name WHERE { ?s foaf:name ?name }".to_string(),
        ];

        let matches = searcher.search("foaf:Person", &history);

        assert!(!matches.is_empty());
        assert_eq!(matches[0].position, 0); // First query matches best
    }

    #[test]
    fn test_token_matching() {
        let searcher = FuzzyHistorySearch::new();
        let history = vec!["SELECT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(?o > 10) }".to_string()];

        let matches = searcher.search("SELECT FILTER", &history);

        assert!(!matches.is_empty());
        assert!(matches[0].score > 0.5); // Both tokens present
    }

    #[test]
    fn test_recency_boost() {
        let config = FuzzyConfig {
            recency_boost: 0.2, // 20% boost
            ..Default::default()
        };
        let searcher = FuzzyHistorySearch::with_config(config);
        let history = vec![
            "SELECT * WHERE { ?s ?p ?o }".to_string(), // Most recent
            "SELECT * WHERE { ?s ?p ?o }".to_string(), // Older
        ];

        let matches = searcher.search("SELECT WHERE", &history);

        assert_eq!(matches.len(), 2);
        // Recent query should have higher score
        assert!(matches[0].position < matches[1].position);
    }

    #[test]
    fn test_min_score_threshold() {
        let config = FuzzyConfig {
            min_score: 0.8, // High threshold
            ..Default::default()
        };
        let searcher = FuzzyHistorySearch::with_config(config);
        let history = vec![
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
            "COMPLETELY DIFFERENT QUERY".to_string(),
        ];

        let matches = searcher.search("SELECT WHERE", &history);

        assert_eq!(matches.len(), 1); // Only good match passes threshold
    }

    #[test]
    fn test_max_results_limit() {
        let config = FuzzyConfig {
            max_results: 2,
            min_score: 0.1,
            ..Default::default()
        };
        let searcher = FuzzyHistorySearch::with_config(config);
        let history = vec![
            "SELECT ?s WHERE { ?s rdf:type ?type }".to_string(),
            "SELECT ?p WHERE { ?s ?p ?o }".to_string(),
            "SELECT ?o WHERE { ?s ?p ?o }".to_string(),
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
        ];

        let matches = searcher.search("SELECT", &history);

        assert_eq!(matches.len(), 2); // Limited to max_results
    }

    #[test]
    fn test_search_with_metadata() {
        let searcher = FuzzyHistorySearch::new();
        let history_entries = vec![
            (
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                Some("2025-11-14T10:00:00Z".to_string()),
                Some(150),
                Some(42),
            ),
            (
                "SELECT ?name WHERE { ?s foaf:name ?name }".to_string(),
                Some("2025-11-14T11:00:00Z".to_string()),
                Some(75),
                Some(15),
            ),
        ];

        let matches = searcher.search_with_metadata("SELECT WHERE", &history_entries);

        assert!(!matches.is_empty());
        assert!(matches[0].timestamp.is_some());
        assert!(matches[0].execution_time_ms.is_some());
        assert!(matches[0].result_count.is_some());
    }

    #[test]
    fn test_filter_by_execution_time() {
        let matches = vec![
            FuzzyMatch::new("query1".to_string(), 0.9, 0).with_execution_time(100),
            FuzzyMatch::new("query2".to_string(), 0.8, 1).with_execution_time(500),
            FuzzyMatch::new("query3".to_string(), 0.7, 2).with_execution_time(1000),
        ];

        let filtered = FuzzyHistorySearch::filter_by_execution_time(matches, Some(200), Some(800));

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].execution_time_ms, Some(500));
    }

    #[test]
    fn test_filter_by_result_count() {
        let matches = vec![
            FuzzyMatch::new("query1".to_string(), 0.9, 0).with_result_count(10),
            FuzzyMatch::new("query2".to_string(), 0.8, 1).with_result_count(50),
            FuzzyMatch::new("query3".to_string(), 0.7, 2).with_result_count(100),
        ];

        let filtered = FuzzyHistorySearch::filter_by_result_count(matches, Some(30), Some(80));

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].result_count, Some(50));
    }

    #[test]
    fn test_empty_pattern() {
        let searcher = FuzzyHistorySearch::new();
        let history = vec!["SELECT * WHERE { ?s ?p ?o }".to_string()];

        let matches = searcher.search("", &history);

        assert!(matches.is_empty());
    }

    #[test]
    fn test_empty_history() {
        let searcher = FuzzyHistorySearch::new();
        let history: Vec<String> = vec![];

        let matches = searcher.search("SELECT", &history);

        assert!(matches.is_empty());
    }

    #[test]
    fn test_strict_configuration() {
        let searcher = FuzzyHistorySearch::with_config(FuzzyConfig::strict());
        let history = vec![
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
            "SOMEWHAT SIMILAR QUERY".to_string(),
        ];

        let matches = searcher.search("SELECT", &history);

        // Strict config has higher threshold, fewer fuzzy matches
        assert!(matches.len() <= 1);
    }

    #[test]
    fn test_lenient_configuration() {
        let searcher = FuzzyHistorySearch::with_config(FuzzyConfig::lenient());
        let history = vec![
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
            "SOMEWHAT SIMILAR QUERY".to_string(),
        ];

        let matches = searcher.search("SELECT", &history);

        // Lenient config accepts more fuzzy matches
        assert!(!matches.is_empty());
    }
}
