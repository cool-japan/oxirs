//! # Response Ranker
//!
//! Multi-criteria ranking of RAG chat response candidates.  Each candidate is
//! scored on relevance, coherence, completeness, conciseness, and factual
//! quality; the scores are linearly combined using configurable weights.

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Weights for the five scoring dimensions.
///
/// Weights need not sum to 1.0 — call [`RankingCriteria::normalized`] to
/// obtain a version where they do.
#[derive(Debug, Clone)]
pub struct RankingCriteria {
    /// Weight given to query-response term overlap.
    pub relevance_weight: f64,
    /// Weight given to sentence structure coherence.
    pub coherence_weight: f64,
    /// Weight given to response length / coverage of query terms.
    pub completeness_weight: f64,
    /// Weight given to brevity and absence of repetition.
    pub conciseness_weight: f64,
    /// Weight given to factual confidence (generation score + source count).
    pub factual_weight: f64,
}

impl Default for RankingCriteria {
    fn default() -> Self {
        Self {
            relevance_weight: 0.3,
            coherence_weight: 0.2,
            completeness_weight: 0.2,
            conciseness_weight: 0.15,
            factual_weight: 0.15,
        }
    }
}

impl RankingCriteria {
    /// Return a copy with weights normalised to sum to 1.0.
    ///
    /// If all weights are zero the original is returned unchanged.
    pub fn normalized(&self) -> RankingCriteria {
        let total = self.relevance_weight
            + self.coherence_weight
            + self.completeness_weight
            + self.conciseness_weight
            + self.factual_weight;
        if total < 1e-15 {
            return self.clone();
        }
        RankingCriteria {
            relevance_weight: self.relevance_weight / total,
            coherence_weight: self.coherence_weight / total,
            completeness_weight: self.completeness_weight / total,
            conciseness_weight: self.conciseness_weight / total,
            factual_weight: self.factual_weight / total,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Candidate / Ranked types
// ─────────────────────────────────────────────────────────────────────────────

/// A candidate response produced by a generation model.
#[derive(Debug, Clone)]
pub struct ResponseCandidate {
    /// Unique identifier within a ranking round.
    pub id: usize,
    /// The generated text.
    pub text: String,
    /// Documents used as retrieval context for this response.
    pub source_documents: Vec<String>,
    /// Raw generation confidence provided by the model (0.0 – 1.0).
    pub generation_score: f64,
    /// Arbitrary numeric metadata (e.g. perplexity, beam score).
    pub metadata: HashMap<String, f64>,
}

impl ResponseCandidate {
    /// Convenience constructor.
    pub fn new(id: usize, text: impl Into<String>, generation_score: f64) -> Self {
        Self {
            id,
            text: text.into(),
            source_documents: vec![],
            generation_score,
            metadata: HashMap::new(),
        }
    }

    /// Add a source document.
    pub fn with_source(mut self, doc: impl Into<String>) -> Self {
        self.source_documents.push(doc.into());
        self
    }
}

/// Scores for a single candidate across all five dimensions.
#[derive(Debug, Clone)]
pub struct ResponseScores {
    /// Term-overlap relevance (0.0 – 1.0).
    pub relevance: f64,
    /// Sentence-structure coherence (0.0 – 1.0).
    pub coherence: f64,
    /// Query coverage / length adequacy (0.0 – 1.0).
    pub completeness: f64,
    /// Brevity and non-repetition (0.0 – 1.0).
    pub conciseness: f64,
    /// Factual confidence (0.0 – 1.0).
    pub factual: f64,
}

impl ResponseScores {
    /// Linearly combine the dimension scores using `criteria`.
    pub fn weighted_score(&self, criteria: &RankingCriteria) -> f64 {
        let c = criteria.normalized();
        self.relevance * c.relevance_weight
            + self.coherence * c.coherence_weight
            + self.completeness * c.completeness_weight
            + self.conciseness * c.conciseness_weight
            + self.factual * c.factual_weight
    }
}

/// A candidate after scoring and ranking.
#[derive(Debug, Clone)]
pub struct RankedResponse {
    /// The original candidate.
    pub candidate: ResponseCandidate,
    /// Per-dimension scores.
    pub scores: ResponseScores,
    /// Weighted final score.
    pub final_score: f64,
    /// 1-based rank (1 = best).
    pub rank: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// ResponseRanker
// ─────────────────────────────────────────────────────────────────────────────

/// Scores and ranks response candidates against a query.
#[derive(Debug, Default)]
pub struct ResponseRanker;

impl ResponseRanker {
    /// Create a new ranker.
    pub fn new() -> Self {
        Self
    }

    // ── Scoring dimensions ────────────────────────────────────────────────────

    /// Term-overlap relevance: |query_terms ∩ response_terms| / |query_terms|.
    ///
    /// Both query and response are lower-cased and split on whitespace.
    /// Returns 0.0 when the query contains no terms.
    pub fn score_relevance(query: &str, response: &str) -> f64 {
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return 0.0;
        }
        let response_terms = tokenize(response);
        let response_set: HashSet<&str> = response_terms.iter().map(String::as_str).collect();
        let overlap = query_terms
            .iter()
            .filter(|t| response_set.contains(t.as_str()))
            .count();
        overlap as f64 / query_terms.len() as f64
    }

    /// Sentence-structure coherence heuristic.
    ///
    /// A sentence is "well-formed" if it starts with an uppercase letter and ends
    /// with `.`, `!`, or `?`.  Returns 0.0 for an empty response.
    pub fn score_coherence(response: &str) -> f64 {
        let sentences: Vec<&str> = split_sentences(response);
        if sentences.is_empty() {
            return 0.0;
        }
        let conforming = sentences
            .iter()
            .filter(|s| {
                let s = s.trim();
                if s.is_empty() {
                    return false;
                }
                let starts_upper = s.chars().next().map(char::is_uppercase).unwrap_or(false);
                let ends_punct = s.ends_with('.') || s.ends_with('!') || s.ends_with('?');
                starts_upper && ends_punct
            })
            .count();
        (conforming as f64 / sentences.len() as f64).max(0.0)
    }

    /// Completeness: covers enough content and addresses query terms.
    ///
    /// - Base 0.5 if `word_count >= min_words`.
    /// - Bonus 0.5 × (query terms covered / total query terms).
    pub fn score_completeness(query: &str, response: &str, min_words: usize) -> f64 {
        let words = word_count(response);
        let base = if words >= min_words {
            0.5
        } else {
            words as f64 / (2.0 * min_words as f64).max(1.0)
        };
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return base;
        }
        let response_terms = tokenize(response);
        let response_set: HashSet<&str> = response_terms.iter().map(String::as_str).collect();
        let covered = query_terms
            .iter()
            .filter(|t| response_set.contains(t.as_str()))
            .count();
        let coverage_bonus = 0.5 * covered as f64 / query_terms.len() as f64;
        (base + coverage_bonus).min(1.0)
    }

    /// Conciseness: penalises responses longer than `max_words` and repeated sentences.
    pub fn score_conciseness(response: &str, max_words: usize) -> f64 {
        let wc = word_count(response);
        let length_score = if wc == 0 {
            0.0
        } else if wc <= max_words {
            1.0
        } else {
            max_words as f64 / wc as f64
        };

        // Repetition penalty: fraction of unique sentences
        let sentences = split_sentences(response);
        let n = sentences.len();
        if n == 0 {
            return length_score;
        }
        let unique: HashSet<String> = sentences.iter().map(|s| s.trim().to_lowercase()).collect();
        let repetition_ratio = unique.len() as f64 / n as f64;

        (length_score * repetition_ratio).clamp(0.0, 1.0)
    }

    /// Factual confidence: uses `generation_score` plus a source-count bonus.
    ///
    /// `score = 0.7 * generation_score + 0.3 * min(sources / 3, 1.0)`
    pub fn score_factual(candidate: &ResponseCandidate) -> f64 {
        let src_bonus = (candidate.source_documents.len() as f64 / 3.0).min(1.0);
        (0.7 * candidate.generation_score.clamp(0.0, 1.0) + 0.3 * src_bonus).clamp(0.0, 1.0)
    }

    // ── Ranking ───────────────────────────────────────────────────────────────

    /// Score and rank all candidates; higher final score = lower (better) rank number.
    pub fn rank(
        query: &str,
        candidates: Vec<ResponseCandidate>,
        criteria: &RankingCriteria,
    ) -> Vec<RankedResponse> {
        let min_words = 20usize;
        let max_words = 200usize;

        let mut ranked: Vec<RankedResponse> = candidates
            .into_iter()
            .map(|c| {
                let scores = ResponseScores {
                    relevance: Self::score_relevance(query, &c.text),
                    coherence: Self::score_coherence(&c.text),
                    completeness: Self::score_completeness(query, &c.text, min_words),
                    conciseness: Self::score_conciseness(&c.text, max_words),
                    factual: Self::score_factual(&c),
                };
                let final_score = scores.weighted_score(criteria);
                RankedResponse {
                    candidate: c,
                    scores,
                    final_score,
                    rank: 0, // assigned below
                }
            })
            .collect();

        // Sort descending by score, then ascending by candidate id for stability
        ranked.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.candidate.id.cmp(&b.candidate.id))
        });

        // Assign 1-based ranks
        for (i, r) in ranked.iter_mut().enumerate() {
            r.rank = i + 1;
        }
        ranked
    }

    /// Return the first `k` elements of a ranked slice.
    pub fn top_k(ranked: &[RankedResponse], k: usize) -> &[RankedResponse] {
        let end = k.min(ranked.len());
        &ranked[..end]
    }

    /// Return all ranked responses whose `final_score >= min_score`.
    pub fn filter_by_threshold(ranked: &[RankedResponse], min_score: f64) -> Vec<&RankedResponse> {
        ranked
            .iter()
            .filter(|r| r.final_score >= min_score)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// String helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Lowercase word tokenisation — removes punctuation that is not internal.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Count whitespace-delimited words in `text`.
fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Split text into sentences on `.`, `!`, `?`.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        if chars[i] == '.' || chars[i] == '!' || chars[i] == '?' {
            // Collect from start to i (inclusive)
            let end_byte = text
                .char_indices()
                .nth(i + 1)
                .map(|(b, _)| b)
                .unwrap_or(text.len());
            let start_byte = text
                .char_indices()
                .nth(start)
                .map(|(b, _)| b)
                .unwrap_or(text.len());
            if end_byte > start_byte {
                let slice = &text[start_byte..end_byte];
                if !slice.trim().is_empty() {
                    sentences.push(slice);
                }
            }
            start = i + 1;
        }
        i += 1;
    }
    // Trailing fragment (no terminal punctuation)
    let start_byte = text
        .char_indices()
        .nth(start)
        .map(|(b, _)| b)
        .unwrap_or(text.len());
    let trailing = &text[start_byte..];
    if !trailing.trim().is_empty() {
        sentences.push(trailing);
    }
    sentences
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn crit_equal() -> RankingCriteria {
        RankingCriteria {
            relevance_weight: 1.0,
            coherence_weight: 1.0,
            completeness_weight: 1.0,
            conciseness_weight: 1.0,
            factual_weight: 1.0,
        }
    }

    // ── RankingCriteria::normalized ──────────────────────────────────────────

    #[test]
    fn test_normalized_sums_to_one() {
        let c = RankingCriteria {
            relevance_weight: 2.0,
            coherence_weight: 1.0,
            completeness_weight: 1.0,
            conciseness_weight: 1.0,
            factual_weight: 0.0,
        };
        let n = c.normalized();
        let sum = n.relevance_weight
            + n.coherence_weight
            + n.completeness_weight
            + n.conciseness_weight
            + n.factual_weight;
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalized_zero_weights_unchanged() {
        let c = RankingCriteria {
            relevance_weight: 0.0,
            coherence_weight: 0.0,
            completeness_weight: 0.0,
            conciseness_weight: 0.0,
            factual_weight: 0.0,
        };
        let n = c.normalized();
        assert!((n.relevance_weight).abs() < 1e-12);
    }

    #[test]
    fn test_normalized_default_sums_to_one() {
        let c = RankingCriteria::default();
        let n = c.normalized();
        let sum = n.relevance_weight
            + n.coherence_weight
            + n.completeness_weight
            + n.conciseness_weight
            + n.factual_weight;
        assert!((sum - 1.0).abs() < 1e-9);
    }

    // ── score_relevance ──────────────────────────────────────────────────────

    #[test]
    fn test_relevance_full_overlap() {
        let score = ResponseRanker::score_relevance("the quick brown fox", "the quick brown fox");
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_relevance_no_overlap() {
        let score = ResponseRanker::score_relevance("apple orange", "banana grape lemon");
        assert!((score).abs() < 1e-9);
    }

    #[test]
    fn test_relevance_partial_overlap() {
        let score = ResponseRanker::score_relevance("quick brown", "quick fox lazy");
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_relevance_empty_query() {
        let score = ResponseRanker::score_relevance("", "some response text");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_relevance_case_insensitive() {
        let score = ResponseRanker::score_relevance("Fox", "the fox ran");
        assert!(score > 0.0);
    }

    // ── score_coherence ──────────────────────────────────────────────────────

    #[test]
    fn test_coherence_perfect() {
        let score = ResponseRanker::score_coherence("The sky is blue. The grass is green.");
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_coherence_no_uppercase_start() {
        let score = ResponseRanker::score_coherence("the sky is blue. the grass is green.");
        // Neither sentence starts with uppercase
        assert!(score < 1.0);
    }

    #[test]
    fn test_coherence_empty_response() {
        assert_eq!(ResponseRanker::score_coherence(""), 0.0);
    }

    #[test]
    fn test_coherence_mixed() {
        // One good, one bad
        let score = ResponseRanker::score_coherence("Good sentence. bad sentence.");
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_coherence_question_mark() {
        let score = ResponseRanker::score_coherence("Is this correct?");
        assert_eq!(score, 1.0);
    }

    // ── score_completeness ───────────────────────────────────────────────────

    #[test]
    fn test_completeness_long_response_with_all_terms() {
        let query = "climate change";
        let response = "Climate change is a major issue affecting all nations. \
                        Change in climate patterns has accelerated due to emissions. \
                        This is a longer text to meet the minimum word count requirement for testing.";
        let score = ResponseRanker::score_completeness(query, response, 10);
        assert!(score > 0.5);
    }

    #[test]
    fn test_completeness_too_short() {
        let score = ResponseRanker::score_completeness("query term", "short", 100);
        assert!(score < 0.5);
    }

    #[test]
    fn test_completeness_empty_query() {
        let score = ResponseRanker::score_completeness(
            "",
            "some long response text here with many words",
            5,
        );
        assert!(score >= 0.5); // word count requirement met
    }

    // ── score_conciseness ────────────────────────────────────────────────────

    #[test]
    fn test_conciseness_short_response() {
        let score = ResponseRanker::score_conciseness("Hello world.", 200);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_conciseness_too_long_response() {
        let long: String = (0..300).map(|i| format!("word{i} ")).collect();
        let score = ResponseRanker::score_conciseness(&long, 100);
        assert!(score < 1.0);
    }

    #[test]
    fn test_conciseness_repeated_sentences() {
        let repeated = "Same sentence. Same sentence. Same sentence.";
        let score = ResponseRanker::score_conciseness(repeated, 100);
        // Repetition should lower the score compared to unique sentences
        let unique = "First sentence. Second sentence. Third sentence.";
        let unique_score = ResponseRanker::score_conciseness(unique, 100);
        assert!(score <= unique_score);
    }

    #[test]
    fn test_conciseness_empty() {
        assert_eq!(ResponseRanker::score_conciseness("", 100), 0.0);
    }

    // ── score_factual ─────────────────────────────────────────────────────────

    #[test]
    fn test_factual_high_generation_score() {
        let c = ResponseCandidate::new(0, "text", 1.0);
        let score = ResponseRanker::score_factual(&c);
        assert!(score >= 0.7);
    }

    #[test]
    fn test_factual_with_sources() {
        let c = ResponseCandidate::new(0, "text", 0.5)
            .with_source("doc1")
            .with_source("doc2")
            .with_source("doc3");
        let score = ResponseRanker::score_factual(&c);
        assert!(score > ResponseRanker::score_factual(&ResponseCandidate::new(0, "text", 0.5)));
    }

    #[test]
    fn test_factual_zero_score() {
        let c = ResponseCandidate::new(0, "text", 0.0);
        let score = ResponseRanker::score_factual(&c);
        assert!(score >= 0.0);
    }

    #[test]
    fn test_factual_clamps_to_one() {
        let c = ResponseCandidate::new(0, "text", 2.0)
            .with_source("a")
            .with_source("b")
            .with_source("c")
            .with_source("d");
        let score = ResponseRanker::score_factual(&c);
        assert!(score <= 1.0);
    }

    // ── ResponseScores::weighted_score ─────────────────────────────────────────

    #[test]
    fn test_weighted_score_all_ones() {
        let scores = ResponseScores {
            relevance: 1.0,
            coherence: 1.0,
            completeness: 1.0,
            conciseness: 1.0,
            factual: 1.0,
        };
        let c = crit_equal();
        let ws = scores.weighted_score(&c);
        assert!((ws - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_score_all_zeros() {
        let scores = ResponseScores {
            relevance: 0.0,
            coherence: 0.0,
            completeness: 0.0,
            conciseness: 0.0,
            factual: 0.0,
        };
        assert_eq!(scores.weighted_score(&crit_equal()), 0.0);
    }

    // ── ResponseRanker::rank ──────────────────────────────────────────────────

    #[test]
    fn test_rank_returns_same_count() {
        let candidates = vec![
            ResponseCandidate::new(0, "The quick brown fox jumps over the lazy dog. Nice.", 0.9),
            ResponseCandidate::new(1, "A completely irrelevant answer with no overlap.", 0.1),
        ];
        let criteria = RankingCriteria::default();
        let ranked = ResponseRanker::rank("quick brown fox", candidates, &criteria);
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn test_rank_assigns_ranks_starting_at_one() {
        let candidates = vec![
            ResponseCandidate::new(0, "response a", 0.5),
            ResponseCandidate::new(1, "response b", 0.8),
        ];
        let ranked = ResponseRanker::rank("query", candidates, &RankingCriteria::default());
        let ranks: HashSet<usize> = ranked.iter().map(|r| r.rank).collect();
        assert!(ranks.contains(&1));
        assert!(ranks.contains(&2));
    }

    #[test]
    fn test_rank_higher_score_gets_lower_rank() {
        let high_quality = "The answer addresses the query directly. It provides relevant context. Sources are cited.";
        let low_quality = "maybe";
        let candidates = vec![
            ResponseCandidate::new(0, high_quality, 0.9)
                .with_source("doc1")
                .with_source("doc2"),
            ResponseCandidate::new(1, low_quality, 0.1),
        ];
        let criteria = RankingCriteria::default();
        let ranked = ResponseRanker::rank("query answer relevant", candidates, &criteria);
        // rank 1 should be the higher-scoring candidate
        assert_eq!(ranked[0].rank, 1);
        assert!(ranked[0].final_score >= ranked[1].final_score);
    }

    #[test]
    fn test_rank_empty_candidates() {
        let ranked = ResponseRanker::rank("query", vec![], &RankingCriteria::default());
        assert!(ranked.is_empty());
    }

    // ── top_k ────────────────────────────────────────────────────────────────

    #[test]
    fn test_top_k_returns_k_elements() {
        let candidates: Vec<ResponseCandidate> = (0..5)
            .map(|i| ResponseCandidate::new(i, format!("response {i}"), i as f64 * 0.1))
            .collect();
        let ranked = ResponseRanker::rank("query", candidates, &RankingCriteria::default());
        let top = ResponseRanker::top_k(&ranked, 3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_top_k_larger_than_ranked() {
        let candidates = vec![ResponseCandidate::new(0, "r", 0.5)];
        let ranked = ResponseRanker::rank("query", candidates, &RankingCriteria::default());
        let top = ResponseRanker::top_k(&ranked, 10);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_top_k_zero() {
        let candidates = vec![ResponseCandidate::new(0, "r", 0.5)];
        let ranked = ResponseRanker::rank("query", candidates, &RankingCriteria::default());
        let top = ResponseRanker::top_k(&ranked, 0);
        assert!(top.is_empty());
    }

    // ── filter_by_threshold ──────────────────────────────────────────────────

    #[test]
    fn test_filter_by_threshold_keeps_high_scores() {
        let candidates: Vec<ResponseCandidate> = (0..4)
            .map(|i| {
                ResponseCandidate::new(
                    i,
                    "relevant response text here for testing",
                    i as f64 * 0.25,
                )
            })
            .collect();
        let ranked =
            ResponseRanker::rank("relevant response", candidates, &RankingCriteria::default());
        let filtered = ResponseRanker::filter_by_threshold(&ranked, 0.0);
        assert!(!filtered.is_empty());
    }

    #[test]
    fn test_filter_by_threshold_excludes_all_when_threshold_too_high() {
        let candidates = vec![ResponseCandidate::new(0, "response", 0.1)];
        let ranked = ResponseRanker::rank("query", candidates, &RankingCriteria::default());
        let filtered = ResponseRanker::filter_by_threshold(&ranked, 2.0);
        assert!(filtered.is_empty());
    }

    // ── ResponseCandidate helpers ─────────────────────────────────────────────

    #[test]
    fn test_candidate_new_defaults() {
        let c = ResponseCandidate::new(42, "text", 0.7);
        assert_eq!(c.id, 42);
        assert_eq!(c.generation_score, 0.7);
        assert!(c.source_documents.is_empty());
    }

    #[test]
    fn test_candidate_with_source() {
        let c = ResponseCandidate::new(0, "text", 0.5).with_source("doc1");
        assert_eq!(c.source_documents.len(), 1);
    }

    // ── ResponseRanker::new ───────────────────────────────────────────────────

    #[test]
    fn test_ranker_new() {
        let _r = ResponseRanker::new();
    }

    // ── RankingCriteria default clone ─────────────────────────────────────────

    #[test]
    fn test_criteria_clone() {
        let c = RankingCriteria::default();
        let c2 = c.clone();
        assert!((c.relevance_weight - c2.relevance_weight).abs() < 1e-12);
    }
}
