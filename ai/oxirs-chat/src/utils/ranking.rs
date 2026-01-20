//! Document ranking algorithms for RAG retrieval
//!
//! This module provides various ranking algorithms to score and rank
//! documents based on relevance to a query.

/// Ranking algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankingAlgorithm {
    /// BM25 (Best Matching 25) - probabilistic relevance ranking
    BM25,
    /// TF-IDF (Term Frequency-Inverse Document Frequency)
    TFIDF,
    /// Cosine similarity between query and document embeddings
    CosineSimilarity,
    /// Linear combination of multiple signals
    LinearCombination,
    /// Learning to rank model
    LearningToRank,
    /// Reciprocal rank fusion for combining multiple rankings
    ReciprocalRankFusion,
}

impl RankingAlgorithm {
    /// Get the name of the ranking algorithm
    pub fn name(&self) -> &'static str {
        match self {
            RankingAlgorithm::BM25 => "BM25",
            RankingAlgorithm::TFIDF => "TF-IDF",
            RankingAlgorithm::CosineSimilarity => "Cosine Similarity",
            RankingAlgorithm::LinearCombination => "Linear Combination",
            RankingAlgorithm::LearningToRank => "Learning to Rank",
            RankingAlgorithm::ReciprocalRankFusion => "Reciprocal Rank Fusion",
        }
    }

    /// Get a description of the algorithm
    pub fn description(&self) -> &'static str {
        match self {
            RankingAlgorithm::BM25 => {
                "Probabilistic ranking function based on term frequency and document length"
            }
            RankingAlgorithm::TFIDF => {
                "Statistical measure of term importance based on frequency and rarity"
            }
            RankingAlgorithm::CosineSimilarity => {
                "Similarity measure between query and document vectors"
            }
            RankingAlgorithm::LinearCombination => {
                "Weighted combination of multiple ranking signals"
            }
            RankingAlgorithm::LearningToRank => "Machine learning-based ranking model",
            RankingAlgorithm::ReciprocalRankFusion => "Method for combining multiple ranked lists",
        }
    }
}

/// Ranking score with breakdown
#[derive(Debug, Clone)]
pub struct RankingScore {
    /// Overall ranking score
    pub score: f64,
    /// Algorithm used for ranking
    pub algorithm: RankingAlgorithm,
    /// Individual signal scores contributing to the final score
    pub signal_scores: Vec<SignalScore>,
    /// Confidence in the ranking
    pub confidence: f64,
    /// Explanation of the score
    pub explanation: Option<String>,
}

impl RankingScore {
    /// Create a new ranking score
    pub fn new(score: f64, algorithm: RankingAlgorithm) -> Self {
        Self {
            score,
            algorithm,
            signal_scores: Vec::new(),
            confidence: 1.0,
            explanation: None,
        }
    }

    /// Add a signal score
    pub fn with_signal(mut self, name: String, score: f64, weight: f64) -> Self {
        self.signal_scores.push(SignalScore {
            name,
            score,
            weight,
        });
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set explanation
    pub fn with_explanation(mut self, explanation: String) -> Self {
        self.explanation = Some(explanation);
        self
    }

    /// Get the weighted score considering confidence
    pub fn weighted_score(&self) -> f64 {
        self.score * self.confidence
    }

    /// Check if the score is above a threshold
    pub fn is_relevant(&self, threshold: f64) -> bool {
        self.score >= threshold
    }
}

/// Individual signal score contributing to ranking
#[derive(Debug, Clone)]
pub struct SignalScore {
    /// Name of the signal
    pub name: String,
    /// Raw score value
    pub score: f64,
    /// Weight of this signal in the final ranking
    pub weight: f64,
}

/// BM25 parameters
#[derive(Debug, Clone)]
pub struct BM25Params {
    /// Term frequency saturation parameter (typically 1.2 to 2.0)
    pub k1: f64,
    /// Length normalization parameter (typically 0.75)
    pub b: f64,
    /// Minimum IDF value
    pub min_idf: f64,
}

impl Default for BM25Params {
    fn default() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            min_idf: 0.0,
        }
    }
}

/// Compute BM25 score for a document
///
/// # Arguments
/// * `term_freq` - Term frequency in the document
/// * `doc_length` - Length of the document
/// * `avg_doc_length` - Average document length in the collection
/// * `num_docs` - Total number of documents
/// * `doc_freq` - Number of documents containing the term
/// * `params` - BM25 parameters
pub fn bm25_score(
    term_freq: f64,
    doc_length: f64,
    avg_doc_length: f64,
    num_docs: f64,
    doc_freq: f64,
    params: &BM25Params,
) -> f64 {
    // IDF component
    let idf = ((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        .ln()
        .max(params.min_idf);

    // TF component with length normalization
    let norm_length = 1.0 - params.b + params.b * (doc_length / avg_doc_length);
    let tf = (term_freq * (params.k1 + 1.0)) / (term_freq + params.k1 * norm_length);

    idf * tf
}

/// Compute TF-IDF score
///
/// # Arguments
/// * `term_freq` - Term frequency in the document
/// * `doc_freq` - Number of documents containing the term
/// * `num_docs` - Total number of documents
pub fn tfidf_score(term_freq: f64, doc_freq: f64, num_docs: f64) -> f64 {
    let tf = term_freq;
    let idf = (num_docs / (doc_freq + 1.0)).ln();
    tf * idf
}

/// Combine multiple ranking scores using linear combination
///
/// # Arguments
/// * `scores` - Vector of (score, weight) pairs
pub fn linear_combination(scores: &[(f64, f64)]) -> f64 {
    let total_weight: f64 = scores.iter().map(|(_, w)| w).sum();
    if total_weight < 1e-10 {
        return 0.0;
    }

    scores.iter().map(|(s, w)| s * w).sum::<f64>() / total_weight
}

/// Reciprocal rank fusion - combine multiple ranked lists
///
/// # Arguments
/// * `ranked_lists` - Vector of ranked document IDs (highest rank first)
/// * `k` - Constant (typically 60)
pub fn reciprocal_rank_fusion(ranked_lists: &[Vec<String>], k: f64) -> Vec<(String, f64)> {
    use std::collections::HashMap;

    let mut scores: HashMap<String, f64> = HashMap::new();

    for ranked_list in ranked_lists {
        for (rank, doc_id) in ranked_list.iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank as f64 + 1.0));
            *scores.entry(doc_id.clone()).or_insert(0.0) += rrf_score;
        }
    }

    let mut result: Vec<(String, f64)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_score() {
        let params = BM25Params::default();
        let score = bm25_score(5.0, 100.0, 80.0, 1000.0, 10.0, &params);
        assert!(score > 0.0);
    }

    #[test]
    fn test_tfidf_score() {
        let score = tfidf_score(3.0, 10.0, 100.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_linear_combination() {
        let scores = vec![(0.8, 0.5), (0.6, 0.3), (0.9, 0.2)];
        let combined = linear_combination(&scores);
        assert!(combined > 0.0 && combined < 1.0);
    }

    #[test]
    fn test_reciprocal_rank_fusion() {
        let list1 = vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()];
        let list2 = vec!["doc2".to_string(), "doc1".to_string(), "doc4".to_string()];
        let result = reciprocal_rank_fusion(&[list1, list2], 60.0);
        assert!(!result.is_empty());
        // doc1 and doc2 should have higher scores as they appear in both lists
        assert!(result[0].1 > 0.0);
    }

    #[test]
    fn test_ranking_score() {
        let score = RankingScore::new(0.85, RankingAlgorithm::BM25)
            .with_signal("term_match".to_string(), 0.9, 0.6)
            .with_signal("semantic".to_string(), 0.8, 0.4)
            .with_confidence(0.95);

        assert_eq!(score.score, 0.85);
        assert_eq!(score.signal_scores.len(), 2);
        assert_eq!(score.confidence, 0.95);
        assert!((score.weighted_score() - 0.8075).abs() < 1e-6);
    }

    #[test]
    fn test_ranking_algorithm_name() {
        assert_eq!(RankingAlgorithm::BM25.name(), "BM25");
        assert_eq!(RankingAlgorithm::TFIDF.name(), "TF-IDF");
    }
}
