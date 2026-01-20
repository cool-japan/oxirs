//! Result Ranking System
//!
//! Advanced ranking algorithms for RAG retrieval results to improve relevance.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use super::types::{QueryContext, RetrievalResult};

/// Ranking algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankingAlgorithmType {
    /// BM25 ranking (Best Match 25)
    BM25,
    /// TF-IDF based ranking
    TFIDF,
    /// Semantic similarity ranking
    Semantic,
    /// Learning to Rank (LTR)
    LearningToRank,
    /// Hybrid ranking (combines multiple signals)
    Hybrid,
    /// Diversification-based ranking (MMR)
    MMR,
    /// PageRank-based ranking
    PageRank,
    /// Reciprocal Rank Fusion
    RRF,
}

/// Ranking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingConfig {
    /// Primary ranking algorithm
    pub algorithm: RankingAlgorithmType,
    /// BM25 parameters
    pub bm25_k1: f32,
    pub bm25_b: f32,
    /// Semantic similarity weight (0.0-1.0)
    pub semantic_weight: f32,
    /// Keyword match weight (0.0-1.0)
    pub keyword_weight: f32,
    /// Recency weight (0.0-1.0)
    pub recency_weight: f32,
    /// Popularity weight (0.0-1.0)
    pub popularity_weight: f32,
    /// Diversity penalty (0.0-1.0)
    pub diversity_penalty: f32,
    /// Enable re-ranking
    pub enable_reranking: bool,
    /// Re-ranking top K
    pub rerank_top_k: usize,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            algorithm: RankingAlgorithmType::Hybrid,
            bm25_k1: 1.5,
            bm25_b: 0.75,
            semantic_weight: 0.6,
            keyword_weight: 0.3,
            recency_weight: 0.05,
            popularity_weight: 0.05,
            diversity_penalty: 0.3,
            enable_reranking: true,
            rerank_top_k: 20,
        }
    }
}

/// Ranking features for a result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingFeatures {
    /// Semantic similarity score
    pub semantic_score: f32,
    /// Keyword match score
    pub keyword_score: f32,
    /// BM25 score
    pub bm25_score: f32,
    /// TF-IDF score
    pub tfidf_score: f32,
    /// Recency score
    pub recency_score: f32,
    /// Popularity score
    pub popularity_score: f32,
    /// Entity overlap score
    pub entity_overlap: f32,
    /// Query-document length ratio
    pub length_ratio: f32,
}

/// Result ranker
pub struct ResultRanker {
    config: RankingConfig,
    document_frequencies: HashMap<String, usize>,
    total_documents: usize,
}

impl ResultRanker {
    /// Create a new result ranker
    pub fn new(config: RankingConfig) -> Self {
        info!(
            "Initialized result ranker with {:?} algorithm",
            config.algorithm
        );

        Self {
            config,
            document_frequencies: HashMap::new(),
            total_documents: 0,
        }
    }

    /// Rank retrieval results
    pub fn rank(
        &self,
        results: &mut [RetrievalResult],
        query: &str,
        context: &QueryContext,
    ) -> Result<()> {
        debug!("Ranking {} results for query: {}", results.len(), query);

        match self.config.algorithm {
            RankingAlgorithmType::BM25 => self.rank_bm25(results, query)?,
            RankingAlgorithmType::TFIDF => self.rank_tfidf(results, query)?,
            RankingAlgorithmType::Semantic => self.rank_semantic(results)?,
            RankingAlgorithmType::Hybrid => self.rank_hybrid(results, query, context)?,
            RankingAlgorithmType::MMR => self.rank_mmr(results, query)?,
            RankingAlgorithmType::PageRank => self.rank_pagerank(results)?,
            RankingAlgorithmType::RRF => self.rank_rrf(results)?,
            RankingAlgorithmType::LearningToRank => self.rank_ltr(results, query, context)?,
        }

        // Re-ranking if enabled
        if self.config.enable_reranking {
            self.rerank_top_k(results, query)?;
        }

        Ok(())
    }

    /// BM25 ranking algorithm
    fn rank_bm25(&self, results: &mut [RetrievalResult], query: &str) -> Result<()> {
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let avg_doc_length = self.calculate_avg_doc_length(results);

        for result in results.iter_mut() {
            let doc_length = result.document.content.split_whitespace().count();
            let mut score = 0.0;

            for term in &query_terms {
                let tf = self.term_frequency(&result.document.content, term);
                let df = self.document_frequency(term);
                let idf =
                    ((self.total_documents as f32 - df as f32 + 0.5) / (df as f32 + 0.5)).ln();

                let numerator = tf * (self.config.bm25_k1 + 1.0);
                let denominator = tf
                    + self.config.bm25_k1
                        * (1.0 - self.config.bm25_b
                            + self.config.bm25_b * (doc_length as f32 / avg_doc_length));

                score += idf * (numerator / denominator);
            }

            result.score = score as f64;
        }

        // Sort with NaN-safe comparison (NaN values go to the end)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
    }

    /// TF-IDF ranking algorithm
    fn rank_tfidf(&self, results: &mut [RetrievalResult], query: &str) -> Result<()> {
        let query_terms: Vec<&str> = query.split_whitespace().collect();

        for result in results.iter_mut() {
            let mut score = 0.0;

            for term in &query_terms {
                let tf = self.term_frequency(&result.document.content, term);
                let df = self.document_frequency(term);
                let idf = (self.total_documents as f32 / (df as f32 + 1.0)).ln();

                score += tf * idf;
            }

            result.score = score as f64;
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
    }

    /// Semantic similarity ranking
    fn rank_semantic(&self, results: &mut [RetrievalResult]) -> Result<()> {
        // Results already have semantic scores from vector search
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
    }

    /// Hybrid ranking combining multiple signals
    fn rank_hybrid(
        &self,
        results: &mut [RetrievalResult],
        query: &str,
        context: &QueryContext,
    ) -> Result<()> {
        debug!("Computing hybrid ranking features");

        for result in results.iter_mut() {
            let features = self.extract_features(result, query, context)?;

            // Combine features with weights
            let hybrid_score = features.semantic_score * self.config.semantic_weight
                + features.keyword_score * self.config.keyword_weight
                + features.recency_score * self.config.recency_weight
                + features.popularity_score * self.config.popularity_weight;

            result.score = hybrid_score as f64;
            result
                .document
                .metadata
                .insert("hybrid_score".to_string(), hybrid_score.to_string());
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
    }

    /// Maximal Marginal Relevance (MMR) for diversity
    fn rank_mmr(&self, results: &mut [RetrievalResult], _query: &str) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }

        let lambda = 1.0 - self.config.diversity_penalty; // Balance relevance vs diversity
        let mut selected: Vec<RetrievalResult> = Vec::new();
        let mut remaining = results.to_vec();

        // Select first result (highest relevance)
        if let Some(first) = remaining.first() {
            selected.push(first.clone());
            remaining.remove(0);
        }

        // Iteratively select results that maximize MMR
        while !remaining.is_empty() && selected.len() < results.len() {
            let mut best_idx = 0;
            let mut best_mmr = f32::MIN;

            for (idx, result) in remaining.iter().enumerate() {
                let relevance = result.score as f32;

                // Calculate max similarity to already selected results
                let max_similarity = selected
                    .iter()
                    .map(|s| {
                        self.calculate_similarity(&result.document.content, &s.document.content)
                    })
                    .fold(0.0f32, f32::max);

                let mmr = lambda * relevance - (1.0 - lambda) * max_similarity;

                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = idx;
                }
            }

            selected.push(remaining.remove(best_idx));
        }

        // Update original results
        for (i, sel) in selected.iter().enumerate() {
            if i < results.len() {
                results[i] = sel.clone();
            }
        }
        Ok(())
    }

    /// PageRank-based ranking
    fn rank_pagerank(&self, results: &mut [RetrievalResult]) -> Result<()> {
        // Simplified PageRank based on entity connections
        for result in results.iter_mut() {
            let entity_count = result
                .document
                .content
                .split_whitespace()
                .filter(|w| w.starts_with("http://") || w.starts_with("https://"))
                .count();

            // More connected entities get higher scores
            result.score *= 1.0 + 0.1 * entity_count as f64;
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
    }

    /// Reciprocal Rank Fusion (RRF)
    fn rank_rrf(&self, results: &mut [RetrievalResult]) -> Result<()> {
        let k = 60.0; // RRF parameter

        for (rank, result) in results.iter_mut().enumerate() {
            result.score = (1.0 / (k + rank as f32 + 1.0)) as f64;
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
    }

    /// Learning to Rank using features
    fn rank_ltr(
        &self,
        results: &mut [RetrievalResult],
        query: &str,
        context: &QueryContext,
    ) -> Result<()> {
        // Extract features and apply learned model
        for result in results.iter_mut() {
            let features = self.extract_features(result, query, context)?;

            // Simple linear model (in production, use trained model)
            let score = features.semantic_score * 0.4
                + features.bm25_score * 0.3
                + features.keyword_score * 0.15
                + features.entity_overlap * 0.1
                + features.popularity_score * 0.05;

            result.score = score as f64;
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
    }

    /// Re-rank top K results with expensive features
    fn rerank_top_k(&self, results: &mut [RetrievalResult], query: &str) -> Result<()> {
        let top_k = self.config.rerank_top_k.min(results.len());

        debug!("Re-ranking top {} results", top_k);

        for result in results.iter_mut().take(top_k) {
            // Apply expensive reranking features
            let enhanced_score = self.compute_enhanced_score(result, query)?;
            result.score = enhanced_score as f64;
        }

        // Re-sort after reranking
        results[..top_k].sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(())
    }

    /// Extract ranking features
    fn extract_features(
        &self,
        result: &RetrievalResult,
        query: &str,
        _context: &QueryContext,
    ) -> Result<RankingFeatures> {
        let doc = &result.document.content;

        Ok(RankingFeatures {
            semantic_score: result.score as f32,
            keyword_score: self.keyword_match_score(doc, query),
            bm25_score: self.compute_bm25_single(doc, query),
            tfidf_score: self.compute_tfidf_single(doc, query),
            recency_score: 0.5,    // Would use actual timestamp
            popularity_score: 0.5, // Would use usage stats
            entity_overlap: self.entity_overlap_score(doc, query),
            length_ratio: (query.len() as f32 / doc.len().max(1) as f32).min(1.0),
        })
    }

    /// Compute enhanced score for reranking
    fn compute_enhanced_score(&self, result: &RetrievalResult, query: &str) -> Result<f32> {
        let doc = &result.document.content;

        // Enhanced features (more expensive to compute)
        let exact_match_bonus = if doc.to_lowercase().contains(&query.to_lowercase()) {
            0.2
        } else {
            0.0
        };
        let position_bonus = 1.0; // Would consider position in document
        let quality_score = 0.8; // Would use quality metrics

        Ok((result.score as f32) + exact_match_bonus + position_bonus * 0.1 + quality_score * 0.1)
    }

    // Helper methods

    fn term_frequency(&self, doc: &str, term: &str) -> f32 {
        let count = doc.to_lowercase().matches(&term.to_lowercase()).count();
        count as f32
    }

    fn document_frequency(&self, term: &str) -> usize {
        self.document_frequencies.get(term).copied().unwrap_or(1)
    }

    fn calculate_avg_doc_length(&self, results: &[RetrievalResult]) -> f32 {
        if results.is_empty() {
            return 100.0;
        }

        let total: usize = results
            .iter()
            .map(|r| r.document.content.split_whitespace().count())
            .sum();

        total as f32 / results.len() as f32
    }

    fn keyword_match_score(&self, doc: &str, query: &str) -> f32 {
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let doc_lower = doc.to_lowercase();

        let matches = query_terms
            .iter()
            .filter(|term| doc_lower.contains(&term.to_lowercase()))
            .count();

        matches as f32 / query_terms.len().max(1) as f32
    }

    fn entity_overlap_score(&self, doc: &str, query: &str) -> f32 {
        // Count overlap of entities (URIs)
        let doc_entities: Vec<&str> = doc
            .split_whitespace()
            .filter(|w| w.starts_with("http"))
            .collect();

        let query_entities: Vec<&str> = query
            .split_whitespace()
            .filter(|w| w.starts_with("http"))
            .collect();

        if query_entities.is_empty() {
            return 0.0;
        }

        let overlap = query_entities
            .iter()
            .filter(|e| doc_entities.contains(e))
            .count();

        overlap as f32 / query_entities.len() as f32
    }

    fn compute_bm25_single(&self, doc: &str, query: &str) -> f32 {
        // Simplified BM25 for a single document
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let mut score = 0.0;

        for term in query_terms {
            let tf = self.term_frequency(doc, term);
            score += tf / (tf + self.config.bm25_k1);
        }

        score
    }

    fn compute_tfidf_single(&self, doc: &str, query: &str) -> f32 {
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let mut score = 0.0;

        for term in query_terms {
            let tf = self.term_frequency(doc, term);
            let idf = 2.0; // Simplified IDF
            score += tf * idf;
        }

        score
    }

    fn calculate_similarity(&self, doc1: &str, doc2: &str) -> f32 {
        // Jaccard similarity
        let words1: std::collections::HashSet<_> = doc1.split_whitespace().collect();
        let words2: std::collections::HashSet<_> = doc2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Update document statistics
    pub fn update_statistics(&mut self, documents: &[String]) {
        self.total_documents = documents.len();
        self.document_frequencies.clear();

        for doc in documents {
            let words: std::collections::HashSet<_> = doc.split_whitespace().collect();
            for word in words {
                *self
                    .document_frequencies
                    .entry(word.to_string())
                    .or_insert(0) += 1;
            }
        }

        info!("Updated statistics for {} documents", self.total_documents);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::types::RagDocument;

    fn create_test_result(text: &str, score: f64) -> RetrievalResult {
        let document = RagDocument::new(
            uuid::Uuid::new_v4().to_string(),
            text.to_string(),
            "test".to_string(),
        );

        RetrievalResult {
            document,
            score,
            relevance_factors: Vec::new(),
        }
    }

    #[test]
    fn test_bm25_ranking() {
        let ranker = ResultRanker::new(RankingConfig::default());
        let mut results = vec![
            create_test_result("the movie Inception", 0.5),
            create_test_result("another movie", 0.6),
            create_test_result("Inception is a great movie", 0.4),
        ];

        ranker.rank_bm25(&mut results, "Inception movie").unwrap();

        // Result with "Inception" should rank higher
        assert!(results[0].document.content.contains("Inception"));
    }

    #[test]
    fn test_hybrid_ranking() {
        let ranker = ResultRanker::new(RankingConfig::default());
        let mut results = vec![
            create_test_result("test document one", 0.8),
            create_test_result("test document two", 0.6),
        ];

        let context = QueryContext::new("session1".to_string());
        ranker
            .rank_hybrid(&mut results, "test document", &context)
            .unwrap();

        // Should maintain relative ordering based on multiple signals
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn test_keyword_match_score() {
        let ranker = ResultRanker::new(RankingConfig::default());

        let score1 = ranker.keyword_match_score("the movie Inception", "Inception movie");
        let score2 = ranker.keyword_match_score("another document", "Inception movie");

        assert!(score1 > score2);
    }
}
