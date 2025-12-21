//! Diversity-aware re-ranking strategies
//!
//! Diversity re-ranking ensures that search results cover different aspects
//! of the query rather than redundant similar documents.
//!
//! ## Strategies
//! - **MMR (Maximal Marginal Relevance)**: Balances relevance and diversity
//! - **Cluster-based**: Groups similar documents and selects from each cluster
//! - **Topic-based**: Ensures topical diversity across results

use crate::reranking::types::{RerankingResult, ScoredCandidate};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Diversity strategy for re-ranking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiversityStrategy {
    /// Maximal Marginal Relevance (MMR)
    MaximalMarginalRelevance,
    /// Cluster-based diversity
    ClusterBased,
    /// Topic-based diversity
    TopicBased,
    /// No diversity (baseline)
    None,
}

/// Diversity re-ranker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityReranker {
    /// Diversity weight (0.0 = relevance only, 1.0 = diversity only)
    weight: f32,
    /// Strategy to use
    strategy: DiversityStrategy,
    /// Similarity threshold for considering documents as similar
    similarity_threshold: f32,
}

impl DiversityReranker {
    /// Create new diversity re-ranker with default strategy
    pub fn new(weight: f32) -> Self {
        Self {
            weight: weight.clamp(0.0, 1.0),
            strategy: DiversityStrategy::MaximalMarginalRelevance,
            similarity_threshold: 0.85,
        }
    }

    /// Create with specific strategy
    pub fn with_strategy(weight: f32, strategy: DiversityStrategy) -> Self {
        Self {
            weight: weight.clamp(0.0, 1.0),
            strategy,
            similarity_threshold: 0.85,
        }
    }

    /// Set similarity threshold
    pub fn set_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Apply diversity re-ranking to candidates
    pub fn apply_diversity(
        &self,
        candidates: &[ScoredCandidate],
    ) -> RerankingResult<Vec<ScoredCandidate>> {
        if candidates.is_empty() || self.weight == 0.0 {
            return Ok(candidates.to_vec());
        }

        match self.strategy {
            DiversityStrategy::MaximalMarginalRelevance => self.mmr_rerank(candidates),
            DiversityStrategy::ClusterBased => self.cluster_based_rerank(candidates),
            DiversityStrategy::TopicBased => self.topic_based_rerank(candidates),
            DiversityStrategy::None => Ok(candidates.to_vec()),
        }
    }

    /// MMR (Maximal Marginal Relevance) re-ranking
    ///
    /// Selects documents that maximize:
    /// MMR = λ * Relevance(d) - (1-λ) * max Similarity(d, already_selected)
    fn mmr_rerank(&self, candidates: &[ScoredCandidate]) -> RerankingResult<Vec<ScoredCandidate>> {
        let lambda = 1.0 - self.weight; // Convert weight to λ parameter
        let mut selected = Vec::new();
        let mut remaining: Vec<_> = candidates.to_vec();

        // Select first candidate (highest relevance)
        if let Some(first) = remaining.first().cloned() {
            selected.push(first);
            remaining.remove(0);
        }

        // Iteratively select documents maximizing MMR
        while !remaining.is_empty() && selected.len() < candidates.len() {
            let mut best_idx = 0;
            let mut best_mmr = f32::NEG_INFINITY;

            for (idx, candidate) in remaining.iter().enumerate() {
                // Relevance component
                let relevance = candidate.effective_score();

                // Diversity component: max similarity to already selected
                let max_similarity = selected
                    .iter()
                    .map(|sel| self.compute_similarity(candidate, sel))
                    .fold(0.0f32, f32::max);

                // MMR score
                let mmr = lambda * relevance - (1.0 - lambda) * max_similarity;

                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = idx;
                }
            }

            // Select best MMR candidate
            if best_idx < remaining.len() {
                selected.push(remaining.remove(best_idx));
            } else {
                break;
            }
        }

        Ok(selected)
    }

    /// Cluster-based diversity re-ranking
    ///
    /// Groups similar documents into clusters and selects
    /// representatives from each cluster to ensure diversity.
    fn cluster_based_rerank(
        &self,
        candidates: &[ScoredCandidate],
    ) -> RerankingResult<Vec<ScoredCandidate>> {
        if candidates.len() <= 2 {
            return Ok(candidates.to_vec());
        }

        // Simple greedy clustering
        let mut clusters: Vec<Vec<ScoredCandidate>> = Vec::new();
        let mut assigned = HashSet::new();

        for (idx, candidate) in candidates.iter().enumerate() {
            if assigned.contains(&idx) {
                continue;
            }

            // Start new cluster
            let mut cluster = vec![candidate.clone()];
            assigned.insert(idx);

            // Find similar candidates
            for (other_idx, other) in candidates.iter().enumerate() {
                if assigned.contains(&other_idx) {
                    continue;
                }

                let similarity = self.compute_similarity(candidate, other);
                if similarity > self.similarity_threshold {
                    cluster.push(other.clone());
                    assigned.insert(other_idx);
                }
            }

            clusters.push(cluster);
        }

        // Select best candidate from each cluster
        let mut result = Vec::new();
        let num_per_cluster = (candidates.len() / clusters.len().max(1)).max(1);

        for cluster in clusters {
            // Sort cluster by score
            let mut sorted_cluster = cluster;
            sorted_cluster.sort_by(|a, b| {
                b.effective_score()
                    .partial_cmp(&a.effective_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Take top candidates from this cluster
            result.extend(sorted_cluster.into_iter().take(num_per_cluster));
        }

        // Sort final result by score
        result.sort_by(|a, b| {
            b.effective_score()
                .partial_cmp(&a.effective_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(result)
    }

    /// Topic-based diversity re-ranking
    ///
    /// Ensures results cover different topics by analyzing
    /// keyword distributions and selecting diverse documents.
    fn topic_based_rerank(
        &self,
        candidates: &[ScoredCandidate],
    ) -> RerankingResult<Vec<ScoredCandidate>> {
        // Extract topics (keywords) from each candidate
        let mut doc_topics: Vec<HashSet<String>> = Vec::new();

        for candidate in candidates {
            let content = candidate.content.as_deref().unwrap_or("");
            let topics = self.extract_topics(content);
            doc_topics.push(topics);
        }

        // Select documents to maximize topic coverage
        let mut selected = Vec::new();
        let mut covered_topics = HashSet::new();
        let mut remaining_indices: Vec<usize> = (0..candidates.len()).collect();

        while !remaining_indices.is_empty() && selected.len() < candidates.len() {
            let mut best_idx = 0;
            let mut best_score = f32::NEG_INFINITY;

            for (list_idx, &doc_idx) in remaining_indices.iter().enumerate() {
                let candidate = &candidates[doc_idx];
                let topics = &doc_topics[doc_idx];

                // Relevance component
                let relevance = candidate.effective_score();

                // Diversity component: number of new topics
                let new_topics = topics.difference(&covered_topics).count() as f32;
                let total_topics = topics.len().max(1) as f32;
                let topic_novelty = new_topics / total_topics;

                // Combined score
                let score = (1.0 - self.weight) * relevance + self.weight * topic_novelty;

                if score > best_score {
                    best_score = score;
                    best_idx = list_idx;
                }
            }

            // Select best candidate
            if best_idx < remaining_indices.len() {
                let doc_idx = remaining_indices.remove(best_idx);
                selected.push(candidates[doc_idx].clone());

                // Update covered topics
                for topic in &doc_topics[doc_idx] {
                    covered_topics.insert(topic.clone());
                }
            } else {
                break;
            }
        }

        Ok(selected)
    }

    /// Compute similarity between two candidates
    fn compute_similarity(&self, a: &ScoredCandidate, b: &ScoredCandidate) -> f32 {
        // Extract keywords from both documents
        let a_content = a.content.as_deref().unwrap_or("");
        let b_content = b.content.as_deref().unwrap_or("");

        let a_words: HashSet<String> = a_content
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3) // Filter short words
            .map(|w| w.to_string())
            .collect();

        let b_words: HashSet<String> = b_content
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(|w| w.to_string())
            .collect();

        if a_words.is_empty() || b_words.is_empty() {
            return 0.0;
        }

        // Jaccard similarity
        let intersection = a_words.intersection(&b_words).count() as f32;
        let union = a_words.union(&b_words).count() as f32;

        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    /// Extract topics (keywords) from document
    fn extract_topics(&self, document: &str) -> HashSet<String> {
        // Simple keyword extraction
        // In production: use TF-IDF, NER, or topic modeling
        document
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 4) // Only meaningful words
            .map(|w| w.to_string())
            .collect()
    }
}

impl Default for DiversityReranker {
    fn default() -> Self {
        Self::new(0.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candidates() -> Vec<ScoredCandidate> {
        vec![
            ScoredCandidate::new("doc1", 0.9, 0)
                .with_content("machine learning deep neural networks")
                .with_reranking_score(0.85),
            ScoredCandidate::new("doc2", 0.85, 1)
                .with_content("machine learning algorithms classification")
                .with_reranking_score(0.8),
            ScoredCandidate::new("doc3", 0.7, 2)
                .with_content("database management systems SQL queries")
                .with_reranking_score(0.75),
            ScoredCandidate::new("doc4", 0.65, 3)
                .with_content("web development JavaScript frameworks")
                .with_reranking_score(0.7),
        ]
    }

    #[test]
    fn test_mmr_rerank() {
        let reranker = DiversityReranker::new(0.5);
        let candidates = create_test_candidates();

        let result = reranker.mmr_rerank(&candidates).unwrap();

        // Should have all candidates
        assert_eq!(result.len(), candidates.len());

        // First should be highest scoring
        assert_eq!(result[0].id, "doc1");

        // Should not have all machine learning docs together
        // (diversity should spread them out)
        let first_three_ids: Vec<_> = result.iter().take(3).map(|c| c.id.as_str()).collect();
        let all_ml = first_three_ids
            .iter()
            .all(|id| id.starts_with("doc1") || id.starts_with("doc2"));
        assert!(!all_ml, "MMR should diversify results");
    }

    #[test]
    fn test_cluster_based_rerank() {
        let reranker = DiversityReranker::with_strategy(0.5, DiversityStrategy::ClusterBased);
        let candidates = create_test_candidates();

        let result = reranker.cluster_based_rerank(&candidates).unwrap();

        assert!(!result.is_empty());
        assert!(result.len() <= candidates.len());
    }

    #[test]
    fn test_topic_based_rerank() {
        let reranker = DiversityReranker::with_strategy(0.6, DiversityStrategy::TopicBased);
        let candidates = create_test_candidates();

        let result = reranker.topic_based_rerank(&candidates).unwrap();

        assert_eq!(result.len(), candidates.len());

        // Verify diversity: first few results should cover different topics
        let first_two = &result[0..2.min(result.len())];
        let similarity = reranker.compute_similarity(&first_two[0], &first_two[1]);

        // Should have lower similarity due to diversity
        assert!(
            similarity < 0.8,
            "Topic-based reranking should increase diversity"
        );
    }

    #[test]
    fn test_no_diversity() {
        let reranker = DiversityReranker::new(0.0); // No diversity
        let candidates = create_test_candidates();

        let result = reranker.apply_diversity(&candidates).unwrap();

        // Should return unchanged
        assert_eq!(result.len(), candidates.len());
        for (orig, res) in candidates.iter().zip(result.iter()) {
            assert_eq!(orig.id, res.id);
        }
    }

    #[test]
    fn test_similarity_computation() {
        let reranker = DiversityReranker::new(0.3);

        let a = ScoredCandidate::new("a", 0.8, 0).with_content("machine learning neural networks");

        let b = ScoredCandidate::new("b", 0.7, 1).with_content("machine learning algorithms");

        let c = ScoredCandidate::new("c", 0.6, 2).with_content("database systems SQL");

        let sim_ab = reranker.compute_similarity(&a, &b);
        let sim_ac = reranker.compute_similarity(&a, &c);

        // a and b should be more similar than a and c
        assert!(sim_ab > sim_ac);
    }

    #[test]
    fn test_topic_extraction() {
        let reranker = DiversityReranker::new(0.3);
        let doc = "machine learning and deep neural networks for classification";

        let topics = reranker.extract_topics(doc);

        assert!(topics.contains("machine"));
        assert!(topics.contains("learning"));
        assert!(topics.contains("neural"));
        assert!(topics.contains("networks"));
        assert!(topics.contains("classification"));

        // Short words should be filtered
        assert!(!topics.contains("and"));
        assert!(!topics.contains("for"));
    }

    #[test]
    fn test_empty_candidates() {
        let reranker = DiversityReranker::new(0.5);
        let candidates = vec![];

        let result = reranker.apply_diversity(&candidates).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_candidate() {
        let reranker = DiversityReranker::new(0.5);
        let candidates = vec![ScoredCandidate::new("doc1", 0.8, 0)
            .with_content("test")
            .with_reranking_score(0.85)];

        let result = reranker.apply_diversity(&candidates).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "doc1");
    }
}
