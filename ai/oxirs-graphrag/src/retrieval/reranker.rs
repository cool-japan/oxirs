//! Cross-encoder reranking for GraphRAG

use crate::{GraphRAGResult, ScoredEntity};
use async_trait::async_trait;

/// Reranker trait for cross-encoder reranking
#[async_trait]
pub trait RerankerTrait: Send + Sync {
    /// Rerank candidates given the original query
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<ScoredEntity>,
    ) -> GraphRAGResult<Vec<ScoredEntity>>;
}

/// Simple score-based reranker (no cross-encoder)
pub struct Reranker {
    /// Boost factor for entities appearing in multiple sources
    multi_source_boost: f64,
    /// Minimum score threshold
    min_score: f64,
}

impl Default for Reranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker {
    pub fn new() -> Self {
        Self {
            multi_source_boost: 1.2,
            min_score: 0.1,
        }
    }

    /// Set multi-source boost factor
    pub fn with_multi_source_boost(mut self, boost: f64) -> Self {
        self.multi_source_boost = boost;
        self
    }

    /// Set minimum score threshold
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }

    /// Rerank candidates based on heuristics
    pub fn rerank(&self, candidates: Vec<ScoredEntity>) -> Vec<ScoredEntity> {
        let mut reranked: Vec<ScoredEntity> = candidates
            .into_iter()
            .filter(|e| e.score >= self.min_score)
            .map(|mut e| {
                // Boost fused results
                if e.source == crate::ScoreSource::Fused {
                    e.score *= self.multi_source_boost;
                }
                e
            })
            .collect();

        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reranked
    }
}

/// Cross-encoder reranker using an embedding model
pub struct CrossEncoderReranker<E>
where
    E: CrossEncoderModel,
{
    model: E,
    batch_size: usize,
}

/// Trait for cross-encoder models
#[async_trait]
pub trait CrossEncoderModel: Send + Sync {
    /// Score a query-document pair
    async fn score(&self, query: &str, document: &str) -> GraphRAGResult<f64>;

    /// Score multiple pairs in batch
    async fn score_batch(&self, query: &str, documents: &[&str]) -> GraphRAGResult<Vec<f64>>;
}

impl<E: CrossEncoderModel> CrossEncoderReranker<E> {
    pub fn new(model: E) -> Self {
        Self {
            model,
            batch_size: 32,
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Rerank using cross-encoder
    pub async fn rerank(
        &self,
        query: &str,
        candidates: Vec<ScoredEntity>,
    ) -> GraphRAGResult<Vec<ScoredEntity>> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // Get document representations (URIs for now)
        let docs: Vec<&str> = candidates.iter().map(|e| e.uri.as_str()).collect();

        // Score in batches
        let mut all_scores = Vec::with_capacity(candidates.len());
        for chunk in docs.chunks(self.batch_size) {
            let scores = self.model.score_batch(query, chunk).await?;
            all_scores.extend(scores);
        }

        // Combine with original scores
        let mut reranked: Vec<ScoredEntity> = candidates
            .into_iter()
            .zip(all_scores)
            .map(|(mut e, cross_score)| {
                // Weighted combination of original and cross-encoder scores
                e.score = e.score * 0.3 + cross_score * 0.7;
                e
            })
            .collect();

        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(reranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScoreSource;
    use std::collections::HashMap;

    #[test]
    fn test_simple_reranker() {
        let reranker = Reranker::new();

        let candidates = vec![
            ScoredEntity {
                uri: "http://a".to_string(),
                score: 0.5,
                source: ScoreSource::Vector,
                metadata: HashMap::new(),
            },
            ScoredEntity {
                uri: "http://b".to_string(),
                score: 0.6,
                source: ScoreSource::Fused,
                metadata: HashMap::new(),
            },
            ScoredEntity {
                uri: "http://c".to_string(),
                score: 0.05,
                source: ScoreSource::Keyword,
                metadata: HashMap::new(),
            },
        ];

        let reranked = reranker.rerank(candidates);

        // 'b' should be first (boosted), 'a' second, 'c' filtered out
        assert_eq!(reranked.len(), 2);
        assert_eq!(reranked[0].uri, "http://b");
    }
}
