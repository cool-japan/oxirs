//! Result fusion strategies for combining retrieval results

use crate::{GraphRAGResult, ScoreSource, ScoredEntity};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Fusion strategy enumeration
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion (RRF)
    #[default]
    ReciprocalRankFusion,
    /// Linear combination of normalized scores
    LinearCombination,
    /// Take the highest score for each entity
    HighestScore,
    /// Geometric mean of scores
    GeometricMean,
    /// Harmonic mean of scores
    HarmonicMean,
}

/// Result fuser for combining multiple retrieval sources
pub struct ResultFuser {
    strategy: FusionStrategy,
    vector_weight: f32,
    keyword_weight: f32,
    rrf_k: f64,
}

impl Default for ResultFuser {
    fn default() -> Self {
        Self::new(FusionStrategy::ReciprocalRankFusion, 0.7, 0.3)
    }
}

impl ResultFuser {
    /// Create a new result fuser
    pub fn new(strategy: FusionStrategy, vector_weight: f32, keyword_weight: f32) -> Self {
        Self {
            strategy,
            vector_weight,
            keyword_weight,
            rrf_k: 60.0,
        }
    }

    /// Set RRF constant k
    pub fn with_rrf_k(mut self, k: f64) -> Self {
        self.rrf_k = k;
        self
    }

    /// Fuse results from vector and keyword search
    pub fn fuse(
        &self,
        vector_results: &[(String, f32)],
        keyword_results: &[(String, f32)],
        max_results: usize,
    ) -> GraphRAGResult<Vec<ScoredEntity>> {
        match self.strategy {
            FusionStrategy::ReciprocalRankFusion => {
                self.reciprocal_rank_fusion(vector_results, keyword_results, max_results)
            }
            FusionStrategy::LinearCombination => {
                self.linear_combination(vector_results, keyword_results, max_results)
            }
            FusionStrategy::HighestScore => {
                self.highest_score(vector_results, keyword_results, max_results)
            }
            FusionStrategy::GeometricMean => {
                self.geometric_mean(vector_results, keyword_results, max_results)
            }
            FusionStrategy::HarmonicMean => {
                self.harmonic_mean(vector_results, keyword_results, max_results)
            }
        }
    }

    /// Reciprocal Rank Fusion
    fn reciprocal_rank_fusion(
        &self,
        vector_results: &[(String, f32)],
        keyword_results: &[(String, f32)],
        max_results: usize,
    ) -> GraphRAGResult<Vec<ScoredEntity>> {
        let mut scores: HashMap<String, (f64, ScoreSource)> = HashMap::new();

        // Add vector scores with RRF
        for (rank, (uri, _)) in vector_results.iter().enumerate() {
            let rrf_score = self.vector_weight as f64 / (self.rrf_k + rank as f64 + 1.0);
            scores.insert(uri.clone(), (rrf_score, ScoreSource::Vector));
        }

        // Add keyword scores with RRF
        for (rank, (uri, _)) in keyword_results.iter().enumerate() {
            let rrf_score = self.keyword_weight as f64 / (self.rrf_k + rank as f64 + 1.0);

            match scores.get(uri).cloned() {
                Some((existing_score, _)) => {
                    let new_score = existing_score + rrf_score;
                    scores.insert(uri.clone(), (new_score, ScoreSource::Fused));
                }
                None => {
                    scores.insert(uri.clone(), (rrf_score, ScoreSource::Keyword));
                }
            }
        }

        self.to_sorted_entities(scores, max_results)
    }

    /// Linear combination of normalized scores
    fn linear_combination(
        &self,
        vector_results: &[(String, f32)],
        keyword_results: &[(String, f32)],
        max_results: usize,
    ) -> GraphRAGResult<Vec<ScoredEntity>> {
        let mut scores: HashMap<String, (f64, ScoreSource)> = HashMap::new();

        // Normalize and add vector scores
        let max_vector = vector_results
            .first()
            .map(|(_, s)| *s)
            .unwrap_or(1.0)
            .max(0.001);
        for (uri, score) in vector_results {
            let normalized = (*score as f64 / max_vector as f64) * self.vector_weight as f64;
            scores.insert(uri.clone(), (normalized, ScoreSource::Vector));
        }

        // Normalize and add keyword scores
        let max_keyword = keyword_results
            .first()
            .map(|(_, s)| *s)
            .unwrap_or(1.0)
            .max(0.001);
        for (uri, score) in keyword_results {
            let normalized = (*score as f64 / max_keyword as f64) * self.keyword_weight as f64;

            match scores.get(uri).cloned() {
                Some((existing_score, _)) => {
                    let new_score = existing_score + normalized;
                    scores.insert(uri.clone(), (new_score, ScoreSource::Fused));
                }
                None => {
                    scores.insert(uri.clone(), (normalized, ScoreSource::Keyword));
                }
            }
        }

        self.to_sorted_entities(scores, max_results)
    }

    /// Take highest score per entity
    fn highest_score(
        &self,
        vector_results: &[(String, f32)],
        keyword_results: &[(String, f32)],
        max_results: usize,
    ) -> GraphRAGResult<Vec<ScoredEntity>> {
        let mut scores: HashMap<String, (f64, ScoreSource)> = HashMap::new();

        // Add vector scores
        for (uri, score) in vector_results {
            let weighted = *score as f64 * self.vector_weight as f64;
            scores.insert(uri.clone(), (weighted, ScoreSource::Vector));
        }

        // Add keyword scores, keeping max
        for (uri, score) in keyword_results {
            let weighted = *score as f64 * self.keyword_weight as f64;

            if let Some((existing_score, _)) = scores.get(uri) {
                if weighted > *existing_score {
                    scores.insert(uri.clone(), (weighted, ScoreSource::Keyword));
                }
            } else {
                scores.insert(uri.clone(), (weighted, ScoreSource::Keyword));
            }
        }

        self.to_sorted_entities(scores, max_results)
    }

    /// Geometric mean of scores
    fn geometric_mean(
        &self,
        vector_results: &[(String, f32)],
        keyword_results: &[(String, f32)],
        max_results: usize,
    ) -> GraphRAGResult<Vec<ScoredEntity>> {
        let vector_map: HashMap<String, f32> = vector_results.iter().cloned().collect();
        let keyword_map: HashMap<String, f32> = keyword_results.iter().cloned().collect();

        let mut scores: HashMap<String, (f64, ScoreSource)> = HashMap::new();

        // For entities in both lists
        for (uri, v_score) in &vector_map {
            if let Some(k_score) = keyword_map.get(uri) {
                let geo_mean = ((*v_score as f64) * (*k_score as f64)).sqrt();
                scores.insert(uri.clone(), (geo_mean, ScoreSource::Fused));
            } else {
                scores.insert(
                    uri.clone(),
                    (
                        *v_score as f64 * self.vector_weight as f64,
                        ScoreSource::Vector,
                    ),
                );
            }
        }

        // Add keyword-only entities
        for (uri, k_score) in &keyword_map {
            if !vector_map.contains_key(uri) {
                scores.insert(
                    uri.clone(),
                    (
                        *k_score as f64 * self.keyword_weight as f64,
                        ScoreSource::Keyword,
                    ),
                );
            }
        }

        self.to_sorted_entities(scores, max_results)
    }

    /// Harmonic mean of scores
    fn harmonic_mean(
        &self,
        vector_results: &[(String, f32)],
        keyword_results: &[(String, f32)],
        max_results: usize,
    ) -> GraphRAGResult<Vec<ScoredEntity>> {
        let vector_map: HashMap<String, f32> = vector_results.iter().cloned().collect();
        let keyword_map: HashMap<String, f32> = keyword_results.iter().cloned().collect();

        let mut scores: HashMap<String, (f64, ScoreSource)> = HashMap::new();

        // For entities in both lists
        for (uri, v_score) in &vector_map {
            if let Some(k_score) = keyword_map.get(uri) {
                let v = *v_score as f64;
                let k = *k_score as f64;
                let harmonic = if v > 0.0 && k > 0.0 {
                    2.0 * v * k / (v + k)
                } else {
                    0.0
                };
                scores.insert(uri.clone(), (harmonic, ScoreSource::Fused));
            } else {
                scores.insert(
                    uri.clone(),
                    (
                        *v_score as f64 * self.vector_weight as f64,
                        ScoreSource::Vector,
                    ),
                );
            }
        }

        // Add keyword-only entities
        for (uri, k_score) in &keyword_map {
            if !vector_map.contains_key(uri) {
                scores.insert(
                    uri.clone(),
                    (
                        *k_score as f64 * self.keyword_weight as f64,
                        ScoreSource::Keyword,
                    ),
                );
            }
        }

        self.to_sorted_entities(scores, max_results)
    }

    /// Convert scores map to sorted entity vector
    fn to_sorted_entities(
        &self,
        scores: HashMap<String, (f64, ScoreSource)>,
        max_results: usize,
    ) -> GraphRAGResult<Vec<ScoredEntity>> {
        let mut entities: Vec<ScoredEntity> = scores
            .into_iter()
            .map(|(uri, (score, source))| ScoredEntity {
                uri,
                score,
                source,
                metadata: HashMap::new(),
            })
            .collect();

        entities.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entities.truncate(max_results);

        Ok(entities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion() {
        let fuser = ResultFuser::new(FusionStrategy::ReciprocalRankFusion, 0.7, 0.3);

        let vector = vec![
            ("http://a".to_string(), 0.9),
            ("http://b".to_string(), 0.8),
            ("http://c".to_string(), 0.7),
        ];

        let keyword = vec![
            ("http://b".to_string(), 5.0),
            ("http://d".to_string(), 4.0),
            ("http://a".to_string(), 3.0),
        ];

        let results = fuser.fuse(&vector, &keyword, 10).unwrap();

        assert!(!results.is_empty());
        // 'b' should be top since it's in both lists
        assert!(results
            .iter()
            .any(|e| e.uri == "http://b" && e.source == ScoreSource::Fused));
    }

    #[test]
    fn test_linear_combination() {
        let fuser = ResultFuser::new(FusionStrategy::LinearCombination, 0.5, 0.5);

        let vector = vec![("http://a".to_string(), 1.0)];
        let keyword = vec![("http://a".to_string(), 1.0)];

        let results = fuser.fuse(&vector, &keyword, 10).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, ScoreSource::Fused);
    }
}
