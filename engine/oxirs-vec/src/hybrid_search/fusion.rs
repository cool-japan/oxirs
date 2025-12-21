//! Rank fusion strategies for combining keyword and semantic results

use super::config::RankFusionStrategy;
use super::types::{DocumentScore, HybridResult, SearchWeights};
use std::collections::{HashMap, HashSet};

/// Rank fusion for combining multiple ranked lists
pub struct RankFusion {
    strategy: RankFusionStrategy,
}

impl RankFusion {
    /// Create a new rank fusion
    pub fn new(strategy: RankFusionStrategy) -> Self {
        Self { strategy }
    }

    /// Fuse keyword and semantic results
    pub fn fuse(
        &self,
        keyword_results: Vec<DocumentScore>,
        semantic_results: Vec<DocumentScore>,
        weights: &SearchWeights,
    ) -> Vec<HybridResult> {
        match self.strategy {
            RankFusionStrategy::WeightedSum => {
                self.weighted_sum(keyword_results, semantic_results, weights)
            }
            RankFusionStrategy::ReciprocalRankFusion => {
                self.reciprocal_rank_fusion(keyword_results, semantic_results)
            }
            RankFusionStrategy::Cascade => self.cascade(keyword_results, semantic_results, weights),
            RankFusionStrategy::Interleave => {
                self.interleave(keyword_results, semantic_results, weights)
            }
        }
    }

    /// Weighted sum of normalized scores
    fn weighted_sum(
        &self,
        keyword_results: Vec<DocumentScore>,
        semantic_results: Vec<DocumentScore>,
        weights: &SearchWeights,
    ) -> Vec<HybridResult> {
        // Normalize scores to [0, 1]
        let keyword_norm = Self::normalize_scores(&keyword_results);
        let semantic_norm = Self::normalize_scores(&semantic_results);

        // Combine scores
        let mut combined: HashMap<String, HybridResult> = HashMap::new();

        for doc in keyword_norm {
            let result = HybridResult::new(doc.doc_id.clone(), doc.score, 0.0, 0.0, weights);
            combined.insert(doc.doc_id, result);
        }

        for doc in semantic_norm {
            combined
                .entry(doc.doc_id.clone())
                .and_modify(|r| {
                    r.score_breakdown.semantic_score = doc.score;
                    r.score = doc.score * weights.semantic_weight
                        + r.score_breakdown.keyword_score * weights.keyword_weight;
                })
                .or_insert_with(|| {
                    HybridResult::new(doc.doc_id.clone(), 0.0, doc.score, 0.0, weights)
                });
        }

        let mut results: Vec<HybridResult> = combined.into_values().collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    /// Reciprocal Rank Fusion (RRF)
    /// Formula: RRF(d) = sum_r(1 / (k + rank_r(d)))
    fn reciprocal_rank_fusion(
        &self,
        keyword_results: Vec<DocumentScore>,
        semantic_results: Vec<DocumentScore>,
    ) -> Vec<HybridResult> {
        const K: f32 = 60.0; // Standard RRF constant

        let mut rrf_scores: HashMap<String, f32> = HashMap::new();
        let mut keyword_ranks: HashMap<String, usize> = HashMap::new();
        let mut semantic_ranks: HashMap<String, usize> = HashMap::new();

        // Calculate RRF scores from keyword results
        for (rank, doc) in keyword_results.iter().enumerate() {
            let rrf = 1.0 / (K + rank as f32 + 1.0);
            *rrf_scores.entry(doc.doc_id.clone()).or_insert(0.0) += rrf;
            keyword_ranks.insert(doc.doc_id.clone(), rank);
        }

        // Add RRF scores from semantic results
        for (rank, doc) in semantic_results.iter().enumerate() {
            let rrf = 1.0 / (K + rank as f32 + 1.0);
            *rrf_scores.entry(doc.doc_id.clone()).or_insert(0.0) += rrf;
            semantic_ranks.insert(doc.doc_id.clone(), rank);
        }

        // Create hybrid results
        let mut results: Vec<HybridResult> = rrf_scores
            .into_iter()
            .map(|(doc_id, score)| {
                let mut result = HybridResult {
                    doc_id: doc_id.clone(),
                    score,
                    score_breakdown: super::types::ScoreBreakdown {
                        keyword_score: 0.0,
                        semantic_score: 0.0,
                        recency_score: 0.0,
                        keyword_rank: keyword_ranks.get(&doc_id).copied(),
                        semantic_rank: semantic_ranks.get(&doc_id).copied(),
                    },
                    metadata: HashMap::new(),
                };

                // Fill in original scores
                if let Some(kw_doc) = keyword_results.iter().find(|d| d.doc_id == doc_id) {
                    result.score_breakdown.keyword_score = kw_doc.score;
                }
                if let Some(sem_doc) = semantic_results.iter().find(|d| d.doc_id == doc_id) {
                    result.score_breakdown.semantic_score = sem_doc.score;
                }

                result
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    /// Cascade: Filter with keyword, re-rank with semantic
    fn cascade(
        &self,
        keyword_results: Vec<DocumentScore>,
        semantic_results: Vec<DocumentScore>,
        weights: &SearchWeights,
    ) -> Vec<HybridResult> {
        // Create a set of keyword doc IDs
        let keyword_docs: HashSet<String> =
            keyword_results.iter().map(|d| d.doc_id.clone()).collect();

        // Filter semantic results to only include keyword matches
        let semantic_map: HashMap<String, f32> = semantic_results
            .iter()
            .filter(|d| keyword_docs.contains(&d.doc_id))
            .map(|d| (d.doc_id.clone(), d.score))
            .collect();

        // Create hybrid results
        let mut results: Vec<HybridResult> = keyword_results
            .into_iter()
            .map(|doc| {
                let semantic_score = semantic_map.get(&doc.doc_id).copied().unwrap_or(0.0);
                HybridResult::new(doc.doc_id, doc.score, semantic_score, 0.0, weights)
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    /// Interleave results from both methods
    fn interleave(
        &self,
        keyword_results: Vec<DocumentScore>,
        semantic_results: Vec<DocumentScore>,
        weights: &SearchWeights,
    ) -> Vec<HybridResult> {
        let mut results = Vec::new();
        let mut seen = HashSet::new();

        let max_len = keyword_results.len().max(semantic_results.len());

        for i in 0..max_len {
            // Take from keyword results
            if i < keyword_results.len() {
                let doc = &keyword_results[i];
                if !seen.contains(&doc.doc_id) {
                    let semantic_score = semantic_results
                        .iter()
                        .find(|d| d.doc_id == doc.doc_id)
                        .map(|d| d.score)
                        .unwrap_or(0.0);

                    results.push(HybridResult::new(
                        doc.doc_id.clone(),
                        doc.score,
                        semantic_score,
                        0.0,
                        weights,
                    ));
                    seen.insert(doc.doc_id.clone());
                }
            }

            // Take from semantic results
            if i < semantic_results.len() {
                let doc = &semantic_results[i];
                if !seen.contains(&doc.doc_id) {
                    let keyword_score = keyword_results
                        .iter()
                        .find(|d| d.doc_id == doc.doc_id)
                        .map(|d| d.score)
                        .unwrap_or(0.0);

                    results.push(HybridResult::new(
                        doc.doc_id.clone(),
                        keyword_score,
                        doc.score,
                        0.0,
                        weights,
                    ));
                    seen.insert(doc.doc_id.clone());
                }
            }
        }

        results
    }

    /// Normalize scores to [0, 1] range
    fn normalize_scores(results: &[DocumentScore]) -> Vec<DocumentScore> {
        if results.is_empty() {
            return Vec::new();
        }

        let max_score = results
            .iter()
            .map(|d| d.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);

        let min_score = results
            .iter()
            .map(|d| d.score)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let range = (max_score - min_score).max(0.001);

        results
            .iter()
            .map(|d| DocumentScore {
                doc_id: d.doc_id.clone(),
                score: (d.score - min_score) / range,
                rank: d.rank,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_results() -> (Vec<DocumentScore>, Vec<DocumentScore>) {
        let keyword = vec![
            DocumentScore {
                doc_id: "doc1".to_string(),
                score: 10.0,
                rank: 0,
            },
            DocumentScore {
                doc_id: "doc2".to_string(),
                score: 8.0,
                rank: 1,
            },
            DocumentScore {
                doc_id: "doc3".to_string(),
                score: 5.0,
                rank: 2,
            },
        ];

        let semantic = vec![
            DocumentScore {
                doc_id: "doc2".to_string(),
                score: 0.95,
                rank: 0,
            },
            DocumentScore {
                doc_id: "doc4".to_string(),
                score: 0.90,
                rank: 1,
            },
            DocumentScore {
                doc_id: "doc1".to_string(),
                score: 0.85,
                rank: 2,
            },
        ];

        (keyword, semantic)
    }

    #[test]
    fn test_weighted_sum() {
        let (keyword, semantic) = create_test_results();
        let fusion = RankFusion::new(RankFusionStrategy::WeightedSum);
        let weights = SearchWeights::default();

        let results = fusion.fuse(keyword, semantic, &weights);
        assert!(!results.is_empty());
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_reciprocal_rank_fusion() {
        let (keyword, semantic) = create_test_results();
        let fusion = RankFusion::new(RankFusionStrategy::ReciprocalRankFusion);
        let weights = SearchWeights::default();

        let results = fusion.fuse(keyword, semantic, &weights);
        assert!(!results.is_empty());

        // doc1 and doc2 appear in both lists, should have higher RRF scores
        let doc1_score = results.iter().find(|r| r.doc_id == "doc1").unwrap().score;
        let doc4_score = results.iter().find(|r| r.doc_id == "doc4").unwrap().score;
        assert!(doc1_score > doc4_score);
    }

    #[test]
    fn test_cascade() {
        let (keyword, semantic) = create_test_results();
        let fusion = RankFusion::new(RankFusionStrategy::Cascade);
        let weights = SearchWeights::default();

        let results = fusion.fuse(keyword, semantic, &weights);

        // Should only include docs that appear in keyword results
        assert!(results.iter().all(|r| r.doc_id != "doc4"));
    }

    #[test]
    fn test_interleave() {
        let (keyword, semantic) = create_test_results();
        let fusion = RankFusion::new(RankFusionStrategy::Interleave);
        let weights = SearchWeights::default();

        let results = fusion.fuse(keyword, semantic, &weights);

        // Should have all unique documents
        let doc_ids: HashSet<String> = results.iter().map(|r| r.doc_id.clone()).collect();
        assert_eq!(doc_ids.len(), 4);
    }

    #[test]
    fn test_normalize_scores() {
        let results = vec![
            DocumentScore {
                doc_id: "doc1".to_string(),
                score: 10.0,
                rank: 0,
            },
            DocumentScore {
                doc_id: "doc2".to_string(),
                score: 5.0,
                rank: 1,
            },
            DocumentScore {
                doc_id: "doc3".to_string(),
                score: 0.0,
                rank: 2,
            },
        ];

        let normalized = RankFusion::normalize_scores(&results);
        assert!((normalized[0].score - 1.0).abs() < 0.001);
        assert!((normalized[2].score - 0.0).abs() < 0.001);
    }
}
