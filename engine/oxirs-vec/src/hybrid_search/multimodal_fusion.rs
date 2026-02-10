//! Multimodal search fusion for combining text, vector, and spatial search results
//!
//! This module provides advanced fusion strategies for combining results from multiple
//! search modalities: text (keyword/BM25), vector (semantic similarity), and spatial
//! (geographic queries). It implements four fusion strategies with score normalization.

use super::types::DocumentScore;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multimodal fusion engine for combining text, vector, and spatial search
pub struct MultimodalFusion {
    config: FusionConfig,
}

/// Configuration for multimodal fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Default fusion strategy to use
    pub default_strategy: FusionStrategy,
    /// Score normalization method
    pub score_normalization: NormalizationMethod,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            default_strategy: FusionStrategy::RankFusion,
            score_normalization: NormalizationMethod::MinMax,
        }
    }
}

/// Fusion strategy for combining multiple modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Weighted linear combination of normalized scores
    Weighted { weights: Vec<f64> },
    /// Sequential filtering: filter with one modality, rank with another
    Sequential { order: Vec<Modality> },
    /// Cascade: progressive filtering with thresholds (fast → expensive)
    Cascade { thresholds: Vec<f64> },
    /// Reciprocal Rank Fusion (RRF) - position-based fusion
    RankFusion,
}

/// Search modality type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Text/keyword search (BM25, TF-IDF)
    Text,
    /// Vector/semantic search (embeddings)
    Vector,
    /// Spatial/geographic search (GeoSPARQL)
    Spatial,
}

/// Score normalization method
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Z-score normalization (mean=0, std=1)
    ZScore,
    /// Sigmoid normalization to (0, 1)
    Sigmoid,
}

/// Result from multimodal fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedResult {
    /// Resource URI
    pub uri: String,
    /// Individual scores per modality
    pub scores: HashMap<Modality, f64>,
    /// Final combined score
    pub total_score: f64,
}

impl FusedResult {
    /// Create a new fused result
    pub fn new(uri: String) -> Self {
        Self {
            uri,
            scores: HashMap::new(),
            total_score: 0.0,
        }
    }

    /// Add a score for a specific modality
    pub fn add_score(&mut self, modality: Modality, score: f64) {
        *self.scores.entry(modality).or_insert(0.0) += score;
    }

    /// Calculate total score from individual scores
    pub fn calculate_total(&mut self) {
        self.total_score = self.scores.values().sum();
    }

    /// Get score for a specific modality
    pub fn get_score(&self, modality: Modality) -> Option<f64> {
        self.scores.get(&modality).copied()
    }
}

impl MultimodalFusion {
    /// Create a new multimodal fusion engine with default configuration
    pub fn new(config: FusionConfig) -> Self {
        Self { config }
    }

    /// Fuse results from multiple modalities
    ///
    /// # Arguments
    /// * `text_results` - Results from text/keyword search
    /// * `vector_results` - Results from vector/semantic search
    /// * `spatial_results` - Results from spatial/geographic search
    /// * `strategy` - Optional fusion strategy (uses default if None)
    ///
    /// # Returns
    /// Fused results sorted by combined score (descending)
    pub fn fuse(
        &self,
        text_results: &[DocumentScore],
        vector_results: &[DocumentScore],
        spatial_results: &[DocumentScore],
        strategy: Option<FusionStrategy>,
    ) -> Result<Vec<FusedResult>> {
        let strat = strategy.unwrap_or_else(|| self.config.default_strategy.clone());

        match strat {
            FusionStrategy::Weighted { weights } => {
                self.fuse_weighted(text_results, vector_results, spatial_results, &weights)
            }
            FusionStrategy::Sequential { order } => {
                self.fuse_sequential(text_results, vector_results, spatial_results, &order)
            }
            FusionStrategy::Cascade { thresholds } => {
                self.fuse_cascade(text_results, vector_results, spatial_results, &thresholds)
            }
            FusionStrategy::RankFusion => {
                self.fuse_rank(text_results, vector_results, spatial_results)
            }
        }
    }

    /// Weighted fusion: Linear combination of normalized scores
    ///
    /// Formula: score(d) = w1·norm(text(d)) + w2·norm(vector(d)) + w3·norm(spatial(d))
    fn fuse_weighted(
        &self,
        text: &[DocumentScore],
        vector: &[DocumentScore],
        spatial: &[DocumentScore],
        weights: &[f64],
    ) -> Result<Vec<FusedResult>> {
        if weights.len() != 3 {
            anyhow::bail!("Weighted fusion requires exactly 3 weights (text, vector, spatial)");
        }

        // Normalize scores to [0, 1]
        let text_norm = self.normalize_scores(text)?;
        let vector_norm = self.normalize_scores(vector)?;
        let spatial_norm = self.normalize_scores(spatial)?;

        // Merge by entity URI
        let mut combined: HashMap<String, FusedResult> = HashMap::new();

        // Add text scores
        for (result, score) in text.iter().zip(text_norm.iter()) {
            combined
                .entry(result.doc_id.clone())
                .or_insert_with(|| FusedResult::new(result.doc_id.clone()))
                .add_score(Modality::Text, score * weights[0]);
        }

        // Add vector scores
        for (result, score) in vector.iter().zip(vector_norm.iter()) {
            combined
                .entry(result.doc_id.clone())
                .or_insert_with(|| FusedResult::new(result.doc_id.clone()))
                .add_score(Modality::Vector, score * weights[1]);
        }

        // Add spatial scores
        for (result, score) in spatial.iter().zip(spatial_norm.iter()) {
            combined
                .entry(result.doc_id.clone())
                .or_insert_with(|| FusedResult::new(result.doc_id.clone()))
                .add_score(Modality::Spatial, score * weights[2]);
        }

        // Calculate total scores and sort
        let mut results: Vec<FusedResult> = combined
            .into_values()
            .map(|mut r| {
                r.calculate_total();
                r
            })
            .collect();

        results.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Sequential fusion: Filter with one modality, rank with another
    ///
    /// Example: Filter with text (fast), rank with vector (accurate)
    fn fuse_sequential(
        &self,
        text: &[DocumentScore],
        vector: &[DocumentScore],
        spatial: &[DocumentScore],
        order: &[Modality],
    ) -> Result<Vec<FusedResult>> {
        if order.len() < 2 {
            anyhow::bail!("Sequential fusion requires at least 2 modalities in order");
        }

        // Get filter results (first modality)
        let filter_results = match order[0] {
            Modality::Text => text,
            Modality::Vector => vector,
            Modality::Spatial => spatial,
        };

        // Create candidate set from filter
        let candidates: HashMap<String, ()> = filter_results
            .iter()
            .map(|r| (r.doc_id.clone(), ()))
            .collect();

        // Get rank results (second modality)
        let rank_results = match order[1] {
            Modality::Text => text,
            Modality::Vector => vector,
            Modality::Spatial => spatial,
        };

        // Normalize ranking scores
        let rank_norm = self.normalize_scores(rank_results)?;

        // Filter and create results
        let mut results: Vec<FusedResult> = rank_results
            .iter()
            .zip(rank_norm.iter())
            .filter(|(r, _)| candidates.contains_key(&r.doc_id))
            .map(|(r, score)| {
                let mut result = FusedResult::new(r.doc_id.clone());
                result.add_score(order[1], *score);
                result.calculate_total();
                result
            })
            .collect();

        results.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Cascade fusion: Progressive filtering (fast → expensive)
    ///
    /// Example: Text (threshold 0.5) → Vector (threshold 0.7) → Spatial (threshold 0.8)
    fn fuse_cascade(
        &self,
        text: &[DocumentScore],
        vector: &[DocumentScore],
        spatial: &[DocumentScore],
        thresholds: &[f64],
    ) -> Result<Vec<FusedResult>> {
        if thresholds.len() != 3 {
            anyhow::bail!("Cascade fusion requires exactly 3 thresholds (text, vector, spatial)");
        }

        // Stage 1: Fast text search with threshold
        let text_norm = self.normalize_scores(text)?;
        let mut candidates: HashMap<String, f64> = text
            .iter()
            .zip(text_norm.iter())
            .filter(|(_, score)| **score >= thresholds[0])
            .map(|(r, score)| (r.doc_id.clone(), *score))
            .collect();

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Stage 2: Vector search on candidates with threshold
        let vector_norm = self.normalize_scores(vector)?;
        let vector_map: HashMap<String, f64> = vector
            .iter()
            .zip(vector_norm.iter())
            .filter(|(r, score)| {
                candidates.contains_key(&r.doc_id) && **score >= thresholds[1]
            })
            .map(|(r, score)| (r.doc_id.clone(), *score))
            .collect();

        // Keep only candidates that passed vector threshold
        candidates.retain(|uri, _| vector_map.contains_key(uri));

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Stage 3: Expensive spatial search on finalists with threshold
        let spatial_norm = self.normalize_scores(spatial)?;
        let mut results: Vec<FusedResult> = spatial
            .iter()
            .zip(spatial_norm.iter())
            .filter(|(r, score)| {
                candidates.contains_key(&r.doc_id) && **score >= thresholds[2]
            })
            .map(|(r, score)| {
                let mut result = FusedResult::new(r.doc_id.clone());
                result.add_score(Modality::Spatial, *score);
                if let Some(&text_score) = candidates.get(&r.doc_id) {
                    result.add_score(Modality::Text, text_score);
                }
                if let Some(&vec_score) = vector_map.get(&r.doc_id) {
                    result.add_score(Modality::Vector, vec_score);
                }
                result.calculate_total();
                result
            })
            .collect();

        results.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Reciprocal Rank Fusion (RRF)
    ///
    /// Formula: RRF(d) = Σ 1/(K + rank(d))
    /// where K=60 is a standard constant
    fn fuse_rank(
        &self,
        text: &[DocumentScore],
        vector: &[DocumentScore],
        spatial: &[DocumentScore],
    ) -> Result<Vec<FusedResult>> {
        const K: f64 = 60.0; // Standard RRF constant

        let mut rrf_scores: HashMap<String, f64> = HashMap::new();

        // Add RRF scores from text results
        for (rank, result) in text.iter().enumerate() {
            *rrf_scores.entry(result.doc_id.clone()).or_insert(0.0) +=
                1.0 / (K + rank as f64 + 1.0);
        }

        // Add RRF scores from vector results
        for (rank, result) in vector.iter().enumerate() {
            *rrf_scores.entry(result.doc_id.clone()).or_insert(0.0) +=
                1.0 / (K + rank as f64 + 1.0);
        }

        // Add RRF scores from spatial results
        for (rank, result) in spatial.iter().enumerate() {
            *rrf_scores.entry(result.doc_id.clone()).or_insert(0.0) +=
                1.0 / (K + rank as f64 + 1.0);
        }

        let mut results: Vec<FusedResult> = rrf_scores
            .into_iter()
            .map(|(uri, score)| {
                let mut result = FusedResult::new(uri);
                result.total_score = score;
                // RRF produces a unified score, store it as Text modality for consistency
                result.scores.insert(Modality::Text, score);
                result
            })
            .collect();

        results.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Normalize scores to [0, 1] range using configured method
    pub fn normalize_scores(&self, results: &[DocumentScore]) -> Result<Vec<f64>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        let scores: Vec<f64> = results.iter().map(|r| r.score as f64).collect();

        match self.config.score_normalization {
            NormalizationMethod::MinMax => self.min_max_normalize(&scores),
            NormalizationMethod::ZScore => self.z_score_normalize(&scores),
            NormalizationMethod::Sigmoid => self.sigmoid_normalize(&scores),
        }
    }

    /// Min-max normalization: (x - min) / (max - min)
    fn min_max_normalize(&self, scores: &[f64]) -> Result<Vec<f64>> {
        if scores.is_empty() {
            return Ok(Vec::new());
        }

        let min_score = scores
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let max_score = scores
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);

        let range = (max_score - min_score).max(1e-10); // Avoid division by zero

        Ok(scores.iter().map(|&s| (s - min_score) / range).collect())
    }

    /// Z-score normalization: (x - mean) / std
    fn z_score_normalize(&self, scores: &[f64]) -> Result<Vec<f64>> {
        if scores.is_empty() {
            return Ok(Vec::new());
        }

        let n = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n;

        let variance = scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt().max(1e-10); // Avoid division by zero

        Ok(scores.iter().map(|&s| (s - mean) / std).collect())
    }

    /// Sigmoid normalization: 1 / (1 + exp(-x))
    fn sigmoid_normalize(&self, scores: &[f64]) -> Result<Vec<f64>> {
        Ok(scores.iter().map(|&s| 1.0 / (1.0 + (-s).exp())).collect())
    }

    /// Get the current configuration
    pub fn config(&self) -> &FusionConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: FusionConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_results() -> (Vec<DocumentScore>, Vec<DocumentScore>, Vec<DocumentScore>) {
        let text = vec![
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

        let vector = vec![
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

        let spatial = vec![
            DocumentScore {
                doc_id: "doc3".to_string(),
                score: 0.99,
                rank: 0,
            },
            DocumentScore {
                doc_id: "doc1".to_string(),
                score: 0.92,
                rank: 1,
            },
            DocumentScore {
                doc_id: "doc5".to_string(),
                score: 0.88,
                rank: 2,
            },
        ];

        (text, vector, spatial)
    }

    #[test]
    fn test_weighted_fusion() {
        let (text, vector, spatial) = create_test_results();
        let fusion = MultimodalFusion::new(FusionConfig::default());

        let weights = vec![0.4, 0.4, 0.2]; // Text, Vector, Spatial
        let strategy = FusionStrategy::Weighted { weights };

        let results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

        assert!(!results.is_empty());
        assert!(results[0].total_score > 0.0);
        // doc1 appears in all three lists, should have high score
        let doc1 = results.iter().find(|r| r.uri == "doc1").unwrap();
        assert!(doc1.scores.len() == 3);
    }

    #[test]
    fn test_sequential_fusion() {
        let (text, vector, spatial) = create_test_results();
        let fusion = MultimodalFusion::new(FusionConfig::default());

        let order = vec![Modality::Text, Modality::Vector];
        let strategy = FusionStrategy::Sequential { order };

        let results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

        assert!(!results.is_empty());
        // Should only include docs that passed text filter
        assert!(results.iter().all(|r| ["doc1", "doc2", "doc3"].contains(&r.uri.as_str())));
    }

    #[test]
    fn test_cascade_fusion() {
        let (text, vector, spatial) = create_test_results();
        let fusion = MultimodalFusion::new(FusionConfig::default());

        let thresholds = vec![0.0, 0.0, 0.0]; // Accept all for testing
        let strategy = FusionStrategy::Cascade { thresholds };

        let results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

        assert!(!results.is_empty());
        // Should have scores from multiple modalities
        if let Some(doc1) = results.iter().find(|r| r.uri == "doc1") {
            assert!(doc1.scores.len() >= 2);
        }
    }

    #[test]
    fn test_rank_fusion() {
        let (text, vector, spatial) = create_test_results();
        let fusion = MultimodalFusion::new(FusionConfig::default());

        let strategy = FusionStrategy::RankFusion;
        let results = fusion.fuse(&text, &vector, &spatial, Some(strategy)).unwrap();

        assert!(!results.is_empty());
        // doc1 appears in all three lists at good positions
        let doc1 = results.iter().find(|r| r.uri == "doc1").unwrap();
        // doc4 appears only in vector list
        let doc4 = results.iter().find(|r| r.uri == "doc4").unwrap();
        // doc1 should have higher RRF score
        assert!(doc1.total_score > doc4.total_score);
    }

    #[test]
    fn test_min_max_normalization() {
        let fusion = MultimodalFusion::new(FusionConfig::default());
        let scores = vec![10.0, 5.0, 0.0];

        let normalized = fusion.min_max_normalize(&scores).unwrap();

        assert!((normalized[0] - 1.0).abs() < 1e-6);
        assert!((normalized[1] - 0.5).abs() < 1e-6);
        assert!((normalized[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_z_score_normalization() {
        let fusion = MultimodalFusion::new(FusionConfig::default());
        let scores = vec![10.0, 5.0, 0.0];

        let normalized = fusion.z_score_normalize(&scores).unwrap();

        // Mean should be ~5.0
        // Z-scores should have mean ~0
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_normalization() {
        let fusion = MultimodalFusion::new(FusionConfig::default());
        let scores = vec![0.0, 1.0, -1.0];

        let normalized = fusion.sigmoid_normalize(&scores).unwrap();

        // Sigmoid of 0 should be 0.5
        assert!((normalized[0] - 0.5).abs() < 1e-6);
        // All values should be in (0, 1)
        assert!(normalized.iter().all(|&s| s > 0.0 && s < 1.0));
    }

    #[test]
    fn test_empty_results() {
        let fusion = MultimodalFusion::new(FusionConfig::default());
        let empty: Vec<DocumentScore> = Vec::new();

        let strategy = FusionStrategy::RankFusion;
        let results = fusion.fuse(&empty, &empty, &empty, Some(strategy)).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_fused_result_operations() {
        let mut result = FusedResult::new("test_doc".to_string());

        result.add_score(Modality::Text, 0.5);
        result.add_score(Modality::Vector, 0.3);
        result.add_score(Modality::Spatial, 0.2);

        assert_eq!(result.get_score(Modality::Text), Some(0.5));
        assert_eq!(result.get_score(Modality::Vector), Some(0.3));
        assert_eq!(result.get_score(Modality::Spatial), Some(0.2));

        result.calculate_total();
        assert!((result.total_score - 1.0).abs() < 1e-6);
    }
}
