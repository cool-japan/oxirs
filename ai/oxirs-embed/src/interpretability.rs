//! Model Interpretability Tools
//!
//! This module provides tools for understanding and interpreting knowledge graph
//! embeddings, including attention analysis, embedding similarity, feature importance,
//! and counterfactual explanations.

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

/// Interpretation method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpretationMethod {
    /// Analyze embedding similarities
    SimilarityAnalysis,
    /// Feature importance (gradient-based)
    FeatureImportance,
    /// Counterfactual explanations
    Counterfactual,
    /// Nearest neighbors analysis
    NearestNeighbors,
}

/// Interpretability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityConfig {
    /// Interpretation method
    pub method: InterpretationMethod,
    /// Top-K most important features/neighbors
    pub top_k: usize,
    /// Similarity threshold
    pub similarity_threshold: f32,
    /// Enable detailed analysis
    pub detailed: bool,
}

impl Default for InterpretabilityConfig {
    fn default() -> Self {
        Self {
            method: InterpretationMethod::SimilarityAnalysis,
            top_k: 10,
            similarity_threshold: 0.7,
            detailed: false,
        }
    }
}

/// Similarity analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityAnalysis {
    /// Entity being analyzed
    pub entity: String,
    /// Most similar entities with scores
    pub similar_entities: Vec<(String, f32)>,
    /// Least similar entities with scores
    pub dissimilar_entities: Vec<(String, f32)>,
    /// Average similarity to all other entities
    pub avg_similarity: f32,
}

/// Feature importance result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Entity being analyzed
    pub entity: String,
    /// Feature indices and their importance scores
    pub important_features: Vec<(usize, f32)>,
    /// Feature statistics
    pub feature_stats: FeatureStats,
}

/// Feature statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    /// Mean feature values
    pub mean: Vec<f32>,
    /// Standard deviation of features
    pub std: Vec<f32>,
    /// Min feature values
    pub min: Vec<f32>,
    /// Max feature values
    pub max: Vec<f32>,
}

/// Counterfactual explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualExplanation {
    /// Original entity
    pub original: String,
    /// Target entity (for comparison)
    pub target: String,
    /// Dimensions that need to change
    pub required_changes: Vec<(usize, f32, f32)>, // (dim, from, to)
    /// Estimated difficulty (0-1, higher is harder)
    pub difficulty: f32,
}

/// Nearest neighbors analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearestNeighborsAnalysis {
    /// Entity being analyzed
    pub entity: String,
    /// Nearest neighbors with distances
    pub neighbors: Vec<(String, f32)>,
    /// Neighbor clusters (if detected)
    pub neighbor_clusters: Vec<Vec<String>>,
}

/// Model interpretability analyzer
pub struct InterpretabilityAnalyzer {
    config: InterpretabilityConfig,
}

impl InterpretabilityAnalyzer {
    /// Create new interpretability analyzer
    pub fn new(config: InterpretabilityConfig) -> Self {
        info!(
            "Initialized interpretability analyzer: method={:?}, top_k={}",
            config.method, config.top_k
        );

        Self { config }
    }

    /// Analyze a specific entity
    pub fn analyze_entity(
        &self,
        entity: &str,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<String> {
        if !embeddings.contains_key(entity) {
            return Err(anyhow!("Entity not found: {}", entity));
        }

        match self.config.method {
            InterpretationMethod::SimilarityAnalysis => {
                let analysis = self.similarity_analysis(entity, embeddings)?;
                Ok(serde_json::to_string_pretty(&analysis)?)
            }
            InterpretationMethod::FeatureImportance => {
                let importance = self.feature_importance(entity, embeddings)?;
                Ok(serde_json::to_string_pretty(&importance)?)
            }
            InterpretationMethod::NearestNeighbors => {
                let neighbors = self.nearest_neighbors_analysis(entity, embeddings)?;
                Ok(serde_json::to_string_pretty(&neighbors)?)
            }
            InterpretationMethod::Counterfactual => {
                Err(anyhow!("Counterfactual requires target entity"))
            }
        }
    }

    /// Analyze similarity between entities
    pub fn similarity_analysis(
        &self,
        entity: &str,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<SimilarityAnalysis> {
        let entity_emb = &embeddings[entity];

        // Compute similarities to all other entities
        let mut similarities: Vec<(String, f32)> = embeddings
            .par_iter()
            .filter(|(e, _)| *e != entity)
            .map(|(other, other_emb)| {
                let sim = self.cosine_similarity(entity_emb, other_emb);
                (other.clone(), sim)
            })
            .collect();

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Get top-K most similar
        let similar_entities: Vec<(String, f32)> = similarities
            .iter()
            .take(self.config.top_k)
            .cloned()
            .collect();

        // Get top-K least similar
        let mut dissimilar_entities: Vec<(String, f32)> = similarities
            .iter()
            .rev()
            .take(self.config.top_k)
            .cloned()
            .collect();
        dissimilar_entities.reverse();

        // Compute average similarity
        let avg_similarity =
            similarities.iter().map(|(_, sim)| sim).sum::<f32>() / similarities.len() as f32;

        info!(
            "Similarity analysis for '{}': avg_similarity={:.4}",
            entity, avg_similarity
        );

        Ok(SimilarityAnalysis {
            entity: entity.to_string(),
            similar_entities,
            dissimilar_entities,
            avg_similarity,
        })
    }

    /// Analyze feature importance for an entity
    pub fn feature_importance(
        &self,
        entity: &str,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<FeatureImportance> {
        let entity_emb = &embeddings[entity];
        let dim = entity_emb.len();

        // Compute global feature statistics
        let feature_stats = self.compute_feature_stats(embeddings);

        // Compute importance as deviation from mean
        let mut important_features: Vec<(usize, f32)> = (0..dim)
            .map(|i| {
                let value = entity_emb[i];
                let mean = feature_stats.mean[i];
                let std = feature_stats.std[i];

                // Z-score based importance
                let importance = if std > 0.0 {
                    ((value - mean) / std).abs()
                } else {
                    0.0
                };

                (i, importance)
            })
            .collect();

        // Sort by importance descending
        important_features
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top-K
        important_features.truncate(self.config.top_k);

        info!(
            "Feature importance for '{}': top feature has importance {:.4}",
            entity,
            important_features
                .first()
                .map(|(_, imp)| *imp)
                .unwrap_or(0.0)
        );

        Ok(FeatureImportance {
            entity: entity.to_string(),
            important_features,
            feature_stats,
        })
    }

    /// Generate counterfactual explanation
    pub fn counterfactual_explanation(
        &self,
        original: &str,
        target: &str,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<CounterfactualExplanation> {
        let original_emb = embeddings
            .get(original)
            .ok_or_else(|| anyhow!("Original entity not found"))?;

        let target_emb = embeddings
            .get(target)
            .ok_or_else(|| anyhow!("Target entity not found"))?;

        // Identify dimensions that differ significantly
        let mut required_changes = Vec::new();
        let mut total_change = 0.0;

        for i in 0..original_emb.len() {
            let diff = (target_emb[i] - original_emb[i]).abs();
            if diff > 0.1 {
                // Threshold for significance
                required_changes.push((i, original_emb[i], target_emb[i]));
                total_change += diff;
            }
        }

        // Sort by magnitude of change
        required_changes.sort_by(|a, b| {
            let diff_a = (a.2 - a.1).abs();
            let diff_b = (b.2 - b.1).abs();
            diff_b
                .partial_cmp(&diff_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep top-K most important changes
        required_changes.truncate(self.config.top_k);

        // Compute difficulty (normalized by embedding norm)
        let norm = original_emb.dot(original_emb).sqrt();
        let difficulty = if norm > 0.0 {
            (total_change / norm).min(1.0)
        } else {
            1.0
        };

        info!(
            "Counterfactual '{}' -> '{}': {} changes, difficulty={:.4}",
            original,
            target,
            required_changes.len(),
            difficulty
        );

        Ok(CounterfactualExplanation {
            original: original.to_string(),
            target: target.to_string(),
            required_changes,
            difficulty,
        })
    }

    /// Analyze nearest neighbors
    pub fn nearest_neighbors_analysis(
        &self,
        entity: &str,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<NearestNeighborsAnalysis> {
        let entity_emb = &embeddings[entity];

        // Find nearest neighbors
        let mut distances: Vec<(String, f32)> = embeddings
            .par_iter()
            .filter(|(e, _)| *e != entity)
            .map(|(other, other_emb)| {
                let dist = self.euclidean_distance(entity_emb, other_emb);
                (other.clone(), dist)
            })
            .collect();

        // Sort by distance ascending
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Get top-K nearest neighbors
        let neighbors: Vec<(String, f32)> =
            distances.iter().take(self.config.top_k).cloned().collect();

        // Attempt to cluster neighbors (simple distance-based clustering)
        let neighbor_clusters = if self.config.detailed {
            self.cluster_neighbors(&neighbors, embeddings)
        } else {
            vec![]
        };

        info!(
            "Nearest neighbors for '{}': closest neighbor at distance {:.4}",
            entity,
            neighbors.first().map(|(_, d)| *d).unwrap_or(0.0)
        );

        Ok(NearestNeighborsAnalysis {
            entity: entity.to_string(),
            neighbors,
            neighbor_clusters,
        })
    }

    /// Batch analysis for multiple entities
    pub fn batch_analysis(
        &self,
        entities: &[String],
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<HashMap<String, String>> {
        let results: HashMap<String, String> = entities
            .par_iter()
            .filter_map(|entity| {
                self.analyze_entity(entity, embeddings)
                    .ok()
                    .map(|analysis| (entity.clone(), analysis))
            })
            .collect();

        Ok(results)
    }

    /// Compute global feature statistics
    fn compute_feature_stats(&self, embeddings: &HashMap<String, Array1<f32>>) -> FeatureStats {
        let n = embeddings.len() as f32;
        let dim = embeddings
            .values()
            .next()
            .expect("embeddings should not be empty")
            .len();

        let mut mean = vec![0.0; dim];
        let mut m2 = vec![0.0; dim]; // For variance calculation
        let mut min = vec![f32::INFINITY; dim];
        let mut max = vec![f32::NEG_INFINITY; dim];

        // Welford's online algorithm for mean and variance
        for (count, emb) in embeddings.values().enumerate() {
            let count_f = (count + 1) as f32;

            for i in 0..dim {
                let value = emb[i];

                // Update min/max
                min[i] = min[i].min(value);
                max[i] = max[i].max(value);

                // Update mean and M2
                let delta = value - mean[i];
                mean[i] += delta / count_f;
                let delta2 = value - mean[i];
                m2[i] += delta * delta2;
            }
        }

        // Compute standard deviation
        let std: Vec<f32> = m2.iter().map(|&m2_val| (m2_val / n).sqrt()).collect();

        FeatureStats {
            mean,
            std,
            min,
            max,
        }
    }

    /// Cluster neighbors based on distance
    fn cluster_neighbors(
        &self,
        neighbors: &[(String, f32)],
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Vec<Vec<String>> {
        if neighbors.len() < 2 {
            return vec![neighbors.iter().map(|(e, _)| e.clone()).collect()];
        }

        // Simple single-linkage clustering
        let mut clusters: Vec<Vec<String>> = Vec::new();
        let distance_threshold = 0.5; // Threshold for clustering

        for (entity, _) in neighbors {
            let entity_emb = &embeddings[entity];
            let mut assigned = false;

            // Try to assign to existing cluster
            for cluster in &mut clusters {
                let cluster_center = cluster
                    .first()
                    .expect("collection validated to be non-empty");
                let center_emb = &embeddings[cluster_center];
                let dist = self.euclidean_distance(entity_emb, center_emb);

                if dist <= distance_threshold {
                    cluster.push(entity.clone());
                    assigned = true;
                    break;
                }
            }

            // Create new cluster if not assigned
            if !assigned {
                clusters.push(vec![entity.clone()]);
            }
        }

        clusters
    }

    /// Cosine similarity between two embeddings
    fn cosine_similarity(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Euclidean distance between two embeddings
    fn euclidean_distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let diff = a - b;
        diff.dot(&diff).sqrt()
    }

    /// Generate interpretation report
    pub fn generate_report(
        &self,
        entity: &str,
        embeddings: &HashMap<String, Array1<f32>>,
    ) -> Result<String> {
        let mut report = String::new();

        report.push_str(&format!("# Interpretability Report for '{}'\n\n", entity));

        // Similarity analysis
        if let Ok(sim_analysis) = self.similarity_analysis(entity, embeddings) {
            report.push_str("## Similarity Analysis\n\n");
            report.push_str(&format!(
                "Average similarity: {:.4}\n\n",
                sim_analysis.avg_similarity
            ));

            report.push_str("### Most Similar Entities:\n");
            for (i, (other, score)) in sim_analysis.similar_entities.iter().enumerate() {
                report.push_str(&format!(
                    "{}. {} (similarity: {:.4})\n",
                    i + 1,
                    other,
                    score
                ));
            }

            report.push_str("\n### Least Similar Entities:\n");
            for (i, (other, score)) in sim_analysis.dissimilar_entities.iter().enumerate() {
                report.push_str(&format!(
                    "{}. {} (similarity: {:.4})\n",
                    i + 1,
                    other,
                    score
                ));
            }
            report.push('\n');
        }

        // Feature importance
        if let Ok(feat_importance) = self.feature_importance(entity, embeddings) {
            report.push_str("## Feature Importance\n\n");
            report.push_str("### Top Important Features:\n");
            for (i, (feature_idx, importance)) in
                feat_importance.important_features.iter().enumerate()
            {
                report.push_str(&format!(
                    "{}. Dimension {} (importance: {:.4})\n",
                    i + 1,
                    feature_idx,
                    importance
                ));
            }
            report.push('\n');
        }

        // Nearest neighbors
        if let Ok(nn_analysis) = self.nearest_neighbors_analysis(entity, embeddings) {
            report.push_str("## Nearest Neighbors\n\n");
            for (i, (neighbor, distance)) in nn_analysis.neighbors.iter().enumerate() {
                report.push_str(&format!(
                    "{}. {} (distance: {:.4})\n",
                    i + 1,
                    neighbor,
                    distance
                ));
            }
            report.push('\n');
        }

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_similarity_analysis() {
        let mut embeddings = HashMap::new();
        embeddings.insert("e1".to_string(), array![1.0, 0.0, 0.0]);
        embeddings.insert("e2".to_string(), array![0.9, 0.1, 0.0]);
        embeddings.insert("e3".to_string(), array![0.0, 1.0, 0.0]);

        let config = InterpretabilityConfig {
            method: InterpretationMethod::SimilarityAnalysis,
            top_k: 2,
            ..Default::default()
        };

        let analyzer = InterpretabilityAnalyzer::new(config);
        let analysis = analyzer.similarity_analysis("e1", &embeddings).unwrap();

        assert_eq!(analysis.entity, "e1");
        assert_eq!(analysis.similar_entities.len(), 2);
        // e2 should be most similar to e1
        assert_eq!(analysis.similar_entities[0].0, "e2");
    }

    #[test]
    fn test_feature_importance() {
        let mut embeddings = HashMap::new();
        embeddings.insert("e1".to_string(), array![1.0, 0.0, 0.0]);
        embeddings.insert("e2".to_string(), array![0.0, 1.0, 0.0]);
        embeddings.insert("e3".to_string(), array![0.0, 0.0, 1.0]);
        embeddings.insert("e4".to_string(), array![5.0, 0.0, 0.0]); // Outlier in dim 0

        let config = InterpretabilityConfig {
            method: InterpretationMethod::FeatureImportance,
            top_k: 3,
            ..Default::default()
        };

        let analyzer = InterpretabilityAnalyzer::new(config);
        let importance = analyzer.feature_importance("e4", &embeddings).unwrap();

        assert_eq!(importance.entity, "e4");
        assert!(!importance.important_features.is_empty());
        // Dimension 0 should be most important for e4 (outlier)
        assert_eq!(importance.important_features[0].0, 0);
    }

    #[test]
    fn test_counterfactual() {
        let mut embeddings = HashMap::new();
        embeddings.insert("e1".to_string(), array![1.0, 0.0, 0.0]);
        embeddings.insert("e2".to_string(), array![0.0, 1.0, 0.0]);

        let config = InterpretabilityConfig::default();
        let analyzer = InterpretabilityAnalyzer::new(config);

        let cf = analyzer
            .counterfactual_explanation("e1", "e2", &embeddings)
            .unwrap();

        assert_eq!(cf.original, "e1");
        assert_eq!(cf.target, "e2");
        assert!(!cf.required_changes.is_empty());
        assert!(cf.difficulty > 0.0);
    }

    #[test]
    fn test_nearest_neighbors() {
        let mut embeddings = HashMap::new();
        embeddings.insert("e1".to_string(), array![1.0, 0.0]);
        embeddings.insert("e2".to_string(), array![1.1, 0.1]);
        embeddings.insert("e3".to_string(), array![5.0, 5.0]);

        let config = InterpretabilityConfig {
            method: InterpretationMethod::NearestNeighbors,
            top_k: 2,
            ..Default::default()
        };

        let analyzer = InterpretabilityAnalyzer::new(config);
        let nn = analyzer
            .nearest_neighbors_analysis("e1", &embeddings)
            .unwrap();

        assert_eq!(nn.entity, "e1");
        assert_eq!(nn.neighbors.len(), 2);
        // e2 should be nearest to e1
        assert_eq!(nn.neighbors[0].0, "e2");
    }

    #[test]
    fn test_generate_report() {
        let mut embeddings = HashMap::new();
        embeddings.insert("e1".to_string(), array![1.0, 0.0, 0.0]);
        embeddings.insert("e2".to_string(), array![0.9, 0.1, 0.0]);

        let config = InterpretabilityConfig::default();
        let analyzer = InterpretabilityAnalyzer::new(config);

        let report = analyzer.generate_report("e1", &embeddings).unwrap();

        assert!(report.contains("Interpretability Report"));
        assert!(report.contains("Similarity Analysis"));
        assert!(report.contains("Feature Importance"));
        assert!(report.contains("Nearest Neighbors"));
    }
}
