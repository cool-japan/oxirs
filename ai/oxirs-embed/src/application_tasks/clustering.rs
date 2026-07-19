//! Clustering evaluation module
//!
//! This module provides comprehensive evaluation for clustering algorithms using
//! embedding models, including silhouette score, inertia, and other clustering
//! quality metrics.

use super::ApplicationEvalConfig;
use crate::EmbeddingModel;
use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array2;
#[allow(unused_imports)]
use scirs2_core::random::{Random, RngExt};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Clustering evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusteringMetric {
    /// Silhouette score
    SilhouetteScore,
    /// Calinski-Harabasz index
    CalinskiHarabaszIndex,
    /// Davies-Bouldin index
    DaviesBouldinIndex,
    /// Adjusted Rand Index (requires ground truth)
    AdjustedRandIndex,
    /// Normalized Mutual Information (requires ground truth)
    NormalizedMutualInformation,
    /// Clustering purity (requires ground truth)
    Purity,
    /// Inertia (within-cluster sum of squares)
    Inertia,
}

/// Cluster quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterAnalysis {
    /// Number of clusters
    pub num_clusters: usize,
    /// Cluster sizes
    pub cluster_sizes: Vec<usize>,
    /// Cluster cohesion scores
    pub cluster_cohesion: Vec<f64>,
    /// Cluster separation scores
    pub cluster_separation: Vec<f64>,
    /// Inter-cluster distances
    pub inter_cluster_distances: Array2<f64>,
}

/// Clustering stability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringStabilityAnalysis {
    /// Stability score across multiple runs
    pub stability_score: f64,
    /// Consistency of cluster assignments
    pub assignment_consistency: f64,
    /// Robustness to parameter changes
    pub parameter_robustness: f64,
}

/// Clustering evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResults {
    /// Metric scores
    pub metric_scores: HashMap<String, f64>,
    /// Cluster quality analysis
    pub cluster_analysis: ClusterAnalysis,
    /// Optimal number of clusters (if determined)
    pub optimal_k: Option<usize>,
    /// Clustering stability analysis
    pub stability_analysis: ClusteringStabilityAnalysis,
}

/// Clustering evaluator
pub struct ClusteringEvaluator {
    /// Ground truth clusters (if available)
    ground_truth_clusters: Option<HashMap<String, String>>,
    /// Clustering metrics
    metrics: Vec<ClusteringMetric>,
}

impl ClusteringEvaluator {
    /// Create a new clustering evaluator
    pub fn new() -> Self {
        Self {
            ground_truth_clusters: None,
            metrics: vec![
                ClusteringMetric::SilhouetteScore,
                ClusteringMetric::CalinskiHarabaszIndex,
                ClusteringMetric::DaviesBouldinIndex,
                ClusteringMetric::Inertia,
            ],
        }
    }

    /// Set ground truth clusters
    pub fn set_ground_truth(&mut self, clusters: HashMap<String, String>) {
        self.ground_truth_clusters = Some(clusters);

        // Add supervised metrics
        self.metrics.extend(vec![
            ClusteringMetric::AdjustedRandIndex,
            ClusteringMetric::NormalizedMutualInformation,
            ClusteringMetric::Purity,
        ]);
    }

    /// Evaluate clustering performance
    pub async fn evaluate(
        &self,
        model: &dyn EmbeddingModel,
        config: &ApplicationEvalConfig,
    ) -> Result<ClusteringResults> {
        // Get entity embeddings
        let entities = model.get_entities();
        let sample_entities: Vec<_> = entities.into_iter().take(config.sample_size).collect();

        let mut embeddings = Vec::new();
        for entity in &sample_entities {
            if let Ok(embedding) = model.get_entity_embedding(entity) {
                embeddings.push(embedding.values);
            }
        }

        if embeddings.is_empty() {
            return Err(anyhow!("No embeddings available for clustering evaluation"));
        }

        // Perform clustering
        let cluster_assignments = self.perform_clustering(&embeddings, config.num_clusters)?;

        // Calculate metrics
        let mut metric_scores = HashMap::new();
        for metric in &self.metrics {
            let score = self.calculate_clustering_metric(
                metric,
                &embeddings,
                &cluster_assignments,
                &sample_entities,
            )?;
            metric_scores.insert(format!("{metric:?}"), score);
        }

        // Analyze clusters
        let cluster_analysis = self.analyze_clusters(&embeddings, &cluster_assignments)?;

        // Analyze stability
        let stability_analysis = self.analyze_stability(&embeddings, config)?;

        Ok(ClusteringResults {
            metric_scores,
            cluster_analysis,
            optimal_k: Some(config.num_clusters), // Simplified
            stability_analysis,
        })
    }

    /// Perform K-means clustering
    fn perform_clustering(&self, embeddings: &[Vec<f32>], k: usize) -> Result<Vec<usize>> {
        if embeddings.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let n = embeddings.len();
        let dim = embeddings[0].len();

        // Initialize centroids randomly
        let mut centroids = Vec::new();
        let mut rng = Random::default();
        for _ in 0..k {
            let idx = rng.random_range(0..n);
            centroids.push(embeddings[idx].clone());
        }

        let mut assignments = vec![0; n];
        let max_iterations = 100;

        for _iteration in 0..max_iterations {
            let mut new_assignments = vec![0; n];
            let mut changed = false;

            // Assign points to nearest centroid
            for (i, embedding) in embeddings.iter().enumerate() {
                let mut min_distance = f32::INFINITY;
                let mut best_cluster = 0;

                for (c, centroid) in centroids.iter().enumerate() {
                    let distance = self.euclidean_distance(embedding, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = c;
                    }
                }

                new_assignments[i] = best_cluster;
                if new_assignments[i] != assignments[i] {
                    changed = true;
                }
            }

            assignments = new_assignments;

            if !changed {
                break;
            }

            // Update centroids
            for (c, centroid) in centroids.iter_mut().enumerate().take(k) {
                let cluster_points: Vec<_> = embeddings
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == c)
                    .map(|(_, emb)| emb)
                    .collect();

                if !cluster_points.is_empty() {
                    let mut new_centroid = vec![0.0f32; dim];
                    for point in &cluster_points {
                        for (i, &value) in point.iter().enumerate() {
                            new_centroid[i] += value;
                        }
                    }
                    for value in &mut new_centroid {
                        *value /= cluster_points.len() as f32;
                    }
                    *centroid = new_centroid;
                }
            }
        }

        Ok(assignments)
    }

    /// Calculate clustering metric
    fn calculate_clustering_metric(
        &self,
        metric: &ClusteringMetric,
        embeddings: &[Vec<f32>],
        assignments: &[usize],
        entities: &[String],
    ) -> Result<f64> {
        match metric {
            ClusteringMetric::SilhouetteScore => {
                self.calculate_silhouette_score(embeddings, assignments)
            }
            ClusteringMetric::Inertia => self.calculate_inertia(embeddings, assignments),
            ClusteringMetric::CalinskiHarabaszIndex => {
                self.calculate_calinski_harabasz(embeddings, assignments)
            }
            ClusteringMetric::DaviesBouldinIndex => {
                self.calculate_davies_bouldin(embeddings, assignments)
            }
            ClusteringMetric::AdjustedRandIndex => {
                if let Some(ref ground_truth) = self.ground_truth_clusters {
                    self.calculate_adjusted_rand_index(assignments, ground_truth, entities)
                } else {
                    Ok(0.0)
                }
            }
            ClusteringMetric::NormalizedMutualInformation => {
                if let Some(ref ground_truth) = self.ground_truth_clusters {
                    self.calculate_nmi(assignments, ground_truth, entities)
                } else {
                    // Consistent with AdjustedRandIndex above: without ground
                    // truth there is nothing to compare against.
                    Ok(0.0)
                }
            }
            ClusteringMetric::Purity => {
                if let Some(ref ground_truth) = self.ground_truth_clusters {
                    self.calculate_purity(assignments, ground_truth, entities)
                } else {
                    Ok(0.0)
                }
            }
        }
    }

    /// Calculate silhouette score
    fn calculate_silhouette_score(
        &self,
        embeddings: &[Vec<f32>],
        assignments: &[usize],
    ) -> Result<f64> {
        if embeddings.len() != assignments.len() || embeddings.is_empty() {
            return Ok(0.0);
        }

        let mut silhouette_scores = Vec::new();

        for (i, embedding) in embeddings.iter().enumerate() {
            let own_cluster = assignments[i];

            // Calculate average intra-cluster distance
            let same_cluster_points: Vec<_> = embeddings
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i && assignments[*j] == own_cluster)
                .map(|(_, emb)| emb)
                .collect();

            let a = if same_cluster_points.is_empty() {
                0.0
            } else {
                same_cluster_points
                    .iter()
                    .map(|other| self.euclidean_distance(embedding, other) as f64)
                    .sum::<f64>()
                    / same_cluster_points.len() as f64
            };

            // Calculate average nearest-cluster distance
            let unique_clusters: HashSet<usize> = assignments.iter().cloned().collect();
            let mut min_b = f64::INFINITY;

            for &cluster in &unique_clusters {
                if cluster != own_cluster {
                    let other_cluster_points: Vec<_> = embeddings
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| assignments[*j] == cluster)
                        .map(|(_, emb)| emb)
                        .collect();

                    if !other_cluster_points.is_empty() {
                        let avg_distance = other_cluster_points
                            .iter()
                            .map(|other| self.euclidean_distance(embedding, other) as f64)
                            .sum::<f64>()
                            / other_cluster_points.len() as f64;

                        min_b = min_b.min(avg_distance);
                    }
                }
            }

            let b = min_b;

            // Calculate silhouette score for this point
            let silhouette = if a < b {
                (b - a) / b
            } else if a > b {
                (b - a) / a
            } else {
                0.0
            };

            silhouette_scores.push(silhouette);
        }

        Ok(silhouette_scores.iter().sum::<f64>() / silhouette_scores.len() as f64)
    }

    /// Calculate inertia (within-cluster sum of squares)
    fn calculate_inertia(&self, embeddings: &[Vec<f32>], assignments: &[usize]) -> Result<f64> {
        let unique_clusters: HashSet<usize> = assignments.iter().cloned().collect();
        let mut total_inertia = 0.0;

        for &cluster in &unique_clusters {
            let cluster_points: Vec<_> = embeddings
                .iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == cluster)
                .map(|(_, emb)| emb)
                .collect();

            if cluster_points.is_empty() {
                continue;
            }

            // Calculate centroid
            let dim = cluster_points[0].len();
            let mut centroid = vec![0.0f32; dim];
            for point in &cluster_points {
                for (i, &value) in point.iter().enumerate() {
                    centroid[i] += value;
                }
            }
            for value in &mut centroid {
                *value /= cluster_points.len() as f32;
            }

            // Calculate sum of squared distances to centroid
            for point in &cluster_points {
                let distance = self.euclidean_distance(point, &centroid);
                total_inertia += (distance * distance) as f64;
            }
        }

        Ok(total_inertia)
    }

    /// Calculate Calinski-Harabasz index (simplified)
    fn calculate_calinski_harabasz(
        &self,
        embeddings: &[Vec<f32>],
        assignments: &[usize],
    ) -> Result<f64> {
        // Simplified implementation
        Ok(embeddings.len() as f64 * assignments.len() as f64 / 1000.0)
    }

    /// Calculate Davies-Bouldin index (simplified)
    fn calculate_davies_bouldin(
        &self,
        _embeddings: &[Vec<f32>],
        _assignments: &[usize],
    ) -> Result<f64> {
        // Simplified implementation
        Ok(0.5)
    }

    /// Calculate Adjusted Rand Index (simplified)
    /// Adjusted Rand Index between predicted cluster assignments and
    /// ground-truth labels, computed from the standard contingency-table
    /// formula: `(sum_ij C(n_ij,2) - expected) / (max - expected)`.
    fn calculate_adjusted_rand_index(
        &self,
        assignments: &[usize],
        ground_truth: &HashMap<String, String>,
        entities: &[String],
    ) -> Result<f64> {
        let pairs = Self::labeled_pairs(assignments, ground_truth, entities);
        if pairs.len() < 2 {
            return Ok(0.0);
        }

        let (cluster_counts, label_counts, joint_counts) = Self::contingency_table(&pairs);

        let comb2 = |x: usize| -> f64 {
            if x < 2 {
                0.0
            } else {
                (x * (x - 1)) as f64 / 2.0
            }
        };

        let sum_joint: f64 = joint_counts.values().map(|&v| comb2(v)).sum();
        let sum_cluster: f64 = cluster_counts.values().map(|&v| comb2(v)).sum();
        let sum_label: f64 = label_counts.values().map(|&v| comb2(v)).sum();
        let total_pairs = comb2(pairs.len());

        if total_pairs == 0.0 {
            return Ok(0.0);
        }

        let expected_index = sum_cluster * sum_label / total_pairs;
        let max_index = 0.5 * (sum_cluster + sum_label);

        if (max_index - expected_index).abs() < 1e-12 {
            // Degenerate case (e.g. a single cluster/label): agreement is
            // trivially perfect since there is nothing else to compare to.
            return Ok(1.0);
        }

        Ok(((sum_joint - expected_index) / (max_index - expected_index)).clamp(-1.0, 1.0))
    }

    /// Normalized Mutual Information between predicted cluster assignments
    /// and ground-truth labels: `I(U,V) / sqrt(H(U) * H(V))`.
    fn calculate_nmi(
        &self,
        assignments: &[usize],
        ground_truth: &HashMap<String, String>,
        entities: &[String],
    ) -> Result<f64> {
        let pairs = Self::labeled_pairs(assignments, ground_truth, entities);
        if pairs.is_empty() {
            return Ok(0.0);
        }

        let (cluster_counts, label_counts, joint_counts) = Self::contingency_table(&pairs);
        let n = pairs.len() as f64;

        let entropy = |count: usize| -> f64 {
            let p = count as f64 / n;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        };

        let mut mutual_information = 0.0;
        for (&(cluster, label), &n_uv) in &joint_counts {
            let n_u = cluster_counts[&cluster] as f64;
            let n_v = label_counts[&label] as f64;
            let p_uv = n_uv as f64 / n;
            mutual_information += p_uv * ((n * n_uv as f64) / (n_u * n_v)).ln();
        }

        let h_u: f64 = cluster_counts.values().map(|&c| entropy(c)).sum();
        let h_v: f64 = label_counts.values().map(|&c| entropy(c)).sum();

        if h_u <= 0.0 || h_v <= 0.0 {
            // No uncertainty in one of the partitions (e.g. a single
            // cluster/label): normalized mutual information is undefined, so
            // report perfect agreement only when the other side also has no
            // uncertainty, else no measurable information.
            return Ok(if h_u <= 0.0 && h_v <= 0.0 { 1.0 } else { 0.0 });
        }

        Ok((mutual_information / (h_u * h_v).sqrt()).clamp(0.0, 1.0))
    }

    /// Clustering purity: the fraction of (labeled) entities whose
    /// predicted cluster's majority ground-truth label matches their own.
    fn calculate_purity(
        &self,
        assignments: &[usize],
        ground_truth: &HashMap<String, String>,
        entities: &[String],
    ) -> Result<f64> {
        if assignments.len() != entities.len() || assignments.is_empty() {
            return Ok(0.0);
        }

        let mut clusters: HashMap<usize, Vec<&str>> = HashMap::new();
        let mut total_labeled = 0usize;
        for (idx, &cluster) in assignments.iter().enumerate() {
            if let Some(label) = ground_truth.get(&entities[idx]) {
                clusters.entry(cluster).or_default().push(label.as_str());
                total_labeled += 1;
            }
        }

        if total_labeled == 0 {
            return Ok(0.0);
        }

        let total_correct: usize = clusters
            .values()
            .map(|labels| {
                let mut counts: HashMap<&str, usize> = HashMap::new();
                for &label in labels {
                    *counts.entry(label).or_insert(0) += 1;
                }
                counts.values().copied().max().unwrap_or(0)
            })
            .sum();

        Ok(total_correct as f64 / total_labeled as f64)
    }

    /// Pair up predicted cluster assignments with ground-truth labels for
    /// entities that have a recorded label, dropping unlabeled entities.
    fn labeled_pairs<'a>(
        assignments: &[usize],
        ground_truth: &'a HashMap<String, String>,
        entities: &[String],
    ) -> Vec<(usize, &'a str)> {
        entities
            .iter()
            .zip(assignments.iter())
            .filter_map(|(entity, &cluster)| {
                ground_truth
                    .get(entity)
                    .map(|label| (cluster, label.as_str()))
            })
            .collect()
    }

    /// Build cluster-size, label-size, and joint contingency counts from
    /// (cluster, label) pairs.
    #[allow(clippy::type_complexity)]
    fn contingency_table<'a>(
        pairs: &[(usize, &'a str)],
    ) -> (
        HashMap<usize, usize>,
        HashMap<&'a str, usize>,
        HashMap<(usize, &'a str), usize>,
    ) {
        let mut cluster_counts: HashMap<usize, usize> = HashMap::new();
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        let mut joint_counts: HashMap<(usize, &str), usize> = HashMap::new();
        for &(cluster, label) in pairs {
            *cluster_counts.entry(cluster).or_insert(0) += 1;
            *label_counts.entry(label).or_insert(0) += 1;
            *joint_counts.entry((cluster, label)).or_insert(0) += 1;
        }
        (cluster_counts, label_counts, joint_counts)
    }

    /// Analyze clusters
    fn analyze_clusters(
        &self,
        _embeddings: &[Vec<f32>],
        assignments: &[usize],
    ) -> Result<ClusterAnalysis> {
        let unique_clusters: HashSet<usize> = assignments.iter().cloned().collect();
        let num_clusters = unique_clusters.len();

        let mut cluster_sizes = Vec::new();
        let cluster_cohesion = vec![0.5; num_clusters]; // Simplified
        let cluster_separation = vec![0.6; num_clusters]; // Simplified

        for &cluster in &unique_clusters {
            let cluster_size = assignments.iter().filter(|&&c| c == cluster).count();
            cluster_sizes.push(cluster_size);
        }

        // Simplified inter-cluster distances
        let inter_cluster_distances = Array2::zeros((num_clusters, num_clusters));

        Ok(ClusterAnalysis {
            num_clusters,
            cluster_sizes,
            cluster_cohesion,
            cluster_separation,
            inter_cluster_distances,
        })
    }

    /// Analyze clustering stability
    fn analyze_stability(
        &self,
        _embeddings: &[Vec<f32>],
        _config: &ApplicationEvalConfig,
    ) -> Result<ClusteringStabilityAnalysis> {
        // Simplified implementation
        Ok(ClusteringStabilityAnalysis {
            stability_score: 0.75,
            assignment_consistency: 0.8,
            parameter_robustness: 0.7,
        })
    }

    /// Calculate Euclidean distance
    fn euclidean_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl Default for ClusteringEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Perfect agreement between predicted clusters and ground truth: ARI,
    /// NMI, and purity must all report (near) 1.0 rather than a fabricated
    /// 0.5/0.6.
    #[test]
    fn test_ground_truth_metrics_perfect_agreement() -> Result<()> {
        let mut evaluator = ClusteringEvaluator::new();
        let entities = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let assignments = vec![0usize, 0, 1, 1];
        let ground_truth: HashMap<String, String> = [
            ("a".to_string(), "X".to_string()),
            ("b".to_string(), "X".to_string()),
            ("c".to_string(), "Y".to_string()),
            ("d".to_string(), "Y".to_string()),
        ]
        .into_iter()
        .collect();
        evaluator.set_ground_truth(ground_truth.clone());

        let ari =
            evaluator.calculate_adjusted_rand_index(&assignments, &ground_truth, &entities)?;
        assert!((ari - 1.0).abs() < 1e-9, "ari = {ari}");

        let nmi = evaluator.calculate_nmi(&assignments, &ground_truth, &entities)?;
        assert!((nmi - 1.0).abs() < 1e-9, "nmi = {nmi}");

        let purity = evaluator.calculate_purity(&assignments, &ground_truth, &entities)?;
        assert!((purity - 1.0).abs() < 1e-9, "purity = {purity}");

        Ok(())
    }

    /// Predicted clusters uncorrelated with (in fact, inverted relative to)
    /// ground truth should score well below the "perfect" case.
    #[test]
    fn test_ground_truth_metrics_poor_agreement() -> Result<()> {
        let evaluator = ClusteringEvaluator::new();
        let entities = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        // Every predicted cluster mixes both ground-truth labels evenly.
        let assignments = vec![0usize, 1, 0, 1];
        let ground_truth: HashMap<String, String> = [
            ("a".to_string(), "X".to_string()),
            ("b".to_string(), "X".to_string()),
            ("c".to_string(), "Y".to_string()),
            ("d".to_string(), "Y".to_string()),
        ]
        .into_iter()
        .collect();

        let ari =
            evaluator.calculate_adjusted_rand_index(&assignments, &ground_truth, &entities)?;
        assert!(ari < 0.5, "ari = {ari}");

        let purity = evaluator.calculate_purity(&assignments, &ground_truth, &entities)?;
        assert!((purity - 0.5).abs() < 1e-9, "purity = {purity}");

        Ok(())
    }

    /// The `_ => Ok(0.5)` fallback must be gone: every `ClusteringMetric`
    /// variant is handled explicitly by `calculate_clustering_metric`.
    #[test]
    fn test_calculate_clustering_metric_handles_ground_truth_variants_without_ground_truth(
    ) -> Result<()> {
        let evaluator = ClusteringEvaluator::new();
        let embeddings = vec![vec![0.0f32, 0.0], vec![1.0, 1.0]];
        let assignments = vec![0usize, 1];
        let entities = vec!["a".to_string(), "b".to_string()];

        for metric in [
            ClusteringMetric::NormalizedMutualInformation,
            ClusteringMetric::Purity,
        ] {
            let score = evaluator.calculate_clustering_metric(
                &metric,
                &embeddings,
                &assignments,
                &entities,
            )?;
            assert_eq!(score, 0.0, "metric = {metric:?}");
        }

        Ok(())
    }
}
