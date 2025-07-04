//! Aggregation strategies for federated learning
//!
//! This module implements various aggregation methods for combining local model
//! updates from multiple participants in federated learning, including Byzantine-
//! resilient methods and robust aggregation techniques.

use super::config::AggregationStrategy;
use super::participant::{LocalUpdate, PrivacyMetrics};
use anyhow::Result;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Aggregation engine for combining local updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationEngine {
    /// Aggregation strategy
    pub strategy: AggregationStrategy,
    /// Aggregation parameters
    pub parameters: HashMap<String, f64>,
    /// Weighting scheme
    pub weighting_scheme: WeightingScheme,
    /// Outlier detection
    pub outlier_detection: OutlierDetection,
}

impl AggregationEngine {
    /// Create new aggregation engine
    pub fn new(strategy: AggregationStrategy) -> Self {
        Self {
            strategy,
            parameters: HashMap::new(),
            weighting_scheme: WeightingScheme::SampleSize,
            outlier_detection: OutlierDetection::default(),
        }
    }

    /// Configure weighting scheme
    pub fn with_weighting_scheme(mut self, scheme: WeightingScheme) -> Self {
        self.weighting_scheme = scheme;
        self
    }

    /// Configure outlier detection
    pub fn with_outlier_detection(mut self, detection: OutlierDetection) -> Self {
        self.outlier_detection = detection;
        self
    }

    /// Aggregate local updates from participants
    pub fn aggregate_updates(
        &self,
        updates: &[LocalUpdate],
    ) -> Result<HashMap<String, Array2<f32>>> {
        if updates.is_empty() {
            return Ok(HashMap::new());
        }

        // Detect and handle outliers
        let filtered_updates = if self.outlier_detection.enabled {
            self.filter_outliers(updates)?
        } else {
            updates.to_vec()
        };

        // Calculate weights for each participant
        let weights = self.calculate_weights(&filtered_updates)?;

        // Perform aggregation based on strategy
        match self.strategy {
            AggregationStrategy::FederatedAveraging => {
                self.federated_averaging(&filtered_updates, &weights)
            }
            AggregationStrategy::WeightedAveraging => {
                self.weighted_averaging(&filtered_updates, &weights)
            }
            AggregationStrategy::SecureAggregation => {
                self.secure_aggregation(&filtered_updates, &weights)
            }
            AggregationStrategy::RobustAggregation => {
                self.robust_aggregation(&filtered_updates, &weights)
            }
            AggregationStrategy::PersonalizedAggregation => {
                self.personalized_aggregation(&filtered_updates, &weights)
            }
            AggregationStrategy::HierarchicalAggregation => {
                self.hierarchical_aggregation(&filtered_updates, &weights)
            }
        }
    }

    /// Standard federated averaging
    fn federated_averaging(
        &self,
        updates: &[LocalUpdate],
        weights: &HashMap<Uuid, f64>,
    ) -> Result<HashMap<String, Array2<f32>>> {
        self.weighted_averaging(updates, weights)
    }

    /// Weighted averaging of updates
    fn weighted_averaging(
        &self,
        updates: &[LocalUpdate],
        weights: &HashMap<Uuid, f64>,
    ) -> Result<HashMap<String, Array2<f32>>> {
        let mut aggregated = HashMap::new();
        let total_weight: f64 = weights.values().sum();

        if total_weight == 0.0 {
            return Err(anyhow::anyhow!("Total weight is zero"));
        }

        // Initialize aggregated parameters with zeros
        if let Some(first_update) = updates.first() {
            for (param_name, param_values) in &first_update.parameter_updates {
                aggregated.insert(
                    param_name.clone(),
                    Array2::zeros(param_values.raw_dim()),
                );
            }
        }

        // Weighted sum of all updates
        for update in updates {
            let weight = weights.get(&update.participant_id).unwrap_or(&0.0) / total_weight;
            
            for (param_name, param_values) in &update.parameter_updates {
                if let Some(aggregated_param) = aggregated.get_mut(param_name) {
                    *aggregated_param = &*aggregated_param + &(param_values * weight as f32);
                }
            }
        }

        Ok(aggregated)
    }

    /// Secure aggregation with privacy preservation
    fn secure_aggregation(
        &self,
        updates: &[LocalUpdate],
        weights: &HashMap<Uuid, f64>,
    ) -> Result<HashMap<String, Array2<f32>>> {
        // For now, use weighted averaging
        // In a full implementation, this would use secure multi-party computation
        self.weighted_averaging(updates, weights)
    }

    /// Robust aggregation resistant to Byzantine failures
    fn robust_aggregation(
        &self,
        updates: &[LocalUpdate],
        _weights: &HashMap<Uuid, f64>,
    ) -> Result<HashMap<String, Array2<f32>>> {
        let mut aggregated = HashMap::new();

        if let Some(first_update) = updates.first() {
            for (param_name, _) in &first_update.parameter_updates {
                // Collect all parameter values for this parameter
                let mut param_matrices: Vec<&Array2<f32>> = updates
                    .iter()
                    .filter_map(|update| update.parameter_updates.get(param_name))
                    .collect();

                if param_matrices.is_empty() {
                    continue;
                }

                // Apply robust aggregation (Krum algorithm approximation)
                let aggregated_param = if param_matrices.len() > 2 {
                    self.krum_aggregation(&param_matrices)?
                } else {
                    // Fallback to averaging for small number of participants
                    self.median_aggregation(&param_matrices)?
                };

                aggregated.insert(param_name.clone(), aggregated_param);
            }
        }

        Ok(aggregated)
    }

    /// Personalized aggregation for participant-specific models
    fn personalized_aggregation(
        &self,
        updates: &[LocalUpdate],
        weights: &HashMap<Uuid, f64>,
    ) -> Result<HashMap<String, Array2<f32>>> {
        // For global model, use weighted averaging
        // Individual personalized models would be handled separately
        self.weighted_averaging(updates, weights)
    }

    /// Hierarchical aggregation for multi-level federation
    fn hierarchical_aggregation(
        &self,
        updates: &[LocalUpdate],
        weights: &HashMap<Uuid, f64>,
    ) -> Result<HashMap<String, Array2<f32>>> {
        // Simplified hierarchical aggregation
        // In practice, this would involve multiple levels of aggregation
        self.weighted_averaging(updates, weights)
    }

    /// Krum aggregation for Byzantine resilience
    fn krum_aggregation(&self, matrices: &[&Array2<f32>]) -> Result<Array2<f32>> {
        if matrices.is_empty() {
            return Err(anyhow::anyhow!("No matrices to aggregate"));
        }

        // Simplified Krum: find the matrix closest to others
        let mut best_idx = 0;
        let mut min_distance = f64::INFINITY;

        for i in 0..matrices.len() {
            let mut total_distance = 0.0;
            for j in 0..matrices.len() {
                if i != j {
                    total_distance += self.matrix_distance(matrices[i], matrices[j]);
                }
            }
            if total_distance < min_distance {
                min_distance = total_distance;
                best_idx = i;
            }
        }

        Ok(matrices[best_idx].clone())
    }

    /// Median aggregation for robustness
    fn median_aggregation(&self, matrices: &[&Array2<f32>]) -> Result<Array2<f32>> {
        if matrices.is_empty() {
            return Err(anyhow::anyhow!("No matrices to aggregate"));
        }

        let shape = matrices[0].raw_dim();
        let mut result = Array2::zeros(shape);

        // Element-wise median
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let mut values: Vec<f32> = matrices.iter()
                    .map(|m| m[[i, j]])
                    .collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                let median = if values.len() % 2 == 0 {
                    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                } else {
                    values[values.len() / 2]
                };
                
                result[[i, j]] = median;
            }
        }

        Ok(result)
    }

    /// Calculate distance between two matrices
    fn matrix_distance(&self, a: &Array2<f32>, b: &Array2<f32>) -> f64 {
        (a - b).iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt()
    }

    /// Calculate participant weights based on weighting scheme
    fn calculate_weights(&self, updates: &[LocalUpdate]) -> Result<HashMap<Uuid, f64>> {
        let mut weights = HashMap::new();

        match &self.weighting_scheme {
            WeightingScheme::Uniform => {
                let uniform_weight = 1.0 / updates.len() as f64;
                for update in updates {
                    weights.insert(update.participant_id, uniform_weight);
                }
            }
            WeightingScheme::SampleSize => {
                let total_samples: usize = updates.iter().map(|u| u.num_samples).sum();
                if total_samples > 0 {
                    for update in updates {
                        let weight = update.num_samples as f64 / total_samples as f64;
                        weights.insert(update.participant_id, weight);
                    }
                }
            }
            WeightingScheme::DataQuality => {
                // Use training accuracy as a proxy for data quality
                let total_accuracy: f64 = updates.iter()
                    .map(|u| u.training_stats.local_accuracy)
                    .sum();
                if total_accuracy > 0.0 {
                    for update in updates {
                        let weight = update.training_stats.local_accuracy / total_accuracy;
                        weights.insert(update.participant_id, weight);
                    }
                }
            }
            WeightingScheme::ComputeContribution => {
                // Use inverse of training time as compute contribution
                let total_compute: f64 = updates.iter()
                    .map(|u| 1.0 / (u.training_stats.training_time_seconds + 1.0))
                    .sum();
                if total_compute > 0.0 {
                    for update in updates {
                        let weight = (1.0 / (update.training_stats.training_time_seconds + 1.0)) / total_compute;
                        weights.insert(update.participant_id, weight);
                    }
                }
            }
            WeightingScheme::TrustScore => {
                // Would require trust scores from participant management
                // For now, fallback to uniform weighting
                let uniform_weight = 1.0 / updates.len() as f64;
                for update in updates {
                    weights.insert(update.participant_id, uniform_weight);
                }
            }
            WeightingScheme::Custom { weights: custom_weights } => {
                for update in updates {
                    let weight = custom_weights.get(&update.participant_id).unwrap_or(&0.0);
                    weights.insert(update.participant_id, *weight);
                }
            }
        }

        Ok(weights)
    }

    /// Filter outliers from updates
    fn filter_outliers(&self, updates: &[LocalUpdate]) -> Result<Vec<LocalUpdate>> {
        match self.outlier_detection.method {
            OutlierDetectionMethod::StatisticalDistance => {
                self.filter_statistical_outliers(updates)
            }
            OutlierDetectionMethod::Clustering => {
                self.filter_clustering_outliers(updates)
            }
            OutlierDetectionMethod::IsolationForest => {
                self.filter_isolation_forest_outliers(updates)
            }
            OutlierDetectionMethod::ByzantineDetection => {
                self.filter_byzantine_outliers(updates)
            }
        }
    }

    /// Filter outliers using statistical distance
    fn filter_statistical_outliers(&self, updates: &[LocalUpdate]) -> Result<Vec<LocalUpdate>> {
        if updates.len() < 3 {
            return Ok(updates.to_vec());
        }

        // Calculate pairwise distances between updates
        let mut distances = Vec::new();
        for i in 0..updates.len() {
            let mut total_distance = 0.0;
            for j in 0..updates.len() {
                if i != j {
                    total_distance += self.calculate_update_distance(&updates[i], &updates[j]);
                }
            }
            distances.push((i, total_distance / (updates.len() - 1) as f64));
        }

        // Calculate mean and std of distances
        let mean_distance: f64 = distances.iter().map(|(_, d)| *d).sum::<f64>() / distances.len() as f64;
        let variance: f64 = distances.iter()
            .map(|(_, d)| (d - mean_distance).powi(2))
            .sum::<f64>() / distances.len() as f64;
        let std_dev = variance.sqrt();

        // Filter outliers
        let threshold = mean_distance + self.outlier_detection.threshold * std_dev;
        let filtered_indices: Vec<usize> = distances.iter()
            .filter(|(_, d)| *d <= threshold)
            .map(|(i, _)| *i)
            .collect();

        Ok(filtered_indices.iter().map(|&i| updates[i].clone()).collect())
    }

    /// Calculate distance between two updates
    fn calculate_update_distance(&self, update1: &LocalUpdate, update2: &LocalUpdate) -> f64 {
        let mut total_distance = 0.0;
        let mut param_count = 0;

        for (param_name, param1) in &update1.parameter_updates {
            if let Some(param2) = update2.parameter_updates.get(param_name) {
                total_distance += self.matrix_distance(param1, param2);
                param_count += 1;
            }
        }

        if param_count > 0 {
            total_distance / param_count as f64
        } else {
            0.0
        }
    }

    /// Filter outliers using clustering (simplified)
    fn filter_clustering_outliers(&self, updates: &[LocalUpdate]) -> Result<Vec<LocalUpdate>> {
        // Simplified clustering-based outlier detection
        // In practice, this would use proper clustering algorithms
        self.filter_statistical_outliers(updates)
    }

    /// Filter outliers using isolation forest (simplified)
    fn filter_isolation_forest_outliers(&self, updates: &[LocalUpdate]) -> Result<Vec<LocalUpdate>> {
        // Simplified isolation forest
        // In practice, this would implement the full isolation forest algorithm
        self.filter_statistical_outliers(updates)
    }

    /// Filter Byzantine failures
    fn filter_byzantine_outliers(&self, updates: &[LocalUpdate]) -> Result<Vec<LocalUpdate>> {
        // Simplified Byzantine detection
        // In practice, this would implement sophisticated Byzantine detection
        self.filter_statistical_outliers(updates)
    }
}

/// Weighting schemes for aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightingScheme {
    /// Equal weights for all participants
    Uniform,
    /// Weight by number of samples
    SampleSize,
    /// Weight by data quality
    DataQuality,
    /// Weight by compute contribution
    ComputeContribution,
    /// Weight by trust score
    TrustScore,
    /// Custom weighting function
    Custom { weights: HashMap<Uuid, f64> },
}

/// Outlier detection for robust aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierDetection {
    /// Enable outlier detection
    pub enabled: bool,
    /// Detection method
    pub method: OutlierDetectionMethod,
    /// Outlier threshold
    pub threshold: f64,
    /// Action on outliers
    pub outlier_action: OutlierAction,
}

impl Default for OutlierDetection {
    fn default() -> Self {
        Self {
            enabled: true,
            method: OutlierDetectionMethod::StatisticalDistance,
            threshold: 2.0,
            outlier_action: OutlierAction::ReduceWeight,
        }
    }
}

/// Outlier detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierDetectionMethod {
    /// Statistical distance-based
    StatisticalDistance,
    /// Clustering-based
    Clustering,
    /// Isolation forest
    IsolationForest,
    /// Byzantine detection
    ByzantineDetection,
}

/// Actions to take on detected outliers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierAction {
    /// Exclude from aggregation
    Exclude,
    /// Reduce weight
    ReduceWeight,
    /// Apply robust aggregation
    RobustAggregation,
    /// Flag for manual review
    FlagForReview,
}

/// Aggregation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationStats {
    /// Number of participants
    pub num_participants: usize,
    /// Number of outliers detected
    pub num_outliers: usize,
    /// Total parameters aggregated
    pub total_parameters: usize,
    /// Aggregation time (seconds)
    pub aggregation_time_seconds: f64,
    /// Consensus measure
    pub consensus_measure: f64,
    /// Privacy budget consumed
    pub privacy_budget_consumed: f64,
}