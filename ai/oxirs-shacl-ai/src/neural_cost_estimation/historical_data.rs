//! Historical data management for neural cost estimation

use oxirs_core::query::algebra::AlgebraTriplePattern;
use scirs2_core::ndarray_ext::{Array1, Array2};
use std::collections::VecDeque;

use super::{
    config::HistoricalDataConfig,
    core::{QueryExecutionContext, TrainingData},
    types::*,
};
use crate::Result;

/// Historical data manager
#[derive(Debug)]
pub struct HistoricalDataManager {
    /// Configuration
    config: HistoricalDataConfig,

    /// Performance records
    performance_records: VecDeque<HistoricalPerformanceRecord>,

    /// Training data cache
    training_data_cache: Option<TrainingData>,
}

/// Historical performance record
#[derive(Debug, Clone)]
pub struct HistoricalPerformanceRecord {
    pub patterns: Vec<AlgebraTriplePattern>,
    pub context: QueryExecutionContext,
    pub actual_cost: f64,
    pub prediction: CostPrediction,
    pub timestamp: std::time::SystemTime,
}

impl HistoricalDataManager {
    pub fn new(config: HistoricalDataConfig) -> Self {
        Self {
            config,
            performance_records: VecDeque::new(),
            training_data_cache: None,
        }
    }

    /// Add a performance record
    pub fn add_performance_record(
        &mut self,
        patterns: &[AlgebraTriplePattern],
        context: &QueryExecutionContext,
        actual_cost: f64,
        prediction: &CostPrediction,
    ) -> Result<()> {
        let record = HistoricalPerformanceRecord {
            patterns: patterns.to_vec(),
            context: context.clone(),
            actual_cost,
            prediction: prediction.clone(),
            timestamp: std::time::SystemTime::now(),
        };

        self.performance_records.push_back(record);

        // Remove old records if exceeding max size
        while self.performance_records.len() > self.config.max_history_size {
            self.performance_records.pop_front();
        }

        // Invalidate training data cache
        self.training_data_cache = None;

        Ok(())
    }

    /// Get training data
    pub fn get_training_data(&mut self) -> Result<&TrainingData> {
        // Check if we need to generate new training data
        let needs_generation = match &self.training_data_cache {
            Some(cached_data) => cached_data.features.is_empty() || cached_data.targets.is_empty(),
            None => true,
        };

        if needs_generation {
            // Generate new training data from historical records
            self.generate_training_data()?;
        }

        Ok(self
            .training_data_cache
            .as_ref()
            .expect("Training data should be generated"))
    }

    /// Generate training data from historical performance records
    fn generate_training_data(&mut self) -> Result<()> {
        if self.performance_records.is_empty() {
            // Create minimal training data for empty case
            self.training_data_cache = Some(TrainingData::default());
            return Ok(());
        }

        let num_records = self.performance_records.len();
        let feature_count = 12; // Number of features we extract per record

        // Initialize feature matrix and target vector
        let mut features = Array2::zeros((num_records, feature_count));
        let mut targets = Array1::zeros(num_records);

        // Convert each performance record to feature vector and target
        for (i, record) in self.performance_records.iter().enumerate() {
            let feature_vector = self.extract_features(record);
            features.row_mut(i).assign(&feature_vector);
            targets[i] = record.actual_cost;
        }

        // Cache the generated training data
        self.training_data_cache = Some(TrainingData { features, targets });

        Ok(())
    }

    /// Extract feature vector from a performance record
    fn extract_features(&self, record: &HistoricalPerformanceRecord) -> Array1<f64> {
        // Extract 12 meaningful features from the performance record
        let pattern_count = record.patterns.len() as f64;
        let cache_size = record.context.cache_size as f64;
        let system_load = record.context.system_load;
        let concurrent_queries = record.context.concurrent_queries as f64;
        let query_complexity = record.context.query_complexity;

        // Extract prediction features
        let predicted_cost = record.prediction.estimated_cost;
        let predicted_cpu = record.prediction.resource_usage.cpu_usage;
        let predicted_memory = record.prediction.resource_usage.memory_usage;
        let prediction_confidence = record.prediction.confidence;
        let prediction_uncertainty = record.prediction.uncertainty;

        // Calculate derived features
        let cost_prediction_error = (record.actual_cost - predicted_cost).abs();
        let normalized_complexity = query_complexity / (1.0 + pattern_count);

        Array1::from(vec![
            pattern_count,
            cache_size,
            system_load,
            concurrent_queries,
            query_complexity,
            predicted_cost,
            predicted_cpu,
            predicted_memory,
            prediction_confidence,
            prediction_uncertainty,
            cost_prediction_error,
            normalized_complexity,
        ])
    }

    /// Clear old records
    pub fn cleanup_old_records(&mut self) -> Result<()> {
        let cutoff_time = std::time::SystemTime::now()
            - std::time::Duration::from_secs(self.config.retention_period_days as u64 * 24 * 3600);

        self.performance_records
            .retain(|record| record.timestamp > cutoff_time);

        Ok(())
    }

    /// Get statistics
    pub fn get_statistics(&self) -> HistoricalDataStatistics {
        HistoricalDataStatistics {
            total_records: self.performance_records.len(),
            average_accuracy: self.calculate_average_accuracy(),
            data_coverage: self.calculate_data_coverage(),
        }
    }

    fn calculate_average_accuracy(&self) -> f64 {
        if self.performance_records.is_empty() {
            return 0.0;
        }

        let total_error: f64 = self
            .performance_records
            .iter()
            .map(|record| {
                (record.prediction.estimated_cost - record.actual_cost).abs()
                    / record.actual_cost.max(1.0)
            })
            .sum();

        1.0 - (total_error / self.performance_records.len() as f64).min(1.0)
    }

    fn calculate_data_coverage(&self) -> f64 {
        // Simplified coverage calculation
        (self.performance_records.len() as f64 / self.config.max_history_size as f64).min(1.0)
    }
}

/// Historical data statistics
#[derive(Debug, Clone)]
pub struct HistoricalDataStatistics {
    pub total_records: usize,
    pub average_accuracy: f64,
    pub data_coverage: f64,
}
