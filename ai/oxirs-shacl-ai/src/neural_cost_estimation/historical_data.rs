//! Historical data management for neural cost estimation

use oxirs_core::query::algebra::AlgebraTriplePattern;
use std::collections::VecDeque;

use super::{
    config::HistoricalDataConfig,
    core::{QueryExecutionContext, TrainingData},
    types::*,
};
use crate::{Result, ShaclAiError};

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
    pub fn get_training_data(&self) -> Result<&TrainingData> {
        // Return cached data or generate new training data
        // For now, return a placeholder
        Err(ShaclAiError::DataProcessing("Training data not implemented".to_string()))
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
