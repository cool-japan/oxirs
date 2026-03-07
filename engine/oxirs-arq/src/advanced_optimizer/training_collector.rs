//! Training Data Collection and Management
//!
//! This module provides training data collection, buffering, and persistence
//! for the ML-based query cost predictor.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::advanced_optimizer::ml_predictor::{QueryCharacteristics, TrainingExample};
use crate::algebra::Algebra;

/// Configuration for training data collector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorConfig {
    /// Maximum number of training examples to keep
    pub max_examples: usize,
    /// Buffer size before flushing to dataset
    pub buffer_size: usize,
    /// Path to persist training data
    pub persistence_path: Option<PathBuf>,
    /// Window strategy for managing dataset size
    pub window_strategy: WindowStrategy,
    /// Auto-flush when buffer reaches threshold
    pub auto_flush: bool,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            max_examples: 10000,
            buffer_size: 100,
            persistence_path: None,
            window_strategy: WindowStrategy::SlidingWindow,
            auto_flush: true,
        }
    }
}

/// Strategy for managing dataset size
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WindowStrategy {
    /// Keep a sliding window of most recent examples
    SlidingWindow,
    /// Keep exactly max_size examples (FIFO)
    FixedSize,
    /// Keep examples within a time window (e.g., last 30 days)
    TimeWindow { days: u64 },
}

/// Training data collector
pub struct TrainingCollector {
    dataset: Arc<RwLock<TrainingDataset>>,
    config: CollectorConfig,
    buffer: VecDeque<TrainingExample>,
}

impl TrainingCollector {
    /// Create a new training collector
    pub fn new(config: CollectorConfig) -> Self {
        let dataset = Arc::new(RwLock::new(TrainingDataset::new(
            config.max_examples,
            config.window_strategy.clone(),
        )));

        let buffer_capacity = config.buffer_size;

        Self {
            dataset,
            config,
            buffer: VecDeque::with_capacity(buffer_capacity),
        }
    }

    /// Create collector and load existing data from disk
    pub fn load_or_create(config: CollectorConfig) -> Result<Self> {
        let persistence_path = config.persistence_path.clone();
        let collector = Self::new(config);

        if let Some(ref path) = persistence_path {
            if path.exists() {
                match TrainingDataset::load(path) {
                    Ok(dataset) => {
                        *collector.dataset.write().map_err(|e| {
                            anyhow::anyhow!("Failed to acquire write lock: {}", e)
                        })? = dataset;
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to load training data from {:?}: {}. Starting fresh.",
                            path,
                            e
                        );
                    }
                }
            }
        }

        Ok(collector)
    }

    /// Record a query execution result
    pub fn record_execution(
        &mut self,
        _query: &Algebra,
        features: Vec<f64>,
        characteristics: QueryCharacteristics,
        actual_cost: f64,
    ) -> Result<()> {
        let example = TrainingExample {
            features,
            target_cost: actual_cost,
            actual_cost,
            query_characteristics: characteristics,
            timestamp: SystemTime::now(),
        };

        self.buffer.push_back(example);

        // Auto-flush if buffer is full
        if self.config.auto_flush && self.buffer.len() >= self.config.buffer_size {
            self.flush()?;
        }

        Ok(())
    }

    /// Flush buffered examples to dataset
    pub fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let mut dataset = self
            .dataset
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire write lock: {}", e))?;

        for example in self.buffer.drain(..) {
            dataset.add(example);
        }

        // Persist if configured
        if let Some(ref path) = self.config.persistence_path {
            dataset.save(path)?;
        }

        Ok(())
    }

    /// Get a batch of training examples
    pub fn get_training_batch(&self, size: usize) -> Result<Vec<TrainingExample>> {
        let dataset = self
            .dataset
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;

        Ok(dataset.get_batch(size))
    }

    /// Get all training examples
    pub fn get_all_examples(&self) -> Result<Vec<TrainingExample>> {
        let dataset = self
            .dataset
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;

        Ok(dataset.examples.clone())
    }

    /// Get total count of examples (including buffer)
    pub fn len(&self) -> usize {
        let dataset_len = self.dataset.read().map(|d| d.len()).unwrap_or(0);

        dataset_len + self.buffer.len()
    }

    /// Check if collector is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear old examples based on window strategy
    pub fn clear_old_examples(&mut self) -> Result<()> {
        let mut dataset = self
            .dataset
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire write lock: {}", e))?;

        dataset.evict_old_examples();

        Ok(())
    }

    /// Get buffer size
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Get dataset reference for direct access
    pub fn dataset(&self) -> Arc<RwLock<TrainingDataset>> {
        Arc::clone(&self.dataset)
    }

    /// Save dataset to disk
    pub fn save(&self) -> Result<()> {
        if let Some(ref path) = self.config.persistence_path {
            let dataset = self
                .dataset
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;

            dataset.save(path)?;
        }

        Ok(())
    }
}

/// Training dataset with automatic size management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataset {
    pub examples: Vec<TrainingExample>,
    max_size: usize,
    windowing_strategy: WindowStrategy,
    total_examples_seen: usize,
}

impl TrainingDataset {
    /// Create a new training dataset
    pub fn new(max_size: usize, strategy: WindowStrategy) -> Self {
        Self {
            examples: Vec::with_capacity(max_size.min(1000)),
            max_size,
            windowing_strategy: strategy,
            total_examples_seen: 0,
        }
    }

    /// Add a training example
    pub fn add(&mut self, example: TrainingExample) {
        self.total_examples_seen += 1;

        match self.windowing_strategy {
            WindowStrategy::SlidingWindow | WindowStrategy::FixedSize => {
                self.examples.push(example);

                // Remove oldest if over capacity
                if self.examples.len() > self.max_size {
                    self.examples.remove(0);
                }
            }
            WindowStrategy::TimeWindow { .. } => {
                // Add example and then evict old ones
                self.examples.push(example);
                self.evict_old_examples();
            }
        }
    }

    /// Get a random batch of examples
    pub fn get_batch(&self, size: usize) -> Vec<TrainingExample> {
        let actual_size = size.min(self.examples.len());

        if actual_size == 0 {
            return Vec::new();
        }

        if actual_size >= self.examples.len() {
            return self.examples.clone();
        }

        // Simple random sampling - select evenly spaced examples
        // For production, we'd use proper random sampling
        let step = self.examples.len() / actual_size;
        let mut selected = Vec::with_capacity(actual_size);

        for i in 0..actual_size {
            let idx = i * step;
            if idx < self.examples.len() {
                selected.push(self.examples[idx].clone());
            }
        }

        selected
    }

    /// Get number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get total examples seen (including evicted ones)
    pub fn total_seen(&self) -> usize {
        self.total_examples_seen
    }

    /// Evict old examples based on strategy
    pub fn evict_old_examples(&mut self) {
        match self.windowing_strategy {
            WindowStrategy::TimeWindow { days } => {
                let now = SystemTime::now();
                let cutoff_duration = std::time::Duration::from_secs(days * 24 * 60 * 60);

                self.examples.retain(|example| {
                    if let Ok(elapsed) = now.duration_since(example.timestamp) {
                        elapsed < cutoff_duration
                    } else {
                        // Keep if we can't determine age
                        true
                    }
                });
            }
            WindowStrategy::SlidingWindow | WindowStrategy::FixedSize => {
                // Already handled in add()
                while self.examples.len() > self.max_size {
                    self.examples.remove(0);
                }
            }
        }
    }

    /// Save dataset to disk
    pub fn save(&self, path: &Path) -> Result<()> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory {:?}", parent))?;
        }

        let contents =
            serde_json::to_string_pretty(self).context("Failed to serialize training dataset")?;

        std::fs::write(path, contents)
            .with_context(|| format!("Failed to write training data to {:?}", path))?;

        Ok(())
    }

    /// Load dataset from disk
    pub fn load(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read training data from {:?}", path))?;

        let dataset: TrainingDataset = serde_json::from_str(&contents)
            .with_context(|| format!("Failed to deserialize training data from {:?}", path))?;

        Ok(dataset)
    }

    /// Get statistics about the dataset
    pub fn statistics(&self) -> DatasetStatistics {
        if self.examples.is_empty() {
            return DatasetStatistics::default();
        }

        let mut total_cost = 0.0;
        let mut min_cost = f64::MAX;
        let mut max_cost = f64::MIN;

        for example in &self.examples {
            let cost = example.actual_cost;
            total_cost += cost;

            if cost < min_cost {
                min_cost = cost;
            }
            if cost > max_cost {
                max_cost = cost;
            }
        }

        let mean_cost = total_cost / self.examples.len() as f64;

        // Calculate variance
        let variance = self
            .examples
            .iter()
            .map(|e| (e.actual_cost - mean_cost).powi(2))
            .sum::<f64>()
            / self.examples.len() as f64;

        let std_dev = variance.sqrt();

        // Find oldest and newest examples
        let oldest = self
            .examples
            .iter()
            .min_by_key(|e| e.timestamp)
            .and_then(|e| e.timestamp.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());

        let newest = self
            .examples
            .iter()
            .max_by_key(|e| e.timestamp)
            .and_then(|e| e.timestamp.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());

        DatasetStatistics {
            total_examples: self.examples.len(),
            total_seen: self.total_examples_seen,
            mean_cost,
            std_dev_cost: std_dev,
            min_cost,
            max_cost,
            oldest_example_timestamp: oldest,
            newest_example_timestamp: newest,
        }
    }

    /// Clear all examples
    pub fn clear(&mut self) {
        self.examples.clear();
    }
}

/// Statistics about the training dataset
#[derive(Debug, Clone, Default)]
pub struct DatasetStatistics {
    pub total_examples: usize,
    pub total_seen: usize,
    pub mean_cost: f64,
    pub std_dev_cost: f64,
    pub min_cost: f64,
    pub max_cost: f64,
    pub oldest_example_timestamp: Option<u64>,
    pub newest_example_timestamp: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Algebra;

    #[test]
    fn test_training_collector_creation() {
        let config = CollectorConfig::default();
        let collector = TrainingCollector::new(config);

        assert_eq!(collector.len(), 0);
        assert!(collector.is_empty());
    }

    #[test]
    fn test_training_dataset_add() {
        let mut dataset = TrainingDataset::new(10, WindowStrategy::FixedSize);

        let example = create_test_example(1.0);
        dataset.add(example);

        assert_eq!(dataset.len(), 1);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_sliding_window() {
        let mut dataset = TrainingDataset::new(5, WindowStrategy::SlidingWindow);

        // Add 10 examples
        for i in 0..10 {
            dataset.add(create_test_example(i as f64));
        }

        // Should only keep last 5
        assert_eq!(dataset.len(), 5);
        assert_eq!(dataset.total_seen(), 10);
    }

    #[test]
    fn test_batch_sampling() {
        let mut dataset = TrainingDataset::new(100, WindowStrategy::FixedSize);

        // Add 20 examples
        for i in 0..20 {
            dataset.add(create_test_example(i as f64));
        }

        // Get batch of 10
        let batch = dataset.get_batch(10);
        assert_eq!(batch.len(), 10);

        // Request more than available
        let batch = dataset.get_batch(100);
        assert_eq!(batch.len(), 20);
    }

    #[test]
    fn test_dataset_statistics() {
        let mut dataset = TrainingDataset::new(100, WindowStrategy::FixedSize);

        // Add examples with known costs
        for i in 1..=5 {
            dataset.add(create_test_example((i * 10) as f64));
        }

        let stats = dataset.statistics();
        assert_eq!(stats.total_examples, 5);
        assert_eq!(stats.mean_cost, 30.0); // (10+20+30+40+50)/5 = 30
        assert_eq!(stats.min_cost, 10.0);
        assert_eq!(stats.max_cost, 50.0);
    }

    #[test]
    fn test_persistence() -> Result<()> {
        use std::env;

        let temp_dir = env::temp_dir();
        let test_path = temp_dir.join("test_training_data.json");

        // Create dataset and add examples
        let mut dataset = TrainingDataset::new(10, WindowStrategy::FixedSize);
        for i in 0..5 {
            dataset.add(create_test_example(i as f64));
        }

        // Save
        dataset.save(&test_path)?;

        // Load
        let loaded = TrainingDataset::load(&test_path)?;

        assert_eq!(loaded.len(), dataset.len());
        assert_eq!(loaded.total_seen(), dataset.total_seen());

        // Cleanup
        std::fs::remove_file(&test_path).ok();

        Ok(())
    }

    #[test]
    fn test_buffer_flush() -> Result<()> {
        let config = CollectorConfig {
            buffer_size: 5,
            auto_flush: false,
            ..Default::default()
        };

        let mut collector = TrainingCollector::new(config);

        // Add examples to buffer
        for i in 0..3 {
            collector.record_execution(
                &Algebra::Empty,
                vec![1.0; 13],
                create_test_characteristics(),
                i as f64,
            )?;
        }

        assert_eq!(collector.buffer_len(), 3);
        assert_eq!(collector.len(), 3); // Buffer + dataset

        // Flush
        collector.flush()?;

        assert_eq!(collector.buffer_len(), 0);
        assert_eq!(collector.len(), 3); // All in dataset now

        Ok(())
    }

    #[test]
    fn test_auto_flush() -> Result<()> {
        let config = CollectorConfig {
            buffer_size: 2,
            auto_flush: true,
            ..Default::default()
        };

        let mut collector = TrainingCollector::new(config);

        // Add 3 examples - should auto-flush after 2
        for i in 0..3 {
            collector.record_execution(
                &Algebra::Empty,
                vec![1.0; 13],
                create_test_characteristics(),
                i as f64,
            )?;
        }

        // Buffer should have been flushed
        assert!(collector.buffer_len() <= 2);

        Ok(())
    }

    // Helper functions for tests
    fn create_test_example(cost: f64) -> TrainingExample {
        TrainingExample {
            features: vec![1.0; 13],
            target_cost: cost,
            actual_cost: cost,
            query_characteristics: create_test_characteristics(),
            timestamp: SystemTime::now(),
        }
    }

    fn create_test_characteristics() -> QueryCharacteristics {
        QueryCharacteristics {
            triple_pattern_count: 1,
            join_count: 0,
            filter_count: 0,
            optional_count: 0,
            has_aggregation: false,
            has_sorting: false,
            estimated_cardinality: 100,
            complexity_score: 1.0,
            query_graph_diameter: 1,
            avg_degree: 0.0,
            max_degree: 0,
        }
    }
}
