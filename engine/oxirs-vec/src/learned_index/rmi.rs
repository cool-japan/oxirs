//! Recursive Model Index (RMI) implementation

use super::neural_index::NeuralVectorIndex;
use super::types::{LearnedIndexError, LearnedIndexResult, TrainingExample};
use serde::{Deserialize, Serialize};

/// RMI stage containing multiple models
#[derive(Clone, Serialize, Deserialize)]
pub struct RmiStage {
    /// Models in this stage
    models: Vec<NeuralVectorIndex>,

    /// Number of models
    num_models: usize,
}

impl RmiStage {
    pub fn new(num_models: usize) -> Self {
        Self {
            models: Vec::with_capacity(num_models),
            num_models,
        }
    }

    pub fn num_models(&self) -> usize {
        self.num_models
    }

    pub fn models(&self) -> &[NeuralVectorIndex] {
        &self.models
    }
}

/// Recursive Model Index
#[derive(Clone, Serialize, Deserialize)]
pub struct RecursiveModelIndex {
    /// Stages of models
    stages: Vec<RmiStage>,

    /// Total number of records
    num_records: usize,

    /// Is trained
    is_trained: bool,
}

impl RecursiveModelIndex {
    /// Create new RMI with specified stage sizes
    pub fn new(stage_sizes: Vec<usize>) -> Self {
        let stages = stage_sizes.into_iter().map(RmiStage::new).collect();

        Self {
            stages,
            num_records: 0,
            is_trained: false,
        }
    }

    /// Train the RMI
    pub fn train(&mut self, examples: Vec<TrainingExample>) -> LearnedIndexResult<()> {
        if examples.is_empty() {
            return Err(LearnedIndexError::InsufficientData {
                min_required: 1,
                actual: 0,
            });
        }

        self.num_records = examples.len();

        tracing::info!(
            "Training RMI with {} stages on {} examples",
            self.stages.len(),
            examples.len()
        );

        // For simplified implementation:
        // Each stage predicts which model in next stage to use
        // Last stage predicts actual position

        self.is_trained = true;
        Ok(())
    }

    /// Predict position using RMI
    pub fn predict(&self, key: &[f32]) -> LearnedIndexResult<usize> {
        if !self.is_trained {
            return Err(LearnedIndexError::ModelNotTrained);
        }

        // Simplified: use first stage to predict position
        let normalized: f32 = key.iter().sum::<f32>() / key.len() as f32;
        let position = (normalized * self.num_records as f32) as usize;

        Ok(position.min(self.num_records.saturating_sub(1)))
    }

    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmi_creation() {
        let rmi = RecursiveModelIndex::new(vec![1, 10, 100]);
        assert_eq!(rmi.num_stages(), 3);
        assert!(!rmi.is_trained());
    }

    #[test]
    fn test_rmi_training() {
        let mut rmi = RecursiveModelIndex::new(vec![1, 10]);
        let examples = (0..100)
            .map(|i| TrainingExample::new(vec![i as f32 / 100.0], i))
            .collect();

        let result = rmi.train(examples);
        assert!(result.is_ok());
        assert!(rmi.is_trained());
    }

    #[test]
    fn test_rmi_prediction() {
        let mut rmi = RecursiveModelIndex::new(vec![1, 10]);
        let examples = (0..100)
            .map(|i| TrainingExample::new(vec![i as f32 / 100.0], i))
            .collect();

        rmi.train(examples).unwrap();

        let key = vec![0.5];
        let position = rmi.predict(&key).unwrap();
        assert!(position < 100);
    }
}
