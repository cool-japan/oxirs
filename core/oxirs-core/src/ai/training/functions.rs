//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use crate::ai::{GraphNeuralNetwork, KnowledgeGraphEmbedding};
use crate::Triple;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(test)]
use std::time::Duration;
/// Trainer trait for different model types
#[async_trait::async_trait]
pub trait Trainer: Send + Sync {
    /// Train knowledge graph embedding model
    async fn train_embedding_model(
        &mut self,
        model: Arc<dyn KnowledgeGraphEmbedding>,
        training_data: &[Triple],
        validation_data: &[Triple],
    ) -> Result<TrainingMetrics>;
    /// Train graph neural network
    async fn train_gnn(
        &mut self,
        model: Arc<dyn GraphNeuralNetwork>,
        training_data: &[Triple],
        validation_data: &[Triple],
    ) -> Result<TrainingMetrics>;
    /// Resume training from checkpoint
    async fn resume_training(
        &mut self,
        checkpoint_path: &str,
        training_data: &[Triple],
        validation_data: &[Triple],
    ) -> Result<TrainingMetrics>;
    /// Evaluate model on test data
    async fn evaluate(
        &self,
        model: Arc<dyn KnowledgeGraphEmbedding>,
        test_data: &[Triple],
        metrics: &[TrainingMetric],
    ) -> Result<HashMap<String, f32>>;
}
/// Create trainer based on configuration
pub fn create_trainer(config: &TrainingConfig) -> Result<Arc<dyn Trainer>> {
    Ok(Arc::new(DefaultTrainer::new(config.clone())))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_training_config_creation() {
        let config = TrainingConfig::default();
        assert_eq!(config.max_epochs, 1000);
        assert_eq!(config.batch_size, 256);
        assert_eq!(config.learning_rate, 0.001);
    }
    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new();
        metrics.update_epoch(
            0,
            0.5,
            Some(0.4),
            Some(0.8),
            Some(0.85),
            0.001,
            Duration::from_millis(100),
        );
        assert_eq!(metrics.train_loss.len(), 1);
        assert_eq!(metrics.val_loss.len(), 1);
        assert_eq!(metrics.best_val_score, 0.4);
        assert_eq!(metrics.best_epoch, 0);
    }
    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig::default();
        let trainer = DefaultTrainer::new(config);
        assert_eq!(trainer.current_lr, 0.001);
    }
    #[test]
    fn test_learning_rate_scheduler() {
        let config = TrainingConfig {
            lr_scheduler: LearningRateScheduler::StepDecay {
                step_size: 100,
                gamma: 0.1,
            },
            ..Default::default()
        };
        let mut trainer = DefaultTrainer::new(config);
        assert_eq!(trainer.current_lr, 0.001);
        trainer.update_learning_rate(100, None);
        assert!((trainer.current_lr - 0.0001).abs() < 1e-8);
    }
}
