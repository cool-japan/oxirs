//! # TrainingConfig - Trait Implementations
//!
//! This module contains trait implementations for `TrainingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 1000,
            batch_size: 256,
            learning_rate: 0.001,
            lr_scheduler: LearningRateScheduler::Constant,
            loss_function: LossFunction::MarginRankingLoss { margin: 1.0 },
            optimizer: Optimizer::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 1e-4,
            },
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 50,
                min_delta: 1e-6,
                monitor_metric: "validation_loss".to_string(),
                mode: MonitorMode::Min,
            },
            validation: ValidationConfig {
                validation_split: 0.1,
                validation_frequency: 10,
                metrics: vec![
                    TrainingMetric::Loss,
                    TrainingMetric::MeanReciprocalRank,
                    TrainingMetric::HitsAtK { k: 1 },
                    TrainingMetric::HitsAtK { k: 3 },
                    TrainingMetric::HitsAtK { k: 10 },
                ],
            },
            regularization: RegularizationConfig {
                l1_weight: 0.0,
                l2_weight: 1e-5,
                dropout_rate: 0.1,
                batch_norm: true,
            },
            gradient_clipping: Some(1.0),
            mixed_precision: true,
            checkpointing: CheckpointConfig {
                enabled: true,
                frequency: 100,
                save_best_only: true,
                save_dir: "./checkpoints".to_string(),
            },
            logging: LoggingConfig {
                log_frequency: 10,
                tensorboard_dir: Some("./logs".to_string()),
                wandb_project: None,
            },
            negative_sampling_strategy: NegativeSamplingStrategy::Random,
        }
    }
}
