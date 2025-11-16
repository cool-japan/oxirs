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
            patience: 50,
            validation_split: 0.2,
            loss_function: LossFunction::CrossEntropy,
            optimizer: Optimizer::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 1e-4,
            },
            gradient_clipping: Some(1.0),
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 50,
                min_delta: 1e-6,
            },
        }
    }
}
