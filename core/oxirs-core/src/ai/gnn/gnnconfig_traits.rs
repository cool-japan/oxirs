//! # GnnConfig - Trait Implementations
//!
//! This module contains trait implementations for `GnnConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for GnnConfig {
    fn default() -> Self {
        Self {
            architecture: GnnArchitecture::GraphConvolutionalNetwork,
            input_dim: 100,
            hidden_dims: vec![256, 128],
            output_dim: 64,
            num_layers: 3,
            dropout: 0.1,
            activation: ActivationFunction::ReLU,
            aggregation: Aggregation::Mean,
            use_residual: true,
            use_batch_norm: true,
            learning_rate: 0.001,
            l2_weight: 1e-4,
            link_prediction_method: LinkPredictionMethod::DotProduct,
        }
    }
}
