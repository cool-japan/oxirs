//! Training infrastructure for SHACL-AI models
//!
//! Provides distributed data-parallel training with gradient synchronisation
//! across worker threads via an in-process AllReduce implementation.

pub mod distributed;

pub use distributed::{
    dequantise_gradients_i8, finite_difference_grad, quantise_gradients_i8, sparsify_gradients,
    AdamOptimiser, AllReduceStrategy, AllReduceSync, DistributedTrainer, DistributedTrainingConfig,
    DistributedTrainingStats, FederatedRound, FederatedShapeTrainer, GradientAccumulator,
    GradientPrivacy, LocalUpdate, ModelWeights, ParameterVector, SgdOptimiser, WorkerConfig,
};
