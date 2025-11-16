//! Advanced features for SHACL-AI v0.1.0-beta.1
//!
//! This module provides cutting-edge AI capabilities for SHACL validation including:
//! - Graph Neural Networks for shape learning
//! - Transfer learning
//! - Active learning
//! - Advanced anomaly detection
//! - Continual learning
//! - Meta-learning

pub mod active_learning;
pub mod advanced_anomaly_detection;
pub mod continual_learning;
pub mod ensemble_methods;
pub mod generative_models;
pub mod graph_neural_networks;
pub mod transfer_learning;

pub use active_learning::{
    ActiveLearner, ActiveLearningConfig, QueryStrategy, SamplingStrategy, UncertaintySampling,
};
pub use advanced_anomaly_detection::{
    AdvancedAnomalyDetector, AnomalyDetectionConfig, CollectiveAnomalyDetector,
    ContextualAnomalyDetector, NoveltyDetector,
};
pub use continual_learning::{
    ContinualLearner, ContinualLearningConfig, MemoryBuffer, PlasticityPreservation,
};
pub use ensemble_methods::{
    EnsembleLearner, EnsembleStrategy, ModelEnsemble, VotingStrategy, WeightedEnsemble,
};
pub use generative_models::{GanModel, GenerativeModel, TestDataGenerator, VariationalAutoencoder};
pub use graph_neural_networks::{
    GnnLayer, GnnLayerType, GraphConvolution, GraphNeuralNetwork, GraphNeuralNetworkConfig,
    MessagePassingConfig, ShapeEmbedding,
};
pub use transfer_learning::{
    DomainAdapter, PretrainedModel, TransferLearner, TransferLearningConfig, TransferStrategy,
};
