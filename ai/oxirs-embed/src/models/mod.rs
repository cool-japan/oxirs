//! Embedding model implementations
//!
//! This module provides various knowledge graph embedding models including:
//! - TransE: Translation-based embeddings
//! - ComplEx: Complex number embeddings for asymmetric relations
//! - DistMult: Bilinear diagonal model
//! - RotatE: Rotation-based embeddings
//! - HolE: Holographic embeddings using circular correlation
//! - ConvE: Convolutional embeddings with 2D CNNs (optional)
//! - TuckER: Tucker decomposition based embeddings (optional)
//! - TransformerEmbedding: Transformer-based embeddings (BERT, RoBERTa, etc.)
//! - GNNEmbedding: Graph Neural Network embeddings (GCN, GraphSAGE, GAT, etc.)
//! - OntologyAwareEmbedding: Embeddings that respect RDF/OWL ontology constraints

pub mod complex;
pub mod distmult;
pub mod gnn;
pub mod hole;
pub mod ontology;
pub mod rotate;
pub mod transe;
pub mod transformer;

#[cfg(feature = "conve")]
pub mod conve;

#[cfg(feature = "tucker")]
pub mod tucker;

#[cfg(feature = "quatd")]
pub mod quatd;

pub mod advanced_models;
pub mod base;
pub mod common;
pub mod scirs_neural;

// Re-export all models
pub use complex::ComplEx;
pub use distmult::DistMult;
pub use gnn::{AggregationType, GNNConfig, GNNEmbedding, GNNType};
pub use hole::{HoLE, HoLEConfig};
pub use ontology::{
    OntologyAwareConfig, OntologyAwareEmbedding, OntologyConstraints, OntologyRelation,
};
pub use rotate::RotatE;
pub use transe::TransE;
pub use transformer::{PoolingStrategy, TransformerConfig, TransformerEmbedding, TransformerType};

#[cfg(feature = "conve")]
pub use conve::{ConvE, ConvEConfig};

#[cfg(feature = "tucker")]
pub use tucker::TuckER;

#[cfg(feature = "quatd")]
pub use quatd::QuatD;

pub use advanced_models::{Lcg as AdvancedLcg, PairRE, Rescal, RotatEPlus};
pub use base::*;
pub use common::*;
pub use scirs_neural::{ActivationType, OptimizerType, SciRS2NeuralConfig, SciRS2NeuralEmbedding};

// GraphSAGE and GAT basic implementations
pub mod gat_basic;
pub mod graphsage;

// Re-exports for GraphSAGE
pub use graphsage::{
    AggregatorType, GraphData, GraphSage, GraphSageConfig, GraphSageEmbeddings,
    GraphSageTrainingMetrics, SimpleLcg,
};

// Re-exports for GAT
pub use gat_basic::{Gat, GatConfig, GatEmbeddings};

// Knowledge Graph Embedding algorithms (TransE, DistMult, RotatE)
pub mod kg_embeddings;
pub use kg_embeddings::{
    deserialize_embeddings, serialize_embeddings, DistMult as KgDistMult, KgEmbeddingConfig,
    KgEmbeddings, KgError, KgModel, KgResult, KgTriple, LinkPredictionEvaluator,
    RotatE as KgRotatE, TrainingHistory, TransE as KgTransE,
};
