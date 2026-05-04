//! Neuro-symbolic module: physics-informed entity scoring for knowledge graphs.
//!
//! This module blends **GNN cosine similarity** (neural signal) with **analytical
//! physics plausibility** (symbolic signal) to produce a combined entity score that
//! is both embedding-aware and physically grounded.
//!
//! # Architecture
//!
//! ```text
//!  KgGraph ──► GraphSageEncoder ──► EntityEmbeddings
//!                                           │
//!  query_embedding ──────────────────┐      │ cosine sim → neural_score
//!                                    └──────┤
//!  KgEntity.properties ──► PhysicsContext ──┤
//!                           (Fo / Re / ε / V)  └──► physics_score
//!                                                         │
//!                    combined = (1-λ)·neural + λ·physics ◄┘
//! ```
//!
//! # Quick start
//!
//! ```rust
//! use std::sync::Arc;
//! use std::collections::HashMap;
//! use oxirs_graphrag::gnn_encoder::{GraphSageConfig, GraphSageEncoder, KgGraph};
//! use oxirs_graphrag::neuro_symbolic::{
//!     PhysicsDomain, PhysicsContext,
//!     KgEntity, PinnEntityScorer,
//! };
//! use scirs2_core::ndarray_ext::{Array1, Array2};
//!
//! let config = GraphSageConfig {
//!     input_dim: 4, hidden_dim: 4, output_dim: 4,
//!     num_layers: 2, dropout: 0.0, k_neighbors: 2, learning_rate: 0.0,
//! };
//! let encoder = Arc::new(GraphSageEncoder::new(&config).expect("encoder"));
//! let ctx = PhysicsContext::new(PhysicsDomain::ThermalDiffusion {
//!     thermal_diffusivity: 1e-5,
//! });
//! let scorer = PinnEntityScorer::new(Arc::clone(&encoder), ctx, 0.3);
//!
//! let kg = KgGraph {
//!     num_nodes: 2,
//!     edges: vec![(0, 1)],
//!     node_features: Array2::zeros((2, 4)),
//! };
//! let entities = vec![
//!     KgEntity { id: "e0".into(), embedding_idx: 0, properties: HashMap::new() },
//!     KgEntity { id: "e1".into(), embedding_idx: 1, properties: HashMap::new() },
//! ];
//! let query = Array1::zeros(4);
//! let ranked = scorer.rank(&kg, &entities, &query).expect("rank");
//! assert_eq!(ranked.len(), 2);
//! ```

pub mod physics_context;
pub mod pinn_scorer;
pub mod retriever;

pub use physics_context::{FlowRegime, PhysicsContext, PhysicsDomain, PlausibilityScore};
pub use pinn_scorer::{KgEntity, PinnEntityScorer, PinnScorerError, ScoredEntity};
pub use retriever::{NeuroSymbolicError, NeuroSymbolicRetriever};
