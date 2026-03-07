//! Federated Query Rewriting and Optimization (v0.3.0)
//!
//! This module provides cross-instance SPARQL federation through query rewriting.
//! It decomposes a single SPARQL query into optimized per-endpoint subqueries,
//! estimates execution costs, and builds a physical execution plan.
//!
//! # Sub-modules
//!
//! - [`decomposer`]: Decomposes a SPARQL query into per-endpoint subqueries.
//! - [`optimizer`]: Optimizes the execution plan (join ordering, subquery merging, etc.).
//! - [`cost_estimator`]: Estimates execution cost per endpoint subquery.
//! - [`plan`]: Data structures representing the physical execution plan.
//! - [`error`]: Error types used across the module.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use oxirs_federate::query_rewrite::{
//!     decomposer::{EndpointInfo, QueryDecomposer},
//!     optimizer::FederationOptimizer,
//! };
//!
//! let decomposer = QueryDecomposer::new(vec![
//!     EndpointInfo::new("http://ep1/sparql").with_affinity("foaf"),
//!     EndpointInfo::new("http://ep2/sparql").with_affinity("schema"),
//! ]);
//!
//! let federated = decomposer
//!     .decompose("SELECT ?s ?name WHERE { ?s a foaf:Person . ?s foaf:name ?name }")
//!     .expect("decomposition failed");
//!
//! let optimizer = FederationOptimizer::new();
//! let plan = optimizer.optimize(federated).expect("optimization failed");
//! println!("{}", plan.summary());
//! ```

pub mod cost_estimator;
pub mod decomposer;
pub mod error;
pub mod optimizer;
pub mod plan;

// Re-export the most commonly used types for ergonomic access.
pub use cost_estimator::{CostEstimator, CostModelConfig};
pub use decomposer::{
    EndpointInfo, EndpointSubquery, FederatedQuery, QueryDecomposer, TriplePattern,
};
pub use error::{FederationError, FederationResult};
pub use optimizer::{FederationOptimizer, OptimizerConfig};
pub use plan::{ExecutionPlan, JoinStrategy, PlanNode};
