//! Cost-based vector index optimizer.
//!
//! This module implements selectivity-aware index selection for vector
//! search.  Given (data_size, dim, requested_recall, query_density) the
//! [`index_dispatcher::OptimizerDispatcher`] picks the lowest-cost index
//! family that meets the recall floor, with a fallback chain when an
//! observed recall trips the SLA.
//!
//! Component split:
//!
//! - [`cost_model`] — explicit per-index cost formulas (HNSW: `O(log n × M × ef)`,
//!   IVF: `O(n / nprobe + n_clusters)`, LSH: `O(K × L × dim + L × bucket)`,
//!   PQ: `O(centroids × subquantizers + n × subquantizers)`).
//! - [`index_dispatcher`] — execution-agnostic brain that consults the cost
//!   model and chooses a primary + fallback chain.
//! - [`query_stats`] — persisted per-family observation aggregates feeding
//!   online cost-weight learning.
//!
//! The crate-level [`crate::index_dispatcher::IndexDispatcher`] wires this
//! brain to concrete vector-index instances (HNSW/IVF/LSH/PQ).

pub mod cost_model;
pub mod index_dispatcher;
pub mod query_stats;

pub use cost_model::{
    CostEstimate, CostModel, CostWeights, IndexFamily, IndexParameters, WorkloadProfile,
};
pub use index_dispatcher::{
    dispatcher_with_families, dispatcher_with_parameters, DispatchError, DispatchPlan,
    DispatcherConfig, OptimizerDispatcher,
};
pub use query_stats::{FamilyStats, QueryObservation, QueryStats};
