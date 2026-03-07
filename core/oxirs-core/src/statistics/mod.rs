//! Statistics modules for the OxiRS cost-based query optimizer.
//!
//! Provides:
//! - [`histogram`]: Equi-depth histogram statistics for predicate-object distributions.
//! - [`graph_stats`]: High-level RDF graph statistics (triple counts, cardinality estimation, sampling).

pub mod graph_stats;
pub mod histogram;

pub use graph_stats::{
    CardinalityEstimator, GraphStatistics, PredicateHistogram, SampledStatistics,
};
pub use histogram::{DatasetStatistics, HistogramBucket, PredicateHistogram as EquiDepthHistogram};
