//! # OxiRS ARQ
//!
//! Jena-style SPARQL algebra with extension points and query optimization.

// Core modules
pub mod algebra;
pub mod algebra_generation;
pub mod bgp_optimizer;
pub mod builtin;
pub mod builtin_fixed;
pub mod cache_integration;
pub mod cost_model;
pub mod distributed;
pub mod executor;
pub mod expression;
pub mod extensions;
pub mod integrated_query_planner;
pub mod materialized_views;
pub mod optimizer;
pub mod parallel;
pub mod path;
pub mod query;
pub mod query_analysis;
pub mod statistics_collector;
pub mod streaming;
pub mod term;
pub mod update;
pub mod vector_query_optimizer;

// Advanced modules
pub mod advanced_optimizer;

// Re-export commonly used types
pub use algebra::{
    Aggregate, Algebra, Binding, GroupCondition, Iri, Literal, OrderCondition, Solution, Term,
    TriplePattern, Variable,
};
pub use executor::{Dataset, ExecutionContext, InMemoryDataset, ParallelConfig, QueryExecutor};

// Common Result type for the crate
pub type Result<T> = anyhow::Result<T>;
