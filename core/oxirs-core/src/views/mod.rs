//! Incremental view maintenance for materialized SPARQL views.
//!
//! This module provides delta-based view invalidation and staleness detection.
//! See [`incremental`] for the core types.

pub mod incremental;

pub use incremental::{
    DeltaChange, IncrementalViewMaintainer, MaterializedView, ViewDefinition, ViewRow,
    ViewStalenessDetector,
};
