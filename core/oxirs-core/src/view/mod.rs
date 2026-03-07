//! Materialized view management for OxiRS.
//!
//! Provides:
//! - [`incremental`]: Incremental view maintenance with delta propagation.
//! - [`graph_view`]: Named graph views (GraphView, FilteredView, UnionView, MergedView).

pub mod graph_view;
pub mod incremental;

pub use graph_view::{FilteredView, GraphView, MergedView, RdfTriple, UnionView, ViewMaterializer};
pub use incremental::{MaterializedView, TripleDelta, ViewDefinition, ViewManager};
