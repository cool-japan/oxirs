//! Data loading utilities for OxiRS TDB
//!
//! Provides high-performance data loading capabilities for RDF datasets.
//! Inspired by Apache Jena TDB2's loader architecture.

pub mod bulk_loader;

pub use bulk_loader::{BulkLoadStats, BulkLoader, BulkLoaderConfig, BulkLoaderFactory};
