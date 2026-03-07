//! Spatial clustering algorithms.
//!
//! This module provides density-based and other clustering algorithms
//! for geo-spatial feature sets.

pub mod dbscan;

pub use dbscan::{ClusterAssignment, ClusterResult, DbscanConfig, SpatialDbscan};
