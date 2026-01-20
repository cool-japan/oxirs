//! Performance utilities for large-scale geometric operations
//!
//! This module provides optimized implementations for processing large batches
//! of geometric data efficiently.
//!
//! # Performance Characteristics (Based on Benchmarks)
//!
//! - **Sequential processing**: Fastest for datasets <50,000 elements
//! - **Parallel processing**: Beneficial for very large datasets (50,000+) or complex operations
//! - **Streaming**: Memory-efficient for datasets that don't fit in memory
//!
//! # Important Notes
//!
//! **Parallel overhead is significant.** Benchmarks show that sequential processing
//! is faster than parallel for most common use cases due to:
//! - Thread spawning and synchronization costs
//! - Work distribution overhead
//! - Cache coherency issues
//!
//! Parallel processing becomes worthwhile for:
//! - Very large batch operations (50,000+ elements)
//! - Large distance matrices (100×100+)
//! - Complex k-NN queries
//! - Memory-constrained streaming scenarios
//!
//! # Usage
//!
//! ```rust
//! use oxirs_geosparql::performance::BatchProcessor;
//! use oxirs_geosparql::geometry::Geometry;
//! use geo_types::{Point, Geometry as GeoGeometry};
//!
//! let processor = BatchProcessor::new();
//! let query = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
//! let targets = vec![
//!     Geometry::new(GeoGeometry::Point(Point::new(1.0, 0.0))),
//!     Geometry::new(GeoGeometry::Point(Point::new(0.0, 1.0))),
//! ];
//!
//! // Automatically selects optimal strategy based on dataset size
//! let distances = processor.distances(&query, &targets).unwrap();
//! ```

pub mod batch;
pub mod parallel;
pub mod profiling;
pub mod simd;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use self::batch::BatchProcessor;
pub use self::parallel::{parallel_distance_matrix, parallel_nearest_neighbors};
pub use self::profiling::{ProfileScope, Profiler, TimingStats};
pub use self::simd::{euclidean_distance, euclidean_distance_squared};

#[cfg(feature = "gpu")]
pub use self::gpu::GpuGeometryContext;
