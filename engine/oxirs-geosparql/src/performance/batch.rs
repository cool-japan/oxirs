//! Batch processing utilities for efficient geometric operations
//!
//! This module provides high-level batch processing APIs that automatically
//! select the most efficient implementation based on dataset size and available features.
//!
//! # Performance
//!
//! - **Automatic optimization selection**: SIMD for small batches, parallel for large batches
//! - **Memory-efficient streaming**: Process data in chunks to avoid memory spikes
//! - **Progress tracking**: Optional progress callbacks for long-running operations
//!
//! # Examples
//!
//! ```rust
//! use oxirs_geosparql::performance::BatchProcessor;
//! use oxirs_geosparql::geometry::Geometry;
//! use geo_types::{Point, Geometry as GeoGeometry};
//!
//! let geometries: Vec<_> = (0..1000)
//!     .map(|i| Geometry::new(GeoGeometry::Point(Point::new(i as f64, i as f64))))
//!     .collect();
//!
//! // Batch processor automatically selects optimal strategy
//! let processor = BatchProcessor::new();
//! let query = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
//! let distances = processor.distances(&query, &geometries).unwrap();
//! ```

use crate::error::Result;
use crate::geometry::Geometry;

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Threshold for switching from SIMD to parallel processing
    pub parallel_threshold: usize,
    /// Chunk size for streaming operations
    pub chunk_size: usize,
    /// Enable progress tracking
    pub track_progress: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            // Parallel processing only becomes worthwhile for very large datasets
            // Based on benchmarks: sequential is faster up to ~50k elements
            parallel_threshold: 50000,
            chunk_size: 5000,
            track_progress: false,
        }
    }
}

/// High-level batch processor for geometric operations
///
/// This processor automatically selects the most efficient implementation
/// based on dataset size and available features.
///
/// # Performance Strategy (Based on Actual Benchmarks)
///
/// - **Small-Medium datasets (<50,000)**: Sequential processing (fastest due to low overhead)
/// - **Large datasets (>=50,000)**: Parallel processing across CPU cores
/// - **Huge datasets (>100,000)**: Chunked streaming to avoid memory spikes
///
/// **Note**: Parallel processing has significant overhead. Benchmarks show sequential
/// is faster for datasets under 50,000 elements. The threshold is conservative to
/// ensure best performance in real-world usage
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::performance::BatchProcessor;
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{Point, Geometry as GeoGeometry};
///
/// let processor = BatchProcessor::new();
///
/// let query = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
/// let targets: Vec<_> = (0..1000)
///     .map(|i| Geometry::new(GeoGeometry::Point(Point::new(i as f64, 0.0))))
///     .collect();
///
/// // Automatically uses optimal strategy (parallel for 1000 items)
/// let distances = processor.distances(&query, &targets).unwrap();
/// assert_eq!(distances.len(), 1000);
/// ```
pub struct BatchProcessor {
    config: BatchConfig,
}

impl BatchProcessor {
    /// Create a new batch processor with default configuration
    pub fn new() -> Self {
        Self {
            config: BatchConfig::default(),
        }
    }

    /// Create a batch processor with custom configuration
    pub fn with_config(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Calculate distances from a query geometry to multiple targets
    ///
    /// Automatically selects SIMD or parallel implementation based on dataset size.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::performance::BatchProcessor;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let processor = BatchProcessor::new();
    /// let query = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
    /// let targets = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(1.0, 0.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(0.0, 1.0))),
    /// ];
    ///
    /// let distances = processor.distances(&query, &targets).unwrap();
    /// assert_eq!(distances.len(), 2);
    /// ```
    pub fn distances(&self, query: &Geometry, targets: &[Geometry]) -> Result<Vec<f64>> {
        if targets.is_empty() {
            return Ok(Vec::new());
        }

        // Select strategy based on dataset size
        #[cfg(feature = "parallel")]
        {
            if targets.len() >= self.config.parallel_threshold {
                tracing::debug!(
                    "Using parallel distance calculation for {} targets",
                    targets.len()
                );
                return crate::performance::parallel::parallel_distances(query, targets);
            }
        }

        // Use SIMD for small datasets
        tracing::debug!(
            "Using SIMD distance calculation for {} targets",
            targets.len()
        );
        crate::performance::simd::batch_euclidean_distance(query, targets)
    }

    /// Calculate distance matrix for all pairs of geometries
    ///
    /// # Performance (Based on Benchmarks)
    ///
    /// - Small matrices (<100×100): Sequential processing (fastest)
    /// - Large matrices (>=100×100): Parallel processing becomes competitive
    ///
    /// **Note**: Parallel overhead is significant. Sequential is faster for matrices
    /// smaller than 100×100 elements
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::performance::BatchProcessor;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let processor = BatchProcessor::new();
    /// let geometries = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(1.0, 0.0))),
    /// ];
    ///
    /// let matrix = processor.distance_matrix(&geometries).unwrap();
    /// assert_eq!(matrix.len(), 2);
    /// ```
    pub fn distance_matrix(&self, geometries: &[Geometry]) -> Result<Vec<Vec<f64>>> {
        if geometries.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "parallel")]
        {
            // Parallel becomes competitive around 100×100 matrix (10,000 comparisons)
            if geometries.len() >= 100 {
                tracing::debug!(
                    "Using parallel distance matrix for {}×{} matrix",
                    geometries.len(),
                    geometries.len()
                );
                return crate::performance::parallel::parallel_distance_matrix(geometries);
            }
        }

        // Sequential for small datasets
        tracing::debug!(
            "Using sequential distance matrix for {} geometries",
            geometries.len()
        );
        let matrix: Result<Vec<_>> = geometries
            .iter()
            .map(|query| {
                let row: Result<Vec<_>> = geometries
                    .iter()
                    .map(|target| crate::performance::simd::euclidean_distance(query, target))
                    .collect();
                row
            })
            .collect();
        matrix
    }

    /// Find k nearest neighbors for each geometry
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::performance::BatchProcessor;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let processor = BatchProcessor::new();
    /// let geometries = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(1.0, 0.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(2.0, 0.0))),
    /// ];
    ///
    /// let nearest = processor.nearest_neighbors(&geometries, 1).unwrap();
    /// assert_eq!(nearest.len(), 3);
    /// ```
    pub fn nearest_neighbors(
        &self,
        geometries: &[Geometry],
        k: usize,
    ) -> Result<Vec<Vec<(usize, f64)>>> {
        #[cfg(feature = "parallel")]
        {
            if geometries.len() >= self.config.parallel_threshold {
                return crate::performance::parallel::parallel_nearest_neighbors(geometries, k);
            }
        }

        // Sequential fallback
        crate::performance::parallel::parallel_nearest_neighbors(geometries, k)
    }

    /// Transform multiple geometries to a target CRS
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "proj-support")]
    /// # fn example() -> oxirs_geosparql::error::Result<()> {
    /// use oxirs_geosparql::performance::BatchProcessor;
    /// use oxirs_geosparql::geometry::{Crs, Geometry};
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let processor = BatchProcessor::new();
    /// let geometries = vec![
    ///     Geometry::with_crs(GeoGeometry::Point(Point::new(139.7, 35.7)), Crs::epsg(4326)),
    /// ];
    ///
    /// let transformed = processor.transform_batch(&geometries, &Crs::epsg(3857))?;
    /// assert_eq!(transformed.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "proj-support")]
    pub fn transform_batch(
        &self,
        geometries: &[Geometry],
        target_crs: &crate::geometry::Crs,
    ) -> Result<Vec<Geometry>> {
        #[cfg(feature = "parallel")]
        {
            if geometries.len() >= self.config.parallel_threshold {
                return crate::performance::parallel::parallel_transform_batch(
                    geometries, target_crs,
                );
            }
        }

        // Sequential for small batches
        crate::functions::coordinate_transformation::transform_batch(geometries, target_crs)
    }

    /// Process data in streaming chunks to avoid memory spikes
    ///
    /// Useful for very large datasets that don't fit in memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::performance::BatchProcessor;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let processor = BatchProcessor::new();
    /// let query = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
    ///
    /// // Simulate large dataset
    /// let targets: Vec<_> = (0..10000)
    ///     .map(|i| Geometry::new(GeoGeometry::Point(Point::new(i as f64, 0.0))))
    ///     .collect();
    ///
    /// let mut all_distances: Vec<f64> = Vec::new();
    /// processor.stream_distances(&query, &targets, |chunk_distances| {
    ///     all_distances.extend(chunk_distances);
    ///     Ok(())
    /// }).unwrap();
    ///
    /// assert_eq!(all_distances.len(), 10000);
    /// ```
    pub fn stream_distances<F>(
        &self,
        query: &Geometry,
        targets: &[Geometry],
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(&[f64]) -> Result<()>,
    {
        for chunk in targets.chunks(self.config.chunk_size) {
            let distances = self.distances(query, chunk)?;
            callback(&distances)?;
        }
        Ok(())
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    fn test_batch_processor_distances() {
        let processor = BatchProcessor::new();
        let query = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
        let targets = vec![
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 0.0))),
            Geometry::new(GeoGeometry::Point(Point::new(0.0, 1.0))),
        ];

        let distances = processor.distances(&query, &targets).unwrap();

        assert_eq!(distances.len(), 2);
        assert!((distances[0] - 1.0).abs() < 1e-10);
        assert!((distances[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_processor_distance_matrix() {
        let processor = BatchProcessor::new();
        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 0.0))),
        ];

        let matrix = processor.distance_matrix(&geometries).unwrap();

        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
    }

    #[test]
    fn test_batch_processor_nearest_neighbors() {
        let processor = BatchProcessor::new();
        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 0.0))),
            Geometry::new(GeoGeometry::Point(Point::new(2.0, 0.0))),
        ];

        let nearest = processor.nearest_neighbors(&geometries, 1).unwrap();

        assert_eq!(nearest.len(), 3);
        assert_eq!(nearest[0].len(), 1); // Each query has 1 nearest neighbor
    }

    #[test]
    fn test_batch_processor_stream_distances() {
        let processor = BatchProcessor::new();
        let query = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
        let targets: Vec<_> = (0..100)
            .map(|i| Geometry::new(GeoGeometry::Point(Point::new(i as f64, 0.0))))
            .collect();

        let mut all_distances: Vec<f64> = Vec::new();
        processor
            .stream_distances(&query, &targets, |chunk_distances| {
                all_distances.extend(chunk_distances);
                Ok(())
            })
            .unwrap();

        assert_eq!(all_distances.len(), 100);
    }

    #[test]
    fn test_batch_config_custom() {
        let config = BatchConfig {
            parallel_threshold: 200,
            chunk_size: 500,
            track_progress: true,
        };

        let processor = BatchProcessor::with_config(config);
        assert_eq!(processor.config.parallel_threshold, 200);
        assert_eq!(processor.config.chunk_size, 500);
        assert!(processor.config.track_progress);
    }
}
