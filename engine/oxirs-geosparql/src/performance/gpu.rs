//! GPU-accelerated geometric operations using SciRS2
//!
//! This module provides GPU-accelerated implementations of spatial operations
//! for large-scale datasets using scirs2-core GPU abstractions.
//!
//! # Current Status
//!
//! This module currently falls back to optimized CPU implementations while
//! awaiting full GPU backend support in scirs2-core. Future versions will
//! provide true GPU acceleration for:
//! - Massive parallelism for 10,000+ geometries
//! - 10-100x speedup for batch operations
//! - Automatic backend selection (CUDA, Metal, WebGPU)
//!
//! # Examples
//!
//! ```rust
//! use oxirs_geosparql::performance::gpu::*;
//! use oxirs_geosparql::geometry::Geometry;
//! use geo_types::{Point, Geometry as GeoGeometry};
//!
//! // Create GPU context (currently uses CPU fallback)
//! let ctx = GpuGeometryContext::new().expect("GPU context creation should succeed");
//!
//! let points: Vec<Geometry> = vec![
//!     Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
//!     Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
//! ];
//!
//! // Batch distance calculation
//! let distances = ctx.pairwise_distance_matrix(&points).expect("distance calculation should succeed");
//! assert_eq!(distances.shape(), &[2, 2]);
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::Centroid;
use geo_types::Point;
use scirs2_core::gpu::GpuBackend;
use scirs2_core::ndarray_ext::Array2;

/// GPU context for accelerated geometric operations
///
/// Currently uses optimized CPU implementations while awaiting
/// full GPU backend support in scirs2-core.
pub struct GpuGeometryContext {
    backend: GpuBackend,
    max_batch_size: usize,
}

impl GpuGeometryContext {
    /// Create a new GPU context with the preferred backend
    ///
    /// Currently always uses CPU fallback.
    pub fn new() -> Result<Self> {
        Ok(Self {
            backend: GpuBackend::Cpu,
            max_batch_size: 100_000,
        })
    }

    /// Create a GPU context with a specific backend
    ///
    /// Currently ignores the backend parameter and uses CPU.
    pub fn with_backend(_backend: GpuBackend) -> Result<Self> {
        Self::new()
    }

    /// Get the active GPU backend
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Get maximum batch size for this GPU
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Calculate pairwise Euclidean distances between all geometries
    ///
    /// Computes an N×N distance matrix where result\[i\]\[j\] is the distance
    /// between geometries\[i\] and geometries\[j\].
    ///
    /// # Arguments
    ///
    /// * `geometries` - Input geometries (all must have compatible CRS)
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::performance::gpu::GpuGeometryContext;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let ctx = GpuGeometryContext::new().expect("GPU context creation should succeed");
    /// let points: Vec<Geometry> = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0))),
    /// ];
    ///
    /// let distance_matrix = ctx.pairwise_distance_matrix(&points).expect("distance calculation should succeed");
    /// assert_eq!(distance_matrix.shape(), &[2, 2]);
    /// assert!((distance_matrix[[0, 1]] - 5.0).abs() < 1e-5);
    /// ```
    pub fn pairwise_distance_matrix(&self, geometries: &[Geometry]) -> Result<Array2<f32>> {
        if geometries.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }

        // Validate CRS compatibility
        let first_crs = &geometries[0].crs;
        for geom in &geometries[1..] {
            if &geom.crs != first_crs {
                return Err(GeoSparqlError::CrsIncompatibility(
                    first_crs.uri.clone(),
                    geom.crs.uri.clone(),
                ));
            }
        }

        // Extract representative points
        let points = extract_representative_points(geometries)?;
        let n = points.len();

        // Compute pairwise distances
        use geo::{Distance, Euclidean};
        let mut distances = Array2::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let dist = Euclidean::distance(points[i], points[j]) as f32;
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        Ok(distances)
    }

    /// Find all geometry pairs within a specified distance
    ///
    /// # Arguments
    ///
    /// * `geometries` - Input geometries
    /// * `max_distance` - Maximum distance threshold
    ///
    /// # Returns
    ///
    /// Vec of (index_i, index_j, distance) tuples for pairs within max_distance
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::performance::gpu::GpuGeometryContext;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let ctx = GpuGeometryContext::new().expect("GPU context creation should succeed");
    /// let points: Vec<Geometry> = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(1.0, 0.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))),
    /// ];
    ///
    /// // Find all pairs within 2.0 units
    /// let pairs = ctx.spatial_join_within_distance(&points, 2.0).expect("spatial join should succeed");
    /// assert_eq!(pairs.len(), 1); // Only (0,1) pair
    /// ```
    pub fn spatial_join_within_distance(
        &self,
        geometries: &[Geometry],
        max_distance: f64,
    ) -> Result<Vec<(usize, usize, f32)>> {
        if geometries.is_empty() {
            return Ok(Vec::new());
        }

        let distances = self.pairwise_distance_matrix(geometries)?;
        let mut result = Vec::new();
        let n = distances.shape()[0];
        let max_dist_f32 = max_distance as f32;

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = distances[[i, j]];
                if dist <= max_dist_f32 {
                    result.push((i, j, dist));
                }
            }
        }

        Ok(result)
    }

    /// GPU-accelerated batch distance calculation
    ///
    /// Computes distances from each query geometry to each target geometry.
    ///
    /// # Arguments
    ///
    /// * `queries` - Query geometries
    /// * `targets` - Target geometries
    ///
    /// # Returns
    ///
    /// M×N matrix where result\[i\]\[j\] is distance from queries\[i\] to targets\[j\]
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::performance::gpu::GpuGeometryContext;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let ctx = GpuGeometryContext::new().expect("GPU context creation should succeed");
    /// let queries: Vec<Geometry> = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
    /// ];
    /// let targets: Vec<Geometry> = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(1.0, 0.0))),
    /// ];
    ///
    /// let distances = ctx.batch_euclidean_distance(&queries, &targets).expect("distance calculation should succeed");
    /// assert_eq!(distances.shape(), &[1, 2]);
    /// assert!((distances[[0, 0]] - 5.0).abs() < 1e-5);
    /// ```
    pub fn batch_euclidean_distance(
        &self,
        queries: &[Geometry],
        targets: &[Geometry],
    ) -> Result<Array2<f32>> {
        if queries.is_empty() || targets.is_empty() {
            return Ok(Array2::zeros((queries.len(), targets.len())));
        }

        let query_points = extract_representative_points(queries)?;
        let target_points = extract_representative_points(targets)?;

        let m = query_points.len();
        let n = target_points.len();
        use geo::{Distance, Euclidean};
        let mut distances = Array2::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                distances[[i, j]] = Euclidean::distance(query_points[i], target_points[j]) as f32;
            }
        }

        Ok(distances)
    }

    /// Find K nearest neighbors for each query geometry
    ///
    /// # Arguments
    ///
    /// * `queries` - Query geometries
    /// * `candidates` - Candidate geometries to search
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Returns
    ///
    /// For each query, returns indices and distances of K nearest candidates
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::performance::gpu::GpuGeometryContext;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let ctx = GpuGeometryContext::new().expect("GPU context creation should succeed");
    /// let queries: Vec<Geometry> = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
    /// ];
    /// let candidates: Vec<Geometry> = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(1.0, 0.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(0.5, 0.0))),
    /// ];
    ///
    /// let results = ctx.k_nearest_neighbors(&queries, &candidates, 2).expect("k-NN search should succeed");
    /// assert_eq!(results.len(), 1);
    /// assert_eq!(results[0].len(), 2);
    /// // Nearest should be index 2 (0.5, 0.0), then index 0 (1.0, 0.0)
    /// assert_eq!(results[0][0].0, 2);
    /// assert_eq!(results[0][1].0, 0);
    /// ```
    pub fn k_nearest_neighbors(
        &self,
        queries: &[Geometry],
        candidates: &[Geometry],
        k: usize,
    ) -> Result<Vec<Vec<(usize, f32)>>> {
        if queries.is_empty() || candidates.is_empty() || k == 0 {
            return Ok(vec![Vec::new(); queries.len()]);
        }

        let distances = self.batch_euclidean_distance(queries, candidates)?;
        let m = distances.shape()[0];
        let n = distances.shape()[1];
        let k_clamped = k.min(n);

        let mut results = Vec::with_capacity(m);

        for i in 0..m {
            let mut dists: Vec<(usize, f32)> = (0..n).map(|j| (j, distances[[i, j]])).collect();

            dists.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .expect("distances should be comparable")
            });
            results.push(dists.into_iter().take(k_clamped).collect());
        }

        Ok(results)
    }
}

impl Default for GpuGeometryContext {
    fn default() -> Self {
        Self::new().expect("CPU backend should always work")
    }
}

/// Extract representative points from geometries
///
/// For non-point geometries, uses the centroid.
fn extract_representative_points(geometries: &[Geometry]) -> Result<Vec<Point<f64>>> {
    let mut points = Vec::with_capacity(geometries.len());

    for geom in geometries {
        let point = match &geom.geom {
            geo_types::Geometry::Point(p) => *p,
            geo_types::Geometry::LineString(ls) => ls.centroid().ok_or_else(|| {
                GeoSparqlError::InvalidGeometryType("Empty LineString".to_string())
            })?,
            geo_types::Geometry::Polygon(poly) => poly
                .centroid()
                .ok_or_else(|| GeoSparqlError::InvalidGeometryType("Empty Polygon".to_string()))?,
            geo_types::Geometry::MultiPoint(mp) => mp.centroid().ok_or_else(|| {
                GeoSparqlError::InvalidGeometryType("Empty MultiPoint".to_string())
            })?,
            geo_types::Geometry::MultiLineString(mls) => mls.centroid().ok_or_else(|| {
                GeoSparqlError::InvalidGeometryType("Empty MultiLineString".to_string())
            })?,
            geo_types::Geometry::MultiPolygon(mpoly) => mpoly.centroid().ok_or_else(|| {
                GeoSparqlError::InvalidGeometryType("Empty MultiPolygon".to_string())
            })?,
            _ => {
                return Err(GeoSparqlError::UnsupportedOperation(
                    "Unsupported geometry type for GPU operations".to_string(),
                ))
            }
        };

        points.push(point);
    }

    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::Geometry as GeoGeometry;

    fn create_test_points(n: usize) -> Vec<Geometry> {
        (0..n)
            .map(|i| {
                let x = (i as f64) * 10.0;
                let y = (i as f64) * 10.0;
                Geometry::new(GeoGeometry::Point(Point::new(x, y)))
            })
            .collect()
    }

    #[test]
    fn test_gpu_context_creation() {
        let ctx = GpuGeometryContext::new();
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_pairwise_distance_small() -> Result<()> {
        let ctx = GpuGeometryContext::new()?;
        let points = create_test_points(10);

        let distances = ctx.pairwise_distance_matrix(&points)?;

        assert_eq!(distances.shape(), &[10, 10]);

        // Distance matrix should be symmetric
        for i in 0..10 {
            for j in 0..10 {
                assert!((distances[[i, j]] - distances[[j, i]]).abs() < 1e-5);
            }
        }

        // Diagonal should be zero
        for i in 0..10 {
            assert!(distances[[i, i]].abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_batch_euclidean_distance() -> Result<()> {
        let ctx = GpuGeometryContext::new()?;
        let queries = create_test_points(5);
        let targets = create_test_points(7);

        let distances = ctx.batch_euclidean_distance(&queries, &targets)?;

        assert_eq!(distances.shape(), &[5, 7]);

        // Verify known distances
        assert!(distances[[0, 0]].abs() < 1e-5);

        let expected = (200.0_f32).sqrt();
        assert!((distances[[0, 1]] - expected).abs() < 1e-3);

        Ok(())
    }

    #[test]
    fn test_spatial_join_within_distance() -> Result<()> {
        let ctx = GpuGeometryContext::new()?;
        let points = create_test_points(20);

        let pairs = ctx.spatial_join_within_distance(&points, 15.0)?;

        assert!(!pairs.is_empty());

        for (_, _, dist) in &pairs {
            assert!(*dist <= 15.0);
        }

        Ok(())
    }

    #[test]
    fn test_k_nearest_neighbors() -> Result<()> {
        let ctx = GpuGeometryContext::new()?;
        let candidates = create_test_points(100);
        let queries = create_test_points(5);

        let results = ctx.k_nearest_neighbors(&queries, &candidates, 3)?;

        assert_eq!(results.len(), 5);

        for neighbors in &results {
            assert_eq!(neighbors.len(), 3);

            for i in 1..neighbors.len() {
                assert!(neighbors[i].1 >= neighbors[i - 1].1);
            }
        }

        Ok(())
    }

    #[test]
    fn test_empty_input() -> Result<()> {
        let ctx = GpuGeometryContext::new()?;
        let empty: Vec<Geometry> = Vec::new();
        let points = create_test_points(5);

        let distances = ctx.batch_euclidean_distance(&empty, &points)?;
        assert_eq!(distances.shape(), &[0, 5]);

        let distances = ctx.batch_euclidean_distance(&points, &empty)?;
        assert_eq!(distances.shape(), &[5, 0]);

        let distances = ctx.pairwise_distance_matrix(&empty)?;
        assert_eq!(distances.shape(), &[0, 0]);

        Ok(())
    }
}
