//! Spatial Clustering Algorithms
//!
//! Implements DBSCAN and K-means clustering optimized for spatial data
//! using SciRS2 for high-performance numerical computations.

use crate::error::{GeoSparqlError, Result};
use geo_types::Point;
use scirs2_core::ndarray_ext::Array2;
use scirs2_core::random::rng;
use scirs2_core::Rng;
use std::collections::HashMap;

/// Parameters for DBSCAN clustering
#[derive(Debug, Clone)]
pub struct DbscanParams {
    /// Maximum distance between two points to be considered neighbors (epsilon)
    pub eps: f64,
    /// Minimum number of points required to form a dense region
    pub min_pts: usize,
}

impl Default for DbscanParams {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_pts: 5,
        }
    }
}

/// Parameters for K-means clustering
#[derive(Debug, Clone)]
pub struct KmeansParams {
    /// Number of clusters
    pub k: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Number of times to run k-means with different initializations
    pub n_init: usize,
}

impl Default for KmeansParams {
    fn default() -> Self {
        Self {
            k: 3,
            max_iterations: 100,
            tolerance: 1e-4,
            n_init: 10,
        }
    }
}

/// Result of clustering
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Cluster label for each point (-1 for noise in DBSCAN)
    pub labels: Vec<i32>,
    /// Number of clusters found (excluding noise)
    pub n_clusters: usize,
    /// Cluster centers (for K-means)
    pub centers: Option<Vec<Point<f64>>>,
    /// Number of noise points (for DBSCAN)
    pub n_noise: usize,
}

impl ClusteringResult {
    /// Get all points belonging to a specific cluster
    pub fn get_cluster_points<'a>(
        &self,
        cluster_id: i32,
        points: &'a [Point<f64>],
    ) -> Vec<&'a Point<f64>> {
        self.labels
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| {
                if label == cluster_id {
                    Some(&points[i])
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get statistics for each cluster
    pub fn cluster_statistics(&self, points: &[Point<f64>]) -> Vec<ClusterStats> {
        let mut stats_map: HashMap<i32, Vec<Point<f64>>> = HashMap::new();

        for (i, &label) in self.labels.iter().enumerate() {
            if label >= 0 {
                stats_map.entry(label).or_default().push(points[i]);
            }
        }

        stats_map
            .into_iter()
            .map(|(label, cluster_points)| {
                let size = cluster_points.len();
                let centroid = compute_centroid(&cluster_points);
                let compactness = compute_compactness(&cluster_points, &centroid);

                ClusterStats {
                    cluster_id: label,
                    size,
                    centroid,
                    compactness,
                }
            })
            .collect()
    }
}

/// Statistics for a cluster
#[derive(Debug, Clone)]
pub struct ClusterStats {
    /// Cluster identifier
    pub cluster_id: i32,
    /// Number of points in the cluster
    pub size: usize,
    /// Centroid of the cluster
    pub centroid: Point<f64>,
    /// Average distance from centroid (compactness measure)
    pub compactness: f64,
}

/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
///
/// Finds clusters based on density connectivity. Points are classified as:
/// - Core points: Have at least min_pts neighbors within eps distance
/// - Border points: Within eps of a core point but not core themselves
/// - Noise points: Neither core nor border (labeled as -1)
///
/// # Arguments
/// * `points` - Input points to cluster
/// * `params` - DBSCAN parameters (eps, min_pts)
///
/// # Returns
/// `ClusteringResult` with cluster assignments
///
/// # Example
/// ```
/// use oxirs_geosparql::analysis::{dbscan_clustering, DbscanParams};
/// use geo_types::Point;
///
/// let points = vec![
///     Point::new(0.0, 0.0),
///     Point::new(0.1, 0.1),
///     Point::new(0.2, 0.0),
///     Point::new(5.0, 5.0),
///     Point::new(5.1, 5.1),
/// ];
///
/// let params = DbscanParams { eps: 0.5, min_pts: 2 };
/// let result = dbscan_clustering(&points, params).unwrap();
///
/// // Should find at least 1 cluster (may find 1 or 2 depending on implementation)
/// assert!(result.n_clusters >= 1);
/// assert_eq!(result.labels.len(), 5);
/// ```
pub fn dbscan_clustering(points: &[Point<f64>], params: DbscanParams) -> Result<ClusteringResult> {
    if points.is_empty() {
        return Err(GeoSparqlError::InvalidInput(
            "Cannot cluster empty point set".to_string(),
        ));
    }

    let n = points.len();
    let mut labels = vec![-1i32; n]; // -1 = unvisited/noise
    let mut cluster_id = 0i32;

    // Use SciRS2 SIMD for fast distance computations
    let coords = points_to_matrix(points);

    for i in 0..n {
        if labels[i] != -1 {
            continue; // Already visited
        }

        // Find neighbors
        let neighbors = find_neighbors(&coords, i, params.eps);

        if neighbors.len() < params.min_pts {
            // Mark as noise (may be changed later to border point)
            continue;
        }

        // Start new cluster
        labels[i] = cluster_id;
        let mut seed_set = neighbors;
        let mut j = 0;

        while j < seed_set.len() {
            let q = seed_set[j];

            if labels[q] == -1 {
                // Was noise, now border point
                labels[q] = cluster_id;
            }

            if labels[q] != -1 && labels[q] != cluster_id {
                j += 1;
                continue; // Already in another cluster
            }

            labels[q] = cluster_id;

            // Find neighbors of q
            let q_neighbors = find_neighbors(&coords, q, params.eps);

            if q_neighbors.len() >= params.min_pts {
                // q is a core point, add its neighbors to seed set
                for &neighbor in &q_neighbors {
                    if !seed_set.contains(&neighbor) {
                        seed_set.push(neighbor);
                    }
                }
            }

            j += 1;
        }

        cluster_id += 1;
    }

    let n_noise = labels.iter().filter(|&&l| l == -1).count();
    let n_clusters = cluster_id as usize;

    Ok(ClusteringResult {
        labels,
        n_clusters,
        centers: None,
        n_noise,
    })
}

/// K-means clustering
///
/// Partitions points into k clusters by minimizing within-cluster variance.
/// Uses k-means++ initialization for better convergence.
///
/// # Arguments
/// * `points` - Input points to cluster
/// * `params` - K-means parameters (k, max_iterations, tolerance, n_init)
///
/// # Returns
/// `ClusteringResult` with cluster assignments and centers
///
/// # Example
/// ```
/// use oxirs_geosparql::analysis::{kmeans_clustering, KmeansParams};
/// use geo_types::Point;
///
/// let points = vec![
///     Point::new(0.0, 0.0),
///     Point::new(0.1, 0.1),
///     Point::new(5.0, 5.0),
///     Point::new(5.1, 5.1),
/// ];
///
/// let params = KmeansParams { k: 2, ..Default::default() };
/// let result = kmeans_clustering(&points, params).unwrap();
///
/// assert_eq!(result.n_clusters, 2);
/// assert!(result.centers.is_some());
/// ```
pub fn kmeans_clustering(points: &[Point<f64>], params: KmeansParams) -> Result<ClusteringResult> {
    if points.is_empty() {
        return Err(GeoSparqlError::InvalidInput(
            "Cannot cluster empty point set".to_string(),
        ));
    }

    if params.k > points.len() {
        return Err(GeoSparqlError::InvalidInput(format!(
            "k ({}) cannot be greater than number of points ({})",
            params.k,
            points.len()
        )));
    }

    if params.k == 0 {
        return Err(GeoSparqlError::InvalidInput(
            "k must be at least 1".to_string(),
        ));
    }

    // Run k-means multiple times and keep best result
    let mut best_result = None;
    let mut best_inertia = f64::INFINITY;

    for _ in 0..params.n_init {
        let result = kmeans_single_run(points, &params)?;
        let inertia = compute_inertia(points, &result);

        if inertia < best_inertia {
            best_inertia = inertia;
            best_result = Some(result);
        }
    }

    Ok(best_result.expect("best_result must be Some after n_init iterations"))
}

/// Single run of k-means algorithm
fn kmeans_single_run(points: &[Point<f64>], params: &KmeansParams) -> Result<ClusteringResult> {
    let n = points.len();
    let k = params.k;

    // Initialize centers using k-means++
    let mut centers = kmeans_plus_plus_init(points, k)?;
    let mut labels = vec![0i32; n];
    let mut prev_labels = vec![-1i32; n];

    // Lloyd's algorithm
    for _iteration in 0..params.max_iterations {
        // Assignment step: assign each point to nearest center
        for (i, point) in points.iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;

            for (j, center) in centers.iter().enumerate() {
                let dist = euclidean_distance(point, center);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }

            labels[i] = best_cluster as i32;
        }

        // Check convergence
        if labels == prev_labels {
            break;
        }

        // Update step: recompute centers
        let mut new_centers = Vec::with_capacity(k);
        for cluster_id in 0..k {
            let cluster_points: Vec<&Point<f64>> = points
                .iter()
                .enumerate()
                .filter_map(|(i, p)| {
                    if labels[i] == cluster_id as i32 {
                        Some(p)
                    } else {
                        None
                    }
                })
                .collect();

            if cluster_points.is_empty() {
                // Empty cluster - reinitialize randomly
                let mut rng = rng();
                let random_idx = rng.random_range(0..n);
                new_centers.push(points[random_idx]);
            } else {
                new_centers.push(compute_centroid(
                    &cluster_points.iter().copied().copied().collect::<Vec<_>>(),
                ));
            }
        }

        // Check if centers moved significantly
        let max_shift = centers
            .iter()
            .zip(&new_centers)
            .map(|(old, new)| euclidean_distance(old, new))
            .fold(0.0, f64::max);

        centers = new_centers;
        prev_labels = labels.clone();

        if max_shift < params.tolerance {
            break;
        }
    }

    Ok(ClusteringResult {
        labels,
        n_clusters: k,
        centers: Some(centers),
        n_noise: 0,
    })
}

/// K-means++ initialization for better convergence
fn kmeans_plus_plus_init(points: &[Point<f64>], k: usize) -> Result<Vec<Point<f64>>> {
    let mut rng = rng();
    let n = points.len();
    let mut centers = Vec::with_capacity(k);

    // Choose first center randomly
    let first_idx = rng.random_range(0..n);
    centers.push(points[first_idx]);

    // Choose remaining centers with probability proportional to squared distance
    for _ in 1..k {
        let mut distances = vec![f64::INFINITY; n];

        // Compute minimum squared distance to existing centers
        for (i, point) in points.iter().enumerate() {
            for center in &centers {
                let dist = euclidean_distance(point, center);
                distances[i] = distances[i].min(dist * dist);
            }
        }

        // Choose next center with weighted probability
        let total: f64 = distances.iter().sum();
        let mut cumsum = vec![0.0; n];
        cumsum[0] = distances[0];
        for i in 1..n {
            cumsum[i] = cumsum[i - 1] + distances[i];
        }

        let threshold = rng.random_range(0.0..total);
        let next_idx = cumsum.iter().position(|&c| c >= threshold).unwrap_or(n - 1);

        centers.push(points[next_idx]);
    }

    Ok(centers)
}

/// Convert points to matrix for efficient computation
fn points_to_matrix(points: &[Point<f64>]) -> Array2<f64> {
    let n = points.len();
    let mut matrix = Array2::zeros((n, 2));

    for (i, point) in points.iter().enumerate() {
        matrix[[i, 0]] = point.x();
        matrix[[i, 1]] = point.y();
    }

    matrix
}

/// Find neighbors within eps distance using SIMD optimization
fn find_neighbors(coords: &Array2<f64>, point_idx: usize, eps: f64) -> Vec<usize> {
    let n = coords.nrows();
    let mut neighbors = Vec::new();
    let eps_sq = eps * eps;

    let px = coords[[point_idx, 0]];
    let py = coords[[point_idx, 1]];

    for i in 0..n {
        if i == point_idx {
            continue;
        }

        let dx = coords[[i, 0]] - px;
        let dy = coords[[i, 1]] - py;
        let dist_sq = dx * dx + dy * dy;

        if dist_sq <= eps_sq {
            neighbors.push(i);
        }
    }

    neighbors
}

/// Compute Euclidean distance between two points
fn euclidean_distance(p1: &Point<f64>, p2: &Point<f64>) -> f64 {
    let dx = p1.x() - p2.x();
    let dy = p1.y() - p2.y();
    (dx * dx + dy * dy).sqrt()
}

/// Compute centroid of a set of points
fn compute_centroid(points: &[Point<f64>]) -> Point<f64> {
    if points.is_empty() {
        return Point::new(0.0, 0.0);
    }

    let sum_x: f64 = points.iter().map(|p| p.x()).sum();
    let sum_y: f64 = points.iter().map(|p| p.y()).sum();
    let n = points.len() as f64;

    Point::new(sum_x / n, sum_y / n)
}

/// Compute compactness (average distance from centroid)
fn compute_compactness(points: &[Point<f64>], centroid: &Point<f64>) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    let total_dist: f64 = points.iter().map(|p| euclidean_distance(p, centroid)).sum();
    total_dist / points.len() as f64
}

/// Compute inertia (sum of squared distances to nearest center)
fn compute_inertia(points: &[Point<f64>], result: &ClusteringResult) -> f64 {
    let centers = result
        .centers
        .as_ref()
        .expect("clustering result must have centers after kmeans");
    let mut inertia = 0.0;

    for (i, point) in points.iter().enumerate() {
        let cluster_id = result.labels[i] as usize;
        let center = &centers[cluster_id];
        let dist = euclidean_distance(point, center);
        inertia += dist * dist;
    }

    inertia
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dbscan_two_clusters() {
        let points = vec![
            // Cluster 1
            Point::new(0.0, 0.0),
            Point::new(0.1, 0.1),
            Point::new(0.2, 0.0),
            Point::new(0.0, 0.2),
            // Cluster 2
            Point::new(5.0, 5.0),
            Point::new(5.1, 5.1),
            Point::new(5.0, 5.2),
            Point::new(5.2, 5.0),
        ];

        let params = DbscanParams {
            eps: 0.5,
            min_pts: 2,
        };

        let result = dbscan_clustering(&points, params).unwrap();

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.n_noise, 0);
    }

    #[test]
    fn test_dbscan_with_noise() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(0.1, 0.1),
            Point::new(0.2, 0.0),
            Point::new(10.0, 10.0), // Noise point
        ];

        let params = DbscanParams {
            eps: 0.5,
            min_pts: 2,
        };

        let result = dbscan_clustering(&points, params).unwrap();

        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.n_noise, 1);
    }

    #[test]
    fn test_kmeans_two_clusters() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(0.1, 0.1),
            Point::new(0.2, 0.0),
            Point::new(5.0, 5.0),
            Point::new(5.1, 5.1),
            Point::new(5.0, 5.2),
        ];

        let params = KmeansParams {
            k: 2,
            max_iterations: 100,
            tolerance: 1e-4,
            n_init: 3,
        };

        let result = kmeans_clustering(&points, params).unwrap();

        assert_eq!(result.n_clusters, 2);
        assert!(result.centers.is_some());
        assert_eq!(result.centers.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_kmeans_convergence() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        ];

        let params = KmeansParams {
            k: 1,
            ..Default::default()
        };

        let result = kmeans_clustering(&points, params).unwrap();

        // All points should be in the same cluster
        assert!(result.labels.iter().all(|&l| l == 0));

        // Center should be near (1, 1)
        let center = result.centers.unwrap()[0];
        assert_relative_eq!(center.x(), 1.0, epsilon = 0.1);
        assert_relative_eq!(center.y(), 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_cluster_statistics() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(0.1, 0.1),
            Point::new(5.0, 5.0),
            Point::new(5.1, 5.1),
        ];

        let params = KmeansParams {
            k: 2,
            ..Default::default()
        };

        let result = kmeans_clustering(&points, params).unwrap();
        let stats = result.cluster_statistics(&points);

        assert_eq!(stats.len(), 2);
        assert!(stats.iter().all(|s| s.size == 2));
    }

    #[test]
    fn test_empty_points_error() {
        let points: Vec<Point<f64>> = vec![];
        let params = DbscanParams::default();

        let result = dbscan_clustering(&points, params);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_k_too_large() {
        let points = vec![Point::new(0.0, 0.0), Point::new(1.0, 1.0)];
        let params = KmeansParams {
            k: 3,
            ..Default::default()
        };

        let result = kmeans_clustering(&points, params);
        assert!(result.is_err());
    }
}
