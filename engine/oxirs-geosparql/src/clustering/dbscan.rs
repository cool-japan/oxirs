//! # Spatial DBSCAN Clustering
//!
//! Density-Based Spatial Clustering of Applications with Noise (DBSCAN) for
//! geographic features using the Haversine distance metric.
//!
//! ## Algorithm
//!
//! DBSCAN groups points that are closely packed together (many nearby
//! neighbours within a distance `eps`) and marks outlier points in
//! low-density regions as noise.
//!
//! ## Features
//!
//! - Haversine distance for WGS-84 lat/lon coordinates
//! - Configurable `eps` (in metres) and `min_points`
//! - Returns cluster assignments with noise label (`-1`)
//! - Cluster statistics (sizes, centroid, bounding box)
//!
//! ## Usage
//!
//! ```rust
//! use oxirs_geosparql::clustering::dbscan::{SpatialDbscan, DbscanConfig, GeoPoint};
//!
//! let config = DbscanConfig {
//!     eps_metres: 500.0,
//!     min_points: 2,
//! };
//! let points = vec![
//!     GeoPoint::new(48.8566, 2.3522), // Paris
//!     GeoPoint::new(48.8570, 2.3525), // Paris nearby
//!     GeoPoint::new(35.6762, 139.6503), // Tokyo
//! ];
//! let result = SpatialDbscan::cluster(&points, &config);
//! assert!(result.num_clusters >= 1);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// GeoPoint
// ---------------------------------------------------------------------------

/// A geographic point (latitude, longitude in degrees).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoPoint {
    /// Latitude in degrees (-90 to 90).
    pub lat: f64,
    /// Longitude in degrees (-180 to 180).
    pub lon: f64,
}

impl GeoPoint {
    /// Create a new point.
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// DBSCAN configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbscanConfig {
    /// Maximum distance (in metres) for two points to be neighbours.
    pub eps_metres: f64,
    /// Minimum number of points required to form a dense region (core point).
    pub min_points: usize,
}

impl Default for DbscanConfig {
    fn default() -> Self {
        Self {
            eps_metres: 1000.0,
            min_points: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Cluster result
// ---------------------------------------------------------------------------

/// Assignment for a single point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterAssignment {
    /// Index of the point in the input array.
    pub point_index: usize,
    /// Cluster label. `-1` means noise.
    pub cluster_id: i32,
}

/// Statistics about a single cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    /// Cluster label.
    pub cluster_id: i32,
    /// Number of points.
    pub size: usize,
    /// Centroid (mean lat/lon).
    pub centroid: GeoPoint,
    /// Bounding box (min_lat, min_lon, max_lat, max_lon).
    pub bbox: (f64, f64, f64, f64),
}

/// The full DBSCAN result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterResult {
    /// Per-point assignments.
    pub assignments: Vec<ClusterAssignment>,
    /// Number of clusters found (excluding noise).
    pub num_clusters: usize,
    /// Number of noise points.
    pub num_noise: usize,
    /// Statistics per cluster (excluding noise).
    pub cluster_stats: Vec<ClusterStats>,
}

impl ClusterResult {
    /// Get the cluster label for a given point index.
    pub fn label_for(&self, point_index: usize) -> Option<i32> {
        self.assignments
            .iter()
            .find(|a| a.point_index == point_index)
            .map(|a| a.cluster_id)
    }

    /// Get point indices belonging to a given cluster.
    pub fn points_in_cluster(&self, cluster_id: i32) -> Vec<usize> {
        self.assignments
            .iter()
            .filter(|a| a.cluster_id == cluster_id)
            .map(|a| a.point_index)
            .collect()
    }

    /// Get all noise points.
    pub fn noise_points(&self) -> Vec<usize> {
        self.points_in_cluster(-1)
    }
}

// ---------------------------------------------------------------------------
// SpatialDbscan
// ---------------------------------------------------------------------------

/// The DBSCAN clustering algorithm with Haversine distance.
pub struct SpatialDbscan;

impl SpatialDbscan {
    /// Perform DBSCAN clustering on the given points.
    pub fn cluster(points: &[GeoPoint], config: &DbscanConfig) -> ClusterResult {
        let n = points.len();
        let mut labels = vec![-2_i32; n]; // -2 = undefined, -1 = noise, >=0 = cluster
        let mut cluster_id = 0_i32;

        for i in 0..n {
            if labels[i] != -2 {
                continue; // already processed
            }
            let neighbours = Self::range_query(points, i, config.eps_metres);
            if neighbours.len() < config.min_points {
                labels[i] = -1; // noise
            } else {
                // Expand cluster
                labels[i] = cluster_id;
                let mut seed_set: Vec<usize> = neighbours.clone();
                let mut j = 0;
                while j < seed_set.len() {
                    let q = seed_set[j];
                    if labels[q] == -1 {
                        labels[q] = cluster_id; // change noise to border
                    }
                    if labels[q] != -2 {
                        j += 1;
                        continue; // already processed
                    }
                    labels[q] = cluster_id;
                    let q_neighbours = Self::range_query(points, q, config.eps_metres);
                    if q_neighbours.len() >= config.min_points {
                        for &nb in &q_neighbours {
                            if !seed_set.contains(&nb) {
                                seed_set.push(nb);
                            }
                        }
                    }
                    j += 1;
                }
                cluster_id += 1;
            }
        }

        // Build result
        let assignments: Vec<ClusterAssignment> = labels
            .iter()
            .enumerate()
            .map(|(i, &label)| ClusterAssignment {
                point_index: i,
                cluster_id: label,
            })
            .collect();

        let num_noise = labels.iter().filter(|&&l| l == -1).count();
        let num_clusters = cluster_id as usize;

        // Compute per-cluster stats
        let mut cluster_points: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            if label >= 0 {
                cluster_points.entry(label).or_default().push(i);
            }
        }

        let mut cluster_stats: Vec<ClusterStats> = Vec::new();
        for cid in 0..cluster_id {
            if let Some(indices) = cluster_points.get(&cid) {
                let size = indices.len();
                let sum_lat: f64 = indices.iter().map(|&i| points[i].lat).sum();
                let sum_lon: f64 = indices.iter().map(|&i| points[i].lon).sum();
                let centroid = GeoPoint::new(sum_lat / size as f64, sum_lon / size as f64);
                let min_lat = indices
                    .iter()
                    .map(|&i| points[i].lat)
                    .fold(f64::MAX, f64::min);
                let min_lon = indices
                    .iter()
                    .map(|&i| points[i].lon)
                    .fold(f64::MAX, f64::min);
                let max_lat = indices
                    .iter()
                    .map(|&i| points[i].lat)
                    .fold(f64::MIN, f64::max);
                let max_lon = indices
                    .iter()
                    .map(|&i| points[i].lon)
                    .fold(f64::MIN, f64::max);

                cluster_stats.push(ClusterStats {
                    cluster_id: cid,
                    size,
                    centroid,
                    bbox: (min_lat, min_lon, max_lat, max_lon),
                });
            }
        }

        ClusterResult {
            assignments,
            num_clusters,
            num_noise,
            cluster_stats,
        }
    }

    /// Find all point indices within `eps_metres` of `points[center_idx]`.
    fn range_query(points: &[GeoPoint], center_idx: usize, eps_metres: f64) -> Vec<usize> {
        let center = &points[center_idx];
        points
            .iter()
            .enumerate()
            .filter(|(i, p)| {
                if *i == center_idx {
                    return true; // include the point itself
                }
                haversine_distance(center, p) <= eps_metres
            })
            .map(|(i, _)| i)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Haversine distance
// ---------------------------------------------------------------------------

/// Earth radius in metres (WGS-84 mean).
const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Compute the Haversine distance in metres between two WGS-84 points.
pub fn haversine_distance(a: &GeoPoint, b: &GeoPoint) -> f64 {
    let lat1 = a.lat.to_radians();
    let lat2 = b.lat.to_radians();
    let dlat = (b.lat - a.lat).to_radians();
    let dlon = (b.lon - a.lon).to_radians();

    let h = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * h.sqrt().asin();

    EARTH_RADIUS_M * c
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Haversine distance tests --

    #[test]
    fn test_haversine_same_point() {
        let p = GeoPoint::new(48.8566, 2.3522);
        let dist = haversine_distance(&p, &p);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_haversine_known_distance() {
        // London to Paris ~ 343 km
        let london = GeoPoint::new(51.5074, -0.1278);
        let paris = GeoPoint::new(48.8566, 2.3522);
        let dist = haversine_distance(&london, &paris);
        assert!(dist > 340_000.0 && dist < 350_000.0);
    }

    #[test]
    fn test_haversine_symmetric() {
        let a = GeoPoint::new(35.6762, 139.6503);
        let b = GeoPoint::new(48.8566, 2.3522);
        let d1 = haversine_distance(&a, &b);
        let d2 = haversine_distance(&b, &a);
        assert!((d1 - d2).abs() < 1e-6);
    }

    #[test]
    fn test_haversine_antipodal() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(0.0, 180.0);
        let dist = haversine_distance(&a, &b);
        // Should be half the circumference ~20_015_000 m
        assert!(dist > 20_000_000.0 && dist < 20_100_000.0);
    }

    #[test]
    fn test_haversine_poles() {
        let north = GeoPoint::new(90.0, 0.0);
        let south = GeoPoint::new(-90.0, 0.0);
        let dist = haversine_distance(&north, &south);
        // ~20_015_000 m
        assert!(dist > 20_000_000.0);
    }

    // -- GeoPoint --

    #[test]
    fn test_geopoint_new() {
        let p = GeoPoint::new(51.5, -0.1);
        assert_eq!(p.lat, 51.5);
        assert_eq!(p.lon, -0.1);
    }

    // -- DbscanConfig --

    #[test]
    fn test_default_config() {
        let config = DbscanConfig::default();
        assert_eq!(config.eps_metres, 1000.0);
        assert_eq!(config.min_points, 3);
    }

    // -- empty input --

    #[test]
    fn test_empty_input() {
        let config = DbscanConfig::default();
        let result = SpatialDbscan::cluster(&[], &config);
        assert_eq!(result.num_clusters, 0);
        assert_eq!(result.num_noise, 0);
        assert!(result.assignments.is_empty());
    }

    // -- single point --

    #[test]
    fn test_single_point_is_noise() {
        let config = DbscanConfig {
            eps_metres: 100.0,
            min_points: 2,
        };
        let points = vec![GeoPoint::new(48.8566, 2.3522)];
        let result = SpatialDbscan::cluster(&points, &config);
        assert_eq!(result.num_clusters, 0);
        assert_eq!(result.num_noise, 1);
        assert_eq!(result.label_for(0), Some(-1));
    }

    // -- two close points --

    #[test]
    fn test_two_close_points_form_cluster() {
        let config = DbscanConfig {
            eps_metres: 500.0,
            min_points: 2,
        };
        let points = vec![
            GeoPoint::new(48.8566, 2.3522),
            GeoPoint::new(48.8567, 2.3523),
        ];
        let result = SpatialDbscan::cluster(&points, &config);
        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.num_noise, 0);
    }

    // -- two far points --

    #[test]
    fn test_two_far_points_are_noise() {
        let config = DbscanConfig {
            eps_metres: 100.0,
            min_points: 2,
        };
        let points = vec![
            GeoPoint::new(48.8566, 2.3522),   // Paris
            GeoPoint::new(35.6762, 139.6503), // Tokyo
        ];
        let result = SpatialDbscan::cluster(&points, &config);
        assert_eq!(result.num_clusters, 0);
        assert_eq!(result.num_noise, 2);
    }

    // -- mixed clusters and noise --

    #[test]
    fn test_two_clusters_and_noise() {
        let config = DbscanConfig {
            eps_metres: 1000.0,
            min_points: 2,
        };
        let points = vec![
            // Cluster 1: Paris
            GeoPoint::new(48.8566, 2.3522),
            GeoPoint::new(48.8570, 2.3525),
            // Cluster 2: Tokyo
            GeoPoint::new(35.6762, 139.6503),
            GeoPoint::new(35.6765, 139.6506),
            // Noise: middle of the Atlantic
            GeoPoint::new(30.0, -40.0),
        ];
        let result = SpatialDbscan::cluster(&points, &config);
        assert_eq!(result.num_clusters, 2);
        assert_eq!(result.num_noise, 1);
        assert_eq!(result.label_for(4), Some(-1));
    }

    // -- ClusterResult methods --

    #[test]
    fn test_points_in_cluster() {
        let config = DbscanConfig {
            eps_metres: 1000.0,
            min_points: 2,
        };
        let points = vec![
            GeoPoint::new(48.8566, 2.3522),
            GeoPoint::new(48.8570, 2.3525),
            GeoPoint::new(35.6762, 139.6503),
        ];
        let result = SpatialDbscan::cluster(&points, &config);
        let cluster0 = result.points_in_cluster(0);
        assert!(cluster0.len() >= 2);
    }

    #[test]
    fn test_noise_points_accessor() {
        let config = DbscanConfig {
            eps_metres: 100.0,
            min_points: 2,
        };
        let points = vec![GeoPoint::new(0.0, 0.0)];
        let result = SpatialDbscan::cluster(&points, &config);
        assert_eq!(result.noise_points(), vec![0]);
    }

    // -- cluster stats --

    #[test]
    fn test_cluster_stats_centroid() {
        let config = DbscanConfig {
            eps_metres: 2000.0,
            min_points: 2,
        };
        let points = vec![
            GeoPoint::new(48.8566, 2.3522),
            GeoPoint::new(48.8570, 2.3525),
        ];
        let result = SpatialDbscan::cluster(&points, &config);
        assert_eq!(result.cluster_stats.len(), 1);
        let stats = &result.cluster_stats[0];
        let expected_lat = (48.8566 + 48.8570) / 2.0;
        assert!((stats.centroid.lat - expected_lat).abs() < 0.001);
    }

    #[test]
    fn test_cluster_stats_bbox() {
        let config = DbscanConfig {
            eps_metres: 2000.0,
            min_points: 2,
        };
        let points = vec![GeoPoint::new(48.0, 2.0), GeoPoint::new(49.0, 3.0)];
        let result = SpatialDbscan::cluster(&points, &config);
        if !result.cluster_stats.is_empty() {
            let bbox = result.cluster_stats[0].bbox;
            assert!(bbox.0 <= bbox.2); // min_lat <= max_lat
            assert!(bbox.1 <= bbox.3); // min_lon <= max_lon
        }
    }

    #[test]
    fn test_cluster_stats_size() {
        let config = DbscanConfig {
            eps_metres: 5000.0,
            min_points: 2,
        };
        let points = vec![
            GeoPoint::new(48.8566, 2.3522),
            GeoPoint::new(48.8570, 2.3525),
            GeoPoint::new(48.8580, 2.3530),
        ];
        let result = SpatialDbscan::cluster(&points, &config);
        assert_eq!(result.cluster_stats.len(), 1);
        assert_eq!(result.cluster_stats[0].size, 3);
    }

    // -- min_points edge cases --

    #[test]
    fn test_min_points_1_all_cluster() {
        let config = DbscanConfig {
            eps_metres: 100.0,
            min_points: 1,
        };
        let points = vec![GeoPoint::new(0.0, 0.0), GeoPoint::new(45.0, 90.0)];
        let result = SpatialDbscan::cluster(&points, &config);
        // Each point is its own cluster with min_points=1
        assert_eq!(result.num_noise, 0);
    }

    #[test]
    fn test_large_eps_single_cluster() {
        let config = DbscanConfig {
            eps_metres: 50_000_000.0, // larger than Earth circumference
            min_points: 2,
        };
        let points = vec![
            GeoPoint::new(48.8566, 2.3522),
            GeoPoint::new(35.6762, 139.6503),
            GeoPoint::new(-33.8688, 151.2093),
        ];
        let result = SpatialDbscan::cluster(&points, &config);
        assert_eq!(result.num_clusters, 1);
    }

    // -- label_for with missing index --

    #[test]
    fn test_label_for_out_of_range() {
        let config = DbscanConfig::default();
        let result = SpatialDbscan::cluster(&[], &config);
        assert_eq!(result.label_for(999), None);
    }

    // -- deterministic results --

    #[test]
    fn test_deterministic() {
        let config = DbscanConfig {
            eps_metres: 500.0,
            min_points: 2,
        };
        let points = vec![
            GeoPoint::new(48.8566, 2.3522),
            GeoPoint::new(48.8570, 2.3525),
            GeoPoint::new(0.0, 0.0),
        ];
        let r1 = SpatialDbscan::cluster(&points, &config);
        let r2 = SpatialDbscan::cluster(&points, &config);
        assert_eq!(r1.assignments.len(), r2.assignments.len());
        for (a, b) in r1.assignments.iter().zip(r2.assignments.iter()) {
            assert_eq!(a.cluster_id, b.cluster_id);
        }
    }

    // -- collinear points --

    #[test]
    fn test_collinear_points() {
        // Points along the equator, 0.001 degrees apart (~111m)
        let config = DbscanConfig {
            eps_metres: 200.0,
            min_points: 2,
        };
        let points: Vec<GeoPoint> = (0..5)
            .map(|i| GeoPoint::new(0.0, i as f64 * 0.001))
            .collect();
        let result = SpatialDbscan::cluster(&points, &config);
        // They should form a chain cluster
        assert!(result.num_clusters >= 1);
    }

    // -- all same point --

    #[test]
    fn test_all_same_location() {
        let config = DbscanConfig {
            eps_metres: 1.0,
            min_points: 2,
        };
        let points = vec![GeoPoint::new(51.5, -0.1); 5];
        let result = SpatialDbscan::cluster(&points, &config);
        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.cluster_stats[0].size, 5);
    }
}
