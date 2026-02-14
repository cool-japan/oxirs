//! Advanced Spatial Analysis
//!
//! This module provides advanced spatial analysis algorithms including:
//! - Voronoi diagrams and Delaunay triangulation
//! - Spatial clustering (DBSCAN, K-means)
//! - Spatial interpolation (IDW, Kriging)
//! - Spatial statistics (Moran's I, Getis-Ord)
//! - Spatial aggregations (union, convex hull, centroid, envelope)
//!
//! All algorithms leverage SciRS2 for high-performance numerical computations
//! with SIMD, parallel processing, and GPU acceleration where applicable.

pub mod aggregations;
pub mod clustering;
pub mod heatmap;
pub mod interpolation;
pub mod network;
pub mod statistics;
pub mod triangulation;
pub mod voronoi;

pub use aggregations::{
    aggregate_centroid, aggregate_convex_hull, aggregate_envelope, aggregate_union,
    spatial_centroid, spatial_convex_hull, spatial_dbscan, spatial_envelope, spatial_union,
    ClusterId,
};
pub use clustering::{
    dbscan_clustering, kmeans_clustering, ClusteringResult, DbscanParams, KmeansParams,
};
pub use heatmap::{generate_heatmap, Heatmap, HeatmapConfig, KernelFunction};
pub use interpolation::{
    idw_interpolation, kriging_interpolation, InterpolationMethod, InterpolationResult, SamplePoint,
};
pub use network::{astar_shortest_path, dijkstra_shortest_path, Network, PathResult};
pub use statistics::{getis_ord_gi_star, morans_i, SpatialAutocorrelation, WeightsMatrixType};
pub use triangulation::{delaunay_triangulation, DelaunayTriangulation, Triangle};
pub use voronoi::{voronoi_diagram, VoronoiCell};
