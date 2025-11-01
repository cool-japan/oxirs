//! Advanced Spatial Analysis
//!
//! This module provides advanced spatial analysis algorithms including:
//! - Voronoi diagrams and Delaunay triangulation
//! - Spatial clustering (DBSCAN, K-means)
//! - Spatial interpolation (IDW, Kriging)
//! - Spatial statistics (Moran's I, Getis-Ord)
//!
//! All algorithms leverage SciRS2 for high-performance numerical computations
//! with SIMD, parallel processing, and GPU acceleration where applicable.

pub mod clustering;
pub mod interpolation;
pub mod statistics;
pub mod triangulation;
pub mod voronoi;

pub use clustering::{
    dbscan_clustering, kmeans_clustering, ClusteringResult, DbscanParams, KmeansParams,
};
pub use interpolation::{
    idw_interpolation, kriging_interpolation, InterpolationMethod, InterpolationResult, SamplePoint,
};
pub use statistics::{getis_ord_gi_star, morans_i, SpatialAutocorrelation, WeightsMatrixType};
pub use triangulation::{delaunay_triangulation, DelaunayTriangulation, Triangle};
pub use voronoi::{voronoi_diagram, VoronoiCell};
