//! Heatmap generation for spatial density analysis
//!
//! This module provides functionality to generate heatmaps (density maps) from
//! point geometries, useful for visualizing spatial patterns and clusters.
//!
//! # Overview
//!
//! Heatmaps represent the density or intensity of point features across a spatial area.
//! Common use cases include:
//! - Crime hotspot analysis
//! - Population density visualization
//! - Disease outbreak mapping
//! - Customer location analysis
//!
//! # Examples
//!
//! ```rust
//! use oxirs_geosparql::geometry::Geometry;
//! use oxirs_geosparql::analysis::heatmap::{generate_heatmap, HeatmapConfig};
//!
//! // Generate heatmap from points
//! let points = vec![
//!     Geometry::from_wkt("POINT(0 0)").unwrap(),
//!     Geometry::from_wkt("POINT(1 1)").unwrap(),
//!     Geometry::from_wkt("POINT(0.5 0.5)").unwrap(),
//! ];
//!
//! let config = HeatmapConfig::default()
//!     .with_grid_size(100, 100)
//!     .with_radius(2.0);
//!
//! let heatmap = generate_heatmap(&points, &config).unwrap();
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo_types::{Coord, Rect};
use scirs2_core::ndarray_ext::Array2;
use serde::{Deserialize, Serialize};

/// Kernel function for heatmap generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelFunction {
    /// Gaussian (normal) distribution kernel
    Gaussian,
    /// Quartic (biweight) kernel
    Quartic,
    /// Epanechnikov kernel
    Epanechnikov,
    /// Triangular kernel
    Triangular,
    /// Uniform (flat) kernel
    Uniform,
}

impl KernelFunction {
    /// Calculate kernel value at distance d with bandwidth h
    pub fn evaluate(&self, d: f64, h: f64) -> f64 {
        let u = d / h;
        if u > 1.0 {
            return 0.0;
        }

        match self {
            KernelFunction::Gaussian => {
                // Gaussian kernel: (1/sqrt(2π)) * exp(-0.5 * u²)
                let normalization = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
                normalization * (-0.5 * u * u).exp()
            }
            KernelFunction::Quartic => {
                // Quartic kernel: (15/16) * (1 - u²)²
                let factor = 1.0 - u * u;
                (15.0 / 16.0) * factor * factor
            }
            KernelFunction::Epanechnikov => {
                // Epanechnikov kernel: (3/4) * (1 - u²)
                (3.0 / 4.0) * (1.0 - u * u)
            }
            KernelFunction::Triangular => {
                // Triangular kernel: (1 - |u|)
                1.0 - u.abs()
            }
            KernelFunction::Uniform => {
                // Uniform kernel: 0.5 for u <= 1
                0.5
            }
        }
    }
}

/// Configuration for heatmap generation
#[derive(Debug, Clone)]
pub struct HeatmapConfig {
    /// Grid width (number of cells in X direction)
    pub grid_width: usize,
    /// Grid height (number of cells in Y direction)
    pub grid_height: usize,
    /// Radius of influence (bandwidth) for each point
    pub radius: f64,
    /// Kernel function to use
    pub kernel: KernelFunction,
    /// Optional bounding box (if None, computed from data)
    pub bounds: Option<Rect<f64>>,
    /// Normalize output values to [0, 1] range
    pub normalize: bool,
    /// Weight values for each point (if None, all points have weight 1.0)
    pub weights: Option<Vec<f64>>,
}

impl Default for HeatmapConfig {
    fn default() -> Self {
        Self {
            grid_width: 100,
            grid_height: 100,
            radius: 1.0,
            kernel: KernelFunction::Gaussian,
            bounds: None,
            normalize: true,
            weights: None,
        }
    }
}

impl HeatmapConfig {
    /// Set the grid size
    pub fn with_grid_size(mut self, width: usize, height: usize) -> Self {
        self.grid_width = width;
        self.grid_height = height;
        self
    }

    /// Set the radius of influence
    pub fn with_radius(mut self, radius: f64) -> Self {
        self.radius = radius;
        self
    }

    /// Set the kernel function
    pub fn with_kernel(mut self, kernel: KernelFunction) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set custom bounds
    pub fn with_bounds(mut self, bounds: Rect<f64>) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Set normalization
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set weights for points
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }
}

/// Heatmap result
#[derive(Debug, Clone)]
pub struct Heatmap {
    /// Density values as 2D grid `[y][x]`
    pub grid: Array2<f64>,
    /// Bounding box of the heatmap
    pub bounds: Rect<f64>,
    /// Cell width in coordinate units
    pub cell_width: f64,
    /// Cell height in coordinate units
    pub cell_height: f64,
    /// Maximum density value
    pub max_value: f64,
    /// Minimum density value
    pub min_value: f64,
}

impl Heatmap {
    /// Get the density value at grid coordinates (x, y)
    pub fn get(&self, x: usize, y: usize) -> Option<f64> {
        if y < self.grid.nrows() && x < self.grid.ncols() {
            Some(self.grid[[y, x]])
        } else {
            None
        }
    }

    /// Get the density value at geographic coordinates
    pub fn get_at_coord(&self, coord: &Coord<f64>) -> Option<f64> {
        let (x, y) = self.coord_to_grid(coord)?;
        self.get(x, y)
    }

    /// Convert geographic coordinate to grid indices
    pub fn coord_to_grid(&self, coord: &Coord<f64>) -> Option<(usize, usize)> {
        if coord.x < self.bounds.min().x
            || coord.x > self.bounds.max().x
            || coord.y < self.bounds.min().y
            || coord.y > self.bounds.max().y
        {
            return None;
        }

        let x = ((coord.x - self.bounds.min().x) / self.cell_width) as usize;
        let y = ((coord.y - self.bounds.min().y) / self.cell_height) as usize;

        let x = x.min(self.grid.ncols() - 1);
        let y = y.min(self.grid.nrows() - 1);

        Some((x, y))
    }

    /// Convert grid indices to geographic coordinate (cell center)
    pub fn grid_to_coord(&self, x: usize, y: usize) -> Option<Coord<f64>> {
        if y >= self.grid.nrows() || x >= self.grid.ncols() {
            return None;
        }

        let coord_x = self.bounds.min().x + (x as f64 + 0.5) * self.cell_width;
        let coord_y = self.bounds.min().y + (y as f64 + 0.5) * self.cell_height;

        Some(Coord {
            x: coord_x,
            y: coord_y,
        })
    }

    /// Export heatmap as normalized values in [0, 1] range
    pub fn to_normalized(&self) -> Array2<f64> {
        if self.max_value == self.min_value {
            return Array2::zeros((self.grid.nrows(), self.grid.ncols()));
        }

        let range = self.max_value - self.min_value;
        self.grid.mapv(|v| (v - self.min_value) / range)
    }

    /// Find local maxima (hotspots) above a threshold percentile
    pub fn find_hotspots(&self, percentile: f64) -> Vec<(usize, usize, f64)> {
        let threshold = self.min_value + (self.max_value - self.min_value) * percentile;
        let mut hotspots = Vec::new();

        for y in 1..(self.grid.nrows() - 1) {
            for x in 1..(self.grid.ncols() - 1) {
                let value = self.grid[[y, x]];

                if value < threshold {
                    continue;
                }

                // Check if this is a local maximum
                let is_local_max = value >= self.grid[[y - 1, x - 1]]
                    && value >= self.grid[[y - 1, x]]
                    && value >= self.grid[[y - 1, x + 1]]
                    && value >= self.grid[[y, x - 1]]
                    && value >= self.grid[[y, x + 1]]
                    && value >= self.grid[[y + 1, x - 1]]
                    && value >= self.grid[[y + 1, x]]
                    && value >= self.grid[[y + 1, x + 1]];

                if is_local_max {
                    hotspots.push((x, y, value));
                }
            }
        }

        hotspots.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        hotspots
    }
}

/// Generate a heatmap from point geometries
///
/// # Arguments
///
/// * `points` - Collection of point geometries
/// * `config` - Heatmap configuration
///
/// # Returns
///
/// A Heatmap containing the density grid and metadata
///
/// # Examples
///
/// ```rust
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::analysis::heatmap::{generate_heatmap, HeatmapConfig};
///
/// let points = vec![
///     Geometry::from_wkt("POINT(0 0)").unwrap(),
///     Geometry::from_wkt("POINT(1 1)").unwrap(),
/// ];
///
/// let config = HeatmapConfig::default();
/// let heatmap = generate_heatmap(&points, &config).unwrap();
/// ```
pub fn generate_heatmap(points: &[Geometry], config: &HeatmapConfig) -> Result<Heatmap> {
    if points.is_empty() {
        return Err(GeoSparqlError::InvalidInput(
            "Cannot generate heatmap from empty point set".to_string(),
        ));
    }

    // Extract point coordinates
    let coords: Vec<Coord<f64>> = points
        .iter()
        .filter_map(|geom| {
            if let geo_types::Geometry::Point(p) = &geom.geom {
                Some(p.0)
            } else {
                None
            }
        })
        .collect();

    if coords.is_empty() {
        return Err(GeoSparqlError::InvalidInput(
            "No valid point geometries found".to_string(),
        ));
    }

    // Determine bounds
    let bounds = if let Some(b) = config.bounds {
        b
    } else {
        compute_bounds(&coords)?
    };

    // Calculate cell dimensions
    let cell_width = (bounds.max().x - bounds.min().x) / config.grid_width as f64;
    let cell_height = (bounds.max().y - bounds.min().y) / config.grid_height as f64;

    // Initialize grid
    let mut grid = Array2::zeros((config.grid_height, config.grid_width));

    // Get weights or use uniform weights
    let default_weights = vec![1.0; coords.len()];
    let weights = config.weights.as_deref().unwrap_or(&default_weights);

    if weights.len() != coords.len() {
        return Err(GeoSparqlError::InvalidParameter(format!(
            "Weight count ({}) doesn't match point count ({})",
            weights.len(),
            coords.len()
        )));
    }

    // Compute density for each grid cell
    for y in 0..config.grid_height {
        for x in 0..config.grid_width {
            let cell_center = Coord {
                x: bounds.min().x + (x as f64 + 0.5) * cell_width,
                y: bounds.min().y + (y as f64 + 0.5) * cell_height,
            };

            let mut density = 0.0;

            for (i, coord) in coords.iter().enumerate() {
                let distance =
                    ((cell_center.x - coord.x).powi(2) + (cell_center.y - coord.y).powi(2)).sqrt();

                if distance <= config.radius {
                    let kernel_value = config.kernel.evaluate(distance, config.radius);
                    density += kernel_value * weights[i];
                }
            }

            grid[[y, x]] = density;
        }
    }

    // Find min and max values
    let max_value = grid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_value = grid.iter().cloned().fold(f64::INFINITY, f64::min);

    // Normalize if requested
    if config.normalize && max_value > min_value {
        let range = max_value - min_value;
        grid.mapv_inplace(|v| (v - min_value) / range);

        // Update min/max to reflect normalization
        let normalized_max = grid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let normalized_min = grid.iter().cloned().fold(f64::INFINITY, f64::min);

        return Ok(Heatmap {
            grid,
            bounds,
            cell_width,
            cell_height,
            max_value: normalized_max,
            min_value: normalized_min,
        });
    }

    Ok(Heatmap {
        grid,
        bounds,
        cell_width,
        cell_height,
        max_value,
        min_value,
    })
}

/// Compute bounding box from coordinates
fn compute_bounds(coords: &[Coord<f64>]) -> Result<Rect<f64>> {
    if coords.is_empty() {
        return Err(GeoSparqlError::InvalidInput(
            "Cannot compute bounds from empty coordinate set".to_string(),
        ));
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for coord in coords {
        min_x = min_x.min(coord.x);
        max_x = max_x.max(coord.x);
        min_y = min_y.min(coord.y);
        max_y = max_y.max(coord.y);
    }

    // Add 10% margin
    let margin_x = (max_x - min_x) * 0.1;
    let margin_y = (max_y - min_y) * 0.1;

    Ok(Rect::new(
        Coord {
            x: min_x - margin_x,
            y: min_y - margin_y,
        },
        Coord {
            x: max_x + margin_x,
            y: max_y + margin_y,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_gaussian() {
        let kernel = KernelFunction::Gaussian;
        let value_at_center = kernel.evaluate(0.0, 1.0);
        let value_at_edge = kernel.evaluate(1.0, 1.0);

        assert!(value_at_center > value_at_edge);
        assert!(value_at_edge > 0.0);
        assert_eq!(kernel.evaluate(2.0, 1.0), 0.0); // Outside radius
    }

    #[test]
    fn test_kernel_uniform() {
        let kernel = KernelFunction::Uniform;
        assert_eq!(kernel.evaluate(0.0, 1.0), 0.5);
        assert_eq!(kernel.evaluate(0.5, 1.0), 0.5);
        assert_eq!(kernel.evaluate(1.0, 1.0), 0.5);
        assert_eq!(kernel.evaluate(1.5, 1.0), 0.0);
    }

    #[test]
    fn test_heatmap_config() {
        let config = HeatmapConfig::default()
            .with_grid_size(50, 50)
            .with_radius(2.0)
            .with_kernel(KernelFunction::Quartic)
            .with_normalization(false);

        assert_eq!(config.grid_width, 50);
        assert_eq!(config.grid_height, 50);
        assert_eq!(config.radius, 2.0);
        assert_eq!(config.kernel, KernelFunction::Quartic);
        assert!(!config.normalize);
    }

    #[test]
    fn test_generate_heatmap_basic() {
        let points = vec![
            Geometry::from_wkt("POINT(0 0)").unwrap(),
            Geometry::from_wkt("POINT(1 1)").unwrap(),
            Geometry::from_wkt("POINT(0.5 0.5)").unwrap(),
        ];

        let config = HeatmapConfig::default()
            .with_grid_size(10, 10)
            .with_radius(2.0);

        let heatmap = generate_heatmap(&points, &config).unwrap();

        assert_eq!(heatmap.grid.nrows(), 10);
        assert_eq!(heatmap.grid.ncols(), 10);
        assert!(heatmap.max_value >= heatmap.min_value);
    }

    #[test]
    fn test_heatmap_with_weights() {
        let points = vec![
            Geometry::from_wkt("POINT(0 0)").unwrap(),
            Geometry::from_wkt("POINT(1 1)").unwrap(),
        ];

        let weights = vec![1.0, 10.0]; // Second point has 10x weight

        let config = HeatmapConfig::default()
            .with_grid_size(10, 10)
            .with_weights(weights);

        let heatmap = generate_heatmap(&points, &config).unwrap();

        // The cell near (1, 1) should have higher density
        assert!(heatmap.max_value > heatmap.min_value);
    }

    #[test]
    fn test_coord_to_grid_conversion() {
        let points = vec![Geometry::from_wkt("POINT(0 0)").unwrap()];

        let config = HeatmapConfig::default().with_grid_size(10, 10);

        let heatmap = generate_heatmap(&points, &config).unwrap();

        let coord = Coord { x: 0.0, y: 0.0 };
        let (x, y) = heatmap.coord_to_grid(&coord).unwrap();

        assert!(x < 10);
        assert!(y < 10);
    }

    #[test]
    fn test_find_hotspots() {
        let points = vec![
            Geometry::from_wkt("POINT(0 0)").unwrap(),
            Geometry::from_wkt("POINT(0.1 0.1)").unwrap(),
            Geometry::from_wkt("POINT(0.2 0.2)").unwrap(),
            Geometry::from_wkt("POINT(5 5)").unwrap(),
        ];

        let config = HeatmapConfig::default()
            .with_grid_size(20, 20)
            .with_radius(1.0);

        let heatmap = generate_heatmap(&points, &config).unwrap();

        let hotspots = heatmap.find_hotspots(0.8); // Top 20%

        // Should find at least one hotspot
        assert!(!hotspots.is_empty());

        // Hotspots should be sorted by density
        if hotspots.len() > 1 {
            assert!(hotspots[0].2 >= hotspots[1].2);
        }
    }

    #[test]
    fn test_empty_points() {
        let points: Vec<Geometry> = vec![];
        let config = HeatmapConfig::default();

        assert!(generate_heatmap(&points, &config).is_err());
    }

    #[test]
    fn test_normalized_output() {
        let points = vec![
            Geometry::from_wkt("POINT(0 0)").unwrap(),
            Geometry::from_wkt("POINT(1 1)").unwrap(),
        ];

        let config = HeatmapConfig::default()
            .with_grid_size(10, 10)
            .with_normalization(true);

        let heatmap = generate_heatmap(&points, &config).unwrap();

        let normalized = heatmap.to_normalized();

        // All values should be in [0, 1]
        for value in normalized.iter() {
            assert!(*value >= 0.0 && *value <= 1.0);
        }
    }
}
