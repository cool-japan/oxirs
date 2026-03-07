//! Raster grid sampling with multiple interpolation methods.
//!
//! Supports nearest-neighbour, bilinear, and bicubic interpolation along
//! with raster statistics and resampling.

// ─────────────────────────────────────────────────
// GridExtent
// ─────────────────────────────────────────────────

/// The geographic extent of a raster grid.
#[derive(Debug, Clone, PartialEq)]
pub struct GridExtent {
    /// Minimum X coordinate (west boundary).
    pub min_x: f64,
    /// Maximum X coordinate (east boundary).
    pub max_x: f64,
    /// Minimum Y coordinate (south boundary).
    pub min_y: f64,
    /// Maximum Y coordinate (north boundary).
    pub max_y: f64,
}

impl GridExtent {
    /// Construct a new grid extent.
    pub fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Self {
        GridExtent {
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    /// Width of the extent in world units.
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Height of the extent in world units.
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }
}

// ─────────────────────────────────────────────────
// RasterGrid
// ─────────────────────────────────────────────────

/// A 2-D raster grid stored as a row-major flat `Vec<f64>`.
/// Row 0 is the top row (north), increasing row index goes south.
#[derive(Debug, Clone)]
pub struct RasterGrid {
    /// Number of columns (pixels) in the grid.
    pub width: usize,
    /// Number of rows (pixels) in the grid.
    pub height: usize,
    /// Geographic extent covered by the grid.
    pub extent: GridExtent,
    /// Row-major flat array of cell values.
    pub values: Vec<f64>,
    /// Sentinel value indicating "no data" (typically `NAN` or `-9999`).
    pub no_data: f64,
}

impl RasterGrid {
    /// Create a new grid filled with `no_data` values.
    pub fn new(width: usize, height: usize, extent: GridExtent) -> Self {
        let no_data = f64::NAN;
        RasterGrid {
            width,
            height,
            extent,
            values: vec![no_data; width * height],
            no_data,
        }
    }

    /// Set the value at grid cell (col, row).
    pub fn set_value(&mut self, col: usize, row: usize, value: f64) {
        if col < self.width && row < self.height {
            self.values[row * self.width + col] = value;
        }
    }

    /// Get the value at grid cell (col, row).
    pub fn get_value(&self, col: usize, row: usize) -> f64 {
        if col < self.width && row < self.height {
            self.values[row * self.width + col]
        } else {
            self.no_data
        }
    }

    /// Convert world coordinates to continuous pixel coordinates (col, row).
    /// Pixel (0, 0) is the top-left corner.
    pub fn world_to_pixel(&self, x: f64, y: f64) -> (f64, f64) {
        let col = (x - self.extent.min_x) / self.extent.width() * (self.width as f64 - 1.0);
        let row = (self.extent.max_y - y) / self.extent.height() * (self.height as f64 - 1.0);
        (col, row)
    }

    /// Convert continuous pixel coordinates to world coordinates.
    pub fn pixel_to_world(&self, col: f64, row: f64) -> (f64, f64) {
        let x = self.extent.min_x + col / (self.width as f64 - 1.0) * self.extent.width();
        let y = self.extent.max_y - row / (self.height as f64 - 1.0) * self.extent.height();
        (x, y)
    }

    /// Check if (col, row) are within the grid bounds.
    pub fn in_bounds(&self, col: f64, row: f64) -> bool {
        col >= 0.0
            && row >= 0.0
            && col <= (self.width as f64 - 1.0)
            && row <= (self.height as f64 - 1.0)
    }

    /// Get value at integer (col, row), returning `no_data` if out of bounds.
    fn cell(&self, col: i64, row: i64) -> f64 {
        if col < 0 || row < 0 || col as usize >= self.width || row as usize >= self.height {
            self.no_data
        } else {
            self.values[row as usize * self.width + col as usize]
        }
    }

    fn is_no_data(&self, v: f64) -> bool {
        v.is_nan() || (self.no_data.is_finite() && (v - self.no_data).abs() < 1e-12)
    }
}

// ─────────────────────────────────────────────────
// InterpolationMethod
// ─────────────────────────────────────────────────

/// The interpolation algorithm to use when sampling at a non-integer pixel position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Snap to the nearest grid cell (fastest, but blocky).
    NearestNeighbor,
    /// Weighted average of the 2×2 surrounding cells.
    Bilinear,
    /// Catmull-Rom cubic interpolation over a 4×4 neighbourhood.
    Bicubic,
}

// ─────────────────────────────────────────────────
// SampleResult
// ─────────────────────────────────────────────────

/// The result of sampling a raster at a single world coordinate.
#[derive(Debug, Clone)]
pub struct SampleResult {
    /// World X coordinate of the sample point.
    pub x: f64,
    /// World Y coordinate of the sample point.
    pub y: f64,
    /// Interpolated value at the sample point.
    pub value: f64,
    /// `true` if the sample position fell outside the grid or on a no-data cell.
    pub is_no_data: bool,
}

// ─────────────────────────────────────────────────
// RasterSampler
// ─────────────────────────────────────────────────

/// Stateless raster sampling utilities.
pub struct RasterSampler;

impl RasterSampler {
    /// Sample using the specified method.
    pub fn sample(grid: &RasterGrid, x: f64, y: f64, method: InterpolationMethod) -> SampleResult {
        match method {
            InterpolationMethod::NearestNeighbor => Self::sample_nearest(grid, x, y),
            InterpolationMethod::Bilinear => Self::sample_bilinear(grid, x, y),
            InterpolationMethod::Bicubic => Self::sample_bicubic(grid, x, y),
        }
    }

    /// Nearest-neighbour sampling.
    pub fn sample_nearest(grid: &RasterGrid, x: f64, y: f64) -> SampleResult {
        let (col, row) = grid.world_to_pixel(x, y);
        if !grid.in_bounds(col, row) {
            return SampleResult {
                x,
                y,
                value: grid.no_data,
                is_no_data: true,
            };
        }
        let c = col.round() as i64;
        let r = row.round() as i64;
        let value = grid.cell(c, r);
        SampleResult {
            x,
            y,
            value,
            is_no_data: grid.is_no_data(value),
        }
    }

    /// Bilinear interpolation sampling.
    pub fn sample_bilinear(grid: &RasterGrid, x: f64, y: f64) -> SampleResult {
        let (col, row) = grid.world_to_pixel(x, y);
        if !grid.in_bounds(col, row) {
            return SampleResult {
                x,
                y,
                value: grid.no_data,
                is_no_data: true,
            };
        }
        let c0 = col.floor() as i64;
        let r0 = row.floor() as i64;
        let c1 = c0 + 1;
        let r1 = r0 + 1;

        let tx = col - c0 as f64;
        let ty = row - r0 as f64;

        let q00 = grid.cell(c0, r0);
        let q10 = grid.cell(c1, r0);
        let q01 = grid.cell(c0, r1);
        let q11 = grid.cell(c1, r1);

        // If any corner is no_data, fall back to nearest
        if grid.is_no_data(q00)
            || grid.is_no_data(q10)
            || grid.is_no_data(q01)
            || grid.is_no_data(q11)
        {
            return Self::sample_nearest(grid, x, y);
        }

        let value = (1.0 - tx) * (1.0 - ty) * q00
            + tx * (1.0 - ty) * q10
            + (1.0 - tx) * ty * q01
            + tx * ty * q11;

        SampleResult {
            x,
            y,
            value,
            is_no_data: grid.is_no_data(value),
        }
    }

    /// Bicubic interpolation sampling.
    ///
    /// Uses the Catmull-Rom variant of bicubic interpolation over a 4×4 neighbourhood.
    pub fn sample_bicubic(grid: &RasterGrid, x: f64, y: f64) -> SampleResult {
        let (col, row) = grid.world_to_pixel(x, y);
        if !grid.in_bounds(col, row) {
            return SampleResult {
                x,
                y,
                value: grid.no_data,
                is_no_data: true,
            };
        }

        let c0 = col.floor() as i64;
        let r0 = row.floor() as i64;
        let tx = col - c0 as f64;
        let ty = row - r0 as f64;

        // Collect 4×4 neighbourhood
        let mut patch = [[0.0_f64; 4]; 4];
        let mut has_no_data = false;
        for j in 0..4_i64 {
            for i in 0..4_i64 {
                let v = grid.cell(c0 - 1 + i, r0 - 1 + j);
                if grid.is_no_data(v) {
                    has_no_data = true;
                }
                patch[j as usize][i as usize] = v;
            }
        }
        if has_no_data {
            return Self::sample_bilinear(grid, x, y);
        }

        let row_interp: Vec<f64> = (0..4)
            .map(|j| cubic_interp(patch[j][0], patch[j][1], patch[j][2], patch[j][3], tx))
            .collect();
        let value = cubic_interp(
            row_interp[0],
            row_interp[1],
            row_interp[2],
            row_interp[3],
            ty,
        );

        SampleResult {
            x,
            y,
            value,
            is_no_data: grid.is_no_data(value),
        }
    }

    /// Sample at `n_samples` evenly-spaced points along the line (x1,y1)→(x2,y2).
    pub fn sample_line(
        grid: &RasterGrid,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        n_samples: usize,
        method: InterpolationMethod,
    ) -> Vec<SampleResult> {
        if n_samples == 0 {
            return Vec::new();
        }
        (0..n_samples)
            .map(|i| {
                let t = if n_samples == 1 {
                    0.0
                } else {
                    i as f64 / (n_samples - 1) as f64
                };
                let x = x1 + t * (x2 - x1);
                let y = y1 + t * (y2 - y1);
                Self::sample(grid, x, y, method.clone())
            })
            .collect()
    }

    /// Compute (min, max, mean, std_dev) of all non-no_data cells.
    pub fn statistics(grid: &RasterGrid) -> (f64, f64, f64, f64) {
        let valid: Vec<f64> = grid
            .values
            .iter()
            .copied()
            .filter(|&v| !grid.is_no_data(v))
            .collect();

        if valid.is_empty() {
            return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        }

        let min = valid.iter().copied().fold(f64::INFINITY, f64::min);
        let max = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let n = valid.len() as f64;
        let mean = valid.iter().sum::<f64>() / n;
        let variance = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        (min, max, mean, std_dev)
    }

    /// Resample the grid to a new (width × height) by sampling with `method`.
    pub fn resample(
        grid: &RasterGrid,
        new_width: usize,
        new_height: usize,
        method: InterpolationMethod,
    ) -> RasterGrid {
        let new_extent = grid.extent.clone();
        let mut new_grid = RasterGrid::new(new_width, new_height, new_extent);
        new_grid.no_data = grid.no_data;

        for r in 0..new_height {
            for c in 0..new_width {
                let (x, y) = new_grid.pixel_to_world(c as f64, r as f64);
                let sample = Self::sample(grid, x, y, method.clone());
                if !sample.is_no_data {
                    new_grid.set_value(c, r, sample.value);
                }
            }
        }
        new_grid
    }
}

// ─────────────────────────────────────────────────
// Cubic interpolation kernel (Catmull-Rom)
// ─────────────────────────────────────────────────

fn cubic_interp(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

// ─────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 3×3 grid spanning [0,2]×[0,2] with values 0..8.
    fn make_3x3() -> RasterGrid {
        let extent = GridExtent::new(0.0, 2.0, 0.0, 2.0);
        let mut g = RasterGrid::new(3, 3, extent);
        g.no_data = -9999.0;
        // Row 0 (top): y=2, Row 2 (bottom): y=0
        for r in 0..3 {
            for c in 0..3 {
                g.set_value(c, r, (r * 3 + c) as f64);
            }
        }
        g
    }

    /// Build a 5×5 uniform grid (all cells = 1.0).
    fn make_uniform(val: f64) -> RasterGrid {
        let extent = GridExtent::new(0.0, 4.0, 0.0, 4.0);
        let mut g = RasterGrid::new(5, 5, extent);
        g.no_data = -9999.0;
        for r in 0..5 {
            for c in 0..5 {
                g.set_value(c, r, val);
            }
        }
        g
    }

    // ── GridExtent ─────────────────────────────────────────────

    #[test]
    fn test_extent_width_height() {
        let e = GridExtent::new(0.0, 10.0, 0.0, 5.0);
        assert!((e.width() - 10.0).abs() < 1e-10);
        assert!((e.height() - 5.0).abs() < 1e-10);
    }

    // ── RasterGrid basics ──────────────────────────────────────

    #[test]
    fn test_grid_set_get_value() {
        let mut g = RasterGrid::new(4, 4, GridExtent::new(0.0, 3.0, 0.0, 3.0));
        g.set_value(1, 2, 42.0);
        assert_eq!(g.get_value(1, 2), 42.0);
    }

    #[test]
    fn test_grid_out_of_bounds_get() {
        let g = RasterGrid::new(3, 3, GridExtent::new(0.0, 2.0, 0.0, 2.0));
        assert!(g.get_value(10, 10).is_nan());
    }

    #[test]
    fn test_world_to_pixel_corners() {
        let g = make_3x3();
        let (c, r) = g.world_to_pixel(0.0, 2.0); // top-left
        assert!((c).abs() < 1e-9);
        assert!((r).abs() < 1e-9);

        let (c, r) = g.world_to_pixel(2.0, 0.0); // bottom-right
        assert!((c - 2.0).abs() < 1e-9);
        assert!((r - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_world_to_pixel_roundtrip() {
        let g = make_3x3();
        let (x0, y0) = (1.0, 1.5);
        let (col, row) = g.world_to_pixel(x0, y0);
        let (x1, y1) = g.pixel_to_world(col, row);
        assert!((x1 - x0).abs() < 1e-9);
        assert!((y1 - y0).abs() < 1e-9);
    }

    #[test]
    fn test_in_bounds_corners() {
        let g = make_3x3();
        assert!(g.in_bounds(0.0, 0.0));
        assert!(g.in_bounds(2.0, 2.0));
        assert!(!g.in_bounds(-0.01, 0.0));
        assert!(!g.in_bounds(0.0, 2.01));
    }

    // ── Nearest-neighbour ─────────────────────────────────────

    #[test]
    fn test_nearest_at_grid_point() {
        let g = make_3x3();
        // Top-left corner: (x=0, y=2) → cell (0,0) = 0
        let s = RasterSampler::sample_nearest(&g, 0.0, 2.0);
        assert!((s.value - 0.0).abs() < 1e-9);
        assert!(!s.is_no_data);
    }

    #[test]
    fn test_nearest_at_center() {
        let g = make_3x3();
        // Centre: (x=1, y=1) → pixel (1,1) = cell (1,1) = 4
        let s = RasterSampler::sample_nearest(&g, 1.0, 1.0);
        assert!((s.value - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_nearest_outside_returns_no_data() {
        let g = make_3x3();
        let s = RasterSampler::sample_nearest(&g, -1.0, -1.0);
        assert!(s.is_no_data);
    }

    #[test]
    fn test_nearest_bottom_right_corner() {
        let g = make_3x3();
        // Bottom-right: (x=2, y=0) → pixel (2,2) = 8
        let s = RasterSampler::sample_nearest(&g, 2.0, 0.0);
        assert!((s.value - 8.0).abs() < 1e-9);
    }

    // ── Bilinear ──────────────────────────────────────────────

    #[test]
    fn test_bilinear_at_grid_points_matches_exact() {
        let g = make_3x3();
        for r in 0..3 {
            for c in 0..3 {
                let (wx, wy) = g.pixel_to_world(c as f64, r as f64);
                let s = RasterSampler::sample_bilinear(&g, wx, wy);
                let expected = (r * 3 + c) as f64;
                assert!(
                    (s.value - expected).abs() < 1e-6,
                    "Bilinear at grid point ({c},{r}) expected {expected}, got {}",
                    s.value
                );
            }
        }
    }

    #[test]
    fn test_bilinear_midpoint_between_four_equal_values() {
        // All cells = 5.0 → any bilinear sample must be 5.0
        let g = make_uniform(5.0);
        let s = RasterSampler::sample_bilinear(&g, 1.5, 2.5);
        assert!((s.value - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_bilinear_outside_returns_no_data() {
        let g = make_3x3();
        let s = RasterSampler::sample_bilinear(&g, 100.0, 100.0);
        assert!(s.is_no_data);
    }

    #[test]
    fn test_bilinear_interpolation_midpoint() {
        // 2×2 grid with known values
        let extent = GridExtent::new(0.0, 1.0, 0.0, 1.0);
        let mut g = RasterGrid::new(2, 2, extent);
        g.no_data = -9999.0;
        g.set_value(0, 0, 0.0); // top-left  (x=0, y=1)
        g.set_value(1, 0, 2.0); // top-right (x=1, y=1)
        g.set_value(0, 1, 2.0); // bot-left  (x=0, y=0)
        g.set_value(1, 1, 4.0); // bot-right (x=1, y=0)
                                // Centre = average = 2.0
        let s = RasterSampler::sample_bilinear(&g, 0.5, 0.5);
        assert!((s.value - 2.0).abs() < 1e-6);
    }

    // ── Bicubic ───────────────────────────────────────────────

    #[test]
    fn test_bicubic_at_grid_points_near_exact() {
        // For a uniform grid, bicubic must also return the uniform value
        let g = make_uniform(7.0);
        let s = RasterSampler::sample_bicubic(&g, 2.0, 2.0);
        assert!((s.value - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_bicubic_outside_returns_no_data() {
        let g = make_3x3();
        let s = RasterSampler::sample_bicubic(&g, -5.0, -5.0);
        assert!(s.is_no_data);
    }

    // ── sample() dispatcher ────────────────────────────────────

    #[test]
    fn test_sample_nearest_via_dispatch() {
        let g = make_3x3();
        let s1 = RasterSampler::sample(&g, 0.0, 2.0, InterpolationMethod::NearestNeighbor);
        let s2 = RasterSampler::sample_nearest(&g, 0.0, 2.0);
        assert!((s1.value - s2.value).abs() < 1e-12);
    }

    #[test]
    fn test_sample_bilinear_via_dispatch() {
        let g = make_uniform(3.0);
        let s1 = RasterSampler::sample(&g, 1.0, 2.0, InterpolationMethod::Bilinear);
        let s2 = RasterSampler::sample_bilinear(&g, 1.0, 2.0);
        assert!((s1.value - s2.value).abs() < 1e-12);
    }

    #[test]
    fn test_sample_bicubic_via_dispatch() {
        let g = make_uniform(9.0);
        let s1 = RasterSampler::sample(&g, 2.0, 2.0, InterpolationMethod::Bicubic);
        let s2 = RasterSampler::sample_bicubic(&g, 2.0, 2.0);
        assert!((s1.value - s2.value).abs() < 1e-12);
    }

    // ── Statistics ────────────────────────────────────────────

    #[test]
    fn test_statistics_uniform() {
        let g = make_uniform(4.0);
        let (min, max, mean, std_dev) = RasterSampler::statistics(&g);
        assert!((min - 4.0).abs() < 1e-9);
        assert!((max - 4.0).abs() < 1e-9);
        assert!((mean - 4.0).abs() < 1e-9);
        assert!(std_dev.abs() < 1e-9);
    }

    #[test]
    fn test_statistics_two_values() {
        let extent = GridExtent::new(0.0, 1.0, 0.0, 1.0);
        let mut g = RasterGrid::new(2, 1, extent);
        g.no_data = -9999.0;
        g.set_value(0, 0, 0.0);
        g.set_value(1, 0, 4.0);
        let (min, max, mean, _std) = RasterSampler::statistics(&g);
        assert!((min - 0.0).abs() < 1e-9);
        assert!((max - 4.0).abs() < 1e-9);
        assert!((mean - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_statistics_excludes_no_data() {
        let extent = GridExtent::new(0.0, 2.0, 0.0, 1.0);
        let mut g = RasterGrid::new(3, 1, extent);
        g.no_data = -9999.0;
        g.set_value(0, 0, 1.0);
        g.set_value(1, 0, -9999.0); // no data
        g.set_value(2, 0, 3.0);
        let (min, max, mean, _) = RasterSampler::statistics(&g);
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 3.0).abs() < 1e-9);
        assert!((mean - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_statistics_all_no_data_returns_nan() {
        let g = RasterGrid::new(2, 2, GridExtent::new(0.0, 1.0, 0.0, 1.0));
        let (min, max, mean, std_dev) = RasterSampler::statistics(&g);
        assert!(min.is_nan() && max.is_nan() && mean.is_nan() && std_dev.is_nan());
    }

    // ── Resample ──────────────────────────────────────────────

    #[test]
    fn test_resample_shape() {
        let g = make_3x3();
        let resampled = RasterSampler::resample(&g, 5, 5, InterpolationMethod::Bilinear);
        assert_eq!(resampled.width, 5);
        assert_eq!(resampled.height, 5);
    }

    #[test]
    fn test_resample_uniform_preserves_value() {
        let g = make_uniform(2.0);
        let resampled = RasterSampler::resample(&g, 3, 3, InterpolationMethod::NearestNeighbor);
        let (min, max, mean, _) = RasterSampler::statistics(&resampled);
        assert!((min - 2.0).abs() < 1e-6);
        assert!((max - 2.0).abs() < 1e-6);
        assert!((mean - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_resample_upscale() {
        let g = make_3x3();
        let large = RasterSampler::resample(&g, 9, 9, InterpolationMethod::Bilinear);
        assert_eq!(large.width, 9);
        assert_eq!(large.height, 9);
    }

    // ── sample_line ───────────────────────────────────────────

    #[test]
    fn test_sample_line_count() {
        let g = make_uniform(1.0);
        let samples = RasterSampler::sample_line(
            &g,
            0.0,
            0.0,
            4.0,
            4.0,
            10,
            InterpolationMethod::NearestNeighbor,
        );
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_sample_line_zero_samples() {
        let g = make_3x3();
        let samples = RasterSampler::sample_line(
            &g,
            0.0,
            0.0,
            2.0,
            2.0,
            0,
            InterpolationMethod::NearestNeighbor,
        );
        assert!(samples.is_empty());
    }

    #[test]
    fn test_sample_line_single_sample_start() {
        let g = make_uniform(5.0);
        let samples =
            RasterSampler::sample_line(&g, 0.0, 4.0, 4.0, 0.0, 1, InterpolationMethod::Bilinear);
        assert_eq!(samples.len(), 1);
        assert!((samples[0].x - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_sample_line_endpoints() {
        let g = make_uniform(3.0);
        let samples = RasterSampler::sample_line(
            &g,
            0.0,
            4.0,
            4.0,
            0.0,
            5,
            InterpolationMethod::NearestNeighbor,
        );
        assert_eq!(samples.len(), 5);
        // First sample at start
        assert!((samples[0].x - 0.0).abs() < 1e-9);
        // Last sample at end
        assert!((samples[4].x - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_sample_line_uniform_values() {
        let g = make_uniform(7.0);
        let samples =
            RasterSampler::sample_line(&g, 1.0, 1.0, 3.0, 3.0, 5, InterpolationMethod::Bilinear);
        for s in &samples {
            if !s.is_no_data {
                assert!((s.value - 7.0).abs() < 1e-6);
            }
        }
    }

    // ── SampleResult fields ────────────────────────────────────

    #[test]
    fn test_sample_result_xy_preserved() {
        let g = make_uniform(1.0);
        let s = RasterSampler::sample_nearest(&g, 1.5, 2.5);
        assert!((s.x - 1.5).abs() < 1e-9);
        assert!((s.y - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_sample_result_not_no_data_for_valid_point() {
        let g = make_uniform(10.0);
        let s = RasterSampler::sample_bilinear(&g, 2.0, 2.0);
        assert!(!s.is_no_data);
    }
}
