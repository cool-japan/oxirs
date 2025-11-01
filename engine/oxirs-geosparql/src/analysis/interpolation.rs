//! Spatial Interpolation
//!
//! Implements IDW (Inverse Distance Weighting) and Kriging interpolation
//! for estimating values at unknown locations from sample points.

use crate::error::{GeoSparqlError, Result};
use geo_types::Point;
use scirs2_core::ndarray_ext::{Array1, Array2};

/// Simple Gaussian elimination solver with partial pivoting
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(GeoSparqlError::ComputationError(
            "Matrix must be square".to_string(),
        ));
    }
    if b.len() != n {
        return Err(GeoSparqlError::ComputationError(
            "RHS vector size mismatch".to_string(),
        ));
    }

    // Create augmented matrix
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = aug[[k, k]].abs();
        for i in (k + 1)..n {
            let val = aug[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Check for singularity
        if max_val < 1e-14 {
            return Err(GeoSparqlError::ComputationError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows
        if max_row != k {
            for j in 0..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Eliminate column
        for i in (k + 1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..=n {
                aug[[i, j]] -= factor * aug[[k, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Interpolation method
#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    /// Inverse Distance Weighting
    Idw {
        /// Power parameter for distance weighting
        power: f64,
    },
    /// Ordinary Kriging
    OrdinaryKriging {
        /// Range parameter of the semivariogram
        range: f64,
        /// Sill parameter of the semivariogram
        sill: f64,
        /// Nugget parameter (measurement error)
        nugget: f64,
    },
    /// Universal Kriging (with trend)
    UniversalKriging {
        /// Range parameter of the semivariogram
        range: f64,
        /// Sill parameter of the semivariogram
        sill: f64,
        /// Nugget parameter (measurement error)
        nugget: f64,
    },
}

/// Sample point with a value
#[derive(Debug, Clone)]
pub struct SamplePoint {
    /// Location of the sample
    pub location: Point<f64>,
    /// Measured value at this location
    pub value: f64,
}

impl SamplePoint {
    /// Create a new sample point
    pub fn new(x: f64, y: f64, value: f64) -> Self {
        Self {
            location: Point::new(x, y),
            value,
        }
    }
}

/// Result of interpolation
#[derive(Debug, Clone)]
pub struct InterpolationResult {
    /// Interpolated value
    pub value: f64,
    /// Estimation variance (for kriging)
    pub variance: Option<f64>,
}

/// Inverse Distance Weighting (IDW) interpolation
///
/// Estimates values using weighted average based on inverse distance.
/// Closer points have higher influence on the interpolated value.
///
/// # Arguments
/// * `samples` - Known sample points with values
/// * `target` - Location where value should be interpolated
/// * `power` - Exponent for distance weighting (typically 2.0)
///
/// # Returns
/// Interpolated value at target location
///
/// # Example
/// ```
/// use oxirs_geosparql::analysis::{idw_interpolation, SamplePoint};
/// use geo_types::Point;
///
/// let samples = vec![
///     SamplePoint::new(0.0, 0.0, 10.0),
///     SamplePoint::new(1.0, 0.0, 20.0),
///     SamplePoint::new(0.0, 1.0, 15.0),
/// ];
///
/// let target = Point::new(0.5, 0.5);
/// let result = idw_interpolation(&samples, &target, 2.0).unwrap();
///
/// // Value should be influenced by all three samples
/// assert!(result.value > 10.0 && result.value < 20.0);
/// ```
pub fn idw_interpolation(
    samples: &[SamplePoint],
    target: &Point<f64>,
    power: f64,
) -> Result<InterpolationResult> {
    if samples.is_empty() {
        return Err(GeoSparqlError::InvalidInput(
            "Need at least one sample point for interpolation".to_string(),
        ));
    }

    if power <= 0.0 {
        return Err(GeoSparqlError::InvalidInput(
            "Power parameter must be positive".to_string(),
        ));
    }

    // Check if target coincides with a sample point
    for sample in samples {
        if is_same_location(&sample.location, target) {
            return Ok(InterpolationResult {
                value: sample.value,
                variance: None,
            });
        }
    }

    let mut weight_sum = 0.0;
    let mut weighted_value_sum = 0.0;

    for sample in samples {
        let dist = euclidean_distance(&sample.location, target);
        if dist < 1e-10 {
            // Target is on a sample point
            return Ok(InterpolationResult {
                value: sample.value,
                variance: None,
            });
        }

        let weight = 1.0 / dist.powf(power);
        weight_sum += weight;
        weighted_value_sum += weight * sample.value;
    }

    let value = weighted_value_sum / weight_sum;

    Ok(InterpolationResult {
        value,
        variance: None,
    })
}

/// Ordinary Kriging interpolation
///
/// Geostatistical interpolation method that provides best linear unbiased
/// prediction (BLUP) based on spatial correlation structure.
///
/// # Arguments
/// * `samples` - Known sample points with values
/// * `target` - Location where value should be interpolated
/// * `range` - Range parameter of variogram (correlation distance)
/// * `sill` - Sill parameter of variogram (total variance)
/// * `nugget` - Nugget parameter of variogram (measurement error variance)
///
/// # Returns
/// Interpolated value with estimation variance
///
/// # Example
/// ```
/// use oxirs_geosparql::analysis::{kriging_interpolation, SamplePoint};
/// use geo_types::Point;
///
/// let samples = vec![
///     SamplePoint::new(0.0, 0.0, 10.0),
///     SamplePoint::new(1.0, 0.0, 20.0),
///     SamplePoint::new(0.0, 1.0, 15.0),
///     SamplePoint::new(1.0, 1.0, 25.0),
/// ];
///
/// let target = Point::new(0.5, 0.5);
/// let result = kriging_interpolation(&samples, &target, 1.0, 10.0, 0.1).unwrap();
///
/// assert!(result.value > 10.0 && result.value < 25.0);
/// assert!(result.variance.is_some());
/// ```
pub fn kriging_interpolation(
    samples: &[SamplePoint],
    target: &Point<f64>,
    range: f64,
    sill: f64,
    nugget: f64,
) -> Result<InterpolationResult> {
    if samples.len() < 3 {
        return Err(GeoSparqlError::InvalidInput(
            "Need at least 3 sample points for kriging".to_string(),
        ));
    }

    if range <= 0.0 || sill < 0.0 || nugget < 0.0 {
        return Err(GeoSparqlError::InvalidInput(
            "Variogram parameters must be non-negative (range > 0)".to_string(),
        ));
    }

    let n = samples.len();

    // Build covariance matrix (kriging matrix)
    let mut cov_matrix = Array2::zeros((n + 1, n + 1));

    for i in 0..n {
        for j in 0..n {
            let dist = euclidean_distance(&samples[i].location, &samples[j].location);
            let gamma = variogram_spherical(dist, range, sill, nugget);
            cov_matrix[[i, j]] = sill - gamma; // Covariance = sill - gamma
        }
        cov_matrix[[i, n]] = 1.0; // Lagrange multiplier
        cov_matrix[[n, i]] = 1.0;
    }
    cov_matrix[[n, n]] = 0.0;

    // Build right-hand side vector
    let mut rhs = Array1::zeros(n + 1);
    for i in 0..n {
        let dist = euclidean_distance(&samples[i].location, target);
        let gamma = variogram_spherical(dist, range, sill, nugget);
        rhs[i] = sill - gamma;
    }
    rhs[n] = 1.0;

    // Solve kriging system
    let weights = solve_linear_system(&cov_matrix, &rhs).map_err(|e| {
        GeoSparqlError::ComputationError(format!("Failed to solve kriging system: {}", e))
    })?;

    // Compute interpolated value
    let mut value = 0.0;
    for i in 0..n {
        value += weights[i] * samples[i].value;
    }

    // Compute kriging variance
    let mut variance = sill;
    for i in 0..n {
        variance -= weights[i] * rhs[i];
    }
    variance -= weights[n]; // Lagrange multiplier contribution

    Ok(InterpolationResult {
        value,
        variance: Some(variance.max(0.0)),
    })
}

/// Spherical variogram model
///
/// γ(h) = nugget + sill * (1.5 * h/range - 0.5 * (h/range)³) for h < range
/// γ(h) = nugget + sill for h >= range
fn variogram_spherical(distance: f64, range: f64, sill: f64, nugget: f64) -> f64 {
    if distance < 1e-10 {
        return 0.0; // γ(0) = 0
    }

    if distance >= range {
        return nugget + sill;
    }

    let h_r = distance / range;
    nugget + sill * (1.5 * h_r - 0.5 * h_r.powi(3))
}

/// Exponential variogram model (alternative to spherical)
#[allow(dead_code)]
fn variogram_exponential(distance: f64, range: f64, sill: f64, nugget: f64) -> f64 {
    if distance < 1e-10 {
        return 0.0;
    }

    nugget + sill * (1.0 - (-3.0 * distance / range).exp())
}

/// Gaussian variogram model (alternative)
#[allow(dead_code)]
fn variogram_gaussian(distance: f64, range: f64, sill: f64, nugget: f64) -> f64 {
    if distance < 1e-10 {
        return 0.0;
    }

    nugget + sill * (1.0 - (-(distance / range).powi(2)).exp())
}

/// Compute Euclidean distance between two points
fn euclidean_distance(p1: &Point<f64>, p2: &Point<f64>) -> f64 {
    let dx = p1.x() - p2.x();
    let dy = p1.y() - p2.y();
    (dx * dx + dy * dy).sqrt()
}

/// Check if two locations are the same
fn is_same_location(p1: &Point<f64>, p2: &Point<f64>) -> bool {
    euclidean_distance(p1, p2) < 1e-10
}

/// Cross-validation for interpolation methods
///
/// Performs leave-one-out cross-validation to assess interpolation accuracy.
pub fn cross_validate_interpolation(
    samples: &[SamplePoint],
    method: InterpolationMethod,
) -> Result<CrossValidationResult> {
    if samples.len() < 4 {
        return Err(GeoSparqlError::InvalidInput(
            "Need at least 4 sample points for cross-validation".to_string(),
        ));
    }

    let mut errors = Vec::new();
    let mut predictions = Vec::new();
    let mut actuals = Vec::new();

    for i in 0..samples.len() {
        // Leave out sample i
        let training: Vec<SamplePoint> = samples
            .iter()
            .enumerate()
            .filter_map(|(j, s)| if j != i { Some(s.clone()) } else { None })
            .collect();

        // Predict at location i
        let result = match method {
            InterpolationMethod::Idw { power } => {
                idw_interpolation(&training, &samples[i].location, power)?
            }
            InterpolationMethod::OrdinaryKriging {
                range,
                sill,
                nugget,
            } => kriging_interpolation(&training, &samples[i].location, range, sill, nugget)?,
            InterpolationMethod::UniversalKriging { .. } => {
                // Simplified: use ordinary kriging for now
                return Err(GeoSparqlError::InvalidInput(
                    "Universal kriging not yet implemented".to_string(),
                ));
            }
        };

        let error = result.value - samples[i].value;
        errors.push(error);
        predictions.push(result.value);
        actuals.push(samples[i].value);
    }

    // Compute statistics
    let mae = errors.iter().map(|e| e.abs()).sum::<f64>() / errors.len() as f64;
    let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();

    let mean_actual = actuals.iter().sum::<f64>() / actuals.len() as f64;
    let ss_tot: f64 = actuals.iter().map(|a| (a - mean_actual).powi(2)).sum();
    let ss_res: f64 = errors.iter().map(|e| e.powi(2)).sum();
    let r_squared = 1.0 - ss_res / ss_tot;

    Ok(CrossValidationResult {
        mae,
        rmse,
        r_squared,
        errors,
        predictions,
        actuals,
    })
}

/// Result of cross-validation
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Mean Absolute Error
    pub mae: f64,
    /// Root Mean Square Error
    pub rmse: f64,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
    /// Individual prediction errors
    pub errors: Vec<f64>,
    /// Predicted values
    pub predictions: Vec<f64>,
    /// Actual values
    pub actuals: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_idw_interpolation_exact_at_sample() {
        let samples = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(1.0, 0.0, 20.0),
            SamplePoint::new(0.0, 1.0, 15.0),
        ];

        // Interpolate at a sample location
        let target = Point::new(0.0, 0.0);
        let result = idw_interpolation(&samples, &target, 2.0).unwrap();

        assert_relative_eq!(result.value, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_idw_interpolation_midpoint() {
        let samples = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(2.0, 0.0, 20.0),
        ];

        // Interpolate at midpoint
        let target = Point::new(1.0, 0.0);
        let result = idw_interpolation(&samples, &target, 2.0).unwrap();

        // Should be close to average
        assert_relative_eq!(result.value, 15.0, epsilon = 0.1);
    }

    #[test]
    fn test_idw_power_effect() {
        let samples = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(2.0, 0.0, 20.0),
        ];

        let target = Point::new(0.5, 0.0);

        // Higher power gives more weight to closer points
        let result1 = idw_interpolation(&samples, &target, 1.0).unwrap();
        let result2 = idw_interpolation(&samples, &target, 3.0).unwrap();

        // With higher power, should be closer to nearest point (10.0)
        assert!(result2.value < result1.value);
    }

    #[test]
    fn test_kriging_interpolation_basic() {
        let samples = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(1.0, 0.0, 20.0),
            SamplePoint::new(0.0, 1.0, 15.0),
            SamplePoint::new(1.0, 1.0, 25.0),
        ];

        let target = Point::new(0.5, 0.5);
        let result = kriging_interpolation(&samples, &target, 1.0, 10.0, 0.1).unwrap();

        // Should be reasonable interpolation
        assert!(result.value > 10.0 && result.value < 25.0);

        // Should have variance estimate
        assert!(result.variance.is_some());
        assert!(result.variance.unwrap() >= 0.0);
    }

    #[test]
    fn test_kriging_exact_at_sample() {
        let samples = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(1.0, 0.0, 20.0),
            SamplePoint::new(0.0, 1.0, 15.0),
        ];

        let target = Point::new(0.0, 0.0);
        let result = kriging_interpolation(&samples, &target, 1.0, 10.0, 0.0).unwrap();

        // Should be exact at sample location
        assert_relative_eq!(result.value, 10.0, epsilon = 1e-6);
    }

    #[test]
    fn test_variogram_spherical() {
        let range = 1.0;
        let sill = 10.0;
        let nugget = 1.0;

        // At distance 0
        let gamma0 = variogram_spherical(0.0, range, sill, nugget);
        assert_relative_eq!(gamma0, 0.0, epsilon = 1e-10);

        // At range distance
        let gamma_range = variogram_spherical(range, range, sill, nugget);
        assert_relative_eq!(gamma_range, nugget + sill, epsilon = 1e-6);

        // Beyond range
        let gamma_beyond = variogram_spherical(2.0 * range, range, sill, nugget);
        assert_relative_eq!(gamma_beyond, nugget + sill, epsilon = 1e-6);
    }

    #[test]
    fn test_cross_validation() {
        let samples = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(1.0, 0.0, 12.0),
            SamplePoint::new(0.0, 1.0, 14.0),
            SamplePoint::new(1.0, 1.0, 16.0),
            SamplePoint::new(0.5, 0.5, 13.0),
        ];

        let method = InterpolationMethod::Idw { power: 2.0 };
        let result = cross_validate_interpolation(&samples, method).unwrap();

        assert!(result.mae >= 0.0);
        assert!(result.rmse >= result.mae);
        assert!(result.r_squared <= 1.0);
        assert_eq!(result.errors.len(), samples.len());
    }

    #[test]
    fn test_idw_empty_samples() {
        let samples: Vec<SamplePoint> = vec![];
        let target = Point::new(0.0, 0.0);

        let result = idw_interpolation(&samples, &target, 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_kriging_too_few_samples() {
        let samples = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(1.0, 0.0, 20.0),
        ];

        let target = Point::new(0.5, 0.0);
        let result = kriging_interpolation(&samples, &target, 1.0, 10.0, 0.1);
        assert!(result.is_err());
    }
}
