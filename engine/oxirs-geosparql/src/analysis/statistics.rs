//! Spatial Statistics
//!
//! Implements spatial autocorrelation and hot spot analysis using
//! Moran's I and Getis-Ord G* statistics.

use crate::error::{GeoSparqlError, Result};
use geo_types::Point;
use scirs2_core::ndarray_ext::Array2;

/// Spatial weights matrix type
#[derive(Debug, Clone, Copy)]
pub enum WeightsMatrixType {
    /// Inverse distance weighting
    InverseDistance {
        /// Power parameter for inverse distance weighting
        power: f64,
    },
    /// Binary weights (1 if within threshold, 0 otherwise)
    BinaryThreshold {
        /// Distance threshold for binary weights
        threshold: f64,
    },
    /// K-nearest neighbors
    KNearestNeighbors {
        /// Number of nearest neighbors
        k: usize,
    },
    /// Distance band (1 if within band, 0 otherwise)
    DistanceBand {
        /// Minimum distance for distance band
        min_dist: f64,
        /// Maximum distance for distance band
        max_dist: f64,
    },
}

/// Spatial autocorrelation result (Moran's I)
#[derive(Debug, Clone)]
pub struct SpatialAutocorrelation {
    /// Moran's I statistic
    pub morans_i: f64,
    /// Expected value under null hypothesis
    pub expected_i: f64,
    /// Variance of I
    pub variance: f64,
    /// Z-score
    pub z_score: f64,
    /// P-value (two-tailed)
    pub p_value: f64,
    /// Interpretation
    pub interpretation: AutocorrelationInterpretation,
}

/// Interpretation of spatial autocorrelation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutocorrelationInterpretation {
    /// Significant positive spatial autocorrelation (clustering)
    PositiveClustering,
    /// Significant negative spatial autocorrelation (dispersion)
    NegativeDispersion,
    /// No significant spatial autocorrelation (random)
    Random,
}

/// Hot spot analysis result (Getis-Ord G*)
#[derive(Debug, Clone)]
pub struct HotSpotResult {
    /// Gi* statistic for each location
    pub gi_star: Vec<f64>,
    /// Z-scores for each location
    pub z_scores: Vec<f64>,
    /// P-values for each location
    pub p_values: Vec<f64>,
    /// Hot spot classification for each location
    pub classifications: Vec<HotSpotClassification>,
}

/// Hot spot classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotSpotClassification {
    /// Hot spot (high values clustered) - 99% confidence
    HotSpot99,
    /// Hot spot - 95% confidence
    HotSpot95,
    /// Hot spot - 90% confidence
    HotSpot90,
    /// Cold spot (low values clustered) - 99% confidence
    ColdSpot99,
    /// Cold spot - 95% confidence
    ColdSpot95,
    /// Cold spot - 90% confidence
    ColdSpot90,
    /// Not significant
    NotSignificant,
}

/// Compute Moran's I spatial autocorrelation statistic
///
/// Measures the degree of spatial autocorrelation in a dataset.
/// Values range from -1 (perfect dispersion) to +1 (perfect clustering),
/// with 0 indicating random spatial pattern.
///
/// # Arguments
/// * `points` - Spatial locations
/// * `values` - Attribute values at each location
/// * `weights_type` - Type of spatial weights matrix
///
/// # Returns
/// `SpatialAutocorrelation` with Moran's I and significance test
///
/// # Example
/// ```
/// use oxirs_geosparql::analysis::{morans_i, WeightsMatrixType};
/// use geo_types::Point;
///
/// let points = vec![
///     Point::new(0.0, 0.0),
///     Point::new(0.1, 0.1),
///     Point::new(1.0, 1.0),
///     Point::new(1.1, 1.1),
/// ];
/// let values = vec![10.0, 12.0, 50.0, 48.0];
///
/// let weights_type = WeightsMatrixType::InverseDistance { power: 1.0 };
/// let result = morans_i(&points, &values, weights_type).unwrap();
///
/// // Should show positive spatial autocorrelation
/// assert!(result.morans_i > 0.0);
/// ```
pub fn morans_i(
    points: &[Point<f64>],
    values: &[f64],
    weights_type: WeightsMatrixType,
) -> Result<SpatialAutocorrelation> {
    if points.len() != values.len() {
        return Err(GeoSparqlError::InvalidInput(
            "Number of points and values must match".to_string(),
        ));
    }

    if points.len() < 3 {
        return Err(GeoSparqlError::InvalidInput(
            "Need at least 3 points for spatial autocorrelation".to_string(),
        ));
    }

    let n = points.len();

    // Build spatial weights matrix
    let weights = build_weights_matrix(points, weights_type)?;

    // Compute mean and deviations
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let deviations: Vec<f64> = values.iter().map(|v| v - mean).collect();

    // Compute Moran's I numerator
    let mut numerator = 0.0;
    let mut w_sum = 0.0;

    for i in 0..n {
        for j in 0..n {
            if i != j {
                numerator += weights[[i, j]] * deviations[i] * deviations[j];
                w_sum += weights[[i, j]];
            }
        }
    }

    // Compute denominator
    let denominator: f64 = deviations.iter().map(|d| d * d).sum();

    // Moran's I
    let morans_i = (n as f64 / w_sum) * (numerator / denominator);

    // Expected value under null hypothesis
    let expected_i = -1.0 / (n as f64 - 1.0);

    // Compute variance using randomization assumption
    let variance = compute_morans_i_variance(n, &weights, w_sum);

    // Z-score
    let z_score = (morans_i - expected_i) / variance.sqrt();

    // P-value (two-tailed)
    let p_value = 2.0 * (1.0 - standard_normal_cdf(z_score.abs()));

    // Interpretation
    let interpretation = if p_value < 0.05 {
        if morans_i > expected_i {
            AutocorrelationInterpretation::PositiveClustering
        } else {
            AutocorrelationInterpretation::NegativeDispersion
        }
    } else {
        AutocorrelationInterpretation::Random
    };

    Ok(SpatialAutocorrelation {
        morans_i,
        expected_i,
        variance,
        z_score,
        p_value,
        interpretation,
    })
}

/// Compute Getis-Ord Gi* hot spot analysis
///
/// Identifies statistically significant hot spots (clusters of high values)
/// and cold spots (clusters of low values).
///
/// # Arguments
/// * `points` - Spatial locations
/// * `values` - Attribute values at each location
/// * `weights_type` - Type of spatial weights matrix
///
/// # Returns
/// `HotSpotResult` with Gi* statistics and classifications
///
/// # Example
/// ```
/// use oxirs_geosparql::analysis::{getis_ord_gi_star, WeightsMatrixType};
/// use geo_types::Point;
///
/// let points = vec![
///     Point::new(0.0, 0.0),
///     Point::new(0.1, 0.1),
///     Point::new(0.2, 0.0),
///     Point::new(5.0, 5.0),
/// ];
/// let values = vec![100.0, 95.0, 105.0, 10.0];
///
/// let weights_type = WeightsMatrixType::InverseDistance { power: 1.0 };
/// let result = getis_ord_gi_star(&points, &values, weights_type).unwrap();
///
/// // First three points should be hot spots
/// assert!(result.z_scores[0] > 0.0);
/// assert!(result.z_scores[1] > 0.0);
/// assert!(result.z_scores[2] > 0.0);
/// ```
pub fn getis_ord_gi_star(
    points: &[Point<f64>],
    values: &[f64],
    weights_type: WeightsMatrixType,
) -> Result<HotSpotResult> {
    if points.len() != values.len() {
        return Err(GeoSparqlError::InvalidInput(
            "Number of points and values must match".to_string(),
        ));
    }

    if points.len() < 3 {
        return Err(GeoSparqlError::InvalidInput(
            "Need at least 3 points for hot spot analysis".to_string(),
        ));
    }

    let n = points.len();

    // Build spatial weights matrix
    let mut weights = build_weights_matrix(points, weights_type)?;

    // For Gi*, include self in weights (set diagonal to 1)
    for i in 0..n {
        weights[[i, i]] = 1.0;
    }

    // Compute global mean and variance
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

    let mut gi_star = Vec::with_capacity(n);
    let mut z_scores = Vec::with_capacity(n);
    let mut p_values = Vec::with_capacity(n);
    let mut classifications = Vec::with_capacity(n);

    for i in 0..n {
        // Compute weighted sum and sum of weights
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut weight_sq_sum = 0.0;

        for j in 0..n {
            weighted_sum += weights[[i, j]] * values[j];
            weight_sum += weights[[i, j]];
            weight_sq_sum += weights[[i, j]] * weights[[i, j]];
        }

        // Gi* statistic
        let gi = weighted_sum;
        gi_star.push(gi);

        // Expected value and variance
        let expected_gi = weight_sum * mean;
        let var_gi =
            ((n as f64 * weight_sq_sum - weight_sum * weight_sum) / (n as f64 - 1.0)) * variance;

        // Z-score
        let z = if var_gi > 1e-10 {
            (gi - expected_gi) / var_gi.sqrt()
        } else {
            0.0
        };
        z_scores.push(z);

        // P-value (two-tailed)
        let p = 2.0 * (1.0 - standard_normal_cdf(z.abs()));
        p_values.push(p);

        // Classification
        let classification = if z > 2.576 {
            HotSpotClassification::HotSpot99
        } else if z > 1.96 {
            HotSpotClassification::HotSpot95
        } else if z > 1.645 {
            HotSpotClassification::HotSpot90
        } else if z < -2.576 {
            HotSpotClassification::ColdSpot99
        } else if z < -1.96 {
            HotSpotClassification::ColdSpot95
        } else if z < -1.645 {
            HotSpotClassification::ColdSpot90
        } else {
            HotSpotClassification::NotSignificant
        };
        classifications.push(classification);
    }

    Ok(HotSpotResult {
        gi_star,
        z_scores,
        p_values,
        classifications,
    })
}

/// Build spatial weights matrix
fn build_weights_matrix(
    points: &[Point<f64>],
    weights_type: WeightsMatrixType,
) -> Result<Array2<f64>> {
    let n = points.len();
    let mut weights = Array2::zeros((n, n));

    match weights_type {
        WeightsMatrixType::InverseDistance { power } => {
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let dist = euclidean_distance(&points[i], &points[j]);
                        weights[[i, j]] = if dist > 1e-10 {
                            1.0 / dist.powf(power)
                        } else {
                            0.0
                        };
                    }
                }
            }

            // Row-standardize
            for i in 0..n {
                let row_sum: f64 = weights.row(i).iter().sum();
                if row_sum > 1e-10 {
                    for j in 0..n {
                        weights[[i, j]] /= row_sum;
                    }
                }
            }
        }
        WeightsMatrixType::BinaryThreshold { threshold } => {
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let dist = euclidean_distance(&points[i], &points[j]);
                        weights[[i, j]] = if dist <= threshold { 1.0 } else { 0.0 };
                    }
                }
            }

            // Row-standardize
            for i in 0..n {
                let row_sum: f64 = weights.row(i).iter().sum();
                if row_sum > 1e-10 {
                    for j in 0..n {
                        weights[[i, j]] /= row_sum;
                    }
                }
            }
        }
        WeightsMatrixType::KNearestNeighbors { k } => {
            if k >= n {
                return Err(GeoSparqlError::InvalidInput(
                    "k must be less than number of points".to_string(),
                ));
            }

            for i in 0..n {
                // Compute distances to all other points
                let mut distances: Vec<(usize, f64)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j, euclidean_distance(&points[i], &points[j])))
                    .collect();

                // Sort by distance
                distances
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                // Set weights for k nearest neighbors
                for &(j, _) in distances.iter().take(k) {
                    weights[[i, j]] = 1.0 / k as f64;
                }
            }
        }
        WeightsMatrixType::DistanceBand { min_dist, max_dist } => {
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let dist = euclidean_distance(&points[i], &points[j]);
                        weights[[i, j]] = if dist >= min_dist && dist <= max_dist {
                            1.0
                        } else {
                            0.0
                        };
                    }
                }
            }

            // Row-standardize
            for i in 0..n {
                let row_sum: f64 = weights.row(i).iter().sum();
                if row_sum > 1e-10 {
                    for j in 0..n {
                        weights[[i, j]] /= row_sum;
                    }
                }
            }
        }
    }

    Ok(weights)
}

/// Compute variance of Moran's I under randomization assumption
fn compute_morans_i_variance(n: usize, weights: &Array2<f64>, w_sum: f64) -> f64 {
    let n_f = n as f64;

    // Compute S1
    let mut s1 = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let wij = weights[[i, j]];
                let wji = weights[[j, i]];
                s1 += (wij + wji) * (wij + wji);
            }
        }
    }
    s1 *= 0.5;

    // Compute S2
    let mut s2 = 0.0;
    for i in 0..n {
        let mut sum_i = 0.0;
        let mut sum_j = 0.0;
        for k in 0..n {
            if k != i {
                sum_i += weights[[i, k]];
                sum_j += weights[[k, i]];
            }
        }
        s2 += (sum_i + sum_j) * (sum_i + sum_j);
    }

    // Variance formula
    let a = n_f * ((n_f * n_f - 3.0 * n_f + 3.0) * s1 - n_f * s2 + 3.0 * w_sum * w_sum);
    let b = (n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0) * w_sum * w_sum;
    let c = (n_f * s1 - 2.0 * n_f * s2 + 6.0 * w_sum * w_sum) / ((n_f - 1.0) * (n_f - 2.0));
    let d = 1.0 / (n_f - 1.0);

    a / b - (c - d) * (c - d)
}

/// Compute Euclidean distance between two points
fn euclidean_distance(p1: &Point<f64>, p2: &Point<f64>) -> f64 {
    let dx = p1.x() - p2.x();
    let dy = p1.y() - p2.y();
    (dx * dx + dy * dy).sqrt()
}

/// Standard normal CDF (cumulative distribution function)
fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / 2.0_f64.sqrt()))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_morans_i_positive_autocorrelation() {
        // Two well-separated clusters with highly distinct values
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(0.1, 0.0),
            Point::new(0.0, 0.1),
            Point::new(0.1, 0.1),
            Point::new(10.0, 10.0),
            Point::new(10.1, 10.0),
            Point::new(10.0, 10.1),
            Point::new(10.1, 10.1),
        ];
        let values = vec![1.0, 2.0, 1.5, 2.5, 100.0, 98.0, 102.0, 99.0];

        let weights_type = WeightsMatrixType::InverseDistance { power: 2.0 };
        let result = morans_i(&points, &values, weights_type).unwrap();

        // Should show positive spatial autocorrelation (Moran's I > expected)
        assert!(
            result.morans_i > result.expected_i,
            "Moran's I should be greater than expected for clustered data"
        );
    }

    #[test]
    fn test_morans_i_random() {
        // Random pattern
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(1.0, 1.0),
        ];
        let values = vec![10.0, 20.0, 15.0, 25.0];

        let weights_type = WeightsMatrixType::InverseDistance { power: 1.0 };
        let result = morans_i(&points, &values, weights_type).unwrap();

        // P-value should be high (not significant)
        assert!(
            result.p_value > 0.05 || result.interpretation == AutocorrelationInterpretation::Random
        );
    }

    #[test]
    fn test_getis_ord_hot_spots() {
        let points = vec![
            // Hot spot cluster
            Point::new(0.0, 0.0),
            Point::new(0.1, 0.1),
            Point::new(0.2, 0.0),
            // Cold spot
            Point::new(5.0, 5.0),
        ];
        let values = vec![100.0, 95.0, 105.0, 10.0];

        let weights_type = WeightsMatrixType::InverseDistance { power: 1.0 };
        let result = getis_ord_gi_star(&points, &values, weights_type).unwrap();

        // First three should have positive z-scores (hot spots)
        assert!(result.z_scores[0] > 0.0);
        assert!(result.z_scores[1] > 0.0);
        assert!(result.z_scores[2] > 0.0);

        // Last should have negative z-score (cold spot)
        assert!(result.z_scores[3] < 0.0);
    }

    #[test]
    fn test_weights_matrix_inverse_distance() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
        ];

        let weights_type = WeightsMatrixType::InverseDistance { power: 1.0 };
        let weights = build_weights_matrix(&points, weights_type).unwrap();

        // Matrix should be row-standardized
        for i in 0..3 {
            let row_sum: f64 = weights.row(i).iter().sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }

        // Diagonal should be zero (before Gi* modification)
        for i in 0..3 {
            assert_eq!(weights[[i, i]], 0.0);
        }
    }

    #[test]
    fn test_weights_matrix_binary_threshold() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(0.5, 0.0),
            Point::new(2.0, 0.0),
        ];

        let weights_type = WeightsMatrixType::BinaryThreshold { threshold: 1.0 };
        let weights = build_weights_matrix(&points, weights_type).unwrap();

        // Points 0 and 1 are within threshold
        assert!(weights[[0, 1]] > 0.0);
        // Points 0 and 2 are not
        assert_eq!(weights[[0, 2]], 0.0);
    }

    #[test]
    fn test_weights_matrix_k_nearest() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(3.0, 0.0),
        ];

        let weights_type = WeightsMatrixType::KNearestNeighbors { k: 2 };
        let weights = build_weights_matrix(&points, weights_type).unwrap();

        // Each row should have exactly k non-zero entries
        for i in 0..4 {
            let non_zero_count = weights.row(i).iter().filter(|&&w| w > 1e-10).count();
            assert_eq!(non_zero_count, 2);
        }
    }

    #[test]
    fn test_standard_normal_cdf() {
        // Test known values
        assert_relative_eq!(standard_normal_cdf(0.0), 0.5, epsilon = 1e-6);
        assert_relative_eq!(standard_normal_cdf(1.96), 0.975, epsilon = 1e-3);
        assert_relative_eq!(standard_normal_cdf(-1.96), 0.025, epsilon = 1e-3);
    }

    #[test]
    fn test_morans_i_too_few_points() {
        let points = vec![Point::new(0.0, 0.0), Point::new(1.0, 0.0)];
        let values = vec![10.0, 20.0];

        let weights_type = WeightsMatrixType::InverseDistance { power: 1.0 };
        let result = morans_i(&points, &values, weights_type);

        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_points_values() {
        let points = vec![Point::new(0.0, 0.0), Point::new(1.0, 0.0)];
        let values = vec![10.0];

        let weights_type = WeightsMatrixType::InverseDistance { power: 1.0 };
        let result = morans_i(&points, &values, weights_type);

        assert!(result.is_err());
    }
}
