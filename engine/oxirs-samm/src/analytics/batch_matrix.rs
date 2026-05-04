//! Batch correlation matrix computation.
//!
//! [`BatchCorrelationMatrix`] computes a symmetric Pearson correlation matrix
//! over a collection of feature columns.  The API mirrors the pattern used in
//! `modelanalytics_compute_property_correlations_group` and
//! `modelanalytics_compute_partial_correlations_group`.
//!
//! # Input orientation
//!
//! `samples` is a slice of *column* slices: each `&[f64]` represents one
//! feature (variable) with `n` observations.  All columns must have the same
//! length; passing ragged slices returns
//! [`BatchCorrelationError::RaggedSamples`].
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_samm::analytics::batch_matrix::BatchCorrelationMatrix;
//!
//! // Two features, 5 observations each
//! let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];  // perfectly correlated
//! let samples: Vec<&[f64]> = vec![x.as_slice(), y.as_slice()];
//!
//! let matrix = BatchCorrelationMatrix::compute(&samples, None).unwrap();
//! assert_eq!(matrix.matrix[[0, 0]], 1.0);
//! assert!((matrix.matrix[[0, 1]] - 1.0).abs() < 1e-9);
//! ```

use scirs2_core::ndarray_ext::Array2;
use scirs2_stats::{CorrelationBuilder, CorrelationMethod};
use thiserror::Error;

/// Errors that can arise during batch correlation matrix computation.
#[derive(Debug, Error)]
pub enum BatchCorrelationError {
    /// All columns must have the same number of observations.
    #[error("ragged samples: columns have different lengths (expected {expected}, got {got} in column {col})")]
    RaggedSamples {
        /// Expected column length.
        expected: usize,
        /// Actual length of the offending column.
        got: usize,
        /// Zero-based index of the offending column.
        col: usize,
    },

    /// At least two features are required to build a correlation matrix.
    #[error("need at least 2 features, got {0}")]
    TooFewFeatures(usize),

    /// Each feature column must have at least two observations.
    #[error("need at least 2 observations, got {0}")]
    TooFewObservations(usize),

    /// The control variable index is out of range.
    #[error("control variable index {index} is out of range for {feature_count} features")]
    ControlIndexOutOfRange {
        /// The index that was provided.
        index: usize,
        /// The number of features in the matrix.
        feature_count: usize,
    },

    /// An internal computation error occurred.
    #[error("computation error: {0}")]
    ComputationError(String),
}

/// A symmetric Pearson correlation matrix computed from raw feature columns.
#[derive(Debug)]
pub struct BatchCorrelationMatrix {
    /// Symmetric `n×n` correlation matrix.  Diagonal entries are `1.0`.
    pub matrix: Array2<f64>,

    /// Labels for each feature (column).  Defaults to `"feature_0"`,
    /// `"feature_1"`, etc. when no labels are supplied.
    pub feature_labels: Vec<String>,

    /// Pairs `(i, j, r)` where `i < j` and `|r| > 0.3`.
    pub significant_pairs: Vec<(usize, usize, f64)>,
}

impl BatchCorrelationMatrix {
    /// Compute the Pearson correlation matrix for the given feature columns.
    ///
    /// # Arguments
    ///
    /// * `samples` – slice of column slices; each inner slice is one feature.
    /// * `labels`  – optional feature names; must match `samples.len()` when
    ///   provided.
    ///
    /// # Errors
    ///
    /// Returns [`BatchCorrelationError`] when the input is invalid.
    pub fn compute(
        samples: &[&[f64]],
        labels: Option<&[&str]>,
    ) -> Result<Self, BatchCorrelationError> {
        let n = samples.len();
        if n < 2 {
            return Err(BatchCorrelationError::TooFewFeatures(n));
        }

        let obs = samples[0].len();
        if obs < 2 {
            return Err(BatchCorrelationError::TooFewObservations(obs));
        }

        // Validate all columns have the same length.
        for (col, col_data) in samples.iter().enumerate().skip(1) {
            if col_data.len() != obs {
                return Err(BatchCorrelationError::RaggedSamples {
                    expected: obs,
                    got: col_data.len(),
                    col,
                });
            }
        }

        // Build feature labels.
        let feature_labels: Vec<String> = match labels {
            Some(lbs) => lbs.iter().map(|s| s.to_string()).collect(),
            None => (0..n).map(|i| format!("feature_{}", i)).collect(),
        };

        // Compute the n×n correlation matrix.
        let mut raw = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    raw[i][j] = 1.0;
                    continue;
                }
                if i > j {
                    raw[i][j] = raw[j][i];
                    continue;
                }
                raw[i][j] = pearson_correlation(samples[i], samples[j]);
            }
        }

        // Convert to ndarray Array2.
        let flat: Vec<f64> = (0..n).flat_map(|r| raw[r].iter().copied()).collect();
        let matrix = Array2::from_shape_vec((n, n), flat)
            .map_err(|e| BatchCorrelationError::ComputationError(e.to_string()))?;

        // Collect significant pairs.
        let significant_pairs = Self::collect_significant_pairs(&raw);

        Ok(Self {
            matrix,
            feature_labels,
            significant_pairs,
        })
    }

    /// Compute the partial correlation matrix controlling for a single variable.
    ///
    /// The partial correlation between features `i` and `j` controlling for
    /// feature `control_idx` is:
    ///
    /// ```text
    /// r(i,j | z) = (r(i,j) - r(i,z)*r(j,z))
    ///              / sqrt((1 - r(i,z)²) * (1 - r(j,z)²))
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BatchCorrelationError`] when the input is invalid or
    /// `control_idx` is out of range.
    pub fn partial_correlation_matrix(
        samples: &[&[f64]],
        control_idx: usize,
    ) -> Result<Self, BatchCorrelationError> {
        let n = samples.len();
        if n < 2 {
            return Err(BatchCorrelationError::TooFewFeatures(n));
        }
        if control_idx >= n {
            return Err(BatchCorrelationError::ControlIndexOutOfRange {
                index: control_idx,
                feature_count: n,
            });
        }
        let obs = samples[0].len();
        if obs < 2 {
            return Err(BatchCorrelationError::TooFewObservations(obs));
        }
        for (col, col_data) in samples.iter().enumerate().skip(1) {
            if col_data.len() != obs {
                return Err(BatchCorrelationError::RaggedSamples {
                    expected: obs,
                    got: col_data.len(),
                    col,
                });
            }
        }

        // Build the full correlation matrix first.
        let mut full = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    full[i][j] = 1.0;
                } else if i > j {
                    full[i][j] = full[j][i];
                } else {
                    full[i][j] = pearson_correlation(samples[i], samples[j]);
                }
            }
        }

        let z = control_idx;

        // Compute partial correlations.
        let mut partial = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    partial[i][j] = 1.0;
                    continue;
                }
                if i > j {
                    partial[i][j] = partial[j][i];
                    continue;
                }
                let r_ij = full[i][j];
                let r_iz = full[i][z];
                let r_jz = full[j][z];
                let denom_sq = (1.0 - r_iz * r_iz) * (1.0 - r_jz * r_jz);
                if denom_sq <= 0.0 {
                    // Perfectly collinear with control — clamp to ±1.
                    partial[i][j] = if r_ij >= 0.0 { 1.0 } else { -1.0 };
                } else {
                    partial[i][j] = (r_ij - r_iz * r_jz) / denom_sq.sqrt();
                }
            }
        }

        let flat: Vec<f64> = (0..n).flat_map(|r| partial[r].iter().copied()).collect();
        let matrix = Array2::from_shape_vec((n, n), flat)
            .map_err(|e| BatchCorrelationError::ComputationError(e.to_string()))?;

        let feature_labels: Vec<String> = (0..n).map(|i| format!("feature_{}", i)).collect();
        let significant_pairs = Self::collect_significant_pairs(&partial);

        Ok(Self {
            matrix,
            feature_labels,
            significant_pairs,
        })
    }

    // ------------------------------------------------------------------ //
    //  Private helpers                                                     //
    // ------------------------------------------------------------------ //

    fn collect_significant_pairs(raw: &[Vec<f64>]) -> Vec<(usize, usize, f64)> {
        let n = raw.len();
        let mut pairs = Vec::new();
        for (i, row_i) in raw.iter().enumerate() {
            for (j, &r) in row_i
                .iter()
                .enumerate()
                .skip(i + 1)
                .take(n.saturating_sub(i + 1))
            {
                if r.abs() > 0.3 {
                    pairs.push((i, j, r));
                }
            }
        }
        pairs
    }
}

// ---------------------------------------------------------------------- //
//  Pearson correlation helper                                              //
// ---------------------------------------------------------------------- //

/// Compute the Pearson correlation coefficient between two equal-length
/// slices.  Returns `0.0` on any degenerate input (zero variance, etc.)
/// so that the caller can treat the result as a number without error
/// handling.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    // Prefer the scirs2_stats CorrelationBuilder when the series is long
    // enough for numeric stability, fall back to a direct formula otherwise.
    use scirs2_core::ndarray_ext::Array1;

    let xa = Array1::from_vec(x.to_vec());
    let ya = Array1::from_vec(y.to_vec());

    let result = CorrelationBuilder::new()
        .method(CorrelationMethod::Pearson)
        .compute(xa.view(), ya.view());

    match result {
        Ok(r) => {
            let v = r.value.correlation;
            if v.is_finite() {
                v.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        }
        Err(_) => {
            // Fall back to direct formula.
            pearson_direct(x, y)
        }
    }
}

/// Direct (two-pass) Pearson correlation formula used as a fallback.
fn pearson_direct(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0_f64;
    let mut var_x = 0.0_f64;
    let mut var_y = 0.0_f64;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < f64::EPSILON {
        0.0
    } else {
        (cov / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_column(data: &[f64]) -> Vec<f64> {
        data.to_vec()
    }

    #[test]
    fn test_pearson_direct_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_direct(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "expected 1.0, got {}", r);
    }

    #[test]
    fn test_pearson_direct_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        let r = pearson_direct(&x, &y);
        assert!((r + 1.0).abs() < 1e-10, "expected -1.0, got {}", r);
    }

    #[test]
    fn test_batch_correlation_basic() {
        let a = make_column(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let b = make_column(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        let c = make_column(&[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let samples: Vec<&[f64]> = vec![&a, &b, &c];
        let labels = ["a", "b", "c"];

        let mat = BatchCorrelationMatrix::compute(&samples, Some(&labels)).unwrap();

        // Diagonal must be 1.
        assert!((mat.matrix[[0, 0]] - 1.0).abs() < 1e-9);
        assert!((mat.matrix[[1, 1]] - 1.0).abs() < 1e-9);
        assert!((mat.matrix[[2, 2]] - 1.0).abs() < 1e-9);

        // Symmetry.
        assert!((mat.matrix[[0, 1]] - mat.matrix[[1, 0]]).abs() < 1e-9);
        assert!((mat.matrix[[0, 2]] - mat.matrix[[2, 0]]).abs() < 1e-9);

        // Feature labels.
        assert_eq!(mat.feature_labels[0], "a");
        assert_eq!(mat.feature_labels[1], "b");
    }

    #[test]
    fn test_batch_correlation_identity() {
        let col = make_column(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        // Both columns are identical → correlation must be 1.
        let samples: Vec<&[f64]> = vec![&col, &col];
        let mat = BatchCorrelationMatrix::compute(&samples, None).unwrap();
        assert!((mat.matrix[[0, 1]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_batch_correlation_negative() {
        let x = make_column(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y: Vec<f64> = x.iter().map(|v| -v).collect();
        let samples: Vec<&[f64]> = vec![&x, &y];
        let mat = BatchCorrelationMatrix::compute(&samples, None).unwrap();
        assert!(
            (mat.matrix[[0, 1]] + 1.0).abs() < 1e-9,
            "expected -1.0, got {}",
            mat.matrix[[0, 1]]
        );
    }

    #[test]
    fn test_batch_correlation_significant_pairs() {
        // Perfectly correlated pair → |r| = 1.0 > 0.3 → must appear.
        let x = make_column(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
        let samples: Vec<&[f64]> = vec![&x, &y];
        let mat = BatchCorrelationMatrix::compute(&samples, None).unwrap();
        assert!(
            !mat.significant_pairs.is_empty(),
            "expected at least one significant pair"
        );
        assert_eq!(mat.significant_pairs[0].0, 0);
        assert_eq!(mat.significant_pairs[0].1, 1);
        assert!((mat.significant_pairs[0].2 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_batch_correlation_too_few_features() {
        let col = make_column(&[1.0, 2.0, 3.0]);
        let samples: Vec<&[f64]> = vec![&col]; // only one feature
        let result = BatchCorrelationMatrix::compute(&samples, None);
        assert!(matches!(
            result,
            Err(BatchCorrelationError::TooFewFeatures(1))
        ));
    }

    #[test]
    fn test_batch_correlation_ragged() {
        let a = make_column(&[1.0, 2.0, 3.0]);
        let b = make_column(&[1.0, 2.0]); // shorter
        let samples: Vec<&[f64]> = vec![&a, &b];
        let result = BatchCorrelationMatrix::compute(&samples, None);
        assert!(matches!(
            result,
            Err(BatchCorrelationError::RaggedSamples { .. })
        ));
    }

    #[test]
    fn test_partial_correlation_basic() {
        // Features: x=1..10, y=2x, z=x+noise
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
        let z: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v + (i % 3) as f64 * 0.5)
            .collect();

        let samples: Vec<&[f64]> = vec![&x, &y, &z];
        let mat = BatchCorrelationMatrix::partial_correlation_matrix(&samples, 2).unwrap();

        // Diagonal must be 1.
        assert!((mat.matrix[[0, 0]] - 1.0).abs() < 1e-9);
        assert!((mat.matrix[[1, 1]] - 1.0).abs() < 1e-9);

        // Symmetry.
        assert!((mat.matrix[[0, 1]] - mat.matrix[[1, 0]]).abs() < 1e-9);
    }

    #[test]
    fn test_partial_correlation_control_out_of_range() {
        let x = make_column(&[1.0, 2.0, 3.0]);
        let y = make_column(&[2.0, 4.0, 6.0]);
        let samples: Vec<&[f64]> = vec![&x, &y];
        let result = BatchCorrelationMatrix::partial_correlation_matrix(&samples, 5);
        assert!(matches!(
            result,
            Err(BatchCorrelationError::ControlIndexOutOfRange { .. })
        ));
    }
}
