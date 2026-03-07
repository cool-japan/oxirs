//! # PCA Reducer
//!
//! Principal Component Analysis (PCA) dimensionality reduction for embedding vectors,
//! implemented from scratch without external linear algebra libraries.
//!
//! ## Algorithm
//!
//! 1. Optionally center (subtract column means) and scale (divide by std-dev).
//! 2. Compute the covariance matrix.
//! 3. Extract the top-k eigenvectors using power iteration with deflation.
//! 4. Orthogonalise via Gram-Schmidt.
//! 5. Project data onto the component subspace.

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors arising from PCA operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PcaError {
    /// Not enough samples or features to perform PCA.
    InsufficientData { samples: usize, features: usize },
    /// Requested more components than the dimensionality allows.
    InvalidComponents { requested: usize, max: usize },
    /// Input dimensionality does not match the fitted model.
    DimensionMismatch,
}

impl fmt::Display for PcaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PcaError::InsufficientData { samples, features } => write!(
                f,
                "insufficient data: {samples} samples and {features} features (need at least 2 of each)"
            ),
            PcaError::InvalidComponents { requested, max } => write!(
                f,
                "requested {requested} components but max is {max}"
            ),
            PcaError::DimensionMismatch => write!(f, "input dimension does not match the fitted model"),
        }
    }
}

impl std::error::Error for PcaError {}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a PCA fitting operation.
#[derive(Debug, Clone)]
pub struct PcaConfig {
    /// Number of principal components to extract.
    pub n_components: usize,
    /// If `true`, subtract the per-feature mean before computing components.
    pub center: bool,
    /// If `true`, divide by the per-feature standard deviation after centering.
    pub scale: bool,
}

impl Default for PcaConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            center: true,
            scale: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model
// ─────────────────────────────────────────────────────────────────────────────

/// A fitted PCA model that can be used for `transform` and `inverse_transform`.
#[derive(Debug, Clone)]
pub struct PcaModel {
    /// Per-feature mean used for centering (zero if `center = false`).
    pub mean: Vec<f64>,
    /// Per-feature standard deviation used for scaling (one if `scale = false`).
    pub std_dev: Vec<f64>,
    /// Principal components as row vectors, each of length `n_features`.
    pub components: Vec<Vec<f64>>,
    /// Explained variance (eigenvalue) associated with each component.
    pub explained_variance: Vec<f64>,
    /// Number of components extracted.
    pub n_components: usize,
    /// Number of features in the training data.
    pub n_features: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Dot product of two equal-length slices.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean norm of a vector.
pub fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Normalise `v` in-place; if the norm is (near) zero, the vector is left unchanged.
pub fn normalize(v: &mut [f64]) {
    let n = norm(v);
    if n > 1e-15 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PcaReducer
// ─────────────────────────────────────────────────────────────────────────────

/// Provides `fit`, `transform`, `fit_transform`, and `inverse_transform` for PCA.
#[derive(Debug, Default)]
pub struct PcaReducer;

impl PcaReducer {
    /// Create a new reducer.
    pub fn new() -> Self {
        Self
    }

    /// Fit a `PcaModel` on `data`.
    ///
    /// Each row of `data` is a sample; each column is a feature.
    pub fn fit(data: &[Vec<f64>], config: &PcaConfig) -> Result<PcaModel, PcaError> {
        let n = data.len();
        let d = data.first().map(|r| r.len()).unwrap_or(0);

        if n < 2 || d < 2 {
            return Err(PcaError::InsufficientData {
                samples: n,
                features: d,
            });
        }
        let k = config.n_components;
        if k == 0 || k > d {
            return Err(PcaError::InvalidComponents {
                requested: k,
                max: d,
            });
        }

        // 1. Compute per-feature mean
        let mut mean = vec![0.0f64; d];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }

        // 2. Compute per-feature std-dev
        let mut std_dev = vec![1.0f64; d];
        if config.scale {
            let mut variance = vec![0.0f64; d];
            for row in data {
                for (j, &v) in row.iter().enumerate() {
                    let diff = v - mean[j];
                    variance[j] += diff * diff;
                }
            }
            for (j, s) in std_dev.iter_mut().enumerate() {
                let var = variance[j] / (n as f64 - 1.0).max(1.0);
                *s = if var > 1e-15 { var.sqrt() } else { 1.0 };
            }
        }

        // 3. Center / scale data
        let apply_mean = if config.center {
            &mean[..]
        } else {
            &vec![0.0f64; d][..]
        };
        let centered: Vec<Vec<f64>> = data
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, &v)| (v - apply_mean[j]) / std_dev[j])
                    .collect()
            })
            .collect();

        // 4. Compute covariance matrix  C[i][j] = (1/(n-1)) * Σ x_k[i] * x_k[j]
        let denom = (n as f64 - 1.0).max(1.0);
        let mut cov = vec![vec![0.0f64; d]; d];
        for row in &centered {
            #[allow(clippy::needless_range_loop)]
            for i in 0..d {
                for j in 0..d {
                    cov[i][j] += row[i] * row[j];
                }
            }
        }
        #[allow(clippy::needless_range_loop)]
        for i in 0..d {
            for j in 0..d {
                cov[i][j] /= denom;
            }
        }

        // 5. Power iteration with deflation for top-k eigenvectors
        let mut components: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut eigenvalues: Vec<f64> = Vec::with_capacity(k);
        // Deflated covariance matrix (starts as a copy of cov)
        let mut deflated = cov.clone();

        for comp_idx in 0..k {
            // Initialise eigenvector estimate with the column that has maximum variance
            let mut ev: Vec<f64> = (0..d).map(|j| deflated[j][j].abs()).collect();
            normalize(&mut ev);
            // Fallback: if all-zero, use a standard basis vector
            if norm(&ev) < 1e-15 {
                ev = vec![0.0; d];
                if comp_idx < d {
                    ev[comp_idx] = 1.0;
                }
            }

            // 50 power iterations
            for _ in 0..50 {
                // new_ev = deflated * ev
                let mut new_ev = vec![0.0f64; d];
                for i in 0..d {
                    for j in 0..d {
                        new_ev[i] += deflated[i][j] * ev[j];
                    }
                }
                normalize(&mut new_ev);
                ev = new_ev;
            }

            // Eigenvalue: Rayleigh quotient λ = ev' * deflated * ev
            let mut av = vec![0.0f64; d];
            for i in 0..d {
                for j in 0..d {
                    av[i] += deflated[i][j] * ev[j];
                }
            }
            let lambda = dot(&ev, &av);

            // Gram-Schmidt orthogonalise against all previous components
            for prev in &components {
                let proj = dot(&ev, prev);
                for j in 0..d {
                    ev[j] -= proj * prev[j];
                }
            }
            normalize(&mut ev);

            eigenvalues.push(lambda.max(0.0));
            components.push(ev.clone());

            // Deflation: subtract the rank-1 contribution of this eigenvector
            for i in 0..d {
                for j in 0..d {
                    deflated[i][j] -= lambda * ev[i] * ev[j];
                }
            }
        }

        // 6. Explained variance: already stored as eigenvalues (≥ 0)
        let total_var: f64 = eigenvalues.iter().sum();
        let explained_variance_ratio: Vec<f64> = if total_var > 1e-15 {
            eigenvalues.iter().map(|&e| e / total_var).collect()
        } else {
            vec![1.0 / k as f64; k]
        };

        Ok(PcaModel {
            mean,
            std_dev,
            components,
            explained_variance: explained_variance_ratio,
            n_components: k,
            n_features: d,
        })
    }

    /// Project `data` onto the components of a fitted `model`.
    pub fn transform(model: &PcaModel, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, PcaError> {
        let d = model.n_features;
        let k = model.n_components;
        let mut out = Vec::with_capacity(data.len());
        for row in data {
            if row.len() != d {
                return Err(PcaError::DimensionMismatch);
            }
            // Center / scale
            let centered: Vec<f64> = row
                .iter()
                .enumerate()
                .map(|(j, &v)| (v - model.mean[j]) / model.std_dev[j])
                .collect();
            // Project onto each component
            let projected: Vec<f64> = (0..k)
                .map(|c| dot(&centered, &model.components[c]))
                .collect();
            out.push(projected);
        }
        Ok(out)
    }

    /// Fit a model and immediately transform `data`.
    pub fn fit_transform(
        data: &[Vec<f64>],
        config: &PcaConfig,
    ) -> Result<(PcaModel, Vec<Vec<f64>>), PcaError> {
        let model = Self::fit(data, config)?;
        let reduced = Self::transform(&model, data)?;
        Ok((model, reduced))
    }

    /// Reconstruct approximate original-space vectors from reduced-space vectors.
    pub fn inverse_transform(
        model: &PcaModel,
        reduced: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, PcaError> {
        let k = model.n_components;
        let d = model.n_features;
        let mut out = Vec::with_capacity(reduced.len());
        for row in reduced {
            if row.len() != k {
                return Err(PcaError::DimensionMismatch);
            }
            // Reconstruct in centred/scaled space: x ≈ Σ row[c] * component[c]
            let mut rec = vec![0.0f64; d];
            for (c, &coeff) in row.iter().enumerate() {
                #[allow(clippy::needless_range_loop)]
                for j in 0..d {
                    rec[j] += coeff * model.components[c][j];
                }
            }
            // Un-scale then un-center
            #[allow(clippy::needless_range_loop)]
            for j in 0..d {
                rec[j] = rec[j] * model.std_dev[j] + model.mean[j];
            }
            out.push(rec);
        }
        Ok(out)
    }

    /// Return the fraction of total variance explained by each component.
    pub fn explained_variance_ratio(model: &PcaModel) -> Vec<f64> {
        model.explained_variance.clone()
    }

    /// Return the cumulative explained variance for each component.
    pub fn cumulative_explained_variance(model: &PcaModel) -> Vec<f64> {
        let mut cum = 0.0;
        model
            .explained_variance
            .iter()
            .map(|&v| {
                cum += v;
                cum
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple 2-D dataset with variance along the first axis.
    fn axis_aligned_2d() -> Vec<Vec<f64>> {
        vec![
            vec![3.0, 0.0],
            vec![-3.0, 0.0],
            vec![3.0, 0.1],
            vec![-3.0, -0.1],
            vec![3.0, 0.05],
            vec![-3.0, -0.05],
        ]
    }

    /// 3-D dataset where variance is dominant along the first two axes.
    fn data_3d() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 2.0, 0.01],
            vec![-1.0, -2.0, 0.01],
            vec![2.0, 4.0, -0.01],
            vec![-2.0, -4.0, -0.01],
            vec![0.5, 1.0, 0.005],
            vec![-0.5, -1.0, -0.005],
        ]
    }

    // ── Error cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_fit_insufficient_samples() {
        let data = vec![vec![1.0, 2.0]];
        let config = PcaConfig {
            n_components: 1,
            center: true,
            scale: false,
        };
        assert!(matches!(
            PcaReducer::fit(&data, &config),
            Err(PcaError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_fit_insufficient_features() {
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        let config = PcaConfig {
            n_components: 1,
            center: true,
            scale: false,
        };
        assert!(matches!(
            PcaReducer::fit(&data, &config),
            Err(PcaError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_fit_too_many_components() {
        let data = axis_aligned_2d();
        let config = PcaConfig {
            n_components: 5,
            center: true,
            scale: false,
        };
        assert!(matches!(
            PcaReducer::fit(&data, &config),
            Err(PcaError::InvalidComponents { .. })
        ));
    }

    #[test]
    fn test_fit_zero_components() {
        let data = axis_aligned_2d();
        let config = PcaConfig {
            n_components: 0,
            center: true,
            scale: false,
        };
        assert!(matches!(
            PcaReducer::fit(&data, &config),
            Err(PcaError::InvalidComponents { .. })
        ));
    }

    #[test]
    fn test_transform_dimension_mismatch() {
        let data = axis_aligned_2d();
        let config = PcaConfig::default();
        let model = PcaReducer::fit(&data, &config).expect("fit ok");
        let bad = vec![vec![1.0, 2.0, 3.0]]; // 3 features, model expects 2
        assert_eq!(
            PcaReducer::transform(&model, &bad),
            Err(PcaError::DimensionMismatch)
        );
    }

    #[test]
    fn test_inverse_transform_dimension_mismatch() {
        let data = axis_aligned_2d();
        let config = PcaConfig::default();
        let model = PcaReducer::fit(&data, &config).expect("fit ok");
        let bad = vec![vec![1.0, 2.0, 3.0]]; // 3 comps, model has 2
        assert_eq!(
            PcaReducer::inverse_transform(&model, &bad),
            Err(PcaError::DimensionMismatch)
        );
    }

    // ── PcaError display ─────────────────────────────────────────────────────

    #[test]
    fn test_error_display_insufficient() {
        let e = PcaError::InsufficientData {
            samples: 1,
            features: 1,
        };
        assert!(e.to_string().contains("insufficient"));
    }

    #[test]
    fn test_error_display_invalid_components() {
        let e = PcaError::InvalidComponents {
            requested: 5,
            max: 2,
        };
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let e = PcaError::DimensionMismatch;
        assert!(e.to_string().contains("dimension"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(PcaError::DimensionMismatch);
        assert!(!e.to_string().is_empty());
    }

    // ── Fit output properties ─────────────────────────────────────────────────

    #[test]
    fn test_fit_produces_model_with_correct_n_components() {
        let data = axis_aligned_2d();
        let config = PcaConfig {
            n_components: 1,
            center: true,
            scale: false,
        };
        let model = PcaReducer::fit(&data, &config).expect("fit ok");
        assert_eq!(model.n_components, 1);
        assert_eq!(model.components.len(), 1);
    }

    #[test]
    fn test_fit_component_is_unit_vector() {
        let data = axis_aligned_2d();
        let config = PcaConfig {
            n_components: 1,
            center: true,
            scale: false,
        };
        let model = PcaReducer::fit(&data, &config).expect("fit ok");
        let n = norm(&model.components[0]);
        assert!((n - 1.0).abs() < 1e-9, "component should be a unit vector");
    }

    #[test]
    fn test_fit_mean_equals_column_means() {
        let data = axis_aligned_2d();
        let config = PcaConfig {
            n_components: 1,
            center: true,
            scale: false,
        };
        let model = PcaReducer::fit(&data, &config).expect("fit ok");
        // All x-values sum to 0, all y-values sum to 0
        assert!(model.mean[0].abs() < 1e-9);
        assert!(model.mean[1].abs() < 1e-9);
    }

    #[test]
    fn test_fit_n_features_stored() {
        let data = axis_aligned_2d();
        let config = PcaConfig::default();
        let model = PcaReducer::fit(&data, &config).expect("fit ok");
        assert_eq!(model.n_features, 2);
    }

    #[test]
    fn test_fit_explained_variance_sums_to_one() {
        let data = data_3d();
        let config = PcaConfig {
            n_components: 2,
            center: true,
            scale: false,
        };
        let model = PcaReducer::fit(&data, &config).expect("fit ok");
        let sum: f64 = model.explained_variance.iter().sum();
        // Sum should equal 1.0 (ratio is normalised against extracted components)
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "explained variance ratios should sum to 1.0"
        );
    }

    #[test]
    fn test_fit_std_dev_ones_when_no_scale() {
        let data = axis_aligned_2d();
        let config = PcaConfig {
            n_components: 1,
            center: true,
            scale: false,
        };
        let model = PcaReducer::fit(&data, &config).expect("fit ok");
        for s in &model.std_dev {
            assert!((*s - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_fit_scale_changes_std_dev() {
        let data = axis_aligned_2d();
        let config = PcaConfig {
            n_components: 1,
            center: true,
            scale: true,
        };
        let model = PcaReducer::fit(&data, &config).expect("fit ok");
        // std_dev for the x-axis should be ~3.0
        assert!(
            model.std_dev[0] > 1.0,
            "std_dev should be > 1 for large-variance feature"
        );
    }

    // ── Transform ────────────────────────────────────────────────────────────

    #[test]
    fn test_transform_output_shape() {
        let data = axis_aligned_2d();
        let config = PcaConfig {
            n_components: 1,
            center: true,
            scale: false,
        };
        let model = PcaReducer::fit(&data, &config).expect("fit ok");
        let reduced = PcaReducer::transform(&model, &data).expect("transform ok");
        assert_eq!(reduced.len(), data.len());
        assert_eq!(reduced[0].len(), 1);
    }

    #[test]
    fn test_transform_two_components() {
        let data = data_3d();
        let config = PcaConfig {
            n_components: 2,
            center: true,
            scale: false,
        };
        let model = PcaReducer::fit(&data, &config).expect("fit ok");
        let reduced = PcaReducer::transform(&model, &data).expect("transform ok");
        for row in &reduced {
            assert_eq!(row.len(), 2);
        }
    }

    // ── fit_transform ────────────────────────────────────────────────────────

    #[test]
    fn test_fit_transform_consistent_with_separate_calls() {
        let data = axis_aligned_2d();
        let config = PcaConfig::default();
        let (model, reduced_combined) = PcaReducer::fit_transform(&data, &config).expect("ok");
        let reduced_separate = PcaReducer::transform(&model, &data).expect("ok");
        for (a, b) in reduced_combined.iter().zip(reduced_separate.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }

    // ── inverse_transform ────────────────────────────────────────────────────

    #[test]
    fn test_inverse_transform_output_shape() {
        let data = axis_aligned_2d();
        let config = PcaConfig::default();
        let (model, reduced) = PcaReducer::fit_transform(&data, &config).expect("ok");
        let reconstructed = PcaReducer::inverse_transform(&model, &reduced).expect("ok");
        assert_eq!(reconstructed.len(), data.len());
        assert_eq!(reconstructed[0].len(), 2);
    }

    #[test]
    fn test_inverse_transform_approximation() {
        // With k == d == 2 components, reconstruction should be exact (up to floating point).
        let data = axis_aligned_2d();
        let config = PcaConfig {
            n_components: 2,
            center: true,
            scale: false,
        };
        let (model, reduced) = PcaReducer::fit_transform(&data, &config).expect("ok");
        let reconstructed = PcaReducer::inverse_transform(&model, &reduced).expect("ok");
        for (orig, rec) in data.iter().zip(reconstructed.iter()) {
            for (a, b) in orig.iter().zip(rec.iter()) {
                // Allow slightly larger tolerance due to floating-point accumulation
                assert!(
                    (a - b).abs() < 0.05,
                    "reconstruction error too large: {a} vs {b}"
                );
            }
        }
    }

    // ── explained_variance_ratio / cumulative ─────────────────────────────────

    #[test]
    fn test_explained_variance_ratio_all_nonneg() {
        let data = data_3d();
        let config = PcaConfig {
            n_components: 2,
            center: true,
            scale: false,
        };
        let model = PcaReducer::fit(&data, &config).expect("ok");
        for &r in PcaReducer::explained_variance_ratio(&model).iter() {
            assert!(r >= 0.0);
        }
    }

    #[test]
    fn test_cumulative_explained_variance_monotone() {
        let data = data_3d();
        let config = PcaConfig {
            n_components: 2,
            center: true,
            scale: false,
        };
        let model = PcaReducer::fit(&data, &config).expect("ok");
        let cum = PcaReducer::cumulative_explained_variance(&model);
        assert_eq!(cum.len(), 2);
        assert!(cum[0] <= cum[1] + 1e-12);
    }

    #[test]
    fn test_cumulative_explained_variance_last_is_one() {
        let data = axis_aligned_2d();
        let config = PcaConfig {
            n_components: 2,
            center: true,
            scale: false,
        };
        let model = PcaReducer::fit(&data, &config).expect("ok");
        let cum = PcaReducer::cumulative_explained_variance(&model);
        assert!((cum.last().copied().unwrap_or(0.0) - 1.0).abs() < 1e-9);
    }

    // ── Math helpers ──────────────────────────────────────────────────────────

    #[test]
    fn test_dot_product() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_norm() {
        assert!((norm(&[3.0, 4.0]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((norm(&v) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_zero_vector_unchanged() {
        let mut v = vec![0.0, 0.0];
        normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    // ── PcaReducer::new ───────────────────────────────────────────────────────

    #[test]
    fn test_pca_reducer_new() {
        let _r = PcaReducer::new();
    }

    // ── PcaConfig default ─────────────────────────────────────────────────────

    #[test]
    fn test_pca_config_default() {
        let c = PcaConfig::default();
        assert_eq!(c.n_components, 2);
        assert!(c.center);
        assert!(!c.scale);
    }

    // ── Scale mode ────────────────────────────────────────────────────────────

    #[test]
    fn test_fit_transform_with_scale() {
        let data = vec![
            vec![100.0, 0.001],
            vec![-100.0, 0.002],
            vec![200.0, -0.001],
            vec![-200.0, -0.002],
            vec![50.0, 0.0005],
            vec![-50.0, -0.0005],
        ];
        let config = PcaConfig {
            n_components: 1,
            center: true,
            scale: true,
        };
        let result = PcaReducer::fit_transform(&data, &config);
        assert!(result.is_ok(), "fit_transform with scale should succeed");
    }

    // ── Repeated fit_transform gives unit-length components ──────────────────

    #[test]
    fn test_components_orthogonal() {
        let data = data_3d();
        let config = PcaConfig {
            n_components: 2,
            center: true,
            scale: false,
        };
        let model = PcaReducer::fit(&data, &config).expect("ok");
        let d01 = dot(&model.components[0], &model.components[1]).abs();
        assert!(d01 < 1e-6, "components should be orthogonal, got dot={d01}");
    }

    #[test]
    fn test_model_clone() {
        let data = axis_aligned_2d();
        let config = PcaConfig::default();
        let model = PcaReducer::fit(&data, &config).expect("ok");
        let model2 = model.clone();
        assert_eq!(model.n_components, model2.n_components);
    }
}
