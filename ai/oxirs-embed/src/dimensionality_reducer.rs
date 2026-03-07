//! # Dimensionality Reducer
//!
//! PCA-based dimensionality reduction for embedding vectors.
//! Implements power-iteration PCA for top-k eigenvector extraction,
//! plus a thin Truncated SVD wrapper.
//!
//! ## Algorithm
//!
//! 1. Center the data (subtract column means).
//! 2. Compute the covariance matrix.
//! 3. Use power iteration with deflation to extract the top-k eigenvectors.
//! 4. Project data onto the component subspace.

use std::fmt;

// ─────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────

/// Errors from dimensionality reduction operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ReductionError {
    /// Not enough data samples to compute PCA.
    InsufficientData,
    /// Requested more components than features.
    TooManyComponents,
    /// Dimension mismatch between training and transform data.
    DimensionMismatch(String),
}

impl fmt::Display for ReductionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientData => write!(f, "insufficient data for PCA"),
            Self::TooManyComponents => write!(f, "requested more components than features"),
            Self::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
        }
    }
}

impl std::error::Error for ReductionError {}

// ─────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────

/// Output of a `fit_transform` call.
#[derive(Debug, Clone)]
pub struct ReductionResult {
    /// Projected data in the reduced space.
    pub reduced: Vec<Vec<f64>>,
    /// Fraction of total variance explained by each component.
    pub explained_variance_ratio: Vec<f64>,
    /// Sum of `explained_variance_ratio`.
    pub total_variance_explained: f64,
}

// ─────────────────────────────────────────────
// PCA reducer
// ─────────────────────────────────────────────

/// PCA dimensionality reducer using power-iteration eigenvector extraction.
#[derive(Debug, Clone)]
pub struct PcaReducer {
    /// Principal components (each row is one eigenvector, shape n_components × n_features).
    pub components: Vec<Vec<f64>>,
    /// Per-feature mean used for centering.
    pub mean: Vec<f64>,
    /// Variance explained by each component.
    pub explained_variance: Vec<f64>,
    /// Number of components retained.
    pub n_components: usize,
}

impl PcaReducer {
    /// Fit PCA on `data` and retain `n_components` principal components.
    ///
    /// Uses 50 power-iteration steps per component.
    pub fn fit(data: &[Vec<f64>], n_components: usize) -> Result<Self, ReductionError> {
        if data.is_empty() {
            return Err(ReductionError::InsufficientData);
        }
        let n_features = data[0].len();
        if n_features == 0 {
            return Err(ReductionError::InsufficientData);
        }
        if n_components > n_features {
            return Err(ReductionError::TooManyComponents);
        }
        if n_components == 0 {
            return Err(ReductionError::TooManyComponents);
        }

        let (centered, mean) = center_data(data);
        let covariance = compute_covariance(&centered);

        let mut components: Vec<Vec<f64>> = Vec::with_capacity(n_components);
        let mut explained_variance: Vec<f64> = Vec::with_capacity(n_components);

        // Deflating copy of the covariance matrix.
        let mut cov = covariance;

        for _ in 0..n_components {
            let (eigenvec, eigenval) = power_iteration(&cov, 50);
            // Deflate: cov ← cov − λ * v * vᵀ
            deflate(&mut cov, &eigenvec, eigenval);
            explained_variance.push(eigenval);
            components.push(eigenvec);
        }

        Ok(Self {
            components,
            mean,
            explained_variance,
            n_components,
        })
    }

    /// Project `data` onto the PCA component space.
    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ReductionError> {
        if data.is_empty() {
            return Ok(vec![]);
        }
        let n_features = data[0].len();
        if n_features != self.mean.len() {
            return Err(ReductionError::DimensionMismatch(format!(
                "expected {} features, got {}",
                self.mean.len(),
                n_features
            )));
        }

        let mut result = Vec::with_capacity(data.len());
        for row in data {
            let centered: Vec<f64> = row
                .iter()
                .zip(self.mean.iter())
                .map(|(x, m)| x - m)
                .collect();
            let projected = self
                .components
                .iter()
                .map(|comp| dot_product(&centered, comp))
                .collect();
            result.push(projected);
        }
        Ok(result)
    }

    /// Fit PCA and immediately project the training data.
    pub fn fit_transform(
        data: &[Vec<f64>],
        n_components: usize,
    ) -> Result<ReductionResult, ReductionError> {
        let reducer = Self::fit(data, n_components)?;
        let reduced = reducer.transform(data)?;

        let total_var: f64 = reducer
            .explained_variance
            .iter()
            .sum::<f64>()
            .max(f64::EPSILON);
        let explained_variance_ratio: Vec<f64> = reducer
            .explained_variance
            .iter()
            .map(|&v| v / total_var)
            .collect();
        let total_variance_explained: f64 = explained_variance_ratio.iter().sum();

        Ok(ReductionResult {
            reduced,
            explained_variance_ratio,
            total_variance_explained,
        })
    }

    /// Approximately reconstruct original-space vectors from reduced-space vectors.
    ///
    /// This is a lossy reconstruction: `x̂ = mean + W * z`
    pub fn inverse_transform(&self, reduced: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_features = self.mean.len();
        let mut result = Vec::with_capacity(reduced.len());

        for row in reduced {
            let mut rec = self.mean.clone();
            for (k, &coeff) in row.iter().enumerate() {
                if let Some(comp) = self.components.get(k) {
                    for (f, &w) in comp.iter().enumerate() {
                        rec[f] += coeff * w;
                    }
                }
            }
            // rec should have n_features entries
            let _ = n_features;
            result.push(rec);
        }
        result
    }
}

// ─────────────────────────────────────────────
// Truncated SVD (thin wrapper)
// ─────────────────────────────────────────────

/// Truncated SVD: operates on the raw (uncentered) data matrix.
#[derive(Debug, Clone)]
pub struct TruncatedSvd {
    /// Right singular vectors (n_components × n_features).
    pub components: Vec<Vec<f64>>,
    /// Singular values.
    pub singular_values: Vec<f64>,
    /// Number of components retained.
    pub n_components: usize,
}

impl TruncatedSvd {
    /// Compute a truncated SVD via power iteration on XᵀX.
    pub fn fit(data: &[Vec<f64>], n_components: usize) -> Result<Self, ReductionError> {
        if data.is_empty() {
            return Err(ReductionError::InsufficientData);
        }
        let n_features = data[0].len();
        if n_components > n_features {
            return Err(ReductionError::TooManyComponents);
        }
        if n_components == 0 {
            return Err(ReductionError::TooManyComponents);
        }

        // Build XᵀX (n_features × n_features)
        let xt_x = compute_gram(data);

        let mut components: Vec<Vec<f64>> = Vec::with_capacity(n_components);
        let mut singular_values: Vec<f64> = Vec::with_capacity(n_components);
        let mut gram = xt_x;

        for _ in 0..n_components {
            let (v, lambda) = power_iteration(&gram, 50);
            deflate(&mut gram, &v, lambda);
            singular_values.push(lambda.sqrt().max(0.0));
            components.push(v);
        }

        Ok(Self {
            components,
            singular_values,
            n_components,
        })
    }
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

/// Center the data by subtracting column means.
/// Returns `(centered_data, column_means)`.
pub fn center_data(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = data.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let d = data[0].len();
    let mut mean = vec![0.0f64; d];
    for row in data {
        for (j, &v) in row.iter().enumerate() {
            mean[j] += v;
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    let centered = data
        .iter()
        .map(|row| row.iter().zip(mean.iter()).map(|(x, m)| x - m).collect())
        .collect();
    (centered, mean)
}

/// Compute the dot product of two equal-length slices.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Normalise a vector in-place (L2 norm). No-op for zero vectors.
pub fn normalize(v: &mut [f64]) {
    let norm = dot_product(v, v).sqrt();
    if norm > f64::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Multiply matrix `mat` (m × n) by vector `vec` (n) → result (m).
pub fn mat_vec_mul(mat: &[Vec<f64>], vec: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| dot_product(row, vec)).collect()
}

/// Compute the covariance matrix (n_features × n_features) of centered data.
fn compute_covariance(centered: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = centered.len();
    if n == 0 {
        return vec![];
    }
    let d = centered[0].len();
    let denom = (n.saturating_sub(1).max(1)) as f64;

    let mut cov = vec![vec![0.0f64; d]; d];
    for row in centered {
        #[allow(clippy::needless_range_loop)]
        for i in 0..d {
            for j in i..d {
                cov[i][j] += row[i] * row[j];
            }
        }
    }
    #[allow(clippy::needless_range_loop)]
    for i in 0..d {
        for j in i..d {
            let val = cov[i][j] / denom;
            cov[i][j] = val;
            cov[j][i] = val;
        }
    }
    cov
}

/// Compute XᵀX (Gram matrix) for SVD.
fn compute_gram(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    let d = data[0].len();
    let denom = n as f64;
    let mut gram = vec![vec![0.0f64; d]; d];
    for row in data {
        #[allow(clippy::needless_range_loop)]
        for i in 0..d {
            for j in i..d {
                gram[i][j] += row[i] * row[j];
            }
        }
    }
    #[allow(clippy::needless_range_loop)]
    for i in 0..d {
        for j in i..d {
            let val = gram[i][j] / denom;
            gram[i][j] = val;
            gram[j][i] = val;
        }
    }
    gram
}

/// Power iteration: find the dominant eigenvector of a symmetric matrix.
/// Returns `(eigenvector, eigenvalue)`.
fn power_iteration(mat: &[Vec<f64>], iterations: usize) -> (Vec<f64>, f64) {
    let d = mat.len();
    if d == 0 {
        return (vec![], 0.0);
    }
    // Initialise with a seeded non-zero vector.
    let mut v: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0).recip()).collect();
    normalize(&mut v);

    for _ in 0..iterations {
        let mut w = mat_vec_mul(mat, &v);
        let eigenval_est = dot_product(&v, &w);
        if eigenval_est.abs() < f64::EPSILON {
            break;
        }
        normalize(&mut w);
        v = w;
    }

    // Rayleigh quotient for eigenvalue.
    let av = mat_vec_mul(mat, &v);
    let eigenvalue = dot_product(&v, &av);

    (v, eigenvalue.max(0.0))
}

/// Deflate a symmetric matrix: `mat ← mat − λ * v * vᵀ`
fn deflate(mat: &mut [Vec<f64>], v: &[f64], lambda: f64) {
    let d = mat.len();
    for i in 0..d {
        for j in 0..d {
            mat[i][j] -= lambda * v[i] * v[j];
        }
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    /// Simple 2-D dataset with strong first principal axis along (1, 1) / √2.
    fn axis_data() -> Vec<Vec<f64>> {
        // Points along the line y = x with small noise
        vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
            vec![5.0, 5.0],
        ]
    }

    fn near(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── center_data ────────────────────────────────────────────────────

    #[test]
    fn test_center_data_zero_mean() {
        let data = vec![vec![1.0, 3.0], vec![3.0, 7.0]];
        let (centered, mean) = center_data(&data);
        assert!(near(mean[0], 2.0, 1e-9));
        assert!(near(mean[1], 5.0, 1e-9));
        assert!(near(centered[0][0], -1.0, 1e-9));
        assert!(near(centered[1][0], 1.0, 1e-9));
    }

    #[test]
    fn test_center_data_empty() {
        let (centered, mean) = center_data(&[]);
        assert!(centered.is_empty());
        assert!(mean.is_empty());
    }

    // ── dot_product ────────────────────────────────────────────────────

    #[test]
    fn test_dot_product_basic() {
        assert!(near(
            dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
            32.0,
            1e-9
        ));
    }

    #[test]
    fn test_dot_product_orthogonal() {
        assert!(near(dot_product(&[1.0, 0.0], &[0.0, 1.0]), 0.0, 1e-9));
    }

    // ── normalize ─────────────────────────────────────────────────────

    #[test]
    fn test_normalize_unit() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        let norm = dot_product(&v, &v).sqrt();
        assert!(near(norm, 1.0, 1e-9));
    }

    #[test]
    fn test_normalize_zero_vector_noop() {
        let mut v = vec![0.0, 0.0];
        normalize(&mut v);
        assert!(near(v[0], 0.0, 1e-9));
    }

    // ── mat_vec_mul ────────────────────────────────────────────────────

    #[test]
    fn test_mat_vec_mul_identity() {
        let eye = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![3.0, 5.0];
        let result = mat_vec_mul(&eye, &v);
        assert!(near(result[0], 3.0, 1e-9));
        assert!(near(result[1], 5.0, 1e-9));
    }

    // ── PcaReducer::fit ────────────────────────────────────────────────

    #[test]
    fn test_fit_success() {
        let data = axis_data();
        let pca = PcaReducer::fit(&data, 1).expect("should fit");
        assert_eq!(pca.n_components, 1);
        assert_eq!(pca.components.len(), 1);
        assert_eq!(pca.mean.len(), 2);
    }

    #[test]
    fn test_fit_empty_data_error() {
        assert!(matches!(
            PcaReducer::fit(&[], 1),
            Err(ReductionError::InsufficientData)
        ));
    }

    #[test]
    fn test_fit_too_many_components_error() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(matches!(
            PcaReducer::fit(&data, 3),
            Err(ReductionError::TooManyComponents)
        ));
    }

    #[test]
    fn test_fit_zero_components_error() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(matches!(
            PcaReducer::fit(&data, 0),
            Err(ReductionError::TooManyComponents)
        ));
    }

    // ── PcaReducer::transform ──────────────────────────────────────────

    #[test]
    fn test_transform_correct_output_shape() {
        let data = axis_data();
        let pca = PcaReducer::fit(&data, 1).expect("fit");
        let reduced = pca.transform(&data).expect("transform");
        assert_eq!(reduced.len(), data.len());
        assert_eq!(reduced[0].len(), 1);
    }

    #[test]
    fn test_transform_empty_input() {
        let data = axis_data();
        let pca = PcaReducer::fit(&data, 1).expect("fit");
        let reduced = pca.transform(&[]).expect("transform");
        assert!(reduced.is_empty());
    }

    #[test]
    fn test_transform_dimension_mismatch() {
        let data = axis_data();
        let pca = PcaReducer::fit(&data, 1).expect("fit");
        let bad_data = vec![vec![1.0, 2.0, 3.0]]; // 3 features, expected 2
        assert!(matches!(
            pca.transform(&bad_data),
            Err(ReductionError::DimensionMismatch(_))
        ));
    }

    // ── PcaReducer::fit_transform ──────────────────────────────────────

    #[test]
    fn test_fit_transform_variance_ratio_sums_to_one() {
        let data: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64, (i * 2) as f64, (i * 3) as f64])
            .collect();
        let result = PcaReducer::fit_transform(&data, 2).expect("fit_transform");
        assert!(
            near(result.total_variance_explained, 1.0, 0.05),
            "total_variance_explained={}",
            result.total_variance_explained
        );
    }

    #[test]
    fn test_fit_transform_shape() {
        let data = axis_data();
        let result = PcaReducer::fit_transform(&data, 1).expect("fit_transform");
        assert_eq!(result.reduced.len(), data.len());
        assert_eq!(result.reduced[0].len(), 1);
        assert_eq!(result.explained_variance_ratio.len(), 1);
    }

    #[test]
    fn test_fit_transform_explained_variance_first_component() {
        // For data along y=x, first PC should capture nearly all variance.
        let data = axis_data();
        let result = PcaReducer::fit_transform(&data, 1).expect("fit_transform");
        assert!(
            result.total_variance_explained > 0.95,
            "expected high explained variance, got {}",
            result.total_variance_explained
        );
    }

    // ── PcaReducer::inverse_transform ─────────────────────────────────

    #[test]
    fn test_inverse_transform_shape() {
        let data = axis_data();
        let pca = PcaReducer::fit(&data, 1).expect("fit");
        let reduced = pca.transform(&data).expect("transform");
        let reconstructed = pca.inverse_transform(&reduced);
        assert_eq!(reconstructed.len(), data.len());
        assert_eq!(reconstructed[0].len(), data[0].len());
    }

    #[test]
    fn test_inverse_transform_approximate_reconstruction() {
        // For data along y=x, reconstruction should be close.
        let data = axis_data();
        let pca = PcaReducer::fit(&data, 1).expect("fit");
        let reduced = pca.transform(&data).expect("transform");
        let reconstructed = pca.inverse_transform(&reduced);
        // Each reconstructed point should be close to original
        for (orig, rec) in data.iter().zip(reconstructed.iter()) {
            let err: f64 = orig
                .iter()
                .zip(rec.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(err < 0.5, "reconstruction error too large: {err}");
        }
    }

    #[test]
    fn test_inverse_transform_empty() {
        let data = axis_data();
        let pca = PcaReducer::fit(&data, 1).expect("fit");
        let reconstructed = pca.inverse_transform(&[]);
        assert!(reconstructed.is_empty());
    }

    // ── TruncatedSvd ───────────────────────────────────────────────────

    #[test]
    fn test_truncated_svd_fit_success() {
        let data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let svd = TruncatedSvd::fit(&data, 1).expect("svd fit");
        assert_eq!(svd.n_components, 1);
        assert_eq!(svd.singular_values.len(), 1);
        assert!(svd.singular_values[0] >= 0.0);
    }

    #[test]
    fn test_truncated_svd_too_many_components() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(matches!(
            TruncatedSvd::fit(&data, 3),
            Err(ReductionError::TooManyComponents)
        ));
    }

    #[test]
    fn test_truncated_svd_empty_data() {
        assert!(matches!(
            TruncatedSvd::fit(&[], 1),
            Err(ReductionError::InsufficientData)
        ));
    }

    // ── error display ─────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = ReductionError::DimensionMismatch("test".to_string());
        assert!(e.to_string().contains("test"));
        let e2 = ReductionError::InsufficientData;
        assert!(!e2.to_string().is_empty());
        let e3 = ReductionError::TooManyComponents;
        assert!(!e3.to_string().is_empty());
    }

    // ── 3D data ────────────────────────────────────────────────────────

    #[test]
    fn test_3d_to_2d_reduction() {
        let data: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let x = i as f64;
                vec![x, x * 2.0, x * 0.5 + 1.0]
            })
            .collect();
        let pca = PcaReducer::fit(&data, 2).expect("fit");
        let reduced = pca.transform(&data).expect("transform");
        assert_eq!(reduced[0].len(), 2);
    }
}
