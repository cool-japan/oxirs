//! # Procrustes Alignment for Embedding Spaces
//!
//! Implements orthogonal Procrustes analysis for aligning embedding spaces from
//! different models or languages. Given a set of anchor/seed pairs, this module
//! finds the optimal rotation matrix that minimizes the sum of squared distances
//! between the aligned source embeddings and the target embeddings.
//!
//! ## Features
//!
//! - **Orthogonal Procrustes**: Find the optimal rotation (orthogonal) matrix via SVD
//! - **Translation alignment**: Optionally center embeddings before rotation
//! - **Alignment quality metrics**: Mean squared error, cosine similarity
//! - **Nearest-neighbor evaluation**: Precision@k for alignment quality
//! - **Batch transformation**: Apply learned alignment to new embeddings

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/// An anchor pair mapping a source embedding index to a target embedding index.
#[derive(Debug, Clone)]
pub struct AnchorPair {
    /// Index into the source embedding matrix.
    pub source_idx: usize,
    /// Index into the target embedding matrix.
    pub target_idx: usize,
    /// Optional label (e.g., entity name).
    pub label: Option<String>,
}

impl AnchorPair {
    pub fn new(source_idx: usize, target_idx: usize) -> Self {
        Self {
            source_idx,
            target_idx,
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Configuration for Procrustes alignment.
#[derive(Debug, Clone)]
pub struct ProcrustesConfig {
    /// Whether to center (mean-subtract) embeddings before alignment.
    pub center: bool,
    /// Whether to normalize embeddings to unit length before alignment.
    pub normalize: bool,
    /// Regularization parameter (small positive value for numerical stability).
    pub regularization: f64,
}

impl Default for ProcrustesConfig {
    fn default() -> Self {
        Self {
            center: true,
            normalize: false,
            regularization: 1e-10,
        }
    }
}

/// The result of a Procrustes alignment.
#[derive(Debug, Clone)]
pub struct ProcrustesResult {
    /// The rotation matrix (dim x dim) that transforms source to target space.
    pub rotation_matrix: Vec<Vec<f64>>,
    /// Source centroid (subtracted before rotation if centering is on).
    pub source_centroid: Vec<f64>,
    /// Target centroid (added after rotation if centering is on).
    pub target_centroid: Vec<f64>,
    /// Mean squared error on the anchor pairs after alignment.
    pub mse: f64,
    /// Mean cosine similarity on anchor pairs after alignment.
    pub mean_cosine_similarity: f64,
    /// Dimensionality.
    pub dim: usize,
}

impl ProcrustesResult {
    /// Transform a single embedding from source space to target space.
    pub fn transform(&self, embedding: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        // 1. Subtract source centroid
        let centered: Vec<f64> = (0..dim)
            .map(|i| embedding.get(i).copied().unwrap_or(0.0) - self.source_centroid[i])
            .collect();

        // 2. Apply rotation
        let mut rotated = vec![0.0; dim];
        for (i, rot_val) in rotated.iter_mut().enumerate().take(dim) {
            for (j, &c_val) in centered.iter().enumerate().take(dim) {
                *rot_val += self.rotation_matrix[i][j] * c_val;
            }
        }

        // 3. Add target centroid
        for (i, val) in rotated.iter_mut().enumerate().take(dim) {
            *val += self.target_centroid[i];
        }

        rotated
    }

    /// Transform a batch of embeddings.
    pub fn transform_batch(&self, embeddings: &[Vec<f64>]) -> Vec<Vec<f64>> {
        embeddings.iter().map(|e| self.transform(e)).collect()
    }
}

/// Alignment quality metrics.
#[derive(Debug, Clone)]
pub struct AlignmentMetrics {
    /// Mean squared error on evaluation pairs.
    pub mse: f64,
    /// Mean cosine similarity on evaluation pairs.
    pub mean_cosine_similarity: f64,
    /// Precision@1 (fraction of source embeddings whose nearest neighbor in target
    /// space is the correct match).
    pub precision_at_1: f64,
    /// Precision@5.
    pub precision_at_5: f64,
    /// Precision@10.
    pub precision_at_10: f64,
    /// Number of evaluation pairs.
    pub eval_pairs: usize,
}

// ─────────────────────────────────────────────
// ProcrustesAligner
// ─────────────────────────────────────────────

/// Procrustes alignment for embedding spaces.
pub struct ProcrustesAligner {
    config: ProcrustesConfig,
}

impl ProcrustesAligner {
    /// Create a new aligner with default configuration.
    pub fn new() -> Self {
        Self {
            config: ProcrustesConfig::default(),
        }
    }

    /// Create a new aligner with the given configuration.
    pub fn with_config(config: ProcrustesConfig) -> Self {
        Self { config }
    }

    /// Compute the optimal alignment.
    ///
    /// `source_embeddings`: rows of source embedding matrix (n x dim)
    /// `target_embeddings`: rows of target embedding matrix (m x dim)
    /// `anchors`: pairs mapping source indices to target indices
    pub fn align(
        &self,
        source_embeddings: &[Vec<f64>],
        target_embeddings: &[Vec<f64>],
        anchors: &[AnchorPair],
    ) -> Result<ProcrustesResult, ProcrustesError> {
        if anchors.is_empty() {
            return Err(ProcrustesError::NoAnchors);
        }

        // Validate anchors
        for anchor in anchors {
            if anchor.source_idx >= source_embeddings.len() {
                return Err(ProcrustesError::InvalidIndex {
                    which: "source",
                    idx: anchor.source_idx,
                    len: source_embeddings.len(),
                });
            }
            if anchor.target_idx >= target_embeddings.len() {
                return Err(ProcrustesError::InvalidIndex {
                    which: "target",
                    idx: anchor.target_idx,
                    len: target_embeddings.len(),
                });
            }
        }

        // Determine dimensionality
        let dim = source_embeddings.first().map(|v| v.len()).unwrap_or(0);
        if dim == 0 {
            return Err(ProcrustesError::EmptyEmbeddings);
        }

        // Extract anchor subsets
        let src_anchors: Vec<Vec<f64>> = anchors
            .iter()
            .map(|a| source_embeddings[a.source_idx].clone())
            .collect();
        let tgt_anchors: Vec<Vec<f64>> = anchors
            .iter()
            .map(|a| target_embeddings[a.target_idx].clone())
            .collect();

        // Compute centroids
        let source_centroid = if self.config.center {
            compute_centroid(&src_anchors, dim)
        } else {
            vec![0.0; dim]
        };
        let target_centroid = if self.config.center {
            compute_centroid(&tgt_anchors, dim)
        } else {
            vec![0.0; dim]
        };

        // Center the anchor embeddings
        let src_centered = center_embeddings(&src_anchors, &source_centroid);
        let tgt_centered = center_embeddings(&tgt_anchors, &target_centroid);

        // Optionally normalize
        let src_final = if self.config.normalize {
            normalize_rows(&src_centered)
        } else {
            src_centered
        };
        let tgt_final = if self.config.normalize {
            normalize_rows(&tgt_centered)
        } else {
            tgt_centered
        };

        // Compute M = X^T Y (cross-covariance matrix)
        let m_matrix = cross_covariance(&src_final, &tgt_final, dim);

        // SVD of M: M = U S V^T => W = V U^T
        let (u, _s, vt) = svd(&m_matrix, dim)?;

        // Rotation W = V^T^T * U^T = V * U^T
        // Actually: W = V * U^T, where V = Vt^T
        let v = transpose(&vt, dim);
        let ut = transpose(&u, dim);
        let rotation = mat_mul(&v, &ut, dim);

        // Compute MSE and cosine similarity on anchors
        let mse = compute_mse(&src_final, &tgt_final, &rotation, dim);
        let mean_cos = compute_mean_cosine(&src_final, &tgt_final, &rotation, dim);

        Ok(ProcrustesResult {
            rotation_matrix: rotation,
            source_centroid,
            target_centroid,
            mse,
            mean_cosine_similarity: mean_cos,
            dim,
        })
    }

    /// Evaluate alignment quality using held-out pairs.
    pub fn evaluate(
        &self,
        result: &ProcrustesResult,
        source_embeddings: &[Vec<f64>],
        target_embeddings: &[Vec<f64>],
        eval_pairs: &[AnchorPair],
    ) -> AlignmentMetrics {
        if eval_pairs.is_empty() {
            return AlignmentMetrics {
                mse: 0.0,
                mean_cosine_similarity: 0.0,
                precision_at_1: 0.0,
                precision_at_5: 0.0,
                precision_at_10: 0.0,
                eval_pairs: 0,
            };
        }

        let mut total_se = 0.0;
        let mut total_cos = 0.0;
        let mut correct_at_1 = 0usize;
        let mut correct_at_5 = 0usize;
        let mut correct_at_10 = 0usize;

        for pair in eval_pairs {
            if pair.source_idx >= source_embeddings.len()
                || pair.target_idx >= target_embeddings.len()
            {
                continue;
            }
            let transformed = result.transform(&source_embeddings[pair.source_idx]);
            let target = &target_embeddings[pair.target_idx];

            // MSE
            let se: f64 = transformed
                .iter()
                .zip(target.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            total_se += se;

            // Cosine similarity
            let cos = cosine_sim(&transformed, target);
            total_cos += cos;

            // Find nearest neighbors in target space
            let neighbors = find_nearest_neighbors(&transformed, target_embeddings, 10);
            if neighbors.first().copied() == Some(pair.target_idx) {
                correct_at_1 += 1;
            }
            if neighbors.iter().take(5).any(|&idx| idx == pair.target_idx) {
                correct_at_5 += 1;
            }
            if neighbors.iter().take(10).any(|&idx| idx == pair.target_idx) {
                correct_at_10 += 1;
            }
        }

        let n = eval_pairs.len() as f64;
        AlignmentMetrics {
            mse: total_se / n,
            mean_cosine_similarity: total_cos / n,
            precision_at_1: correct_at_1 as f64 / n,
            precision_at_5: correct_at_5 as f64 / n,
            precision_at_10: correct_at_10 as f64 / n,
            eval_pairs: eval_pairs.len(),
        }
    }
}

impl Default for ProcrustesAligner {
    fn default() -> Self {
        Self::new()
    }
}

/// Error type for Procrustes alignment.
#[derive(Debug, Clone)]
pub enum ProcrustesError {
    NoAnchors,
    EmptyEmbeddings,
    InvalidIndex {
        which: &'static str,
        idx: usize,
        len: usize,
    },
    SvdFailed(String),
}

impl std::fmt::Display for ProcrustesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcrustesError::NoAnchors => write!(f, "no anchor pairs provided"),
            ProcrustesError::EmptyEmbeddings => write!(f, "embeddings are empty"),
            ProcrustesError::InvalidIndex { which, idx, len } => {
                write!(f, "invalid {which} index {idx} (length {len})")
            }
            ProcrustesError::SvdFailed(msg) => write!(f, "SVD failed: {msg}"),
        }
    }
}

impl std::error::Error for ProcrustesError {}

// ─────────────────────────────────────────────
// Linear algebra helpers
// ─────────────────────────────────────────────

fn compute_centroid(embeddings: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let n = embeddings.len() as f64;
    if n < 1.0 {
        return vec![0.0; dim];
    }
    let mut centroid = vec![0.0; dim];
    for emb in embeddings {
        for i in 0..dim.min(emb.len()) {
            centroid[i] += emb[i];
        }
    }
    for v in &mut centroid {
        *v /= n;
    }
    centroid
}

fn center_embeddings(embeddings: &[Vec<f64>], centroid: &[f64]) -> Vec<Vec<f64>> {
    embeddings
        .iter()
        .map(|emb| {
            emb.iter()
                .enumerate()
                .map(|(i, &v)| v - centroid.get(i).copied().unwrap_or(0.0))
                .collect()
        })
        .collect()
}

fn normalize_rows(embeddings: &[Vec<f64>]) -> Vec<Vec<f64>> {
    embeddings
        .iter()
        .map(|emb| {
            let norm: f64 = emb.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm < 1e-12 {
                emb.clone()
            } else {
                emb.iter().map(|v| v / norm).collect()
            }
        })
        .collect()
}

fn cross_covariance(src: &[Vec<f64>], tgt: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    // M[i][j] = sum_k src[k][i] * tgt[k][j]
    let mut m = vec![vec![0.0; dim]; dim];
    for k in 0..src.len().min(tgt.len()) {
        for (i, m_row) in m.iter_mut().enumerate().take(dim) {
            let si = src[k].get(i).copied().unwrap_or(0.0);
            for (j, m_val) in m_row.iter_mut().enumerate().take(dim) {
                let tj = tgt[k].get(j).copied().unwrap_or(0.0);
                *m_val += si * tj;
            }
        }
    }
    m
}

/// Result type for SVD decomposition: (U, singular_values, V^T).
type SvdResult = (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>);

/// Simple Jacobi-style SVD for small matrices.
/// Returns (U, singular_values, V^T) for a dim x dim matrix.
fn svd(matrix: &[Vec<f64>], dim: usize) -> Result<SvdResult, ProcrustesError> {
    // Compute A^T A
    let ata = mat_mul(&transpose(matrix, dim), matrix, dim);

    // Eigendecomposition of A^T A via Jacobi iteration
    let (eigenvalues, eigenvectors) = jacobi_eigendecomposition(&ata, dim, 200)?;

    // Singular values = sqrt(eigenvalues)
    let mut singular_values: Vec<f64> = eigenvalues
        .iter()
        .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
        .collect();

    // V = eigenvectors (columns), V^T = transpose
    let vt = transpose(&eigenvectors, dim);

    // U = A * V * S^{-1}
    let av = mat_mul(matrix, &eigenvectors, dim);
    let mut u = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            if singular_values[j].abs() > 1e-12 {
                u[i][j] = av[i][j] / singular_values[j];
            }
        }
    }

    // Sort by descending singular value
    let mut indices: Vec<usize> = (0..dim).collect();
    indices.sort_by(|&a, &b| {
        singular_values[b]
            .partial_cmp(&singular_values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_s: Vec<f64> = indices.iter().map(|&i| singular_values[i]).collect();
    let sorted_u: Vec<Vec<f64>> = (0..dim)
        .map(|row| indices.iter().map(|&col| u[row][col]).collect())
        .collect();
    let sorted_vt: Vec<Vec<f64>> = indices.iter().map(|&i| vt[i].clone()).collect();

    singular_values = sorted_s;

    Ok((sorted_u, singular_values, sorted_vt))
}

fn jacobi_eigendecomposition(
    matrix: &[Vec<f64>],
    dim: usize,
    max_iter: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), ProcrustesError> {
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut v: Vec<Vec<f64>> = (0..dim)
        .map(|i| (0..dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for (i, a_row) in a.iter().enumerate().take(dim) {
            for (j, a_val) in a_row.iter().enumerate().take(dim).skip(i + 1) {
                if a_val.abs() > max_val {
                    max_val = a_val.abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-12 {
            break;
        }

        // Compute rotation angle
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply Givens rotation
        let mut new_a = a.clone();
        for i in 0..dim {
            new_a[i][p] = cos_t * a[i][p] + sin_t * a[i][q];
            new_a[i][q] = -sin_t * a[i][p] + cos_t * a[i][q];
        }
        let a_tmp = new_a.clone();
        for j in 0..dim {
            new_a[p][j] = cos_t * a_tmp[p][j] + sin_t * a_tmp[q][j];
            new_a[q][j] = -sin_t * a_tmp[p][j] + cos_t * a_tmp[q][j];
        }
        a = new_a;

        // Update eigenvectors
        let mut new_v = v.clone();
        for i in 0..dim {
            new_v[i][p] = cos_t * v[i][p] + sin_t * v[i][q];
            new_v[i][q] = -sin_t * v[i][p] + cos_t * v[i][q];
        }
        v = new_v;
    }

    let eigenvalues: Vec<f64> = (0..dim).map(|i| a[i][i]).collect();
    Ok((eigenvalues, v))
}

fn transpose(matrix: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0; dim]; dim];
    for (i, m_row) in matrix.iter().enumerate().take(dim) {
        for (j, &val) in m_row.iter().enumerate().take(dim) {
            t[j][i] = val;
        }
    }
    t
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; dim]; dim];
    for (i, res_row) in result.iter_mut().enumerate().take(dim) {
        for k in 0..dim {
            let aik = a.get(i).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
            if aik.abs() < 1e-15 {
                continue;
            }
            for (j, res_val) in res_row.iter_mut().enumerate().take(dim) {
                let bkj = b.get(k).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                *res_val += aik * bkj;
            }
        }
    }
    result
}

fn compute_mse(src: &[Vec<f64>], tgt: &[Vec<f64>], rotation: &[Vec<f64>], dim: usize) -> f64 {
    let n = src.len().min(tgt.len());
    if n == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for k in 0..n {
        let mut rotated = vec![0.0; dim];
        for (i, rot_val) in rotated.iter_mut().enumerate().take(dim) {
            for (j, &r_ij) in rotation[i].iter().enumerate().take(dim) {
                *rot_val += r_ij * src[k].get(j).copied().unwrap_or(0.0);
            }
        }
        let se: f64 = rotated
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - tgt[k].get(i).copied().unwrap_or(0.0)).powi(2))
            .sum();
        total += se;
    }
    total / n as f64
}

fn compute_mean_cosine(
    src: &[Vec<f64>],
    tgt: &[Vec<f64>],
    rotation: &[Vec<f64>],
    dim: usize,
) -> f64 {
    let n = src.len().min(tgt.len());
    if n == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for k in 0..n {
        let mut rotated = vec![0.0; dim];
        for (i, rot_val) in rotated.iter_mut().enumerate().take(dim) {
            for (j, &r_ij) in rotation[i].iter().enumerate().take(dim) {
                *rot_val += r_ij * src[k].get(j).copied().unwrap_or(0.0);
            }
        }
        total += cosine_sim(&rotated, &tgt[k]);
    }
    total / n as f64
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn find_nearest_neighbors(query: &[f64], candidates: &[Vec<f64>], k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let dist: f64 = query
                .iter()
                .zip(c.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (i, dist)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.iter().take(k).map(|(idx, _)| *idx).collect()
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a set of embeddings as rows.
    fn make_embeddings(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
        // Simple deterministic pseudo-random using a linear congruential generator
        let mut state = seed;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
                    })
                    .collect()
            })
            .collect()
    }

    /// Apply a known rotation to embeddings (90-degree rotation in 2D).
    fn rotate_90_2d(embeddings: &[Vec<f64>]) -> Vec<Vec<f64>> {
        embeddings
            .iter()
            .map(|e| {
                // [x, y] -> [-y, x]
                vec![-e[1], e[0]]
            })
            .collect()
    }

    // ═══ AnchorPair tests ════════════════════════════════

    #[test]
    fn test_anchor_pair_creation() {
        let pair = AnchorPair::new(0, 1);
        assert_eq!(pair.source_idx, 0);
        assert_eq!(pair.target_idx, 1);
        assert!(pair.label.is_none());
    }

    #[test]
    fn test_anchor_pair_with_label() {
        let pair = AnchorPair::new(0, 1).with_label("cat");
        assert_eq!(pair.label, Some("cat".to_string()));
    }

    // ═══ ProcrustesConfig tests ══════════════════════════

    #[test]
    fn test_default_config() {
        let config = ProcrustesConfig::default();
        assert!(config.center);
        assert!(!config.normalize);
        assert!(config.regularization > 0.0);
    }

    // ═══ Error tests ═════════════════════════════════════

    #[test]
    fn test_no_anchors_error() {
        let aligner = ProcrustesAligner::new();
        let src = make_embeddings(10, 3, 42);
        let tgt = make_embeddings(10, 3, 99);
        let result = aligner.align(&src, &tgt, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_source_index() {
        let aligner = ProcrustesAligner::new();
        let src = make_embeddings(5, 3, 42);
        let tgt = make_embeddings(5, 3, 99);
        let anchors = vec![AnchorPair::new(10, 0)]; // 10 >= 5
        let result = aligner.align(&src, &tgt, &anchors);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_target_index() {
        let aligner = ProcrustesAligner::new();
        let src = make_embeddings(5, 3, 42);
        let tgt = make_embeddings(5, 3, 99);
        let anchors = vec![AnchorPair::new(0, 10)]; // 10 >= 5
        let result = aligner.align(&src, &tgt, &anchors);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_embeddings() {
        let aligner = ProcrustesAligner::new();
        let src: Vec<Vec<f64>> = Vec::new();
        let tgt: Vec<Vec<f64>> = Vec::new();
        let anchors = vec![AnchorPair::new(0, 0)];
        let result = aligner.align(&src, &tgt, &anchors);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_display() {
        let err = ProcrustesError::NoAnchors;
        assert!(format!("{err}").contains("anchor"));
    }

    // ═══ Alignment tests (2D rotation) ═══════════════════

    #[test]
    fn test_identity_alignment() {
        let aligner = ProcrustesAligner::new();
        let src = make_embeddings(20, 3, 42);
        let tgt = src.clone(); // identical
        let anchors: Vec<AnchorPair> = (0..10).map(|i| AnchorPair::new(i, i)).collect();
        let result = aligner.align(&src, &tgt, &anchors);
        assert!(result.is_ok());
        let res = result.expect("alignment should succeed");
        assert!(res.mse < 1e-6);
    }

    #[test]
    fn test_2d_rotation_alignment() {
        let src = make_embeddings(20, 2, 42);
        let tgt = rotate_90_2d(&src);
        let anchors: Vec<AnchorPair> = (0..10).map(|i| AnchorPair::new(i, i)).collect();

        let aligner = ProcrustesAligner::new();
        let result = aligner.align(&src, &tgt, &anchors);
        assert!(result.is_ok());
        let res = result.expect("alignment should succeed");

        // MSE should be small
        assert!(res.mse < 0.5, "MSE too high: {}", res.mse);
        // Cosine similarity should be high
        assert!(
            res.mean_cosine_similarity > 0.5,
            "Cosine too low: {}",
            res.mean_cosine_similarity
        );
    }

    #[test]
    fn test_alignment_dim() {
        let src = make_embeddings(10, 5, 42);
        let tgt = make_embeddings(10, 5, 99);
        let anchors: Vec<AnchorPair> = (0..5).map(|i| AnchorPair::new(i, i)).collect();

        let aligner = ProcrustesAligner::new();
        let result = aligner.align(&src, &tgt, &anchors).expect("should align");
        assert_eq!(result.dim, 5);
        assert_eq!(result.rotation_matrix.len(), 5);
        assert_eq!(result.rotation_matrix[0].len(), 5);
    }

    // ═══ Transform tests ═════════════════════════════════

    #[test]
    fn test_transform_preserves_dim() {
        let src = make_embeddings(10, 4, 42);
        let tgt = make_embeddings(10, 4, 99);
        let anchors: Vec<AnchorPair> = (0..5).map(|i| AnchorPair::new(i, i)).collect();
        let aligner = ProcrustesAligner::new();
        let result = aligner.align(&src, &tgt, &anchors).expect("should align");

        let transformed = result.transform(&src[0]);
        assert_eq!(transformed.len(), 4);
    }

    #[test]
    fn test_transform_batch() {
        let src = make_embeddings(10, 3, 42);
        let tgt = make_embeddings(10, 3, 99);
        let anchors: Vec<AnchorPair> = (0..5).map(|i| AnchorPair::new(i, i)).collect();
        let aligner = ProcrustesAligner::new();
        let result = aligner.align(&src, &tgt, &anchors).expect("should align");

        let batch = result.transform_batch(&src);
        assert_eq!(batch.len(), 10);
    }

    // ═══ Evaluation tests ════════════════════════════════

    #[test]
    fn test_evaluate_identity() {
        let src = make_embeddings(20, 3, 42);
        let tgt = src.clone();
        let anchors: Vec<AnchorPair> = (0..10).map(|i| AnchorPair::new(i, i)).collect();
        let eval_pairs: Vec<AnchorPair> = (10..20).map(|i| AnchorPair::new(i, i)).collect();

        let aligner = ProcrustesAligner::new();
        let result = aligner.align(&src, &tgt, &anchors).expect("should align");
        let metrics = aligner.evaluate(&result, &src, &tgt, &eval_pairs);

        assert_eq!(metrics.eval_pairs, 10);
        assert!(metrics.mse < 1e-4);
        assert!(metrics.precision_at_1 > 0.8);
    }

    #[test]
    fn test_evaluate_empty() {
        let src = make_embeddings(10, 3, 42);
        let tgt = make_embeddings(10, 3, 99);
        let anchors: Vec<AnchorPair> = (0..5).map(|i| AnchorPair::new(i, i)).collect();

        let aligner = ProcrustesAligner::new();
        let result = aligner.align(&src, &tgt, &anchors).expect("should align");
        let metrics = aligner.evaluate(&result, &src, &tgt, &[]);
        assert_eq!(metrics.eval_pairs, 0);
    }

    // ═══ Cosine similarity helper tests ══════════════════

    #[test]
    fn test_cosine_sim_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_sim(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_sim(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_sim_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_sim(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_sim_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_sim(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    // ═══ Linear algebra helper tests ═════════════════════

    #[test]
    fn test_centroid_computation() {
        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let centroid = compute_centroid(&embeddings, 2);
        assert!((centroid[0] - 2.0).abs() < 1e-10);
        assert!((centroid[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_center_embeddings_fn() {
        let embeddings = vec![vec![2.0, 4.0], vec![4.0, 6.0]];
        let centroid = vec![3.0, 5.0];
        let centered = center_embeddings(&embeddings, &centroid);
        assert!((centered[0][0] - (-1.0)).abs() < 1e-10);
        assert!((centered[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_rows_fn() {
        let embeddings = vec![vec![3.0, 4.0]];
        let normalized = normalize_rows(&embeddings);
        let norm: f64 = normalized[0].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_transpose_identity() {
        let m = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let t = transpose(&m, 2);
        assert!((t[0][0] - 1.0).abs() < 1e-10);
        assert!((t[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_mul_identity() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = mat_mul(&a, &identity, 2);
        assert!((result[0][0] - 1.0).abs() < 1e-10);
        assert!((result[0][1] - 2.0).abs() < 1e-10);
        assert!((result[1][0] - 3.0).abs() < 1e-10);
        assert!((result[1][1] - 4.0).abs() < 1e-10);
    }

    // ═══ Nearest neighbor tests ══════════════════════════

    #[test]
    fn test_find_nearest_neighbors() {
        let query = vec![0.0, 0.0];
        let candidates = vec![
            vec![10.0, 10.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
        ];
        let nn = find_nearest_neighbors(&query, &candidates, 2);
        assert_eq!(nn.len(), 2);
        // Closest should be [1,0] (idx=1) or [0,1] (idx=2)
        assert!(nn[0] == 1 || nn[0] == 2);
    }

    // ═══ Config with normalize ═══════════════════════════

    #[test]
    fn test_alignment_with_normalization() {
        let config = ProcrustesConfig {
            center: true,
            normalize: true,
            regularization: 1e-10,
        };
        let aligner = ProcrustesAligner::with_config(config);
        let src = make_embeddings(20, 3, 42);
        let tgt = src.clone();
        let anchors: Vec<AnchorPair> = (0..10).map(|i| AnchorPair::new(i, i)).collect();
        let result = aligner.align(&src, &tgt, &anchors);
        assert!(result.is_ok());
    }

    // ═══ Default aligner test ════════════════════════════

    #[test]
    fn test_default_aligner() {
        let aligner = ProcrustesAligner::default();
        let src = make_embeddings(10, 2, 1);
        let tgt = make_embeddings(10, 2, 2);
        let anchors = vec![AnchorPair::new(0, 0), AnchorPair::new(1, 1)];
        let result = aligner.align(&src, &tgt, &anchors);
        assert!(result.is_ok());
    }
}
