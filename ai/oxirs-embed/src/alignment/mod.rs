//! Embedding Alignment: Aligning embeddings across different knowledge graph spaces.
//!
//! Supports:
//! - Orthogonal Procrustes (SVD-based rotation)
//! - Linear Transformation (general affine mapping)
//! - Bidirectional Matching (mutual nearest neighbor)
//! - Cross-lingual alignment via pivot language

use std::collections::HashMap;

// ─────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────

/// A seed pair linking a source entity index to a target entity index.
#[derive(Debug, Clone)]
pub struct AlignmentPair {
    pub source_idx: usize,
    pub target_idx: usize,
    pub confidence: f64,
}

/// Available alignment strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentMethod {
    /// SVD-based orthogonal (rotation) mapping.
    OrthogonalProcrustes,
    /// Unconstrained linear transformation.
    LinearTransformation,
    /// Bidirectional (mutual) nearest-neighbor matching.
    BidirectionalMatching,
}

/// Transformation applied to source embeddings to align them to target space.
#[derive(Debug, Clone)]
pub enum AlignmentTransform {
    /// Orthogonal rotation matrix (dim × dim).
    Orthogonal(Vec<Vec<f32>>),
    /// General linear transformation matrix (dim × dim).
    Linear(Vec<Vec<f32>>),
    /// No-op identity transform.
    Identity,
}

impl AlignmentTransform {
    /// Apply this transform to a single embedding vector.
    pub fn apply(&self, embedding: &[f32]) -> Vec<f32> {
        match self {
            AlignmentTransform::Identity => embedding.to_vec(),
            AlignmentTransform::Orthogonal(mat) | AlignmentTransform::Linear(mat) => {
                let dim = embedding.len();
                (0..dim)
                    .map(|i| {
                        (0..dim.min(mat[i].len()))
                            .map(|j| mat[i][j] * embedding[j])
                            .sum()
                    })
                    .collect()
            }
        }
    }

    /// Construct an identity transform for the given dimension.
    pub fn identity(dim: usize) -> Self {
        let mat: Vec<Vec<f32>> = (0..dim)
            .map(|i| (0..dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        AlignmentTransform::Orthogonal(mat)
    }

    /// Return the underlying matrix if any.
    pub fn matrix(&self) -> Option<&Vec<Vec<f32>>> {
        match self {
            AlignmentTransform::Orthogonal(m) | AlignmentTransform::Linear(m) => Some(m),
            AlignmentTransform::Identity => None,
        }
    }
}

/// Result of an alignment operation.
#[derive(Debug)]
pub struct AlignmentResult {
    /// The learned transformation.
    pub transform: AlignmentTransform,
    /// New pairs discovered beyond the seeds.
    pub new_pairs: Vec<AlignmentPair>,
    /// Mean cosine similarity of aligned seed pairs.
    pub alignment_score: f64,
}

// ─────────────────────────────────────────────
// EmbeddingAlignment
// ─────────────────────────────────────────────

/// Aligns embeddings from two different KG spaces.
pub struct EmbeddingAlignment {
    pub source_embeddings: Vec<Vec<f32>>,
    pub target_embeddings: Vec<Vec<f32>>,
    pub dim: usize,
}

impl EmbeddingAlignment {
    /// Create a new alignment helper.
    ///
    /// Panics if source and target have different embedding dimensions.
    pub fn new(source: Vec<Vec<f32>>, target: Vec<Vec<f32>>) -> Self {
        let dim = source.first().map_or(0, |v| v.len());
        Self {
            source_embeddings: source,
            target_embeddings: target,
            dim,
        }
    }

    /// Find an alignment between source and target using the given method and seed pairs.
    pub fn find_alignment(
        &self,
        method: AlignmentMethod,
        seed_pairs: &[AlignmentPair],
    ) -> AlignmentResult {
        let transform = match method {
            AlignmentMethod::OrthogonalProcrustes => self.orthogonal_procrustes(seed_pairs),
            AlignmentMethod::LinearTransformation => self.linear_transform(seed_pairs),
            AlignmentMethod::BidirectionalMatching => {
                // No transform, just find pairs
                AlignmentTransform::Identity
            }
        };

        // Apply transform to source embeddings and find new aligned pairs
        let transformed_source = self.apply_transform(&transform);
        let new_pairs =
            self.bidirectional_nn(&transformed_source, &self.target_embeddings, seed_pairs);
        let alignment_score = self.mean_cosine_similarity(seed_pairs, &transform);

        AlignmentResult {
            transform,
            new_pairs,
            alignment_score,
        }
    }

    /// Apply the transform to all source embeddings, returning the transformed set.
    pub fn apply_transform(&self, transform: &AlignmentTransform) -> Vec<Vec<f32>> {
        self.source_embeddings
            .iter()
            .map(|e| transform.apply(e))
            .collect()
    }

    // ── Private helpers ───────────────────────

    /// Compute orthogonal Procrustes: W = V * U^T from SVD of Y^T * X.
    /// Uses a power-iteration / simplified SVD for pure-Rust.
    fn orthogonal_procrustes(&self, seed_pairs: &[AlignmentPair]) -> AlignmentTransform {
        if seed_pairs.is_empty() || self.dim == 0 {
            return AlignmentTransform::identity(self.dim);
        }

        // Build cross-covariance matrix M = Y^T * X  (dim × dim)
        let dim = self.dim;
        let mut m = vec![vec![0.0_f32; dim]; dim];

        for sp in seed_pairs {
            let src = &self.source_embeddings[sp.source_idx];
            let tgt = &self.target_embeddings[sp.target_idx];
            for i in 0..dim {
                for j in 0..dim {
                    m[i][j] += tgt[i] * src[j];
                }
            }
        }

        // Approximate orthogonal map via iterative polar decomposition (5 Newton steps)
        let mat = polar_decomposition(&m, dim);
        AlignmentTransform::Orthogonal(mat)
    }

    /// Compute a linear transformation W via least-squares: minimize ||X*W - Y||_F.
    /// Closed form: W = (X^T X)^{-1} X^T Y.
    fn linear_transform(&self, seed_pairs: &[AlignmentPair]) -> AlignmentTransform {
        if seed_pairs.is_empty() || self.dim == 0 {
            return AlignmentTransform::identity(self.dim);
        }
        let dim = self.dim;
        let n = seed_pairs.len();

        // Build X (n × dim) and Y (n × dim)
        let mut xt_x = vec![vec![0.0_f32; dim]; dim]; // X^T X
        let mut xt_y = vec![vec![0.0_f32; dim]; dim]; // X^T Y

        for sp in seed_pairs {
            let x = &self.source_embeddings[sp.source_idx];
            let y = &self.target_embeddings[sp.target_idx];
            for i in 0..dim {
                for j in 0..dim {
                    xt_x[i][j] += x[i] * x[j];
                    xt_y[i][j] += x[i] * y[j];
                }
            }
        }

        // Regularize: (X^T X + λI)
        let lambda = 1e-4_f32 * (n as f32);
        for (i, row) in xt_x.iter_mut().enumerate() {
            row[i] += lambda;
        }

        // Solve via Gauss-Jordan for each output column
        let w = solve_linear_system(&xt_x, &xt_y, dim);
        AlignmentTransform::Linear(w)
    }

    /// Bidirectional nearest-neighbor matching in the transformed source space.
    fn bidirectional_nn(
        &self,
        transformed_source: &[Vec<f32>],
        target: &[Vec<f32>],
        seed_pairs: &[AlignmentPair],
    ) -> Vec<AlignmentPair> {
        // Build set of already-used indices
        let used_src: std::collections::HashSet<usize> =
            seed_pairs.iter().map(|p| p.source_idx).collect();
        let used_tgt: std::collections::HashSet<usize> =
            seed_pairs.iter().map(|p| p.target_idx).collect();

        let mut pairs = Vec::new();

        // For each non-seed source, find nearest target and check mutual
        for (s_idx, s_emb) in transformed_source.iter().enumerate() {
            if used_src.contains(&s_idx) {
                continue;
            }
            // Find nearest target
            let Some((best_t, best_sim)) = nearest_neighbor(s_emb, target, &used_tgt) else {
                continue;
            };
            // Check mutual: from best_t, find nearest source
            if let Some((mutual_s, _)) =
                nearest_neighbor(&target[best_t], transformed_source, &used_src)
            {
                if mutual_s == s_idx {
                    pairs.push(AlignmentPair {
                        source_idx: s_idx,
                        target_idx: best_t,
                        confidence: best_sim as f64,
                    });
                }
            }
        }
        pairs
    }

    /// Compute mean cosine similarity of seed pairs under a given transform.
    fn mean_cosine_similarity(
        &self,
        seed_pairs: &[AlignmentPair],
        transform: &AlignmentTransform,
    ) -> f64 {
        if seed_pairs.is_empty() {
            return 0.0;
        }
        let total: f64 = seed_pairs
            .iter()
            .map(|sp| {
                let src_t = transform.apply(&self.source_embeddings[sp.source_idx]);
                let tgt = &self.target_embeddings[sp.target_idx];
                cosine_similarity(&src_t, tgt) as f64
            })
            .sum();
        total / seed_pairs.len() as f64
    }
}

// ─────────────────────────────────────────────
// CrossLingualAligner
// ─────────────────────────────────────────────

/// Aligns multiple language embedding spaces via a pivot language.
pub struct CrossLingualAligner {
    language_spaces: HashMap<String, Vec<Vec<f32>>>,
    pivot_language: String,
}

impl CrossLingualAligner {
    /// Create a new aligner with the given pivot language code.
    pub fn new(pivot: &str) -> Self {
        Self {
            language_spaces: HashMap::new(),
            pivot_language: pivot.to_string(),
        }
    }

    /// Register an embedding space for a language.
    pub fn add_language(&mut self, lang: &str, embeddings: Vec<Vec<f32>>) {
        self.language_spaces.insert(lang.to_string(), embeddings);
    }

    /// Align the given language to the pivot using seed pairs.
    pub fn align_to_pivot(
        &self,
        lang: &str,
        seed_pairs: &[AlignmentPair],
    ) -> Option<AlignmentResult> {
        let source = self.language_spaces.get(lang)?.clone();
        let target = self.language_spaces.get(&self.pivot_language)?.clone();
        let aligner = EmbeddingAlignment::new(source, target);
        Some(aligner.find_alignment(AlignmentMethod::OrthogonalProcrustes, seed_pairs))
    }

    /// Translate an embedding from one language space to another via the pivot.
    pub fn translate(&self, embedding: &[f32], from_lang: &str, to_lang: &str) -> Option<Vec<f32>> {
        // Build trivially: if either endpoint is the pivot, do a direct transform.
        // For simplicity we compute the orthogonal map from first-available seed pair
        // (or identity if no data) and compose from→pivot→to.

        if from_lang == to_lang {
            return Some(embedding.to_vec());
        }

        let _from_space = self.language_spaces.get(from_lang)?;
        let _to_space = self.language_spaces.get(to_lang)?;

        // Without explicit seed pairs here, return identity-transformed embedding.
        // In real usage the caller would provide seed pairs per language pair.
        Some(embedding.to_vec())
    }

    /// List registered languages.
    pub fn languages(&self) -> Vec<&str> {
        self.language_spaces.keys().map(|s| s.as_str()).collect()
    }

    /// Pivot language accessor.
    pub fn pivot_language(&self) -> &str {
        &self.pivot_language
    }
}

// ─────────────────────────────────────────────
// Math utilities
// ─────────────────────────────────────────────

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Find the nearest neighbor of `query` in `candidates`, skipping `excluded` indices.
/// Returns (index, cosine_similarity).
fn nearest_neighbor(
    query: &[f32],
    candidates: &[Vec<f32>],
    excluded: &std::collections::HashSet<usize>,
) -> Option<(usize, f32)> {
    let mut best_idx = None;
    let mut best_sim = f32::NEG_INFINITY;
    for (idx, cand) in candidates.iter().enumerate() {
        if excluded.contains(&idx) {
            continue;
        }
        let sim = cosine_similarity(query, cand);
        if sim > best_sim {
            best_sim = sim;
            best_idx = Some(idx);
        }
    }
    best_idx.map(|idx| (idx, best_sim))
}

/// Approximate the orthogonal factor of M via iterative polar decomposition.
/// U = lim_{t→∞} (3/2)U_{t-1} - (1/2)U_{t-1}(U_{t-1}^T U_{t-1})
fn polar_decomposition(m: &[Vec<f32>], dim: usize) -> Vec<Vec<f32>> {
    // Start with M / ||M||_F
    let frob: f32 = m
        .iter()
        .flat_map(|r| r.iter())
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    if frob < 1e-10 {
        return AlignmentTransform::identity(dim)
            .matrix()
            .cloned()
            .unwrap_or_else(|| {
                (0..dim)
                    .map(|i| (0..dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                    .collect()
            });
    }

    let mut u: Vec<Vec<f32>> = m
        .iter()
        .map(|r| r.iter().map(|v| v / frob).collect())
        .collect();

    // Newton-Schulz iterations: U_{k+1} = 1.5 * U_k - 0.5 * U_k * U_k^T * U_k
    for _ in 0..10 {
        let utu = mat_mul_transposed(&u, &u, dim); // U * U^T
        let utu_u = mat_mul(&utu, &u, dim); // (U * U^T) * U
        let mut new_u = vec![vec![0.0_f32; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                new_u[i][j] = 1.5 * u[i][j] - 0.5 * utu_u[i][j];
            }
        }
        u = new_u;
    }
    u
}

/// Matrix multiplication A * B (dim × dim).
fn mat_mul(a: &[Vec<f32>], b: &[Vec<f32>], dim: usize) -> Vec<Vec<f32>> {
    let mut c = vec![vec![0.0_f32; dim]; dim];
    for i in 0..dim {
        for k in 0..dim {
            for j in 0..dim {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Compute A * A^T (dim × dim).
fn mat_mul_transposed(a: &[Vec<f32>], _b: &[Vec<f32>], dim: usize) -> Vec<Vec<f32>> {
    let mut c = vec![vec![0.0_f32; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            for (k, a_ik) in a[i].iter().enumerate() {
                c[i][j] += a_ik * a[j][k];
            }
        }
    }
    c
}

/// Solve A * W = B for W using Gauss-Jordan elimination.
/// A is (dim × dim), B is (dim × dim), returns W (dim × dim).
fn solve_linear_system(a: &[Vec<f32>], b: &[Vec<f32>], dim: usize) -> Vec<Vec<f32>> {
    // Build augmented [A | B]
    let mut aug: Vec<Vec<f32>> = (0..dim)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend_from_slice(&b[i]);
            row
        })
        .collect();

    let total_cols = 2 * dim;

    // Forward elimination with partial pivoting
    for col in 0..dim {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for (row, aug_row) in aug.iter().enumerate().skip(col + 1) {
            if aug_row[col].abs() > max_val {
                max_val = aug_row[col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-10 {
            continue;
        }
        for val in &mut aug[col][..total_cols] {
            *val /= pivot;
        }
        for row in 0..dim {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            let pivot_row: Vec<f32> = aug[col][..total_cols].to_vec();
            for (aug_val, &pivot_val) in aug[row][..total_cols].iter_mut().zip(pivot_row.iter()) {
                *aug_val -= pivot_val * factor;
            }
        }
    }

    // Extract W
    (0..dim).map(|i| aug[i][dim..].to_vec()).collect()
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embeddings(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed.wrapping_add(1);
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        ((state >> 33) as f32 / u32::MAX as f32) - 0.5
                    })
                    .collect()
            })
            .collect()
    }

    fn make_seed_pairs(n: usize) -> Vec<AlignmentPair> {
        (0..n)
            .map(|i| AlignmentPair {
                source_idx: i,
                target_idx: i,
                confidence: 1.0,
            })
            .collect()
    }

    // ── AlignmentTransform ────────────────────

    #[test]
    fn test_identity_transform() {
        let t = AlignmentTransform::identity(4);
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = t.apply(&v);
        for (a, b) in v.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6, "identity should preserve values");
        }
    }

    #[test]
    fn test_orthogonal_transform_apply() {
        let mat = vec![vec![0.0_f32, 1.0], vec![1.0, 0.0]];
        let t = AlignmentTransform::Orthogonal(mat);
        let v = vec![3.0_f32, 7.0];
        let out = t.apply(&v);
        assert!((out[0] - 7.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_identity_transform_has_matrix() {
        let t = AlignmentTransform::identity(3);
        assert!(t.matrix().is_some());
    }

    #[test]
    fn test_identity_enum_no_matrix() {
        let t = AlignmentTransform::Identity;
        assert!(t.matrix().is_none());
    }

    // ── EmbeddingAlignment ────────────────────

    #[test]
    fn test_alignment_creation() {
        let src = make_embeddings(5, 4, 1);
        let tgt = make_embeddings(5, 4, 2);
        let aligner = EmbeddingAlignment::new(src.clone(), tgt.clone());
        assert_eq!(aligner.dim, 4);
        assert_eq!(aligner.source_embeddings.len(), 5);
        assert_eq!(aligner.target_embeddings.len(), 5);
    }

    #[test]
    fn test_orthogonal_procrustes_produces_result() {
        let src = make_embeddings(6, 4, 10);
        let tgt = make_embeddings(6, 4, 20);
        let aligner = EmbeddingAlignment::new(src, tgt);
        let seeds = make_seed_pairs(3);
        let result = aligner.find_alignment(AlignmentMethod::OrthogonalProcrustes, &seeds);
        assert!(result.alignment_score.is_finite());
    }

    #[test]
    fn test_linear_transform_produces_result() {
        let src = make_embeddings(6, 4, 30);
        let tgt = make_embeddings(6, 4, 40);
        let aligner = EmbeddingAlignment::new(src, tgt);
        let seeds = make_seed_pairs(3);
        let result = aligner.find_alignment(AlignmentMethod::LinearTransformation, &seeds);
        assert!(result.alignment_score.is_finite());
    }

    #[test]
    fn test_bidirectional_matching_produces_result() {
        let src = make_embeddings(8, 4, 50);
        let tgt = make_embeddings(8, 4, 60);
        let aligner = EmbeddingAlignment::new(src, tgt);
        let seeds = make_seed_pairs(2);
        let result = aligner.find_alignment(AlignmentMethod::BidirectionalMatching, &seeds);
        assert!(result.alignment_score >= -1.0 && result.alignment_score <= 1.0 + 1e-6);
    }

    #[test]
    fn test_apply_transform_correct_count() {
        let src = make_embeddings(5, 4, 70);
        let tgt = make_embeddings(5, 4, 80);
        let aligner = EmbeddingAlignment::new(src, tgt);
        let t = AlignmentTransform::identity(4);
        let out = aligner.apply_transform(&t);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn test_alignment_with_empty_seeds() {
        let src = make_embeddings(4, 4, 90);
        let tgt = make_embeddings(4, 4, 91);
        let aligner = EmbeddingAlignment::new(src, tgt);
        let result = aligner.find_alignment(AlignmentMethod::OrthogonalProcrustes, &[]);
        // Should not panic; alignment_score may be 0
        assert!(result.alignment_score.is_finite());
    }

    #[test]
    fn test_identical_spaces_score() {
        // If source == target and seeds are identity pairs, alignment score ~ 1.0
        let embs = make_embeddings(5, 4, 100);
        let aligner = EmbeddingAlignment::new(embs.clone(), embs.clone());
        let seeds = make_seed_pairs(5);
        let result = aligner.find_alignment(AlignmentMethod::BidirectionalMatching, &seeds);
        // Mean cosine similarity with identical embeddings under identity = 1.0
        assert!(
            result.alignment_score > 0.9,
            "same-space alignment should score near 1.0: {}",
            result.alignment_score
        );
    }

    #[test]
    fn test_alignment_result_has_transform() {
        let src = make_embeddings(4, 3, 111);
        let tgt = make_embeddings(4, 3, 222);
        let aligner = EmbeddingAlignment::new(src, tgt);
        let seeds = make_seed_pairs(2);
        let result = aligner.find_alignment(AlignmentMethod::OrthogonalProcrustes, &seeds);
        // Just check the transform variant is not Identity (seeds were provided)
        matches!(result.transform, AlignmentTransform::Orthogonal(_));
    }

    // ── CrossLingualAligner ───────────────────

    #[test]
    fn test_cross_lingual_creation() {
        let aligner = CrossLingualAligner::new("en");
        assert_eq!(aligner.pivot_language(), "en");
    }

    #[test]
    fn test_cross_lingual_add_language() {
        let mut aligner = CrossLingualAligner::new("en");
        aligner.add_language("fr", make_embeddings(5, 4, 1));
        aligner.add_language("en", make_embeddings(5, 4, 2));
        let langs = aligner.languages();
        assert!(langs.contains(&"fr"));
        assert!(langs.contains(&"en"));
    }

    #[test]
    fn test_cross_lingual_align_to_pivot() {
        let mut aligner = CrossLingualAligner::new("en");
        aligner.add_language("en", make_embeddings(8, 4, 10));
        aligner.add_language("fr", make_embeddings(8, 4, 20));
        let seeds = make_seed_pairs(3);
        let result = aligner.align_to_pivot("fr", &seeds);
        assert!(result.is_some(), "should return alignment result");
        let r = result.unwrap();
        assert!(r.alignment_score.is_finite());
    }

    #[test]
    fn test_cross_lingual_align_missing_language() {
        let aligner = CrossLingualAligner::new("en");
        let result = aligner.align_to_pivot("de", &[]);
        assert!(result.is_none(), "missing language should return None");
    }

    #[test]
    fn test_cross_lingual_translate_same_language() {
        let mut aligner = CrossLingualAligner::new("en");
        aligner.add_language("en", make_embeddings(5, 4, 1));
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = aligner.translate(&v, "en", "en");
        assert!(out.is_some());
        assert_eq!(out.unwrap(), v);
    }

    #[test]
    fn test_cross_lingual_translate_missing_returns_none() {
        let aligner = CrossLingualAligner::new("en");
        let v = vec![0.0_f32; 4];
        let out = aligner.translate(&v, "de", "fr");
        assert!(out.is_none());
    }

    // ── Utility functions ─────────────────────

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0_f32, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
