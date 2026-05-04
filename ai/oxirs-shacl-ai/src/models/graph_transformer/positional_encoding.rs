//! Positional encodings for graph transformer architectures.
//!
//! Two encodings:
//! - `LaplacianPE` — top-k eigenvectors of the normalised Laplacian (used by GT).
//! - `CentralityEncoding` — learnable in/out degree embeddings (used by Graphormer).

use scirs2_core::ndarray_ext::{Array1, Array2};

use super::GraphTransformerError;

/// Simple LCG-based deterministic PRNG for reproducible parameter initialisation.
pub(crate) struct DetRng {
    state: u64,
}

impl DetRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E3779B97F4A7C15,
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform float in `[low, high)`.
    pub(crate) fn uniform(&mut self, low: f64, high: f64) -> f64 {
        let u = self.next_u64() as f64 / u64::MAX as f64;
        low + u * (high - low)
    }

    /// Xavier uniform limit: `sqrt(6 / (fan_in + fan_out))`.
    pub(crate) fn xavier(&mut self, fan_in: usize, fan_out: usize) -> f64 {
        let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
        self.uniform(-limit, limit)
    }
}

// ---------------------------------------------------------------------------
// Laplacian Eigenvector PE
// ---------------------------------------------------------------------------

/// Laplacian eigenvector positional encoding.
///
/// Computes the top-`k` eigenvectors of the normalised graph Laplacian
/// `L = I - D^{-1/2} A D^{-1/2}` via power-deflation iteration.
/// The returned matrix is `[n, k]`; column `i` is eigenvector `i`.
/// Signs are random-flipped for training stability.
#[derive(Debug, Clone)]
pub struct LaplacianPE {
    /// Number of eigenvectors to extract.
    pub k: usize,
}

impl LaplacianPE {
    /// Create a new `LaplacianPE` encoder.
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Compute top-`k` Laplacian eigenvectors for adjacency matrix `adj` (`[n, n]`).
    ///
    /// Returns `[n, k]` eigenvector matrix. Uses power-deflation (a matrix deflation
    /// that subtracts the contribution of already-found eigenvectors before the next
    /// power iteration). O(n² × k × iters).
    pub fn compute(
        &self,
        adj: &Array2<f64>,
        seed: u64,
    ) -> Result<Array2<f64>, GraphTransformerError> {
        let n = adj.nrows();
        if adj.ncols() != n {
            return Err(GraphTransformerError::NonSquareAdjacency {
                rows: n,
                cols: adj.ncols(),
            });
        }
        if n == 0 {
            return Err(GraphTransformerError::EmptyGraph);
        }

        let k = self.k.min(n);
        // Compute normalised Laplacian L = I - D^{-1/2} A D^{-1/2}.
        let lap = normalised_laplacian(adj, n);

        let mut rng = DetRng::new(seed);
        let mut ev = Array2::<f64>::zeros((n, k));

        // Deflated power iteration: iterate through each eigenvector.
        let mut deflation: Vec<Array1<f64>> = Vec::with_capacity(k);

        for col in 0..k {
            // Initialise random unit vector.
            let mut v = random_unit_vec(n, &mut rng);

            // Orthogonalise against already-found eigenvectors.
            gram_schmidt_orth(&mut v, &deflation);

            // Power iteration (converges to the dominant eigenvector of the
            // deflated matrix).
            for _iter in 0..200 {
                let mut mv = matvec(&lap, &v);
                // Subtract contributions of prior eigenvectors (deflation).
                for prior in &deflation {
                    let dot: f64 = prior.iter().zip(mv.iter()).map(|(&a, &b)| a * b).sum();
                    for (x, p) in mv.iter_mut().zip(prior.iter()) {
                        *x -= dot * p;
                    }
                }
                // Normalise.
                let norm = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-14 {
                    break;
                }
                for x in mv.iter_mut() {
                    *x /= norm;
                }
                // Check convergence.
                let diff: f64 = v.iter().zip(mv.iter()).map(|(&a, &b)| (a - b).abs()).sum();
                v = mv;
                if diff < 1e-10 {
                    break;
                }
            }

            // Random sign flip for stability.
            let flip = if rng.uniform(0.0, 1.0) < 0.5 {
                -1.0_f64
            } else {
                1.0_f64
            };
            for i in 0..n {
                ev[[i, col]] = flip * v[i];
            }
            deflation.push(v);
        }

        Ok(ev)
    }
}

/// Build normalised Laplacian from adjacency matrix.
fn normalised_laplacian(adj: &Array2<f64>, n: usize) -> Array2<f64> {
    // Degree vector (sum of each row).
    let mut d_inv_sqrt = Array1::<f64>::zeros(n);
    for i in 0..n {
        let deg: f64 = (0..n).map(|j| adj[[i, j]].abs()).sum();
        d_inv_sqrt[i] = if deg > 1e-12 { 1.0 / deg.sqrt() } else { 0.0 };
    }

    // L = I - D^{-1/2} A D^{-1/2}
    let mut lap = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        lap[[i, i]] = 1.0;
        for j in 0..n {
            lap[[i, j]] -= d_inv_sqrt[i] * adj[[i, j]] * d_inv_sqrt[j];
        }
    }
    lap
}

/// Matrix-vector product: `A v`.
fn matvec(a: &Array2<f64>, v: &Array1<f64>) -> Array1<f64> {
    let n = v.len();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..n {
            s += a[[i, j]] * v[j];
        }
        out[i] = s;
    }
    out
}

/// Create a random unit vector of length `n`.
fn random_unit_vec(n: usize, rng: &mut DetRng) -> Array1<f64> {
    let mut v = Array1::<f64>::zeros(n);
    for i in 0..n {
        v[i] = rng.uniform(-1.0, 1.0);
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-14 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    } else {
        v[0] = 1.0; // fallback
    }
    v
}

/// Gram-Schmidt orthogonalisation of `v` against a list of unit vectors.
fn gram_schmidt_orth(v: &mut Array1<f64>, basis: &[Array1<f64>]) {
    for b in basis {
        let dot: f64 = b.iter().zip(v.iter()).map(|(&a, &x)| a * x).sum();
        for (x, bi) in v.iter_mut().zip(b.iter()) {
            *x -= dot * bi;
        }
    }
    // Re-normalise.
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-14 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Centrality Encoding
// ---------------------------------------------------------------------------

/// Degree-centrality positional encoding (used by Graphormer).
///
/// Maintains two learnable embedding tables: one for in-degree, one for
/// out-degree. Each node's encoding is `in_emb[in_deg] + out_emb[out_deg]`.
#[derive(Debug, Clone)]
pub struct CentralityEncoding {
    /// Maximum degree index. Degrees > `max_degree` are clamped to `max_degree`.
    pub max_degree: usize,
    /// Embedding dimension (must match the model's `hidden_dim`).
    pub hidden_dim: usize,
    /// In-degree embedding table: `[max_degree+1, hidden_dim]`.
    pub in_degree_emb: Array2<f64>,
    /// Out-degree embedding table: `[max_degree+1, hidden_dim]`.
    pub out_degree_emb: Array2<f64>,
}

impl CentralityEncoding {
    /// Create a new centrality encoding with Xavier-uniform initialised tables.
    pub fn new(max_degree: usize, hidden_dim: usize, seed: u64) -> Self {
        let rows = max_degree + 1;
        let mut rng = DetRng::new(seed);
        let mut in_degree_emb = Array2::<f64>::zeros((rows, hidden_dim));
        let mut out_degree_emb = Array2::<f64>::zeros((rows, hidden_dim));
        for i in 0..rows {
            for j in 0..hidden_dim {
                in_degree_emb[[i, j]] = rng.xavier(rows, hidden_dim);
                out_degree_emb[[i, j]] = rng.xavier(rows, hidden_dim);
            }
        }
        Self {
            max_degree,
            hidden_dim,
            in_degree_emb,
            out_degree_emb,
        }
    }

    /// Compute centrality encodings for a batch of nodes.
    ///
    /// Returns `[n, hidden_dim]` matrix where each row is
    /// `in_emb[in_deg[i]] + out_emb[out_deg[i]]`.
    pub fn encode(&self, in_degrees: &[usize], out_degrees: &[usize]) -> Array2<f64> {
        let n = in_degrees.len().max(out_degrees.len());
        let mut out = Array2::<f64>::zeros((n, self.hidden_dim));
        for i in 0..n {
            let in_d = in_degrees.get(i).copied().unwrap_or(0).min(self.max_degree);
            let out_d = out_degrees
                .get(i)
                .copied()
                .unwrap_or(0)
                .min(self.max_degree);
            for j in 0..self.hidden_dim {
                out[[i, j]] = self.in_degree_emb[[in_d, j]] + self.out_degree_emb[[out_d, j]];
            }
        }
        out
    }

    /// Backward pass: update embedding tables from upstream gradient.
    ///
    /// `grad`: `[n, hidden_dim]` gradient w.r.t. output of `encode`.
    /// `in_degrees`, `out_degrees`: the same indices used in the forward pass.
    /// `lr`: learning rate.
    pub fn backward(
        &mut self,
        grad: &Array2<f64>,
        in_degrees: &[usize],
        out_degrees: &[usize],
        lr: f64,
    ) {
        let n = grad.nrows();
        for i in 0..n {
            let in_d = in_degrees.get(i).copied().unwrap_or(0).min(self.max_degree);
            let out_d = out_degrees
                .get(i)
                .copied()
                .unwrap_or(0)
                .min(self.max_degree);
            for j in 0..self.hidden_dim {
                self.in_degree_emb[[in_d, j]] -= lr * grad[[i, j]];
                self.out_degree_emb[[out_d, j]] -= lr * grad[[i, j]];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_adj(n: usize) -> Array2<f64> {
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, (i + 1) % n]] = 1.0;
            a[[(i + 1) % n, i]] = 1.0;
        }
        a
    }

    #[test]
    fn test_laplacian_pe_shape() {
        let adj = ring_adj(6);
        let pe = LaplacianPE::new(3);
        let ev = pe.compute(&adj, 42).expect("eigenvectors");
        assert_eq!(ev.nrows(), 6);
        assert_eq!(ev.ncols(), 3);
    }

    #[test]
    fn test_laplacian_pe_orthogonality() {
        let adj = ring_adj(8);
        let pe = LaplacianPE::new(4);
        let ev = pe.compute(&adj, 99).expect("eigenvectors");
        let n = ev.nrows();
        let k = ev.ncols();
        for c1 in 0..k {
            for c2 in (c1 + 1)..k {
                let dot: f64 = (0..n).map(|i| ev[[i, c1]] * ev[[i, c2]]).sum();
                assert!(
                    dot.abs() < 0.15,
                    "columns {c1} and {c2} not approx orthogonal: dot={dot:.6}"
                );
            }
        }
    }

    #[test]
    fn test_centrality_encoding_varies() {
        let enc = CentralityEncoding::new(10, 8, 7);
        // degree-2 node vs. degree-4 node
        let enc2 = enc.encode(&[2], &[2]);
        let enc4 = enc.encode(&[4], &[4]);
        let diff: f64 = (0..8).map(|j| (enc2[[0, j]] - enc4[[0, j]]).abs()).sum();
        assert!(diff > 1e-6, "degree-2 and degree-4 encodings should differ");
    }

    #[test]
    fn test_centrality_encoding_shape() {
        let enc = CentralityEncoding::new(8, 16, 1);
        let out = enc.encode(&[0, 2, 4, 8], &[1, 3, 5, 8]);
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 16);
    }

    #[test]
    fn test_laplacian_pe_non_square_error() {
        let bad = Array2::<f64>::zeros((3, 4));
        let pe = LaplacianPE::new(2);
        let result = pe.compute(&bad, 0);
        assert!(result.is_err());
    }
}
