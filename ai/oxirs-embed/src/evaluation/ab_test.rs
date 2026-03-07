//! A/B Testing Framework for Embedding Model Comparison
//!
//! Provides statistical significance testing to determine which embedding
//! model performs better on a given evaluation metric.
//!
//! Supported statistical tests:
//! - Student's paired t-test (parametric, assumes normality)
//! - Bootstrap permutation test (non-parametric)
//! - Wilcoxon signed-rank test (non-parametric, distribution-free)
//!
//! Effect size is measured using Cohen's d for interpretability.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Metric used to compare embedding models
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbedMetric {
    /// Mean Reciprocal Rank on link prediction
    MeanReciprocalRank,
    /// Hits@K on knowledge graph completion
    HitsAtK(usize),
    /// Average cosine similarity of known-similar pairs
    SimilarityScore,
    /// Clustering quality via silhouette coefficient
    SilhouetteScore,
    /// Linear classification accuracy on node labels
    ClassificationAccuracy,
    /// Custom named metric
    Custom(String),
}

impl std::fmt::Display for EmbedMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedMetric::MeanReciprocalRank => write!(f, "MRR"),
            EmbedMetric::HitsAtK(k) => write!(f, "Hits@{}", k),
            EmbedMetric::SimilarityScore => write!(f, "SimilarityScore"),
            EmbedMetric::SilhouetteScore => write!(f, "SilhouetteScore"),
            EmbedMetric::ClassificationAccuracy => write!(f, "ClassificationAccuracy"),
            EmbedMetric::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Evaluation result for a single model on a given metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvalResult {
    /// Identifier for the model being evaluated
    pub model_id: String,
    /// Metric being measured
    pub metric: EmbedMetric,
    /// Per-sample scores (e.g., per-query MRR, per-entity accuracy)
    pub scores: Vec<f64>,
    /// Arithmetic mean of scores
    pub mean: f64,
    /// Standard deviation of scores
    pub std_dev: f64,
    /// Number of samples
    pub sample_count: usize,
}

impl ModelEvalResult {
    /// Create a new evaluation result, computing statistics automatically
    pub fn new(model_id: String, metric: EmbedMetric, scores: Vec<f64>) -> Result<Self> {
        if scores.is_empty() {
            return Err(anyhow!("scores must not be empty for model '{}'", model_id));
        }

        let mean = stats_mean(&scores);
        let std_dev = stats_std_dev(&scores, mean);
        let sample_count = scores.len();

        Ok(Self {
            model_id,
            metric,
            scores,
            mean,
            std_dev,
            sample_count,
        })
    }

    /// Compute a two-sided confidence interval for the mean.
    ///
    /// Uses the t-distribution approximation.
    /// `alpha` is the significance level (e.g., 0.05 for 95% CI).
    ///
    /// Returns (lower_bound, upper_bound).
    pub fn confidence_interval(&self, alpha: f64) -> (f64, f64) {
        let n = self.sample_count as f64;
        if n < 2.0 {
            return (self.mean, self.mean);
        }
        let se = self.std_dev / n.sqrt();
        // t critical value approximation using normal distribution for large n,
        // or a simple lookup for common alpha values
        let t_crit = t_critical_value(n as usize - 1, alpha / 2.0);
        let margin = t_crit * se;
        (self.mean - margin, self.mean + margin)
    }
}

/// Statistical test to use for comparing two models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatTest {
    /// Student's paired t-test (assumes paired samples from same test set)
    TTest,
    /// Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
    WilcoxonSignedRank,
    /// Bootstrap permutation test (model-free, most general)
    Bootstrap {
        /// Number of permutation iterations
        n_permutations: usize,
        /// Random seed for reproducibility
        seed: u64,
    },
}

/// Result of an A/B test comparing two embedding models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTestResult {
    /// Identifier of model A
    pub model_a: String,
    /// Identifier of model B
    pub model_b: String,
    /// Metric being compared
    pub metric: EmbedMetric,
    /// p-value from the statistical test
    pub p_value: f64,
    /// Effect size (Cohen's d): magnitude of the difference
    pub effect_size: f64,
    /// Whether the result is statistically significant (p_value < alpha)
    pub is_significant: bool,
    /// Significance level used
    pub alpha: f64,
    /// The winning model (None if result is not significant)
    pub winner: Option<String>,
    /// Mean score for model A
    pub mean_a: f64,
    /// Mean score for model B
    pub mean_b: f64,
    /// 95% confidence interval for model A mean
    pub ci_a: (f64, f64),
    /// 95% confidence interval for model B mean
    pub ci_b: (f64, f64),
    /// Which statistical test was used
    pub test_used: StatTest,
}

impl AbTestResult {
    /// Returns a human-readable summary of the test result
    pub fn summary(&self) -> String {
        let sig_str = if self.is_significant {
            "significant"
        } else {
            "not significant"
        };
        let winner_str = match &self.winner {
            Some(w) => format!("Winner: {}", w),
            None => "No clear winner".to_string(),
        };
        format!(
            "A/B Test [{}] ({}) -- {} vs {}: p={:.4}, effect={:.3} ({}). {}",
            self.metric,
            self.test_used.name(),
            self.model_a,
            self.model_b,
            self.p_value,
            self.effect_size,
            sig_str,
            winner_str
        )
    }
}

impl StatTest {
    fn name(&self) -> &'static str {
        match self {
            StatTest::TTest => "paired t-test",
            StatTest::WilcoxonSignedRank => "Wilcoxon signed-rank",
            StatTest::Bootstrap { .. } => "bootstrap permutation",
        }
    }
}

/// Runner for A/B testing experiments
pub struct AbTestRunner {
    /// Significance level (default: 0.05)
    alpha: f64,
    /// Statistical test to use
    test: StatTest,
}

impl Default for AbTestRunner {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            test: StatTest::TTest,
        }
    }
}

impl AbTestRunner {
    /// Create a new A/B test runner with default settings (t-test, alpha=0.05)
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the significance level (e.g., 0.05 for 95% confidence)
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the statistical test to use
    pub fn with_test(mut self, test: StatTest) -> Self {
        self.test = test;
        self
    }

    /// Compare two models on the same evaluation metric.
    ///
    /// The scores must be paired (same test samples evaluated by both models).
    /// If scores have different lengths, only the common prefix is used.
    pub fn compare(
        &self,
        result_a: &ModelEvalResult,
        result_b: &ModelEvalResult,
    ) -> Result<AbTestResult> {
        if result_a.metric != result_b.metric {
            return Err(anyhow!(
                "Cannot compare models on different metrics: {:?} vs {:?}",
                result_a.metric,
                result_b.metric
            ));
        }
        if result_a.scores.is_empty() || result_b.scores.is_empty() {
            return Err(anyhow!("Both models must have non-empty scores"));
        }

        // Use the minimum length for paired comparison
        let n = result_a.scores.len().min(result_b.scores.len());
        let scores_a = &result_a.scores[..n];
        let scores_b = &result_b.scores[..n];

        let p_value = match &self.test {
            StatTest::TTest => self.t_test_paired(scores_a, scores_b)?,
            StatTest::WilcoxonSignedRank => self.wilcoxon_signed_rank(scores_a, scores_b)?,
            StatTest::Bootstrap {
                n_permutations,
                seed,
            } => self.bootstrap_test(scores_a, scores_b, *n_permutations, *seed)?,
        };

        let effect_size = cohens_d(scores_a, scores_b);
        let is_significant = p_value < self.alpha;

        let mean_a = stats_mean(scores_a);
        let mean_b = stats_mean(scores_b);

        // Determine winner (higher mean is better for most metrics)
        let winner = if is_significant {
            if mean_a >= mean_b {
                Some(result_a.model_id.clone())
            } else {
                Some(result_b.model_id.clone())
            }
        } else {
            None
        };

        let ci_a = result_a.confidence_interval(self.alpha);
        let ci_b = result_b.confidence_interval(self.alpha);

        Ok(AbTestResult {
            model_a: result_a.model_id.clone(),
            model_b: result_b.model_id.clone(),
            metric: result_a.metric.clone(),
            p_value,
            effect_size,
            is_significant,
            alpha: self.alpha,
            winner,
            mean_a,
            mean_b,
            ci_a,
            ci_b,
            test_used: self.test.clone(),
        })
    }

    /// Compare multiple models and produce a ranking by mean score.
    ///
    /// Returns a list of (model_id, mean_score, is_significantly_best) tuples,
    /// sorted by mean score descending.
    pub fn rank_models(&self, results: &[ModelEvalResult]) -> Result<Vec<(String, f64, bool)>> {
        if results.is_empty() {
            return Err(anyhow!("Need at least one model to rank"));
        }

        // Verify all metrics are the same
        let metric = &results[0].metric;
        for r in results.iter().skip(1) {
            if &r.metric != metric {
                return Err(anyhow!(
                    "All models must use the same metric for ranking, found {:?} and {:?}",
                    metric,
                    r.metric
                ));
            }
        }

        // Sort by mean score descending
        let mut ranked: Vec<(String, f64)> = results
            .iter()
            .map(|r| (r.model_id.clone(), r.mean))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Determine if best model is significantly better than runner-up
        let mut is_best_significant = false;
        if ranked.len() >= 2 {
            let best = results
                .iter()
                .find(|r| r.model_id == ranked[0].0)
                .expect("best model must be in results");
            let second = results
                .iter()
                .find(|r| r.model_id == ranked[1].0)
                .expect("second model must be in results");
            if let Ok(cmp) = self.compare(best, second) {
                is_best_significant = cmp.is_significant;
            }
        } else {
            is_best_significant = true; // Only one model, trivially best
        }

        let result: Vec<(String, f64, bool)> = ranked
            .into_iter()
            .enumerate()
            .map(|(i, (id, mean))| (id, mean, i == 0 && is_best_significant))
            .collect();

        Ok(result)
    }

    /// Paired t-test: tests whether mean difference is significantly != 0.
    ///
    /// H0: mean(a - b) = 0
    /// Returns two-sided p-value.
    fn t_test_paired(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        let n = a.len();
        if n < 2 {
            return Err(anyhow!("t-test requires at least 2 paired samples"));
        }

        // Compute differences
        let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        let mean_diff = stats_mean(&diffs);
        let std_diff = stats_std_dev(&diffs, mean_diff);

        if std_diff < 1e-15 {
            // No variation in differences - perfectly tied or identical
            return Ok(if mean_diff.abs() < 1e-15 { 1.0 } else { 0.0 });
        }

        let se = std_diff / (n as f64).sqrt();
        let t_stat = mean_diff / se;
        let df = (n - 1) as f64;

        // Two-sided p-value: 2 * P(T > |t_stat|)
        let p = 2.0 * (1.0 - t_distribution_cdf(t_stat.abs(), df));
        Ok(p.clamp(0.0, 1.0))
    }

    /// Wilcoxon signed-rank test (non-parametric paired comparison).
    ///
    /// Ranks the absolute differences and tests whether positive and
    /// negative ranks are symmetrically distributed.
    fn wilcoxon_signed_rank(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        let n = a.len();
        if n < 2 {
            return Err(anyhow!(
                "Wilcoxon signed-rank test requires at least 2 paired samples"
            ));
        }

        // Compute differences, ignoring zeros
        let diffs: Vec<f64> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| x - y)
            .filter(|d| d.abs() > 1e-15)
            .collect();

        let m = diffs.len();
        if m == 0 {
            return Ok(1.0); // All tied
        }

        // Rank the absolute differences (with ties averaged)
        let abs_diffs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
        let ranks = rank_with_ties(&abs_diffs);

        // Sum of positive and negative ranks
        let w_plus: f64 = diffs
            .iter()
            .zip(ranks.iter())
            .filter(|(d, _)| **d > 0.0)
            .map(|(_, r)| r)
            .sum();

        let w_minus: f64 = diffs
            .iter()
            .zip(ranks.iter())
            .filter(|(d, _)| **d < 0.0)
            .map(|(_, r)| r)
            .sum();

        let w_stat = w_plus.min(w_minus);

        // Normal approximation for p-value (valid for m >= 10)
        let m_f = m as f64;
        let expected = m_f * (m_f + 1.0) / 4.0;
        let variance = m_f * (m_f + 1.0) * (2.0 * m_f + 1.0) / 24.0;

        if variance < 1e-15 {
            return Ok(1.0);
        }

        let z = (w_stat - expected) / variance.sqrt();
        // Two-sided p-value using normal approximation
        let p = 2.0 * standard_normal_cdf(-z.abs());
        Ok(p.clamp(0.0, 1.0))
    }

    /// Bootstrap permutation test: estimates p-value by random label swapping.
    ///
    /// Under H0 (no difference), swapping labels between A and B should
    /// produce test statistics as extreme as the observed difference with
    /// probability equal to the p-value.
    fn bootstrap_test(
        &self,
        a: &[f64],
        b: &[f64],
        n_permutations: usize,
        seed: u64,
    ) -> Result<f64> {
        if n_permutations == 0 {
            return Err(anyhow!("n_permutations must be > 0"));
        }
        let n = a.len();
        if n < 2 {
            return Err(anyhow!("Bootstrap test requires at least 2 paired samples"));
        }

        let observed_diff = (stats_mean(a) - stats_mean(b)).abs();

        // LCG for reproducible random permutations
        let mut rng = BootstrapRng::new(seed);
        let mut extreme_count = 0usize;

        for _ in 0..n_permutations {
            // For each pair (a_i, b_i), randomly swap with probability 0.5
            let (perm_a, perm_b): (Vec<f64>, Vec<f64>) = a
                .iter()
                .zip(b.iter())
                .map(
                    |(&ai, &bi)| {
                        if rng.next_bool() {
                            (ai, bi)
                        } else {
                            (bi, ai)
                        }
                    },
                )
                .unzip();

            let perm_diff = (stats_mean(&perm_a) - stats_mean(&perm_b)).abs();
            if perm_diff >= observed_diff {
                extreme_count += 1;
            }
        }

        let p = (extreme_count as f64 + 1.0) / (n_permutations as f64 + 1.0);
        Ok(p.clamp(0.0, 1.0))
    }
}

/// Evaluate link prediction quality on a set of embeddings.
///
/// For each query node, computes the rank of the positive target
/// among all candidates, then returns mean reciprocal rank scores.
///
/// # Arguments
/// * `embeddings` - Node embedding matrix (indexed by node ID)
/// * `positive_pairs` - Known positive (source, target) pairs
/// * `negative_pairs` - Known negative (source, non-target) pairs
pub fn evaluate_link_prediction(
    model_id: String,
    embeddings: &[Vec<f64>],
    positive_pairs: &[(usize, usize)],
    negative_pairs: &[(usize, usize)],
) -> Result<ModelEvalResult> {
    if embeddings.is_empty() {
        return Err(anyhow!("embeddings must not be empty"));
    }
    if positive_pairs.is_empty() {
        return Err(anyhow!("positive_pairs must not be empty"));
    }

    // For each positive pair (u, v), compute its rank among all negatives
    let mut mrr_scores: Vec<f64> = Vec::with_capacity(positive_pairs.len());

    for &(u, v) in positive_pairs {
        let emb_u = match embeddings.get(u) {
            Some(e) => e,
            None => continue,
        };
        let emb_v = match embeddings.get(v) {
            Some(e) => e,
            None => continue,
        };

        let pos_score = cosine_similarity_slice(emb_u, emb_v);

        // Count how many negatives score higher than this positive
        let mut higher_count = 0usize;
        for &(nu, nv) in negative_pairs {
            let emb_nu = match embeddings.get(nu) {
                Some(e) => e,
                None => continue,
            };
            let emb_nv = match embeddings.get(nv) {
                Some(e) => e,
                None => continue,
            };
            let neg_score = cosine_similarity_slice(emb_nu, emb_nv);
            if neg_score >= pos_score {
                higher_count += 1;
            }
        }

        let rank = (higher_count + 1) as f64;
        mrr_scores.push(1.0 / rank);
    }

    if mrr_scores.is_empty() {
        return Err(anyhow!("No valid positive pairs found in embeddings"));
    }

    ModelEvalResult::new(model_id, EmbedMetric::MeanReciprocalRank, mrr_scores)
}

/// Evaluate Hits@K: fraction of positive pairs ranked in top-K
pub fn evaluate_hits_at_k(
    model_id: String,
    embeddings: &[Vec<f64>],
    positive_pairs: &[(usize, usize)],
    negative_pairs: &[(usize, usize)],
    k: usize,
) -> Result<ModelEvalResult> {
    if embeddings.is_empty() {
        return Err(anyhow!("embeddings must not be empty"));
    }
    if positive_pairs.is_empty() {
        return Err(anyhow!("positive_pairs must not be empty"));
    }
    if k == 0 {
        return Err(anyhow!("k must be > 0"));
    }

    let mut hit_scores: Vec<f64> = Vec::with_capacity(positive_pairs.len());

    for &(u, v) in positive_pairs {
        let emb_u = match embeddings.get(u) {
            Some(e) => e,
            None => continue,
        };
        let emb_v = match embeddings.get(v) {
            Some(e) => e,
            None => continue,
        };

        let pos_score = cosine_similarity_slice(emb_u, emb_v);

        let mut higher_count = 0usize;
        for &(nu, nv) in negative_pairs {
            let emb_nu = match embeddings.get(nu) {
                Some(e) => e,
                None => continue,
            };
            let emb_nv = match embeddings.get(nv) {
                Some(e) => e,
                None => continue,
            };
            let neg_score = cosine_similarity_slice(emb_nu, emb_nv);
            if neg_score >= pos_score {
                higher_count += 1;
            }
        }

        // Hit if rank <= k
        let rank = higher_count + 1;
        hit_scores.push(if rank <= k { 1.0 } else { 0.0 });
    }

    if hit_scores.is_empty() {
        return Err(anyhow!("No valid positive pairs found in embeddings"));
    }

    ModelEvalResult::new(model_id, EmbedMetric::HitsAtK(k), hit_scores)
}

/// Evaluate silhouette score for node clustering quality.
///
/// The silhouette coefficient measures how similar each node is to its own cluster
/// compared to other clusters. Range: [-1, 1], higher is better.
pub fn evaluate_silhouette(
    model_id: String,
    embeddings: &[Vec<f64>],
    cluster_labels: &[usize],
) -> Result<ModelEvalResult> {
    let n = embeddings.len();
    if n < 2 {
        return Err(anyhow!("Need at least 2 nodes for silhouette score"));
    }
    if cluster_labels.len() != n {
        return Err(anyhow!(
            "cluster_labels length {} != embeddings length {}",
            cluster_labels.len(),
            n
        ));
    }

    // Collect unique clusters
    let unique_clusters: std::collections::HashSet<usize> =
        cluster_labels.iter().copied().collect();
    if unique_clusters.len() < 2 {
        return Err(anyhow!(
            "Need at least 2 distinct clusters for silhouette score"
        ));
    }

    let mut silhouette_scores: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let my_cluster = cluster_labels[i];

        // Intra-cluster: mean distance to all other nodes in same cluster
        let my_cluster_nodes: Vec<usize> = (0..n)
            .filter(|&j| j != i && cluster_labels[j] == my_cluster)
            .collect();

        let a = if my_cluster_nodes.is_empty() {
            0.0
        } else {
            my_cluster_nodes
                .iter()
                .map(|&j| euclidean_distance(&embeddings[i], &embeddings[j]))
                .sum::<f64>()
                / my_cluster_nodes.len() as f64
        };

        // Inter-cluster: minimum mean distance to any other cluster
        let b = unique_clusters
            .iter()
            .filter(|&&c| c != my_cluster)
            .map(|&c| {
                let other_nodes: Vec<usize> = (0..n).filter(|&j| cluster_labels[j] == c).collect();
                if other_nodes.is_empty() {
                    f64::INFINITY
                } else {
                    other_nodes
                        .iter()
                        .map(|&j| euclidean_distance(&embeddings[i], &embeddings[j]))
                        .sum::<f64>()
                        / other_nodes.len() as f64
                }
            })
            .fold(f64::INFINITY, f64::min);

        // Silhouette coefficient
        let s = if a < b {
            1.0 - a / b
        } else if a > b {
            b / a - 1.0
        } else {
            0.0
        };

        silhouette_scores.push(s);
    }

    ModelEvalResult::new(model_id, EmbedMetric::SilhouetteScore, silhouette_scores)
}

// ============================================================================
// Statistical utility functions
// ============================================================================

/// Compute arithmetic mean
fn stats_mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Compute population standard deviation
fn stats_std_dev(v: &[f64], mean: f64) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let variance: f64 =
        v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (v.len() - 1) as f64;
    variance.sqrt()
}

/// Cohen's d: standardized effect size between two samples
fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let mean_a = stats_mean(a);
    let mean_b = stats_mean(b);
    let std_a = stats_std_dev(a, mean_a);
    let std_b = stats_std_dev(b, mean_b);

    // Pooled standard deviation
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    let pooled_var =
        ((n_a - 1.0) * std_a * std_a + (n_b - 1.0) * std_b * std_b) / (n_a + n_b - 2.0);

    if pooled_var < 1e-15 {
        return 0.0;
    }

    (mean_a - mean_b) / pooled_var.sqrt()
}

/// CDF of the Student's t-distribution.
///
/// Uses a simple and robust normal approximation for large |t|,
/// and the exact continued fraction for small |t|.
/// Two-sided p-value computation: p = 2 * (1 - CDF(|t|)).
fn t_distribution_cdf(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 0.5;
    }
    // For very large t, the CDF is essentially 1.0
    if t.abs() > 1e6 {
        return if t >= 0.0 { 1.0 } else { 0.0 };
    }
    // Regularized incomplete beta: P(T <= t | df) = 1 - 0.5 * I_x(df/2, 1/2)
    // where x = df / (df + t^2)
    let x = df / (df + t * t);
    // I_x(df/2, 1/2) computed via the beta regularized function
    let beta_inc = betai(df / 2.0, 0.5, x);
    // CDF(t) = 1 - 0.5 * betai for t >= 0
    let cdf = 1.0 - 0.5 * beta_inc;
    cdf.clamp(0.0, 1.0)
}

/// Regularized incomplete beta function I_x(a, b).
///
/// Uses the continued fraction method from Numerical Recipes.
/// Returns a value in [0, 1].
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    // Use the symmetry relation for numerical stability
    let bt =
        (log_gamma(a + b) - log_gamma(a) - log_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();

    // Evaluate on the smaller argument side for better CF convergence
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for the incomplete beta function (Numerical Recipes).
/// Evaluates the continued fraction via Lentz's method.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3.0e-10;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m2 = 2 * m;
        // Even step
        let aa = m as f64 * (b - m as f64) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step
        let aa = -(a + m as f64) * (qab + m as f64) * x / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Log of the Gamma function using Lanczos approximation (Numerical Recipes).
/// Accurate to ~15 decimal places for z > 0.
fn log_gamma(z: f64) -> f64 {
    if z <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos coefficients for g=7
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        // Reflection formula: Gamma(z)*Gamma(1-z) = pi/sin(pi*z)
        std::f64::consts::PI.ln() - (std::f64::consts::PI * z).sin().abs().ln() - log_gamma(1.0 - z)
    } else {
        let z = z - 1.0;
        let mut x = C[0];
        for (i, &c) in C[1..].iter().enumerate() {
            x += c / (z + i as f64 + 1.0);
        }
        let t = z + G + 0.5;
        (std::f64::consts::TAU.sqrt()).ln() + x.ln() + (z + 0.5) * t.ln() - t
    }
}

/// t critical value approximation using inverse t-distribution.
///
/// For common significance levels, returns the t-critical value
/// for the given degrees of freedom and tail probability.
fn t_critical_value(df: usize, tail_prob: f64) -> f64 {
    // Use normal approximation for large df (>= 30)
    if df >= 30 {
        return normal_quantile(1.0 - tail_prob);
    }
    // Simple lookup for small df (common in practice)
    // These are approximate values for two-sided alpha=0.05 (tail=0.025)
    match df {
        1 => 12.706,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        5 => 2.571,
        6 => 2.447,
        7 => 2.365,
        8 => 2.306,
        9 => 2.262,
        10 => 2.228,
        11 => 2.201,
        12 => 2.179,
        13 => 2.160,
        14 => 2.145,
        15 => 2.131,
        16 => 2.120,
        17 => 2.110,
        18 => 2.101,
        19 => 2.093,
        20 => 2.086,
        21..=25 => 2.064,
        26..=29 => 2.048,
        _ => normal_quantile(1.0 - tail_prob),
    }
}

/// Standard normal CDF using rational approximation (Abramowitz & Stegun 26.2.17)
fn standard_normal_cdf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989422820 * (-z * z / 2.0).exp();
    let poly = t
        * (0.3193815306
            + t * (-0.3565637813 + t * (1.7814779372 + t * (-1.8212559978 + t * 1.3302744929))));
    if z >= 0.0 {
        1.0 - d * poly
    } else {
        d * poly
    }
}

/// Inverse normal (quantile function) - approximation (Peter Acklam)
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Rational approximation for central region
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        q * (((2_509.080_928_730_122_7 * r + 33_430.575_583_588_13) * r + 67_265.770_927_008_7) * r
            + 45_921.953_931_549_87)
            / (((5_226.495_278_852_854 * r + 28_729.085_735_721_943) * r + 39_307.895_800_092_71)
                * r
                + 10_765.120_437_959_045
                + 1.0)
    } else {
        let r = if q < 0.0 { p } else { 1.0 - p };
        let r = (-r.ln()).sqrt();
        let x = (((2.990_113_295_264_179 * r + 4.740_220_281_696_907_5) * r
            + 3.343_057_558_358_813)
            * r
            + 0.675_865_739_902_174_9)
            / ((5.104_063_170_295_205_5 * r + 3.874_403_263_689_304_7) * r
                + 0.732_672_856_137_836_4
                + 1.0);
        if q < 0.0 {
            -x
        } else {
            x
        }
    }
}

/// Rank a slice with ties averaged
fn rank_with_ties(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        // Assign average rank to ties
        let avg_rank = (i + j + 1) as f64 / 2.0; // average of ranks i+1..j
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Cosine similarity between two slices
fn cosine_similarity_slice(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Euclidean distance between two vectors
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Simple PRNG for bootstrap tests
struct BootstrapRng {
    state: u64,
}

impl BootstrapRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scores(values: &[f64], model_id: &str) -> ModelEvalResult {
        ModelEvalResult::new(
            model_id.to_string(),
            EmbedMetric::MeanReciprocalRank,
            values.to_vec(),
        )
        .expect("scores should be valid")
    }

    #[test]
    fn test_model_eval_result_basic() {
        let scores = vec![0.8, 0.7, 0.9, 0.6, 0.85];
        let result = ModelEvalResult::new(
            "ModelA".to_string(),
            EmbedMetric::MeanReciprocalRank,
            scores.clone(),
        )
        .expect("should construct");

        assert_eq!(result.sample_count, 5);
        assert!((result.mean - 0.77).abs() < 0.01);
        assert!(result.std_dev > 0.0);
    }

    #[test]
    fn test_model_eval_result_empty_scores() {
        let result =
            ModelEvalResult::new("Model".to_string(), EmbedMetric::SilhouetteScore, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_confidence_interval() {
        let scores: Vec<f64> = (0..50).map(|i| i as f64 * 0.02).collect();
        let result =
            ModelEvalResult::new("M".to_string(), EmbedMetric::ClassificationAccuracy, scores)
                .expect("should construct");
        let (lo, hi) = result.confidence_interval(0.05);
        assert!(lo < result.mean);
        assert!(hi > result.mean);
        assert!(lo >= 0.0);
        assert!(hi <= 1.0 + 1e-6);
    }

    #[test]
    fn test_ab_test_clearly_different() {
        // Model A is clearly better (scores ~0.9 vs ~0.1)
        let a_scores: Vec<f64> = vec![0.85, 0.90, 0.88, 0.92, 0.87, 0.91, 0.89, 0.93, 0.86, 0.94];
        let b_scores: Vec<f64> = vec![0.10, 0.12, 0.09, 0.11, 0.13, 0.08, 0.10, 0.12, 0.11, 0.09];

        let result_a = make_scores(&a_scores, "ModelA");
        let result_b = make_scores(&b_scores, "ModelB");

        let runner = AbTestRunner::new();
        let ab_result = runner
            .compare(&result_a, &result_b)
            .expect("comparison should succeed");

        assert!(ab_result.is_significant, "should be significant");
        assert_eq!(ab_result.winner, Some("ModelA".to_string()));
        assert!(ab_result.p_value < 0.001);
        assert!(ab_result.effect_size.abs() > 1.0); // Large effect size
    }

    #[test]
    fn test_ab_test_similar_models() {
        // Models A and B with very similar scores (just noise)
        let a_scores: Vec<f64> = vec![0.500, 0.501, 0.499, 0.502, 0.498, 0.501, 0.500, 0.499];
        let b_scores: Vec<f64> = vec![0.499, 0.500, 0.501, 0.500, 0.501, 0.499, 0.500, 0.501];

        let result_a = make_scores(&a_scores, "ModelA");
        let result_b = make_scores(&b_scores, "ModelB");

        let runner = AbTestRunner::new();
        let ab_result = runner
            .compare(&result_a, &result_b)
            .expect("comparison should succeed");

        assert!(
            !ab_result.is_significant,
            "should not be significant for similar scores"
        );
        assert!(ab_result.winner.is_none());
        assert!(ab_result.p_value > 0.05);
    }

    #[test]
    fn test_ab_test_different_metrics_error() {
        let result_a = ModelEvalResult::new(
            "A".to_string(),
            EmbedMetric::MeanReciprocalRank,
            vec![0.5; 10],
        )
        .expect("ok");
        let result_b =
            ModelEvalResult::new("B".to_string(), EmbedMetric::SilhouetteScore, vec![0.5; 10])
                .expect("ok");

        let runner = AbTestRunner::new();
        assert!(runner.compare(&result_a, &result_b).is_err());
    }

    #[test]
    fn test_rank_models() {
        let scores_a: Vec<f64> = vec![0.8; 20];
        let scores_b: Vec<f64> = vec![0.6; 20];
        let scores_c: Vec<f64> = vec![0.7; 20];

        let results = vec![
            ModelEvalResult::new("B".to_string(), EmbedMetric::HitsAtK(10), scores_b).expect("ok"),
            ModelEvalResult::new("A".to_string(), EmbedMetric::HitsAtK(10), scores_a).expect("ok"),
            ModelEvalResult::new("C".to_string(), EmbedMetric::HitsAtK(10), scores_c).expect("ok"),
        ];

        let runner = AbTestRunner::new();
        let ranking = runner
            .rank_models(&results)
            .expect("ranking should succeed");

        assert_eq!(ranking.len(), 3);
        assert_eq!(ranking[0].0, "A"); // Highest mean
        assert_eq!(ranking[1].0, "C");
        assert_eq!(ranking[2].0, "B");
    }

    #[test]
    fn test_bootstrap_test() {
        let a_scores: Vec<f64> = vec![0.85, 0.90, 0.88, 0.92, 0.87, 0.91, 0.89, 0.93, 0.86, 0.94];
        let b_scores: Vec<f64> = vec![0.10, 0.12, 0.09, 0.11, 0.13, 0.08, 0.10, 0.12, 0.11, 0.09];

        let result_a = make_scores(&a_scores, "A");
        let result_b = make_scores(&b_scores, "B");

        let runner = AbTestRunner::new().with_test(StatTest::Bootstrap {
            n_permutations: 999,
            seed: 42,
        });
        let ab_result = runner
            .compare(&result_a, &result_b)
            .expect("bootstrap should succeed");

        assert!(ab_result.is_significant);
        assert!(ab_result.p_value < 0.05);
    }

    #[test]
    fn test_wilcoxon_test() {
        let a_scores: Vec<f64> = vec![0.9, 0.85, 0.92, 0.88, 0.91, 0.87, 0.93, 0.86, 0.94, 0.89];
        let b_scores: Vec<f64> = vec![0.1, 0.12, 0.09, 0.11, 0.13, 0.08, 0.1, 0.12, 0.11, 0.09];

        let result_a = make_scores(&a_scores, "A");
        let result_b = make_scores(&b_scores, "B");

        let runner = AbTestRunner::new().with_test(StatTest::WilcoxonSignedRank);
        let ab_result = runner
            .compare(&result_a, &result_b)
            .expect("wilcoxon should succeed");

        assert!(ab_result.is_significant);
    }

    #[test]
    fn test_evaluate_link_prediction() {
        // Create simple embeddings: nodes 0,1 are similar; 2,3 are different from 0
        let embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let positive_pairs = vec![(0, 1)];
        let negative_pairs = vec![(0, 2), (0, 3)];

        let result = evaluate_link_prediction(
            "test_model".to_string(),
            &embeddings,
            &positive_pairs,
            &negative_pairs,
        )
        .expect("link prediction eval should succeed");

        assert_eq!(result.model_id, "test_model");
        assert_eq!(result.metric, EmbedMetric::MeanReciprocalRank);
        assert!(!result.scores.is_empty());
        // Rank 1 => MRR = 1.0, node 0-1 should score higher than 0-2
        assert!((result.scores[0] - 1.0).abs() < 1e-10, "MRR should be 1.0");
    }

    #[test]
    fn test_evaluate_hits_at_k() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let positive_pairs = vec![(0, 1)];
        let negative_pairs = vec![(0, 2)];

        let result = evaluate_hits_at_k(
            "model".to_string(),
            &embeddings,
            &positive_pairs,
            &negative_pairs,
            1,
        )
        .expect("hits@k eval should succeed");

        assert_eq!(result.metric, EmbedMetric::HitsAtK(1));
        assert!(!result.scores.is_empty());
        // Positive pair has higher cosine similarity => rank 1 => hit@1 = 1.0
        assert!((result.scores[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_silhouette() {
        // Two well-separated clusters
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.95, 0.05],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
            vec![0.05, 0.95],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];

        let result = evaluate_silhouette("model".to_string(), &embeddings, &labels)
            .expect("silhouette eval should succeed");

        assert_eq!(result.metric, EmbedMetric::SilhouetteScore);
        assert_eq!(result.sample_count, 6);
        // Well-separated clusters should have positive mean silhouette
        assert!(
            result.mean > 0.0,
            "mean silhouette should be positive for well-separated clusters"
        );
    }

    #[test]
    fn test_cohens_d_interpretation() {
        // d > 0.8 is "large" effect; identical means => d = 0
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((cohens_d(&a, &b)).abs() < 1e-10);

        let c = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let d = cohens_d(&c, &a);
        assert!(
            d.abs() > 1.0,
            "Large difference should give large Cohen's d"
        );
    }

    #[test]
    fn test_rank_with_ties() {
        let values = vec![3.0, 1.0, 1.0, 2.0];
        let ranks = rank_with_ties(&values);
        assert_eq!(ranks.len(), 4);
        // value 1.0 appears at positions 1,2 => avg rank = (1+2)/2 = 1.5
        assert!((ranks[1] - 1.5).abs() < 1e-10);
        assert!((ranks[2] - 1.5).abs() < 1e-10);
        // value 2.0 at position 3 => rank 3
        assert!((ranks[3] - 3.0).abs() < 1e-10);
        // value 3.0 at position 0 => rank 4
        assert!((ranks[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_hits_at_k_zero_error() {
        let result = evaluate_hits_at_k(
            "m".to_string(),
            &[vec![1.0f64]],
            &[(0, 0)],
            &[],
            0, // k=0 is invalid
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_embed_metric_display() {
        assert_eq!(EmbedMetric::MeanReciprocalRank.to_string(), "MRR");
        assert_eq!(EmbedMetric::HitsAtK(10).to_string(), "Hits@10");
        assert_eq!(
            EmbedMetric::Custom("MyMetric".to_string()).to_string(),
            "MyMetric"
        );
    }

    #[test]
    fn test_ab_test_summary() {
        let a_scores = vec![0.8f64; 10];
        let b_scores = vec![0.2f64; 10];
        let result_a = make_scores(&a_scores, "ModelA");
        let result_b = make_scores(&b_scores, "ModelB");

        let runner = AbTestRunner::new();
        let ab_result = runner.compare(&result_a, &result_b).expect("ok");
        let summary = ab_result.summary();
        assert!(summary.contains("ModelA"));
        assert!(summary.contains("ModelB"));
        assert!(summary.contains("significant"));
    }
}
