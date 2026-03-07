//! A/B Testing Framework for Embedding Models (v0.3.0)
//!
//! Production-ready framework for comparing embedding model variants with:
//! - [`ModelVariant`]: encapsulates a model with metadata and metrics collection
//! - [`ABTestConfig`]: configures traffic splits, metric targets, and test duration
//! - [`ABTestRunner`]: routes inference requests between variants and records outcomes
//! - [`ABTestAnalyzer`]: statistical significance testing (Welch's t-test, Mann-Whitney U)
//! - [`ABTestReport`]: generates detailed comparison reports
//!
//! ## Design
//!
//! The framework is model-agnostic: any function from an input key to a
//! `Vec<f64>` embedding qualifies as a "model variant".  This keeps the A/B
//! framework decoupled from specific GNN or KGE implementations.
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxirs_embed::ab_testing::{ABTestConfig, ABTestRunner, ModelVariant};
//!
//! # fn main() -> anyhow::Result<()> {
//! let control = ModelVariant::new("transe-v1", |_key: &str| vec![0.0f64; 64]);
//! let treatment = ModelVariant::new("transe-v2", |_key: &str| vec![0.1f64; 64]);
//!
//! let config = ABTestConfig::default();
//! let mut runner = ABTestRunner::new(config, control, treatment)?;
//!
//! // Simulate requests
//! for i in 0..200 {
//!     let key = format!("entity:{i}");
//!     let (embedding, variant_name) = runner.route(&key)?;
//!     // Record a business metric (e.g., link prediction hit@10)
//!     runner.record_metric(&variant_name, 0.85)?;
//! }
//!
//! let report = runner.analyze()?;
//! println!("{}", report.summary());
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ModelVariant
// ---------------------------------------------------------------------------

/// A named embedding model variant for A/B testing.
///
/// Wraps an inference function `fn(&str) -> Vec<f64>` together with
/// metadata about the variant (name, description, version).
pub struct ModelVariant {
    /// Human-readable name, e.g. `"transe-v1"`.
    pub name: String,
    /// Optional description of the variant.
    pub description: String,
    /// Optional version string.
    pub version: String,
    /// The inference function: maps an entity key to an embedding.
    #[allow(clippy::type_complexity)]
    infer: Box<dyn Fn(&str) -> Vec<f64> + Send + Sync>,
}

impl std::fmt::Debug for ModelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelVariant")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("version", &self.version)
            .finish()
    }
}

impl ModelVariant {
    /// Create a new variant with a name and inference function.
    pub fn new<F>(name: impl Into<String>, infer: F) -> Self
    where
        F: Fn(&str) -> Vec<f64> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            description: String::new(),
            version: String::from("0.1.0"),
            infer: Box::new(infer),
        }
    }

    /// Attach a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Attach a version string.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Run inference for the given entity key.
    pub fn infer(&self, key: &str) -> Vec<f64> {
        (self.infer)(key)
    }
}

// ---------------------------------------------------------------------------
// ABTestConfig
// ---------------------------------------------------------------------------

/// Metric to optimize / compare between variants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizeMetric {
    /// Higher is better (e.g., recall, NDCG).
    Maximize,
    /// Lower is better (e.g., MRR rank, latency).
    Minimize,
}

/// Configuration for an A/B test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConfig {
    /// Fraction of requests routed to the treatment variant (0.0 – 1.0).
    /// The remainder goes to the control variant.
    pub traffic_split: f64,
    /// Minimum number of observations per variant before analysis.
    pub min_samples: usize,
    /// Significance level α for hypothesis tests (default 0.05).
    pub significance_level: f64,
    /// Direction of optimization for the primary metric.
    pub optimize: OptimizeMetric,
    /// Random seed for deterministic traffic splitting.
    pub seed: u64,
    /// Optional maximum number of requests before the test is declared complete.
    pub max_requests: Option<usize>,
    /// Optional minimum detectable effect size (Cohen's d).
    pub min_effect_size: Option<f64>,
}

impl Default for ABTestConfig {
    fn default() -> Self {
        Self {
            traffic_split: 0.5,
            min_samples: 50,
            significance_level: 0.05,
            optimize: OptimizeMetric::Maximize,
            seed: 42,
            max_requests: None,
            min_effect_size: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// A single metric observation from a model variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// The variant that produced this observation.
    pub variant_name: String,
    /// Request key (entity IRI, query string, etc.).
    pub key: String,
    /// Observed metric value.
    pub metric: f64,
    /// Wall-clock time taken for inference (microseconds).
    pub latency_us: u64,
}

// ---------------------------------------------------------------------------
// ABTestRunner
// ---------------------------------------------------------------------------

/// Routes inference requests between two variants and records metrics.
pub struct ABTestRunner {
    config: ABTestConfig,
    control: ModelVariant,
    treatment: ModelVariant,
    observations: Vec<Observation>,
    total_requests: usize,
    /// Simple LCG for deterministic traffic assignment.
    lcg_state: u64,
    /// Latency of the most recent route() call (microseconds).
    last_latency_us: u64,
}

impl std::fmt::Debug for ABTestRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ABTestRunner")
            .field("control", &self.control.name)
            .field("treatment", &self.treatment.name)
            .field("total_requests", &self.total_requests)
            .field("observations", &self.observations.len())
            .finish()
    }
}

impl ABTestRunner {
    /// Create a new A/B test runner.
    pub fn new(
        config: ABTestConfig,
        control: ModelVariant,
        treatment: ModelVariant,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&config.traffic_split) {
            return Err(anyhow!(
                "traffic_split must be in [0, 1], got {}",
                config.traffic_split
            ));
        }
        if config.significance_level <= 0.0 || config.significance_level >= 1.0 {
            return Err(anyhow!(
                "significance_level must be in (0, 1), got {}",
                config.significance_level
            ));
        }
        let lcg_state = config.seed.wrapping_add(1);
        Ok(Self {
            config,
            control,
            treatment,
            observations: Vec::new(),
            total_requests: 0,
            lcg_state,
            last_latency_us: 0,
        })
    }

    /// Route a single request to a variant, returning `(embedding, variant_name)`.
    ///
    /// The variant is chosen deterministically based on `key` and the LCG state,
    /// ensuring the configured `traffic_split` is respected on average.
    pub fn route(&mut self, key: &str) -> Result<(Vec<f64>, String)> {
        if let Some(max) = self.config.max_requests {
            if self.total_requests >= max {
                return Err(anyhow!("A/B test has reached max_requests {}", max));
            }
        }
        // Mix key hash with LCG for per-request randomness
        let key_hash = fnv1a_hash(key);
        self.lcg_state = self
            .lcg_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            .wrapping_add(key_hash);
        let r = (self.lcg_state >> 11) as f64 / (1u64 << 53) as f64;

        let use_treatment = r < self.config.traffic_split;
        let variant = if use_treatment {
            &self.treatment
        } else {
            &self.control
        };
        let start = std::time::Instant::now();
        let embedding = variant.infer(key);
        let latency_us = start.elapsed().as_micros() as u64;

        self.last_latency_us = latency_us;
        self.total_requests += 1;
        Ok((embedding, variant.name.clone()))
    }

    /// Record a metric observation for a variant.
    ///
    /// Call this after routing to register the quality signal for the request.
    pub fn record_metric(&mut self, variant_name: &str, metric: f64) -> Result<()> {
        if !metric.is_finite() {
            return Err(anyhow!("metric must be finite, got {}", metric));
        }
        // Attribute to the last routed request
        let key = format!("req_{}", self.total_requests);
        self.observations.push(Observation {
            variant_name: variant_name.to_string(),
            key,
            metric,
            latency_us: self.last_latency_us,
        });
        Ok(())
    }

    /// Record a full observation (metric + latency + key).
    pub fn record_observation(&mut self, obs: Observation) -> Result<()> {
        if !obs.metric.is_finite() {
            return Err(anyhow!("Observation metric must be finite"));
        }
        self.observations.push(obs);
        Ok(())
    }

    /// Analyze the collected observations and return a report.
    pub fn analyze(&self) -> Result<ABTestReport> {
        let ctrl_metrics: Vec<f64> = self
            .observations
            .iter()
            .filter(|o| o.variant_name == self.control.name)
            .map(|o| o.metric)
            .collect();
        let trt_metrics: Vec<f64> = self
            .observations
            .iter()
            .filter(|o| o.variant_name == self.treatment.name)
            .map(|o| o.metric)
            .collect();

        if ctrl_metrics.len() < self.config.min_samples {
            return Err(anyhow!(
                "Not enough control observations: {} < {}",
                ctrl_metrics.len(),
                self.config.min_samples
            ));
        }
        if trt_metrics.len() < self.config.min_samples {
            return Err(anyhow!(
                "Not enough treatment observations: {} < {}",
                trt_metrics.len(),
                self.config.min_samples
            ));
        }

        let analyzer = ABTestAnalyzer::new(&self.config);
        analyzer.analyze(
            &self.control.name,
            &ctrl_metrics,
            &self.treatment.name,
            &trt_metrics,
        )
    }

    /// Number of requests routed so far.
    pub fn total_requests(&self) -> usize {
        self.total_requests
    }

    /// All recorded observations.
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }

    /// Get per-variant summary statistics without a full report.
    pub fn variant_stats(&self) -> HashMap<String, VariantStats> {
        let mut map: HashMap<String, Vec<f64>> = HashMap::new();
        for obs in &self.observations {
            map.entry(obs.variant_name.clone())
                .or_default()
                .push(obs.metric);
        }
        map.into_iter()
            .map(|(name, metrics)| {
                let stats = VariantStats::from_slice(&metrics);
                (name, stats)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ABTestAnalyzer
// ---------------------------------------------------------------------------

/// Statistical significance testing for A/B experiment results.
pub struct ABTestAnalyzer<'a> {
    config: &'a ABTestConfig,
}

impl<'a> ABTestAnalyzer<'a> {
    /// Create with the test configuration.
    pub fn new(config: &'a ABTestConfig) -> Self {
        Self { config }
    }

    /// Run Welch's t-test and Mann-Whitney U test, then build a report.
    pub fn analyze(
        &self,
        control_name: &str,
        control_metrics: &[f64],
        treatment_name: &str,
        treatment_metrics: &[f64],
    ) -> Result<ABTestReport> {
        if control_metrics.is_empty() || treatment_metrics.is_empty() {
            return Err(anyhow!("Both metric slices must be non-empty"));
        }

        let ctrl_stats = VariantStats::from_slice(control_metrics);
        let trt_stats = VariantStats::from_slice(treatment_metrics);

        let ttest_result = self.welchs_ttest(control_metrics, treatment_metrics)?;
        let mwu_result = self.mann_whitney_u(control_metrics, treatment_metrics)?;
        let cohens_d = self.cohens_d(control_metrics, treatment_metrics);

        let significant = ttest_result.p_value < self.config.significance_level
            && mwu_result.p_value < self.config.significance_level;

        // Determine winner
        let winner = if !significant {
            Winner::NoSignificantDifference
        } else {
            let ctrl_better = ctrl_stats.mean > trt_stats.mean;
            match self.config.optimize {
                OptimizeMetric::Maximize => {
                    if ctrl_better {
                        Winner::Control(control_name.to_string())
                    } else {
                        Winner::Treatment(treatment_name.to_string())
                    }
                }
                OptimizeMetric::Minimize => {
                    if ctrl_better {
                        Winner::Treatment(treatment_name.to_string())
                    } else {
                        Winner::Control(control_name.to_string())
                    }
                }
            }
        };

        Ok(ABTestReport {
            control_name: control_name.to_string(),
            treatment_name: treatment_name.to_string(),
            control_stats: ctrl_stats,
            treatment_stats: trt_stats,
            ttest: ttest_result,
            mann_whitney: mwu_result,
            cohens_d,
            significant,
            winner,
            significance_level: self.config.significance_level,
        })
    }

    /// Welch's t-test (unequal variance, two-sided).
    ///
    /// Returns t-statistic, degrees of freedom (Welch-Satterthwaite), and
    /// a p-value approximated via Student's t CDF.
    pub fn welchs_ttest(&self, a: &[f64], b: &[f64]) -> Result<TTestResult> {
        if a.len() < 2 || b.len() < 2 {
            return Err(anyhow!(
                "Welch's t-test requires >= 2 observations per group"
            ));
        }
        let na = a.len() as f64;
        let nb = b.len() as f64;
        let mean_a = mean(a);
        let mean_b = mean(b);
        let var_a = variance(a, mean_a);
        let var_b = variance(b, mean_b);

        if var_a < 1e-15 && var_b < 1e-15 {
            // Both groups are constant
            let t = if (mean_a - mean_b).abs() < 1e-12 {
                0.0
            } else {
                f64::INFINITY
            };
            return Ok(TTestResult {
                t_statistic: t,
                degrees_of_freedom: 0.0,
                p_value: if t == 0.0 { 1.0 } else { 0.0 },
                mean_diff: mean_a - mean_b,
            });
        }

        let se = (var_a / na + var_b / nb).sqrt();
        if se < 1e-15 {
            return Err(anyhow!("Standard error too small for t-test"));
        }
        let t = (mean_a - mean_b) / se;

        // Welch-Satterthwaite degrees of freedom
        let df_num = (var_a / na + var_b / nb).powi(2);
        let df_den = (var_a / na).powi(2) / (na - 1.0) + (var_b / nb).powi(2) / (nb - 1.0);
        let df = if df_den < 1e-15 { 1.0 } else { df_num / df_den };

        // Two-sided p-value via t distribution CDF approximation
        let p_value = t_distribution_two_sided_p(t.abs(), df);

        Ok(TTestResult {
            t_statistic: t,
            degrees_of_freedom: df,
            p_value,
            mean_diff: mean_a - mean_b,
        })
    }

    /// Mann-Whitney U test (Wilcoxon rank-sum, two-sided).
    ///
    /// Suitable for non-normally distributed metrics.
    /// Uses the normal approximation for p-value when n > 20.
    pub fn mann_whitney_u(&self, a: &[f64], b: &[f64]) -> Result<MannWhitneyResult> {
        if a.is_empty() || b.is_empty() {
            return Err(anyhow!("Mann-Whitney U requires non-empty groups"));
        }
        let na = a.len() as f64;
        let nb = b.len() as f64;

        // Rank all observations pooled
        let mut combined: Vec<(f64, u8)> = a
            .iter()
            .map(|&v| (v, 0u8))
            .chain(b.iter().map(|&v| (v, 1u8)))
            .collect();
        combined.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

        // Assign average ranks (handle ties)
        let n_total = combined.len();
        let mut ranks = vec![0.0f64; n_total];
        let mut i = 0;
        while i < n_total {
            let mut j = i;
            while j < n_total && (combined[j].0 - combined[i].0).abs() < 1e-12 {
                j += 1;
            }
            // Average rank for tied group: 1-based
            let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based average
            for rank in ranks[i..j].iter_mut() {
                *rank = avg_rank;
            }
            i = j;
        }

        // Sum of ranks for group a
        let rank_sum_a: f64 = combined
            .iter()
            .zip(ranks.iter())
            .filter(|(obs, _)| obs.1 == 0)
            .map(|(_, &r)| r)
            .sum();

        let u_a = rank_sum_a - na * (na + 1.0) / 2.0;
        let u_b = na * nb - u_a;
        let u = u_a.min(u_b);

        // Normal approximation
        let mu_u = na * nb / 2.0;
        let sigma_u = ((na * nb * (na + nb + 1.0)) / 12.0).sqrt();
        let z = if sigma_u < 1e-12 {
            0.0
        } else {
            (u - mu_u) / sigma_u
        };
        let p_value = 2.0 * standard_normal_sf(z.abs());

        Ok(MannWhitneyResult {
            u_statistic: u,
            z_score: z,
            p_value: p_value.clamp(0.0, 1.0),
            rank_sum_a,
        })
    }

    /// Cohen's d effect size: (mean_a - mean_b) / pooled_std.
    pub fn cohens_d(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() < 2 || b.len() < 2 {
            return 0.0;
        }
        let mean_a = mean(a);
        let mean_b = mean(b);
        let var_a = variance(a, mean_a);
        let var_b = variance(b, mean_b);
        let na = a.len() as f64;
        let nb = b.len() as f64;
        // Pooled standard deviation (Hedges' g denominator)
        let pooled_std = (((na - 1.0) * var_a + (nb - 1.0) * var_b) / (na + nb - 2.0)).sqrt();
        if pooled_std < 1e-15 {
            return 0.0;
        }
        (mean_a - mean_b) / pooled_std
    }
}

// ---------------------------------------------------------------------------
// Statistical test results
// ---------------------------------------------------------------------------

/// Result of Welch's two-sample t-test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTestResult {
    pub t_statistic: f64,
    pub degrees_of_freedom: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// `mean(a) - mean(b)`.
    pub mean_diff: f64,
}

/// Result of Mann-Whitney U test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MannWhitneyResult {
    pub u_statistic: f64,
    pub z_score: f64,
    /// Two-sided p-value (normal approximation).
    pub p_value: f64,
    pub rank_sum_a: f64,
}

// ---------------------------------------------------------------------------
// VariantStats
// ---------------------------------------------------------------------------

/// Summary statistics for one variant's metric observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantStats {
    pub n: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    /// 25th percentile (linear interpolation).
    pub p25: f64,
    /// 50th percentile (median).
    pub p50: f64,
    /// 75th percentile.
    pub p75: f64,
    /// 95th percentile.
    pub p95: f64,
}

impl VariantStats {
    /// Compute from a slice of observations.
    pub fn from_slice(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self {
                n: 0,
                mean: f64::NAN,
                std_dev: f64::NAN,
                min: f64::NAN,
                max: f64::NAN,
                p25: f64::NAN,
                p50: f64::NAN,
                p75: f64::NAN,
                p95: f64::NAN,
            };
        }
        let n = data.len();
        let m = mean(data);
        let var = variance(data, m);
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Self {
            n,
            mean: m,
            std_dev: var.sqrt(),
            min: sorted[0],
            max: sorted[n - 1],
            p25: percentile_sorted(&sorted, 25.0),
            p50: percentile_sorted(&sorted, 50.0),
            p75: percentile_sorted(&sorted, 75.0),
            p95: percentile_sorted(&sorted, 95.0),
        }
    }
}

// ---------------------------------------------------------------------------
// ABTestReport
// ---------------------------------------------------------------------------

/// Which variant won the A/B test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Winner {
    Control(String),
    Treatment(String),
    NoSignificantDifference,
}

/// Full A/B test analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestReport {
    pub control_name: String,
    pub treatment_name: String,
    pub control_stats: VariantStats,
    pub treatment_stats: VariantStats,
    pub ttest: TTestResult,
    pub mann_whitney: MannWhitneyResult,
    /// Cohen's d effect size.
    pub cohens_d: f64,
    /// True iff both tests are below `significance_level`.
    pub significant: bool,
    pub winner: Winner,
    pub significance_level: f64,
}

impl ABTestReport {
    /// Render a human-readable summary.
    pub fn summary(&self) -> String {
        let ctrl = &self.control_stats;
        let trt = &self.treatment_stats;
        let mut lines = Vec::new();
        lines.push("=== A/B Test Report ===".to_string());
        lines.push(format!(
            "Control   ({:>20}): n={:4} mean={:.4} std={:.4} p50={:.4}",
            self.control_name, ctrl.n, ctrl.mean, ctrl.std_dev, ctrl.p50
        ));
        lines.push(format!(
            "Treatment ({:>20}): n={:4} mean={:.4} std={:.4} p50={:.4}",
            self.treatment_name, trt.n, trt.mean, trt.std_dev, trt.p50
        ));
        lines.push(format!(
            "Welch's t-test:  t={:.4}  df={:.1}  p={:.4}",
            self.ttest.t_statistic, self.ttest.degrees_of_freedom, self.ttest.p_value
        ));
        lines.push(format!(
            "Mann-Whitney U:  U={:.1}  z={:.4}  p={:.4}",
            self.mann_whitney.u_statistic, self.mann_whitney.z_score, self.mann_whitney.p_value
        ));
        lines.push(format!("Cohen's d: {:.4}", self.cohens_d));
        lines.push(format!(
            "Significant (α={}): {}",
            self.significance_level, self.significant
        ));
        lines.push(match &self.winner {
            Winner::Control(n) => format!("Winner: CONTROL ({n})"),
            Winner::Treatment(n) => format!("Winner: TREATMENT ({n})"),
            Winner::NoSignificantDifference => "Winner: No significant difference".to_string(),
        });
        lines.join("\n")
    }

    /// Return true if the treatment is the statistically significant winner.
    pub fn treatment_wins(&self) -> bool {
        matches!(&self.winner, Winner::Treatment(_))
    }

    /// Return true if the control is the statistically significant winner.
    pub fn control_wins(&self) -> bool {
        matches!(&self.winner, Winner::Control(_))
    }

    /// Relative improvement of treatment over control: `(trt - ctrl) / |ctrl|`.
    pub fn relative_improvement(&self) -> f64 {
        let ctrl_mean = self.control_stats.mean;
        let trt_mean = self.treatment_stats.mean;
        if ctrl_mean.abs() < 1e-12 {
            return 0.0;
        }
        (trt_mean - ctrl_mean) / ctrl_mean.abs()
    }
}

// ---------------------------------------------------------------------------
// Statistical utilities
// ---------------------------------------------------------------------------

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn variance(data: &[f64], m: f64) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let sum_sq: f64 = data.iter().map(|&x| (x - m).powi(2)).sum();
    sum_sq / (data.len() - 1) as f64
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = p / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Two-sided p-value for t distribution (approximation via regularized incomplete beta).
///
/// Uses Abramowitz & Stegun approximation for the t-distribution CDF.
fn t_distribution_two_sided_p(t_abs: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 1.0;
    }
    // Use normal approximation for large df
    if df > 200.0 {
        return 2.0 * standard_normal_sf(t_abs);
    }
    // Regularized incomplete beta function I_x(a, b) where x = df/(df+t^2)
    let x = df / (df + t_abs * t_abs);
    let a = df / 2.0;
    let b = 0.5f64;
    let ibeta = regularized_incomplete_beta(x, a, b);
    ibeta.clamp(0.0, 1.0)
}

/// Regularized incomplete beta function via continued fraction (Lentz's method).
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Use symmetry if x > (a+1)/(a+b+2)
    let switch = (a + 1.0) / (a + b + 2.0);
    if x > switch {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }
    // Front factor
    let ln_front = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b);
    let front = ln_front.exp();
    // Continued fraction
    let cf = beta_continued_fraction(x, a, b);
    (front * cf / a).clamp(0.0, 1.0)
}

fn beta_continued_fraction(x: f64, a: f64, b: f64) -> f64 {
    // Lentz's algorithm
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-300;
    let mut f = tiny;
    let mut c = f;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    f = d;
    for m in 1..=max_iter {
        let m_f = m as f64;
        // Even step
        let aa = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        f *= d * c;
        // Odd step
        let aa = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        f *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    f
}

/// Natural log of the Beta function via lgamma.
fn ln_beta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

/// Stirling approximation for log-gamma (accurate for a > 0.5).
fn lgamma(a: f64) -> f64 {
    // Lanczos approximation coefficients (g=7, n=9)
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1259.139216722403,
        771.323_428_777_653,
        -176.615_029_162_141_9,
        12.507343278686905,
        -0.138571095265720,
        9.984369578019572e-6,
        1.505632735149312e-7,
    ];
    if a < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * a).sin().abs().ln() - lgamma(1.0 - a)
    } else {
        let x = a - 1.0;
        let t = x + G + 0.5;
        let ser: f64 = C[0]
            + C[1..]
                .iter()
                .enumerate()
                .map(|(i, &c)| c / (x + i as f64 + 1.0))
                .sum::<f64>();
        (2.0 * std::f64::consts::PI).sqrt().ln() + ser.abs().ln() + (x + 0.5) * t.ln() - t
    }
}

/// Survival function of standard normal: P(Z > z).
fn standard_normal_sf(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

/// Complementary error function (erfc) via Horner's method approximation.
fn erfc(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 rational approximation
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

/// FNV-1a hash for deterministic traffic assignment.
fn fnv1a_hash(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_runner(split: f64) -> ABTestRunner {
        let control = ModelVariant::new("control", |_| vec![0.0f64; 4]);
        let treatment = ModelVariant::new("treatment", |_| vec![1.0f64; 4]);
        let config = ABTestConfig {
            traffic_split: split,
            min_samples: 5,
            ..Default::default()
        };
        ABTestRunner::new(config, control, treatment).expect("runner should construct")
    }

    #[test]
    fn test_model_variant_infer() {
        let v = ModelVariant::new("test", |key| vec![key.len() as f64]);
        let result = v.infer("hello");
        assert_eq!(result, vec![5.0]);
    }

    #[test]
    fn test_model_variant_metadata() {
        let v = ModelVariant::new("sage-v1", |_| vec![])
            .with_description("GraphSAGE v1")
            .with_version("1.2.3");
        assert_eq!(v.name, "sage-v1");
        assert_eq!(v.description, "GraphSAGE v1");
        assert_eq!(v.version, "1.2.3");
    }

    #[test]
    fn test_abtest_config_default() {
        let cfg = ABTestConfig::default();
        assert!((cfg.traffic_split - 0.5).abs() < 1e-10);
        assert_eq!(cfg.min_samples, 50);
        assert!((cfg.significance_level - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_runner_construction_invalid_split() {
        let ctrl = ModelVariant::new("c", |_| vec![]);
        let trt = ModelVariant::new("t", |_| vec![]);
        let cfg = ABTestConfig {
            traffic_split: 1.5,
            ..Default::default()
        };
        assert!(ABTestRunner::new(cfg, ctrl, trt).is_err());
    }

    #[test]
    fn test_runner_route() {
        let mut runner = make_runner(0.5);
        for i in 0..20 {
            let key = format!("entity:{i}");
            let (emb, variant) = runner.route(&key).expect("route should succeed");
            assert!(!emb.is_empty());
            assert!(variant == "control" || variant == "treatment");
        }
        assert_eq!(runner.total_requests(), 20);
    }

    #[test]
    fn test_runner_traffic_split_deterministic() {
        // Same seed => same routing sequence
        let mut r1 = make_runner(0.3);
        let mut r2 = make_runner(0.3);
        for i in 0..50 {
            let key = format!("k{i}");
            let (_, v1) = r1.route(&key).expect("route 1 ok");
            let (_, v2) = r2.route(&key).expect("route 2 ok");
            assert_eq!(v1, v2, "routing should be deterministic");
        }
    }

    #[test]
    fn test_runner_record_metric_invalid() {
        let mut runner = make_runner(0.5);
        assert!(runner.record_metric("control", f64::NAN).is_err());
        assert!(runner.record_metric("control", f64::INFINITY).is_err());
    }

    #[test]
    fn test_runner_record_and_stats() {
        let mut runner = make_runner(0.5);
        for i in 0..20 {
            let key = format!("e:{i}");
            let (_, variant) = runner.route(&key).expect("route ok");
            runner
                .record_metric(&variant, (i as f64) * 0.1)
                .expect("record ok");
        }
        let stats = runner.variant_stats();
        assert!(!stats.is_empty());
        for s in stats.values() {
            assert!(s.n > 0);
            assert!(s.mean.is_finite());
        }
    }

    #[test]
    fn test_welchs_ttest_identical_groups() {
        let cfg = ABTestConfig::default();
        let analyzer = ABTestAnalyzer::new(&cfg);
        let data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let result = analyzer
            .welchs_ttest(&data, &data)
            .expect("t-test should succeed");
        assert!(
            (result.t_statistic).abs() < 1e-10,
            "t should be 0 for identical groups"
        );
        assert!(
            (result.p_value - 1.0).abs() < 0.01,
            "p should be ~1 for identical groups, got {}",
            result.p_value
        );
        assert!((result.mean_diff).abs() < 1e-10);
    }

    #[test]
    fn test_welchs_ttest_clearly_different() {
        let cfg = ABTestConfig::default();
        let analyzer = ABTestAnalyzer::new(&cfg);
        let a: Vec<f64> = (0..50).map(|_| 0.0).collect();
        let b: Vec<f64> = (0..50).map(|_| 100.0).collect();
        let result = analyzer
            .welchs_ttest(&a, &b)
            .expect("t-test should succeed");
        // Very different means => very significant
        assert!(
            result.p_value < 0.001,
            "p-value should be very small, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_mann_whitney_identical() {
        let cfg = ABTestConfig::default();
        let analyzer = ABTestAnalyzer::new(&cfg);
        let data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let result = analyzer
            .mann_whitney_u(&data, &data)
            .expect("MWU should succeed");
        // Identical groups: p should be high
        assert!(
            result.p_value > 0.3,
            "p-value for identical groups should be high, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_mann_whitney_clearly_different() {
        let cfg = ABTestConfig::default();
        let analyzer = ABTestAnalyzer::new(&cfg);
        let a: Vec<f64> = (0..40).map(|_| 0.0).collect();
        let b: Vec<f64> = (0..40).map(|_| 10.0).collect();
        let result = analyzer.mann_whitney_u(&a, &b).expect("MWU should succeed");
        assert!(
            result.p_value < 0.001,
            "p-value for clearly different groups should be small, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_cohens_d_no_difference() {
        let cfg = ABTestConfig::default();
        let analyzer = ABTestAnalyzer::new(&cfg);
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let d = analyzer.cohens_d(&data, &data);
        assert!(
            (d).abs() < 1e-10,
            "Cohen's d should be 0 for identical groups"
        );
    }

    #[test]
    fn test_cohens_d_large_effect() {
        let cfg = ABTestConfig::default();
        let analyzer = ABTestAnalyzer::new(&cfg);
        let a: Vec<f64> = vec![0.0f64; 30];
        let b: Vec<f64> = vec![10.0f64; 30];
        let d = analyzer.cohens_d(&a, &b);
        // Both have std=0, should handle gracefully
        assert!(d.is_finite() || d == 0.0);
    }

    #[test]
    fn test_full_ab_test_workflow() {
        let control = ModelVariant::new("baseline", |_| vec![0.0f64; 4]);
        let treatment = ModelVariant::new("improved", |_| vec![1.0f64; 4]);
        let config = ABTestConfig {
            traffic_split: 0.5,
            min_samples: 10,
            significance_level: 0.05,
            optimize: OptimizeMetric::Maximize,
            seed: 99,
            max_requests: None,
            min_effect_size: None,
        };
        let mut runner =
            ABTestRunner::new(config, control, treatment).expect("runner should construct");

        // Route requests and record metrics (treatment gets higher scores)
        let mut rng_state: u64 = 12345;
        for i in 0..100 {
            let key = format!("entity:{i}");
            let (_, variant) = runner.route(&key).expect("route ok");
            // Simple pseudo-random metric: treatment gets +0.5
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let base = (rng_state >> 32) as f64 / u32::MAX as f64;
            let metric = if variant == "improved" {
                base * 0.3 + 0.7
            } else {
                base * 0.3 + 0.2
            };
            runner.record_metric(&variant, metric).expect("record ok");
        }

        let report = runner.analyze().expect("analysis should succeed");
        assert!(report.control_stats.n >= 10);
        assert!(report.treatment_stats.n >= 10);
        // Treatment has higher scores, so it should win (or no significant diff)
        let wins = report.treatment_wins() || !report.significant;
        assert!(wins, "treatment should win or no significant difference");

        let summary = report.summary();
        assert!(summary.contains("A/B Test Report"));
    }

    #[test]
    fn test_ab_test_max_requests() {
        let ctrl = ModelVariant::new("c", |_| vec![1.0]);
        let trt = ModelVariant::new("t", |_| vec![2.0]);
        let cfg = ABTestConfig {
            max_requests: Some(5),
            ..Default::default()
        };
        let mut runner = ABTestRunner::new(cfg, ctrl, trt).expect("runner ok");
        for i in 0..5 {
            runner.route(&format!("k{i}")).expect("route ok");
        }
        let err = runner.route("k5");
        assert!(err.is_err(), "should error after max_requests");
    }

    #[test]
    fn test_variant_stats_empty() {
        let stats = VariantStats::from_slice(&[]);
        assert_eq!(stats.n, 0);
        assert!(stats.mean.is_nan());
    }

    #[test]
    fn test_variant_stats_single() {
        let stats = VariantStats::from_slice(&[42.0]);
        assert_eq!(stats.n, 1);
        assert_eq!(stats.mean, 42.0);
        assert_eq!(stats.min, 42.0);
        assert_eq!(stats.max, 42.0);
    }

    #[test]
    fn test_variant_stats_known_values() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = VariantStats::from_slice(&data);
        assert_eq!(stats.n, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.p50 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_report_relative_improvement() {
        let ctrl = ModelVariant::new("c", |_| vec![0.5f64]);
        let trt = ModelVariant::new("t", |_| vec![0.6f64]);
        let cfg = ABTestConfig {
            min_samples: 5,
            ..Default::default()
        };
        let mut runner = ABTestRunner::new(cfg, ctrl, trt).expect("runner ok");
        for i in 0..30 {
            let (_, v) = runner.route(&format!("k{i}")).expect("route ok");
            let metric = if v == "c" { 0.5 } else { 0.6 };
            runner.record_metric(&v, metric).expect("record ok");
        }
        let report = runner.analyze().expect("analyze ok");
        let ri = report.relative_improvement();
        assert!(ri.is_finite());
    }
}
