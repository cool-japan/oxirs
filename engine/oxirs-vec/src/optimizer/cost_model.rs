//! Per-index cost formulas for vector search query optimization.
//!
//! Each formula reflects the asymptotic search cost of a specific approximate
//! nearest-neighbour index, expressed as **expected number of distance
//! computations** for a single k-NN query.  Distance computations dominate
//! query latency for vector search, so distance-count is a robust proxy for
//! wall-clock time.
//!
//! The formulas implemented here are:
//!
//! - **HNSW**: `O(log n × M × ef)` — `ef * (M * log n)` expected probes
//!   through the hierarchy.
//! - **IVF**:  `O(n / nprobe + n_clusters)` — `n_clusters` for coarse
//!   centroid selection, then `n / nprobe` for fine search inside the
//!   probed cells.
//! - **LSH**:  `O(L × bucket_size + K × L × dim)` — `K * L * dim` for hash
//!   evaluation and `L * bucket_size` for candidate scoring.
//! - **PQ**:   `O(centroids × subquantizers + n × subquantizers)` — codebook
//!   lookup table build then asymmetric distance against every encoded vector.
//!
//! Each cost is multiplied by a tunable per-index weight obtained from
//! historical statistics, enabling online cost-model adaptation when actual
//! latency measurements drift from the static formula.
//!
//! All costs are expressed in **abstract distance-equivalent units**.  Higher
//! is more expensive; the dispatcher picks the lowest-cost index that meets
//! the requested recall.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Index families recognised by the optimizer cost model.
///
/// Each variant maps to a concrete cost formula encoded in
/// [`CostModel::estimate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IndexFamily {
    /// Hierarchical Navigable Small World graph.
    Hnsw,
    /// Inverted File index (Voronoi-cell partitioning).
    Ivf,
    /// Locality-Sensitive Hashing.
    Lsh,
    /// Product Quantization (asymmetric distance against coded vectors).
    Pq,
}

impl IndexFamily {
    /// All families known to the cost model.
    pub fn all() -> [IndexFamily; 4] {
        [
            IndexFamily::Hnsw,
            IndexFamily::Ivf,
            IndexFamily::Lsh,
            IndexFamily::Pq,
        ]
    }

    /// Stable string identifier for serialization and metrics.
    pub fn as_str(&self) -> &'static str {
        match self {
            IndexFamily::Hnsw => "hnsw",
            IndexFamily::Ivf => "ivf",
            IndexFamily::Lsh => "lsh",
            IndexFamily::Pq => "pq",
        }
    }
}

/// Workload characteristics the cost model needs to estimate per-index cost.
///
/// `data_size` is the total number of indexed vectors; `dim` is the
/// dimensionality; `requested_recall` is in `[0.0, 1.0]`; `query_density`
/// expresses the expected fraction of the dataset that satisfies the query
/// predicate (e.g. 1.0 = no filtering, 0.1 = 10% of data is candidate).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkloadProfile {
    /// Number of vectors currently indexed.
    pub data_size: usize,
    /// Vector dimensionality.
    pub dim: usize,
    /// Minimum acceptable recall (0.0 to 1.0).
    pub requested_recall: f32,
    /// Expected fraction of data passing pre-filter (0.0 to 1.0).
    ///
    /// Use `1.0` for unfiltered queries.  Lower values indicate selective
    /// filters that should bias the optimizer toward indices that benefit
    /// from filtering (LSH and PQ degrade gracefully; HNSW does not).
    pub query_density: f32,
    /// Number of nearest neighbours requested.
    pub k: usize,
}

impl WorkloadProfile {
    /// Create a profile with sensible defaults: density=1.0 (unfiltered), k=10.
    pub fn new(data_size: usize, dim: usize, requested_recall: f32) -> Self {
        Self {
            data_size,
            dim,
            requested_recall,
            query_density: 1.0,
            k: 10,
        }
    }

    /// Set the query density (filter selectivity).
    pub fn with_query_density(mut self, density: f32) -> Self {
        self.query_density = density.clamp(0.0, 1.0);
        self
    }

    /// Set the number of nearest neighbours.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k.max(1);
        self
    }
}

/// Tunable per-index parameters that affect the cost formula.
///
/// Defaults match the canonical defaults used elsewhere in the crate:
/// `HnswConfig::m = 16`, `HnswConfig::ef = 50`, `IvfConfig::n_clusters = 256`,
/// `IvfConfig::n_probes = 8`, `LshConfig::num_tables = 10`,
/// `LshConfig::num_hash_functions = 8`, `PQConfig::n_subquantizers = 8`,
/// `PQConfig::n_centroids = 256`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexParameters {
    /// HNSW max degree per layer.
    pub hnsw_m: usize,
    /// HNSW search beam width.
    pub hnsw_ef: usize,
    /// IVF coarse centroids.
    pub ivf_n_clusters: usize,
    /// IVF cells probed per query.
    pub ivf_n_probes: usize,
    /// LSH hash tables.
    pub lsh_tables: usize,
    /// LSH hash functions per table.
    pub lsh_hash_functions: usize,
    /// LSH expected bucket size (data_size / (tables * 2^hash_functions) clamped).
    pub lsh_avg_bucket_size: usize,
    /// PQ subquantizers.
    pub pq_subquantizers: usize,
    /// PQ centroids per subquantizer.
    pub pq_centroids: usize,
}

impl Default for IndexParameters {
    fn default() -> Self {
        Self {
            hnsw_m: 16,
            hnsw_ef: 50,
            ivf_n_clusters: 256,
            ivf_n_probes: 8,
            lsh_tables: 10,
            lsh_hash_functions: 8,
            lsh_avg_bucket_size: 64,
            pq_subquantizers: 8,
            pq_centroids: 256,
        }
    }
}

/// Online-learnable weights applied to each per-index formula output.
///
/// `weight = 1.0` means "trust the formula"; higher values indicate the
/// formula systematically underestimates real latency for that family;
/// lower values indicate it overestimates.  Weights are updated by
/// [`crate::optimizer::query_stats::QueryStats::recommended_weights`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CostWeights {
    weights: BTreeMap<IndexFamily, f64>,
}

impl Default for CostWeights {
    fn default() -> Self {
        let mut weights = BTreeMap::new();
        for fam in IndexFamily::all() {
            weights.insert(fam, 1.0);
        }
        Self { weights }
    }
}

impl CostWeights {
    /// Get weight for a family (defaults to 1.0 if not yet set).
    pub fn get(&self, family: IndexFamily) -> f64 {
        self.weights.get(&family).copied().unwrap_or(1.0)
    }

    /// Set weight for a family.  Values are clamped to `[0.05, 20.0]` to
    /// prevent runaway feedback loops from outlier observations.
    pub fn set(&mut self, family: IndexFamily, weight: f64) {
        let clamped = weight.clamp(0.05, 20.0);
        self.weights.insert(family, clamped);
    }
}

/// Expected recall floor per family at default parameters.
///
/// These are conservative empirical lower bounds — the dispatcher uses them
/// to filter out indices that *cannot meet* the requested recall before any
/// cost comparison is performed.  Indices with adaptive parameters (HNSW
/// `ef`, IVF `nprobe`) can usually exceed these floors when tuned.
fn expected_recall_floor(family: IndexFamily) -> f32 {
    match family {
        IndexFamily::Hnsw => 0.95,
        IndexFamily::Ivf => 0.85,
        IndexFamily::Lsh => 0.75,
        IndexFamily::Pq => 0.88,
    }
}

/// Cost-model entrypoint used by the optimizer dispatcher.
#[derive(Debug, Clone, Default)]
pub struct CostModel {
    parameters: IndexParameters,
    weights: CostWeights,
}

impl CostModel {
    /// Construct a cost model with explicit parameters and weights.
    pub fn new(parameters: IndexParameters, weights: CostWeights) -> Self {
        Self {
            parameters,
            weights,
        }
    }

    /// Return mutable access to the weights for online adaptation.
    pub fn weights_mut(&mut self) -> &mut CostWeights {
        &mut self.weights
    }

    /// Borrow the current weights.
    pub fn weights(&self) -> &CostWeights {
        &self.weights
    }

    /// Borrow the current parameters.
    pub fn parameters(&self) -> &IndexParameters {
        &self.parameters
    }

    /// Recall floor that an index family is expected to deliver at its
    /// default parameter configuration.
    pub fn recall_floor(family: IndexFamily) -> f32 {
        expected_recall_floor(family)
    }

    /// Estimate the (cost, recall) pair for executing `workload` against
    /// `family`.
    ///
    /// `cost` is in abstract distance-equivalent units (higher = slower);
    /// `recall` is a 0..1 estimate.  Cost is multiplied by the
    /// learned per-family weight.
    pub fn estimate(&self, family: IndexFamily, workload: &WorkloadProfile) -> CostEstimate {
        // Scale by query density: highly selective queries benefit from
        // filtered scans; less selective queries amortize over more probes.
        // Density of 1.0 means no scaling (the default unfiltered query).
        let density_scale = (workload.query_density.clamp(0.01, 1.0)) as f64;
        let n = workload.data_size.max(1) as f64;
        let dim = workload.dim.max(1) as f64;
        let k = workload.k.max(1) as f64;

        let raw_cost = match family {
            IndexFamily::Hnsw => self.estimate_hnsw(n, k),
            IndexFamily::Ivf => self.estimate_ivf(n),
            IndexFamily::Lsh => self.estimate_lsh(dim),
            IndexFamily::Pq => self.estimate_pq(n),
        };

        // Density boosts indices that benefit from filtering (LSH, PQ) and
        // penalises HNSW which has no native filtering primitive.
        let density_factor = match family {
            IndexFamily::Hnsw => 1.0 / density_scale.max(0.1),
            IndexFamily::Ivf => 1.0,
            IndexFamily::Lsh => density_scale.max(0.5),
            IndexFamily::Pq => density_scale.max(0.5),
        };

        let weight = self.weights.get(family);
        let cost = raw_cost * weight * density_factor;

        // Estimate recall: at default parameters use the family floor;
        // adapt slightly based on dim and requested_recall.
        let recall = self.estimate_recall(family, workload);

        CostEstimate {
            family,
            cost,
            recall,
        }
    }

    /// HNSW: `ef * M * log n + k`
    fn estimate_hnsw(&self, n: f64, k: f64) -> f64 {
        let p = &self.parameters;
        let log_n = n.ln().max(1.0);
        (p.hnsw_ef as f64) * (p.hnsw_m as f64) * log_n + k
    }

    /// IVF: `n_clusters + (n / max(nprobe, 1)) * (nprobe / n_clusters)`
    /// = `n_clusters + n / n_clusters * nprobe` …simplifies to
    /// `n_clusters + n * (n_probes / n_clusters)`.
    fn estimate_ivf(&self, n: f64) -> f64 {
        let p = &self.parameters;
        let n_clusters = p.ivf_n_clusters.max(1) as f64;
        let n_probes = p.ivf_n_probes.max(1) as f64;
        n_clusters + n * (n_probes / n_clusters)
    }

    /// LSH: `K * L * dim + L * avg_bucket_size`
    fn estimate_lsh(&self, dim: f64) -> f64 {
        let p = &self.parameters;
        let l = p.lsh_tables.max(1) as f64;
        let kk = p.lsh_hash_functions.max(1) as f64;
        let bucket = p.lsh_avg_bucket_size.max(1) as f64;
        kk * l * dim + l * bucket
    }

    /// PQ: `centroids * subquantizers + n * subquantizers / 8`
    /// (codebook precompute, then table lookup per coded vector).
    fn estimate_pq(&self, n: f64) -> f64 {
        let p = &self.parameters;
        let cents = p.pq_centroids.max(1) as f64;
        let subs = p.pq_subquantizers.max(1) as f64;
        cents * subs + n * subs / 8.0
    }

    /// Estimate recall for a family at current parameters.
    fn estimate_recall(&self, family: IndexFamily, workload: &WorkloadProfile) -> f32 {
        let floor = expected_recall_floor(family);
        // Tighter beam widths/probes lift recall closer to 1.0.
        let lift = match family {
            IndexFamily::Hnsw => {
                let ef = self.parameters.hnsw_ef as f32;
                ((ef - 32.0) / 200.0).clamp(0.0, 0.04)
            }
            IndexFamily::Ivf => {
                let probes = self.parameters.ivf_n_probes as f32;
                ((probes - 4.0) / 64.0).clamp(0.0, 0.08)
            }
            IndexFamily::Lsh => {
                let l = self.parameters.lsh_tables as f32;
                ((l - 4.0) / 64.0).clamp(0.0, 0.10)
            }
            IndexFamily::Pq => {
                let cents = self.parameters.pq_centroids as f32;
                ((cents - 64.0) / 1024.0).clamp(0.0, 0.06)
            }
        };
        // Higher dimensionality slightly degrades approximate recall.
        let dim_penalty = if workload.dim > 512 {
            ((workload.dim as f32 - 512.0) / 4096.0).min(0.05)
        } else {
            0.0
        };
        (floor + lift - dim_penalty).clamp(0.0, 1.0)
    }
}

/// One row of cost-model output: family + estimated cost + estimated recall.
#[derive(Debug, Clone, PartialEq)]
pub struct CostEstimate {
    /// Index family being estimated.
    pub family: IndexFamily,
    /// Abstract cost units (lower is better).
    pub cost: f64,
    /// Estimated recall in `[0.0, 1.0]`.
    pub recall: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workload(n: usize, dim: usize, recall: f32) -> WorkloadProfile {
        WorkloadProfile::new(n, dim, recall)
    }

    #[test]
    fn index_family_all_returns_four_distinct() {
        let all = IndexFamily::all();
        assert_eq!(all.len(), 4);
        let strs: Vec<_> = all.iter().map(|f| f.as_str()).collect();
        assert_eq!(strs, vec!["hnsw", "ivf", "lsh", "pq"]);
    }

    #[test]
    fn cost_weights_default_is_unit() {
        let w = CostWeights::default();
        for f in IndexFamily::all() {
            assert!((w.get(f) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn cost_weights_set_clamps_outliers() {
        let mut w = CostWeights::default();
        w.set(IndexFamily::Hnsw, 1000.0);
        assert!((w.get(IndexFamily::Hnsw) - 20.0).abs() < 1e-12);
        w.set(IndexFamily::Pq, 0.0);
        assert!((w.get(IndexFamily::Pq) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn hnsw_cost_grows_with_log_n() {
        let cm = CostModel::default();
        let small = cm.estimate(IndexFamily::Hnsw, &workload(1_000, 128, 0.9));
        let large = cm.estimate(IndexFamily::Hnsw, &workload(1_000_000, 128, 0.9));
        assert!(
            large.cost > small.cost,
            "HNSW cost must grow with data size"
        );
        // Log scaling: 1000x more data should be only ~2x more cost.
        assert!(large.cost < small.cost * 4.0);
    }

    #[test]
    fn ivf_cost_grows_with_n() {
        let cm = CostModel::default();
        let small = cm.estimate(IndexFamily::Ivf, &workload(10_000, 128, 0.9));
        let large = cm.estimate(IndexFamily::Ivf, &workload(1_000_000, 128, 0.9));
        // IVF is roughly linear (n_probes / n_clusters fraction of n).
        assert!(large.cost > small.cost);
        assert!(large.cost > small.cost * 10.0);
    }

    #[test]
    fn lsh_cost_independent_of_n() {
        let cm = CostModel::default();
        let small = cm.estimate(IndexFamily::Lsh, &workload(1_000, 128, 0.8));
        let large = cm.estimate(IndexFamily::Lsh, &workload(1_000_000, 128, 0.8));
        // LSH has bucket scan, but average bucket size is parameterised
        // independently from n in our cost model.
        assert!((large.cost - small.cost).abs() < 1e-9);
    }

    #[test]
    fn pq_cost_grows_with_n() {
        let cm = CostModel::default();
        let small = cm.estimate(IndexFamily::Pq, &workload(1_000, 128, 0.9));
        let large = cm.estimate(IndexFamily::Pq, &workload(100_000, 128, 0.9));
        assert!(large.cost > small.cost);
    }

    #[test]
    fn weights_scale_cost_linearly() {
        let mut cm = CostModel::default();
        let baseline = cm.estimate(IndexFamily::Hnsw, &workload(10_000, 128, 0.9));
        cm.weights_mut().set(IndexFamily::Hnsw, 2.0);
        let scaled = cm.estimate(IndexFamily::Hnsw, &workload(10_000, 128, 0.9));
        assert!((scaled.cost - 2.0 * baseline.cost).abs() < 1e-6);
    }

    #[test]
    fn recall_floors_match_expectations() {
        assert!((CostModel::recall_floor(IndexFamily::Hnsw) - 0.95).abs() < 1e-6);
        assert!((CostModel::recall_floor(IndexFamily::Pq) - 0.88).abs() < 1e-6);
        assert!(
            CostModel::recall_floor(IndexFamily::Lsh) < CostModel::recall_floor(IndexFamily::Hnsw)
        );
    }

    #[test]
    fn high_dim_penalises_recall_estimate() {
        let cm = CostModel::default();
        let low_dim = cm.estimate(IndexFamily::Hnsw, &workload(10_000, 128, 0.9));
        let high_dim = cm.estimate(IndexFamily::Hnsw, &workload(10_000, 4096, 0.9));
        assert!(high_dim.recall < low_dim.recall);
    }

    #[test]
    fn density_biases_toward_filterable_indices() {
        let cm = CostModel::default();
        let unfiltered = cm.estimate(
            IndexFamily::Hnsw,
            &workload(10_000, 128, 0.9).with_query_density(1.0),
        );
        let very_selective = cm.estimate(
            IndexFamily::Hnsw,
            &workload(10_000, 128, 0.9).with_query_density(0.05),
        );
        // HNSW gets more expensive when the query is very selective
        // because it cannot exploit the filter.
        assert!(very_selective.cost > unfiltered.cost);
    }

    #[test]
    fn density_helps_lsh_and_pq() {
        let cm = CostModel::default();
        let unfiltered = cm.estimate(
            IndexFamily::Lsh,
            &workload(10_000, 128, 0.8).with_query_density(1.0),
        );
        let selective = cm.estimate(
            IndexFamily::Lsh,
            &workload(10_000, 128, 0.8).with_query_density(0.5),
        );
        assert!(selective.cost <= unfiltered.cost);
    }
}
