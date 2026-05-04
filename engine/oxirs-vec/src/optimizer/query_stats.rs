//! Persistent runtime statistics for online cost-model adaptation.
//!
//! Each query through the optimizer reports a [`QueryObservation`] capturing
//! the family it dispatched to, the wall-clock latency, and the observed
//! recall (if measurable).  [`QueryStats`] aggregates these observations into
//! per-family running averages and produces [`recommended_weights`] that the
//! cost model uses to correct systematic over/underestimates.
//!
//! Persistence uses **`serde_json`** so the file is grep-able for operators;
//! atomic writes go through a temporary `.tmp` sibling and a rename.
//!
//! [`recommended_weights`]: QueryStats::recommended_weights

use crate::optimizer::cost_model::{CostWeights, IndexFamily};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Single query observation reported back by the dispatcher after execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryObservation {
    /// Family that served the query.
    pub family: IndexFamily,
    /// Whether the query produced any results (used for hit/miss counting).
    pub hit: bool,
    /// Observed wall-clock latency in microseconds.
    pub latency_us: f64,
    /// Observed recall in `[0.0, 1.0]`, or `None` when ground truth is unknown.
    pub recall: Option<f32>,
    /// The cost the model predicted for this family at dispatch time.
    pub predicted_cost: f64,
}

impl QueryObservation {
    /// Convenience constructor.
    pub fn new(
        family: IndexFamily,
        hit: bool,
        latency_us: f64,
        recall: Option<f32>,
        predicted_cost: f64,
    ) -> Self {
        Self {
            family,
            hit,
            latency_us,
            recall,
            predicted_cost,
        }
    }
}

/// Per-family aggregate running statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct FamilyStats {
    /// Total number of queries dispatched to this family.
    pub queries: u64,
    /// Number of queries that returned at least one result.
    pub hits: u64,
    /// Sum of observed latencies (microseconds) — divide by `queries` for mean.
    pub total_latency_us: f64,
    /// Mean observed recall across queries that reported a recall.
    pub mean_recall: f64,
    /// Number of queries that contributed to `mean_recall`.
    pub recall_samples: u64,
    /// Mean predicted cost at dispatch time.
    pub mean_predicted_cost: f64,
}

impl FamilyStats {
    /// Mean latency in microseconds (returns 0.0 with no samples).
    pub fn mean_latency_us(&self) -> f64 {
        if self.queries == 0 {
            0.0
        } else {
            self.total_latency_us / self.queries as f64
        }
    }

    /// Hit rate in `[0.0, 1.0]` (returns 1.0 with no samples — assume best).
    pub fn hit_rate(&self) -> f64 {
        if self.queries == 0 {
            1.0
        } else {
            self.hits as f64 / self.queries as f64
        }
    }

    /// Update this aggregate with a single observation using running mean
    /// formulas (no buffer growth).
    fn update(&mut self, obs: &QueryObservation) {
        self.queries += 1;
        if obs.hit {
            self.hits += 1;
        }
        self.total_latency_us += obs.latency_us;

        // Running mean for predicted cost.
        let n = self.queries as f64;
        self.mean_predicted_cost =
            self.mean_predicted_cost + (obs.predicted_cost - self.mean_predicted_cost) / n;

        // Recall mean is updated only when the observation reports a recall.
        if let Some(r) = obs.recall {
            self.recall_samples += 1;
            let m = self.recall_samples as f64;
            self.mean_recall = self.mean_recall + (r as f64 - self.mean_recall) / m;
        }
    }
}

/// Aggregated statistics across all index families.
///
/// `version` is bumped when the on-disk layout changes; loaders refuse to
/// read incompatible versions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryStats {
    /// Storage format version.
    pub version: u32,
    /// Per-family running aggregates.
    pub families: BTreeMap<IndexFamily, FamilyStats>,
    /// Total number of observations recorded since creation.
    pub total_observations: u64,
}

impl Default for QueryStats {
    fn default() -> Self {
        let mut families = BTreeMap::new();
        for fam in IndexFamily::all() {
            families.insert(fam, FamilyStats::default());
        }
        Self {
            version: 1,
            families,
            total_observations: 0,
        }
    }
}

impl QueryStats {
    /// On-disk format version this build emits.
    pub const CURRENT_VERSION: u32 = 1;

    /// Construct an empty stats container.
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow stats for a specific family (always present after `default()`).
    pub fn family_stats(&self, family: IndexFamily) -> &FamilyStats {
        // BTreeMap was populated by Default::default() with all families;
        // if a deserialized file is missing one we fall back to a stable
        // pointer into a static cell to avoid panicking.
        self.families.get(&family).unwrap_or(&FALLBACK_FAMILY_STATS)
    }

    /// Record a new observation, updating running aggregates.
    pub fn record(&mut self, obs: QueryObservation) {
        let family = obs.family;
        let entry = self.families.entry(family).or_default();
        entry.update(&obs);
        self.total_observations += 1;
    }

    /// Recommend cost-model weights from accumulated observations.
    ///
    /// The weight for a family is set to `mean_observed_latency_us /
    /// mean_predicted_cost`, capped to the safe range enforced by
    /// [`CostWeights::set`].  Families with no observations keep their
    /// previous weights.
    ///
    /// Pass the current weights as `prior` so untouched families retain
    /// their existing values.
    pub fn recommended_weights(&self, prior: &CostWeights) -> CostWeights {
        let mut next = prior.clone();
        for fam in IndexFamily::all() {
            if let Some(stats) = self.families.get(&fam) {
                if stats.queries == 0 || stats.mean_predicted_cost <= 0.0 {
                    continue;
                }
                let mean_lat = stats.mean_latency_us();
                if mean_lat <= 0.0 {
                    continue;
                }
                let new_weight = mean_lat / stats.mean_predicted_cost;
                next.set(fam, new_weight);
            }
        }
        next
    }

    /// Serialise to a JSON file, atomically replacing any existing copy.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("QueryStats::save: failed to create parent dir {:?}", parent)
                })?;
            }
        }
        let tmp_path = tmp_sibling(path);
        let json = serde_json::to_string_pretty(self)
            .context("QueryStats::save: serde_json encode failed")?;
        fs::write(&tmp_path, json).with_context(|| {
            format!("QueryStats::save: write to temp file {:?} failed", tmp_path)
        })?;
        fs::rename(&tmp_path, path).with_context(|| {
            format!(
                "QueryStats::save: rename {:?} -> {:?} failed",
                tmp_path, path
            )
        })?;
        Ok(())
    }

    /// Load from a JSON file.  Refuses to read versions newer than this build.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            fs::read(path).with_context(|| format!("QueryStats::load: read {:?} failed", path))?;
        let stats: QueryStats = serde_json::from_slice(&bytes)
            .with_context(|| format!("QueryStats::load: parse {:?} failed", path))?;
        if stats.version > Self::CURRENT_VERSION {
            return Err(anyhow!(
                "QueryStats::load: version {} is newer than this build's {}",
                stats.version,
                Self::CURRENT_VERSION
            ));
        }
        Ok(stats)
    }
}

/// Fallback value returned by `family_stats()` when a family is absent
/// from a deserialized file (defensive — `default()` populates all families).
static FALLBACK_FAMILY_STATS: FamilyStats = FamilyStats {
    queries: 0,
    hits: 0,
    total_latency_us: 0.0,
    mean_recall: 0.0,
    recall_samples: 0,
    mean_predicted_cost: 0.0,
};

/// Compute the temporary sibling file path used during atomic writes.
fn tmp_sibling(path: &Path) -> PathBuf {
    let mut tmp = path.to_path_buf();
    let file_name = path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "query_stats".to_string());
    tmp.set_file_name(format!("{}.tmp", file_name));
    tmp
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn unique_path(label: &str) -> PathBuf {
        let mut p = temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("oxirs_vec_optstats_{}_{}.json", label, stamp));
        p
    }

    #[test]
    fn family_stats_default_is_zeroed() {
        let s = FamilyStats::default();
        assert_eq!(s.queries, 0);
        assert_eq!(s.hits, 0);
        assert_eq!(s.total_latency_us, 0.0);
        assert!(s.mean_recall.abs() < 1e-12);
        assert!(s.hit_rate() == 1.0); // no data → assume best
    }

    #[test]
    fn record_updates_running_means() {
        let mut stats = QueryStats::new();
        stats.record(QueryObservation::new(
            IndexFamily::Hnsw,
            true,
            100.0,
            Some(0.95),
            80.0,
        ));
        stats.record(QueryObservation::new(
            IndexFamily::Hnsw,
            true,
            200.0,
            Some(0.93),
            80.0,
        ));
        let s = stats.family_stats(IndexFamily::Hnsw);
        assert_eq!(s.queries, 2);
        assert_eq!(s.hits, 2);
        assert!((s.mean_latency_us() - 150.0).abs() < 1e-6);
        assert!((s.mean_recall - 0.94).abs() < 1e-3);
        assert_eq!(stats.total_observations, 2);
    }

    #[test]
    fn record_handles_missing_recall() {
        let mut stats = QueryStats::new();
        stats.record(QueryObservation::new(
            IndexFamily::Lsh,
            true,
            50.0,
            None,
            40.0,
        ));
        let s = stats.family_stats(IndexFamily::Lsh);
        assert_eq!(s.queries, 1);
        assert_eq!(s.recall_samples, 0);
        assert!(s.mean_recall.abs() < 1e-12);
    }

    #[test]
    fn hit_rate_reflects_misses() {
        let mut stats = QueryStats::new();
        stats.record(QueryObservation::new(
            IndexFamily::Pq,
            true,
            10.0,
            None,
            10.0,
        ));
        stats.record(QueryObservation::new(
            IndexFamily::Pq,
            false,
            12.0,
            None,
            10.0,
        ));
        stats.record(QueryObservation::new(
            IndexFamily::Pq,
            false,
            14.0,
            None,
            10.0,
        ));
        let r = stats.family_stats(IndexFamily::Pq).hit_rate();
        assert!((r - (1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn recommended_weights_derive_from_observed_vs_predicted() {
        let mut stats = QueryStats::new();
        // Observed average 200µs, predicted 100 → weight should be 2.0.
        for _ in 0..10 {
            stats.record(QueryObservation::new(
                IndexFamily::Hnsw,
                true,
                200.0,
                Some(0.95),
                100.0,
            ));
        }
        let w = stats.recommended_weights(&CostWeights::default());
        assert!((w.get(IndexFamily::Hnsw) - 2.0).abs() < 1e-6);
        // Untouched families keep prior 1.0.
        assert!((w.get(IndexFamily::Ivf) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn recommended_weights_clamped_for_outliers() {
        let mut stats = QueryStats::new();
        // Predicted near zero → would yield enormous weight; must clamp.
        stats.record(QueryObservation::new(
            IndexFamily::Lsh,
            true,
            5_000.0,
            None,
            0.001,
        ));
        let w = stats.recommended_weights(&CostWeights::default());
        // Clamp ceiling is 20.0 in CostWeights::set.
        assert!((w.get(IndexFamily::Lsh) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn save_load_roundtrip() -> Result<()> {
        let path = unique_path("roundtrip");
        let mut original = QueryStats::new();
        original.record(QueryObservation::new(
            IndexFamily::Ivf,
            true,
            150.0,
            Some(0.91),
            120.0,
        ));
        original.save(&path)?;
        let loaded = QueryStats::load(&path)?;
        // JSON serialisation can introduce ≤1 ULP float drift on the recall
        // mean (f32→f64→f32); fields-level equality with epsilon comparison
        // is the right test, not bitwise.
        assert_eq!(loaded.version, original.version);
        assert_eq!(loaded.total_observations, original.total_observations);
        let lhs = loaded.family_stats(IndexFamily::Ivf);
        let rhs = original.family_stats(IndexFamily::Ivf);
        assert_eq!(lhs.queries, rhs.queries);
        assert_eq!(lhs.hits, rhs.hits);
        assert!((lhs.total_latency_us - rhs.total_latency_us).abs() < 1e-9);
        assert!((lhs.mean_recall - rhs.mean_recall).abs() < 1e-6);
        assert_eq!(lhs.recall_samples, rhs.recall_samples);
        assert!((lhs.mean_predicted_cost - rhs.mean_predicted_cost).abs() < 1e-9);
        let _ = fs::remove_file(&path);
        Ok(())
    }

    #[test]
    fn load_rejects_future_version() -> Result<()> {
        let path = unique_path("future");
        let mut stats = QueryStats::new();
        stats.version = QueryStats::CURRENT_VERSION + 1;
        let json = serde_json::to_string_pretty(&stats)?;
        fs::write(&path, json)?;
        let res = QueryStats::load(&path);
        assert!(res.is_err(), "future version must be rejected");
        let _ = fs::remove_file(&path);
        Ok(())
    }

    #[test]
    fn load_rejects_corrupt_json() {
        let path = unique_path("corrupt");
        fs::write(&path, b"{not json}").expect("temp write");
        let res = QueryStats::load(&path);
        assert!(res.is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn fallback_returned_for_missing_family() {
        // Construct stats *without* using Default to simulate an old
        // serialization that omitted some families.
        let stats = QueryStats {
            version: 1,
            families: BTreeMap::new(),
            total_observations: 0,
        };
        let s = stats.family_stats(IndexFamily::Hnsw);
        assert_eq!(s.queries, 0);
    }
}
