//! Equi-depth histogram statistics for cardinality estimation in the cost-based optimizer.
//!
//! This module provides column-level statistics (histograms) over predicate-object distributions
//! in an RDF dataset, enabling the optimizer to make better cardinality estimates when planning
//! join orders and filter evaluation.
//!
//! # Design
//!
//! Each predicate gets its own [`PredicateHistogram`] that maintains an equi-depth histogram over
//! the distribution of object values observed for that predicate. The histogram is built from raw
//! value streams using a sorting-and-splitting approach, and can be updated incrementally (with a
//! degraded accuracy guarantee) as new triples arrive.
//!
//! A [`DatasetStatistics`] struct aggregates all per-predicate histograms and provides the
//! dataset-level estimates (subject/predicate/object counts) used by the optimizer.

use std::collections::HashMap;
use std::time::Instant;

/// A single bucket in an equi-depth histogram.
///
/// All values in the range `[lower_bound, upper_bound]` (lexicographic) fall into this bucket.
#[derive(Debug, Clone)]
pub struct HistogramBucket {
    /// Minimum value in this bucket (inclusive, string-encoded).
    pub lower_bound: String,
    /// Maximum value in this bucket (inclusive, string-encoded).
    pub upper_bound: String,
    /// Approximate number of values (triples) whose object falls in this bucket.
    pub count: u64,
    /// Approximate number of distinct values in this bucket.
    pub distinct_count: u64,
    /// Number of null / missing values encountered when building this bucket.
    pub null_count: u64,
}

impl HistogramBucket {
    /// Returns true if `value` falls within the closed range `[lower_bound, upper_bound]`.
    pub fn contains(&self, value: &str) -> bool {
        value >= self.lower_bound.as_str() && value <= self.upper_bound.as_str()
    }

    /// Returns the fraction `[0, 1]` of the bucket that is "covered" by the closed range
    /// `[lo, hi]`.  Used for range selectivity estimation with linear interpolation.
    pub fn overlap_fraction(&self, lo: &str, hi: &str) -> f64 {
        // Both empty: full coverage.
        if self.lower_bound.is_empty() && self.upper_bound.is_empty() {
            return 1.0;
        }
        let lb = self.lower_bound.as_str();
        let ub = self.upper_bound.as_str();

        // No overlap at all.
        if hi < lb || lo > ub {
            return 0.0;
        }

        // Full containment.
        if lo <= lb && hi >= ub {
            return 1.0;
        }

        // Partial overlap — approximate via linear interpolation over lexicographic key space.
        // We convert strings to a rough u64 "position" by looking at the first 8 bytes.
        let to_pos = |s: &str| -> u64 {
            let bytes = s.as_bytes();
            let mut v: u64 = 0;
            for (i, &b) in bytes.iter().enumerate().take(8) {
                v |= (b as u64) << (56 - i * 8);
            }
            v
        };

        let pos_lb = to_pos(lb);
        let pos_ub = to_pos(ub);
        let pos_lo = to_pos(lo).max(pos_lb);
        let pos_hi = to_pos(hi).min(pos_ub);

        if pos_ub == pos_lb {
            return 1.0;
        }

        let overlap = pos_hi.saturating_sub(pos_lo) as f64;
        let total = (pos_ub - pos_lb) as f64;
        (overlap / total).clamp(0.0, 1.0)
    }
}

/// Equi-depth histogram for the object values of a specific predicate.
///
/// All buckets have approximately equal `count` (depth), spread across the ordered value space.
#[derive(Debug, Clone)]
pub struct PredicateHistogram {
    /// IRI of the predicate this histogram covers.
    pub predicate_iri: String,
    /// Total number of non-null values seen.
    pub total_count: u64,
    /// Number of null / missing object values.
    pub null_count: u64,
    /// Number of globally distinct object values (approximate, from build phase).
    pub distinct_count: u64,
    /// Equi-depth buckets (sorted by lower_bound ascending).
    pub buckets: Vec<HistogramBucket>,
}

impl PredicateHistogram {
    /// Create an empty histogram for `predicate` with `num_buckets` bucket slots.
    pub fn new(predicate: &str, num_buckets: usize) -> Self {
        Self {
            predicate_iri: predicate.to_owned(),
            total_count: 0,
            null_count: 0,
            distinct_count: 0,
            buckets: Vec::with_capacity(num_buckets),
        }
    }

    /// Build an equi-depth histogram from `values`, targeting `num_buckets` buckets.
    ///
    /// The approach is:
    /// 1. Sort all (non-empty) values lexicographically.
    /// 2. Split into `num_buckets` equal-sized slices.
    /// 3. For each slice compute distinct count via deduplication.
    ///
    /// Complexity: O(n log n) where n = `values.len()`.
    pub fn build_from_values(predicate: &str, values: &[String], num_buckets: usize) -> Self {
        let num_buckets = num_buckets.max(1);

        let null_count = values.iter().filter(|v| v.is_empty()).count() as u64;

        let mut sorted: Vec<&String> = values.iter().filter(|v| !v.is_empty()).collect();
        sorted.sort_unstable();

        let total_count = sorted.len() as u64;

        // Count distinct values.
        let mut distinct_count = 0u64;
        {
            let mut prev: Option<&str> = None;
            for v in &sorted {
                if prev != Some(v.as_str()) {
                    distinct_count += 1;
                    prev = Some(v.as_str());
                }
            }
        }

        let mut buckets: Vec<HistogramBucket> = Vec::with_capacity(num_buckets);

        if sorted.is_empty() {
            return Self {
                predicate_iri: predicate.to_owned(),
                total_count,
                null_count,
                distinct_count,
                buckets,
            };
        }

        let target_depth = (sorted.len() as f64 / num_buckets as f64).ceil() as usize;
        let target_depth = target_depth.max(1);

        let mut cursor = 0usize;
        while cursor < sorted.len() {
            let end = (cursor + target_depth).min(sorted.len());
            let slice = &sorted[cursor..end];

            // Move `end` forward to not split equal values across adjacent buckets.
            let lower_bound = slice[0].clone();
            let upper_bound = slice[slice.len() - 1].clone();

            // Distinct count in this bucket.
            let mut bucket_distinct = 0u64;
            let mut prev: Option<&str> = None;
            for v in slice {
                if prev != Some(v.as_str()) {
                    bucket_distinct += 1;
                    prev = Some(v.as_str());
                }
            }

            buckets.push(HistogramBucket {
                lower_bound,
                upper_bound,
                count: slice.len() as u64,
                distinct_count: bucket_distinct,
                null_count: 0,
            });

            cursor = end;
        }

        Self {
            predicate_iri: predicate.to_owned(),
            total_count,
            null_count,
            distinct_count,
            buckets,
        }
    }

    /// Estimate the selectivity (fraction of rows) for an equality predicate `object = value`.
    ///
    /// Returns a value in `[0, 1]`.  Uses a uniform assumption within the containing bucket.
    pub fn estimate_selectivity_eq(&self, value: &str) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        for bucket in &self.buckets {
            if bucket.contains(value) {
                let distinct = bucket.distinct_count.max(1) as f64;
                let bucket_sel = bucket.count as f64 / self.total_count as f64;
                // Assume uniform distribution within bucket.
                return bucket_sel / distinct;
            }
        }

        // Value not in any bucket — assume extremely rare.
        // Use (distinct_count + 1) as denominator to model "one extra unseen value beyond all known
        // distinct ones", which is strictly less than 1/distinct_count and much smaller than any
        // in-bucket estimate.  Adding 1 prevents the edge case where distinct_count == total_count
        // would yield exactly 1/total_count (i.e., 0.01 for 10 distinct values in 10 total).
        let denominator =
            (self.distinct_count.max(1) + 1) as f64 * (self.total_count.max(1) as f64);
        1.0 / denominator
    }

    /// Estimate the selectivity for a range predicate `lo <= object <= hi`.
    ///
    /// Returns a value in `[0, 1]`.
    pub fn estimate_selectivity_range(&self, lo: &str, hi: &str) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        let mut covered_rows: f64 = 0.0;
        for bucket in &self.buckets {
            let frac = bucket.overlap_fraction(lo, hi);
            if frac > 0.0 {
                covered_rows += frac * bucket.count as f64;
            }
        }

        (covered_rows / self.total_count as f64).clamp(0.0, 1.0)
    }

    /// Estimate the expected cardinality for a pattern that filters this predicate's objects.
    ///
    /// If `predicate_filter` is `Some(value)` an equality filter is applied; otherwise all
    /// triples with this predicate are returned.
    ///
    /// `total_triples` is used as a fallback denominator when `self.total_count` is zero.
    pub fn estimate_cardinality(&self, total_triples: u64, predicate_filter: Option<&str>) -> u64 {
        let base = if self.total_count > 0 {
            self.total_count
        } else {
            total_triples
        };

        if base == 0 {
            return 0;
        }

        match predicate_filter {
            None => base,
            Some(value) => {
                let sel = self.estimate_selectivity_eq(value);
                ((base as f64 * sel).ceil() as u64).max(1)
            }
        }
    }

    /// Incrementally update the histogram with additional observed values.
    ///
    /// This is an approximate update: new values are inserted into the most appropriate existing
    /// bucket (lowest `lower_bound` not greater than the value), and global counts are adjusted.
    /// After many incremental updates, accuracy degrades and a full rebuild is recommended.
    pub fn update_incremental(&mut self, new_values: &[String]) {
        for value in new_values {
            if value.is_empty() {
                self.null_count += 1;
                continue;
            }

            self.total_count += 1;

            // Find the bucket whose range contains the value, or the last bucket.
            let idx = self
                .buckets
                .iter()
                .rposition(|b| b.lower_bound.as_str() <= value.as_str())
                .unwrap_or(0);

            if let Some(bucket) = self.buckets.get_mut(idx) {
                bucket.count += 1;
                // Heuristic: if value is outside current range, expand bounds.
                if value.as_str() < bucket.lower_bound.as_str() {
                    bucket.lower_bound = value.clone();
                }
                if value.as_str() > bucket.upper_bound.as_str() {
                    bucket.upper_bound = value.clone();
                    // Ripple update: adjust distinct count (approximate).
                    bucket.distinct_count += 1;
                    self.distinct_count += 1;
                }
            } else if !self.buckets.is_empty() {
                // Append to the last bucket.
                let last = self
                    .buckets
                    .last_mut()
                    .expect("buckets non-empty checked above");
                last.count += 1;
                if value.as_str() > last.upper_bound.as_str() {
                    last.upper_bound = value.clone();
                    last.distinct_count += 1;
                    self.distinct_count += 1;
                }
            } else {
                // No buckets yet — create a single bucket.
                self.buckets.push(HistogramBucket {
                    lower_bound: value.clone(),
                    upper_bound: value.clone(),
                    count: 1,
                    distinct_count: 1,
                    null_count: 0,
                });
                self.distinct_count += 1;
            }
        }
    }

    /// Returns the number of buckets in this histogram.
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }
}

/// Dataset-level statistics aggregating per-predicate histograms.
///
/// Provides the interface expected by the cost-based optimizer for pattern cardinality estimation.
pub struct DatasetStatistics {
    /// Per-predicate equi-depth histograms, keyed by predicate IRI.
    histograms: HashMap<String, PredicateHistogram>,
    /// Total number of triples in the dataset.
    total_triples: u64,
    /// Approximate number of distinct subjects.
    distinct_subjects: u64,
    /// Approximate number of distinct predicates.
    distinct_predicates: u64,
    /// Approximate number of distinct objects.
    distinct_objects: u64,
    /// When statistics were last updated.
    last_updated: Instant,
    /// Target number of buckets per histogram (default: 100).
    num_buckets: usize,
}

impl DatasetStatistics {
    /// Create a new empty statistics store with the default bucket count (100).
    pub fn new() -> Self {
        Self::with_num_buckets(100)
    }

    /// Create a new statistics store with a custom bucket count.
    pub fn with_num_buckets(num_buckets: usize) -> Self {
        Self {
            histograms: HashMap::new(),
            total_triples: 0,
            distinct_subjects: 0,
            distinct_predicates: 0,
            distinct_objects: 0,
            last_updated: Instant::now(),
            num_buckets: num_buckets.max(1),
        }
    }

    /// Record the insertion of a single triple `(subject, predicate, object)`.
    ///
    /// Updates global counts and incrementally updates the per-predicate histogram.
    pub fn record_triple(&mut self, predicate: &str, object: &str) {
        self.total_triples += 1;

        let num_buckets = self.num_buckets;
        let histogram = self
            .histograms
            .entry(predicate.to_owned())
            .or_insert_with(|| PredicateHistogram::new(predicate, num_buckets));

        histogram.update_incremental(&[object.to_owned()]);

        self.distinct_predicates = self.histograms.len() as u64;
        self.last_updated = Instant::now();
    }

    /// Return the histogram for `predicate`, if one exists.
    pub fn get_histogram(&self, predicate: &str) -> Option<&PredicateHistogram> {
        self.histograms.get(predicate)
    }

    /// Estimate the cardinality of the triple pattern `(?s, predicate?, object?)`.
    ///
    /// Pattern selectivity is computed as:
    /// - Unknown subject and unknown predicate → `total_triples`.
    /// - Known predicate, unknown object → `histogram.total_count`.
    /// - Known predicate + known object → equality filter estimate.
    /// - Known object, unknown predicate → brute-force across all histograms.
    /// - Unknown predicate and unknown object → `total_triples`.
    pub fn estimate_pattern_cardinality(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> u64 {
        if self.total_triples == 0 {
            return 0;
        }

        // Subject binding multiplies selectivity by ~1/distinct_subjects.
        let subject_selectivity = if subject.is_some() {
            if self.distinct_subjects > 1 {
                1.0 / self.distinct_subjects as f64
            } else {
                0.001
            }
        } else {
            1.0
        };

        let base_cardinality: f64 = match (predicate, object) {
            (None, None) => self.total_triples as f64,

            (Some(pred), None) => {
                // All triples with this predicate.
                self.histograms
                    .get(pred)
                    .map(|h| h.total_count as f64)
                    .unwrap_or_else(|| {
                        // Unknown predicate — assume uniform distribution.
                        if self.distinct_predicates > 0 {
                            self.total_triples as f64 / self.distinct_predicates as f64
                        } else {
                            self.total_triples as f64 * 0.1
                        }
                    })
            }

            (Some(pred), Some(obj)) => {
                // Equality filter on object for a specific predicate.
                match self.histograms.get(pred) {
                    Some(hist) => {
                        let sel = hist.estimate_selectivity_eq(obj);
                        (hist.total_count as f64 * sel).max(1.0)
                    }
                    None => 1.0, // Unknown predicate → assume at most 1 result.
                }
            }

            (None, Some(obj)) => {
                // Sum estimates across all predicates for the given object value.
                if self.histograms.is_empty() {
                    return 0;
                }
                self.histograms
                    .values()
                    .map(|hist| {
                        let sel = hist.estimate_selectivity_eq(obj);
                        hist.total_count as f64 * sel
                    })
                    .sum::<f64>()
                    .max(1.0)
            }
        };

        ((base_cardinality * subject_selectivity).ceil() as u64).max(1)
    }

    /// Rebuild all histograms from a full scan of `(predicate, object)` pairs.
    ///
    /// This is an expensive O(n log n) operation and should be run during off-peak periods or
    /// triggered by the auto-analyze subsystem.
    pub fn rebuild_histograms(&mut self, values: &[(String, String)]) {
        self.histograms.clear();
        self.total_triples = values.len() as u64;

        // Group by predicate.
        let mut by_predicate: HashMap<&str, Vec<&String>> = HashMap::new();
        for (pred, obj) in values {
            by_predicate.entry(pred.as_str()).or_default().push(obj);
        }

        let num_buckets = self.num_buckets;
        for (pred, objs) in &by_predicate {
            let owned_objs: Vec<String> = objs.iter().map(|s| (*s).clone()).collect();
            let histogram = PredicateHistogram::build_from_values(pred, &owned_objs, num_buckets);
            self.histograms.insert(pred.to_string(), histogram);
        }

        // Update distinct counts.
        self.distinct_predicates = self.histograms.len() as u64;
        // Approximate distinct objects as the sum of distinct objects per predicate
        // (may overcount if the same literal appears under different predicates).
        self.distinct_objects = self.histograms.values().map(|h| h.distinct_count).sum();

        self.last_updated = Instant::now();
    }

    /// Return the total number of triples tracked by these statistics.
    pub fn total_triples(&self) -> u64 {
        self.total_triples
    }

    /// Update the known distinct subject count.
    ///
    /// Called by the store layer after it recomputes subject cardinality.
    pub fn set_distinct_subjects(&mut self, count: u64) {
        self.distinct_subjects = count;
    }

    /// Update the known distinct object count.
    pub fn set_distinct_objects(&mut self, count: u64) {
        self.distinct_objects = count;
    }

    /// Return when the statistics were last updated.
    pub fn last_updated(&self) -> Instant {
        self.last_updated
    }

    /// Return the number of predicates with histograms.
    pub fn predicate_count(&self) -> usize {
        self.histograms.len()
    }
}

impl Default for DatasetStatistics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- HistogramBucket tests ---

    #[test]
    fn test_bucket_contains() {
        let bucket = HistogramBucket {
            lower_bound: "apple".to_string(),
            upper_bound: "mango".to_string(),
            count: 10,
            distinct_count: 5,
            null_count: 0,
        };
        assert!(bucket.contains("apple"));
        assert!(bucket.contains("cherry"));
        assert!(bucket.contains("mango"));
        assert!(!bucket.contains("aardvark")); // < lower_bound
        assert!(!bucket.contains("zebra")); // > upper_bound
    }

    #[test]
    fn test_bucket_overlap_fraction_full() {
        let bucket = HistogramBucket {
            lower_bound: "b".to_string(),
            upper_bound: "m".to_string(),
            count: 100,
            distinct_count: 50,
            null_count: 0,
        };
        // Full containment
        let frac = bucket.overlap_fraction("a", "z");
        assert!(frac > 0.99, "Expected ~1.0 but got {}", frac);
    }

    #[test]
    fn test_bucket_overlap_fraction_none() {
        let bucket = HistogramBucket {
            lower_bound: "m".to_string(),
            upper_bound: "z".to_string(),
            count: 100,
            distinct_count: 50,
            null_count: 0,
        };
        // No overlap
        let frac = bucket.overlap_fraction("a", "b");
        assert_eq!(frac, 0.0);
    }

    // --- PredicateHistogram tests ---

    #[test]
    fn test_build_from_empty_values() {
        let hist = PredicateHistogram::build_from_values("http://example.org/p", &[], 10);
        assert_eq!(hist.total_count, 0);
        assert_eq!(hist.buckets.len(), 0);
    }

    #[test]
    fn test_build_from_values_single_bucket() {
        let values: Vec<String> = (0..5).map(|i| format!("value{}", i)).collect();
        let hist = PredicateHistogram::build_from_values("p", &values, 1);
        assert_eq!(hist.total_count, 5);
        assert_eq!(hist.buckets.len(), 1);
        assert_eq!(hist.buckets[0].count, 5);
    }

    #[test]
    fn test_build_from_values_multiple_buckets() {
        let values: Vec<String> = (0..100).map(|i| format!("value{:03}", i)).collect();
        let hist = PredicateHistogram::build_from_values("p", &values, 10);
        assert_eq!(hist.total_count, 100);
        assert_eq!(hist.distinct_count, 100);
        // 10 buckets, each ~10 entries
        assert!(hist.buckets.len() >= 9 && hist.buckets.len() <= 11);

        // Total count across buckets must equal total_count
        let sum: u64 = hist.buckets.iter().map(|b| b.count).sum();
        assert_eq!(sum, 100);
    }

    #[test]
    fn test_build_handles_nulls() {
        let mut values: Vec<String> = (0..10).map(|i| format!("v{}", i)).collect();
        values.push(String::new()); // null
        values.push(String::new()); // null
        let hist = PredicateHistogram::build_from_values("p", &values, 5);
        assert_eq!(hist.null_count, 2);
        assert_eq!(hist.total_count, 10);
    }

    #[test]
    fn test_estimate_selectivity_eq_within_bucket() {
        let values: Vec<String> = vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
            "delta".to_string(),
            "epsilon".to_string(),
        ];
        let hist = PredicateHistogram::build_from_values("p", &values, 2);
        // Selectivity for a value in the dataset must be > 0
        let sel = hist.estimate_selectivity_eq("beta");
        assert!(sel > 0.0 && sel <= 1.0, "Selectivity {} out of range", sel);
    }

    #[test]
    fn test_estimate_selectivity_eq_missing_value() {
        let values: Vec<String> = (0..10).map(|i| format!("value{}", i)).collect();
        let hist = PredicateHistogram::build_from_values("p", &values, 5);
        let sel = hist.estimate_selectivity_eq("zzz_not_in_dataset");
        // Should be very small
        assert!(
            sel < 0.01,
            "Selectivity for missing value too high: {}",
            sel
        );
    }

    #[test]
    fn test_estimate_selectivity_range() {
        let values: Vec<String> = (0..100).map(|i| format!("{:03}", i)).collect();
        let hist = PredicateHistogram::build_from_values("p", &values, 10);
        // Range covering ~50% of values
        let sel = hist.estimate_selectivity_range("000", "049");
        assert!(
            sel > 0.3 && sel < 0.7,
            "Range selectivity {} unexpected",
            sel
        );
    }

    #[test]
    fn test_estimate_cardinality_no_filter() {
        let values: Vec<String> = (0..50).map(|i| format!("v{}", i)).collect();
        let hist = PredicateHistogram::build_from_values("p", &values, 5);
        let card = hist.estimate_cardinality(1000, None);
        assert_eq!(card, 50);
    }

    #[test]
    fn test_estimate_cardinality_with_filter() {
        let values: Vec<String> = (0..100).map(|i| format!("{:03}", i)).collect();
        let hist = PredicateHistogram::build_from_values("p", &values, 10);
        let card = hist.estimate_cardinality(1000, Some("050"));
        // Should return at least 1
        assert!(card >= 1);
    }

    #[test]
    fn test_update_incremental() {
        let initial: Vec<String> = (0..10).map(|i| format!("v{}", i)).collect();
        let mut hist = PredicateHistogram::build_from_values("p", &initial, 3);
        let before_count = hist.total_count;

        let new_vals: Vec<String> = vec!["newA".to_string(), "newB".to_string()];
        hist.update_incremental(&new_vals);

        assert_eq!(hist.total_count, before_count + 2);
    }

    #[test]
    fn test_update_incremental_null() {
        let initial: Vec<String> = (0..5).map(|i| format!("v{}", i)).collect();
        let mut hist = PredicateHistogram::build_from_values("p", &initial, 2);
        hist.update_incremental(&[String::new()]);
        assert_eq!(hist.null_count, 1);
    }

    // --- DatasetStatistics tests ---

    #[test]
    fn test_dataset_statistics_new() {
        let stats = DatasetStatistics::new();
        assert_eq!(stats.total_triples(), 0);
        assert_eq!(stats.predicate_count(), 0);
    }

    #[test]
    fn test_dataset_record_triple() {
        let mut stats = DatasetStatistics::new();
        stats.record_triple("http://example.org/age", "42");
        stats.record_triple("http://example.org/age", "25");
        stats.record_triple("http://example.org/name", "Alice");

        assert_eq!(stats.total_triples(), 3);
        assert_eq!(stats.predicate_count(), 2);
        assert!(stats.get_histogram("http://example.org/age").is_some());
    }

    #[test]
    fn test_dataset_estimate_pattern_cardinality_no_bindings() {
        let mut stats = DatasetStatistics::new();
        for i in 0..20 {
            stats.record_triple("http://p", &format!("obj{}", i));
        }
        let card = stats.estimate_pattern_cardinality(None, None, None);
        assert_eq!(card, 20);
    }

    #[test]
    fn test_dataset_estimate_pattern_cardinality_known_predicate() {
        let mut stats = DatasetStatistics::new();
        for i in 0..30 {
            stats.record_triple("http://p/name", &format!("name{}", i));
        }
        for i in 0..10 {
            stats.record_triple("http://p/age", &format!("{}", i));
        }
        let card = stats.estimate_pattern_cardinality(None, Some("http://p/name"), None);
        assert_eq!(card, 30);
    }

    #[test]
    fn test_dataset_estimate_pattern_cardinality_eq_filter() {
        let mut stats = DatasetStatistics::with_num_buckets(5);
        for i in 0..100 {
            stats.record_triple("http://p/color", &format!("color{:03}", i));
        }
        let card =
            stats.estimate_pattern_cardinality(None, Some("http://p/color"), Some("color050"));
        assert!(card >= 1, "Equality filter should return at least 1");
    }

    #[test]
    fn test_dataset_rebuild_histograms() {
        let mut stats = DatasetStatistics::new();
        let pairs: Vec<(String, String)> = (0..200)
            .map(|i| ("http://p".to_string(), format!("v{:04}", i)))
            .collect();
        stats.rebuild_histograms(&pairs);

        assert_eq!(stats.total_triples(), 200);
        assert_eq!(stats.predicate_count(), 1);

        let hist = stats
            .get_histogram("http://p")
            .expect("histogram must exist");
        assert_eq!(hist.total_count, 200);
    }

    #[test]
    fn test_dataset_set_distinct_subjects() {
        let mut stats = DatasetStatistics::new();
        stats.record_triple("p", "o");
        stats.set_distinct_subjects(500);
        // Estimate with subject binding should give lower cardinality.
        let with_subject = stats.estimate_pattern_cardinality(Some("s"), Some("p"), None);
        let without_subject = stats.estimate_pattern_cardinality(None, Some("p"), None);
        assert!(with_subject <= without_subject);
    }

    #[test]
    fn test_dataset_unknown_predicate_estimate() {
        let mut stats = DatasetStatistics::new();
        for i in 0..90 {
            stats.record_triple("http://p/known", &format!("v{}", i));
        }
        // Unknown predicate — should fall back to uniform estimate.
        let card = stats.estimate_pattern_cardinality(None, Some("http://p/unknown"), None);
        assert!(card >= 1);
    }
}
