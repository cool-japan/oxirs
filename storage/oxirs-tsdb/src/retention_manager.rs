//! Time-series data retention policies.
//!
//! Supports keeping the most-recent N points, keeping data within a duration
//! window, keeping data after a threshold, downsampling old data, and a
//! no-op (keep everything) policy.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Data types
// ────────────────────────────────────────────────────────────────────────────

/// A single time-series measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct DataPoint {
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
    pub value: f64,
    pub tags: HashMap<String, String>,
}

impl DataPoint {
    pub fn new(timestamp: i64, value: f64) -> Self {
        Self {
            timestamp,
            value,
            tags: HashMap::new(),
        }
    }

    pub fn with_tag(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.tags.insert(key.into(), val.into());
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Retention policy
// ────────────────────────────────────────────────────────────────────────────

/// Specifies how data should be retained or discarded.
#[derive(Debug, Clone)]
pub enum RetentionPolicy {
    /// Keep the most-recent `n` data points (by timestamp, ascending order).
    KeepLast(usize),
    /// Keep all points whose `timestamp >= current_time_ms - duration_ms`.
    KeepDuration(i64),
    /// Keep all points with `timestamp >= threshold`.
    KeepAfter(i64),
    /// Downsample data older than `keep_duration_ms` by averaging into
    /// `resolution_ms`-wide buckets; data within the recent window is kept as-is.
    DownsampleThen {
        resolution_ms: i64,
        keep_duration_ms: i64,
    },
    /// No-op: keep every point.
    None,
}

/// Configuration wrapper for a retention pass.
#[derive(Debug, Clone)]
pub struct RetentionConfig {
    pub policy: RetentionPolicy,
    /// When `true`, compute the result but do **not** return the trimmed data;
    /// the original data is returned unchanged while statistics still reflect
    /// what *would* have been removed.
    pub dry_run: bool,
}

impl RetentionConfig {
    pub fn new(policy: RetentionPolicy) -> Self {
        Self {
            policy,
            dry_run: false,
        }
    }

    pub fn dry_run(mut self) -> Self {
        self.dry_run = true;
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Result
// ────────────────────────────────────────────────────────────────────────────

/// Summary of a retention pass.
#[derive(Debug, Clone, Default)]
pub struct RetentionResult {
    pub kept: usize,
    pub removed: usize,
    pub oldest_kept: Option<i64>,
    pub newest_kept: Option<i64>,
}

impl RetentionResult {
    fn from_kept(kept_points: &[DataPoint], removed: usize) -> Self {
        let oldest_kept = kept_points.iter().map(|p| p.timestamp).min();
        let newest_kept = kept_points.iter().map(|p| p.timestamp).max();
        Self {
            kept: kept_points.len(),
            removed,
            oldest_kept,
            newest_kept,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Manager
// ────────────────────────────────────────────────────────────────────────────

pub struct RetentionManager {
    config: RetentionConfig,
}

impl RetentionManager {
    pub fn new(config: RetentionConfig) -> Self {
        Self { config }
    }

    /// Apply the configured policy to `data`.
    ///
    /// Returns `(resulting_data, stats)`.  If `dry_run` is set, the original
    /// `data` is returned unchanged but `stats` reflects what would have happened.
    pub fn apply(
        &self,
        data: Vec<DataPoint>,
        current_time_ms: i64,
    ) -> (Vec<DataPoint>, RetentionResult) {
        let (kept, result) = match &self.config.policy {
            RetentionPolicy::KeepLast(n) => {
                let (k, r) = Self::apply_keep_last(data.clone(), *n);
                (k, r)
            }
            RetentionPolicy::KeepDuration(duration_ms) => {
                let (k, r) = Self::apply_keep_duration(data.clone(), *duration_ms, current_time_ms);
                (k, r)
            }
            RetentionPolicy::KeepAfter(threshold) => {
                let removed_count = data.iter().filter(|p| p.timestamp < *threshold).count();
                let k: Vec<DataPoint> = data.iter().filter(|p| p.timestamp >= *threshold).cloned().collect();
                let r = RetentionResult::from_kept(&k, removed_count);
                (k, r)
            }
            RetentionPolicy::DownsampleThen {
                resolution_ms,
                keep_duration_ms,
            } => {
                let (k, r) =
                    Self::apply_downsample(data.clone(), *resolution_ms, *keep_duration_ms, current_time_ms);
                (k, r)
            }
            RetentionPolicy::None => {
                let r = RetentionResult::from_kept(&data, 0);
                (data.clone(), r)
            }
        };

        if self.config.dry_run {
            // Return original data, but report stats as-if applied
            (data, result)
        } else {
            (kept, result)
        }
    }

    // ── static policy helpers ────────────────────────────────────────────────

    /// Keep the most-recent `n` data points (sorted ascending by timestamp).
    pub fn apply_keep_last(mut data: Vec<DataPoint>, n: usize) -> (Vec<DataPoint>, RetentionResult) {
        data.sort_by_key(|p| p.timestamp);
        let total = data.len();
        let kept = if n >= total {
            data
        } else {
            data.into_iter().skip(total - n).collect()
        };
        let removed = total - kept.len();
        let result = RetentionResult::from_kept(&kept, removed);
        (kept, result)
    }

    /// Keep all points with `timestamp >= current_time_ms - duration_ms`.
    pub fn apply_keep_duration(
        data: Vec<DataPoint>,
        duration_ms: i64,
        current_time_ms: i64,
    ) -> (Vec<DataPoint>, RetentionResult) {
        let cutoff = current_time_ms - duration_ms;
        let removed = data.iter().filter(|p| p.timestamp < cutoff).count();
        let kept: Vec<DataPoint> = data.into_iter().filter(|p| p.timestamp >= cutoff).collect();
        let result = RetentionResult::from_kept(&kept, removed);
        (kept, result)
    }

    /// Downsample data older than `keep_duration_ms` into `resolution_ms` buckets.
    ///
    /// For each bucket the representative point has:
    /// - `timestamp` = bucket start
    /// - `value`     = average of all values in the bucket
    /// - `tags`      = merged union (later values overwrite earlier on conflict)
    pub fn apply_downsample(
        data: Vec<DataPoint>,
        resolution_ms: i64,
        keep_duration_ms: i64,
        current_time_ms: i64,
    ) -> (Vec<DataPoint>, RetentionResult) {
        if data.is_empty() {
            return (vec![], RetentionResult::default());
        }

        let cutoff = current_time_ms - keep_duration_ms;

        let (recent, old): (Vec<DataPoint>, Vec<DataPoint>) =
            data.into_iter().partition(|p| p.timestamp >= cutoff);

        let original_count = recent.len() + old.len();

        // Downsample old points into resolution-ms buckets
        let mut buckets: std::collections::BTreeMap<i64, Vec<DataPoint>> =
            std::collections::BTreeMap::new();

        for point in old {
            let bucket_start = if resolution_ms > 0 {
                (point.timestamp / resolution_ms) * resolution_ms
            } else {
                point.timestamp
            };
            buckets.entry(bucket_start).or_default().push(point);
        }

        let mut downsampled: Vec<DataPoint> = buckets
            .into_iter()
            .map(|(bucket_start, points)| {
                let avg = points.iter().map(|p| p.value).sum::<f64>() / points.len() as f64;
                let mut merged_tags: HashMap<String, String> = HashMap::new();
                for p in &points {
                    merged_tags.extend(p.tags.clone());
                }
                DataPoint {
                    timestamp: bucket_start,
                    value: avg,
                    tags: merged_tags,
                }
            })
            .collect();

        downsampled.extend(recent);
        downsampled.sort_by_key(|p| p.timestamp);

        let kept_count = downsampled.len();
        let removed = original_count - kept_count;
        let result = RetentionResult::from_kept(&downsampled, removed);
        (downsampled, result)
    }

    /// Rough byte estimate for a slice of data points.
    ///
    /// Uses: 8 (timestamp) + 8 (value) + 32 per tag key-value pair.
    pub fn estimate_storage(data: &[DataPoint]) -> usize {
        data.iter()
            .map(|p| {
                let tag_bytes: usize = p.tags.iter().map(|(k, v)| k.len() + v.len() + 2).sum();
                8 + 8 + tag_bytes
            })
            .sum()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn points(timestamps: Vec<i64>) -> Vec<DataPoint> {
        timestamps
            .into_iter()
            .map(|t| DataPoint::new(t, t as f64))
            .collect()
    }

    // ── KeepLast ──────────────────────────────────────────────────────────────

    #[test]
    fn test_keep_last_trims_oldest() {
        let (kept, result) = RetentionManager::apply_keep_last(points(vec![1, 2, 3, 4, 5]), 3);
        let ts: Vec<i64> = kept.iter().map(|p| p.timestamp).collect();
        assert_eq!(ts, vec![3, 4, 5]);
        assert_eq!(result.kept, 3);
        assert_eq!(result.removed, 2);
    }

    #[test]
    fn test_keep_last_n_greater_than_len_keeps_all() {
        let (kept, result) = RetentionManager::apply_keep_last(points(vec![1, 2, 3]), 10);
        assert_eq!(kept.len(), 3);
        assert_eq!(result.removed, 0);
    }

    #[test]
    fn test_keep_last_zero_keeps_nothing() {
        let (kept, result) = RetentionManager::apply_keep_last(points(vec![1, 2, 3]), 0);
        assert!(kept.is_empty());
        assert_eq!(result.removed, 3);
    }

    #[test]
    fn test_keep_last_oldest_newest_correct() {
        let (_, result) = RetentionManager::apply_keep_last(points(vec![10, 20, 30, 40, 50]), 3);
        assert_eq!(result.oldest_kept, Some(30));
        assert_eq!(result.newest_kept, Some(50));
    }

    #[test]
    fn test_keep_last_unsorted_input() {
        let (kept, _) = RetentionManager::apply_keep_last(points(vec![50, 10, 30, 20, 40]), 3);
        let ts: Vec<i64> = kept.iter().map(|p| p.timestamp).collect();
        assert_eq!(ts, vec![30, 40, 50]);
    }

    // ── KeepDuration ──────────────────────────────────────────────────────────

    #[test]
    fn test_keep_duration_removes_old() {
        // current=1000, duration=500 → keep timestamp >= 500
        let (kept, result) = RetentionManager::apply_keep_duration(
            points(vec![100, 400, 600, 800, 1000]),
            500,
            1000,
        );
        let ts: Vec<i64> = kept.iter().map(|p| p.timestamp).collect();
        assert_eq!(ts, vec![600, 800, 1000]);
        assert_eq!(result.kept, 3);
        assert_eq!(result.removed, 2);
    }

    #[test]
    fn test_keep_duration_keeps_all_recent() {
        let (kept, result) =
            RetentionManager::apply_keep_duration(points(vec![900, 950, 1000]), 500, 1000);
        assert_eq!(kept.len(), 3);
        assert_eq!(result.removed, 0);
    }

    #[test]
    fn test_keep_duration_removes_all_old() {
        let (kept, result) =
            RetentionManager::apply_keep_duration(points(vec![1, 2, 3]), 500, 1_000_000);
        assert!(kept.is_empty());
        assert_eq!(result.removed, 3);
    }

    // ── KeepAfter ─────────────────────────────────────────────────────────────

    #[test]
    fn test_keep_after_threshold() {
        let manager = RetentionManager::new(RetentionConfig::new(RetentionPolicy::KeepAfter(500)));
        let (kept, result) = manager.apply(points(vec![100, 300, 500, 700, 900]), 1000);
        assert_eq!(result.kept, 3);
        assert_eq!(result.removed, 2);
        assert!(kept.iter().all(|p| p.timestamp >= 500));
    }

    #[test]
    fn test_keep_after_all_pass() {
        let manager = RetentionManager::new(RetentionConfig::new(RetentionPolicy::KeepAfter(0)));
        let (kept, result) = manager.apply(points(vec![1, 2, 3]), 1000);
        assert_eq!(kept.len(), 3);
        assert_eq!(result.removed, 0);
    }

    #[test]
    fn test_keep_after_oldest_newest_correct() {
        let manager = RetentionManager::new(RetentionConfig::new(RetentionPolicy::KeepAfter(300)));
        let (_, result) = manager.apply(points(vec![100, 300, 500, 700]), 1000);
        assert_eq!(result.oldest_kept, Some(300));
        assert_eq!(result.newest_kept, Some(700));
    }

    // ── DownsampleThen ────────────────────────────────────────────────────────

    #[test]
    fn test_downsample_reduces_old_count() {
        // current=10_000, keep_duration=2_000 → cutoff=8_000
        // old: 1000, 1100, 1200 → bucket 1000 (resolution=1000)
        // recent: 9000, 9500
        let data = points(vec![1000, 1100, 1200, 9000, 9500]);
        let (kept, result) = RetentionManager::apply_downsample(data, 1_000, 2_000, 10_000);
        // 3 old → 1 downsampled bucket; 2 recent → total 3
        assert_eq!(kept.len(), 3);
        assert_eq!(result.removed, 2); // 5 original - 3 kept
    }

    #[test]
    fn test_downsample_average_value() {
        let data = vec![
            DataPoint::new(1000, 10.0),
            DataPoint::new(1100, 20.0),
            DataPoint::new(1200, 30.0),
        ];
        // All old (current=5000, keep_duration=1000 → cutoff=4000)
        let (kept, _) = RetentionManager::apply_downsample(data, 1_000, 1_000, 5_000);
        assert_eq!(kept.len(), 1);
        // avg = (10+20+30)/3 = 20
        assert!((kept[0].value - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_downsample_recent_untouched() {
        let data = vec![
            DataPoint::new(9_000, 5.0),
            DataPoint::new(9_500, 10.0),
        ];
        let (kept, result) = RetentionManager::apply_downsample(data, 1_000, 2_000, 10_000);
        assert_eq!(kept.len(), 2);
        assert_eq!(result.removed, 0);
    }

    #[test]
    fn test_downsample_empty_input() {
        let (kept, result) = RetentionManager::apply_downsample(vec![], 1_000, 5_000, 10_000);
        assert!(kept.is_empty());
        assert_eq!(result.kept, 0);
        assert_eq!(result.removed, 0);
    }

    // ── None (keep all) ───────────────────────────────────────────────────────

    #[test]
    fn test_none_policy_keeps_all() {
        let manager = RetentionManager::new(RetentionConfig::new(RetentionPolicy::None));
        let data = points(vec![1, 2, 3, 4, 5]);
        let (kept, result) = manager.apply(data, 1000);
        assert_eq!(kept.len(), 5);
        assert_eq!(result.removed, 0);
    }

    // ── dry_run ───────────────────────────────────────────────────────────────

    #[test]
    fn test_dry_run_returns_original_data() {
        let manager =
            RetentionManager::new(RetentionConfig::new(RetentionPolicy::KeepLast(2)).dry_run());
        let data = points(vec![1, 2, 3, 4, 5]);
        let (kept, result) = manager.apply(data, 1000);
        // Original data is returned unchanged
        assert_eq!(kept.len(), 5);
        // But stats reflect what would have happened
        assert_eq!(result.kept, 2);
        assert_eq!(result.removed, 3);
    }

    #[test]
    fn test_dry_run_duration_returns_original() {
        let manager = RetentionManager::new(
            RetentionConfig::new(RetentionPolicy::KeepDuration(100)).dry_run(),
        );
        let data = points(vec![1, 2, 900, 950, 1000]);
        let (kept, result) = manager.apply(data, 1000);
        assert_eq!(kept.len(), 5); // returned unchanged
        assert_eq!(result.removed, 2); // 1 and 2 would have been removed
    }

    // ── RetentionResult counters ──────────────────────────────────────────────

    #[test]
    fn test_result_counters_correct() {
        let (_, result) = RetentionManager::apply_keep_last(points(vec![1, 2, 3, 4, 5]), 3);
        assert_eq!(result.kept + result.removed, 5);
    }

    #[test]
    fn test_oldest_newest_none_for_empty_result() {
        let (_, result) = RetentionManager::apply_keep_last(points(vec![1, 2, 3]), 0);
        assert_eq!(result.oldest_kept, None);
        assert_eq!(result.newest_kept, None);
    }

    // ── empty input ───────────────────────────────────────────────────────────

    #[test]
    fn test_empty_input_keep_last() {
        let (kept, result) = RetentionManager::apply_keep_last(vec![], 10);
        assert!(kept.is_empty());
        assert_eq!(result.removed, 0);
        assert_eq!(result.oldest_kept, None);
        assert_eq!(result.newest_kept, None);
    }

    #[test]
    fn test_empty_input_keep_duration() {
        let (kept, result) = RetentionManager::apply_keep_duration(vec![], 1000, 5000);
        assert!(kept.is_empty());
        assert_eq!(result.removed, 0);
    }

    // ── estimate_storage ─────────────────────────────────────────────────────

    #[test]
    fn test_estimate_storage_no_tags() {
        // 2 points, no tags: 2 * (8 + 8) = 32 bytes
        let data = vec![DataPoint::new(1, 1.0), DataPoint::new(2, 2.0)];
        assert_eq!(RetentionManager::estimate_storage(&data), 32);
    }

    #[test]
    fn test_estimate_storage_with_tags() {
        let p = DataPoint::new(1, 1.0).with_tag("host", "srv1");
        let bytes = RetentionManager::estimate_storage(&[p]);
        // 8 + 8 + (4 + 4 + 2) = 26
        assert_eq!(bytes, 26);
    }

    #[test]
    fn test_estimate_storage_empty() {
        assert_eq!(RetentionManager::estimate_storage(&[]), 0);
    }

    // ── via manager.apply (integration) ──────────────────────────────────────

    #[test]
    fn test_manager_apply_keep_last_integration() {
        let manager =
            RetentionManager::new(RetentionConfig::new(RetentionPolicy::KeepLast(2)));
        let (kept, _) = manager.apply(points(vec![10, 20, 30, 40, 50]), 999);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_manager_apply_duration_integration() {
        let manager =
            RetentionManager::new(RetentionConfig::new(RetentionPolicy::KeepDuration(500)));
        let (kept, _) = manager.apply(points(vec![100, 400, 600, 800, 1000]), 1000);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn test_data_point_with_tags() {
        let p = DataPoint::new(100, 42.0)
            .with_tag("region", "us-east")
            .with_tag("env", "prod");
        assert_eq!(p.tags.get("region"), Some(&"us-east".to_string()));
        assert_eq!(p.tags.get("env"), Some(&"prod".to_string()));
    }
}
