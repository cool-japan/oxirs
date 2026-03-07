//! # Downsampling / Rollup Engine
//!
//! Provides multi-tier downsampling (rollup) for time-series data:
//!   raw (1s) -> minute (1m) -> hour (1h) -> day (1d)
//!
//! Each tier stores pre-aggregated data that can be queried much faster
//! than computing aggregates on the fly from raw data.
//!
//! ## Architecture
//!
//! ```text
//! Raw data (1s resolution)
//!   |
//!   +--> Minute rollup (avg, min, max, sum, count per minute)
//!          |
//!          +--> Hour rollup (avg, min, max, sum, count per hour)
//!                 |
//!                 +--> Day rollup (avg, min, max, sum, count per day)
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use oxirs_tsdb::analytics::rollup_engine::{RollupEngine, RollupConfig, RollupTier};
//!
//! let config = RollupConfig::default();
//! let engine = RollupEngine::new(config);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Rollup tier
// ---------------------------------------------------------------------------

/// The time resolution tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RollupTier {
    /// Raw data (no aggregation).
    Raw,
    /// One-minute aggregation.
    Minute,
    /// One-hour aggregation.
    Hour,
    /// One-day aggregation.
    Day,
}

impl RollupTier {
    /// Duration of one bucket in seconds.
    pub fn bucket_seconds(&self) -> u64 {
        match self {
            RollupTier::Raw => 1,
            RollupTier::Minute => 60,
            RollupTier::Hour => 3600,
            RollupTier::Day => 86400,
        }
    }

    /// The next coarser tier (if any).
    pub fn next_tier(&self) -> Option<RollupTier> {
        match self {
            RollupTier::Raw => Some(RollupTier::Minute),
            RollupTier::Minute => Some(RollupTier::Hour),
            RollupTier::Hour => Some(RollupTier::Day),
            RollupTier::Day => None,
        }
    }

    /// The previous finer tier (if any).
    pub fn prev_tier(&self) -> Option<RollupTier> {
        match self {
            RollupTier::Raw => None,
            RollupTier::Minute => Some(RollupTier::Raw),
            RollupTier::Hour => Some(RollupTier::Minute),
            RollupTier::Day => Some(RollupTier::Hour),
        }
    }

    /// All tiers from finest to coarsest.
    pub fn all_tiers() -> &'static [RollupTier] {
        &[
            RollupTier::Raw,
            RollupTier::Minute,
            RollupTier::Hour,
            RollupTier::Day,
        ]
    }
}

impl fmt::Display for RollupTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RollupTier::Raw => write!(f, "raw"),
            RollupTier::Minute => write!(f, "minute"),
            RollupTier::Hour => write!(f, "hour"),
            RollupTier::Day => write!(f, "day"),
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregated data point
// ---------------------------------------------------------------------------

/// A pre-aggregated data point for a rollup bucket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollupDataPoint {
    /// Start timestamp of the bucket (epoch seconds).
    pub timestamp: i64,
    /// Average value in the bucket.
    pub avg: f64,
    /// Minimum value in the bucket.
    pub min: f64,
    /// Maximum value in the bucket.
    pub max: f64,
    /// Sum of values in the bucket.
    pub sum: f64,
    /// Number of raw data points in the bucket.
    pub count: u64,
    /// First value in the bucket.
    pub first: f64,
    /// Last value in the bucket.
    pub last: f64,
}

impl RollupDataPoint {
    /// Create from a set of raw values.
    pub fn from_values(timestamp: i64, values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let count = values.len() as u64;
        let sum: f64 = values.iter().sum();
        let avg = sum / count as f64;
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let first = values[0];
        let last = values[values.len() - 1];

        Some(Self {
            timestamp,
            avg,
            min,
            max,
            sum,
            count,
            first,
            last,
        })
    }

    /// Merge two rollup points (for re-aggregation to a coarser tier).
    pub fn merge(a: &RollupDataPoint, b: &RollupDataPoint) -> Self {
        let total_count = a.count + b.count;
        let total_sum = a.sum + b.sum;
        let avg = if total_count > 0 {
            total_sum / total_count as f64
        } else {
            0.0
        };

        Self {
            timestamp: a.timestamp.min(b.timestamp),
            avg,
            min: a.min.min(b.min),
            max: a.max.max(b.max),
            sum: total_sum,
            count: total_count,
            first: if a.timestamp <= b.timestamp {
                a.first
            } else {
                b.first
            },
            last: if a.timestamp >= b.timestamp {
                a.last
            } else {
                b.last
            },
        }
    }

    /// Merge an iterator of rollup points into one.
    pub fn merge_many(points: &[RollupDataPoint]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let mut result = points[0].clone();
        for p in &points[1..] {
            result = Self::merge(&result, p);
        }
        Some(result)
    }
}

// ---------------------------------------------------------------------------
// Raw data point (simple timestamp + value)
// ---------------------------------------------------------------------------

/// A raw time-series data point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawDataPoint {
    /// Timestamp (epoch seconds).
    pub timestamp: i64,
    /// Value.
    pub value: f64,
}

impl RawDataPoint {
    pub fn new(timestamp: i64, value: f64) -> Self {
        Self { timestamp, value }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Per-tier configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    /// Whether this tier is enabled.
    pub enabled: bool,
    /// Retention duration in seconds (0 = keep forever).
    pub retention_secs: u64,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retention_secs: 0,
        }
    }
}

/// Overall rollup configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollupConfig {
    /// Per-tier configurations.
    pub tiers: HashMap<RollupTier, TierConfig>,
    /// Whether to keep raw data after rollup.
    pub keep_raw_after_rollup: bool,
    /// Maximum number of raw points per rollup batch.
    pub batch_size: usize,
}

impl Default for RollupConfig {
    fn default() -> Self {
        let mut tiers = HashMap::new();
        tiers.insert(
            RollupTier::Raw,
            TierConfig {
                enabled: true,
                retention_secs: 7 * 86400, // 7 days
            },
        );
        tiers.insert(
            RollupTier::Minute,
            TierConfig {
                enabled: true,
                retention_secs: 30 * 86400, // 30 days
            },
        );
        tiers.insert(
            RollupTier::Hour,
            TierConfig {
                enabled: true,
                retention_secs: 365 * 86400, // 1 year
            },
        );
        tiers.insert(
            RollupTier::Day,
            TierConfig {
                enabled: true,
                retention_secs: 0, // Keep forever
            },
        );

        Self {
            tiers,
            keep_raw_after_rollup: true,
            batch_size: 10_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Rollup statistics
// ---------------------------------------------------------------------------

/// Statistics from rollup operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RollupStats {
    /// Number of rollup operations performed.
    pub total_rollups: u64,
    /// Total raw points processed.
    pub total_raw_points_processed: u64,
    /// Total rollup points produced.
    pub total_rollup_points_produced: u64,
    /// Per-tier rollup counts.
    pub per_tier_rollups: HashMap<String, u64>,
    /// Per-tier point counts.
    pub per_tier_points: HashMap<String, u64>,
}

// ---------------------------------------------------------------------------
// Rollup engine
// ---------------------------------------------------------------------------

/// The main rollup engine that performs downsampling.
pub struct RollupEngine {
    config: RollupConfig,
    /// Stored rollup data per series per tier.
    data: HashMap<String, HashMap<RollupTier, Vec<RollupDataPoint>>>,
    /// Statistics.
    stats: RollupStats,
}

impl RollupEngine {
    /// Create a new rollup engine.
    pub fn new(config: RollupConfig) -> Self {
        Self {
            config,
            data: HashMap::new(),
            stats: RollupStats::default(),
        }
    }

    /// Ingest raw data points for a series and produce rollups.
    pub fn ingest(
        &mut self,
        series_id: &str,
        points: &[RawDataPoint],
    ) -> Vec<(RollupTier, Vec<RollupDataPoint>)> {
        let mut results = Vec::new();

        if points.is_empty() {
            return results;
        }

        // Produce minute rollups from raw data
        if self.is_tier_enabled(RollupTier::Minute) {
            let minute_rollups = self.aggregate(points, RollupTier::Minute);
            if !minute_rollups.is_empty() {
                self.store(series_id, RollupTier::Minute, &minute_rollups);
                results.push((RollupTier::Minute, minute_rollups.clone()));

                // Produce hour rollups from minute rollups
                if self.is_tier_enabled(RollupTier::Hour) {
                    let hour_rollups = self.aggregate_rollups(&minute_rollups, RollupTier::Hour);
                    if !hour_rollups.is_empty() {
                        self.store(series_id, RollupTier::Hour, &hour_rollups);
                        results.push((RollupTier::Hour, hour_rollups.clone()));

                        // Produce day rollups from hour rollups
                        if self.is_tier_enabled(RollupTier::Day) {
                            let day_rollups =
                                self.aggregate_rollups(&hour_rollups, RollupTier::Day);
                            if !day_rollups.is_empty() {
                                self.store(series_id, RollupTier::Day, &day_rollups);
                                results.push((RollupTier::Day, day_rollups));
                            }
                        }
                    }
                }
            }
        }

        // Update statistics
        self.stats.total_rollups += 1;
        self.stats.total_raw_points_processed += points.len() as u64;

        for (tier, rollup_points) in &results {
            self.stats.total_rollup_points_produced += rollup_points.len() as u64;
            *self
                .stats
                .per_tier_rollups
                .entry(tier.to_string())
                .or_insert(0) += 1;
            *self
                .stats
                .per_tier_points
                .entry(tier.to_string())
                .or_insert(0) += rollup_points.len() as u64;
        }

        results
    }

    /// Aggregate raw data points into rollup buckets for the given tier.
    pub fn aggregate(&self, points: &[RawDataPoint], tier: RollupTier) -> Vec<RollupDataPoint> {
        if points.is_empty() {
            return Vec::new();
        }

        let bucket_secs = tier.bucket_seconds() as i64;

        // Group points by bucket
        let mut buckets: HashMap<i64, Vec<f64>> = HashMap::new();
        for p in points {
            let bucket_start = (p.timestamp / bucket_secs) * bucket_secs;
            buckets.entry(bucket_start).or_default().push(p.value);
        }

        // Convert to rollup points
        let mut result: Vec<RollupDataPoint> = buckets
            .into_iter()
            .filter_map(|(ts, values)| RollupDataPoint::from_values(ts, &values))
            .collect();

        result.sort_by_key(|r| r.timestamp);
        result
    }

    /// Re-aggregate existing rollup points into a coarser tier.
    pub fn aggregate_rollups(
        &self,
        points: &[RollupDataPoint],
        target_tier: RollupTier,
    ) -> Vec<RollupDataPoint> {
        if points.is_empty() {
            return Vec::new();
        }

        let bucket_secs = target_tier.bucket_seconds() as i64;

        // Group by bucket
        let mut buckets: HashMap<i64, Vec<RollupDataPoint>> = HashMap::new();
        for p in points {
            let bucket_start = (p.timestamp / bucket_secs) * bucket_secs;
            buckets.entry(bucket_start).or_default().push(p.clone());
        }

        // Merge within each bucket
        let mut result: Vec<RollupDataPoint> = buckets
            .into_iter()
            .filter_map(|(ts, bucket_points)| {
                RollupDataPoint::merge_many(&bucket_points).map(|mut merged| {
                    merged.timestamp = ts;
                    merged
                })
            })
            .collect();

        result.sort_by_key(|r| r.timestamp);
        result
    }

    /// Store rollup data for a series.
    fn store(&mut self, series_id: &str, tier: RollupTier, points: &[RollupDataPoint]) {
        let series_data = self.data.entry(series_id.to_string()).or_default();
        let tier_data = series_data.entry(tier).or_default();
        tier_data.extend(points.iter().cloned());
        tier_data.sort_by_key(|r| r.timestamp);
        tier_data.dedup_by_key(|r| r.timestamp);
    }

    /// Query rollup data for a series and tier.
    pub fn query(
        &self,
        series_id: &str,
        tier: RollupTier,
        start_ts: i64,
        end_ts: i64,
    ) -> Vec<RollupDataPoint> {
        self.data
            .get(series_id)
            .and_then(|sd| sd.get(&tier))
            .map(|points| {
                points
                    .iter()
                    .filter(|p| p.timestamp >= start_ts && p.timestamp <= end_ts)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Automatically select the best tier for a query time range.
    pub fn select_tier(&self, range_secs: u64) -> RollupTier {
        if range_secs <= 3600 {
            RollupTier::Raw
        } else if range_secs <= 86400 {
            RollupTier::Minute
        } else if range_secs <= 7 * 86400 {
            RollupTier::Hour
        } else {
            RollupTier::Day
        }
    }

    /// Check if a tier is enabled.
    pub fn is_tier_enabled(&self, tier: RollupTier) -> bool {
        self.config
            .tiers
            .get(&tier)
            .map(|c| c.enabled)
            .unwrap_or(false)
    }

    /// Get all series IDs.
    pub fn series_ids(&self) -> Vec<&str> {
        self.data.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of rollup points for a series and tier.
    pub fn point_count(&self, series_id: &str, tier: RollupTier) -> usize {
        self.data
            .get(series_id)
            .and_then(|sd| sd.get(&tier))
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get statistics.
    pub fn stats(&self) -> &RollupStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = RollupStats::default();
    }

    /// Get configuration.
    pub fn config(&self) -> &RollupConfig {
        &self.config
    }

    /// Apply retention: remove rollup points older than the retention period.
    pub fn apply_retention(&mut self, now_secs: i64) -> u64 {
        let mut total_removed = 0u64;

        for series_data in self.data.values_mut() {
            for (tier, points) in series_data.iter_mut() {
                if let Some(tier_config) = self.config.tiers.get(tier) {
                    if tier_config.retention_secs > 0 {
                        let cutoff = now_secs - tier_config.retention_secs as i64;
                        let before = points.len();
                        points.retain(|p| p.timestamp >= cutoff);
                        total_removed += (before - points.len()) as u64;
                    }
                }
            }
        }

        total_removed
    }

    /// Clear all data for a series.
    pub fn clear_series(&mut self, series_id: &str) -> bool {
        self.data.remove(series_id).is_some()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_raw_points(start: i64, count: usize, interval_secs: i64) -> Vec<RawDataPoint> {
        (0..count)
            .map(|i| RawDataPoint::new(start + i as i64 * interval_secs, 10.0 + i as f64))
            .collect()
    }

    // --- RollupTier ---

    #[test]
    fn test_tier_bucket_seconds() {
        assert_eq!(RollupTier::Raw.bucket_seconds(), 1);
        assert_eq!(RollupTier::Minute.bucket_seconds(), 60);
        assert_eq!(RollupTier::Hour.bucket_seconds(), 3600);
        assert_eq!(RollupTier::Day.bucket_seconds(), 86400);
    }

    #[test]
    fn test_tier_next() {
        assert_eq!(RollupTier::Raw.next_tier(), Some(RollupTier::Minute));
        assert_eq!(RollupTier::Minute.next_tier(), Some(RollupTier::Hour));
        assert_eq!(RollupTier::Hour.next_tier(), Some(RollupTier::Day));
        assert_eq!(RollupTier::Day.next_tier(), None);
    }

    #[test]
    fn test_tier_prev() {
        assert_eq!(RollupTier::Raw.prev_tier(), None);
        assert_eq!(RollupTier::Minute.prev_tier(), Some(RollupTier::Raw));
        assert_eq!(RollupTier::Hour.prev_tier(), Some(RollupTier::Minute));
        assert_eq!(RollupTier::Day.prev_tier(), Some(RollupTier::Hour));
    }

    #[test]
    fn test_tier_all() {
        assert_eq!(RollupTier::all_tiers().len(), 4);
    }

    #[test]
    fn test_tier_display() {
        assert_eq!(RollupTier::Raw.to_string(), "raw");
        assert_eq!(RollupTier::Minute.to_string(), "minute");
        assert_eq!(RollupTier::Hour.to_string(), "hour");
        assert_eq!(RollupTier::Day.to_string(), "day");
    }

    #[test]
    fn test_tier_ordering() {
        assert!(RollupTier::Raw < RollupTier::Minute);
        assert!(RollupTier::Minute < RollupTier::Hour);
        assert!(RollupTier::Hour < RollupTier::Day);
    }

    // --- RollupDataPoint ---

    #[test]
    fn test_rollup_from_values() {
        let point = RollupDataPoint::from_values(1000, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(point.is_some());
        let p = point.expect("should exist");
        assert_eq!(p.timestamp, 1000);
        assert!((p.avg - 3.0).abs() < 0.001);
        assert!((p.min - 1.0).abs() < 0.001);
        assert!((p.max - 5.0).abs() < 0.001);
        assert!((p.sum - 15.0).abs() < 0.001);
        assert_eq!(p.count, 5);
        assert!((p.first - 1.0).abs() < 0.001);
        assert!((p.last - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_rollup_from_values_empty() {
        assert!(RollupDataPoint::from_values(0, &[]).is_none());
    }

    #[test]
    fn test_rollup_from_values_single() {
        let p = RollupDataPoint::from_values(100, &[42.0]).expect("should exist");
        assert_eq!(p.count, 1);
        assert!((p.avg - 42.0).abs() < 0.001);
        assert!((p.min - 42.0).abs() < 0.001);
        assert!((p.max - 42.0).abs() < 0.001);
    }

    #[test]
    fn test_rollup_merge() {
        let a = RollupDataPoint::from_values(100, &[1.0, 2.0]).expect("exists");
        let b = RollupDataPoint::from_values(200, &[3.0, 4.0]).expect("exists");
        let merged = RollupDataPoint::merge(&a, &b);

        assert_eq!(merged.count, 4);
        assert!((merged.sum - 10.0).abs() < 0.001);
        assert!((merged.avg - 2.5).abs() < 0.001);
        assert!((merged.min - 1.0).abs() < 0.001);
        assert!((merged.max - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_rollup_merge_many() {
        let points = vec![
            RollupDataPoint::from_values(100, &[1.0]).expect("exists"),
            RollupDataPoint::from_values(200, &[2.0]).expect("exists"),
            RollupDataPoint::from_values(300, &[3.0]).expect("exists"),
        ];
        let merged = RollupDataPoint::merge_many(&points).expect("exists");
        assert_eq!(merged.count, 3);
        assert!((merged.sum - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_rollup_merge_many_empty() {
        assert!(RollupDataPoint::merge_many(&[]).is_none());
    }

    // --- RawDataPoint ---

    #[test]
    fn test_raw_data_point() {
        let p = RawDataPoint::new(1000, 42.5);
        assert_eq!(p.timestamp, 1000);
        assert!((p.value - 42.5).abs() < 0.001);
    }

    // --- RollupConfig ---

    #[test]
    fn test_config_default() {
        let config = RollupConfig::default();
        assert!(config.keep_raw_after_rollup);
        assert_eq!(config.batch_size, 10_000);
        assert_eq!(config.tiers.len(), 4);
    }

    #[test]
    fn test_config_tier_retention() {
        let config = RollupConfig::default();
        let raw_config = config.tiers.get(&RollupTier::Raw).expect("should exist");
        assert_eq!(raw_config.retention_secs, 7 * 86400);

        let day_config = config.tiers.get(&RollupTier::Day).expect("should exist");
        assert_eq!(day_config.retention_secs, 0); // Forever
    }

    // --- RollupEngine ---

    #[test]
    fn test_engine_creation() {
        let engine = RollupEngine::new(RollupConfig::default());
        assert!(engine.series_ids().is_empty());
    }

    #[test]
    fn test_engine_aggregate_empty() {
        let engine = RollupEngine::new(RollupConfig::default());
        let result = engine.aggregate(&[], RollupTier::Minute);
        assert!(result.is_empty());
    }

    #[test]
    fn test_engine_aggregate_minute() {
        let engine = RollupEngine::new(RollupConfig::default());
        // 120 points at 1-second intervals = 2 minute buckets
        let points = make_raw_points(0, 120, 1);
        let result = engine.aggregate(&points, RollupTier::Minute);
        assert_eq!(result.len(), 2);

        // First bucket: 0-59
        assert_eq!(result[0].count, 60);
        // Second bucket: 60-119
        assert_eq!(result[1].count, 60);
    }

    #[test]
    fn test_engine_aggregate_hour() {
        let engine = RollupEngine::new(RollupConfig::default());
        // 7200 points = 2 hours
        let points = make_raw_points(0, 7200, 1);
        let result = engine.aggregate(&points, RollupTier::Hour);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_engine_ingest_basic() {
        let mut engine = RollupEngine::new(RollupConfig::default());
        let points = make_raw_points(0, 120, 1);

        let results = engine.ingest("sensor_1", &points);
        assert!(!results.is_empty());

        // Should have minute rollups
        let minute_count = engine.point_count("sensor_1", RollupTier::Minute);
        assert!(minute_count > 0);
    }

    #[test]
    fn test_engine_ingest_empty() {
        let mut engine = RollupEngine::new(RollupConfig::default());
        let results = engine.ingest("sensor_1", &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_engine_query() {
        let mut engine = RollupEngine::new(RollupConfig::default());
        let points = make_raw_points(0, 120, 1);
        engine.ingest("sensor_1", &points);

        let result = engine.query("sensor_1", RollupTier::Minute, 0, 120);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_engine_query_empty_series() {
        let engine = RollupEngine::new(RollupConfig::default());
        let result = engine.query("nonexistent", RollupTier::Minute, 0, 100);
        assert!(result.is_empty());
    }

    #[test]
    fn test_engine_query_time_filter() {
        let mut engine = RollupEngine::new(RollupConfig::default());
        let points = make_raw_points(0, 300, 1); // 5 minutes of data
        engine.ingest("sensor_1", &points);

        // Query only the first 2 minutes
        let result = engine.query("sensor_1", RollupTier::Minute, 0, 120);
        let full = engine.query("sensor_1", RollupTier::Minute, 0, 300);
        assert!(result.len() <= full.len());
    }

    #[test]
    fn test_engine_select_tier() {
        let engine = RollupEngine::new(RollupConfig::default());
        assert_eq!(engine.select_tier(600), RollupTier::Raw); // 10 minutes
        assert_eq!(engine.select_tier(3600), RollupTier::Raw); // 1 hour
        assert_eq!(engine.select_tier(86400), RollupTier::Minute); // 1 day
        assert_eq!(engine.select_tier(3 * 86400), RollupTier::Hour); // 3 days
        assert_eq!(engine.select_tier(30 * 86400), RollupTier::Day); // 30 days
    }

    #[test]
    fn test_engine_is_tier_enabled() {
        let engine = RollupEngine::new(RollupConfig::default());
        assert!(engine.is_tier_enabled(RollupTier::Raw));
        assert!(engine.is_tier_enabled(RollupTier::Minute));
        assert!(engine.is_tier_enabled(RollupTier::Hour));
        assert!(engine.is_tier_enabled(RollupTier::Day));
    }

    #[test]
    fn test_engine_disabled_tier() {
        let mut config = RollupConfig::default();
        config
            .tiers
            .get_mut(&RollupTier::Hour)
            .expect("exists")
            .enabled = false;
        let engine = RollupEngine::new(config);
        assert!(!engine.is_tier_enabled(RollupTier::Hour));
    }

    #[test]
    fn test_engine_stats() {
        let mut engine = RollupEngine::new(RollupConfig::default());
        let points = make_raw_points(0, 120, 1);
        engine.ingest("sensor_1", &points);

        let stats = engine.stats();
        assert_eq!(stats.total_rollups, 1);
        assert_eq!(stats.total_raw_points_processed, 120);
        assert!(stats.total_rollup_points_produced > 0);
    }

    #[test]
    fn test_engine_reset_stats() {
        let mut engine = RollupEngine::new(RollupConfig::default());
        let points = make_raw_points(0, 60, 1);
        engine.ingest("sensor_1", &points);
        assert!(engine.stats().total_rollups > 0);

        engine.reset_stats();
        assert_eq!(engine.stats().total_rollups, 0);
    }

    #[test]
    fn test_engine_clear_series() {
        let mut engine = RollupEngine::new(RollupConfig::default());
        let points = make_raw_points(0, 60, 1);
        engine.ingest("sensor_1", &points);
        assert!(!engine.series_ids().is_empty());

        assert!(engine.clear_series("sensor_1"));
        assert!(engine.series_ids().is_empty());
    }

    #[test]
    fn test_engine_clear_nonexistent_series() {
        let mut engine = RollupEngine::new(RollupConfig::default());
        assert!(!engine.clear_series("nonexistent"));
    }

    #[test]
    fn test_engine_apply_retention() {
        let mut engine = RollupEngine::new(RollupConfig::default());
        // Create data at timestamp 0 (very old)
        let points = make_raw_points(0, 60, 1);
        engine.ingest("sensor_1", &points);

        let before = engine.point_count("sensor_1", RollupTier::Minute);
        assert!(before > 0);

        // Apply retention with "now" far in the future
        let removed = engine.apply_retention(100_000_000);
        assert!(removed > 0);

        let after = engine.point_count("sensor_1", RollupTier::Minute);
        assert!(after < before);
    }

    #[test]
    fn test_engine_apply_retention_no_removal() {
        let mut engine = RollupEngine::new(RollupConfig::default());
        let now = 1_000_000_000i64;
        let points = make_raw_points(now - 60, 60, 1);
        engine.ingest("sensor_1", &points);

        // Apply retention with "now" only slightly ahead
        let removed = engine.apply_retention(now);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_engine_multiple_series() {
        let mut engine = RollupEngine::new(RollupConfig::default());
        let points1 = make_raw_points(0, 60, 1);
        let points2 = make_raw_points(0, 120, 1);

        engine.ingest("temp", &points1);
        engine.ingest("humidity", &points2);

        assert_eq!(engine.series_ids().len(), 2);
        assert!(engine.point_count("temp", RollupTier::Minute) > 0);
        assert!(engine.point_count("humidity", RollupTier::Minute) > 0);
    }

    #[test]
    fn test_engine_config_access() {
        let config = RollupConfig {
            batch_size: 5000,
            ..Default::default()
        };
        let engine = RollupEngine::new(config);
        assert_eq!(engine.config().batch_size, 5000);
    }

    #[test]
    fn test_aggregate_rollups_to_coarser_tier() {
        let engine = RollupEngine::new(RollupConfig::default());

        // Create 120 minute rollups (2 hours)
        let minute_rollups: Vec<RollupDataPoint> = (0..120)
            .map(|i| RollupDataPoint::from_values(i * 60, &[10.0 + i as f64]).expect("exists"))
            .collect();

        let hour_rollups = engine.aggregate_rollups(&minute_rollups, RollupTier::Hour);
        assert_eq!(hour_rollups.len(), 2);
        assert_eq!(hour_rollups[0].count, 60);
        assert_eq!(hour_rollups[1].count, 60);
    }

    #[test]
    fn test_rollup_data_preserves_min_max() {
        let engine = RollupEngine::new(RollupConfig::default());
        let points = vec![
            RawDataPoint::new(0, 100.0),
            RawDataPoint::new(1, -50.0),
            RawDataPoint::new(2, 200.0),
            RawDataPoint::new(30, 0.0),
        ];

        let rollups = engine.aggregate(&points, RollupTier::Minute);
        assert_eq!(rollups.len(), 1);
        assert!((rollups[0].min - (-50.0)).abs() < 0.001);
        assert!((rollups[0].max - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_per_tier_stats() {
        let mut engine = RollupEngine::new(RollupConfig::default());
        let points = make_raw_points(0, 60, 1);
        engine.ingest("s1", &points);

        let stats = engine.stats();
        assert!(stats.per_tier_rollups.contains_key("minute"));
        assert!(stats.per_tier_points.contains_key("minute"));
    }
}
