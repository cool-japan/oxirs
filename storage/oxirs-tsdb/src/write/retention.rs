//! Retention Policy Enforcement for Time-Series Data
//!
//! This module provides automatic data expiration and downsampling
//! based on configurable retention policies.
//!
//! ## Retention Strategies
//!
//! 1. **Time-based expiration** - Delete data older than N days
//! 2. **Downsampling** - Reduce resolution for old data (1s → 1m → 1h)
//! 3. **Selective retention** - Keep aggregates, discard raw data
//!
//! ## Example Policy
//!
//! ```text
//! Raw data (1s):     Keep for 7 days
//! 1-minute avg:      Keep for 30 days
//! 1-hour avg:        Keep for 1 year
//! Daily summary:     Keep forever
//! ```

use crate::config::{AggregationFunction, RetentionPolicy};
use crate::error::{TsdbError, TsdbResult};
use crate::query::{Aggregation, ResampleBucket, Resampler};
use crate::series::DataPoint;
use crate::storage::{ChunkEntry, ColumnarStore, TimeChunk};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use tokio::time::interval;

/// Retention enforcer statistics
#[derive(Debug, Clone, Default)]
pub struct RetentionStats {
    /// Number of enforcement runs
    pub runs: u64,
    /// Number of chunks deleted
    pub chunks_deleted: u64,
    /// Number of chunks downsampled
    pub chunks_downsampled: u64,
    /// Total bytes freed
    pub bytes_freed: u64,
    /// Last run time
    pub last_run: Option<DateTime<Utc>>,
}

/// Retention policy enforcer
#[derive(Debug)]
pub struct RetentionEnforcer {
    /// Retention policies to enforce
    policies: Vec<RetentionPolicy>,

    /// Statistics
    stats: Arc<RwLock<RetentionStats>>,

    /// Total points deleted
    points_deleted: Arc<AtomicU64>,

    /// Enforcement interval (default: 1 day)
    enforcement_interval: ChronoDuration,
}

impl RetentionEnforcer {
    /// Create a new retention enforcer
    pub fn new(policies: Vec<RetentionPolicy>) -> Self {
        Self {
            policies,
            stats: Arc::new(RwLock::new(RetentionStats::default())),
            points_deleted: Arc::new(AtomicU64::new(0)),
            enforcement_interval: ChronoDuration::days(1),
        }
    }

    /// Set enforcement interval
    pub fn set_enforcement_interval(&mut self, interval: ChronoDuration) {
        self.enforcement_interval = interval;
    }

    /// Start background retention enforcement
    pub async fn start(&self, store: Arc<ColumnarStore>) -> TsdbResult<()> {
        let interval_secs = self.enforcement_interval.num_seconds() as u64;
        let mut ticker = interval(std::time::Duration::from_secs(interval_secs));

        loop {
            ticker.tick().await;

            if let Err(e) = self.enforce_once(&store).await {
                eprintln!("Retention enforcement error: {e}");
            }
        }
    }

    /// Run a single retention enforcement cycle
    pub async fn enforce_once(&self, store: &ColumnarStore) -> TsdbResult<()> {
        let now = Utc::now();
        let index = store.index();
        let series_ids = index.series_ids()?;

        let mut total_deleted = 0;
        let mut total_downsampled = 0;
        let mut bytes_freed = 0;

        for series_id in series_ids {
            let (deleted, downsampled, freed) = self.enforce_series(store, series_id, now).await?;
            total_deleted += deleted;
            total_downsampled += downsampled;
            bytes_freed += freed;
        }

        // Update statistics
        {
            let mut stats = self
                .stats
                .write()
                .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
            stats.runs += 1;
            stats.chunks_deleted += total_deleted;
            stats.chunks_downsampled += total_downsampled;
            stats.bytes_freed += bytes_freed;
            stats.last_run = Some(now);
        }

        Ok(())
    }

    /// Enforce retention for a single series
    async fn enforce_series(
        &self,
        store: &ColumnarStore,
        series_id: u64,
        now: DateTime<Utc>,
    ) -> TsdbResult<(u64, u64, u64)> {
        let index = store.index();
        let chunks = index.get_chunks_for_series(series_id)?;

        let mut deleted_count = 0;
        let mut downsampled_count = 0;
        let mut bytes_freed = 0;

        for chunk_entry in chunks {
            // Find the most senior retention tier this chunk's age has
            // actually exceeded (if any).
            let age = now - chunk_entry.end_time;
            let policy = self.find_policy_for_age(age);

            match policy {
                Some(policy) => {
                    // Data is older than this tier's retention period.
                    if let Some(downsampling) = &policy.downsampling {
                        // Downsample the chunk
                        let (success, saved) = self
                            .downsample_chunk(store, &chunk_entry, downsampling)
                            .await?;
                        if success {
                            downsampled_count += 1;
                            bytes_freed += saved;
                        }
                    } else {
                        // Delete the chunk
                        let freed = self.delete_chunk(store, &chunk_entry).await?;
                        deleted_count += 1;
                        bytes_freed += freed;
                    }
                }
                None => {
                    // No policy exceeded yet (or no policies configured at
                    // all): the chunk is still within every configured
                    // retention window, so keep it untouched.
                }
            }
        }

        Ok((deleted_count, downsampled_count, bytes_freed))
    }

    /// Find the retention tier whose duration this age has *exceeded*.
    ///
    /// Retention policies form a set of tiers keyed by how long data may
    /// live before that tier's action (downsample or delete) applies. A
    /// chunk that has outlived more than one tier's duration should be
    /// enforced under the most senior (largest-duration) tier it has
    /// exceeded — e.g. data that has outlived both a 7-day and a 30-day
    /// tier is handled by the 30-day tier's action, not re-processed under
    /// the 7-day tier. Returns `None` only when the age has not yet
    /// exceeded any configured policy (including when no policies are
    /// configured at all), in which case the data must be kept untouched.
    fn find_policy_for_age(&self, age: ChronoDuration) -> Option<&RetentionPolicy> {
        let age_secs = age.num_seconds().max(0) as u64;
        self.policies
            .iter()
            .filter(|p| age_secs > p.duration.as_secs())
            .max_by_key(|p| p.duration.as_secs())
    }

    /// Downsample a chunk according to downsampling config
    async fn downsample_chunk(
        &self,
        store: &ColumnarStore,
        chunk_entry: &ChunkEntry,
        downsampling: &crate::config::Downsampling,
    ) -> TsdbResult<(bool, u64)> {
        // Read chunk
        let chunk = store.read_chunk(chunk_entry.chunk_id)?;
        let points = chunk.decompress()?;

        if points.is_empty() {
            return Ok((false, 0));
        }

        // Convert to_resolution to ResampleBucket
        let bucket_secs = downsampling.to_resolution.as_secs();
        let bucket = if bucket_secs == 60 {
            ResampleBucket::Minute(1)
        } else if bucket_secs == 3600 {
            ResampleBucket::Hour(1)
        } else if bucket_secs == 86400 {
            ResampleBucket::Day(1)
        } else {
            return Err(TsdbError::Config(format!(
                "Unsupported downsampling resolution: {}s",
                bucket_secs
            )));
        };

        // Convert aggregation
        let aggregation = match downsampling.aggregation {
            AggregationFunction::Average => Aggregation::Avg,
            AggregationFunction::Min => Aggregation::Min,
            AggregationFunction::Max => Aggregation::Max,
            AggregationFunction::Sum => Aggregation::Sum,
            AggregationFunction::First => Aggregation::First,
            AggregationFunction::Last => Aggregation::Last,
        };

        // Resample
        let resampler = Resampler::new(bucket, aggregation);
        let buckets = resampler.resample(&points)?;

        // Convert buckets to data points
        let downsampled_points: Vec<DataPoint> = buckets
            .into_iter()
            .map(|b| DataPoint::new(b.timestamp, b.value))
            .collect();

        if downsampled_points.is_empty() {
            return Ok((false, 0));
        }

        // Create new chunk with downsampled data
        let chunk_duration = ChronoDuration::seconds(downsampling.to_resolution.as_secs() as i64);
        let new_chunk = TimeChunk::new(
            chunk.series_id,
            chunk.start_time,
            chunk_duration,
            downsampled_points,
        )?;

        // Write new chunk
        let new_entry = store.write_chunk(&new_chunk)?;

        // Delete old chunk
        let bytes_saved = chunk_entry
            .compressed_size
            .saturating_sub(new_entry.compressed_size);
        let _freed = self.delete_chunk(store, chunk_entry).await?;

        Ok((true, bytes_saved as u64))
    }

    /// Delete a chunk
    async fn delete_chunk(
        &self,
        store: &ColumnarStore,
        chunk_entry: &ChunkEntry,
    ) -> TsdbResult<u64> {
        let bytes_freed = chunk_entry.compressed_size as u64;

        // Remove from index
        store.index().remove_chunk(chunk_entry.chunk_id)?;

        // Delete file
        if let Some(path) = &chunk_entry.file_path {
            std::fs::remove_file(path)?;
        }

        self.points_deleted
            .fetch_add(chunk_entry.point_count as u64, Ordering::SeqCst);

        Ok(bytes_freed)
    }

    /// Get retention statistics
    pub fn stats(&self) -> TsdbResult<RetentionStats> {
        let stats = self
            .stats
            .read()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
        Ok(stats.clone())
    }

    /// Reset statistics
    pub fn reset_stats(&self) -> TsdbResult<()> {
        let mut stats = self
            .stats
            .write()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
        *stats = RetentionStats::default();
        self.points_deleted.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// Get total points deleted
    pub fn points_deleted(&self) -> u64 {
        self.points_deleted.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Downsampling;
    use std::env;

    fn create_test_chunk(
        series_id: u64,
        start_timestamp: i64,
        count: usize,
    ) -> TsdbResult<TimeChunk> {
        let start_time = DateTime::from_timestamp(start_timestamp, 0).expect("valid timestamp");
        let mut points = Vec::new();

        for i in 0..count {
            points.push(DataPoint::new(
                start_time + ChronoDuration::seconds(i as i64),
                20.0 + (i as f64 * 0.1),
            ));
        }

        TimeChunk::new(series_id, start_time, ChronoDuration::hours(2), points)
    }

    #[tokio::test]
    async fn test_retention_enforcer_creation() {
        let policy = RetentionPolicy {
            name: "7days".to_string(),
            duration: std::time::Duration::from_secs(7 * 24 * 3600),
            downsampling: None,
        };

        let enforcer = RetentionEnforcer::new(vec![policy]);
        let stats = enforcer.stats().expect("operation should succeed");
        assert_eq!(stats.runs, 0);
    }

    #[tokio::test]
    async fn regression_find_policy_for_age_selects_exceeded_tier() {
        // find_policy_for_age must return the tier whose duration the age
        // has *exceeded*, not one it still fits within (the original bug:
        // filtering `age <= duration` made the enforcement guard
        // unreachable). With two tiers (7 days, 30 days):
        //   - age younger than every tier -> None (nothing exceeded yet)
        //   - age between 7 and 30 days -> the 7-day tier is exceeded
        //   - age older than both -> the most senior (30-day) exceeded
        //     tier applies, not None.
        let policies = vec![
            RetentionPolicy {
                name: "7days".to_string(),
                duration: std::time::Duration::from_secs(7 * 24 * 3600),
                downsampling: None,
            },
            RetentionPolicy {
                name: "30days".to_string(),
                duration: std::time::Duration::from_secs(30 * 24 * 3600),
                downsampling: None,
            },
        ];

        let enforcer = RetentionEnforcer::new(policies);

        let age_5days = ChronoDuration::days(5);
        let policy = enforcer.find_policy_for_age(age_5days);
        assert!(
            policy.is_none(),
            "5-day-old data has not exceeded any tier yet"
        );

        let age_20days = ChronoDuration::days(20);
        let policy = enforcer.find_policy_for_age(age_20days);
        assert!(policy.is_some());
        assert_eq!(
            policy.expect("operation should succeed").name,
            "7days",
            "20-day-old data has exceeded only the 7-day tier"
        );

        let age_40days = ChronoDuration::days(40);
        let policy = enforcer.find_policy_for_age(age_40days);
        assert!(policy.is_some());
        assert_eq!(
            policy.expect("operation should succeed").name,
            "30days",
            "data exceeding every configured tier must be enforced under \
             the most senior tier, never kept forever"
        );
    }

    #[tokio::test]
    async fn test_delete_chunk() -> TsdbResult<()> {
        let temp_dir = env::temp_dir().join("tsdb_retention_delete_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut store = ColumnarStore::new(&temp_dir, ChronoDuration::hours(2), 100)?;
        store.set_fsync(false);

        let chunk = create_test_chunk(100, 1000, 50)?;
        let entry = store.write_chunk(&chunk)?;

        assert_eq!(store.index().chunk_count()?, 1);

        let enforcer = RetentionEnforcer::new(vec![]);
        let bytes_freed = enforcer.delete_chunk(&store, &entry).await?;

        assert!(bytes_freed > 0);
        assert_eq!(store.index().chunk_count()?, 0);
        assert_eq!(enforcer.points_deleted(), 50);

        std::fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_stats_tracking() -> TsdbResult<()> {
        let enforcer = RetentionEnforcer::new(vec![]);

        let stats = enforcer.stats()?;
        assert_eq!(stats.runs, 0);

        // Simulate enforcement
        {
            let mut stats = enforcer.stats.write().expect("lock should not be poisoned");
            stats.runs += 1;
            stats.chunks_deleted += 10;
            stats.bytes_freed += 50_000;
            stats.last_run = Some(Utc::now());
        }

        let stats = enforcer.stats()?;
        assert_eq!(stats.runs, 1);
        assert_eq!(stats.chunks_deleted, 10);
        assert_eq!(stats.bytes_freed, 50_000);
        assert!(stats.last_run.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_reset_stats() -> TsdbResult<()> {
        let enforcer = RetentionEnforcer::new(vec![]);

        {
            let mut stats = enforcer.stats.write().expect("lock should not be poisoned");
            stats.runs = 5;
            stats.chunks_deleted = 20;
        }

        enforcer.reset_stats()?;

        let stats = enforcer.stats()?;
        assert_eq!(stats.runs, 0);
        assert_eq!(stats.chunks_deleted, 0);

        Ok(())
    }

    #[tokio::test]
    async fn regression_enforce_once_deletes_expired_chunk() -> TsdbResult<()> {
        // End-to-end: write a chunk whose data is far older than the
        // configured retention duration, run a real enforcement cycle
        // against a real ColumnarStore, and confirm the chunk is actually
        // deleted and stats reflect it. Prior to the fix, enforce_series's
        // guard was unreachable and this chunk would never be removed.
        let temp_dir = env::temp_dir().join("tsdb_retention_enforce_e2e_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut store = ColumnarStore::new(&temp_dir, ChronoDuration::hours(2), 100)?;
        store.set_fsync(false);

        // Chunk data is from 1970 (effectively infinitely old).
        let chunk = create_test_chunk(200, 10_000, 50)?;
        store.write_chunk(&chunk)?;
        assert_eq!(store.index().chunk_count()?, 1);

        // A single short retention tier with no downsampling -> delete.
        let policy = RetentionPolicy {
            name: "1hour".to_string(),
            duration: std::time::Duration::from_secs(3600),
            downsampling: None,
        };
        let enforcer = RetentionEnforcer::new(vec![policy]);

        enforcer.enforce_once(&store).await?;

        let stats = enforcer.stats()?;
        assert_eq!(stats.runs, 1);
        assert!(
            stats.chunks_deleted > 0,
            "expected at least one chunk to be deleted, got stats: {stats:?}"
        );
        assert!(stats.bytes_freed > 0);
        assert_eq!(
            store.index().chunk_count()?,
            0,
            "expired chunk must actually be removed from the index"
        );

        std::fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }

    #[tokio::test]
    async fn regression_enforce_once_keeps_fresh_chunk() -> TsdbResult<()> {
        // A chunk whose age is within every configured tier must survive
        // an enforcement cycle untouched.
        let temp_dir = env::temp_dir().join("tsdb_retention_enforce_keep_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut store = ColumnarStore::new(&temp_dir, ChronoDuration::hours(2), 100)?;
        store.set_fsync(false);

        let now_ts = Utc::now().timestamp();
        let chunk = create_test_chunk(201, now_ts, 5)?;
        store.write_chunk(&chunk)?;
        assert_eq!(store.index().chunk_count()?, 1);

        let policy = RetentionPolicy {
            name: "30days".to_string(),
            duration: std::time::Duration::from_secs(30 * 24 * 3600),
            downsampling: None,
        };
        let enforcer = RetentionEnforcer::new(vec![policy]);

        enforcer.enforce_once(&store).await?;

        let stats = enforcer.stats()?;
        assert_eq!(stats.chunks_deleted, 0);
        assert_eq!(stats.chunks_downsampled, 0);
        assert_eq!(
            store.index().chunk_count()?,
            1,
            "fresh chunk within retention window must be kept"
        );

        std::fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_downsampling_config() {
        let downsampling = Downsampling {
            from_resolution: std::time::Duration::from_secs(1),
            to_resolution: std::time::Duration::from_secs(60),
            aggregation: AggregationFunction::Average,
        };

        assert_eq!(downsampling.from_resolution.as_secs(), 1);
        assert_eq!(downsampling.to_resolution.as_secs(), 60);
        assert_eq!(downsampling.aggregation, AggregationFunction::Average);
    }
}
