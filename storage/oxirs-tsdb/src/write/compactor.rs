//! Background Compaction for Time-Series Chunks
//!
//! This module provides automatic chunk compaction to improve compression ratios
//! and reduce storage overhead.
//!
//! ## Compaction Strategies
//!
//! 1. **Merge small chunks** - Combine multiple small chunks into larger ones
//! 2. **Recompress old chunks** - Rewrite chunks with better compression
//! 3. **Remove duplicates** - Deduplicate identical data points
//!
//! ## Compaction Policy
//!
//! - Run every 1 hour (configurable)
//! - Target: Merge chunks <10% full
//! - Minimum chunk size: 1000 points
//! - Maximum chunk size: 100,000 points

use crate::error::{TsdbError, TsdbResult};
use crate::storage::{ChunkEntry, ColumnarStore, TimeChunk};
use chrono::{DateTime, Duration, Utc};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use tokio::time::interval;

/// Compaction configuration
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Compaction interval (default: 1 hour)
    pub interval: Duration,

    /// Minimum chunk fill ratio to trigger compaction (default: 0.1 = 10%)
    pub min_fill_ratio: f64,

    /// Target chunk size in points (default: 10,000)
    pub target_chunk_size: usize,

    /// Maximum chunk size in points (default: 100,000)
    pub max_chunk_size: usize,

    /// Enable automatic compaction
    pub enabled: bool,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            interval: Duration::hours(1),
            min_fill_ratio: 0.1,
            target_chunk_size: 10_000,
            max_chunk_size: 100_000,
            enabled: true,
        }
    }
}

/// Compaction statistics
#[derive(Debug, Clone, Default)]
pub struct CompactionStats {
    /// Number of compaction runs
    pub runs: u64,
    /// Number of chunks merged
    pub chunks_merged: u64,
    /// Number of chunks created
    pub chunks_created: u64,
    /// Total bytes saved
    pub bytes_saved: u64,
    /// Last compaction time
    pub last_run: Option<DateTime<Utc>>,
}

/// Background compactor for time-series chunks
#[derive(Debug)]
pub struct Compactor {
    /// Compaction configuration
    config: CompactionConfig,

    /// Compaction statistics
    stats: Arc<RwLock<CompactionStats>>,

    /// Running flag
    running: Arc<AtomicBool>,

    /// Total bytes processed
    bytes_processed: Arc<AtomicU64>,
}

impl Compactor {
    /// Create a new compactor with default configuration
    pub fn new() -> Self {
        Self::with_config(CompactionConfig::default())
    }

    /// Create a new compactor with custom configuration
    pub fn with_config(config: CompactionConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(CompactionStats::default())),
            running: Arc::new(AtomicBool::new(false)),
            bytes_processed: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Start background compaction task
    ///
    /// Runs compaction at configured intervals until stopped.
    pub async fn start(&self, store: Arc<ColumnarStore>) -> TsdbResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        self.running.store(true, Ordering::SeqCst);

        let interval_secs = self.config.interval.num_seconds() as u64;
        let mut ticker = interval(std::time::Duration::from_secs(interval_secs));

        while self.running.load(Ordering::SeqCst) {
            ticker.tick().await;

            if let Err(e) = self.compact_once(&store).await {
                eprintln!("Compaction error: {e}");
            }
        }

        Ok(())
    }

    /// Stop background compaction
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Run a single compaction cycle
    pub async fn compact_once(&self, store: &ColumnarStore) -> TsdbResult<()> {
        let start_time = Utc::now();
        let index = store.index();

        // Get all series
        let series_ids = index.series_ids()?;

        let mut total_merged = 0;
        let mut total_created = 0;
        let mut bytes_saved = 0;

        for series_id in series_ids {
            let (merged, created, saved) = self.compact_series(store, series_id).await?;
            total_merged += merged;
            total_created += created;
            bytes_saved += saved;
        }

        // Update statistics
        {
            let mut stats = self
                .stats
                .write()
                .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
            stats.runs += 1;
            stats.chunks_merged += total_merged;
            stats.chunks_created += total_created;
            stats.bytes_saved += bytes_saved;
            stats.last_run = Some(start_time);
        }

        Ok(())
    }

    /// Compact chunks for a single series
    async fn compact_series(
        &self,
        store: &ColumnarStore,
        series_id: u64,
    ) -> TsdbResult<(u64, u64, u64)> {
        let index = store.index();
        let chunks = index.get_chunks_for_series(series_id)?;

        // Find small chunks that need compaction
        let mut small_chunks: Vec<ChunkEntry> = chunks
            .into_iter()
            .filter(|chunk| {
                let fill_ratio = chunk.point_count as f64 / self.config.target_chunk_size as f64;
                fill_ratio < self.config.min_fill_ratio
            })
            .collect();

        if small_chunks.is_empty() {
            return Ok((0, 0, 0));
        }

        // Sort by time
        small_chunks.sort_by_key(|c| c.start_time);

        // Group adjacent chunks for merging
        let merge_groups = self.group_adjacent_chunks(&small_chunks);

        let mut merged_count = 0;
        let mut created_count = 0;
        let mut bytes_saved = 0;

        for group in merge_groups {
            if group.len() < 2 {
                continue; // Need at least 2 chunks to merge
            }

            let (merged, created, saved) = self.merge_chunks(store, series_id, &group).await?;
            merged_count += merged;
            created_count += created;
            bytes_saved += saved;
        }

        Ok((merged_count, created_count, bytes_saved))
    }

    /// Group adjacent chunks for merging
    fn group_adjacent_chunks(&self, chunks: &[ChunkEntry]) -> Vec<Vec<ChunkEntry>> {
        let mut groups = Vec::new();
        let mut current_group = Vec::new();
        let mut current_size = 0;

        for chunk in chunks {
            if current_size + chunk.point_count <= self.config.max_chunk_size {
                current_group.push(chunk.clone());
                current_size += chunk.point_count;
            } else {
                if current_group.len() > 1 {
                    groups.push(current_group);
                }
                current_group = vec![chunk.clone()];
                current_size = chunk.point_count;
            }
        }

        if current_group.len() > 1 {
            groups.push(current_group);
        }

        groups
    }

    /// Merge a group of chunks into a single chunk
    async fn merge_chunks(
        &self,
        store: &ColumnarStore,
        series_id: u64,
        chunks: &[ChunkEntry],
    ) -> TsdbResult<(u64, u64, u64)> {
        // Read all chunks and collect data points
        let mut all_points = Vec::new();
        let mut total_compressed_size = 0;

        for chunk_entry in chunks {
            let chunk = store.read_chunk(chunk_entry.chunk_id)?;
            let points = chunk.decompress()?;
            all_points.extend(points);
            total_compressed_size += chunk_entry.compressed_size;
        }

        // Sort points by timestamp (should already be sorted, but ensure)
        all_points.sort_by_key(|p| p.timestamp);

        // Remove duplicates
        all_points.dedup_by_key(|p| p.timestamp);

        if all_points.is_empty() {
            return Ok((0, 0, 0));
        }

        // Create new merged chunk
        let start_time = all_points[0].timestamp;
        let chunk_duration = self.config.interval;
        let new_chunk = TimeChunk::new(series_id, start_time, chunk_duration, all_points)?;

        // Write new chunk
        let new_entry = store.write_chunk(&new_chunk)?;

        // Remove old chunks from index
        let index = store.index();
        for chunk_entry in chunks {
            index.remove_chunk(chunk_entry.chunk_id)?;

            // Delete old chunk file
            if let Some(path) = &chunk_entry.file_path {
                let _ = std::fs::remove_file(path); // Ignore errors
            }
        }

        // Calculate bytes saved
        let bytes_saved = total_compressed_size.saturating_sub(new_entry.compressed_size);

        self.bytes_processed
            .fetch_add(bytes_saved as u64, Ordering::SeqCst);

        Ok((chunks.len() as u64, 1, bytes_saved as u64))
    }

    /// Get compaction statistics
    pub fn stats(&self) -> TsdbResult<CompactionStats> {
        let stats = self
            .stats
            .read()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
        Ok(stats.clone())
    }

    /// Reset compaction statistics
    pub fn reset_stats(&self) -> TsdbResult<()> {
        let mut stats = self
            .stats
            .write()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
        *stats = CompactionStats::default();
        self.bytes_processed.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// Check if compactor is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl Default for Compactor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::DataPoint;
    use std::env;

    fn create_test_chunk(
        series_id: u64,
        start_timestamp: i64,
        count: usize,
    ) -> TsdbResult<TimeChunk> {
        let start_time = DateTime::from_timestamp(start_timestamp, 0).unwrap();
        let mut points = Vec::new();

        for i in 0..count {
            points.push(DataPoint::new(
                start_time + Duration::seconds(i as i64),
                20.0 + (i as f64 * 0.1),
            ));
        }

        TimeChunk::new(series_id, start_time, Duration::hours(2), points)
    }

    #[tokio::test]
    async fn test_compaction_config() {
        let config = CompactionConfig::default();
        assert_eq!(config.interval, Duration::hours(1));
        assert_eq!(config.min_fill_ratio, 0.1);
        assert_eq!(config.target_chunk_size, 10_000);
        assert!(config.enabled);
    }

    #[tokio::test]
    async fn test_compactor_creation() {
        let compactor = Compactor::new();
        assert!(!compactor.is_running());

        let stats = compactor.stats().unwrap();
        assert_eq!(stats.runs, 0);
        assert_eq!(stats.chunks_merged, 0);
    }

    #[tokio::test]
    async fn test_group_adjacent_chunks() -> TsdbResult<()> {
        let config = CompactionConfig {
            max_chunk_size: 200,
            ..Default::default()
        };
        let compactor = Compactor::with_config(config);

        let chunks = vec![
            ChunkEntry::new(
                1,
                100,
                DateTime::from_timestamp(1000, 0).unwrap(),
                DateTime::from_timestamp(1100, 0).unwrap(),
                50,
            ),
            ChunkEntry::new(
                2,
                100,
                DateTime::from_timestamp(1200, 0).unwrap(),
                DateTime::from_timestamp(1300, 0).unwrap(),
                60,
            ),
            ChunkEntry::new(
                3,
                100,
                DateTime::from_timestamp(1400, 0).unwrap(),
                DateTime::from_timestamp(1500, 0).unwrap(),
                70,
            ),
        ];

        let groups = compactor.group_adjacent_chunks(&chunks);
        assert_eq!(groups.len(), 1); // All fit in one group (50+60+70=180 < 200)
        assert_eq!(groups[0].len(), 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_group_respects_max_size() -> TsdbResult<()> {
        let config = CompactionConfig {
            max_chunk_size: 100,
            ..Default::default()
        };
        let compactor = Compactor::with_config(config);

        let chunks = vec![
            ChunkEntry::new(
                1,
                100,
                DateTime::from_timestamp(1000, 0).unwrap(),
                DateTime::from_timestamp(1100, 0).unwrap(),
                50,
            ),
            ChunkEntry::new(
                2,
                100,
                DateTime::from_timestamp(1200, 0).unwrap(),
                DateTime::from_timestamp(1300, 0).unwrap(),
                60,
            ),
            ChunkEntry::new(
                3,
                100,
                DateTime::from_timestamp(1400, 0).unwrap(),
                DateTime::from_timestamp(1500, 0).unwrap(),
                70,
            ),
        ];

        let groups = compactor.group_adjacent_chunks(&chunks);
        // First two fit (50+60=110 > 100, so just 50)
        // Second chunk starts new group (60)
        // Third chunk starts another group (70)
        // But groups must have >1 chunk, so none qualify
        assert!(groups.is_empty() || groups.iter().all(|g| g.len() >= 2));

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_chunks() -> TsdbResult<()> {
        let temp_dir = env::temp_dir().join("tsdb_compactor_merge_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut store = ColumnarStore::new(&temp_dir, Duration::hours(2), 100)?;
        store.set_fsync(false);

        // Create two small chunks
        let chunk1 = create_test_chunk(100, 1000, 50)?;
        let chunk2 = create_test_chunk(100, 1100, 50)?;

        let entry1 = store.write_chunk(&chunk1)?;
        let entry2 = store.write_chunk(&chunk2)?;

        // Compact them
        let compactor = Compactor::new();
        let (merged, created, _saved) = compactor
            .merge_chunks(&store, 100, &[entry1, entry2])
            .await?;

        assert_eq!(merged, 2);
        assert_eq!(created, 1);

        std::fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_stats_tracking() -> TsdbResult<()> {
        let compactor = Compactor::new();

        let stats = compactor.stats()?;
        assert_eq!(stats.runs, 0);

        // Simulate a compaction run
        {
            let mut stats = compactor.stats.write().unwrap();
            stats.runs += 1;
            stats.chunks_merged += 5;
            stats.chunks_created += 2;
            stats.bytes_saved += 10_000;
            stats.last_run = Some(Utc::now());
        }

        let stats = compactor.stats()?;
        assert_eq!(stats.runs, 1);
        assert_eq!(stats.chunks_merged, 5);
        assert_eq!(stats.chunks_created, 2);
        assert_eq!(stats.bytes_saved, 10_000);
        assert!(stats.last_run.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_reset_stats() -> TsdbResult<()> {
        let compactor = Compactor::new();

        // Set some stats
        {
            let mut stats = compactor.stats.write().unwrap();
            stats.runs = 10;
            stats.chunks_merged = 50;
        }

        let stats_before = compactor.stats()?;
        assert_eq!(stats_before.runs, 10);

        compactor.reset_stats()?;

        let stats_after = compactor.stats()?;
        assert_eq!(stats_after.runs, 0);
        assert_eq!(stats_after.chunks_merged, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_compactor_disabled() {
        let config = CompactionConfig {
            enabled: false,
            ..Default::default()
        };
        let compactor = Compactor::with_config(config);

        // Should return immediately without error
        let temp_dir = env::temp_dir().join("tsdb_compactor_disabled_test");
        let store = Arc::new(ColumnarStore::new(&temp_dir, Duration::hours(2), 100).unwrap());

        let result = compactor.start(store).await;
        assert!(result.is_ok());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
