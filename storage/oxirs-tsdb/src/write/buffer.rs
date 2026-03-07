//! In-memory write buffer for batching data points
//!
//! Buffers incoming data points before compressing and writing to disk.
//! Provides flush triggers based on size and time.

use crate::error::TsdbResult;
use crate::series::DataPoint;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Write buffer configuration
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Maximum number of points before auto-flush
    pub max_points: usize,
    /// Maximum time before auto-flush (in seconds)
    pub max_age_secs: u64,
    /// Enable automatic background flushing
    pub auto_flush: bool,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_points: 100_000, // 100K points default
            max_age_secs: 60,    // 1 minute default
            auto_flush: true,
        }
    }
}

/// In-memory buffer for a single time series
#[derive(Debug, Clone)]
struct SeriesBuffer {
    /// Buffered data points
    points: Vec<DataPoint>,
    /// Time when first point was added
    first_insert: Instant,
}

impl SeriesBuffer {
    fn new() -> Self {
        Self {
            points: Vec::new(),
            first_insert: Instant::now(),
        }
    }

    fn push(&mut self, point: DataPoint) {
        if self.points.is_empty() {
            self.first_insert = Instant::now();
        }
        self.points.push(point);
    }

    fn age_secs(&self) -> u64 {
        self.first_insert.elapsed().as_secs()
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    fn drain(&mut self) -> Vec<DataPoint> {
        let points = std::mem::take(&mut self.points);
        self.first_insert = Instant::now();
        points
    }
}

/// Write buffer for time-series data
///
/// Provides thread-safe buffering with automatic flush triggers.
pub struct WriteBuffer {
    /// Configuration
    config: BufferConfig,
    /// Per-series buffers (series_id → buffer)
    buffers: Arc<RwLock<HashMap<u64, SeriesBuffer>>>,
}

impl WriteBuffer {
    /// Create new write buffer
    pub fn new(config: BufferConfig) -> Self {
        Self {
            config,
            buffers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(BufferConfig::default())
    }

    /// Insert a data point into the buffer
    ///
    /// Returns true if buffer should be flushed
    pub async fn insert(&self, series_id: u64, point: DataPoint) -> TsdbResult<bool> {
        let mut buffers = self.buffers.write().await;
        let buffer = buffers.entry(series_id).or_insert_with(SeriesBuffer::new);

        buffer.push(point);

        // Check if flush is needed
        let should_flush =
            buffer.len() >= self.config.max_points || buffer.age_secs() >= self.config.max_age_secs;

        Ok(should_flush)
    }

    /// Insert multiple points in a batch
    pub async fn insert_batch(&self, entries: &[(u64, DataPoint)]) -> TsdbResult<bool> {
        let mut buffers = self.buffers.write().await;
        let mut should_flush = false;

        for (series_id, point) in entries {
            let buffer = buffers.entry(*series_id).or_insert_with(SeriesBuffer::new);

            buffer.push(*point);

            if buffer.len() >= self.config.max_points
                || buffer.age_secs() >= self.config.max_age_secs
            {
                should_flush = true;
            }
        }

        Ok(should_flush)
    }

    /// Flush all buffers for a specific series
    pub async fn flush_series(&self, series_id: u64) -> TsdbResult<Vec<DataPoint>> {
        let mut buffers = self.buffers.write().await;

        if let Some(buffer) = buffers.get_mut(&series_id) {
            Ok(buffer.drain())
        } else {
            Ok(Vec::new())
        }
    }

    /// Flush all buffers
    ///
    /// Returns map of series_id → data points
    pub async fn flush_all(&self) -> TsdbResult<HashMap<u64, Vec<DataPoint>>> {
        let mut buffers = self.buffers.write().await;
        let mut result = HashMap::new();

        for (series_id, buffer) in buffers.iter_mut() {
            if !buffer.is_empty() {
                result.insert(*series_id, buffer.drain());
            }
        }

        Ok(result)
    }

    /// Get buffer statistics
    pub async fn stats(&self) -> BufferStats {
        let buffers = self.buffers.read().await;

        let total_points: usize = buffers.values().map(|b| b.len()).sum();
        let num_series = buffers.len();
        let oldest_age = buffers
            .values()
            .filter(|b| !b.is_empty())
            .map(|b| b.age_secs())
            .max()
            .unwrap_or(0);

        BufferStats {
            total_points,
            num_series,
            oldest_age_secs: oldest_age,
        }
    }

    /// Check which series need flushing
    pub async fn series_needing_flush(&self) -> Vec<u64> {
        let buffers = self.buffers.read().await;

        buffers
            .iter()
            .filter(|(_, buffer)| {
                buffer.len() >= self.config.max_points
                    || buffer.age_secs() >= self.config.max_age_secs
            })
            .map(|(series_id, _)| *series_id)
            .collect()
    }
}

/// Buffer statistics
#[derive(Debug, Clone)]
pub struct BufferStats {
    /// Total buffered data points across all series
    pub total_points: usize,
    /// Number of series with buffered data
    pub num_series: usize,
    /// Age of oldest buffered data (in seconds)
    pub oldest_age_secs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[tokio::test]
    async fn test_buffer_insert() {
        let buffer = WriteBuffer::with_defaults();

        let point = DataPoint {
            timestamp: Utc::now(),
            value: 22.5,
        };

        let should_flush = buffer
            .insert(1, point)
            .await
            .expect("async operation should succeed");
        assert!(!should_flush); // Should not flush with just 1 point

        let stats = buffer.stats().await;
        assert_eq!(stats.total_points, 1);
        assert_eq!(stats.num_series, 1);
    }

    #[tokio::test]
    async fn test_buffer_flush_on_size() {
        let config = BufferConfig {
            max_points: 10,
            max_age_secs: 3600,
            auto_flush: false,
        };
        let buffer = WriteBuffer::new(config);

        let base_time = Utc::now();

        // Insert 9 points - should not trigger flush
        for i in 0..9 {
            let point = DataPoint {
                timestamp: base_time + chrono::Duration::seconds(i),
                value: i as f64,
            };
            let should_flush = buffer
                .insert(1, point)
                .await
                .expect("async operation should succeed");
            assert!(!should_flush);
        }

        // Insert 10th point - should trigger flush
        let point = DataPoint {
            timestamp: base_time + chrono::Duration::seconds(9),
            value: 9.0,
        };
        let should_flush = buffer
            .insert(1, point)
            .await
            .expect("async operation should succeed");
        assert!(should_flush);
    }

    #[tokio::test]
    async fn test_buffer_flush_series() {
        let buffer = WriteBuffer::with_defaults();

        let base_time = Utc::now();

        // Insert to series 1
        for i in 0..5 {
            let point = DataPoint {
                timestamp: base_time + chrono::Duration::seconds(i),
                value: i as f64,
            };
            buffer
                .insert(1, point)
                .await
                .expect("async operation should succeed");
        }

        // Insert to series 2
        for i in 0..3 {
            let point = DataPoint {
                timestamp: base_time + chrono::Duration::seconds(i),
                value: (i + 100) as f64,
            };
            buffer
                .insert(2, point)
                .await
                .expect("async operation should succeed");
        }

        // Flush only series 1
        let points = buffer
            .flush_series(1)
            .await
            .expect("async operation should succeed");
        assert_eq!(points.len(), 5);

        // Series 2 should still have data
        let stats = buffer.stats().await;
        assert_eq!(stats.total_points, 3);
    }

    #[tokio::test]
    async fn test_buffer_flush_all() {
        let buffer = WriteBuffer::with_defaults();

        let base_time = Utc::now();

        // Insert to multiple series
        for series_id in 1_u64..=5_u64 {
            for i in 0..10_i64 {
                let point = DataPoint {
                    timestamp: base_time + chrono::Duration::seconds(i),
                    value: (series_id * 100 + i as u64) as f64,
                };
                buffer
                    .insert(series_id, point)
                    .await
                    .expect("async operation should succeed");
            }
        }

        let stats = buffer.stats().await;
        assert_eq!(stats.total_points, 50); // 5 series × 10 points
        assert_eq!(stats.num_series, 5);

        // Flush all
        let flushed = buffer
            .flush_all()
            .await
            .expect("async operation should succeed");
        assert_eq!(flushed.len(), 5);
        assert_eq!(flushed.get(&1).expect("key should exist").len(), 10);

        // Buffer should be empty
        let stats = buffer.stats().await;
        assert_eq!(stats.total_points, 0);
    }

    #[tokio::test]
    async fn test_batch_insert() {
        let buffer = WriteBuffer::with_defaults();

        let base_time = Utc::now();
        let mut batch = Vec::new();

        for i in 0..100 {
            let point = DataPoint {
                timestamp: base_time + chrono::Duration::seconds(i),
                value: i as f64,
            };
            batch.push((1, point));
        }

        buffer
            .insert_batch(&batch)
            .await
            .expect("async operation should succeed");

        let stats = buffer.stats().await;
        assert_eq!(stats.total_points, 100);
    }

    #[tokio::test]
    async fn test_series_needing_flush() {
        let config = BufferConfig {
            max_points: 10,
            max_age_secs: 3600,
            auto_flush: false,
        };
        let buffer = WriteBuffer::new(config);

        let base_time = Utc::now();

        // Series 1: 5 points (no flush needed)
        for i in 0..5 {
            let point = DataPoint {
                timestamp: base_time + chrono::Duration::seconds(i),
                value: i as f64,
            };
            buffer
                .insert(1, point)
                .await
                .expect("async operation should succeed");
        }

        // Series 2: 10 points (flush needed)
        for i in 0..10_i64 {
            let point = DataPoint {
                timestamp: base_time + chrono::Duration::seconds(i),
                value: i as f64,
            };
            buffer
                .insert(2, point)
                .await
                .expect("async operation should succeed");
        }

        let to_flush = buffer.series_needing_flush().await;
        assert_eq!(to_flush.len(), 1);
        assert_eq!(to_flush[0], 2);
    }
}
