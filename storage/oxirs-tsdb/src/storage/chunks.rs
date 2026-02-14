//! Time-based chunk management
//!
//! Time-series data is organized into fixed-duration chunks (default: 2 hours)
//! for efficient compression and querying.

use crate::error::{TsdbError, TsdbResult};
use crate::series::DataPoint;
use crate::storage::compression::{
    DeltaOfDeltaCompressor, DeltaOfDeltaDecompressor, GorillaCompressor, GorillaDecompressor,
};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Time chunk containing compressed time-series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeChunk {
    /// Series identifier
    pub series_id: u64,

    /// Chunk start time (aligned to chunk interval)
    pub start_time: DateTime<Utc>,

    /// Chunk end time
    pub end_time: DateTime<Utc>,

    /// Compressed timestamps (Delta-of-delta encoding)
    pub compressed_timestamps: Vec<u8>,

    /// Compressed values (Gorilla encoding)
    pub compressed_values: Vec<u8>,

    /// Chunk metadata
    pub metadata: ChunkMetadata,
}

/// Metadata about a time chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Number of data points in chunk
    pub count: usize,

    /// Minimum value in chunk
    pub min_value: f64,

    /// Maximum value in chunk
    pub max_value: f64,

    /// Average value in chunk
    pub avg_value: f64,

    /// Uncompressed size (bytes)
    pub uncompressed_size: usize,

    /// Compressed size (bytes)
    pub compressed_size: usize,
}

impl TimeChunk {
    /// Create a new chunk from data points
    ///
    /// # Arguments
    ///
    /// * `series_id` - Unique series identifier
    /// * `start_time` - Chunk start time (should be aligned)
    /// * `chunk_duration` - Duration of this chunk (e.g., 2 hours)
    /// * `points` - Data points to compress
    pub fn new(
        series_id: u64,
        start_time: DateTime<Utc>,
        chunk_duration: Duration,
        points: Vec<DataPoint>,
    ) -> TsdbResult<Self> {
        if points.is_empty() {
            return Err(TsdbError::Compression(
                "Cannot create chunk from empty data".to_string(),
            ));
        }

        // Compress timestamps
        let compressed_timestamps = Self::compress_timestamps(&points)?;

        // Compress values
        let compressed_values = Self::compress_values(&points)?;

        // Calculate metadata
        let min_value = points
            .iter()
            .map(|p| p.value)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("collection should not be empty");

        let max_value = points
            .iter()
            .map(|p| p.value)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("collection should not be empty");

        let sum: f64 = points.iter().map(|p| p.value).sum();
        let avg_value = sum / points.len() as f64;

        let metadata = ChunkMetadata {
            count: points.len(),
            min_value,
            max_value,
            avg_value,
            uncompressed_size: points.len() * 16, // 8 bytes timestamp + 8 bytes value
            compressed_size: compressed_timestamps.len() + compressed_values.len(),
        };

        Ok(Self {
            series_id,
            start_time,
            end_time: start_time + chunk_duration,
            compressed_timestamps,
            compressed_values,
            metadata,
        })
    }

    /// Compress timestamps using Delta-of-delta encoding
    fn compress_timestamps(points: &[DataPoint]) -> TsdbResult<Vec<u8>> {
        let first_ts = points[0].timestamp.timestamp_millis();
        let mut compressor = DeltaOfDeltaCompressor::new(first_ts);

        for point in &points[1..] {
            let ts = point.timestamp.timestamp_millis();
            compressor.compress(ts);
        }

        Ok(compressor.finish())
    }

    /// Compress values using Gorilla encoding
    fn compress_values(points: &[DataPoint]) -> TsdbResult<Vec<u8>> {
        let first_value = points[0].value;
        let mut compressor = GorillaCompressor::new(first_value);

        for point in &points[1..] {
            compressor.compress(point.value);
        }

        Ok(compressor.finish())
    }

    /// Decompress all data points in chunk
    pub fn decompress(&self) -> TsdbResult<Vec<DataPoint>> {
        let timestamps = self.decompress_timestamps()?;
        let values = self.decompress_values()?;

        if timestamps.len() != values.len() {
            return Err(TsdbError::Decompression(format!(
                "Timestamp count ({}) != value count ({})",
                timestamps.len(),
                values.len()
            )));
        }

        Ok(timestamps
            .into_iter()
            .zip(values)
            .map(|(ts_millis, value)| DataPoint {
                timestamp: DateTime::from_timestamp_millis(ts_millis).unwrap_or_else(Utc::now),
                value,
            })
            .collect())
    }

    /// Decompress timestamps
    fn decompress_timestamps(&self) -> TsdbResult<Vec<i64>> {
        let decompressor = DeltaOfDeltaDecompressor::new(&self.compressed_timestamps)?;
        Ok(decompressor.decompress_all())
    }

    /// Decompress values
    fn decompress_values(&self) -> TsdbResult<Vec<f64>> {
        let decompressor = GorillaDecompressor::new(&self.compressed_values)?;
        Ok(decompressor.decompress_all())
    }

    /// Query data points within time range
    ///
    /// This is more efficient than decompress() + filter for range queries.
    pub fn query_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> TsdbResult<Vec<DataPoint>> {
        // Current implementation: decompress all and filter
        // Performance optimization opportunity: Early-exit during decompression when
        // timestamp exceeds end range (requires streaming decompressor API)
        let all_points = self.decompress()?;

        Ok(all_points
            .into_iter()
            .filter(|p| p.timestamp >= start && p.timestamp < end)
            .collect())
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        self.metadata.uncompressed_size as f64 / self.metadata.compressed_size as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_chunk_creation() {
        let series_id = 1;
        let start_time = Utc::now();
        let chunk_duration = Duration::hours(2);

        // Create 100 data points with small variations
        let mut points = Vec::new();
        for i in 0..100 {
            points.push(DataPoint {
                timestamp: start_time + Duration::seconds(i),
                value: 20.0 + (i as f64 * 0.1),
            });
        }

        let chunk = TimeChunk::new(series_id, start_time, chunk_duration, points.clone())
            .expect("construction should succeed");

        assert_eq!(chunk.series_id, 1);
        assert_eq!(chunk.metadata.count, 100);
        assert!(chunk.metadata.min_value >= 20.0);
        assert!(chunk.metadata.max_value <= 30.0);
        assert!(chunk.compression_ratio() > 1.0);
    }

    #[test]
    fn test_chunk_round_trip() {
        let series_id = 1;
        let start_time = Utc::now();
        let chunk_duration = Duration::hours(2);

        let mut points = Vec::new();
        for i in 0..1000 {
            points.push(DataPoint {
                timestamp: start_time + Duration::seconds(i),
                value: 22.5 + (i as f64 % 10_f64) * 0.1,
            });
        }

        let chunk = TimeChunk::new(series_id, start_time, chunk_duration, points.clone())
            .expect("construction should succeed");
        let decompressed = chunk.decompress().expect("start should succeed");

        assert_eq!(points.len(), decompressed.len());

        // Verify all values match
        for (original, decompressed_point) in points.iter().zip(decompressed.iter()) {
            assert_eq!(
                original.timestamp.timestamp_millis(),
                decompressed_point.timestamp.timestamp_millis()
            );
            assert_eq!(original.value, decompressed_point.value);
        }
    }

    #[test]
    fn test_chunk_range_query() {
        let series_id = 1;
        let start_time = Utc::now();
        let chunk_duration = Duration::hours(2);

        let mut points = Vec::new();
        for i in 0..100 {
            points.push(DataPoint {
                timestamp: start_time + Duration::seconds(i * 10),
                value: 20.0,
            });
        }

        let chunk = TimeChunk::new(series_id, start_time, chunk_duration, points)
            .expect("construction should succeed");

        // Query middle 50 points
        let query_start = start_time + Duration::seconds(250);
        let query_end = start_time + Duration::seconds(750);

        let results = chunk
            .query_range(query_start, query_end)
            .expect("query should succeed");

        // Should get points in range [250s, 750s) = indices 25-74 = 50 points
        assert!(results.len() >= 45 && results.len() <= 55); // Allow some tolerance
    }

    #[test]
    fn test_high_compression_ratio() {
        // Stable sensor reading (temperature in controlled environment)
        let series_id = 1;
        let start_time = Utc::now();
        let chunk_duration = Duration::hours(2);

        let mut points = Vec::new();
        for i in 0..7200 {
            // 1 Hz for 2 hours = 7200 points
            points.push(DataPoint {
                timestamp: start_time + Duration::seconds(i),
                value: 22.5, // Constant temperature
            });
        }

        let chunk = TimeChunk::new(series_id, start_time, chunk_duration, points)
            .expect("construction should succeed");

        let ratio = chunk.compression_ratio();
        println!("Stable sensor compression ratio: {:.1}:1", ratio);

        // Should achieve very high compression (>50:1) for constant values
        assert!(ratio > 50.0);
    }
}
