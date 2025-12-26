//! Columnar storage for efficient time-series data access
//!
//! This module provides column-oriented storage with memory-mapped file I/O
//! for efficient compression and query performance.

use crate::error::{TsdbError, TsdbResult};
use crate::series::DataPoint;
use crate::storage::{ChunkEntry, SeriesIndex, TimeChunk};
use chrono::{DateTime, Duration, Utc};
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Columnar storage engine for time-series data
///
/// Features:
/// - Column-oriented layout for better compression
/// - Memory-mapped file I/O for large datasets
/// - Atomic writes with fsync
/// - Crash recovery support
#[derive(Debug)]
pub struct ColumnarStore {
    /// Base directory for chunk files
    base_path: PathBuf,

    /// Series index for chunk lookups
    index: Arc<SeriesIndex>,

    /// Chunk duration (default: 2 hours)
    #[allow(dead_code)]
    chunk_duration: Duration,

    /// In-memory chunk cache
    chunk_cache: Arc<RwLock<lru::LruCache<u64, TimeChunk>>>,

    /// Enable fsync for writes
    fsync_enabled: bool,
}

impl ColumnarStore {
    /// Create a new columnar store
    ///
    /// # Arguments
    ///
    /// - `base_path` - Directory for storing chunk files
    /// - `chunk_duration` - Duration of each chunk (e.g., 2 hours)
    /// - `cache_size` - Number of chunks to keep in memory
    pub fn new<P: AsRef<Path>>(
        base_path: P,
        chunk_duration: Duration,
        cache_size: usize,
    ) -> TsdbResult<Self> {
        let base_path = base_path.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        create_dir_all(&base_path)?;

        // Create index file path
        let index_path = base_path.join("index.json");
        let index = Arc::new(SeriesIndex::with_file(index_path)?);

        Ok(Self {
            base_path,
            index,
            chunk_duration,
            chunk_cache: Arc::new(RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(cache_size).unwrap(),
            ))),
            fsync_enabled: true,
        })
    }

    /// Set fsync behavior (disable for better performance in testing)
    pub fn set_fsync(&mut self, enabled: bool) {
        self.fsync_enabled = enabled;
    }

    /// Write a chunk to disk
    ///
    /// Creates a binary file with the following format:
    /// - Header (32 bytes):
    ///   - Magic bytes: "TSDB" (4 bytes)
    ///   - Version: u32 (4 bytes)
    ///   - Series ID: u64 (8 bytes)
    ///   - Timestamp count: u32 (4 bytes)
    ///   - Timestamps size: u32 (4 bytes)
    ///   - Values size: u32 (4 bytes)
    ///   - Reserved: [u8; 4] (4 bytes)
    /// - Compressed timestamps (variable length)
    /// - Compressed values (variable length)
    /// - Metadata (serialized as JSON)
    pub fn write_chunk(&self, chunk: &TimeChunk) -> TsdbResult<ChunkEntry> {
        // Create chunk file path
        let chunk_id = chunk.start_time.timestamp();
        let file_path = self.chunk_file_path(chunk.series_id, chunk_id);

        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            create_dir_all(parent)?;
        }

        // Create temporary file for atomic write
        let temp_path = file_path.with_extension("tmp");
        let file = File::create(&temp_path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(b"TSDB")?; // Magic
        writer.write_all(&1u32.to_le_bytes())?; // Version
        writer.write_all(&chunk.series_id.to_le_bytes())?; // Series ID
        writer.write_all(&(chunk.metadata.count as u32).to_le_bytes())?; // Count
        writer.write_all(&(chunk.compressed_timestamps.len() as u32).to_le_bytes())?; // Timestamps size
        writer.write_all(&(chunk.compressed_values.len() as u32).to_le_bytes())?; // Values size
        writer.write_all(&[0u8; 4])?; // Reserved

        // Write compressed data
        writer.write_all(&chunk.compressed_timestamps)?;
        writer.write_all(&chunk.compressed_values)?;

        // Write metadata as JSON
        let metadata_json = serde_json::to_vec(&chunk.metadata)
            .map_err(|e| TsdbError::Config(format!("Failed to serialize metadata: {e}")))?;
        writer.write_all(&(metadata_json.len() as u32).to_le_bytes())?;
        writer.write_all(&metadata_json)?;

        // Flush and fsync if enabled
        writer.flush()?;
        let file = writer.into_inner().map_err(|e| TsdbError::Io(e.into()))?;
        if self.fsync_enabled {
            file.sync_all()?;
        }
        drop(file);

        // Atomic rename
        std::fs::rename(&temp_path, &file_path)?;

        // Create index entry
        let mut entry = ChunkEntry::new(
            chunk_id as u64,
            chunk.series_id,
            chunk.start_time,
            chunk.end_time,
            chunk.metadata.count,
        );
        entry.file_path = Some(file_path);
        entry.compressed_size = chunk.metadata.compressed_size;
        entry.uncompressed_size = chunk.metadata.uncompressed_size;

        // Update index
        let chunk_id = self.index.insert_chunk(entry.clone())?;
        entry.chunk_id = chunk_id;

        // Save index
        self.index.save()?;

        // Cache the chunk
        {
            let mut cache = self
                .chunk_cache
                .write()
                .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
            cache.put(chunk_id, chunk.clone());
        }

        Ok(entry)
    }

    /// Read a chunk from disk
    pub fn read_chunk(&self, chunk_id: u64) -> TsdbResult<TimeChunk> {
        // Check cache first
        {
            let mut cache = self
                .chunk_cache
                .write()
                .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
            if let Some(chunk) = cache.get(&chunk_id) {
                return Ok(chunk.clone());
            }
        }

        // Get chunk entry from index
        let entry = self
            .index
            .get_chunk(chunk_id)?
            .ok_or(TsdbError::ChunkNotFound {
                series_id: 0,
                timestamp: chunk_id as i64,
            })?;

        let file_path = entry
            .file_path
            .ok_or_else(|| TsdbError::Query("Chunk has no file path".to_string()))?;

        // Read file
        let data = std::fs::read(&file_path)?;

        // Parse header
        if data.len() < 32 {
            return Err(TsdbError::Decompression("Chunk file too small".to_string()));
        }

        let magic = &data[0..4];
        if magic != b"TSDB" {
            return Err(TsdbError::Decompression("Invalid magic bytes".to_string()));
        }

        let series_id = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let count = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let timestamps_size = u32::from_le_bytes(data[20..24].try_into().unwrap()) as usize;
        let values_size = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;

        // Extract compressed data
        let mut offset = 32;
        let compressed_timestamps = data[offset..offset + timestamps_size].to_vec();
        offset += timestamps_size;
        let compressed_values = data[offset..offset + values_size].to_vec();
        offset += values_size;

        // Parse metadata
        let metadata_size =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let metadata_json = &data[offset..offset + metadata_size];
        let metadata = serde_json::from_slice(metadata_json)
            .map_err(|e| TsdbError::Config(format!("Failed to deserialize metadata: {e}")))?;

        let chunk = TimeChunk {
            series_id,
            start_time: entry.start_time,
            end_time: entry.end_time,
            compressed_timestamps,
            compressed_values,
            metadata,
        };

        // Verify count matches
        if chunk.metadata.count != count {
            return Err(TsdbError::Decompression(format!(
                "Metadata count mismatch: header={count}, metadata={}",
                chunk.metadata.count
            )));
        }

        // Cache the chunk
        {
            let mut cache = self
                .chunk_cache
                .write()
                .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
            cache.put(chunk_id, chunk.clone());
        }

        Ok(chunk)
    }

    /// Query data points within a time range
    pub fn query_range(
        &self,
        series_id: u64,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> TsdbResult<Vec<DataPoint>> {
        // Find relevant chunks
        let chunks = self.index.get_chunks_in_range(series_id, start, end)?;

        let mut points = Vec::new();

        // Read and decompress each chunk
        for entry in chunks {
            let chunk = self.read_chunk(entry.chunk_id)?;
            let chunk_points = chunk.query_range(start, end)?;
            points.extend(chunk_points);
        }

        // Sort by timestamp
        points.sort_by_key(|p| p.timestamp);

        Ok(points)
    }

    /// Get chunk file path
    fn chunk_file_path(&self, series_id: u64, chunk_id: i64) -> PathBuf {
        // Organize chunks into subdirectories by series
        self.base_path
            .join(format!("series_{series_id}"))
            .join(format!("chunk_{chunk_id}.bin"))
    }

    /// Get series index (for advanced operations)
    pub fn index(&self) -> &SeriesIndex {
        &self.index
    }

    /// Flush cache to disk
    pub fn flush_cache(&self) -> TsdbResult<()> {
        let mut cache = self
            .chunk_cache
            .write()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
        cache.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn test_columnar_store_creation() -> TsdbResult<()> {
        let temp_dir = env::temp_dir().join("tsdb_columnar_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let _store = ColumnarStore::new(&temp_dir, Duration::hours(2), 100)?;
        assert!(temp_dir.exists());

        std::fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }

    #[test]
    fn test_write_and_read_chunk() -> TsdbResult<()> {
        let temp_dir = env::temp_dir().join("tsdb_columnar_write_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut store = ColumnarStore::new(&temp_dir, Duration::hours(2), 100)?;
        store.set_fsync(false); // Faster for tests

        let chunk = create_test_chunk(100, 1000, 50)?;
        let entry = store.write_chunk(&chunk)?;

        assert!(entry.file_path.is_some());
        assert_eq!(entry.series_id, 100);

        let read_chunk = store.read_chunk(entry.chunk_id)?;
        assert_eq!(read_chunk.series_id, 100);
        assert_eq!(read_chunk.metadata.count, 50);

        std::fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }

    #[test]
    fn test_query_range() -> TsdbResult<()> {
        let temp_dir = env::temp_dir().join("tsdb_columnar_query_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut store = ColumnarStore::new(&temp_dir, Duration::hours(2), 100)?;
        store.set_fsync(false);

        // Write two chunks with non-overlapping time ranges
        let chunk1 = create_test_chunk(100, 1000, 50)?;
        let chunk2 = create_test_chunk(100, 2000, 50)?;

        store.write_chunk(&chunk1)?;
        store.write_chunk(&chunk2)?;

        // Query across both chunks
        let points = store.query_range(
            100,
            DateTime::from_timestamp(1000, 0).unwrap(),
            DateTime::from_timestamp(2100, 0).unwrap(),
        )?;

        // Should include points from both chunks (50 from chunk1, some from chunk2)
        assert!(
            points.len() >= 50,
            "Expected at least 50 points, got {}",
            points.len()
        );

        std::fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }

    #[test]
    fn test_chunk_cache() -> TsdbResult<()> {
        let temp_dir = env::temp_dir().join("tsdb_columnar_cache_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut store = ColumnarStore::new(&temp_dir, Duration::hours(2), 100)?;
        store.set_fsync(false);

        let chunk = create_test_chunk(100, 1000, 50)?;
        let entry = store.write_chunk(&chunk)?;

        // First read (from cache)
        let read1 = store.read_chunk(entry.chunk_id)?;
        assert_eq!(read1.metadata.count, 50);

        // Second read (should also be from cache)
        let read2 = store.read_chunk(entry.chunk_id)?;
        assert_eq!(read2.metadata.count, 50);

        // Flush cache
        store.flush_cache()?;

        // Third read (from disk)
        let read3 = store.read_chunk(entry.chunk_id)?;
        assert_eq!(read3.metadata.count, 50);

        std::fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }

    #[test]
    fn test_multiple_series() -> TsdbResult<()> {
        let temp_dir = env::temp_dir().join("tsdb_columnar_multi_series_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut store = ColumnarStore::new(&temp_dir, Duration::hours(2), 100)?;
        store.set_fsync(false);

        // Write chunks for different series
        let chunk1 = create_test_chunk(100, 1000, 50)?;
        let chunk2 = create_test_chunk(200, 1000, 50)?;
        let chunk3 = create_test_chunk(300, 1000, 50)?;

        store.write_chunk(&chunk1)?;
        store.write_chunk(&chunk2)?;
        store.write_chunk(&chunk3)?;

        // Query each series
        let points1 = store.query_range(
            100,
            DateTime::from_timestamp(0, 0).unwrap(),
            DateTime::from_timestamp(2000, 0).unwrap(),
        )?;
        let points2 = store.query_range(
            200,
            DateTime::from_timestamp(0, 0).unwrap(),
            DateTime::from_timestamp(2000, 0).unwrap(),
        )?;
        let points3 = store.query_range(
            300,
            DateTime::from_timestamp(0, 0).unwrap(),
            DateTime::from_timestamp(2000, 0).unwrap(),
        )?;

        assert_eq!(points1.len(), 50);
        assert_eq!(points2.len(), 50);
        assert_eq!(points3.len(), 50);

        std::fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }
}
