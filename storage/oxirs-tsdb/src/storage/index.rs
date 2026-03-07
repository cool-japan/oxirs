//! Series indexing for efficient time-series lookups
//!
//! This module provides indexing structures for mapping series IDs to chunks
//! and performing efficient time-based lookups.

use crate::error::{TsdbError, TsdbResult};
use crate::storage::TimeChunk;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Chunk map type: timestamp -> ChunkEntry
type ChunkMap = BTreeMap<DateTime<Utc>, ChunkEntry>;

/// Series map type: series_id -> ChunkMap
type SeriesMap = HashMap<u64, ChunkMap>;

/// Index entry for a single chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkEntry {
    /// Chunk ID (unique identifier)
    pub chunk_id: u64,
    /// Series ID this chunk belongs to
    pub series_id: u64,
    /// Start timestamp of the chunk
    pub start_time: DateTime<Utc>,
    /// End timestamp of the chunk
    pub end_time: DateTime<Utc>,
    /// Number of data points in the chunk
    pub point_count: usize,
    /// File path (for disk-backed chunks)
    pub file_path: Option<PathBuf>,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Uncompressed size in bytes
    pub uncompressed_size: usize,
}

impl ChunkEntry {
    /// Create a new chunk entry
    pub fn new(
        chunk_id: u64,
        series_id: u64,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        point_count: usize,
    ) -> Self {
        Self {
            chunk_id,
            series_id,
            start_time,
            end_time,
            point_count,
            file_path: None,
            compressed_size: 0,
            uncompressed_size: 0,
        }
    }

    /// Check if a timestamp falls within this chunk
    pub fn contains(&self, timestamp: DateTime<Utc>) -> bool {
        timestamp >= self.start_time && timestamp <= self.end_time
    }

    /// Check if a time range overlaps with this chunk
    pub fn overlaps(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> bool {
        !(end < self.start_time || start > self.end_time)
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_size > 0 {
            self.uncompressed_size as f64 / self.compressed_size as f64
        } else {
            1.0
        }
    }
}

/// Series index mapping series IDs to chunks
///
/// Provides efficient lookups for:
/// - All chunks for a given series
/// - Chunks overlapping a time range
/// - Series metadata
#[derive(Debug)]
pub struct SeriesIndex {
    /// Mapping: series_id → list of chunk entries (sorted by time)
    series_to_chunks: Arc<RwLock<SeriesMap>>,

    /// Mapping: chunk_id → chunk entry
    chunk_by_id: Arc<RwLock<HashMap<u64, ChunkEntry>>>,

    /// Next chunk ID to assign
    next_chunk_id: Arc<RwLock<u64>>,

    /// Index file path (for persistence)
    index_file: Option<PathBuf>,
}

impl SeriesIndex {
    /// Create a new in-memory series index
    pub fn new() -> Self {
        Self {
            series_to_chunks: Arc::new(RwLock::new(HashMap::new())),
            chunk_by_id: Arc::new(RwLock::new(HashMap::new())),
            next_chunk_id: Arc::new(RwLock::new(1)),
            index_file: None,
        }
    }

    /// Create a series index with persistence
    pub fn with_file<P: AsRef<Path>>(path: P) -> TsdbResult<Self> {
        let index_file = path.as_ref().to_path_buf();
        let mut index = Self {
            series_to_chunks: Arc::new(RwLock::new(HashMap::new())),
            chunk_by_id: Arc::new(RwLock::new(HashMap::new())),
            next_chunk_id: Arc::new(RwLock::new(1)),
            index_file: Some(index_file.clone()),
        };

        // Load existing index if file exists
        if index_file.exists() {
            index.load()?;
        }

        Ok(index)
    }

    /// Insert a chunk into the index
    pub fn insert_chunk(&self, mut entry: ChunkEntry) -> TsdbResult<u64> {
        // Assign chunk ID if not set
        if entry.chunk_id == 0 {
            let mut next_id = self
                .next_chunk_id
                .write()
                .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
            entry.chunk_id = *next_id;
            *next_id += 1;
        }

        let chunk_id = entry.chunk_id;
        let series_id = entry.series_id;
        let start_time = entry.start_time;

        // Insert into chunk_by_id
        {
            let mut chunks = self
                .chunk_by_id
                .write()
                .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
            chunks.insert(chunk_id, entry.clone());
        }

        // Insert into series_to_chunks
        {
            let mut series_map = self
                .series_to_chunks
                .write()
                .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
            series_map
                .entry(series_id)
                .or_insert_with(BTreeMap::new)
                .insert(start_time, entry);
        }

        Ok(chunk_id)
    }

    /// Get all chunks for a series
    pub fn get_chunks_for_series(&self, series_id: u64) -> TsdbResult<Vec<ChunkEntry>> {
        let series_map = self
            .series_to_chunks
            .read()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;

        Ok(series_map
            .get(&series_id)
            .map(|chunks| chunks.values().cloned().collect())
            .unwrap_or_default())
    }

    /// Get chunks for a series within a time range
    pub fn get_chunks_in_range(
        &self,
        series_id: u64,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> TsdbResult<Vec<ChunkEntry>> {
        let series_map = self
            .series_to_chunks
            .read()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;

        if let Some(chunks) = series_map.get(&series_id) {
            Ok(chunks
                .values()
                .filter(|chunk| chunk.overlaps(start, end))
                .cloned()
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Get a specific chunk by ID
    pub fn get_chunk(&self, chunk_id: u64) -> TsdbResult<Option<ChunkEntry>> {
        let chunks = self
            .chunk_by_id
            .read()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
        Ok(chunks.get(&chunk_id).cloned())
    }

    /// Remove a chunk from the index
    pub fn remove_chunk(&self, chunk_id: u64) -> TsdbResult<bool> {
        // Get the chunk entry first
        let entry = {
            let chunks = self
                .chunk_by_id
                .read()
                .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
            chunks.get(&chunk_id).cloned()
        };

        if let Some(entry) = entry {
            // Remove from chunk_by_id
            {
                let mut chunks = self
                    .chunk_by_id
                    .write()
                    .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
                chunks.remove(&chunk_id);
            }

            // Remove from series_to_chunks
            {
                let mut series_map = self
                    .series_to_chunks
                    .write()
                    .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
                if let Some(chunks) = series_map.get_mut(&entry.series_id) {
                    chunks.remove(&entry.start_time);
                    if chunks.is_empty() {
                        series_map.remove(&entry.series_id);
                    }
                }
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get the number of chunks in the index
    pub fn chunk_count(&self) -> TsdbResult<usize> {
        let chunks = self
            .chunk_by_id
            .read()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
        Ok(chunks.len())
    }

    /// Get the number of series in the index
    pub fn series_count(&self) -> TsdbResult<usize> {
        let series_map = self
            .series_to_chunks
            .read()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
        Ok(series_map.len())
    }

    /// Get all series IDs
    pub fn series_ids(&self) -> TsdbResult<Vec<u64>> {
        let series_map = self
            .series_to_chunks
            .read()
            .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
        Ok(series_map.keys().copied().collect())
    }

    /// Save index to disk
    pub fn save(&self) -> TsdbResult<()> {
        if let Some(path) = &self.index_file {
            let chunks = self
                .chunk_by_id
                .read()
                .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;

            let entries: Vec<&ChunkEntry> = chunks.values().collect();
            let json = serde_json::to_string_pretty(&entries)
                .map_err(|e| TsdbError::Config(format!("Failed to serialize index: {e}")))?;

            std::fs::write(path, json)?;
        }
        Ok(())
    }

    /// Load index from disk
    pub fn load(&mut self) -> TsdbResult<()> {
        if let Some(path) = &self.index_file {
            let json = std::fs::read_to_string(path)?;

            let entries: Vec<ChunkEntry> = serde_json::from_str(&json)
                .map_err(|e| TsdbError::Config(format!("Failed to deserialize index: {e}")))?;

            // Clear existing data
            {
                let mut chunks = self
                    .chunk_by_id
                    .write()
                    .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
                chunks.clear();
            }
            {
                let mut series_map = self
                    .series_to_chunks
                    .write()
                    .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
                series_map.clear();
            }

            // Rebuild index
            let mut max_chunk_id = 0;
            for entry in entries {
                max_chunk_id = max_chunk_id.max(entry.chunk_id);
                self.insert_chunk(entry)?;
            }

            // Update next_chunk_id
            {
                let mut next_id = self
                    .next_chunk_id
                    .write()
                    .map_err(|e| TsdbError::Query(format!("Lock poisoned: {e}")))?;
                *next_id = max_chunk_id + 1;
            }
        }
        Ok(())
    }

    /// Rebuild index from chunks (used during recovery)
    pub fn rebuild_from_chunks(&self, chunks: &[TimeChunk]) -> TsdbResult<()> {
        for chunk in chunks {
            let mut entry = ChunkEntry::new(
                0, // Will be assigned
                chunk.series_id,
                chunk.start_time,
                chunk.end_time,
                chunk.metadata.count,
            );
            entry.compressed_size = chunk.metadata.compressed_size;
            entry.uncompressed_size = chunk.metadata.uncompressed_size;
            self.insert_chunk(entry)?;
        }
        Ok(())
    }
}

impl Default for SeriesIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_chunk_entry_contains() {
        let start = DateTime::from_timestamp(1000, 0).expect("valid timestamp");
        let end = DateTime::from_timestamp(2000, 0).expect("valid timestamp");
        let entry = ChunkEntry::new(1, 100, start, end, 50);

        assert!(entry.contains(DateTime::from_timestamp(1500, 0).expect("valid timestamp")));
        assert!(entry.contains(start));
        assert!(entry.contains(end));
        assert!(!entry.contains(DateTime::from_timestamp(500, 0).expect("valid timestamp")));
        assert!(!entry.contains(DateTime::from_timestamp(2500, 0).expect("valid timestamp")));
    }

    #[test]
    fn test_chunk_entry_overlaps() {
        let start = DateTime::from_timestamp(1000, 0).expect("valid timestamp");
        let end = DateTime::from_timestamp(2000, 0).expect("valid timestamp");
        let entry = ChunkEntry::new(1, 100, start, end, 50);

        // Completely overlaps
        assert!(entry.overlaps(
            DateTime::from_timestamp(1200, 0).expect("valid timestamp"),
            DateTime::from_timestamp(1800, 0).expect("valid timestamp")
        ));

        // Partial overlap (left)
        assert!(entry.overlaps(
            DateTime::from_timestamp(500, 0).expect("valid timestamp"),
            DateTime::from_timestamp(1500, 0).expect("valid timestamp")
        ));

        // Partial overlap (right)
        assert!(entry.overlaps(
            DateTime::from_timestamp(1500, 0).expect("valid timestamp"),
            DateTime::from_timestamp(2500, 0).expect("valid timestamp")
        ));

        // No overlap (before)
        assert!(!entry.overlaps(
            DateTime::from_timestamp(0, 0).expect("valid timestamp"),
            DateTime::from_timestamp(500, 0).expect("valid timestamp")
        ));

        // No overlap (after)
        assert!(!entry.overlaps(
            DateTime::from_timestamp(2500, 0).expect("valid timestamp"),
            DateTime::from_timestamp(3000, 0).expect("valid timestamp")
        ));
    }

    #[test]
    fn test_chunk_entry_compression_ratio() {
        let mut entry = ChunkEntry::new(
            1,
            100,
            DateTime::from_timestamp(1000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(2000, 0).expect("valid timestamp"),
            50,
        );

        entry.uncompressed_size = 8000;
        entry.compressed_size = 200;

        assert_eq!(entry.compression_ratio(), 40.0);
    }

    #[test]
    fn test_series_index_insert_and_get() -> TsdbResult<()> {
        let index = SeriesIndex::new();

        let entry1 = ChunkEntry::new(
            0,
            100,
            DateTime::from_timestamp(1000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(2000, 0).expect("valid timestamp"),
            50,
        );

        let chunk_id = index.insert_chunk(entry1)?;
        assert_eq!(chunk_id, 1);

        let retrieved = index.get_chunk(chunk_id)?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.expect("operation should succeed").series_id, 100);

        Ok(())
    }

    #[test]
    fn test_series_index_get_chunks_for_series() -> TsdbResult<()> {
        let index = SeriesIndex::new();

        let entry1 = ChunkEntry::new(
            0,
            100,
            DateTime::from_timestamp(1000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(2000, 0).expect("valid timestamp"),
            50,
        );
        let entry2 = ChunkEntry::new(
            0,
            100,
            DateTime::from_timestamp(3000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(4000, 0).expect("valid timestamp"),
            50,
        );
        let entry3 = ChunkEntry::new(
            0,
            200,
            DateTime::from_timestamp(1000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(2000, 0).expect("valid timestamp"),
            50,
        );

        index.insert_chunk(entry1)?;
        index.insert_chunk(entry2)?;
        index.insert_chunk(entry3)?;

        let chunks = index.get_chunks_for_series(100)?;
        assert_eq!(chunks.len(), 2);

        let chunks = index.get_chunks_for_series(200)?;
        assert_eq!(chunks.len(), 1);

        let chunks = index.get_chunks_for_series(999)?;
        assert_eq!(chunks.len(), 0);

        Ok(())
    }

    #[test]
    fn test_series_index_get_chunks_in_range() -> TsdbResult<()> {
        let index = SeriesIndex::new();

        let entry1 = ChunkEntry::new(
            0,
            100,
            DateTime::from_timestamp(1000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(2000, 0).expect("valid timestamp"),
            50,
        );
        let entry2 = ChunkEntry::new(
            0,
            100,
            DateTime::from_timestamp(3000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(4000, 0).expect("valid timestamp"),
            50,
        );
        let entry3 = ChunkEntry::new(
            0,
            100,
            DateTime::from_timestamp(5000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(6000, 0).expect("valid timestamp"),
            50,
        );

        index.insert_chunk(entry1)?;
        index.insert_chunk(entry2)?;
        index.insert_chunk(entry3)?;

        // Query range that overlaps first two chunks
        let chunks = index.get_chunks_in_range(
            100,
            DateTime::from_timestamp(1500, 0).expect("valid timestamp"),
            DateTime::from_timestamp(3500, 0).expect("valid timestamp"),
        )?;
        assert_eq!(chunks.len(), 2);

        // Query range that overlaps all chunks
        let chunks = index.get_chunks_in_range(
            100,
            DateTime::from_timestamp(0, 0).expect("valid timestamp"),
            DateTime::from_timestamp(10000, 0).expect("valid timestamp"),
        )?;
        assert_eq!(chunks.len(), 3);

        // Query range with no overlap
        let chunks = index.get_chunks_in_range(
            100,
            DateTime::from_timestamp(7000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(8000, 0).expect("valid timestamp"),
        )?;
        assert_eq!(chunks.len(), 0);

        Ok(())
    }

    #[test]
    fn test_series_index_remove_chunk() -> TsdbResult<()> {
        let index = SeriesIndex::new();

        let entry = ChunkEntry::new(
            0,
            100,
            DateTime::from_timestamp(1000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(2000, 0).expect("valid timestamp"),
            50,
        );

        let chunk_id = index.insert_chunk(entry)?;
        assert_eq!(index.chunk_count()?, 1);

        let removed = index.remove_chunk(chunk_id)?;
        assert!(removed);
        assert_eq!(index.chunk_count()?, 0);

        let removed_again = index.remove_chunk(chunk_id)?;
        assert!(!removed_again);

        Ok(())
    }

    #[test]
    fn test_series_index_counts() -> TsdbResult<()> {
        let index = SeriesIndex::new();

        assert_eq!(index.chunk_count()?, 0);
        assert_eq!(index.series_count()?, 0);

        let entry1 = ChunkEntry::new(
            0,
            100,
            DateTime::from_timestamp(1000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(2000, 0).expect("valid timestamp"),
            50,
        );
        let entry2 = ChunkEntry::new(
            0,
            200,
            DateTime::from_timestamp(1000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(2000, 0).expect("valid timestamp"),
            50,
        );

        index.insert_chunk(entry1)?;
        index.insert_chunk(entry2)?;

        assert_eq!(index.chunk_count()?, 2);
        assert_eq!(index.series_count()?, 2);

        Ok(())
    }

    #[test]
    fn test_series_index_persistence() -> TsdbResult<()> {
        let temp_file = env::temp_dir().join("tsdb_index_test.json");

        // Create index and save
        {
            let index = SeriesIndex::with_file(&temp_file)?;
            let entry = ChunkEntry::new(
                0,
                100,
                DateTime::from_timestamp(1000, 0).expect("valid timestamp"),
                DateTime::from_timestamp(2000, 0).expect("valid timestamp"),
                50,
            );
            index.insert_chunk(entry)?;
            index.save()?;
        }

        // Load index and verify
        {
            let index = SeriesIndex::with_file(&temp_file)?;
            assert_eq!(index.chunk_count()?, 1);
            assert_eq!(index.series_count()?, 1);
        }

        // Cleanup
        let _ = std::fs::remove_file(temp_file);

        Ok(())
    }

    #[test]
    fn test_series_ids() -> TsdbResult<()> {
        let index = SeriesIndex::new();

        let entry1 = ChunkEntry::new(
            0,
            100,
            DateTime::from_timestamp(1000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(2000, 0).expect("valid timestamp"),
            50,
        );
        let entry2 = ChunkEntry::new(
            0,
            200,
            DateTime::from_timestamp(1000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(2000, 0).expect("valid timestamp"),
            50,
        );
        let entry3 = ChunkEntry::new(
            0,
            300,
            DateTime::from_timestamp(1000, 0).expect("valid timestamp"),
            DateTime::from_timestamp(2000, 0).expect("valid timestamp"),
            50,
        );

        index.insert_chunk(entry1)?;
        index.insert_chunk(entry2)?;
        index.insert_chunk(entry3)?;

        let mut ids = index.series_ids()?;
        ids.sort();

        assert_eq!(ids, vec![100, 200, 300]);

        Ok(())
    }
}
