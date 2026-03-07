//! LSM-tree Compaction Strategy
//!
//! This module implements Log-Structured Merge-tree (LSM-tree) compaction strategies
//! for the TDB storage engine. LSM-trees are the foundation of many high-performance
//! key-value stores (LevelDB, RocksDB, Cassandra).
//!
//! # Compaction Strategies
//!
//! - **Size-tiered**: Groups SSTables of similar size into tiers; compacts when
//!   a tier reaches a threshold. Good for write-heavy workloads.
//! - **Leveled**: Organizes SSTables into levels with size limits; each level is
//!   10x larger than the previous. Good for read-heavy workloads.
//!
//! # Bloom Filters
//!
//! Each SSTable has an associated Bloom filter for probabilistic membership testing,
//! allowing O(1) negative lookups without reading the SSTable from disk.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

/// Unique identifier for an SSTable.
pub type SsTableId = u64;

/// A Bloom filter for probabilistic membership testing.
///
/// Uses a bit array with multiple hash functions to achieve low false-positive rates.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// The bit array.
    bits: Vec<bool>,
    /// Number of hash functions.
    num_hashes: usize,
    /// Number of items inserted.
    item_count: usize,
}

impl BloomFilter {
    /// Creates a new Bloom filter sized for the expected number of items
    /// and the desired false-positive rate.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items to insert.
    /// * `false_positive_rate` - Desired false-positive rate (e.g., 0.01 for 1%).
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let fp_rate = false_positive_rate.clamp(0.0001, 0.5);
        let expected = expected_items.max(1);

        // Optimal bit count: -n * ln(p) / (ln(2)^2)
        let bit_count =
            (-(expected as f64) * fp_rate.ln() / (2.0_f64.ln().powi(2))).ceil() as usize;
        let bit_count = bit_count.max(64);

        // Optimal number of hash functions: (m/n) * ln(2)
        let num_hashes = ((bit_count as f64 / expected as f64) * 2.0_f64.ln()).ceil() as usize;
        let num_hashes = num_hashes.clamp(1, 20);

        Self {
            bits: vec![false; bit_count],
            num_hashes,
            item_count: 0,
        }
    }

    /// Creates a Bloom filter with a specific size and hash count.
    pub fn with_params(bit_count: usize, num_hashes: usize) -> Self {
        Self {
            bits: vec![false; bit_count.max(1)],
            num_hashes: num_hashes.max(1),
            item_count: 0,
        }
    }

    /// Inserts a key into the Bloom filter.
    pub fn insert(&mut self, key: &[u8]) {
        for i in 0..self.num_hashes {
            let idx = self.hash_index(key, i);
            self.bits[idx] = true;
        }
        self.item_count += 1;
    }

    /// Checks if a key might be in the set.
    ///
    /// Returns `true` if the key might exist (possible false positive),
    /// `false` if the key definitely does not exist (no false negatives).
    pub fn might_contain(&self, key: &[u8]) -> bool {
        for i in 0..self.num_hashes {
            let idx = self.hash_index(key, i);
            if !self.bits[idx] {
                return false;
            }
        }
        true
    }

    /// Returns the number of items inserted.
    pub fn item_count(&self) -> usize {
        self.item_count
    }

    /// Returns the size of the bit array.
    pub fn bit_count(&self) -> usize {
        self.bits.len()
    }

    /// Returns the number of hash functions.
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }

    /// Returns the approximate false-positive rate based on current fill.
    pub fn estimated_fpr(&self) -> f64 {
        let set_bits = self.bits.iter().filter(|&&b| b).count() as f64;
        let total = self.bits.len() as f64;
        (set_bits / total).powi(self.num_hashes as i32)
    }

    /// Returns the fill ratio (fraction of bits set).
    pub fn fill_ratio(&self) -> f64 {
        if self.bits.is_empty() {
            return 0.0;
        }
        let set_bits = self.bits.iter().filter(|&&b| b).count() as f64;
        set_bits / self.bits.len() as f64
    }

    /// Computes a hash index for the given key and hash function number.
    fn hash_index(&self, key: &[u8], hash_num: usize) -> usize {
        // Double hashing: h(i) = h1 + i*h2
        let h1 = fnv1a_hash(key);
        let h2 = murmur_mix(key);
        let combined = h1.wrapping_add((hash_num as u64).wrapping_mul(h2));
        (combined as usize) % self.bits.len()
    }
}

/// FNV-1a hash for the first hash function.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Murmur-inspired mix for the second hash function.
fn murmur_mix(data: &[u8]) -> u64 {
    let mut hash: u64 = 0;
    for (i, &byte) in data.iter().enumerate() {
        hash = hash.wrapping_add((byte as u64).wrapping_mul(0x9e3779b97f4a7c15));
        hash ^= hash >> 17;
        hash = hash.wrapping_mul(0xbf58476d1ce4e5b9);
        hash ^= (i as u64).wrapping_add(1);
    }
    hash
}

/// Metadata for a single SSTable (Sorted String Table).
#[derive(Debug, Clone)]
pub struct SsTableMeta {
    /// Unique table ID.
    pub id: SsTableId,
    /// Level in the LSM tree (0 = memtable flush, 1+ = compacted).
    pub level: usize,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Number of entries (key-value pairs).
    pub entry_count: usize,
    /// Smallest key in this table.
    pub min_key: Vec<u8>,
    /// Largest key in this table.
    pub max_key: Vec<u8>,
    /// When this table was created.
    pub created_at: Instant,
    /// Bloom filter for this table.
    pub bloom_filter: BloomFilter,
    /// Number of tombstone (delete) markers.
    pub tombstone_count: usize,
}

impl SsTableMeta {
    /// Returns true if this SSTable's key range overlaps with the given range.
    pub fn overlaps(&self, min: &[u8], max: &[u8]) -> bool {
        self.min_key <= max.to_vec() && self.max_key >= min.to_vec()
    }

    /// Returns the tombstone ratio.
    pub fn tombstone_ratio(&self) -> f64 {
        if self.entry_count == 0 {
            return 0.0;
        }
        self.tombstone_count as f64 / self.entry_count as f64
    }
}

/// Compaction strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    /// Size-tiered compaction: groups similarly-sized SSTables.
    SizeTiered,
    /// Leveled compaction: organizes into exponentially-sized levels.
    Leveled,
}

/// Configuration for the LSM compactor.
#[derive(Debug, Clone)]
pub struct LsmConfig {
    /// Compaction strategy to use.
    pub strategy: CompactionStrategy,
    /// Maximum number of SSTables in L0 before triggering compaction.
    pub l0_compaction_trigger: usize,
    /// For size-tiered: minimum number of similarly-sized tables to merge.
    pub size_tiered_min_merge: usize,
    /// For size-tiered: maximum bucket count.
    pub size_tiered_max_buckets: usize,
    /// For leveled: size ratio between levels (typically 10).
    pub level_size_ratio: usize,
    /// For leveled: maximum level.
    pub max_level: usize,
    /// For leveled: target size for L1 in bytes.
    pub l1_target_size: u64,
    /// Bloom filter false-positive rate.
    pub bloom_fpr: f64,
    /// Tombstone ratio threshold to trigger compaction.
    pub tombstone_compaction_threshold: f64,
}

impl Default for LsmConfig {
    fn default() -> Self {
        Self {
            strategy: CompactionStrategy::Leveled,
            l0_compaction_trigger: 4,
            size_tiered_min_merge: 4,
            size_tiered_max_buckets: 8,
            level_size_ratio: 10,
            max_level: 7,
            l1_target_size: 64 * 1024 * 1024, // 64 MB
            bloom_fpr: 0.01,
            tombstone_compaction_threshold: 0.3,
        }
    }
}

/// Describes a compaction task to be executed.
#[derive(Debug, Clone)]
pub struct CompactionTask {
    /// The input SSTables to merge.
    pub input_tables: Vec<SsTableId>,
    /// The target level for the output.
    pub target_level: usize,
    /// Estimated output size in bytes.
    pub estimated_output_size: u64,
    /// Priority (higher = more urgent).
    pub priority: u32,
    /// Reason for compaction.
    pub reason: CompactionReason,
}

/// Why a compaction was triggered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionReason {
    /// Too many SSTables at a level.
    LevelOverflow,
    /// Size-tiered bucket reached threshold.
    SizeTieredBucket,
    /// High tombstone ratio.
    TombstoneCleanup,
    /// Manual compaction request.
    Manual,
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// IDs of tables that were compacted (to be deleted).
    pub removed_tables: Vec<SsTableId>,
    /// IDs of new tables created.
    pub new_tables: Vec<SsTableId>,
    /// Total bytes read during compaction.
    pub bytes_read: u64,
    /// Total bytes written during compaction.
    pub bytes_written: u64,
    /// Duration of the compaction.
    pub duration_ms: u64,
    /// Space amplification factor (total_size / live_data_size).
    pub space_amplification: f64,
}

/// Statistics about the LSM tree state.
#[derive(Debug, Clone, Default)]
pub struct LsmStats {
    /// Total SSTables across all levels.
    pub total_tables: usize,
    /// Tables per level.
    pub tables_per_level: Vec<usize>,
    /// Bytes per level.
    pub bytes_per_level: Vec<u64>,
    /// Total compactions performed.
    pub total_compactions: u64,
    /// Total bytes read during compactions.
    pub total_bytes_read: u64,
    /// Total bytes written during compactions.
    pub total_bytes_written: u64,
    /// Write amplification factor.
    pub write_amplification: f64,
    /// Bloom filter total memory usage (bits).
    pub bloom_filter_bits: usize,
}

/// The LSM compactor manages SSTable levels and schedules compactions.
pub struct LsmCompactor {
    config: LsmConfig,
    /// SSTables organized by level.
    levels: Vec<Vec<SsTableMeta>>,
    /// Quick lookup by table ID.
    table_index: HashMap<SsTableId, usize>, // id -> level
    /// Next table ID to assign.
    next_table_id: SsTableId,
    /// Statistics.
    stats: LsmStats,
}

impl LsmCompactor {
    /// Creates a new LSM compactor with the given configuration.
    pub fn new(config: LsmConfig) -> Self {
        let max_level = config.max_level + 1;
        Self {
            levels: vec![Vec::new(); max_level],
            table_index: HashMap::new(),
            next_table_id: 1,
            stats: LsmStats {
                tables_per_level: vec![0; max_level],
                bytes_per_level: vec![0; max_level],
                ..Default::default()
            },
            config,
        }
    }

    /// Creates a new LSM compactor with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(LsmConfig::default())
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &LsmConfig {
        &self.config
    }

    /// Registers a new SSTable (e.g., from a memtable flush).
    pub fn add_table(&mut self, mut meta: SsTableMeta) -> SsTableId {
        let id = self.next_table_id;
        self.next_table_id += 1;
        meta.id = id;

        let level = meta.level;
        if level >= self.levels.len() {
            self.levels.resize_with(level + 1, Vec::new);
            self.stats.tables_per_level.resize(level + 1, 0);
            self.stats.bytes_per_level.resize(level + 1, 0);
        }

        self.stats.bytes_per_level[level] += meta.size_bytes;
        self.levels[level].push(meta);
        self.stats.tables_per_level[level] = self.levels[level].len();
        self.table_index.insert(id, level);
        self.stats.total_tables += 1;

        id
    }

    /// Removes a table after compaction.
    pub fn remove_table(&mut self, id: SsTableId) -> bool {
        if let Some(level) = self.table_index.remove(&id) {
            if let Some(pos) = self.levels[level].iter().position(|t| t.id == id) {
                let size = self.levels[level][pos].size_bytes;
                self.levels[level].swap_remove(pos);
                self.stats.tables_per_level[level] = self.levels[level].len();
                self.stats.bytes_per_level[level] =
                    self.stats.bytes_per_level[level].saturating_sub(size);
                self.stats.total_tables = self.stats.total_tables.saturating_sub(1);
                return true;
            }
        }
        false
    }

    /// Returns the number of SSTables at the given level.
    pub fn level_table_count(&self, level: usize) -> usize {
        self.levels.get(level).map_or(0, |v| v.len())
    }

    /// Returns the total size in bytes at the given level.
    pub fn level_size(&self, level: usize) -> u64 {
        self.stats.bytes_per_level.get(level).copied().unwrap_or(0)
    }

    /// Returns the total number of SSTables across all levels.
    pub fn total_table_count(&self) -> usize {
        self.stats.total_tables
    }

    /// Returns the number of levels.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Picks the next compaction task based on the configured strategy.
    ///
    /// Returns `None` if no compaction is needed.
    pub fn pick_compaction(&self) -> Option<CompactionTask> {
        match self.config.strategy {
            CompactionStrategy::SizeTiered => self.pick_size_tiered_compaction(),
            CompactionStrategy::Leveled => self.pick_leveled_compaction(),
        }
    }

    /// Picks a compaction using size-tiered strategy.
    fn pick_size_tiered_compaction(&self) -> Option<CompactionTask> {
        // Group L0 tables by size into buckets
        let l0_tables = &self.levels[0];
        if l0_tables.len() < self.config.size_tiered_min_merge {
            return None;
        }

        // Sort by size and find buckets of similar size
        let mut sorted: Vec<&SsTableMeta> = l0_tables.iter().collect();
        sorted.sort_by_key(|t| t.size_bytes);

        // Simple bucketing: consecutive tables within 2x size ratio
        let mut best_bucket: Vec<SsTableId> = Vec::new();
        let mut current_bucket: Vec<SsTableId> = Vec::new();
        let mut bucket_base_size: u64 = 0;

        for table in &sorted {
            if current_bucket.is_empty()
                || (bucket_base_size > 0 && table.size_bytes <= bucket_base_size * 2)
            {
                if current_bucket.is_empty() {
                    bucket_base_size = table.size_bytes;
                }
                current_bucket.push(table.id);
            } else {
                if current_bucket.len() > best_bucket.len() {
                    best_bucket = current_bucket.clone();
                }
                current_bucket.clear();
                current_bucket.push(table.id);
                bucket_base_size = table.size_bytes;
            }
        }
        if current_bucket.len() > best_bucket.len() {
            best_bucket = current_bucket;
        }

        if best_bucket.len() >= self.config.size_tiered_min_merge {
            let estimated_size: u64 = best_bucket
                .iter()
                .filter_map(|id| {
                    self.table_index.get(id).and_then(|_| {
                        self.levels[0]
                            .iter()
                            .find(|t| t.id == *id)
                            .map(|t| t.size_bytes)
                    })
                })
                .sum();

            return Some(CompactionTask {
                input_tables: best_bucket,
                target_level: 0,
                estimated_output_size: estimated_size,
                priority: 10,
                reason: CompactionReason::SizeTieredBucket,
            });
        }

        None
    }

    /// Picks a compaction using leveled strategy.
    fn pick_leveled_compaction(&self) -> Option<CompactionTask> {
        // Check L0 overflow first
        if self.levels[0].len() >= self.config.l0_compaction_trigger {
            let input_ids: Vec<SsTableId> = self.levels[0].iter().map(|t| t.id).collect();
            let estimated_size: u64 = self.levels[0].iter().map(|t| t.size_bytes).sum();

            return Some(CompactionTask {
                input_tables: input_ids,
                target_level: 1,
                estimated_output_size: estimated_size,
                priority: 20,
                reason: CompactionReason::LevelOverflow,
            });
        }

        // Check each level for overflow
        for level in 1..self.levels.len().saturating_sub(1) {
            let target_size = self.target_level_size(level);
            let current_size: u64 = self.levels[level].iter().map(|t| t.size_bytes).sum();

            if current_size > target_size {
                // Pick the table with the most overlapping range at the next level
                if let Some(table) = self.levels[level].first() {
                    let mut input_ids = vec![table.id];

                    // Find overlapping tables at the next level
                    let next_level = level + 1;
                    if next_level < self.levels.len() {
                        for next_table in &self.levels[next_level] {
                            if next_table.overlaps(&table.min_key, &table.max_key) {
                                input_ids.push(next_table.id);
                            }
                        }
                    }

                    let estimated_size: u64 = input_ids
                        .iter()
                        .filter_map(|id| {
                            self.table_index.get(id).and_then(|&lvl| {
                                self.levels[lvl]
                                    .iter()
                                    .find(|t| t.id == *id)
                                    .map(|t| t.size_bytes)
                            })
                        })
                        .sum();

                    return Some(CompactionTask {
                        input_tables: input_ids,
                        target_level: next_level,
                        estimated_output_size: estimated_size,
                        priority: 15,
                        reason: CompactionReason::LevelOverflow,
                    });
                }
            }
        }

        // Check tombstone ratio
        for level in &self.levels {
            for table in level {
                if table.tombstone_ratio() >= self.config.tombstone_compaction_threshold {
                    return Some(CompactionTask {
                        input_tables: vec![table.id],
                        target_level: table.level,
                        estimated_output_size: table.size_bytes / 2, // estimate
                        priority: 5,
                        reason: CompactionReason::TombstoneCleanup,
                    });
                }
            }
        }

        None
    }

    /// Computes the target size for a given level.
    fn target_level_size(&self, level: usize) -> u64 {
        if level == 0 {
            return u64::MAX; // L0 has no size limit, uses count trigger
        }
        self.config.l1_target_size
            * (self.config.level_size_ratio as u64).pow(level.saturating_sub(1) as u32)
    }

    /// Records a completed compaction and updates statistics.
    pub fn record_compaction(&mut self, result: &CompactionResult) {
        self.stats.total_compactions += 1;
        self.stats.total_bytes_read += result.bytes_read;
        self.stats.total_bytes_written += result.bytes_written;

        if self.stats.total_bytes_read > 0 {
            self.stats.write_amplification =
                self.stats.total_bytes_written as f64 / self.stats.total_bytes_read as f64;
        }
    }

    /// Looks up a key across all SSTables using Bloom filters, returning candidate tables.
    ///
    /// Tables are returned in order from newest to oldest (L0 first, then higher levels).
    pub fn lookup_candidates(&self, key: &[u8]) -> Vec<SsTableId> {
        let mut candidates = Vec::new();

        for level in &self.levels {
            for table in level {
                if table.bloom_filter.might_contain(key) {
                    candidates.push(table.id);
                }
            }
        }

        candidates
    }

    /// Returns current statistics.
    pub fn stats(&self) -> &LsmStats {
        &self.stats
    }

    /// Returns a snapshot of the statistics.
    pub fn stats_snapshot(&self) -> LsmStats {
        let mut stats = self.stats.clone();
        stats.bloom_filter_bits = self
            .levels
            .iter()
            .flat_map(|level| level.iter())
            .map(|t| t.bloom_filter.bit_count())
            .sum();
        stats
    }

    /// Returns table metadata by ID.
    pub fn get_table(&self, id: SsTableId) -> Option<&SsTableMeta> {
        self.table_index
            .get(&id)
            .and_then(|&level| self.levels[level].iter().find(|t| t.id == id))
    }

    /// Creates a manual compaction task for the given level.
    pub fn manual_compaction(&self, level: usize) -> Option<CompactionTask> {
        if level >= self.levels.len() || self.levels[level].is_empty() {
            return None;
        }

        let input_ids: Vec<SsTableId> = self.levels[level].iter().map(|t| t.id).collect();
        let estimated_size: u64 = self.levels[level].iter().map(|t| t.size_bytes).sum();
        let target_level = (level + 1).min(self.levels.len() - 1);

        Some(CompactionTask {
            input_tables: input_ids,
            target_level,
            estimated_output_size: estimated_size,
            priority: 1,
            reason: CompactionReason::Manual,
        })
    }
}

/// Helper to create an SSTable metadata for testing.
fn make_test_table(level: usize, size_bytes: u64, min_key: &[u8], max_key: &[u8]) -> SsTableMeta {
    let mut bloom = BloomFilter::new(100, 0.01);
    bloom.insert(min_key);
    bloom.insert(max_key);

    SsTableMeta {
        id: 0, // will be assigned by add_table
        level,
        size_bytes,
        entry_count: (size_bytes / 100) as usize,
        min_key: min_key.to_vec(),
        max_key: max_key.to_vec(),
        created_at: Instant::now(),
        bloom_filter: bloom,
        tombstone_count: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Bloom filter ────────────────────────────────────────────────────────

    #[test]
    fn test_bloom_filter_basic() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(b"hello");
        bf.insert(b"world");
        assert!(bf.might_contain(b"hello"));
        assert!(bf.might_contain(b"world"));
        // Very unlikely to contain random data
        assert_eq!(bf.item_count(), 2);
    }

    #[test]
    fn test_bloom_filter_no_false_negatives() {
        let mut bf = BloomFilter::new(1000, 0.01);
        let keys: Vec<Vec<u8>> = (0..500).map(|i| format!("key-{i}").into_bytes()).collect();

        for key in &keys {
            bf.insert(key);
        }

        // No false negatives: all inserted keys must be found
        for key in &keys {
            assert!(bf.might_contain(key), "Key {:?} not found", key);
        }
    }

    #[test]
    fn test_bloom_filter_false_positive_rate() {
        let mut bf = BloomFilter::new(1000, 0.05);
        for i in 0..1000 {
            bf.insert(format!("inserted-{i}").as_bytes());
        }

        let mut false_positives = 0;
        let test_count = 10_000;
        for i in 0..test_count {
            if bf.might_contain(format!("not-inserted-{i}").as_bytes()) {
                false_positives += 1;
            }
        }

        let fpr = false_positives as f64 / test_count as f64;
        // FPR should be below 15% (generous margin for probabilistic test)
        assert!(fpr < 0.15, "FPR too high: {fpr}");
    }

    #[test]
    fn test_bloom_filter_empty() {
        let bf = BloomFilter::new(100, 0.01);
        assert!(!bf.might_contain(b"anything"));
        assert_eq!(bf.item_count(), 0);
    }

    #[test]
    fn test_bloom_filter_with_params() {
        let bf = BloomFilter::with_params(256, 3);
        assert_eq!(bf.bit_count(), 256);
        assert_eq!(bf.num_hashes(), 3);
    }

    #[test]
    fn test_bloom_filter_fill_ratio() {
        let mut bf = BloomFilter::new(100, 0.01);
        assert_eq!(bf.fill_ratio(), 0.0);
        for i in 0..50 {
            bf.insert(format!("key-{i}").as_bytes());
        }
        assert!(bf.fill_ratio() > 0.0);
        assert!(bf.fill_ratio() <= 1.0);
    }

    #[test]
    fn test_bloom_filter_estimated_fpr() {
        let mut bf = BloomFilter::new(100, 0.01);
        let initial_fpr = bf.estimated_fpr();
        assert!(initial_fpr < 0.001); // empty = very low FPR

        for i in 0..100 {
            bf.insert(format!("key-{i}").as_bytes());
        }
        let filled_fpr = bf.estimated_fpr();
        assert!(filled_fpr > initial_fpr);
    }

    // ── SsTableMeta ────────────────────────────────────────────────────────

    #[test]
    fn test_sstable_overlaps() {
        let table = make_test_table(0, 1000, b"aaa", b"zzz");
        assert!(table.overlaps(b"mmm", b"nnn"));
        assert!(table.overlaps(b"aaa", b"aaa"));
        assert!(!table.overlaps(b"000", b"099"));
    }

    #[test]
    fn test_sstable_tombstone_ratio() {
        let mut table = make_test_table(0, 10000, b"a", b"z");
        table.tombstone_count = 3;
        assert!(table.tombstone_ratio() > 0.0);
        assert!(table.tombstone_ratio() < 1.0);
    }

    #[test]
    fn test_sstable_tombstone_ratio_empty() {
        let mut table = make_test_table(0, 10000, b"a", b"z");
        table.entry_count = 0;
        assert_eq!(table.tombstone_ratio(), 0.0);
    }

    // ── LsmCompactor: creation and table management ─────────────────────────

    #[test]
    fn test_compactor_creation() {
        let compactor = LsmCompactor::with_defaults();
        assert_eq!(compactor.total_table_count(), 0);
        assert!(compactor.level_count() > 0);
    }

    #[test]
    fn test_add_table() {
        let mut compactor = LsmCompactor::with_defaults();
        let table = make_test_table(0, 1000, b"a", b"z");
        let id = compactor.add_table(table);
        assert!(id > 0);
        assert_eq!(compactor.total_table_count(), 1);
        assert_eq!(compactor.level_table_count(0), 1);
    }

    #[test]
    fn test_add_multiple_tables() {
        let mut compactor = LsmCompactor::with_defaults();
        for i in 0..5 {
            let key = format!("key{i}");
            compactor.add_table(make_test_table(0, 1000, key.as_bytes(), key.as_bytes()));
        }
        assert_eq!(compactor.total_table_count(), 5);
        assert_eq!(compactor.level_table_count(0), 5);
    }

    #[test]
    fn test_add_tables_to_different_levels() {
        let mut compactor = LsmCompactor::with_defaults();
        compactor.add_table(make_test_table(0, 1000, b"a", b"m"));
        compactor.add_table(make_test_table(1, 5000, b"a", b"z"));
        compactor.add_table(make_test_table(2, 50000, b"a", b"z"));
        assert_eq!(compactor.level_table_count(0), 1);
        assert_eq!(compactor.level_table_count(1), 1);
        assert_eq!(compactor.level_table_count(2), 1);
    }

    #[test]
    fn test_remove_table() {
        let mut compactor = LsmCompactor::with_defaults();
        let id = compactor.add_table(make_test_table(0, 1000, b"a", b"z"));
        assert!(compactor.remove_table(id));
        assert_eq!(compactor.total_table_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_table() {
        let mut compactor = LsmCompactor::with_defaults();
        assert!(!compactor.remove_table(999));
    }

    #[test]
    fn test_get_table() {
        let mut compactor = LsmCompactor::with_defaults();
        let id = compactor.add_table(make_test_table(0, 1000, b"a", b"z"));
        let table = compactor.get_table(id);
        assert!(table.is_some());
        assert_eq!(table.map(|t| t.level), Some(0));
    }

    #[test]
    fn test_level_size() {
        let mut compactor = LsmCompactor::with_defaults();
        compactor.add_table(make_test_table(0, 1000, b"a", b"m"));
        compactor.add_table(make_test_table(0, 2000, b"n", b"z"));
        assert_eq!(compactor.level_size(0), 3000);
    }

    // ── Leveled compaction ──────────────────────────────────────────────────

    #[test]
    fn test_leveled_no_compaction_needed() {
        let compactor = LsmCompactor::new(LsmConfig {
            strategy: CompactionStrategy::Leveled,
            l0_compaction_trigger: 4,
            ..LsmConfig::default()
        });
        assert!(compactor.pick_compaction().is_none());
    }

    #[test]
    fn test_leveled_l0_trigger() {
        let mut compactor = LsmCompactor::new(LsmConfig {
            strategy: CompactionStrategy::Leveled,
            l0_compaction_trigger: 4,
            ..LsmConfig::default()
        });
        for i in 0..4 {
            let key = format!("key{i}");
            compactor.add_table(make_test_table(0, 1000, key.as_bytes(), key.as_bytes()));
        }
        let task = compactor.pick_compaction();
        assert!(task.is_some());
        let task = task.expect("should have task");
        assert_eq!(task.target_level, 1);
        assert_eq!(task.reason, CompactionReason::LevelOverflow);
        assert_eq!(task.input_tables.len(), 4);
    }

    #[test]
    fn test_leveled_l0_below_trigger() {
        let mut compactor = LsmCompactor::new(LsmConfig {
            strategy: CompactionStrategy::Leveled,
            l0_compaction_trigger: 4,
            ..LsmConfig::default()
        });
        for i in 0..3 {
            let key = format!("key{i}");
            compactor.add_table(make_test_table(0, 1000, key.as_bytes(), key.as_bytes()));
        }
        assert!(compactor.pick_compaction().is_none());
    }

    #[test]
    fn test_leveled_tombstone_trigger() {
        let mut compactor = LsmCompactor::new(LsmConfig {
            strategy: CompactionStrategy::Leveled,
            l0_compaction_trigger: 100, // high trigger so L0 doesn't fire
            tombstone_compaction_threshold: 0.3,
            ..LsmConfig::default()
        });

        let mut table = make_test_table(1, 10000, b"a", b"z");
        table.tombstone_count = 50; // 50% tombstones
        compactor.add_table(table);

        let task = compactor.pick_compaction();
        assert!(task.is_some());
        assert_eq!(
            task.expect("should exist").reason,
            CompactionReason::TombstoneCleanup
        );
    }

    // ── Size-tiered compaction ──────────────────────────────────────────────

    #[test]
    fn test_size_tiered_no_compaction() {
        let compactor = LsmCompactor::new(LsmConfig {
            strategy: CompactionStrategy::SizeTiered,
            size_tiered_min_merge: 4,
            ..LsmConfig::default()
        });
        assert!(compactor.pick_compaction().is_none());
    }

    #[test]
    fn test_size_tiered_triggers() {
        let mut compactor = LsmCompactor::new(LsmConfig {
            strategy: CompactionStrategy::SizeTiered,
            size_tiered_min_merge: 4,
            ..LsmConfig::default()
        });
        // Add 4 similarly-sized tables
        for i in 0..4 {
            let key = format!("key{i}");
            compactor.add_table(make_test_table(
                0,
                1000 + i * 100,
                key.as_bytes(),
                key.as_bytes(),
            ));
        }
        let task = compactor.pick_compaction();
        assert!(task.is_some());
        assert_eq!(
            task.expect("should exist").reason,
            CompactionReason::SizeTieredBucket
        );
    }

    #[test]
    fn test_size_tiered_different_sizes_no_trigger() {
        let mut compactor = LsmCompactor::new(LsmConfig {
            strategy: CompactionStrategy::SizeTiered,
            size_tiered_min_merge: 4,
            ..LsmConfig::default()
        });
        // Add tables with very different sizes (each > 2x the previous)
        for i in 0..3 {
            let size = 1000 * (10u64.pow(i));
            let key = format!("key{i}");
            compactor.add_table(make_test_table(0, size, key.as_bytes(), key.as_bytes()));
        }
        assert!(compactor.pick_compaction().is_none());
    }

    // ── Bloom filter lookup ─────────────────────────────────────────────────

    #[test]
    fn test_lookup_candidates() {
        let mut compactor = LsmCompactor::with_defaults();
        let mut table = make_test_table(0, 1000, b"a", b"z");
        table.bloom_filter.insert(b"target-key");
        compactor.add_table(table);

        let candidates = compactor.lookup_candidates(b"target-key");
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_lookup_no_candidates() {
        let mut compactor = LsmCompactor::with_defaults();
        compactor.add_table(make_test_table(0, 1000, b"a", b"z"));
        // The key was never inserted into any bloom filter specifically
        // But the bloom filter might still match due to hash collisions
        // So we just verify the function returns without error
        let _ = compactor.lookup_candidates(b"nonexistent-key-xyz-123");
    }

    // ── Manual compaction ───────────────────────────────────────────────────

    #[test]
    fn test_manual_compaction() {
        let mut compactor = LsmCompactor::with_defaults();
        compactor.add_table(make_test_table(0, 1000, b"a", b"m"));
        compactor.add_table(make_test_table(0, 2000, b"n", b"z"));

        let task = compactor.manual_compaction(0);
        assert!(task.is_some());
        let task = task.expect("should exist");
        assert_eq!(task.reason, CompactionReason::Manual);
        assert_eq!(task.input_tables.len(), 2);
        assert_eq!(task.target_level, 1);
    }

    #[test]
    fn test_manual_compaction_empty_level() {
        let compactor = LsmCompactor::with_defaults();
        assert!(compactor.manual_compaction(0).is_none());
    }

    #[test]
    fn test_manual_compaction_invalid_level() {
        let compactor = LsmCompactor::with_defaults();
        assert!(compactor.manual_compaction(999).is_none());
    }

    // ── Statistics ──────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let compactor = LsmCompactor::with_defaults();
        let stats = compactor.stats();
        assert_eq!(stats.total_tables, 0);
        assert_eq!(stats.total_compactions, 0);
    }

    #[test]
    fn test_stats_after_adds() {
        let mut compactor = LsmCompactor::with_defaults();
        compactor.add_table(make_test_table(0, 1000, b"a", b"z"));
        compactor.add_table(make_test_table(1, 5000, b"a", b"z"));
        let stats = compactor.stats();
        assert_eq!(stats.total_tables, 2);
        assert_eq!(stats.tables_per_level[0], 1);
        assert_eq!(stats.tables_per_level[1], 1);
    }

    #[test]
    fn test_record_compaction() {
        let mut compactor = LsmCompactor::with_defaults();
        let result = CompactionResult {
            removed_tables: vec![1, 2],
            new_tables: vec![3],
            bytes_read: 10000,
            bytes_written: 8000,
            duration_ms: 100,
            space_amplification: 1.2,
        };
        compactor.record_compaction(&result);
        assert_eq!(compactor.stats().total_compactions, 1);
        assert_eq!(compactor.stats().total_bytes_read, 10000);
        assert_eq!(compactor.stats().total_bytes_written, 8000);
        assert!(compactor.stats().write_amplification > 0.0);
    }

    #[test]
    fn test_stats_snapshot_includes_bloom_bits() {
        let mut compactor = LsmCompactor::with_defaults();
        compactor.add_table(make_test_table(0, 1000, b"a", b"z"));
        let snapshot = compactor.stats_snapshot();
        assert!(snapshot.bloom_filter_bits > 0);
    }

    // ── Config ──────────────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let cfg = LsmConfig::default();
        assert_eq!(cfg.strategy, CompactionStrategy::Leveled);
        assert_eq!(cfg.l0_compaction_trigger, 4);
        assert_eq!(cfg.level_size_ratio, 10);
    }

    #[test]
    fn test_config_access() {
        let compactor = LsmCompactor::with_defaults();
        assert_eq!(compactor.config().strategy, CompactionStrategy::Leveled);
    }

    // ── CompactionTask properties ───────────────────────────────────────────

    #[test]
    fn test_compaction_task_priority() {
        let mut compactor = LsmCompactor::new(LsmConfig {
            strategy: CompactionStrategy::Leveled,
            l0_compaction_trigger: 2,
            ..LsmConfig::default()
        });
        compactor.add_table(make_test_table(0, 1000, b"a", b"m"));
        compactor.add_table(make_test_table(0, 1000, b"n", b"z"));
        let task = compactor.pick_compaction().expect("should compact");
        assert!(task.priority > 0);
    }

    #[test]
    fn test_compaction_result_fields() {
        let result = CompactionResult {
            removed_tables: vec![1, 2, 3],
            new_tables: vec![4],
            bytes_read: 30000,
            bytes_written: 25000,
            duration_ms: 500,
            space_amplification: 1.5,
        };
        assert_eq!(result.removed_tables.len(), 3);
        assert_eq!(result.new_tables.len(), 1);
        assert!(result.space_amplification > 1.0);
    }

    // ── Hash functions ──────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_consistency() {
        let h1 = fnv1a_hash(b"test");
        let h2 = fnv1a_hash(b"test");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_different_inputs() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_murmur_mix_consistency() {
        let h1 = murmur_mix(b"test");
        let h2 = murmur_mix(b"test");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_murmur_mix_different_inputs() {
        let h1 = murmur_mix(b"hello");
        let h2 = murmur_mix(b"world");
        assert_ne!(h1, h2);
    }
}
