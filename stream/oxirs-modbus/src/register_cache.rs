//! # Register Block Caching with Change Detection
//!
//! Provides an in-memory cache for Modbus register blocks that detects value
//! changes between polls and emits change events. Reduces unnecessary
//! network traffic and RDF triple generation by only propagating changed values.
//!
//! ## Features
//!
//! - **Block-level caching**: Cache contiguous register blocks
//! - **Change detection**: Detect which registers changed between polls
//! - **Dead-band filtering**: Suppress small oscillations below threshold
//! - **TTL-based invalidation**: Automatically invalidate stale cache entries
//! - **Statistics**: Track cache hit rates, change rates, and bandwidth savings
//! - **History buffer**: Configurable ring buffer for recent values

use crate::error::ModbusError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for register block caching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterCacheConfig {
    /// Time-to-live for cached register values (default: 5s).
    pub ttl: Duration,
    /// Dead-band threshold (absolute) for change detection (default: 0).
    /// Values that change by less than this amount are not reported.
    pub dead_band: u16,
    /// Maximum number of cached register blocks (default: 256).
    pub max_blocks: usize,
    /// Number of historical values to keep per register (default: 10).
    pub history_depth: usize,
    /// Whether to track per-register statistics (default: true).
    pub track_statistics: bool,
}

impl Default for RegisterCacheConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(5),
            dead_band: 0,
            max_blocks: 256,
            history_depth: 10,
            track_statistics: true,
        }
    }
}

// ─────────────────────────────────────────────
// Register Block Key
// ─────────────────────────────────────────────

/// Identifies a contiguous block of registers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegisterBlockKey {
    /// Unit (slave) ID.
    pub unit_id: u8,
    /// Starting register address.
    pub start_address: u16,
    /// Number of registers in the block.
    pub count: u16,
}

impl RegisterBlockKey {
    /// Create a new register block key.
    pub fn new(unit_id: u8, start_address: u16, count: u16) -> Self {
        Self {
            unit_id,
            start_address,
            count,
        }
    }
}

// ─────────────────────────────────────────────
// Change Event
// ─────────────────────────────────────────────

/// Describes a change detected in a single register.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterChange {
    /// Register address.
    pub address: u16,
    /// Previous value.
    pub old_value: u16,
    /// New value.
    pub new_value: u16,
    /// Absolute change magnitude.
    pub delta: u16,
    /// When the change was detected.
    pub timestamp: DateTime<Utc>,
}

/// Describes changes in a register block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockChangeEvent {
    /// Which block changed.
    pub block_key: RegisterBlockKey,
    /// Individual register changes.
    pub changes: Vec<RegisterChange>,
    /// Whether any register in the block changed.
    pub has_changes: bool,
    /// Total number of registers in the block.
    pub total_registers: usize,
    /// Number of changed registers.
    pub changed_count: usize,
    /// Timestamp of this event.
    pub timestamp: DateTime<Utc>,
}

// ─────────────────────────────────────────────
// Cached Block
// ─────────────────────────────────────────────

/// Cached state for a register block.
#[derive(Debug, Clone)]
struct CachedBlock {
    /// Current register values.
    values: Vec<u16>,
    /// When the block was last updated.
    last_updated: Instant,
    /// UTC timestamp of last update.
    last_updated_utc: DateTime<Utc>,
    /// Per-register history (ring buffer).
    history: Vec<Vec<(DateTime<Utc>, u16)>>,
    /// Per-register change counter.
    change_counts: Vec<u64>,
    /// Total number of updates (polls).
    poll_count: u64,
}

// ─────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────

/// Statistics for the register cache.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegisterCacheStats {
    /// Total cache lookups.
    pub total_lookups: u64,
    /// Cache hits (valid, non-expired entry found).
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
    /// Cache hit rate.
    pub hit_rate: f64,
    /// Total register updates (polls).
    pub total_updates: u64,
    /// Total individual register changes detected.
    pub total_changes_detected: u64,
    /// Updates suppressed by dead-band filter.
    pub dead_band_filtered: u64,
    /// Number of cached blocks.
    pub cached_blocks: usize,
    /// Total registers cached.
    pub total_registers_cached: usize,
    /// Estimated bandwidth savings (bytes not re-read).
    pub bandwidth_saved_bytes: u64,
    /// Blocks evicted due to TTL expiry.
    pub blocks_expired: u64,
}

// ─────────────────────────────────────────────
// Register Cache
// ─────────────────────────────────────────────

/// In-memory cache for Modbus register blocks with change detection.
pub struct RegisterBlockCache {
    config: RegisterCacheConfig,
    blocks: HashMap<RegisterBlockKey, CachedBlock>,
    stats: RegisterCacheStats,
}

impl RegisterBlockCache {
    /// Create a new register block cache.
    pub fn new(config: RegisterCacheConfig) -> Self {
        Self {
            config,
            blocks: HashMap::new(),
            stats: RegisterCacheStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RegisterCacheConfig::default())
    }

    /// Update the cache with new register values and detect changes.
    ///
    /// Returns a `BlockChangeEvent` describing which registers changed.
    pub fn update(
        &mut self,
        key: &RegisterBlockKey,
        new_values: &[u16],
    ) -> Result<BlockChangeEvent, ModbusError> {
        if new_values.len() != key.count as usize {
            return Err(ModbusError::Config(format!(
                "Value count {} does not match block count {}",
                new_values.len(),
                key.count
            )));
        }

        self.stats.total_updates += 1;
        let now = Instant::now();
        let now_utc = Utc::now();

        // Clone old values to avoid borrow conflict with detect_changes
        let old_values_opt = self.blocks.get(key).map(|cached| cached.values.clone());
        let changes = if let Some(old_values) = old_values_opt {
            self.detect_changes(key, &old_values, new_values, now_utc)
        } else {
            // First time seeing this block — all values are "new"
            Vec::new()
        };

        let changed_count = changes.len();
        let has_changes = !changes.is_empty();

        if has_changes {
            self.stats.total_changes_detected += changed_count as u64;
        }

        // Update or insert the cached block
        let history_depth = self.config.history_depth;
        let block = self
            .blocks
            .entry(key.clone())
            .or_insert_with(|| CachedBlock {
                values: vec![0; key.count as usize],
                last_updated: now,
                last_updated_utc: now_utc,
                history: vec![Vec::with_capacity(history_depth); key.count as usize],
                change_counts: vec![0; key.count as usize],
                poll_count: 0,
            });

        // Update history for changed registers
        for change in &changes {
            let idx = (change.address - key.start_address) as usize;
            if idx < block.history.len() {
                block.history[idx].push((change.timestamp, change.new_value));
                if block.history[idx].len() > history_depth {
                    block.history[idx].remove(0);
                }
                block.change_counts[idx] += 1;
            }
        }

        block.values = new_values.to_vec();
        block.last_updated = now;
        block.last_updated_utc = now_utc;
        block.poll_count += 1;

        self.stats.cached_blocks = self.blocks.len();
        self.stats.total_registers_cached = self.blocks.values().map(|b| b.values.len()).sum();

        // Enforce max blocks
        if self.blocks.len() > self.config.max_blocks {
            self.evict_oldest();
        }

        let event = BlockChangeEvent {
            block_key: key.clone(),
            changes,
            has_changes,
            total_registers: key.count as usize,
            changed_count,
            timestamp: now_utc,
        };

        if has_changes {
            debug!(
                block = %format!("{}:{}", key.unit_id, key.start_address),
                changed = changed_count,
                total = key.count,
                "Register changes detected"
            );
        }

        Ok(event)
    }

    /// Look up cached values for a register block.
    ///
    /// Returns `None` if the block is not cached or has expired.
    pub fn get(&mut self, key: &RegisterBlockKey) -> Option<&[u16]> {
        self.stats.total_lookups += 1;

        // Check existence and expiry without holding an immutable borrow across mutable calls
        let status = self
            .blocks
            .get(key)
            .map(|block| block.last_updated.elapsed() <= self.config.ttl);

        match status {
            Some(true) => {
                // Cache hit
                self.stats.cache_hits += 1;
                self.stats.bandwidth_saved_bytes += (key.count as u64) * 2; // 2 bytes per register
                self.update_hit_rate();
                // Re-borrow immutably now that all mutable operations are done
                self.blocks.get(key).map(|b| b.values.as_slice())
            }
            Some(false) => {
                // Expired
                self.stats.blocks_expired += 1;
                self.stats.cache_misses += 1;
                self.update_hit_rate();
                None
            }
            None => {
                // Not found
                self.stats.cache_misses += 1;
                self.update_hit_rate();
                None
            }
        }
    }

    /// Check if a block is cached and not expired.
    pub fn contains(&self, key: &RegisterBlockKey) -> bool {
        self.blocks
            .get(key)
            .map(|b| b.last_updated.elapsed() <= self.config.ttl)
            .unwrap_or(false)
    }

    /// Get the change history for a specific register.
    pub fn register_history(&self, unit_id: u8, address: u16) -> Option<Vec<(DateTime<Utc>, u16)>> {
        for (key, block) in &self.blocks {
            if key.unit_id == unit_id
                && address >= key.start_address
                && address < key.start_address + key.count
            {
                let idx = (address - key.start_address) as usize;
                if idx < block.history.len() {
                    return Some(block.history[idx].clone());
                }
            }
        }
        None
    }

    /// Get the change count for a specific register.
    pub fn register_change_count(&self, unit_id: u8, address: u16) -> u64 {
        for (key, block) in &self.blocks {
            if key.unit_id == unit_id
                && address >= key.start_address
                && address < key.start_address + key.count
            {
                let idx = (address - key.start_address) as usize;
                if idx < block.change_counts.len() {
                    return block.change_counts[idx];
                }
            }
        }
        0
    }

    /// Invalidate (remove) a specific block.
    pub fn invalidate(&mut self, key: &RegisterBlockKey) -> bool {
        let removed = self.blocks.remove(key).is_some();
        if removed {
            self.stats.cached_blocks = self.blocks.len();
        }
        removed
    }

    /// Invalidate all blocks for a given unit ID.
    pub fn invalidate_unit(&mut self, unit_id: u8) -> usize {
        let keys: Vec<RegisterBlockKey> = self
            .blocks
            .keys()
            .filter(|k| k.unit_id == unit_id)
            .cloned()
            .collect();
        let count = keys.len();
        for key in keys {
            self.blocks.remove(&key);
        }
        self.stats.cached_blocks = self.blocks.len();
        count
    }

    /// Clear all cached blocks.
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.stats.cached_blocks = 0;
        self.stats.total_registers_cached = 0;
        info!("Register cache cleared");
    }

    /// Get current statistics.
    pub fn stats(&self) -> &RegisterCacheStats {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &RegisterCacheConfig {
        &self.config
    }

    /// Get number of cached blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Detect changes between old and new values, applying dead-band filter.
    fn detect_changes(
        &mut self,
        key: &RegisterBlockKey,
        old: &[u16],
        new: &[u16],
        timestamp: DateTime<Utc>,
    ) -> Vec<RegisterChange> {
        let mut changes = Vec::new();
        let dead_band = self.config.dead_band;

        for (i, (&old_val, &new_val)) in old.iter().zip(new.iter()).enumerate() {
            let delta = new_val.abs_diff(old_val);

            if delta > dead_band {
                changes.push(RegisterChange {
                    address: key.start_address + i as u16,
                    old_value: old_val,
                    new_value: new_val,
                    delta,
                    timestamp,
                });
            } else if delta > 0 {
                self.stats.dead_band_filtered += 1;
            }
        }
        changes
    }

    /// Evict the oldest cached block (by last_updated).
    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .blocks
            .iter()
            .min_by_key(|(_, b)| b.last_updated)
            .map(|(k, _)| k.clone())
        {
            self.blocks.remove(&oldest_key);
            self.stats.cached_blocks = self.blocks.len();
        }
    }

    /// Update the hit rate statistic.
    fn update_hit_rate(&mut self) {
        let total = self.stats.cache_hits + self.stats.cache_misses;
        self.stats.hit_rate = if total > 0 {
            self.stats.cache_hits as f64 / total as f64
        } else {
            0.0
        };
    }

    /// Expire all stale blocks (older than TTL).
    pub fn expire_stale(&mut self) -> usize {
        let ttl = self.config.ttl;
        let stale_keys: Vec<RegisterBlockKey> = self
            .blocks
            .iter()
            .filter(|(_, b)| b.last_updated.elapsed() > ttl)
            .map(|(k, _)| k.clone())
            .collect();
        let count = stale_keys.len();
        for key in stale_keys {
            self.blocks.remove(&key);
        }
        self.stats.blocks_expired += count as u64;
        self.stats.cached_blocks = self.blocks.len();
        count
    }

    /// Get the poll count for a block.
    pub fn block_poll_count(&self, key: &RegisterBlockKey) -> u64 {
        self.blocks.get(key).map_or(0, |b| b.poll_count)
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cache() -> RegisterBlockCache {
        RegisterBlockCache::with_defaults()
    }

    fn block_key(unit: u8, start: u16, count: u16) -> RegisterBlockKey {
        RegisterBlockKey::new(unit, start, count)
    }

    #[test]
    fn test_default_config() {
        let config = RegisterCacheConfig::default();
        assert_eq!(config.ttl, Duration::from_secs(5));
        assert_eq!(config.dead_band, 0);
        assert_eq!(config.max_blocks, 256);
        assert_eq!(config.history_depth, 10);
    }

    #[test]
    fn test_initial_update_no_changes() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 3);
        let event = cache.update(&key, &[100, 200, 300]).expect("update failed");
        assert!(!event.has_changes, "First update should not report changes");
        assert_eq!(event.changed_count, 0);
    }

    #[test]
    fn test_value_change_detection() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 3);
        cache.update(&key, &[100, 200, 300]).expect("update failed");

        let event = cache.update(&key, &[100, 250, 300]).expect("update failed");
        assert!(event.has_changes);
        assert_eq!(event.changed_count, 1);
        assert_eq!(event.changes[0].address, 1);
        assert_eq!(event.changes[0].old_value, 200);
        assert_eq!(event.changes[0].new_value, 250);
        assert_eq!(event.changes[0].delta, 50);
    }

    #[test]
    fn test_no_change_same_values() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 2);
        cache.update(&key, &[100, 200]).expect("update failed");

        let event = cache.update(&key, &[100, 200]).expect("update failed");
        assert!(!event.has_changes);
    }

    #[test]
    fn test_dead_band_filtering() {
        let mut cache = RegisterBlockCache::new(RegisterCacheConfig {
            dead_band: 10,
            ..Default::default()
        });
        let key = block_key(1, 0, 2);
        cache.update(&key, &[100, 200]).expect("update failed");

        // Change within dead band — should not report
        let event = cache.update(&key, &[105, 200]).expect("update failed");
        assert!(!event.has_changes);
        assert_eq!(cache.stats().dead_band_filtered, 1);

        // Change outside dead band — should report
        let event = cache.update(&key, &[120, 200]).expect("update failed");
        assert!(event.has_changes);
        assert_eq!(event.changes[0].delta, 15); // 120 - 105 = 15 > dead_band of 10
    }

    #[test]
    fn test_cache_get_hit() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 3);
        cache.update(&key, &[10, 20, 30]).expect("update failed");

        let values = cache.get(&key);
        assert!(values.is_some());
        assert_eq!(values.expect("should have values"), &[10, 20, 30]);
        assert_eq!(cache.stats().cache_hits, 1);
    }

    #[test]
    fn test_cache_get_miss() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 3);
        let values = cache.get(&key);
        assert!(values.is_none());
        assert_eq!(cache.stats().cache_misses, 1);
    }

    #[test]
    fn test_cache_ttl_expiry() {
        let mut cache = RegisterBlockCache::new(RegisterCacheConfig {
            ttl: Duration::from_millis(50),
            ..Default::default()
        });
        let key = block_key(1, 0, 2);
        cache.update(&key, &[10, 20]).expect("update failed");

        assert!(cache.get(&key).is_some());

        std::thread::sleep(Duration::from_millis(100));
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_contains() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 2);
        assert!(!cache.contains(&key));

        cache.update(&key, &[10, 20]).expect("update failed");
        assert!(cache.contains(&key));
    }

    #[test]
    fn test_invalidate() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 2);
        cache.update(&key, &[10, 20]).expect("update failed");
        assert!(cache.invalidate(&key));
        assert!(!cache.contains(&key));
    }

    #[test]
    fn test_invalidate_nonexistent() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 2);
        assert!(!cache.invalidate(&key));
    }

    #[test]
    fn test_invalidate_unit() {
        let mut cache = default_cache();
        cache
            .update(&block_key(1, 0, 2), &[10, 20])
            .expect("update failed");
        cache
            .update(&block_key(1, 100, 2), &[30, 40])
            .expect("update failed");
        cache
            .update(&block_key(2, 0, 2), &[50, 60])
            .expect("update failed");

        let removed = cache.invalidate_unit(1);
        assert_eq!(removed, 2);
        assert_eq!(cache.block_count(), 1);
    }

    #[test]
    fn test_clear() {
        let mut cache = default_cache();
        cache
            .update(&block_key(1, 0, 2), &[10, 20])
            .expect("update failed");
        cache
            .update(&block_key(2, 0, 2), &[30, 40])
            .expect("update failed");
        cache.clear();
        assert_eq!(cache.block_count(), 0);
    }

    #[test]
    fn test_max_blocks_eviction() {
        let mut cache = RegisterBlockCache::new(RegisterCacheConfig {
            max_blocks: 3,
            ..Default::default()
        });

        for i in 0..5 {
            cache
                .update(&block_key(1, i * 10, 2), &[10, 20])
                .expect("update failed");
        }

        assert!(cache.block_count() <= 4); // max_blocks + possible last insert
    }

    #[test]
    fn test_wrong_count_error() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 3);
        let result = cache.update(&key, &[10, 20]); // Only 2 values for 3 registers
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_changes() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 4);
        cache
            .update(&key, &[10, 20, 30, 40])
            .expect("update failed");

        let event = cache
            .update(&key, &[15, 20, 35, 45])
            .expect("update failed");
        assert_eq!(event.changed_count, 3);
    }

    #[test]
    fn test_register_history() {
        let mut cache = default_cache();
        let key = block_key(1, 100, 2);
        cache.update(&key, &[10, 20]).expect("update failed");
        cache.update(&key, &[15, 20]).expect("update failed");
        cache.update(&key, &[25, 20]).expect("update failed");

        let history = cache.register_history(1, 100);
        assert!(history.is_some());
        let hist = history.expect("should have history");
        assert_eq!(hist.len(), 2); // Two changes
    }

    #[test]
    fn test_register_history_not_found() {
        let cache = default_cache();
        assert!(cache.register_history(1, 999).is_none());
    }

    #[test]
    fn test_register_change_count() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 2);
        cache.update(&key, &[10, 20]).expect("update failed");
        cache.update(&key, &[15, 20]).expect("update failed");
        cache.update(&key, &[25, 20]).expect("update failed");

        assert_eq!(cache.register_change_count(1, 0), 2);
        assert_eq!(cache.register_change_count(1, 1), 0);
    }

    #[test]
    fn test_block_poll_count() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 2);
        cache.update(&key, &[10, 20]).expect("update failed");
        cache.update(&key, &[15, 20]).expect("update failed");
        cache.update(&key, &[25, 20]).expect("update failed");

        assert_eq!(cache.block_poll_count(&key), 3);
    }

    #[test]
    fn test_block_poll_count_unknown() {
        let cache = default_cache();
        let key = block_key(1, 0, 2);
        assert_eq!(cache.block_poll_count(&key), 0);
    }

    #[test]
    fn test_stats_bandwidth_savings() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 5);
        cache.update(&key, &[1, 2, 3, 4, 5]).expect("update failed");

        let _ = cache.get(&key); // Cache hit
        assert_eq!(cache.stats().bandwidth_saved_bytes, 10); // 5 regs * 2 bytes
    }

    #[test]
    fn test_hit_rate_calculation() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 2);
        cache.update(&key, &[10, 20]).expect("update failed");

        let _ = cache.get(&key); // hit
        let _ = cache.get(&block_key(1, 100, 2)); // miss

        let stats = cache.stats();
        assert!((stats.hit_rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_expire_stale() {
        let mut cache = RegisterBlockCache::new(RegisterCacheConfig {
            ttl: Duration::from_millis(50),
            ..Default::default()
        });

        cache
            .update(&block_key(1, 0, 2), &[10, 20])
            .expect("update failed");
        std::thread::sleep(Duration::from_millis(100));

        let expired = cache.expire_stale();
        assert_eq!(expired, 1);
        assert_eq!(cache.block_count(), 0);
    }

    #[test]
    fn test_expire_stale_none() {
        let mut cache = default_cache();
        cache
            .update(&block_key(1, 0, 2), &[10, 20])
            .expect("update failed");
        let expired = cache.expire_stale();
        assert_eq!(expired, 0);
    }

    #[test]
    fn test_block_key_equality() {
        let k1 = block_key(1, 100, 10);
        let k2 = block_key(1, 100, 10);
        assert_eq!(k1, k2);

        let k3 = block_key(2, 100, 10);
        assert_ne!(k1, k3);
    }

    #[test]
    fn test_config_serialization() {
        let config = RegisterCacheConfig::default();
        let json = serde_json::to_string(&config).expect("serialize failed");
        assert!(json.contains("dead_band"));
    }

    #[test]
    fn test_change_event_serialization() {
        let event = BlockChangeEvent {
            block_key: block_key(1, 0, 2),
            changes: vec![RegisterChange {
                address: 0,
                old_value: 10,
                new_value: 20,
                delta: 10,
                timestamp: Utc::now(),
            }],
            has_changes: true,
            total_registers: 2,
            changed_count: 1,
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&event).expect("serialize failed");
        assert!(json.contains("block_key"));
    }

    #[test]
    fn test_multiple_units() {
        let mut cache = default_cache();
        cache
            .update(&block_key(1, 0, 2), &[10, 20])
            .expect("update failed");
        cache
            .update(&block_key(2, 0, 2), &[30, 40])
            .expect("update failed");
        cache
            .update(&block_key(3, 0, 2), &[50, 60])
            .expect("update failed");

        assert_eq!(cache.block_count(), 3);
        assert_eq!(
            cache.get(&block_key(2, 0, 2)).expect("should hit"),
            &[30, 40]
        );
    }

    #[test]
    fn test_history_depth_limit() {
        let mut cache = RegisterBlockCache::new(RegisterCacheConfig {
            history_depth: 3,
            ..Default::default()
        });
        let key = block_key(1, 0, 1);

        for i in 0u16..10 {
            cache.update(&key, &[i * 10]).expect("update failed");
        }

        let history = cache.register_history(1, 0).expect("should have history");
        assert!(history.len() <= 3);
    }

    #[test]
    fn test_stats_total_updates() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 2);
        cache.update(&key, &[10, 20]).expect("update failed");
        cache.update(&key, &[15, 25]).expect("update failed");
        assert_eq!(cache.stats().total_updates, 2);
    }

    #[test]
    fn test_stats_total_changes() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 3);
        cache.update(&key, &[10, 20, 30]).expect("update failed");
        cache.update(&key, &[15, 25, 30]).expect("update failed");
        assert_eq!(cache.stats().total_changes_detected, 2);
    }

    #[test]
    fn test_decrease_change_detection() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 1);
        cache.update(&key, &[100]).expect("update failed");
        let event = cache.update(&key, &[50]).expect("update failed");
        assert!(event.has_changes);
        assert_eq!(event.changes[0].delta, 50);
    }

    #[test]
    fn test_zero_value_changes() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 2);
        cache.update(&key, &[0, 0]).expect("update failed");
        let event = cache.update(&key, &[0, 1]).expect("update failed");
        assert_eq!(event.changed_count, 1);
    }

    #[test]
    fn test_max_value_changes() {
        let mut cache = default_cache();
        let key = block_key(1, 0, 1);
        cache.update(&key, &[0]).expect("update failed");
        let event = cache.update(&key, &[65535]).expect("update failed");
        assert!(event.has_changes);
        assert_eq!(event.changes[0].delta, 65535);
    }
}
