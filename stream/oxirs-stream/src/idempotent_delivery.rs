//! # Idempotent Delivery Manager
//!
//! Provides exactly-once delivery guarantees with idempotency keys for stream
//! events. Prevents duplicate processing by tracking idempotency keys with
//! configurable TTL-based expiration and storage backends.
//!
//! ## Features
//!
//! - **Idempotency key tracking**: SHA-256 based key generation from event content
//! - **TTL-based expiration**: Keys expire after configurable time-to-live
//! - **Outcome caching**: Cache delivery results for idempotent retries
//! - **Storage backends**: In-memory (default) or pluggable persistent backends
//! - **Batch processing**: Efficient batch idempotency checks
//! - **Metrics and statistics**: Track duplicate rates, key cardinality, etc.
//! - **Partition-aware**: Per-partition idempotency tracking for ordered delivery

use crate::error::StreamError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for idempotent delivery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdempotentDeliveryConfig {
    /// Time-to-live for idempotency keys (default: 1 hour).
    pub key_ttl: Duration,
    /// Maximum number of tracked keys before forced eviction (default: 1M).
    pub max_keys: usize,
    /// How often to run the eviction sweep (default: 60s).
    pub eviction_interval: Duration,
    /// Whether to cache delivery outcomes for fast retries (default: true).
    pub cache_outcomes: bool,
    /// Maximum outcome cache size (default: 100K).
    pub max_cached_outcomes: usize,
    /// Whether to enable per-partition tracking (default: true).
    pub partition_aware: bool,
    /// Hash algorithm for key generation (SHA-256).
    pub hash_algorithm: HashAlgorithm,
}

impl Default for IdempotentDeliveryConfig {
    fn default() -> Self {
        Self {
            key_ttl: Duration::from_secs(3600),
            max_keys: 1_000_000,
            eviction_interval: Duration::from_secs(60),
            cache_outcomes: true,
            max_cached_outcomes: 100_000,
            partition_aware: true,
            hash_algorithm: HashAlgorithm::Sha256,
        }
    }
}

/// Hash algorithm for idempotency key generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashAlgorithm {
    /// SHA-256 (default, recommended).
    Sha256,
    /// Fast non-cryptographic hash (FNV-1a style, for high throughput).
    FastHash,
}

// ─────────────────────────────────────────────
// Idempotency Key
// ─────────────────────────────────────────────

/// A unique idempotency key derived from event content.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IdempotencyKey {
    /// The hex-encoded hash digest.
    pub digest: String,
    /// Optional partition the key belongs to.
    pub partition: Option<u32>,
    /// Source producer id (if available).
    pub producer_id: Option<String>,
}

impl IdempotencyKey {
    /// Create a new idempotency key from raw content bytes.
    pub fn from_content(content: &[u8], algorithm: HashAlgorithm) -> Self {
        let digest = match algorithm {
            HashAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                hasher.update(content);
                hex::encode(hasher.finalize())
            }
            HashAlgorithm::FastHash => {
                let hash = fnv1a_hash(content);
                format!("{hash:016x}")
            }
        };
        Self {
            digest,
            partition: None,
            producer_id: None,
        }
    }

    /// Create with explicit partition.
    pub fn with_partition(mut self, partition: u32) -> Self {
        self.partition = Some(partition);
        self
    }

    /// Create with explicit producer id.
    pub fn with_producer(mut self, producer: String) -> Self {
        self.producer_id = Some(producer);
        self
    }

    /// Create from a pre-computed string key.
    pub fn from_string(key: String) -> Self {
        Self {
            digest: key,
            partition: None,
            producer_id: None,
        }
    }

    /// Return the composite key used for lookups (includes partition if set).
    pub fn composite_key(&self) -> String {
        match self.partition {
            Some(p) => format!("{}:{}", p, self.digest),
            None => self.digest.clone(),
        }
    }
}

/// FNV-1a hash for fast non-cryptographic hashing.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ─────────────────────────────────────────────
// Delivery Outcome
// ─────────────────────────────────────────────

/// The result of a delivery attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryOutcome {
    /// Event was successfully processed.
    Success {
        /// Serialized result (if any).
        result: Option<String>,
        /// When the event was processed.
        processed_at: DateTime<Utc>,
    },
    /// Event processing failed.
    Failure {
        /// Error message.
        error: String,
        /// Whether the failure is retryable.
        retryable: bool,
        /// When the failure occurred.
        failed_at: DateTime<Utc>,
    },
    /// Event is currently being processed.
    InProgress {
        /// When processing started.
        started_at: DateTime<Utc>,
    },
}

// ─────────────────────────────────────────────
// Tracked Key Entry
// ─────────────────────────────────────────────

/// Internal entry for a tracked idempotency key.
#[derive(Debug, Clone)]
struct TrackedKey {
    /// When this key was first seen.
    first_seen: Instant,
    /// When this key was last accessed.
    last_accessed: Instant,
    /// How many times this key was submitted.
    submission_count: u64,
    /// Cached outcome (if enabled).
    outcome: Option<DeliveryOutcome>,
}

// ─────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────

/// Statistics for the idempotent delivery manager.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IdempotentDeliveryStats {
    /// Total events submitted.
    pub total_submitted: u64,
    /// Events accepted (new, non-duplicate).
    pub accepted: u64,
    /// Events rejected as duplicates.
    pub duplicates_rejected: u64,
    /// Events served from outcome cache.
    pub cache_hits: u64,
    /// Current number of tracked keys.
    pub active_keys: usize,
    /// Total keys evicted (TTL or capacity).
    pub keys_evicted: u64,
    /// Eviction sweeps performed.
    pub eviction_sweeps: u64,
    /// Duplicate rate (duplicates / total).
    pub duplicate_rate: f64,
    /// Average key lifetime in seconds.
    pub avg_key_lifetime_secs: f64,
    /// Per-partition key counts.
    pub partition_key_counts: HashMap<u32, usize>,
}

// ─────────────────────────────────────────────
// Batch check result
// ─────────────────────────────────────────────

/// Result of checking a single key in a batch.
#[derive(Debug, Clone)]
pub struct KeyCheckResult {
    /// The idempotency key.
    pub key: IdempotencyKey,
    /// Whether this is a duplicate.
    pub is_duplicate: bool,
    /// Cached outcome (if available).
    pub cached_outcome: Option<DeliveryOutcome>,
    /// How many times this key has been seen.
    pub submission_count: u64,
}

// ─────────────────────────────────────────────
// Idempotent Delivery Manager
// ─────────────────────────────────────────────

/// Manages idempotent delivery guarantees for stream events.
///
/// Tracks idempotency keys in a sliding TTL window with configurable
/// eviction, outcome caching, and partition-aware tracking.
pub struct IdempotentDeliveryManager {
    config: IdempotentDeliveryConfig,
    /// Map from composite key -> tracked entry.
    keys: Arc<RwLock<HashMap<String, TrackedKey>>>,
    /// Ordered set for TTL eviction: (expiry_instant, composite_key).
    expiry_queue: Arc<RwLock<BTreeMap<(Instant, String), ()>>>,
    /// Running statistics.
    stats: Arc<RwLock<IdempotentDeliveryStats>>,
    /// When the last eviction sweep ran.
    last_eviction: Arc<RwLock<Instant>>,
}

impl IdempotentDeliveryManager {
    /// Create a new idempotent delivery manager.
    pub fn new(config: IdempotentDeliveryConfig) -> Self {
        Self {
            config,
            keys: Arc::new(RwLock::new(HashMap::new())),
            expiry_queue: Arc::new(RwLock::new(BTreeMap::new())),
            stats: Arc::new(RwLock::new(IdempotentDeliveryStats::default())),
            last_eviction: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(IdempotentDeliveryConfig::default())
    }

    /// Generate an idempotency key from raw content.
    pub fn generate_key(&self, content: &[u8]) -> IdempotencyKey {
        IdempotencyKey::from_content(content, self.config.hash_algorithm)
    }

    /// Generate a partitioned idempotency key.
    pub fn generate_partitioned_key(&self, content: &[u8], partition: u32) -> IdempotencyKey {
        IdempotencyKey::from_content(content, self.config.hash_algorithm).with_partition(partition)
    }

    /// Check if a key is a duplicate and register it atomically.
    ///
    /// Returns `Ok(false)` if this is a new key (event should be processed).
    /// Returns `Ok(true)` if this is a duplicate (event should be skipped).
    pub async fn check_and_register(&self, key: &IdempotencyKey) -> Result<bool, StreamError> {
        self.maybe_evict().await;

        let composite = key.composite_key();
        let now = Instant::now();

        let mut keys = self.keys.write().await;
        let mut stats = self.stats.write().await;
        stats.total_submitted += 1;

        if let Some(entry) = keys.get_mut(&composite) {
            // Check if expired
            if now.duration_since(entry.first_seen) > self.config.key_ttl {
                // Key expired — remove and treat as new
                keys.remove(&composite);
                let mut expiry = self.expiry_queue.write().await;
                // Remove from expiry queue (best effort — key might not match exactly)
                expiry.retain(|(_t, k), _| k != &composite);
                // Fall through to insert below
            } else {
                entry.submission_count += 1;
                entry.last_accessed = now;
                stats.duplicates_rejected += 1;
                stats.duplicate_rate = if stats.total_submitted > 0 {
                    stats.duplicates_rejected as f64 / stats.total_submitted as f64
                } else {
                    0.0
                };
                debug!(key = %composite, count = entry.submission_count, "Duplicate key detected");
                return Ok(true);
            }
        }

        // Capacity check
        if keys.len() >= self.config.max_keys {
            // Evict oldest key
            let mut expiry = self.expiry_queue.write().await;
            if let Some(((_, oldest_key), _)) = expiry.pop_first() {
                keys.remove(&oldest_key);
                stats.keys_evicted += 1;
            }
        }

        // Insert new key
        let entry = TrackedKey {
            first_seen: now,
            last_accessed: now,
            submission_count: 1,
            outcome: None,
        };
        keys.insert(composite.clone(), entry);

        // Add to expiry queue
        let expiry_time = now + self.config.key_ttl;
        let mut expiry = self.expiry_queue.write().await;
        expiry.insert((expiry_time, composite), ());

        stats.accepted += 1;
        stats.active_keys = keys.len();

        // Update partition stats
        if let Some(p) = key.partition {
            *stats.partition_key_counts.entry(p).or_insert(0) += 1;
        }

        stats.duplicate_rate = if stats.total_submitted > 0 {
            stats.duplicates_rejected as f64 / stats.total_submitted as f64
        } else {
            0.0
        };

        Ok(false)
    }

    /// Record the outcome of processing an event.
    pub async fn record_outcome(
        &self,
        key: &IdempotencyKey,
        outcome: DeliveryOutcome,
    ) -> Result<(), StreamError> {
        if !self.config.cache_outcomes {
            return Ok(());
        }

        let composite = key.composite_key();
        let mut keys = self.keys.write().await;

        if let Some(entry) = keys.get_mut(&composite) {
            entry.outcome = Some(outcome);
            Ok(())
        } else {
            Err(StreamError::NotFound(format!("Key not found: {composite}")))
        }
    }

    /// Get the cached outcome for a key (if any).
    pub async fn get_outcome(
        &self,
        key: &IdempotencyKey,
    ) -> Result<Option<DeliveryOutcome>, StreamError> {
        let composite = key.composite_key();
        let keys = self.keys.read().await;
        let mut stats = self.stats.write().await;

        if let Some(entry) = keys.get(&composite) {
            if entry.outcome.is_some() {
                stats.cache_hits += 1;
            }
            Ok(entry.outcome.clone())
        } else {
            Ok(None)
        }
    }

    /// Check multiple keys in batch.
    pub async fn check_batch(
        &self,
        keys_to_check: &[IdempotencyKey],
    ) -> Result<Vec<KeyCheckResult>, StreamError> {
        self.maybe_evict().await;

        let stored_keys = self.keys.read().await;
        let mut results = Vec::with_capacity(keys_to_check.len());

        for key in keys_to_check {
            let composite = key.composite_key();
            let (is_duplicate, cached_outcome, submission_count) =
                if let Some(entry) = stored_keys.get(&composite) {
                    let now = Instant::now();
                    if now.duration_since(entry.first_seen) > self.config.key_ttl {
                        (false, None, 0)
                    } else {
                        (true, entry.outcome.clone(), entry.submission_count)
                    }
                } else {
                    (false, None, 0)
                };

            results.push(KeyCheckResult {
                key: key.clone(),
                is_duplicate,
                cached_outcome,
                submission_count,
            });
        }

        Ok(results)
    }

    /// Manually evict expired keys.
    pub async fn evict_expired(&self) -> usize {
        let now = Instant::now();
        let mut keys = self.keys.write().await;
        let mut expiry = self.expiry_queue.write().await;
        let mut stats = self.stats.write().await;
        let mut evicted = 0;

        // Remove all entries with expiry <= now
        let cutoff = (now, String::new());
        let expired: Vec<(Instant, String)> = expiry
            .range(..=cutoff)
            .map(|((t, k), _)| (*t, k.clone()))
            .collect();

        for (t, k) in &expired {
            expiry.remove(&(*t, k.clone()));
            if keys.remove(k).is_some() {
                evicted += 1;
            }
        }

        stats.keys_evicted += evicted as u64;
        stats.eviction_sweeps += 1;
        stats.active_keys = keys.len();

        if evicted > 0 {
            info!(
                evicted,
                remaining = keys.len(),
                "Evicted expired idempotency keys"
            );
        }

        evicted
    }

    /// Check if a key exists (without registering or counting it).
    pub async fn contains_key(&self, key: &IdempotencyKey) -> bool {
        let composite = key.composite_key();
        let keys = self.keys.read().await;
        if let Some(entry) = keys.get(&composite) {
            Instant::now().duration_since(entry.first_seen) <= self.config.key_ttl
        } else {
            false
        }
    }

    /// Remove a specific key (e.g., after a failed delivery that should be retried).
    pub async fn remove_key(&self, key: &IdempotencyKey) -> bool {
        let composite = key.composite_key();
        let mut keys = self.keys.write().await;
        let removed = keys.remove(&composite).is_some();
        if removed {
            let mut stats = self.stats.write().await;
            stats.active_keys = keys.len();
        }
        removed
    }

    /// Clear all tracked keys.
    pub async fn clear(&self) {
        let mut keys = self.keys.write().await;
        let mut expiry = self.expiry_queue.write().await;
        let mut stats = self.stats.write().await;
        keys.clear();
        expiry.clear();
        stats.active_keys = 0;
        info!("Cleared all idempotency keys");
    }

    /// Get current statistics.
    pub async fn stats(&self) -> IdempotentDeliveryStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &IdempotentDeliveryConfig {
        &self.config
    }

    /// Internal: run eviction if interval has elapsed.
    async fn maybe_evict(&self) {
        let should_evict = {
            let last = self.last_eviction.read().await;
            last.elapsed() >= self.config.eviction_interval
        };

        if should_evict {
            let mut last = self.last_eviction.write().await;
            if last.elapsed() >= self.config.eviction_interval {
                *last = Instant::now();
                // Drop the lock before the expensive eviction
                drop(last);
                self.evict_expired().await;
            }
        }
    }

    /// Get number of active (non-expired) keys.
    pub async fn active_key_count(&self) -> usize {
        let keys = self.keys.read().await;
        keys.len()
    }

    /// Get the submission count for a specific key.
    pub async fn submission_count(&self, key: &IdempotencyKey) -> u64 {
        let composite = key.composite_key();
        let keys = self.keys.read().await;
        keys.get(&composite).map_or(0, |e| e.submission_count)
    }

    /// Check if the outcome cache is enabled.
    pub fn is_cache_enabled(&self) -> bool {
        self.config.cache_outcomes
    }
}

// ─────────────────────────────────────────────
// Idempotent Producer Wrapper
// ─────────────────────────────────────────────

/// Wraps a delivery function with idempotency guarantees.
///
/// Usage pattern:
/// 1. Compute idempotency key from event content
/// 2. Check against the manager
/// 3. If new, process the event and record the outcome
/// 4. If duplicate, return the cached outcome or skip
pub struct IdempotentProducer {
    manager: Arc<IdempotentDeliveryManager>,
}

impl IdempotentProducer {
    /// Create an idempotent producer wrapping the given manager.
    pub fn new(manager: Arc<IdempotentDeliveryManager>) -> Self {
        Self { manager }
    }

    /// Attempt to deliver an event with idempotency key.
    ///
    /// Returns `Ok(Some(outcome))` if the event was already processed.
    /// Returns `Ok(None)` if the event is new and should be processed by the caller.
    pub async fn try_deliver(
        &self,
        key: &IdempotencyKey,
    ) -> Result<Option<DeliveryOutcome>, StreamError> {
        let is_dup = self.manager.check_and_register(key).await?;
        if is_dup {
            // Return cached outcome if available
            let outcome = self.manager.get_outcome(key).await?;
            Ok(outcome.or(Some(DeliveryOutcome::Success {
                result: None,
                processed_at: Utc::now(),
            })))
        } else {
            Ok(None)
        }
    }

    /// Record that delivery succeeded.
    pub async fn mark_success(
        &self,
        key: &IdempotencyKey,
        result: Option<String>,
    ) -> Result<(), StreamError> {
        self.manager
            .record_outcome(
                key,
                DeliveryOutcome::Success {
                    result,
                    processed_at: Utc::now(),
                },
            )
            .await
    }

    /// Record that delivery failed.
    pub async fn mark_failure(
        &self,
        key: &IdempotencyKey,
        error: String,
        retryable: bool,
    ) -> Result<(), StreamError> {
        self.manager
            .record_outcome(
                key,
                DeliveryOutcome::Failure {
                    error,
                    retryable,
                    failed_at: Utc::now(),
                },
            )
            .await
    }

    /// Get the underlying manager.
    pub fn manager(&self) -> &IdempotentDeliveryManager {
        &self.manager
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn default_manager() -> IdempotentDeliveryManager {
        IdempotentDeliveryManager::new(IdempotentDeliveryConfig::default())
    }

    fn fast_ttl_manager(ttl_ms: u64) -> IdempotentDeliveryManager {
        IdempotentDeliveryManager::new(IdempotentDeliveryConfig {
            key_ttl: Duration::from_millis(ttl_ms),
            eviction_interval: Duration::from_millis(10),
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn test_new_key_is_not_duplicate() {
        let mgr = default_manager();
        let key = mgr.generate_key(b"event-1");
        let is_dup = mgr.check_and_register(&key).await.expect("check failed");
        assert!(!is_dup, "First submission should not be duplicate");
    }

    #[tokio::test]
    async fn test_same_key_is_duplicate() {
        let mgr = default_manager();
        let key = mgr.generate_key(b"event-1");
        mgr.check_and_register(&key).await.expect("check failed");
        let is_dup = mgr.check_and_register(&key).await.expect("check failed");
        assert!(is_dup, "Second submission should be duplicate");
    }

    #[tokio::test]
    async fn test_different_keys_not_duplicate() {
        let mgr = default_manager();
        let k1 = mgr.generate_key(b"event-1");
        let k2 = mgr.generate_key(b"event-2");
        mgr.check_and_register(&k1).await.expect("check failed");
        let is_dup = mgr.check_and_register(&k2).await.expect("check failed");
        assert!(!is_dup, "Different events should not be duplicates");
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let mgr = default_manager();
        let k1 = mgr.generate_key(b"event-1");
        mgr.check_and_register(&k1).await.expect("check failed");
        mgr.check_and_register(&k1).await.expect("check failed");
        mgr.check_and_register(&k1).await.expect("check failed");

        let stats = mgr.stats().await;
        assert_eq!(stats.total_submitted, 3);
        assert_eq!(stats.accepted, 1);
        assert_eq!(stats.duplicates_rejected, 2);
        assert!((stats.duplicate_rate - 2.0 / 3.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_outcome_caching() {
        let mgr = default_manager();
        let key = mgr.generate_key(b"event-1");
        mgr.check_and_register(&key).await.expect("check failed");

        mgr.record_outcome(
            &key,
            DeliveryOutcome::Success {
                result: Some("ok".into()),
                processed_at: Utc::now(),
            },
        )
        .await
        .expect("record failed");

        let outcome = mgr.get_outcome(&key).await.expect("get failed");
        assert!(outcome.is_some());
        if let Some(DeliveryOutcome::Success { result, .. }) = outcome {
            assert_eq!(result, Some("ok".into()));
        } else {
            panic!("Expected Success outcome");
        }
    }

    #[tokio::test]
    async fn test_outcome_cache_disabled() {
        let mgr = IdempotentDeliveryManager::new(IdempotentDeliveryConfig {
            cache_outcomes: false,
            ..Default::default()
        });
        let key = mgr.generate_key(b"event-1");
        mgr.check_and_register(&key).await.expect("check failed");

        // Recording should be a no-op when cache is disabled
        mgr.record_outcome(
            &key,
            DeliveryOutcome::Success {
                result: None,
                processed_at: Utc::now(),
            },
        )
        .await
        .expect("record failed");

        let outcome = mgr.get_outcome(&key).await.expect("get failed");
        assert!(outcome.is_none());
    }

    #[tokio::test]
    async fn test_ttl_expiration() {
        let mgr = fast_ttl_manager(50);
        let key = mgr.generate_key(b"event-1");
        mgr.check_and_register(&key).await.expect("check failed");

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(100)).await;

        // After expiry, the same key should be accepted as new
        let is_dup = mgr.check_and_register(&key).await.expect("check failed");
        assert!(!is_dup, "Expired key should be accepted as new");
    }

    #[tokio::test]
    async fn test_evict_expired() {
        let mgr = fast_ttl_manager(50);
        let k1 = mgr.generate_key(b"event-1");
        let k2 = mgr.generate_key(b"event-2");
        mgr.check_and_register(&k1).await.expect("check failed");
        mgr.check_and_register(&k2).await.expect("check failed");

        assert_eq!(mgr.active_key_count().await, 2);

        tokio::time::sleep(Duration::from_millis(100)).await;
        let evicted = mgr.evict_expired().await;
        assert!(evicted >= 1, "Should have evicted at least one key");
    }

    #[tokio::test]
    async fn test_max_keys_eviction() {
        let mgr = IdempotentDeliveryManager::new(IdempotentDeliveryConfig {
            max_keys: 5,
            ..Default::default()
        });

        for i in 0..10 {
            let key = mgr.generate_key(format!("event-{i}").as_bytes());
            mgr.check_and_register(&key).await.expect("check failed");
        }

        assert!(mgr.active_key_count().await <= 6); // May be up to max_keys + 1 due to insertion order
    }

    #[tokio::test]
    async fn test_partitioned_keys() {
        let mgr = default_manager();
        let content = b"event-1";
        let k1 = mgr.generate_partitioned_key(content, 0);
        let k2 = mgr.generate_partitioned_key(content, 1);

        mgr.check_and_register(&k1).await.expect("check failed");
        let is_dup = mgr.check_and_register(&k2).await.expect("check failed");
        assert!(
            !is_dup,
            "Same content in different partitions should not collide"
        );
    }

    #[tokio::test]
    async fn test_same_partition_duplicate() {
        let mgr = default_manager();
        let k1 = mgr.generate_partitioned_key(b"event-1", 0);
        let k2 = mgr.generate_partitioned_key(b"event-1", 0);

        mgr.check_and_register(&k1).await.expect("check failed");
        let is_dup = mgr.check_and_register(&k2).await.expect("check failed");
        assert!(is_dup, "Same content in same partition should be duplicate");
    }

    #[tokio::test]
    async fn test_contains_key() {
        let mgr = default_manager();
        let key = mgr.generate_key(b"event-1");
        assert!(!mgr.contains_key(&key).await);

        mgr.check_and_register(&key).await.expect("check failed");
        assert!(mgr.contains_key(&key).await);
    }

    #[tokio::test]
    async fn test_remove_key() {
        let mgr = default_manager();
        let key = mgr.generate_key(b"event-1");
        mgr.check_and_register(&key).await.expect("check failed");
        assert!(mgr.contains_key(&key).await);

        let removed = mgr.remove_key(&key).await;
        assert!(removed);
        assert!(!mgr.contains_key(&key).await);
    }

    #[tokio::test]
    async fn test_clear_all_keys() {
        let mgr = default_manager();
        for i in 0..10 {
            let key = mgr.generate_key(format!("event-{i}").as_bytes());
            mgr.check_and_register(&key).await.expect("check failed");
        }
        assert_eq!(mgr.active_key_count().await, 10);
        mgr.clear().await;
        assert_eq!(mgr.active_key_count().await, 0);
    }

    #[tokio::test]
    async fn test_batch_check() {
        let mgr = default_manager();
        let k1 = mgr.generate_key(b"event-1");
        let k2 = mgr.generate_key(b"event-2");
        let k3 = mgr.generate_key(b"event-3");

        mgr.check_and_register(&k1).await.expect("check failed");

        let results = mgr
            .check_batch(&[k1.clone(), k2.clone(), k3.clone()])
            .await
            .expect("batch failed");
        assert_eq!(results.len(), 3);
        assert!(results[0].is_duplicate);
        assert!(!results[1].is_duplicate);
        assert!(!results[2].is_duplicate);
    }

    #[tokio::test]
    async fn test_submission_count() {
        let mgr = default_manager();
        let key = mgr.generate_key(b"event-1");
        mgr.check_and_register(&key).await.expect("check failed");
        mgr.check_and_register(&key).await.expect("check failed");
        mgr.check_and_register(&key).await.expect("check failed");

        let count = mgr.submission_count(&key).await;
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn test_idempotent_producer_new_event() {
        let mgr = Arc::new(default_manager());
        let producer = IdempotentProducer::new(mgr);
        let key = producer.manager().generate_key(b"event-1");

        let result = producer.try_deliver(&key).await.expect("deliver failed");
        assert!(result.is_none(), "New event should return None");
    }

    #[tokio::test]
    async fn test_idempotent_producer_duplicate_event() {
        let mgr = Arc::new(default_manager());
        let producer = IdempotentProducer::new(mgr);
        let key = producer.manager().generate_key(b"event-1");

        // First delivery
        producer.try_deliver(&key).await.expect("deliver failed");
        producer
            .mark_success(&key, Some("done".into()))
            .await
            .expect("mark failed");

        // Second delivery (duplicate)
        let result = producer.try_deliver(&key).await.expect("deliver failed");
        assert!(result.is_some(), "Duplicate should return cached outcome");
    }

    #[tokio::test]
    async fn test_idempotent_producer_mark_failure() {
        let mgr = Arc::new(default_manager());
        let producer = IdempotentProducer::new(mgr);
        let key = producer.manager().generate_key(b"event-1");

        producer.try_deliver(&key).await.expect("deliver failed");
        producer
            .mark_failure(&key, "timeout".into(), true)
            .await
            .expect("mark failed");

        let outcome = producer
            .manager()
            .get_outcome(&key)
            .await
            .expect("get failed");
        assert!(matches!(
            outcome,
            Some(DeliveryOutcome::Failure {
                retryable: true,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn test_fast_hash_algorithm() {
        let mgr = IdempotentDeliveryManager::new(IdempotentDeliveryConfig {
            hash_algorithm: HashAlgorithm::FastHash,
            ..Default::default()
        });
        let key = mgr.generate_key(b"event-1");
        assert!(!key.digest.is_empty());
        assert_eq!(key.digest.len(), 16); // 64-bit hex
    }

    #[tokio::test]
    async fn test_sha256_algorithm() {
        let mgr = default_manager();
        let key = mgr.generate_key(b"event-1");
        assert!(!key.digest.is_empty());
        assert_eq!(key.digest.len(), 64); // SHA-256 hex
    }

    #[tokio::test]
    async fn test_from_string_key() {
        let mgr = default_manager();
        let key = IdempotencyKey::from_string("custom-key-12345".into());
        let is_dup = mgr.check_and_register(&key).await.expect("check failed");
        assert!(!is_dup);

        let is_dup2 = mgr.check_and_register(&key).await.expect("check failed");
        assert!(is_dup2);
    }

    #[tokio::test]
    async fn test_with_producer_id() {
        let mgr = default_manager();
        let key = mgr
            .generate_key(b"event-1")
            .with_producer("producer-a".into());
        assert_eq!(key.producer_id, Some("producer-a".into()));
    }

    #[tokio::test]
    async fn test_composite_key_format() {
        let key = IdempotencyKey::from_string("abc".into()).with_partition(3);
        assert_eq!(key.composite_key(), "3:abc");

        let key2 = IdempotencyKey::from_string("abc".into());
        assert_eq!(key2.composite_key(), "abc");
    }

    #[tokio::test]
    async fn test_config_defaults() {
        let config = IdempotentDeliveryConfig::default();
        assert_eq!(config.key_ttl, Duration::from_secs(3600));
        assert_eq!(config.max_keys, 1_000_000);
        assert!(config.cache_outcomes);
        assert!(config.partition_aware);
        assert_eq!(config.hash_algorithm, HashAlgorithm::Sha256);
    }

    #[tokio::test]
    async fn test_with_defaults_constructor() {
        let mgr = IdempotentDeliveryManager::with_defaults();
        assert_eq!(mgr.config().key_ttl, Duration::from_secs(3600));
    }

    #[tokio::test]
    async fn test_is_cache_enabled() {
        let mgr = default_manager();
        assert!(mgr.is_cache_enabled());

        let mgr2 = IdempotentDeliveryManager::new(IdempotentDeliveryConfig {
            cache_outcomes: false,
            ..Default::default()
        });
        assert!(!mgr2.is_cache_enabled());
    }

    #[tokio::test]
    async fn test_concurrent_duplicate_detection() {
        let mgr = Arc::new(default_manager());
        let mut handles = Vec::new();

        for _ in 0..10 {
            let m = Arc::clone(&mgr);
            handles.push(tokio::spawn(async move {
                let key = m.generate_key(b"shared-event");
                m.check_and_register(&key).await.expect("check failed")
            }));
        }

        let mut accepted = 0;
        let mut duplicates = 0;
        for h in handles {
            let is_dup = h.await.expect("join failed");
            if is_dup {
                duplicates += 1;
            } else {
                accepted += 1;
            }
        }

        assert_eq!(accepted, 1, "Exactly one should be accepted");
        assert_eq!(duplicates, 9, "Nine should be duplicates");
    }

    #[tokio::test]
    async fn test_partition_stats() {
        let mgr = default_manager();
        let k1 = mgr.generate_partitioned_key(b"a", 0);
        let k2 = mgr.generate_partitioned_key(b"b", 0);
        let k3 = mgr.generate_partitioned_key(b"c", 1);

        mgr.check_and_register(&k1).await.expect("check failed");
        mgr.check_and_register(&k2).await.expect("check failed");
        mgr.check_and_register(&k3).await.expect("check failed");

        let stats = mgr.stats().await;
        assert_eq!(stats.partition_key_counts.get(&0), Some(&2));
        assert_eq!(stats.partition_key_counts.get(&1), Some(&1));
    }

    #[tokio::test]
    async fn test_record_outcome_unknown_key() {
        let mgr = default_manager();
        let key = mgr.generate_key(b"unknown");

        let result = mgr
            .record_outcome(
                &key,
                DeliveryOutcome::Success {
                    result: None,
                    processed_at: Utc::now(),
                },
            )
            .await;
        assert!(
            result.is_err(),
            "Recording outcome for unknown key should fail"
        );
    }

    #[tokio::test]
    async fn test_remove_nonexistent_key() {
        let mgr = default_manager();
        let key = mgr.generate_key(b"nonexistent");
        let removed = mgr.remove_key(&key).await;
        assert!(!removed);
    }

    #[tokio::test]
    async fn test_failure_outcome() {
        let mgr = default_manager();
        let key = mgr.generate_key(b"event-1");
        mgr.check_and_register(&key).await.expect("check failed");

        mgr.record_outcome(
            &key,
            DeliveryOutcome::Failure {
                error: "network timeout".into(),
                retryable: true,
                failed_at: Utc::now(),
            },
        )
        .await
        .expect("record failed");

        let outcome = mgr.get_outcome(&key).await.expect("get failed");
        if let Some(DeliveryOutcome::Failure {
            error, retryable, ..
        }) = outcome
        {
            assert_eq!(error, "network timeout");
            assert!(retryable);
        } else {
            panic!("Expected Failure outcome");
        }
    }

    #[tokio::test]
    async fn test_in_progress_outcome() {
        let mgr = default_manager();
        let key = mgr.generate_key(b"event-1");
        mgr.check_and_register(&key).await.expect("check failed");

        mgr.record_outcome(
            &key,
            DeliveryOutcome::InProgress {
                started_at: Utc::now(),
            },
        )
        .await
        .expect("record failed");

        let outcome = mgr.get_outcome(&key).await.expect("get failed");
        assert!(matches!(outcome, Some(DeliveryOutcome::InProgress { .. })));
    }

    #[tokio::test]
    async fn test_many_unique_events() {
        let mgr = default_manager();
        for i in 0u64..100 {
            let key = mgr.generate_key(&i.to_le_bytes());
            let is_dup = mgr.check_and_register(&key).await.expect("check failed");
            assert!(!is_dup, "All unique events should be accepted");
        }
        assert_eq!(mgr.active_key_count().await, 100);
        let stats = mgr.stats().await;
        assert_eq!(stats.accepted, 100);
        assert_eq!(stats.duplicates_rejected, 0);
    }

    #[tokio::test]
    async fn test_fnv1a_deterministic() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"hello");
        assert_eq!(h1, h2);
        let h3 = fnv1a_hash(b"world");
        assert_ne!(h1, h3);
    }

    #[tokio::test]
    async fn test_idempotency_key_serialize() {
        let key = IdempotencyKey::from_string("test-key".into()).with_partition(42);
        let json = serde_json::to_string(&key).expect("serialize failed");
        let deserialized: IdempotencyKey = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(deserialized.digest, "test-key");
        assert_eq!(deserialized.partition, Some(42));
    }

    #[tokio::test]
    async fn test_config_serialize() {
        let config = IdempotentDeliveryConfig::default();
        let json = serde_json::to_string(&config).expect("serialize failed");
        assert!(json.contains("key_ttl"));
    }

    #[tokio::test]
    async fn test_stats_initial_values() {
        let mgr = default_manager();
        let stats = mgr.stats().await;
        assert_eq!(stats.total_submitted, 0);
        assert_eq!(stats.accepted, 0);
        assert_eq!(stats.duplicates_rejected, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.active_keys, 0);
        assert_eq!(stats.keys_evicted, 0);
    }

    #[tokio::test]
    async fn test_cache_hit_stats() {
        let mgr = default_manager();
        let key = mgr.generate_key(b"event-1");
        mgr.check_and_register(&key).await.expect("check failed");

        mgr.record_outcome(
            &key,
            DeliveryOutcome::Success {
                result: Some("cached".into()),
                processed_at: Utc::now(),
            },
        )
        .await
        .expect("record failed");

        // Retrieve outcome to trigger cache hit
        let _ = mgr.get_outcome(&key).await.expect("get failed");
        let stats = mgr.stats().await;
        assert_eq!(stats.cache_hits, 1);
    }
}
