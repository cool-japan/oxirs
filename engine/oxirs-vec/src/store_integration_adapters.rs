use crate::store_integration_types::*;
use crate::{
    embeddings::EmbeddingStrategy, rdf_integration::RdfVectorConfig,
    sparql_integration::SparqlVectorService, VectorStoreTrait,
};
use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, SystemTime};

pub fn new_integrated_store(
    config: StoreIntegrationConfig,
    embedding_strategy: EmbeddingStrategy,
) -> Result<IntegratedVectorStore> {
    let vector_store = Arc::new(RwLock::new(
        crate::VectorStore::with_embedding_strategy(embedding_strategy.clone())?.with_config(
            crate::VectorStoreConfig {
                auto_embed: true,
                cache_embeddings: config.cache_config.enable_vector_cache,
                similarity_threshold: 0.7,
                max_results: 1000,
            },
        ),
    ));

    let rdf_config = RdfVectorConfig::default();
    let vector_store_wrapper = VectorStoreWrapper {
        store: vector_store.clone(),
    };
    // RdfVectorIntegration now uses parking_lot::RwLock (no lock poisoning);
    // match that type here rather than std::sync::RwLock.
    let vector_store_trait: Arc<parking_lot::RwLock<dyn VectorStoreTrait>> =
        Arc::new(parking_lot::RwLock::new(vector_store_wrapper));
    let rdf_integration = Arc::new(RwLock::new(
        crate::rdf_integration::RdfVectorIntegration::new(rdf_config, vector_store_trait),
    ));

    let sparql_config = crate::sparql_integration::VectorServiceConfig::default();
    let sparql_service = Arc::new(RwLock::new(SparqlVectorService::new(
        sparql_config,
        embedding_strategy,
    )?));

    let transaction_manager = Arc::new(TransactionManager::new(config.clone()));
    let streaming_engine = Arc::new(StreamingEngine::new(config.streaming_config.clone()));
    let cache_manager = Arc::new(CacheManager::new(config.cache_config.clone()));
    let replication_manager = Arc::new(ReplicationManager::new(config.replication_config.clone()));
    let consistency_manager = Arc::new(ConsistencyManager::new(config.consistency_level));
    let change_log = Arc::new(ChangeLog::new(10000));
    let metrics = Arc::new(StoreMetrics::default());

    Ok(IntegratedVectorStore {
        config,
        vector_store,
        rdf_integration,
        sparql_service,
        transaction_manager,
        streaming_engine,
        cache_manager,
        replication_manager,
        consistency_manager,
        change_log,
        metrics,
    })
}

impl Default for WriteAheadLog {
    fn default() -> Self {
        Self::new()
    }
}

impl WriteAheadLog {
    /// Create an in-memory-only WAL: entries are kept in `log_entries` for
    /// fast reads but are **not** durable — a process crash loses every
    /// buffered entry. Use [`WriteAheadLog::with_file`] for real
    /// append-only file durability.
    pub fn new() -> Self {
        Self {
            log_entries: Arc::new(RwLock::new(std::collections::VecDeque::new())),
            log_file: None,
            file_handle: None,
            checkpoint_interval: Duration::from_secs(60),
            last_checkpoint: Arc::new(RwLock::new(SystemTime::now())),
        }
    }

    /// Create a durable, file-backed WAL. Every [`WriteAheadLog::append`]
    /// writes a length-prefixed, checksummed record to `path` and calls
    /// `sync_data()` (fsync) before returning, so a crash immediately after
    /// `append()` returns cannot lose the entry. The file is opened in
    /// append mode and created if it does not already exist.
    pub fn with_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("Failed to open WAL file: {}", path.display()))?;

        Ok(Self {
            log_entries: Arc::new(RwLock::new(std::collections::VecDeque::new())),
            log_file: Some(path),
            file_handle: Some(Arc::new(parking_lot::Mutex::new(file))),
            checkpoint_interval: Duration::from_secs(60),
            last_checkpoint: Arc::new(RwLock::new(SystemTime::now())),
        })
    }

    pub fn append(
        &self,
        transaction_id: TransactionId,
        operation: SerializableOperation,
    ) -> Result<()> {
        let mut entry = LogEntry {
            lsn: self.generate_lsn(),
            transaction_id,
            operation,
            timestamp: SystemTime::now(),
            checksum: 0,
        };
        entry.checksum = Self::compute_checksum(&entry);

        if let Some(ref file_handle) = self.file_handle {
            Self::write_entry_to_file(file_handle, &entry)?;
        }

        let mut log = self.log_entries.write();
        log.push_back(entry);
        drop(log);

        if self.should_checkpoint() {
            self.checkpoint()?;
        }

        Ok(())
    }

    /// Real integrity checksum for a log entry (BLAKE3 over the LSN,
    /// transaction ID, serialized operation payload, and timestamp,
    /// truncated to 64 bits) — replaces the previous hardcoded `0`, which
    /// meant corruption could never be detected.
    fn compute_checksum(entry: &LogEntry) -> u64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&entry.lsn.to_le_bytes());
        hasher.update(&entry.transaction_id.to_le_bytes());
        if let Ok(bytes) =
            oxicode::serde::encode_to_vec(&entry.operation, oxicode::config::standard())
        {
            hasher.update(&bytes);
        }
        if let Ok(duration) = entry.timestamp.duration_since(std::time::UNIX_EPOCH) {
            hasher.update(&duration.as_nanos().to_le_bytes());
        }
        let hash = hasher.finalize();
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&hash.as_bytes()[0..8]);
        u64::from_le_bytes(buf)
    }

    /// Append one length-prefixed, oxicode-serialized record to the durable
    /// WAL file and fsync it (`sync_data`) before returning.
    fn write_entry_to_file(
        file_handle: &Arc<parking_lot::Mutex<std::fs::File>>,
        entry: &LogEntry,
    ) -> Result<()> {
        let bytes = oxicode::serde::encode_to_vec(entry, oxicode::config::standard())
            .map_err(|e| anyhow::anyhow!("Failed to serialize WAL entry: {e}"))?;

        let mut file = file_handle.lock();
        file.write_all(&(bytes.len() as u32).to_le_bytes())?;
        file.write_all(&bytes)?;
        file.flush()?;
        file.sync_data()
            .with_context(|| "Failed to fsync WAL file after append")?;
        Ok(())
    }

    /// Replay all durably-written entries from the WAL file, in append
    /// order (for crash recovery). Returns an empty list for in-memory-only
    /// WALs (no `log_file` configured).
    pub fn replay_from_file(&self) -> Result<Vec<LogEntry>> {
        let Some(ref path) = self.log_file else {
            return Ok(Vec::new());
        };

        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open WAL file for replay: {}", path.display()))?;
        let mut entries = Vec::new();

        loop {
            let mut len_bytes = [0u8; 4];
            match file.read_exact(&mut len_bytes) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u32::from_le_bytes(len_bytes) as usize;
            let mut buf = vec![0u8; len];
            file.read_exact(&mut buf)
                .with_context(|| "Truncated WAL record: expected more bytes than file contained")?;
            let (entry, _): (LogEntry, usize) =
                oxicode::serde::decode_from_slice(&buf, oxicode::config::standard())
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize WAL entry: {e}"))?;
            entries.push(entry);
        }

        Ok(entries)
    }

    fn generate_lsn(&self) -> u64 {
        static LSN_COUNTER: AtomicU64 = AtomicU64::new(0);
        LSN_COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    fn should_checkpoint(&self) -> bool {
        let last_checkpoint = *self.last_checkpoint.read();
        last_checkpoint
            .elapsed()
            .expect("SystemTime should not go backwards")
            > self.checkpoint_interval
    }

    /// Update the checkpoint timestamp; also fsyncs the durable WAL file
    /// (if configured) so a checkpoint boundary is a real durability
    /// guarantee, not just a timestamp update.
    fn checkpoint(&self) -> Result<()> {
        if let Some(ref file_handle) = self.file_handle {
            file_handle
                .lock()
                .sync_all()
                .with_context(|| "Failed to fsync WAL file at checkpoint")?;
        }
        let mut last_checkpoint = self.last_checkpoint.write();
        *last_checkpoint = SystemTime::now();
        Ok(())
    }
}

impl Default for LockManager {
    fn default() -> Self {
        Self::new()
    }
}

impl LockManager {
    pub fn new() -> Self {
        Self {
            locks: Arc::new(RwLock::new(HashMap::new())),
            deadlock_detector: Arc::new(DeadlockDetector::new()),
        }
    }

    pub fn acquire_lock(
        &self,
        transaction_id: TransactionId,
        resource: &str,
        lock_type: LockType,
    ) -> Result<()> {
        // Holders blocking this request, captured while `locks` is held so
        // the lock table isn't held across the deadlock check below.
        let blocking_holders = {
            let mut locks = self.locks.write();
            let lock_info = locks
                .entry(resource.to_string())
                .or_insert_with(|| LockInfo {
                    lock_type: LockType::Shared,
                    holders: std::collections::HashSet::new(),
                    waiters: std::collections::VecDeque::new(),
                    granted_time: SystemTime::now(),
                });

            if self.can_grant_lock(lock_info, lock_type) {
                lock_info.holders.insert(transaction_id);
                lock_info.lock_type = lock_type;
                lock_info.granted_time = SystemTime::now();
                None
            } else {
                lock_info.waiters.push_back((transaction_id, lock_type));
                Some(lock_info.holders.clone())
            }
        };

        match blocking_holders {
            None => {
                // Lock granted: this transaction is no longer waiting on
                // anyone, so drop its stale wait-for edges (if any).
                self.deadlock_detector.clear_waits(transaction_id);
                Ok(())
            }
            Some(holders) => {
                self.deadlock_detector.record_wait(transaction_id, holders);
                self.deadlock_detector.check_deadlock(transaction_id)?;
                Err(anyhow::anyhow!("Lock not available, transaction waiting"))
            }
        }
    }

    pub fn release_transaction_locks(&self, transaction_id: TransactionId) {
        self.deadlock_detector.clear_waits(transaction_id);

        let mut locks = self.locks.write();
        let mut to_remove = Vec::new();

        for (resource, lock_info) in locks.iter_mut() {
            lock_info.holders.remove(&transaction_id);
            lock_info.waiters.retain(|(tid, _)| *tid != transaction_id);

            if lock_info.holders.is_empty() {
                to_remove.push(resource.clone());
            }
        }

        for resource in to_remove {
            locks.remove(&resource);
        }
    }

    fn can_grant_lock(&self, lock_info: &LockInfo, requested_type: LockType) -> bool {
        if lock_info.holders.is_empty() {
            return true;
        }

        matches!(
            (lock_info.lock_type, requested_type),
            (LockType::Shared, LockType::Shared)
        )
    }
}

impl Default for DeadlockDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DeadlockDetector {
    pub fn new() -> Self {
        Self {
            wait_for_graph: Arc::new(RwLock::new(HashMap::new())),
            detection_interval: Duration::from_secs(1),
        }
    }

    /// Record that `waiter` is now blocked on each transaction in `holders`
    /// (i.e. add `waiter -> holder` edges to the wait-for graph).
    pub fn record_wait(
        &self,
        waiter: TransactionId,
        holders: std::collections::HashSet<TransactionId>,
    ) {
        if holders.is_empty() {
            return;
        }
        let mut graph = self.wait_for_graph.write();
        graph.entry(waiter).or_default().extend(holders);
    }

    /// Remove all wait-for edges originating from `transaction_id` (call
    /// once it stops waiting: its lock request was granted, it committed,
    /// or it was aborted). Without this, resolved waits would linger in the
    /// graph forever and eventually produce false-positive cycles.
    pub fn clear_waits(&self, transaction_id: TransactionId) {
        let mut graph = self.wait_for_graph.write();
        graph.remove(&transaction_id);
    }

    /// Detect whether `transaction_id` participates in a wait-for cycle
    /// (deadlock): a real depth-first search over `wait_for_graph` for a
    /// path from `transaction_id` back to itself, replacing the previous
    /// unconditional `Ok(())` that never actually inspected the graph.
    pub fn check_deadlock(&self, transaction_id: TransactionId) -> Result<()> {
        let graph = self.wait_for_graph.read();

        if Self::reaches_target(&graph, transaction_id, transaction_id, &mut HashSet::new()) {
            return Err(anyhow::anyhow!(
                "Deadlock detected: transaction {} is part of a circular wait-for chain",
                transaction_id
            ));
        }

        Ok(())
    }

    /// DFS helper: does any path starting at `current`'s direct successors
    /// eventually reach `target`?
    fn reaches_target(
        graph: &HashMap<TransactionId, std::collections::HashSet<TransactionId>>,
        current: TransactionId,
        target: TransactionId,
        visited: &mut HashSet<TransactionId>,
    ) -> bool {
        let Some(successors) = graph.get(&current) else {
            return false;
        };

        for &next in successors {
            if next == target {
                return true;
            }
            if visited.insert(next) && Self::reaches_target(graph, next, target, visited) {
                return true;
            }
        }

        false
    }
}

impl StreamingEngine {
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            stream_buffer: Arc::new(RwLock::new(std::collections::VecDeque::new())),
            processor_thread: None,
            backpressure_controller: Arc::new(BackpressureController::new()),
            stream_metrics: Arc::new(StreamingMetrics::default()),
        }
    }

    pub fn submit_operation(&self, operation: StreamingOperation) -> Result<()> {
        if self.backpressure_controller.should_apply_backpressure() {
            return Err(anyhow::anyhow!("Backpressure applied, operation rejected"));
        }

        let mut buffer = self.stream_buffer.write();
        buffer.push_back(operation);

        self.stream_metrics
            .operations_pending
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn get_metrics(&self) -> &StreamingMetrics {
        &self.stream_metrics
    }
}

impl Default for BackpressureController {
    fn default() -> Self {
        Self::new()
    }
}

impl BackpressureController {
    pub fn new() -> Self {
        Self {
            current_load: Arc::new(RwLock::new(0.0)),
            max_load_threshold: 0.8,
            adaptive_batching: true,
            load_shedding: true,
        }
    }

    pub fn should_apply_backpressure(&self) -> bool {
        let load = *self.current_load.read();
        load > self.max_load_threshold
    }
}

impl CacheManager {
    pub fn new(config: StoreCacheConfig) -> Self {
        Self {
            vector_cache: Arc::new(RwLock::new(HashMap::new())),
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            cache_stats: Arc::new(CacheStats::default()),
            eviction_policy: EvictionPolicy::LRU,
        }
    }

    pub fn get_vector(&self, uri: &str) -> Option<CachedVector> {
        let cache = self.vector_cache.read();
        if let Some(cached) = cache.get(uri) {
            self.cache_stats
                .vector_cache_hits
                .fetch_add(1, Ordering::Relaxed);
            Some(cached.clone())
        } else {
            self.cache_stats
                .vector_cache_misses
                .fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn cache_vector(&self, uri: String, vector: crate::Vector) {
        let cached_vector = CachedVector {
            vector,
            last_accessed: SystemTime::now(),
            access_count: 1,
            compression_ratio: 1.0,
            cache_level: CacheLevel::Memory,
        };

        let mut cache = self.vector_cache.write();
        cache.insert(uri, cached_vector);
    }

    pub fn get_cached_query(&self, query_hash: &u64) -> Option<CachedQueryResult> {
        let cache = self.query_cache.read();
        if let Some(cached) = cache.get(&query_hash.to_string()) {
            self.cache_stats
                .query_cache_hits
                .fetch_add(1, Ordering::Relaxed);
            Some(cached.clone())
        } else {
            self.cache_stats
                .query_cache_misses
                .fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn cache_query_result(&self, query_hash: u64, results: Vec<(String, f32)>) {
        let cached_result = CachedQueryResult {
            results,
            query_hash,
            last_accessed: SystemTime::now(),
            ttl: self.config.cache_ttl,
            hit_count: 0,
        };

        let mut cache = self.query_cache.write();
        cache.insert(query_hash.to_string(), cached_result);
    }

    pub fn get_stats(&self) -> &CacheStats {
        &self.cache_stats
    }
}

impl ReplicationManager {
    pub fn new(config: ReplicationConfig) -> Self {
        Self {
            config,
            replicas: Arc::new(RwLock::new(Vec::new())),
            replication_log: Arc::new(RwLock::new(std::collections::VecDeque::new())),
            consensus_algorithm: ConsensusAlgorithm::SimpleMajority,
            health_checker: Arc::new(HealthChecker::new()),
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
        }
    }
}

impl ConsistencyManager {
    pub fn new(consistency_level: ConsistencyLevel) -> Self {
        Self {
            consistency_level,
            vector_clocks: Arc::new(RwLock::new(HashMap::new())),
            conflict_resolver: Arc::new(ConflictResolver::new()),
            causal_order_tracker: Arc::new(CausalOrderTracker::new()),
        }
    }
}

impl Default for ConflictResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ConflictResolver {
    pub fn new() -> Self {
        Self {
            strategy: ConflictResolution::LastWriteWins,
            custom_resolvers: HashMap::new(),
        }
    }
}

impl Default for CausalOrderTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalOrderTracker {
    pub fn new() -> Self {
        Self {
            happens_before: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl ChangeLog {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(std::collections::VecDeque::new())),
            max_entries,
            subscribers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn add_entry(&self, entry: ChangeLogEntry) {
        let mut entries = self.entries.write();
        entries.push_back(entry.clone());

        if entries.len() > self.max_entries {
            entries.pop_front();
        }

        let subscribers = self.subscribers.read();
        for subscriber in subscribers.iter() {
            let _ = subscriber.on_change(&entry);
        }
    }

    pub fn add_subscriber(&self, subscriber: Arc<dyn ChangeSubscriber>) {
        let mut subscribers = self.subscribers.write();
        subscribers.push(subscriber);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the P1 finding: `WriteAheadLog` used to be
    /// purely in-memory (never wrote to disk) with a hardcoded `checksum: 0`
    /// on every entry. `with_file` must now durably persist entries with a
    /// real, non-zero checksum, and `replay_from_file` must recover them.
    #[test]
    fn test_write_ahead_log_file_backed_round_trip() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("oxirs_vec_wal_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir)?;
        let wal_path = dir.join("wal.log");

        let wal = WriteAheadLog::with_file(&wal_path)?;
        wal.append(
            1,
            SerializableOperation::Insert {
                uri: "http://example.org/a".to_string(),
                vector_data: vec![1.0, 2.0, 3.0],
            },
        )?;
        wal.append(1, SerializableOperation::Commit { transaction_id: 1 })?;

        // The in-memory ring must reflect real (non-zero) checksums.
        let in_memory: Vec<LogEntry> = wal.log_entries.read().iter().cloned().collect();
        assert_eq!(in_memory.len(), 2);
        assert!(
            in_memory.iter().all(|e| e.checksum != 0),
            "checksum must be a real computed value, not the hardcoded 0"
        );

        // A fresh read from disk (simulating post-crash recovery) must
        // recover the same entries with matching checksums.
        let replayed = wal.replay_from_file()?;
        assert_eq!(replayed.len(), 2);
        assert_eq!(replayed[0].transaction_id, 1);
        assert_eq!(replayed[0].checksum, in_memory[0].checksum);
        assert_eq!(replayed[1].checksum, in_memory[1].checksum);

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// The in-memory-only constructor must remain honest about not being
    /// durable: `replay_from_file` returns an empty list rather than
    /// pretending to recover anything.
    #[test]
    fn test_write_ahead_log_in_memory_only_has_no_replay() -> Result<()> {
        let wal = WriteAheadLog::new();
        wal.append(
            1,
            SerializableOperation::Delete {
                uri: "http://example.org/a".to_string(),
            },
        )?;
        assert!(wal.replay_from_file()?.is_empty());
        Ok(())
    }

    /// Regression test for the P2 finding: `DeadlockDetector::check_deadlock`
    /// used to always return `Ok(())` regardless of the wait-for graph. A
    /// genuine circular wait (tx1 holds A, wants B; tx2 holds B, wants A)
    /// must now be detected via `LockManager::acquire_lock`.
    #[test]
    fn test_lock_manager_detects_real_deadlock() {
        let manager = LockManager::new();

        // tx1 holds A, tx2 holds B.
        manager
            .acquire_lock(1, "A", LockType::Exclusive)
            .expect("tx1 should acquire A");
        manager
            .acquire_lock(2, "B", LockType::Exclusive)
            .expect("tx2 should acquire B");

        // tx1 now wants B (held by tx2): blocked, but not yet a cycle.
        let tx1_wants_b = manager.acquire_lock(1, "B", LockType::Exclusive);
        assert!(tx1_wants_b.is_err());

        // tx2 now wants A (held by tx1): this closes the cycle
        // 1 -> 2 -> 1, so it must fail with a deadlock error specifically
        // (not just the generic "lock not available" contention error).
        let tx2_wants_a = manager.acquire_lock(2, "A", LockType::Exclusive);
        let err = tx2_wants_a.expect_err("circular wait must be detected as a deadlock");
        assert!(
            err.to_string().contains("Deadlock"),
            "expected a deadlock-specific error, got: {err}"
        );
    }

    /// Ordinary (non-circular) lock contention must NOT be misreported as a
    /// deadlock.
    #[test]
    fn test_lock_manager_no_false_positive_deadlock() {
        let manager = LockManager::new();
        manager
            .acquire_lock(1, "A", LockType::Exclusive)
            .expect("tx1 should acquire A");

        // tx2 waits on tx1 for A; there is no cycle (tx1 isn't waiting on
        // anything), so this must be ordinary contention, not a deadlock.
        let err = manager
            .acquire_lock(2, "A", LockType::Exclusive)
            .expect_err("tx2 should be blocked");
        assert!(!err.to_string().contains("Deadlock"));
    }
}
