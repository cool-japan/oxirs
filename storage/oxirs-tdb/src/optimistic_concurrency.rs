//! # Optimistic Concurrency Control with Advanced Validation
//!
//! High-performance optimistic concurrency control implementation with comprehensive
//! validation phases, conflict detection, and automatic retry mechanisms for TDB storage.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Transaction phase tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionPhase {
    /// Transaction has begun and is executing
    Active,
    /// Transaction is in validation phase
    Validating,
    /// Transaction has passed validation and is committing
    Committing,
    /// Transaction has been committed successfully
    Committed,
    /// Transaction has been aborted
    Aborted,
}

/// Conflict type classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictType {
    /// Read-Write conflict (phantom read)
    ReadWrite,
    /// Write-Write conflict (lost update)
    WriteWrite,
    /// Write-Read conflict (dirty read)
    WriteRead,
    /// Timestamp ordering violation
    TimestampViolation,
    /// Serializability violation
    SerializabilityViolation,
}

/// Validation result with detailed conflict information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationResult {
    /// Transaction is valid and can commit
    Valid,
    /// Conflict detected with detailed information
    Conflict {
        conflict_type: ConflictType,
        conflicting_transaction: u64,
        conflicting_key: String,
        reason: String,
    },
    /// Validation timeout
    Timeout,
    /// Internal validation error
    Error(String),
}

/// Read set entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadEntry {
    /// Key that was read
    pub key: String,
    /// Version that was read
    pub version: u64,
    /// Timestamp when read occurred
    pub read_timestamp: SystemTime,
    /// Predicate information for phantom read detection
    pub predicate: Option<String>,
}

/// Write set entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteEntry {
    /// Key that was written
    pub key: String,
    /// New version being written
    pub version: u64,
    /// Old version being replaced (if any)
    pub old_version: Option<u64>,
    /// Timestamp when write occurred
    pub write_timestamp: SystemTime,
    /// Write operation type
    pub operation: WriteOperation,
}

/// Write operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WriteOperation {
    Insert,
    Update,
    Delete,
}

/// Comprehensive transaction metadata
#[derive(Debug, Clone)]
pub struct TransactionInfo {
    /// Transaction identifier
    pub id: u64,
    /// Current phase
    pub phase: TransactionPhase,
    /// Transaction start time
    pub start_time: SystemTime,
    /// Transaction start version
    pub start_version: u64,
    /// Read set with detailed metadata
    pub read_set: Vec<ReadEntry>,
    /// Write set with detailed metadata
    pub write_set: Vec<WriteEntry>,
    /// Priority for conflict resolution
    pub priority: i32,
    /// Retry count
    pub retry_count: u32,
    /// Validation attempts
    pub validation_attempts: u32,
}

impl TransactionInfo {
    pub fn new(id: u64, start_version: u64) -> Self {
        Self {
            id,
            phase: TransactionPhase::Active,
            start_time: SystemTime::now(),
            start_version,
            read_set: Vec::new(),
            write_set: Vec::new(),
            priority: 0,
            retry_count: 0,
            validation_attempts: 0,
        }
    }

    /// Add read operation to transaction
    pub fn add_read(&mut self, key: String, version: u64, predicate: Option<String>) {
        self.read_set.push(ReadEntry {
            key,
            version,
            read_timestamp: SystemTime::now(),
            predicate,
        });
    }

    /// Add write operation to transaction
    pub fn add_write(&mut self, key: String, version: u64, old_version: Option<u64>, operation: WriteOperation) {
        self.write_set.push(WriteEntry {
            key,
            version,
            old_version,
            write_timestamp: SystemTime::now(),
            operation,
        });
    }

    /// Get all keys accessed by this transaction
    pub fn get_accessed_keys(&self) -> HashSet<String> {
        let mut keys = HashSet::new();
        for read in &self.read_set {
            keys.insert(read.key.clone());
        }
        for write in &self.write_set {
            keys.insert(write.key.clone());
        }
        keys
    }

    /// Get transaction duration
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed().unwrap_or_default()
    }
}

/// Version vector for distributed timestamp ordering
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionVector {
    /// Node timestamps
    pub timestamps: HashMap<u64, u64>,
}

impl VersionVector {
    pub fn new() -> Self {
        Self {
            timestamps: HashMap::new(),
        }
    }

    /// Increment timestamp for a node
    pub fn increment(&mut self, node_id: u64) {
        *self.timestamps.entry(node_id).or_insert(0) += 1;
    }

    /// Update with another version vector
    pub fn update(&mut self, other: &VersionVector) {
        for (&node_id, &timestamp) in &other.timestamps {
            let current = self.timestamps.entry(node_id).or_insert(0);
            *current = (*current).max(timestamp);
        }
    }

    /// Check if this version vector happens before another
    pub fn happens_before(&self, other: &VersionVector) -> bool {
        let mut all_leq = true;
        let mut some_less = false;

        for (&node_id, &other_time) in &other.timestamps {
            let self_time = self.timestamps.get(&node_id).unwrap_or(&0);
            if self_time > &other_time {
                all_leq = false;
                break;
            }
            if self_time < &other_time {
                some_less = true;
            }
        }

        all_leq && some_less
    }
}

/// Configuration for optimistic concurrency control
#[derive(Debug, Clone)]
pub struct OptimisticConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Base backoff time in milliseconds
    pub base_backoff_ms: u64,
    /// Maximum backoff time in milliseconds
    pub max_backoff_ms: u64,
    /// Validation timeout in milliseconds
    pub validation_timeout_ms: u64,
    /// Maximum number of validation attempts
    pub max_validation_attempts: u32,
    /// Enable advanced conflict detection
    pub enable_advanced_conflict_detection: bool,
    /// Enable phantom read detection
    pub enable_phantom_read_detection: bool,
    /// Enable priority-based conflict resolution
    pub enable_priority_conflict_resolution: bool,
    /// History size for validation
    pub validation_history_size: usize,
}

impl Default for OptimisticConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            base_backoff_ms: 1,
            max_backoff_ms: 1000,
            validation_timeout_ms: 5000,
            max_validation_attempts: 3,
            enable_advanced_conflict_detection: true,
            enable_phantom_read_detection: true,
            enable_priority_conflict_resolution: true,
            validation_history_size: 10000,
        }
    }
}

/// Statistics for optimistic concurrency control
#[derive(Debug, Clone, Default, Serialize)]
pub struct OptimisticStats {
    pub total_transactions: u64,
    pub committed_transactions: u64,
    pub aborted_transactions: u64,
    pub retry_count: u64,
    pub validation_failures: u64,
    pub conflicts_by_type: HashMap<ConflictType, u64>,
    pub avg_validation_time_ms: f64,
    pub max_validation_time_ms: u64,
    pub current_active_transactions: usize,
    pub current_validating_transactions: usize,
}

/// Advanced optimistic concurrency control manager
pub struct OptimisticConcurrencyController {
    /// Active transactions
    active_transactions: Arc<RwLock<HashMap<u64, TransactionInfo>>>,
    /// Transaction counter
    transaction_counter: Arc<Mutex<u64>>,
    /// Version counter
    version_counter: Arc<Mutex<u64>>,
    /// Committed transaction history for validation
    commit_history: Arc<RwLock<VecDeque<TransactionInfo>>>,
    /// Configuration
    config: OptimisticConfig,
    /// Statistics
    stats: Arc<Mutex<OptimisticStats>>,
    /// Global version vector
    version_vector: Arc<Mutex<VersionVector>>,
    /// Node identifier
    node_id: u64,
}

impl OptimisticConcurrencyController {
    /// Create new optimistic concurrency controller
    pub fn new(node_id: u64) -> Self {
        Self::with_config(node_id, OptimisticConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(node_id: u64, config: OptimisticConfig) -> Self {
        Self {
            active_transactions: Arc::new(RwLock::new(HashMap::new())),
            transaction_counter: Arc::new(Mutex::new(0)),
            version_counter: Arc::new(Mutex::new(0)),
            commit_history: Arc::new(RwLock::new(VecDeque::new())),
            config,
            stats: Arc::new(Mutex::new(OptimisticStats::default())),
            version_vector: Arc::new(Mutex::new(VersionVector::new())),
            node_id,
        }
    }

    /// Begin new transaction
    pub fn begin_transaction(&self) -> Result<u64> {
        let mut counter = self.transaction_counter.lock()
            .map_err(|_| anyhow!("Failed to acquire transaction counter lock"))?;
        *counter += 1;
        let tx_id = *counter;

        let version = *self.version_counter.lock()
            .map_err(|_| anyhow!("Failed to acquire version counter lock"))?;

        let tx_info = TransactionInfo::new(tx_id, version);

        {
            let mut active = self.active_transactions.write()
                .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
            active.insert(tx_id, tx_info);
        }

        // Update version vector
        {
            let mut vv = self.version_vector.lock()
                .map_err(|_| anyhow!("Failed to acquire version vector lock"))?;
            vv.increment(self.node_id);
        }

        // Update stats
        {
            let mut stats = self.stats.lock()
                .map_err(|_| anyhow!("Failed to acquire stats lock"))?;
            stats.total_transactions += 1;
            stats.current_active_transactions += 1;
        }

        Ok(tx_id)
    }

    /// Record read operation
    pub fn record_read(&self, tx_id: u64, key: String, version: u64, predicate: Option<String>) -> Result<()> {
        let mut active = self.active_transactions.write()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
        
        if let Some(tx_info) = active.get_mut(&tx_id) {
            tx_info.add_read(key, version, predicate);
            Ok(())
        } else {
            Err(anyhow!("Transaction {} not found", tx_id))
        }
    }

    /// Record write operation
    pub fn record_write(&self, tx_id: u64, key: String, version: u64, old_version: Option<u64>, operation: WriteOperation) -> Result<()> {
        let mut active = self.active_transactions.write()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
        
        if let Some(tx_info) = active.get_mut(&tx_id) {
            tx_info.add_write(key, version, old_version, operation);
            Ok(())
        } else {
            Err(anyhow!("Transaction {} not found", tx_id))
        }
    }

    /// Validate transaction for commit
    pub fn validate_transaction(&self, tx_id: u64) -> Result<ValidationResult> {
        let validation_start = Instant::now();

        // Update transaction phase
        {
            let mut active = self.active_transactions.write()
                .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
            if let Some(tx_info) = active.get_mut(&tx_id) {
                tx_info.phase = TransactionPhase::Validating;
                tx_info.validation_attempts += 1;

                // Update stats
                {
                    let active_count = active.values().filter(|tx| tx.phase == TransactionPhase::Active).count();
                    let validating_count = active.values().filter(|tx| tx.phase == TransactionPhase::Validating).count();
                    
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.current_active_transactions = active_count;
                        stats.current_validating_transactions = validating_count;
                    }
                }
            } else {
                return Err(anyhow!("Transaction {} not found", tx_id));
            }
        }

        // Perform comprehensive validation
        let result = self.perform_validation_phases(tx_id)?;

        // Record validation time
        let validation_time = validation_start.elapsed();
        {
            if let Ok(mut stats) = self.stats.lock() {
                let validation_ms = validation_time.as_millis() as u64;
                stats.max_validation_time_ms = stats.max_validation_time_ms.max(validation_ms);
                
                // Update average validation time
                let total_validations = stats.validation_failures + stats.committed_transactions;
                if total_validations > 0 {
                    stats.avg_validation_time_ms = (stats.avg_validation_time_ms * (total_validations - 1) as f64 + validation_ms as f64) / total_validations as f64;
                }

                // Update conflict statistics
                if let ValidationResult::Conflict { conflict_type, .. } = &result {
                    *stats.conflicts_by_type.entry(*conflict_type).or_insert(0) += 1;
                    stats.validation_failures += 1;
                }
            }
        }

        Ok(result)
    }

    /// Perform comprehensive validation phases
    fn perform_validation_phases(&self, tx_id: u64) -> Result<ValidationResult> {
        // Phase 1: Basic conflict detection
        if let Some(conflict) = self.detect_basic_conflicts(tx_id)? {
            return Ok(conflict);
        }

        // Phase 2: Advanced conflict detection (if enabled)
        if self.config.enable_advanced_conflict_detection {
            if let Some(conflict) = self.detect_advanced_conflicts(tx_id)? {
                return Ok(conflict);
            }
        }

        // Phase 3: Phantom read detection (if enabled)
        if self.config.enable_phantom_read_detection {
            if let Some(conflict) = self.detect_phantom_reads(tx_id)? {
                return Ok(conflict);
            }
        }

        // Phase 4: Serializability validation
        if let Some(conflict) = self.validate_serializability(tx_id)? {
            return Ok(conflict);
        }

        // Phase 5: Timestamp ordering validation
        if let Some(conflict) = self.validate_timestamp_ordering(tx_id)? {
            return Ok(conflict);
        }

        Ok(ValidationResult::Valid)
    }

    /// Detect basic read-write and write-write conflicts
    fn detect_basic_conflicts(&self, tx_id: u64) -> Result<Option<ValidationResult>> {
        let active = self.active_transactions.read()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
        let history = self.commit_history.read()
            .map_err(|_| anyhow!("Failed to acquire commit history lock"))?;

        let tx_info = active.get(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        // Check conflicts with committed transactions
        for committed_tx in history.iter() {
            if committed_tx.start_time > tx_info.start_time {
                // Check read-write conflicts
                for read_entry in &tx_info.read_set {
                    for write_entry in &committed_tx.write_set {
                        if read_entry.key == write_entry.key {
                            return Ok(Some(ValidationResult::Conflict {
                                conflict_type: ConflictType::ReadWrite,
                                conflicting_transaction: committed_tx.id,
                                conflicting_key: read_entry.key.clone(),
                                reason: format!("Read key '{}' at version {} conflicts with committed write at version {}", 
                                    read_entry.key, read_entry.version, write_entry.version),
                            }));
                        }
                    }
                }

                // Check write-write conflicts
                for write_entry in &tx_info.write_set {
                    for committed_write in &committed_tx.write_set {
                        if write_entry.key == committed_write.key {
                            return Ok(Some(ValidationResult::Conflict {
                                conflict_type: ConflictType::WriteWrite,
                                conflicting_transaction: committed_tx.id,
                                conflicting_key: write_entry.key.clone(),
                                reason: format!("Write key '{}' conflicts with committed transaction {}", 
                                    write_entry.key, committed_tx.id),
                            }));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Detect advanced conflicts using dependency analysis
    fn detect_advanced_conflicts(&self, tx_id: u64) -> Result<Option<ValidationResult>> {
        let active = self.active_transactions.read()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;

        let tx_info = active.get(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        // Check for conflicts with other active transactions
        for (other_tx_id, other_tx) in active.iter() {
            if *other_tx_id == tx_id || other_tx.phase == TransactionPhase::Aborted {
                continue;
            }

            // Priority-based conflict resolution
            if self.config.enable_priority_conflict_resolution {
                if tx_info.priority < other_tx.priority {
                    // Check if there's any overlap in accessed keys
                    let tx_keys = tx_info.get_accessed_keys();
                    let other_keys = other_tx.get_accessed_keys();
                    
                    if !tx_keys.is_disjoint(&other_keys) {
                        return Ok(Some(ValidationResult::Conflict {
                            conflict_type: ConflictType::SerializabilityViolation,
                            conflicting_transaction: *other_tx_id,
                            conflicting_key: tx_keys.intersection(&other_keys).next().unwrap().clone(),
                            reason: format!("Priority conflict with transaction {} (priority {} vs {})", 
                                other_tx_id, tx_info.priority, other_tx.priority),
                        }));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Detect phantom reads using predicate analysis
    fn detect_phantom_reads(&self, tx_id: u64) -> Result<Option<ValidationResult>> {
        let active = self.active_transactions.read()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
        let history = self.commit_history.read()
            .map_err(|_| anyhow!("Failed to acquire commit history lock"))?;

        let tx_info = active.get(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        // Check for phantom reads with committed transactions
        for committed_tx in history.iter() {
            if committed_tx.start_time > tx_info.start_time {
                for read_entry in &tx_info.read_set {
                    if let Some(predicate) = &read_entry.predicate {
                        for write_entry in &committed_tx.write_set {
                            if self.predicate_matches_write(predicate, write_entry) {
                                return Ok(Some(ValidationResult::Conflict {
                                    conflict_type: ConflictType::ReadWrite,
                                    conflicting_transaction: committed_tx.id,
                                    conflicting_key: write_entry.key.clone(),
                                    reason: format!("Phantom read detected: predicate '{}' matches new insert '{}'", 
                                        predicate, write_entry.key),
                                }));
                            }
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Check if a predicate matches a write operation (simplified implementation)
    fn predicate_matches_write(&self, predicate: &str, write_entry: &WriteEntry) -> bool {
        // This is a simplified predicate matching - in a real implementation,
        // this would involve sophisticated predicate analysis
        predicate.contains(&write_entry.key) || write_entry.key.contains(predicate)
    }

    /// Validate serializability using conflict graph analysis
    fn validate_serializability(&self, tx_id: u64) -> Result<Option<ValidationResult>> {
        // This is a simplified serializability check
        // A full implementation would build and analyze the conflict graph
        
        let active = self.active_transactions.read()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
        let history = self.commit_history.read()
            .map_err(|_| anyhow!("Failed to acquire commit history lock"))?;

        let tx_info = active.get(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        // Simple cycle detection in dependency graph
        let mut dependencies = HashMap::new();
        
        for committed_tx in history.iter() {
            if committed_tx.start_time > tx_info.start_time {
                // Build dependency relationships
                for read_entry in &tx_info.read_set {
                    for write_entry in &committed_tx.write_set {
                        if read_entry.key == write_entry.key {
                            dependencies.insert(committed_tx.id, tx_id);
                        }
                    }
                }
            }
        }

        // Check for potential cycles (simplified)
        if dependencies.len() > 2 {
            return Ok(Some(ValidationResult::Conflict {
                conflict_type: ConflictType::SerializabilityViolation,
                conflicting_transaction: dependencies.keys().next().copied().unwrap_or(0),
                conflicting_key: "multiple".to_string(),
                reason: "Potential serializability violation detected".to_string(),
            }));
        }

        Ok(None)
    }

    /// Validate timestamp ordering
    fn validate_timestamp_ordering(&self, tx_id: u64) -> Result<Option<ValidationResult>> {
        let active = self.active_transactions.read()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;

        let tx_info = active.get(&tx_id)
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;

        // Check timestamp ordering constraints
        let current_version = *self.version_counter.lock()
            .map_err(|_| anyhow!("Failed to acquire version counter lock"))?;

        if tx_info.start_version > current_version {
            return Ok(Some(ValidationResult::Conflict {
                conflict_type: ConflictType::TimestampViolation,
                conflicting_transaction: 0,
                conflicting_key: "timestamp".to_string(),
                reason: format!("Timestamp ordering violation: start version {} > current version {}", 
                    tx_info.start_version, current_version),
            }));
        }

        Ok(None)
    }

    /// Commit validated transaction
    pub fn commit_transaction(&self, tx_id: u64) -> Result<u64> {
        let mut version_counter = self.version_counter.lock()
            .map_err(|_| anyhow!("Failed to acquire version counter lock"))?;
        *version_counter += 1;
        let commit_version = *version_counter;

        // Move transaction from active to history
        let tx_info = {
            let mut active = self.active_transactions.write()
                .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
            let mut tx_info = active.remove(&tx_id)
                .ok_or_else(|| anyhow!("Transaction {} not found", tx_id))?;
            tx_info.phase = TransactionPhase::Committed;
            tx_info
        };

        // Add to commit history
        {
            let mut history = self.commit_history.write()
                .map_err(|_| anyhow!("Failed to acquire commit history lock"))?;
            history.push_back(tx_info);

            // Limit history size
            while history.len() > self.config.validation_history_size {
                history.pop_front();
            }
        }

        // Update version vector
        {
            let mut vv = self.version_vector.lock()
                .map_err(|_| anyhow!("Failed to acquire version vector lock"))?;
            vv.increment(self.node_id);
        }

        // Update stats
        {
            let mut stats = self.stats.lock()
                .map_err(|_| anyhow!("Failed to acquire stats lock"))?;
            stats.committed_transactions += 1;
            stats.current_active_transactions = stats.current_active_transactions.saturating_sub(1);
        }

        Ok(commit_version)
    }

    /// Abort transaction
    pub fn abort_transaction(&self, tx_id: u64) -> Result<()> {
        let mut active = self.active_transactions.write()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
        
        if let Some(mut tx_info) = active.remove(&tx_id) {
            tx_info.phase = TransactionPhase::Aborted;

            // Update stats
            {
                let mut stats = self.stats.lock()
                    .map_err(|_| anyhow!("Failed to acquire stats lock"))?;
                stats.aborted_transactions += 1;
                stats.current_active_transactions = stats.current_active_transactions.saturating_sub(1);
            }
        }

        Ok(())
    }

    /// Execute transaction with retry logic
    pub fn execute_with_retry<F, R>(&self, mut operation: F) -> Result<R>
    where
        F: FnMut() -> Result<R>,
    {
        let mut attempts = 0;
        let mut backoff_ms = self.config.base_backoff_ms;

        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.config.max_retries {
                        return Err(anyhow!("Transaction failed after {} attempts: {}", attempts, e));
                    }

                    // Update retry stats
                    {
                        if let Ok(mut stats) = self.stats.lock() {
                            stats.retry_count += 1;
                        }
                    }

                    // Exponential backoff with jitter
                    let jitter = fastrand::u64(0..backoff_ms / 4);
                    let sleep_time = Duration::from_millis(backoff_ms + jitter);
                    std::thread::sleep(sleep_time);

                    backoff_ms = (backoff_ms * 2).min(self.config.max_backoff_ms);
                }
            }
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> Result<OptimisticStats> {
        let stats = self.stats.lock()
            .map_err(|_| anyhow!("Failed to acquire stats lock"))?;
        Ok(stats.clone())
    }

    /// Get transaction information
    pub fn get_transaction_info(&self, tx_id: u64) -> Result<Option<TransactionInfo>> {
        let active = self.active_transactions.read()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
        Ok(active.get(&tx_id).cloned())
    }

    /// Get all active transaction IDs
    pub fn get_active_transactions(&self) -> Result<Vec<u64>> {
        let active = self.active_transactions.read()
            .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
        Ok(active.keys().copied().collect())
    }

    /// Clear all transactions (for testing/cleanup)
    pub fn clear(&self) -> Result<()> {
        {
            let mut active = self.active_transactions.write()
                .map_err(|_| anyhow!("Failed to acquire active transactions lock"))?;
            active.clear();
        }

        {
            let mut history = self.commit_history.write()
                .map_err(|_| anyhow!("Failed to acquire commit history lock"))?;
            history.clear();
        }

        {
            let mut stats = self.stats.lock()
                .map_err(|_| anyhow!("Failed to acquire stats lock"))?;
            *stats = OptimisticStats::default();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_lifecycle() {
        let controller = OptimisticConcurrencyController::new(1);
        
        // Begin transaction
        let tx_id = controller.begin_transaction().unwrap();
        assert!(tx_id > 0);

        // Record operations
        controller.record_read(tx_id, "key1".to_string(), 1, None).unwrap();
        controller.record_write(tx_id, "key2".to_string(), 2, Some(1), WriteOperation::Update).unwrap();

        // Validate transaction
        let result = controller.validate_transaction(tx_id).unwrap();
        assert_eq!(result, ValidationResult::Valid);

        // Commit transaction
        let version = controller.commit_transaction(tx_id).unwrap();
        assert!(version > 0);

        // Check stats
        let stats = controller.get_stats().unwrap();
        assert_eq!(stats.total_transactions, 1);
        assert_eq!(stats.committed_transactions, 1);
    }

    #[test]
    fn test_write_write_conflict() {
        let controller = OptimisticConcurrencyController::new(1);
        
        // First transaction
        let tx1 = controller.begin_transaction().unwrap();
        controller.record_write(tx1, "key1".to_string(), 1, None, WriteOperation::Insert).unwrap();
        controller.validate_transaction(tx1).unwrap();
        controller.commit_transaction(tx1).unwrap();

        // Second transaction with conflicting write
        let tx2 = controller.begin_transaction().unwrap();
        controller.record_write(tx2, "key1".to_string(), 2, Some(1), WriteOperation::Update).unwrap();
        
        let result = controller.validate_transaction(tx2).unwrap();
        match result {
            ValidationResult::Conflict { conflict_type, .. } => {
                assert_eq!(conflict_type, ConflictType::WriteWrite);
            }
            _ => panic!("Expected write-write conflict"),
        }
    }

    #[test]
    fn test_phantom_read_detection() {
        let controller = OptimisticConcurrencyController::new(1);
        
        // Transaction with predicate read
        let tx1 = controller.begin_transaction().unwrap();
        controller.record_read(tx1, "range_query".to_string(), 1, Some("name=John".to_string())).unwrap();

        // Concurrent transaction inserts matching record
        let tx2 = controller.begin_transaction().unwrap();
        controller.record_write(tx2, "name=John_Smith".to_string(), 2, None, WriteOperation::Insert).unwrap();
        controller.validate_transaction(tx2).unwrap();
        controller.commit_transaction(tx2).unwrap();

        // First transaction should detect phantom read
        let result = controller.validate_transaction(tx1).unwrap();
        match result {
            ValidationResult::Conflict { conflict_type, .. } => {
                assert_eq!(conflict_type, ConflictType::ReadWrite);
            }
            _ => panic!("Expected phantom read conflict"),
        }
    }

    #[test]
    fn test_statistics_collection() {
        let controller = OptimisticConcurrencyController::new(1);
        
        // Execute multiple transactions
        for i in 0..5 {
            let tx = controller.begin_transaction().unwrap();
            controller.record_write(tx, format!("key{}", i), i + 1, None, WriteOperation::Insert).unwrap();
            controller.validate_transaction(tx).unwrap();
            controller.commit_transaction(tx).unwrap();
        }

        let stats = controller.get_stats().unwrap();
        assert_eq!(stats.total_transactions, 5);
        assert_eq!(stats.committed_transactions, 5);
        assert_eq!(stats.aborted_transactions, 0);
    }

    #[test]
    fn test_version_vector() {
        let mut vv1 = VersionVector::new();
        let mut vv2 = VersionVector::new();

        vv1.increment(1);
        vv2.increment(2);

        assert!(!vv1.happens_before(&vv2));
        assert!(!vv2.happens_before(&vv1));

        vv1.update(&vv2);
        assert!(vv2.happens_before(&vv1));
    }
}