//! Transaction management for SPARQL UPDATE operations
//!
//! This module provides transaction support for grouping multiple SPARQL UPDATE
//! operations into atomic units that can be committed or rolled back.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

use crate::cli::error::{CliError, CliResult};

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionState {
    /// No active transaction
    Idle,
    /// Transaction in progress
    Active,
    /// Transaction preparing to commit
    Preparing,
    /// Transaction committed
    Committed,
    /// Transaction rolled back
    RolledBack,
    /// Transaction failed
    Failed,
}

impl fmt::Display for TransactionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransactionState::Idle => write!(f, "idle"),
            TransactionState::Active => write!(f, "active"),
            TransactionState::Preparing => write!(f, "preparing"),
            TransactionState::Committed => write!(f, "committed"),
            TransactionState::RolledBack => write!(f, "rolled back"),
            TransactionState::Failed => write!(f, "failed"),
        }
    }
}

/// Isolation level for transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IsolationLevel {
    /// Read uncommitted - lowest isolation, highest performance
    ReadUncommitted,
    /// Read committed - prevents dirty reads
    #[default]
    ReadCommitted,
    /// Repeatable read - prevents non-repeatable reads
    RepeatableRead,
    /// Serializable - highest isolation, lowest performance
    Serializable,
}

impl fmt::Display for IsolationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IsolationLevel::ReadUncommitted => write!(f, "read-uncommitted"),
            IsolationLevel::ReadCommitted => write!(f, "read-committed"),
            IsolationLevel::RepeatableRead => write!(f, "repeatable-read"),
            IsolationLevel::Serializable => write!(f, "serializable"),
        }
    }
}

/// A single operation within a transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionOperation {
    /// The SPARQL UPDATE query
    pub query: String,
    /// Timestamp when operation was added
    pub timestamp: DateTime<Utc>,
    /// Optional operation description
    pub description: Option<String>,
}

impl TransactionOperation {
    /// Create a new transaction operation
    pub fn new(query: String) -> Self {
        Self {
            query,
            timestamp: Utc::now(),
            description: None,
        }
    }

    /// Create an operation with description
    pub fn with_description(query: String, description: String) -> Self {
        Self {
            query,
            timestamp: Utc::now(),
            description: Some(description),
        }
    }
}

/// Transaction metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionMetadata {
    /// Transaction ID
    pub id: String,
    /// Transaction state
    pub state: TransactionState,
    /// Isolation level
    pub isolation_level: IsolationLevel,
    /// Start timestamp
    pub started_at: DateTime<Utc>,
    /// End timestamp (commit/rollback)
    pub ended_at: Option<DateTime<Utc>>,
    /// Number of operations
    pub operation_count: usize,
    /// Read-only transaction
    pub read_only: bool,
    /// Optional transaction name/label
    pub name: Option<String>,
}

/// Transaction configuration
#[derive(Debug, Clone)]
pub struct TransactionConfig {
    /// Isolation level
    pub isolation_level: IsolationLevel,
    /// Read-only mode
    pub read_only: bool,
    /// Maximum number of operations per transaction
    pub max_operations: usize,
    /// Transaction timeout in seconds (None = no timeout)
    pub timeout_seconds: Option<u64>,
    /// Auto-commit on success
    pub auto_commit: bool,
}

impl Default for TransactionConfig {
    fn default() -> Self {
        Self {
            isolation_level: IsolationLevel::default(),
            read_only: false,
            max_operations: 1000,
            timeout_seconds: Some(300), // 5 minutes
            auto_commit: false,
        }
    }
}

impl TransactionConfig {
    /// Create a read-only transaction config
    pub fn read_only() -> Self {
        Self {
            read_only: true,
            ..Default::default()
        }
    }

    /// Create a config with specific isolation level
    pub fn with_isolation(isolation_level: IsolationLevel) -> Self {
        Self {
            isolation_level,
            ..Default::default()
        }
    }

    /// Set auto-commit
    pub fn auto_commit(mut self, enabled: bool) -> Self {
        self.auto_commit = enabled;
        self
    }

    /// Set timeout
    pub fn timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = Some(seconds);
        self
    }
}

/// Transaction manager for SPARQL operations
pub struct TransactionManager {
    /// Current transaction metadata
    metadata: Option<TransactionMetadata>,
    /// Queued operations
    operations: VecDeque<TransactionOperation>,
    /// Configuration
    config: TransactionConfig,
    /// Transaction counter for generating IDs
    transaction_counter: u64,
}

impl TransactionManager {
    /// Create a new transaction manager
    pub fn new() -> Self {
        Self::with_config(TransactionConfig::default())
    }

    /// Create a transaction manager with custom config
    pub fn with_config(config: TransactionConfig) -> Self {
        Self {
            metadata: None,
            operations: VecDeque::new(),
            config,
            transaction_counter: 0,
        }
    }

    /// Check if a transaction is active
    pub fn is_active(&self) -> bool {
        matches!(
            self.metadata.as_ref().map(|m| m.state),
            Some(TransactionState::Active)
        )
    }

    /// Get current transaction state
    pub fn state(&self) -> TransactionState {
        self.metadata
            .as_ref()
            .map(|m| m.state)
            .unwrap_or(TransactionState::Idle)
    }

    /// Get current transaction metadata
    pub fn metadata(&self) -> Option<&TransactionMetadata> {
        self.metadata.as_ref()
    }

    /// Get number of operations in current transaction
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Begin a new transaction
    pub fn begin(&mut self) -> CliResult<String> {
        self.begin_with_name(None)
    }

    /// Begin a new transaction with a name
    pub fn begin_with_name(&mut self, name: Option<String>) -> CliResult<String> {
        if self.is_active() {
            return Err(CliError::invalid_arguments(
                "A transaction is already active. Commit or rollback the current transaction first.",
            ));
        }

        self.transaction_counter += 1;
        let tx_id = format!("tx-{}", self.transaction_counter);

        self.metadata = Some(TransactionMetadata {
            id: tx_id.clone(),
            state: TransactionState::Active,
            isolation_level: self.config.isolation_level,
            started_at: Utc::now(),
            ended_at: None,
            operation_count: 0,
            read_only: self.config.read_only,
            name,
        });

        self.operations.clear();

        Ok(tx_id)
    }

    /// Add an operation to the current transaction
    pub fn add_operation(&mut self, query: String) -> CliResult<()> {
        self.add_operation_with_description(query, None)
    }

    /// Add an operation with description
    pub fn add_operation_with_description(
        &mut self,
        query: String,
        description: Option<String>,
    ) -> CliResult<()> {
        if !self.is_active() {
            return Err(CliError::invalid_arguments(
                "No active transaction. Use BEGIN to start a transaction.",
            ));
        }

        if self.operations.len() >= self.config.max_operations {
            return Err(CliError::invalid_arguments(format!(
                "Transaction operation limit reached ({}). Commit or rollback the transaction.",
                self.config.max_operations
            )));
        }

        // Check if transaction has timed out
        if let Some(timeout) = self.config.timeout_seconds {
            if let Some(ref meta) = self.metadata {
                let elapsed = Utc::now()
                    .signed_duration_since(meta.started_at)
                    .num_seconds() as u64;
                if elapsed > timeout {
                    self.fail("Transaction timeout")?;
                    return Err(CliError::invalid_arguments(format!(
                        "Transaction timed out after {} seconds",
                        timeout
                    )));
                }
            }
        }

        let operation = if let Some(desc) = description {
            TransactionOperation::with_description(query, desc)
        } else {
            TransactionOperation::new(query)
        };

        self.operations.push_back(operation);

        if let Some(ref mut meta) = self.metadata {
            meta.operation_count = self.operations.len();
        }

        Ok(())
    }

    /// Get all operations in the current transaction
    pub fn operations(&self) -> &VecDeque<TransactionOperation> {
        &self.operations
    }

    /// Prepare transaction for commit
    pub fn prepare(&mut self) -> CliResult<()> {
        if !self.is_active() {
            return Err(CliError::invalid_arguments("No active transaction"));
        }

        if let Some(ref mut meta) = self.metadata {
            meta.state = TransactionState::Preparing;
        }

        Ok(())
    }

    /// Commit the current transaction
    pub fn commit(&mut self) -> CliResult<Vec<String>> {
        if !self.is_active() && self.state() != TransactionState::Preparing {
            return Err(CliError::invalid_arguments(
                "No active transaction to commit",
            ));
        }

        // Collect all queries
        let queries: Vec<String> = self.operations.iter().map(|op| op.query.clone()).collect();

        if let Some(ref mut meta) = self.metadata {
            meta.state = TransactionState::Committed;
            meta.ended_at = Some(Utc::now());
        }

        self.operations.clear();

        Ok(queries)
    }

    /// Rollback the current transaction
    pub fn rollback(&mut self) -> CliResult<()> {
        if !self.is_active() && self.state() != TransactionState::Preparing {
            return Err(CliError::invalid_arguments(
                "No active transaction to rollback",
            ));
        }

        if let Some(ref mut meta) = self.metadata {
            meta.state = TransactionState::RolledBack;
            meta.ended_at = Some(Utc::now());
        }

        self.operations.clear();

        Ok(())
    }

    /// Mark transaction as failed
    pub fn fail(&mut self, reason: &str) -> CliResult<()> {
        if let Some(ref mut meta) = self.metadata {
            meta.state = TransactionState::Failed;
            meta.ended_at = Some(Utc::now());
        }

        self.operations.clear();

        Err(CliError::invalid_arguments(format!(
            "Transaction failed: {}",
            reason
        )))
    }

    /// Get transaction statistics
    pub fn stats(&self) -> TransactionStats {
        TransactionStats {
            total_transactions: self.transaction_counter,
            current_operations: self.operations.len(),
            current_state: self.state(),
            is_active: self.is_active(),
            started_at: self.metadata.as_ref().map(|m| m.started_at),
            operation_limit: self.config.max_operations,
            timeout_seconds: self.config.timeout_seconds,
        }
    }

    /// Clear transaction history
    pub fn clear(&mut self) {
        self.metadata = None;
        self.operations.clear();
    }

    /// Get configuration
    pub fn config(&self) -> &TransactionConfig {
        &self.config
    }

    /// Update configuration (only when no transaction is active)
    pub fn set_config(&mut self, config: TransactionConfig) -> CliResult<()> {
        if self.is_active() {
            return Err(CliError::invalid_arguments(
                "Cannot change configuration while transaction is active",
            ));
        }

        self.config = config;
        Ok(())
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Transaction statistics
#[derive(Debug, Clone)]
pub struct TransactionStats {
    /// Total number of transactions started
    pub total_transactions: u64,
    /// Number of operations in current transaction
    pub current_operations: usize,
    /// Current transaction state
    pub current_state: TransactionState,
    /// Whether a transaction is active
    pub is_active: bool,
    /// When current transaction started
    pub started_at: Option<DateTime<Utc>>,
    /// Maximum operations per transaction
    pub operation_limit: usize,
    /// Timeout in seconds
    pub timeout_seconds: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_lifecycle() {
        let mut manager = TransactionManager::new();

        // Initially idle
        assert_eq!(manager.state(), TransactionState::Idle);
        assert!(!manager.is_active());

        // Begin transaction
        let tx_id = manager.begin().unwrap();
        assert!(!tx_id.is_empty());
        assert_eq!(manager.state(), TransactionState::Active);
        assert!(manager.is_active());

        // Add operations
        manager
            .add_operation("INSERT DATA { <x> <y> <z> }".to_string())
            .unwrap();
        assert_eq!(manager.operation_count(), 1);

        manager
            .add_operation("DELETE DATA { <a> <b> <c> }".to_string())
            .unwrap();
        assert_eq!(manager.operation_count(), 2);

        // Commit
        let queries = manager.commit().unwrap();
        assert_eq!(queries.len(), 2);
        assert_eq!(manager.state(), TransactionState::Committed);
        assert!(!manager.is_active());
        assert_eq!(manager.operation_count(), 0);
    }

    #[test]
    fn test_transaction_rollback() {
        let mut manager = TransactionManager::new();

        manager.begin().unwrap();
        manager
            .add_operation("INSERT DATA { <x> <y> <z> }".to_string())
            .unwrap();
        manager
            .add_operation("DELETE DATA { <a> <b> <c> }".to_string())
            .unwrap();

        assert_eq!(manager.operation_count(), 2);

        manager.rollback().unwrap();

        assert_eq!(manager.state(), TransactionState::RolledBack);
        assert_eq!(manager.operation_count(), 0);
        assert!(!manager.is_active());
    }

    #[test]
    fn test_nested_transaction_prevention() {
        let mut manager = TransactionManager::new();

        manager.begin().unwrap();

        let result = manager.begin();
        assert!(result.is_err());
    }

    #[test]
    fn test_operation_without_transaction() {
        let mut manager = TransactionManager::new();

        let result = manager.add_operation("INSERT DATA { <x> <y> <z> }".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_commit_without_transaction() {
        let mut manager = TransactionManager::new();

        let result = manager.commit();
        assert!(result.is_err());
    }

    #[test]
    fn test_rollback_without_transaction() {
        let mut manager = TransactionManager::new();

        let result = manager.rollback();
        assert!(result.is_err());
    }

    #[test]
    fn test_transaction_with_name() {
        let mut manager = TransactionManager::new();

        manager
            .begin_with_name(Some("test-transaction".to_string()))
            .unwrap();

        let meta = manager.metadata().unwrap();
        assert_eq!(meta.name, Some("test-transaction".to_string()));
    }

    #[test]
    fn test_operation_with_description() {
        let mut manager = TransactionManager::new();

        manager.begin().unwrap();
        manager
            .add_operation_with_description(
                "INSERT DATA { <x> <y> <z> }".to_string(),
                Some("Add test triple".to_string()),
            )
            .unwrap();

        let ops = manager.operations();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].description, Some("Add test triple".to_string()));
    }

    #[test]
    fn test_operation_limit() {
        let config = TransactionConfig {
            max_operations: 2,
            ..Default::default()
        };
        let mut manager = TransactionManager::with_config(config);

        manager.begin().unwrap();
        manager
            .add_operation("INSERT DATA { <x1> <y> <z> }".to_string())
            .unwrap();
        manager
            .add_operation("INSERT DATA { <x2> <y> <z> }".to_string())
            .unwrap();

        // Third operation should fail
        let result = manager.add_operation("INSERT DATA { <x3> <y> <z> }".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_transaction_stats() {
        let mut manager = TransactionManager::new();

        let stats = manager.stats();
        assert_eq!(stats.total_transactions, 0);
        assert_eq!(stats.current_operations, 0);
        assert!(!stats.is_active);

        manager.begin().unwrap();
        manager
            .add_operation("INSERT DATA { <x> <y> <z> }".to_string())
            .unwrap();

        let stats = manager.stats();
        assert_eq!(stats.total_transactions, 1);
        assert_eq!(stats.current_operations, 1);
        assert!(stats.is_active);
    }

    #[test]
    fn test_read_only_config() {
        let config = TransactionConfig::read_only();
        let mut manager = TransactionManager::with_config(config);

        manager.begin().unwrap();

        let meta = manager.metadata().unwrap();
        assert!(meta.read_only);
    }

    #[test]
    fn test_isolation_levels() {
        let config = TransactionConfig::with_isolation(IsolationLevel::Serializable);
        let mut manager = TransactionManager::with_config(config);

        manager.begin().unwrap();

        let meta = manager.metadata().unwrap();
        assert_eq!(meta.isolation_level, IsolationLevel::Serializable);
    }

    #[test]
    fn test_clear_transaction() {
        let mut manager = TransactionManager::new();

        manager.begin().unwrap();
        manager
            .add_operation("INSERT DATA { <x> <y> <z> }".to_string())
            .unwrap();

        manager.clear();

        assert_eq!(manager.state(), TransactionState::Idle);
        assert_eq!(manager.operation_count(), 0);
    }

    #[test]
    fn test_config_update_while_active() {
        let mut manager = TransactionManager::new();

        manager.begin().unwrap();

        let result = manager.set_config(TransactionConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_transactions() {
        let mut manager = TransactionManager::new();

        // First transaction
        manager.begin().unwrap();
        manager
            .add_operation("INSERT DATA { <x1> <y> <z> }".to_string())
            .unwrap();
        manager.commit().unwrap();

        // Second transaction
        manager.begin().unwrap();
        manager
            .add_operation("INSERT DATA { <x2> <y> <z> }".to_string())
            .unwrap();
        manager.commit().unwrap();

        let stats = manager.stats();
        assert_eq!(stats.total_transactions, 2);
    }
}
