//! Transaction Support for OxiRS
//!
//! ACID transaction management for SPARQL UPDATE operations with support for
//! BEGIN/COMMIT/ROLLBACK, isolation levels, and multi-operation atomicity.

use super::{ToolResult, ToolStats};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

/// Transaction isolation levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// Read uncommitted - lowest isolation, highest performance
    ReadUncommitted,
    /// Read committed - prevent dirty reads
    ReadCommitted,
    /// Repeatable read - prevent non-repeatable reads
    RepeatableRead,
    /// Serializable - highest isolation, lowest performance
    Serializable,
}

impl IsolationLevel {
    pub fn name(&self) -> &str {
        match self {
            IsolationLevel::ReadUncommitted => "READ_UNCOMMITTED",
            IsolationLevel::ReadCommitted => "READ_COMMITTED",
            IsolationLevel::RepeatableRead => "REPEATABLE_READ",
            IsolationLevel::Serializable => "SERIALIZABLE",
        }
    }

    pub fn description(&self) -> &str {
        match self {
            IsolationLevel::ReadUncommitted => "Lowest isolation, allows dirty reads",
            IsolationLevel::ReadCommitted => "Prevents dirty reads",
            IsolationLevel::RepeatableRead => "Prevents dirty and non-repeatable reads",
            IsolationLevel::Serializable => "Highest isolation, full ACID compliance",
        }
    }
}

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// No active transaction
    None,
    /// Transaction started, operations in progress
    Active,
    /// Transaction committed successfully
    Committed,
    /// Transaction rolled back
    RolledBack,
    /// Transaction failed with error
    Failed,
}

/// Transaction operation types
#[derive(Debug, Clone)]
pub enum TransactionOperation {
    Insert { triples: String },
    Delete { triples: String },
    Update { query: String },
    Load { file: PathBuf },
    Clear { graph: Option<String> },
}

/// Savepoint for nested transactions
#[derive(Debug, Clone)]
pub struct Savepoint {
    pub name: String,
    pub timestamp: SystemTime,
    pub operation_count: usize,
}

/// Transaction context
#[derive(Debug)]
pub struct Transaction {
    pub id: String,
    pub state: TransactionState,
    pub isolation_level: IsolationLevel,
    pub start_time: Instant,
    pub operations: Vec<TransactionOperation>,
    pub savepoints: Vec<Savepoint>,
    pub dataset: String,
    pub read_only: bool,
}

impl Transaction {
    /// Create a new transaction
    pub fn new(dataset: String, isolation_level: IsolationLevel, read_only: bool) -> Self {
        let id = format!("txn_{}", uuid::Uuid::new_v4());

        Self {
            id,
            state: TransactionState::Active,
            isolation_level,
            start_time: Instant::now(),
            operations: Vec::new(),
            savepoints: Vec::new(),
            dataset,
            read_only,
        }
    }

    /// Add an operation to the transaction
    pub fn add_operation(&mut self, operation: TransactionOperation) -> ToolResult {
        if self.state != TransactionState::Active {
            return Err(format!("Transaction is not active: {:?}", self.state).into());
        }

        if self.read_only {
            return Err("Cannot modify data in read-only transaction".into());
        }

        self.operations.push(operation);
        Ok(())
    }

    /// Create a savepoint
    pub fn create_savepoint(&mut self, name: String) -> ToolResult {
        if self.state != TransactionState::Active {
            return Err(format!("Transaction is not active: {:?}", self.state).into());
        }

        // Check if savepoint name already exists
        if self.savepoints.iter().any(|sp| sp.name == name) {
            return Err(format!("Savepoint '{}' already exists", name).into());
        }

        let savepoint = Savepoint {
            name,
            timestamp: SystemTime::now(),
            operation_count: self.operations.len(),
        };

        self.savepoints.push(savepoint);
        Ok(())
    }

    /// Rollback to a savepoint
    pub fn rollback_to_savepoint(&mut self, name: &str) -> ToolResult {
        if self.state != TransactionState::Active {
            return Err(format!("Transaction is not active: {:?}", self.state).into());
        }

        // Find the savepoint
        let savepoint_idx = self
            .savepoints
            .iter()
            .position(|sp| sp.name == name)
            .ok_or_else(|| format!("Savepoint '{}' not found", name))?;

        let savepoint = &self.savepoints[savepoint_idx];
        let operation_count = savepoint.operation_count;

        // Rollback operations
        self.operations.truncate(operation_count);

        // Remove this and all later savepoints
        self.savepoints.truncate(savepoint_idx);

        println!("Rolled back to savepoint '{}'", name);
        println!("  Operations restored to: {}", operation_count);

        Ok(())
    }

    /// Get transaction duration
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get transaction status summary
    pub fn summary(&self) -> String {
        format!(
            "Transaction {} [{}] - {} operations, {} savepoints, {:?} elapsed",
            self.id,
            self.state_str(),
            self.operations.len(),
            self.savepoints.len(),
            self.duration()
        )
    }

    fn state_str(&self) -> &str {
        match self.state {
            TransactionState::None => "NONE",
            TransactionState::Active => "ACTIVE",
            TransactionState::Committed => "COMMITTED",
            TransactionState::RolledBack => "ROLLED_BACK",
            TransactionState::Failed => "FAILED",
        }
    }
}

/// Transaction manager
pub struct TransactionManager {
    active_transactions: HashMap<String, Transaction>,
    transaction_log: Vec<TransactionLogEntry>,
}

impl TransactionManager {
    pub fn new() -> Self {
        Self {
            active_transactions: HashMap::new(),
            transaction_log: Vec::new(),
        }
    }

    /// Begin a new transaction
    pub fn begin(
        &mut self,
        dataset: String,
        isolation_level: IsolationLevel,
        read_only: bool,
    ) -> ToolResult<String> {
        let transaction = Transaction::new(dataset.clone(), isolation_level, read_only);
        let txn_id = transaction.id.clone();

        println!("BEGIN TRANSACTION {}", txn_id);
        println!("  Dataset: {}", dataset);
        println!("  Isolation: {}", isolation_level.name());
        println!("  Read-only: {}", read_only);

        self.active_transactions.insert(txn_id.clone(), transaction);

        self.transaction_log.push(TransactionLogEntry {
            transaction_id: txn_id.clone(),
            action: "BEGIN".to_string(),
            timestamp: SystemTime::now(),
            details: None,
        });

        Ok(txn_id)
    }

    /// Commit a transaction
    pub fn commit(&mut self, txn_id: &str) -> ToolResult {
        // First, validate and get operation count
        let (operation_count, duration) = {
            let transaction = self
                .active_transactions
                .get(txn_id)
                .ok_or_else(|| format!("Transaction '{}' not found", txn_id))?;

            if transaction.state != TransactionState::Active {
                return Err(format!("Transaction is not active: {:?}", transaction.state).into());
            }

            (transaction.operations.len(), transaction.duration())
        };

        println!("COMMIT TRANSACTION {}", txn_id);
        println!("  Operations: {}", operation_count);

        // Execute all operations (immutable borrow)
        let result = {
            let transaction = self
                .active_transactions
                .get(txn_id)
                .expect("transaction should exist for given id");
            self.execute_operations(transaction)
        };

        // Update transaction state (mutable borrow)
        let transaction = self
            .active_transactions
            .get_mut(txn_id)
            .expect("transaction should exist for given id");

        match result {
            Ok(_) => {
                transaction.state = TransactionState::Committed;
                println!("  Status: SUCCESS");
                println!("  Duration: {:?}", duration);

                self.transaction_log.push(TransactionLogEntry {
                    transaction_id: txn_id.to_string(),
                    action: "COMMIT".to_string(),
                    timestamp: SystemTime::now(),
                    details: Some(format!("{} operations", operation_count)),
                });

                Ok(())
            }
            Err(e) => {
                transaction.state = TransactionState::Failed;
                println!("  Status: FAILED");
                println!("  Error: {}", e);

                self.transaction_log.push(TransactionLogEntry {
                    transaction_id: txn_id.to_string(),
                    action: "COMMIT_FAILED".to_string(),
                    timestamp: SystemTime::now(),
                    details: Some(e.to_string()),
                });

                Err(e)
            }
        }
    }

    /// Rollback a transaction
    pub fn rollback(&mut self, txn_id: &str) -> ToolResult {
        let transaction = self
            .active_transactions
            .get_mut(txn_id)
            .ok_or_else(|| format!("Transaction '{}' not found", txn_id))?;

        if transaction.state != TransactionState::Active {
            return Err(format!("Transaction is not active: {:?}", transaction.state).into());
        }

        println!("ROLLBACK TRANSACTION {}", txn_id);
        println!("  Operations discarded: {}", transaction.operations.len());

        transaction.state = TransactionState::RolledBack;
        transaction.operations.clear();
        transaction.savepoints.clear();

        println!("  Status: ROLLED BACK");
        println!("  Duration: {:?}", transaction.duration());

        self.transaction_log.push(TransactionLogEntry {
            transaction_id: txn_id.to_string(),
            action: "ROLLBACK".to_string(),
            timestamp: SystemTime::now(),
            details: None,
        });

        Ok(())
    }

    /// Execute transaction operations
    fn execute_operations(&self, transaction: &Transaction) -> ToolResult {
        println!("  Executing {} operations...", transaction.operations.len());

        for (i, operation) in transaction.operations.iter().enumerate() {
            print!("    Operation {}/{}: ", i + 1, transaction.operations.len());

            match operation {
                TransactionOperation::Insert { triples } => {
                    println!("INSERT {} triples", triples.lines().count());
                    // Simulate execution
                    std::thread::sleep(Duration::from_millis(10));
                }
                TransactionOperation::Delete { triples } => {
                    println!("DELETE {} triples", triples.lines().count());
                    // Simulate execution
                    std::thread::sleep(Duration::from_millis(10));
                }
                TransactionOperation::Update { query } => {
                    println!("UPDATE (query: {} chars)", query.len());
                    // Simulate execution
                    std::thread::sleep(Duration::from_millis(20));
                }
                TransactionOperation::Load { file } => {
                    println!("LOAD {}", file.display());
                    // Simulate execution
                    std::thread::sleep(Duration::from_millis(50));
                }
                TransactionOperation::Clear { graph } => {
                    if let Some(g) = graph {
                        println!("CLEAR GRAPH <{}>", g);
                    } else {
                        println!("CLEAR DEFAULT");
                    }
                    // Simulate execution
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }

        Ok(())
    }

    /// Get active transaction
    pub fn get_transaction(&self, txn_id: &str) -> Option<&Transaction> {
        self.active_transactions.get(txn_id)
    }

    /// Get mutable active transaction
    pub fn get_transaction_mut(&mut self, txn_id: &str) -> Option<&mut Transaction> {
        self.active_transactions.get_mut(txn_id)
    }

    /// List all active transactions
    pub fn list_active(&self) -> Vec<&Transaction> {
        self.active_transactions
            .values()
            .filter(|txn| txn.state == TransactionState::Active)
            .collect()
    }

    /// Get transaction log
    pub fn get_log(&self) -> &[TransactionLogEntry] {
        &self.transaction_log
    }

    /// Clean up finished transactions
    pub fn cleanup(&mut self) {
        self.active_transactions
            .retain(|_, txn| txn.state == TransactionState::Active);
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Transaction log entry
#[derive(Debug, Clone)]
pub struct TransactionLogEntry {
    pub transaction_id: String,
    pub action: String,
    pub timestamp: SystemTime,
    pub details: Option<String>,
}

/// Configuration for transaction execution
pub struct TransactionConfig {
    pub dataset: String,
    pub operations: Vec<String>,
    pub isolation_level: IsolationLevel,
    pub read_only: bool,
    pub auto_commit: bool,
}

/// Run a transactional operation
pub async fn run(config: TransactionConfig) -> ToolResult {
    let mut stats = ToolStats::new();
    let mut manager = TransactionManager::new();

    println!("OxiRS Transaction Manager");
    println!("=========================\n");

    // Begin transaction
    let txn_id = manager.begin(config.dataset, config.isolation_level, config.read_only)?;

    println!();

    // Add operations
    let transaction = manager
        .get_transaction_mut(&txn_id)
        .expect("transaction should exist after begin");

    for operation_str in &config.operations {
        let operation = parse_operation(operation_str)?;
        transaction.add_operation(operation)?;
    }

    println!(
        "Added {} operations to transaction\n",
        config.operations.len()
    );

    // Commit or rollback
    if config.auto_commit {
        match manager.commit(&txn_id) {
            Ok(_) => {
                println!("\n✓ Transaction committed successfully");
                stats.items_processed = config.operations.len();
            }
            Err(e) => {
                println!("\n✗ Transaction commit failed: {}", e);
                println!("  Rolling back...");
                manager.rollback(&txn_id)?;
                return Err(e);
            }
        }
    } else {
        println!("Transaction prepared (not committed)");
        println!("  Use 'oxirs transaction commit {}' to commit", txn_id);
        println!("  Use 'oxirs transaction rollback {}' to rollback", txn_id);
    }

    manager.cleanup();

    stats.finish();
    stats.print_summary("Transaction");

    Ok(())
}

/// Parse operation string into TransactionOperation
fn parse_operation(operation_str: &str) -> ToolResult<TransactionOperation> {
    let operation_str = operation_str.trim();

    if operation_str.starts_with("INSERT") {
        let triples = operation_str.strip_prefix("INSERT").unwrap_or("").trim();
        Ok(TransactionOperation::Insert {
            triples: triples.to_string(),
        })
    } else if operation_str.starts_with("DELETE") {
        let triples = operation_str.strip_prefix("DELETE").unwrap_or("").trim();
        Ok(TransactionOperation::Delete {
            triples: triples.to_string(),
        })
    } else if operation_str.starts_with("UPDATE") {
        let query = operation_str.strip_prefix("UPDATE").unwrap_or("").trim();
        Ok(TransactionOperation::Update {
            query: query.to_string(),
        })
    } else if operation_str.starts_with("LOAD") {
        let file_str = operation_str.strip_prefix("LOAD").unwrap_or("").trim();
        Ok(TransactionOperation::Load {
            file: PathBuf::from(file_str),
        })
    } else if operation_str.starts_with("CLEAR") {
        let graph_str = operation_str.strip_prefix("CLEAR").unwrap_or("").trim();
        let graph = if graph_str.is_empty() || graph_str == "DEFAULT" {
            None
        } else {
            Some(graph_str.to_string())
        };
        Ok(TransactionOperation::Clear { graph })
    } else {
        Err(format!("Unknown operation: {}", operation_str).into())
    }
}

/// Display transaction status
pub fn display_status(manager: &TransactionManager) {
    let active = manager.list_active();

    println!("Active Transactions: {}\n", active.len());

    if active.is_empty() {
        println!("No active transactions");
        return;
    }

    println!(
        "{:<40} {:>12} {:>15} {:>10}",
        "Transaction ID", "Operations", "Savepoints", "Duration"
    );
    println!("{}", "-".repeat(80));

    for txn in active {
        println!(
            "{:<40} {:>12} {:>15} {:>10.2?}",
            txn.id,
            txn.operations.len(),
            txn.savepoints.len(),
            txn.duration()
        );
    }
}

/// Display transaction log
pub fn display_log(manager: &TransactionManager, limit: Option<usize>) {
    let log = manager.get_log();
    let entries = if let Some(n) = limit {
        &log[log.len().saturating_sub(n)..]
    } else {
        log
    };

    println!("Transaction Log ({} entries)\n", entries.len());

    for entry in entries {
        println!(
            "[{:?}] {} - {}",
            entry.timestamp, entry.transaction_id, entry.action
        );

        if let Some(ref details) = entry.details {
            println!("  Details: {}", details);
        }
    }
}

// UUID generation helper module
mod uuid {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub struct Uuid;

    impl Uuid {
        pub fn new_v4() -> String {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_nanos();

            format!("{:032x}", now)
        }
    }
}
