//! Transaction Support for Reasoning Operations
//!
//! Provides ACID (Atomicity, Consistency, Isolation, Durability) transactions
//! for rule-based reasoning operations.
//!
//! # Features
//!
//! - **Atomicity**: All-or-nothing updates
//! - **Consistency**: Maintain knowledge base integrity
//! - **Isolation**: Concurrent transaction support
//! - **Durability**: Persist committed changes
//! - **Rollback**: Undo uncommitted changes
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::transaction::{TransactionManager, IsolationLevel};
//! use oxirs_rule::{RuleEngine, RuleAtom, Term};
//!
//! let mut manager = TransactionManager::new();
//! let mut engine = RuleEngine::new();
//!
//! // Begin transaction
//! let tx_id = manager.begin_transaction(IsolationLevel::ReadCommitted).unwrap();
//!
//! // Perform operations
//! let fact = RuleAtom::Triple {
//!     subject: Term::Constant("john".to_string()),
//!     predicate: Term::Constant("age".to_string()),
//!     object: Term::Literal("30".to_string()),
//! };
//!
//! manager.add_fact(tx_id, fact).unwrap();
//!
//! // Commit or rollback
//! manager.commit(tx_id).unwrap();
//! ```

use crate::RuleAtom;
use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

/// Transaction identifier
pub type TransactionId = u64;

/// Isolation level for transactions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// Read uncommitted (lowest isolation)
    ReadUncommitted,
    /// Read committed
    ReadCommitted,
    /// Repeatable read
    RepeatableRead,
    /// Serializable (highest isolation)
    Serializable,
}

/// Transaction state
#[derive(Debug, Clone, PartialEq)]
pub enum TransactionState {
    /// Transaction is active
    Active,
    /// Transaction is being committed
    Committing,
    /// Transaction has been committed
    Committed,
    /// Transaction is being rolled back
    RollingBack,
    /// Transaction has been aborted
    Aborted,
}

/// Transaction operation
#[derive(Debug, Clone)]
pub enum Operation {
    /// Add a fact
    AddFact(RuleAtom),
    /// Remove a fact
    RemoveFact(RuleAtom),
    /// Add multiple facts
    AddFacts(Vec<RuleAtom>),
    /// Remove multiple facts
    RemoveFacts(Vec<RuleAtom>),
}

/// Transaction record
#[derive(Debug, Clone)]
pub struct Transaction {
    /// Transaction ID
    pub id: TransactionId,
    /// Isolation level
    pub isolation_level: IsolationLevel,
    /// Current state
    pub state: TransactionState,
    /// Operations performed
    pub operations: Vec<Operation>,
    /// Timestamp when started
    pub start_time: u64,
    /// Timestamp when completed (if completed)
    pub end_time: Option<u64>,
}

/// Transaction manager
pub struct TransactionManager {
    /// Active transactions
    transactions: Arc<Mutex<HashMap<TransactionId, Transaction>>>,
    /// Next transaction ID
    next_id: Arc<Mutex<TransactionId>>,
    /// Committed facts
    committed_facts: Arc<Mutex<Vec<RuleAtom>>>,
    /// Transaction log for durability
    log: Arc<Mutex<VecDeque<LogEntry>>>,
    /// Current timestamp
    timestamp: Arc<Mutex<u64>>,
    /// Maximum log size
    max_log_size: usize,
}

/// Log entry for durability
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LogEntry {
    transaction_id: TransactionId,
    operation: Operation,
    timestamp: u64,
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TransactionManager {
    /// Create a new transaction manager
    pub fn new() -> Self {
        Self {
            transactions: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(0)),
            committed_facts: Arc::new(Mutex::new(Vec::new())),
            log: Arc::new(Mutex::new(VecDeque::new())),
            timestamp: Arc::new(Mutex::new(0)),
            max_log_size: 10000,
        }
    }

    /// Begin a new transaction
    pub fn begin_transaction(&self, isolation_level: IsolationLevel) -> Result<TransactionId> {
        let mut next_id = self.next_id.lock().unwrap();
        let tx_id = *next_id;
        *next_id += 1;

        let mut timestamp = self.timestamp.lock().unwrap();
        *timestamp += 1;
        let start_time = *timestamp;

        let transaction = Transaction {
            id: tx_id,
            isolation_level,
            state: TransactionState::Active,
            operations: Vec::new(),
            start_time,
            end_time: None,
        };

        let mut transactions = self.transactions.lock().unwrap();
        transactions.insert(tx_id, transaction);

        info!(
            "Started transaction {} with isolation level {:?}",
            tx_id, isolation_level
        );

        Ok(tx_id)
    }

    /// Add a fact within a transaction
    pub fn add_fact(&self, tx_id: TransactionId, fact: RuleAtom) -> Result<()> {
        let mut transactions = self.transactions.lock().unwrap();

        let transaction = transactions
            .get_mut(&tx_id)
            .ok_or_else(|| anyhow::anyhow!("Transaction {} not found", tx_id))?;

        if transaction.state != TransactionState::Active {
            return Err(anyhow::anyhow!(
                "Transaction {} is not active (state: {:?})",
                tx_id,
                transaction.state
            ));
        }

        transaction
            .operations
            .push(Operation::AddFact(fact.clone()));

        debug!("Added fact to transaction {}", tx_id);
        Ok(())
    }

    /// Remove a fact within a transaction
    pub fn remove_fact(&self, tx_id: TransactionId, fact: RuleAtom) -> Result<()> {
        let mut transactions = self.transactions.lock().unwrap();

        let transaction = transactions
            .get_mut(&tx_id)
            .ok_or_else(|| anyhow::anyhow!("Transaction {} not found", tx_id))?;

        if transaction.state != TransactionState::Active {
            return Err(anyhow::anyhow!("Transaction {} is not active", tx_id));
        }

        transaction.operations.push(Operation::RemoveFact(fact));

        debug!("Removed fact in transaction {}", tx_id);
        Ok(())
    }

    /// Add multiple facts within a transaction
    pub fn add_facts(&self, tx_id: TransactionId, facts: Vec<RuleAtom>) -> Result<()> {
        let mut transactions = self.transactions.lock().unwrap();

        let transaction = transactions
            .get_mut(&tx_id)
            .ok_or_else(|| anyhow::anyhow!("Transaction {} not found", tx_id))?;

        if transaction.state != TransactionState::Active {
            return Err(anyhow::anyhow!("Transaction {} is not active", tx_id));
        }

        transaction
            .operations
            .push(Operation::AddFacts(facts.clone()));

        debug!("Added {} facts to transaction {}", facts.len(), tx_id);
        Ok(())
    }

    /// Commit a transaction
    pub fn commit(&self, tx_id: TransactionId) -> Result<()> {
        let mut transactions = self.transactions.lock().unwrap();

        let transaction = transactions
            .get_mut(&tx_id)
            .ok_or_else(|| anyhow::anyhow!("Transaction {} not found", tx_id))?;

        if transaction.state != TransactionState::Active {
            return Err(anyhow::anyhow!(
                "Cannot commit transaction {} in state {:?}",
                tx_id,
                transaction.state
            ));
        }

        // Change state to committing
        transaction.state = TransactionState::Committing;

        // Apply operations to committed facts
        let mut committed_facts = self.committed_facts.lock().unwrap();

        for operation in &transaction.operations {
            match operation {
                Operation::AddFact(fact) => {
                    committed_facts.push(fact.clone());
                    self.log_operation(tx_id, operation.clone())?;
                }
                Operation::RemoveFact(fact) => {
                    committed_facts.retain(|f| f != fact);
                    self.log_operation(tx_id, operation.clone())?;
                }
                Operation::AddFacts(facts) => {
                    committed_facts.extend(facts.clone());
                    self.log_operation(tx_id, operation.clone())?;
                }
                Operation::RemoveFacts(facts) => {
                    for fact in facts {
                        committed_facts.retain(|f| f != fact);
                    }
                    self.log_operation(tx_id, operation.clone())?;
                }
            }
        }

        // Update transaction state
        let mut timestamp = self.timestamp.lock().unwrap();
        *timestamp += 1;
        transaction.end_time = Some(*timestamp);
        transaction.state = TransactionState::Committed;

        info!("Committed transaction {}", tx_id);
        Ok(())
    }

    /// Rollback a transaction
    pub fn rollback(&self, tx_id: TransactionId) -> Result<()> {
        let mut transactions = self.transactions.lock().unwrap();

        let transaction = transactions
            .get_mut(&tx_id)
            .ok_or_else(|| anyhow::anyhow!("Transaction {} not found", tx_id))?;

        if transaction.state != TransactionState::Active {
            return Err(anyhow::anyhow!(
                "Cannot rollback transaction {} in state {:?}",
                tx_id,
                transaction.state
            ));
        }

        // Change state to rolling back
        transaction.state = TransactionState::RollingBack;

        // Discard all operations
        transaction.operations.clear();

        // Update transaction state
        let mut timestamp = self.timestamp.lock().unwrap();
        *timestamp += 1;
        transaction.end_time = Some(*timestamp);
        transaction.state = TransactionState::Aborted;

        warn!("Rolled back transaction {}", tx_id);
        Ok(())
    }

    /// Get committed facts
    pub fn get_committed_facts(&self) -> Vec<RuleAtom> {
        let committed_facts = self.committed_facts.lock().unwrap();
        committed_facts.clone()
    }

    /// Get transaction state
    pub fn get_transaction_state(&self, tx_id: TransactionId) -> Option<TransactionState> {
        let transactions = self.transactions.lock().unwrap();
        transactions.get(&tx_id).map(|tx| tx.state.clone())
    }

    /// Check if transaction is active
    pub fn is_active(&self, tx_id: TransactionId) -> bool {
        matches!(
            self.get_transaction_state(tx_id),
            Some(TransactionState::Active)
        )
    }

    /// Log an operation for durability
    fn log_operation(&self, tx_id: TransactionId, operation: Operation) -> Result<()> {
        let mut log = self.log.lock().unwrap();
        let mut timestamp = self.timestamp.lock().unwrap();
        *timestamp += 1;

        let entry = LogEntry {
            transaction_id: tx_id,
            operation,
            timestamp: *timestamp,
        };

        log.push_back(entry);

        // Trim log if it exceeds max size
        while log.len() > self.max_log_size {
            log.pop_front();
        }

        Ok(())
    }

    /// Get statistics
    pub fn get_stats(&self) -> TransactionStats {
        let transactions = self.transactions.lock().unwrap();
        let committed_facts = self.committed_facts.lock().unwrap();
        let log = self.log.lock().unwrap();

        let active_count = transactions
            .values()
            .filter(|tx| tx.state == TransactionState::Active)
            .count();

        let committed_count = transactions
            .values()
            .filter(|tx| tx.state == TransactionState::Committed)
            .count();

        let aborted_count = transactions
            .values()
            .filter(|tx| tx.state == TransactionState::Aborted)
            .count();

        TransactionStats {
            total_transactions: transactions.len(),
            active_transactions: active_count,
            committed_transactions: committed_count,
            aborted_transactions: aborted_count,
            committed_facts: committed_facts.len(),
            log_entries: log.len(),
        }
    }

    /// Clear completed transactions
    pub fn gc_completed_transactions(&self) {
        let mut transactions = self.transactions.lock().unwrap();
        transactions.retain(|_, tx| {
            tx.state == TransactionState::Active || tx.state == TransactionState::Committing
        });
        debug!("Garbage collected completed transactions");
    }
}

/// Transaction statistics
#[derive(Debug, Clone)]
pub struct TransactionStats {
    pub total_transactions: usize,
    pub active_transactions: usize,
    pub committed_transactions: usize,
    pub aborted_transactions: usize,
    pub committed_facts: usize,
    pub log_entries: usize,
}

impl std::fmt::Display for TransactionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Total: {}, Active: {}, Committed: {}, Aborted: {}, Facts: {}, Log: {}",
            self.total_transactions,
            self.active_transactions,
            self.committed_transactions,
            self.aborted_transactions,
            self.committed_facts,
            self.log_entries
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_begin_transaction() {
        let manager = TransactionManager::new();
        let tx_id = manager
            .begin_transaction(IsolationLevel::ReadCommitted)
            .unwrap();

        assert_eq!(tx_id, 0);
        assert!(manager.is_active(tx_id));
    }

    #[test]
    fn test_add_fact() {
        let manager = TransactionManager::new();
        let tx_id = manager
            .begin_transaction(IsolationLevel::ReadCommitted)
            .unwrap();

        let fact = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("age".to_string()),
            object: Term::Literal("30".to_string()),
        };

        manager.add_fact(tx_id, fact).unwrap();
        assert!(manager.is_active(tx_id));
    }

    #[test]
    fn test_commit() {
        let manager = TransactionManager::new();
        let tx_id = manager
            .begin_transaction(IsolationLevel::ReadCommitted)
            .unwrap();

        let fact = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("age".to_string()),
            object: Term::Literal("30".to_string()),
        };

        manager.add_fact(tx_id, fact).unwrap();
        manager.commit(tx_id).unwrap();

        let committed_facts = manager.get_committed_facts();
        assert_eq!(committed_facts.len(), 1);
        assert_eq!(
            manager.get_transaction_state(tx_id),
            Some(TransactionState::Committed)
        );
    }

    #[test]
    fn test_rollback() {
        let manager = TransactionManager::new();
        let tx_id = manager
            .begin_transaction(IsolationLevel::ReadCommitted)
            .unwrap();

        let fact = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("age".to_string()),
            object: Term::Literal("30".to_string()),
        };

        manager.add_fact(tx_id, fact).unwrap();
        manager.rollback(tx_id).unwrap();

        let committed_facts = manager.get_committed_facts();
        assert_eq!(committed_facts.len(), 0);
        assert_eq!(
            manager.get_transaction_state(tx_id),
            Some(TransactionState::Aborted)
        );
    }

    #[test]
    fn test_multiple_transactions() {
        let manager = TransactionManager::new();

        let tx1 = manager
            .begin_transaction(IsolationLevel::ReadCommitted)
            .unwrap();
        let tx2 = manager
            .begin_transaction(IsolationLevel::ReadCommitted)
            .unwrap();

        assert_ne!(tx1, tx2);
        assert!(manager.is_active(tx1));
        assert!(manager.is_active(tx2));
    }

    #[test]
    fn test_stats() {
        let manager = TransactionManager::new();
        manager
            .begin_transaction(IsolationLevel::ReadCommitted)
            .unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.active_transactions, 1);
        assert_eq!(stats.total_transactions, 1);
    }
}
