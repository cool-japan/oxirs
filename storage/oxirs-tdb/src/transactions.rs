//! # TDB Transaction Support
//!
//! Transaction management for TDB storage.

use anyhow::Result;
use std::collections::HashMap;

/// Transaction isolation levels
#[derive(Debug, Clone, Copy)]
pub enum IsolationLevel {
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

/// Transaction state
#[derive(Debug, Clone, Copy)]
pub enum TransactionState {
    Active,
    Committed,
    Aborted,
}

/// Transaction manager
pub struct TransactionManager {
    active_transactions: HashMap<String, TransactionInfo>,
    next_transaction_id: u64,
}

/// Transaction information
#[derive(Debug, Clone)]
pub struct TransactionInfo {
    pub id: String,
    pub isolation_level: IsolationLevel,
    pub state: TransactionState,
    pub start_time: std::time::SystemTime,
}

impl TransactionManager {
    pub fn new() -> Self {
        Self {
            active_transactions: HashMap::new(),
            next_transaction_id: 1,
        }
    }
    
    /// Begin a new transaction
    pub fn begin_transaction(&mut self, isolation_level: IsolationLevel) -> Result<String> {
        let tx_id = format!("tx_{}", self.next_transaction_id);
        self.next_transaction_id += 1;
        
        let tx_info = TransactionInfo {
            id: tx_id.clone(),
            isolation_level,
            state: TransactionState::Active,
            start_time: std::time::SystemTime::now(),
        };
        
        self.active_transactions.insert(tx_id.clone(), tx_info);
        Ok(tx_id)
    }
    
    /// Commit a transaction
    pub fn commit_transaction(&mut self, tx_id: &str) -> Result<()> {
        if let Some(tx_info) = self.active_transactions.get_mut(tx_id) {
            if !matches!(tx_info.state, TransactionState::Active) {
                return Err(anyhow::anyhow!("Transaction {} is not active", tx_id));
            }
            
            tx_info.state = TransactionState::Committed;
            self.active_transactions.remove(tx_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Transaction {} not found", tx_id))
        }
    }
    
    /// Abort a transaction
    pub fn abort_transaction(&mut self, tx_id: &str) -> Result<()> {
        if let Some(tx_info) = self.active_transactions.get_mut(tx_id) {
            if !matches!(tx_info.state, TransactionState::Active) {
                return Err(anyhow::anyhow!("Transaction {} is not active", tx_id));
            }
            
            tx_info.state = TransactionState::Aborted;
            self.active_transactions.remove(tx_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Transaction {} not found", tx_id))
        }
    }
    
    /// Get transaction info
    pub fn get_transaction_info(&self, tx_id: &str) -> Option<&TransactionInfo> {
        self.active_transactions.get(tx_id)
    }
    
    /// List active transactions
    pub fn active_transactions(&self) -> Vec<&TransactionInfo> {
        self.active_transactions.values().collect()
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}