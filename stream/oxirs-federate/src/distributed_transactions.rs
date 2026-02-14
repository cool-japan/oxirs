//! # Distributed Transaction Coordinator
//!
//! Provides distributed transaction support for federated queries using:
//! - **Two-Phase Commit (2PC)** for strong consistency
//! - **Saga Pattern** for long-running distributed transactions with compensation

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use scirs2_core::random::rng;
use scirs2_core::RngCore;

/// Distributed transaction coordinator
#[derive(Debug, Clone)]
pub struct DistributedTransactionCoordinator {
    transactions: Arc<RwLock<HashMap<String, Transaction>>>,
    config: TransactionConfig,
}

/// Transaction configuration
#[derive(Debug, Clone)]
pub struct TransactionConfig {
    /// Timeout for transaction execution
    pub timeout: Duration,
    /// Maximum retry attempts for failed operations
    pub max_retries: u32,
    /// Enable automatic compensation on failure
    pub enable_auto_compensation: bool,
    /// Transaction isolation level
    pub isolation_level: IsolationLevel,
    /// Preferred transaction protocol
    pub protocol: TransactionProtocol,
}

impl Default for TransactionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_retries: 3,
            enable_auto_compensation: true,
            isolation_level: IsolationLevel::ReadCommitted,
            protocol: TransactionProtocol::TwoPhaseCommit,
        }
    }
}

/// Transaction isolation levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

/// Transaction protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionProtocol {
    /// Two-Phase Commit for strong consistency
    TwoPhaseCommit,
    /// Saga pattern for long-running transactions
    Saga,
    /// Best-effort eventual consistency
    EventualConsistency,
}

/// Distributed transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub transaction_id: String,
    pub protocol: TransactionProtocol,
    pub state: TransactionState,
    pub participants: Vec<Participant>,
    pub operations: Vec<Operation>,
    pub saga_log: Option<SagaLog>,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub timeout: Duration,
}

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionState {
    /// Initial state
    Initiated,
    /// Preparing phase (2PC)
    Preparing,
    /// Prepared and waiting for commit
    Prepared,
    /// Committing changes
    Committing,
    /// Successfully committed
    Committed,
    /// Rolling back changes
    Aborting,
    /// Successfully rolled back
    Aborted,
    /// Compensating (Saga)
    Compensating,
    /// Successfully compensated
    Compensated,
    /// Failed state
    Failed,
}

/// Transaction participant (service)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    pub service_id: String,
    pub service_url: String,
    pub state: ParticipantState,
    pub prepared_at: Option<SystemTime>,
    pub committed_at: Option<SystemTime>,
    pub aborted_at: Option<SystemTime>,
}

/// Participant state in transaction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticipantState {
    Idle,
    Preparing,
    Prepared,
    Committing,
    Committed,
    Aborting,
    Aborted,
    Failed,
}

/// Transaction operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub operation_id: String,
    pub service_id: String,
    pub operation_type: OperationType,
    pub query: String,
    pub compensation_query: Option<String>,
    pub state: OperationState,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
}

/// Operation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationType {
    Read,
    Write,
    Update,
    Delete,
    Custom(String),
}

/// Operation state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationState {
    Pending,
    Executing,
    Completed,
    Failed,
    Compensating,
    Compensated,
}

/// Saga log for compensation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SagaLog {
    pub saga_id: String,
    pub steps: Vec<SagaStep>,
    pub compensation_order: Vec<String>, // Step IDs in reverse order
    pub current_step: usize,
}

/// Saga transaction step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SagaStep {
    pub step_id: String,
    pub operation: Operation,
    pub compensation: Option<Operation>,
    pub state: SagaStepState,
}

/// Saga step state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SagaStepState {
    Pending,
    Executing,
    Completed,
    Failed,
    Compensating,
    Compensated,
}

/// Transaction result
#[derive(Debug, Clone)]
pub struct TransactionResult {
    pub transaction_id: String,
    pub success: bool,
    pub final_state: TransactionState,
    pub results: Vec<OperationResult>,
    pub errors: Vec<String>,
    pub execution_time: Duration,
}

/// Operation execution result
#[derive(Debug, Clone)]
pub struct OperationResult {
    pub operation_id: String,
    pub service_id: String,
    pub success: bool,
    pub data: Option<serde_json::Value>,
    pub error: Option<String>,
}

impl DistributedTransactionCoordinator {
    /// Create a new transaction coordinator
    pub fn new() -> Self {
        Self {
            transactions: Arc::new(RwLock::new(HashMap::new())),
            config: TransactionConfig::default(),
        }
    }

    /// Create coordinator with custom configuration
    pub fn with_config(config: TransactionConfig) -> Self {
        Self {
            transactions: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Begin a new distributed transaction
    pub async fn begin_transaction(
        &self,
        participants: Vec<Participant>,
        operations: Vec<Operation>,
    ) -> Result<String> {
        let transaction_id = Uuid::new_v4().to_string();

        let mut rng = rng();
        debug!(
            "Beginning transaction {} with {} participants (random seed: {})",
            transaction_id,
            participants.len(),
            rng.next_u64()
        );

        let saga_log = if self.config.protocol == TransactionProtocol::Saga {
            Some(SagaLog {
                saga_id: Uuid::new_v4().to_string(),
                steps: operations
                    .iter()
                    .map(|op| SagaStep {
                        step_id: Uuid::new_v4().to_string(),
                        operation: op.clone(),
                        compensation: op.compensation_query.as_ref().map(|comp| Operation {
                            operation_id: Uuid::new_v4().to_string(),
                            service_id: op.service_id.clone(),
                            operation_type: op.operation_type.clone(),
                            query: comp.clone(),
                            compensation_query: None,
                            state: OperationState::Pending,
                            result: None,
                            error: None,
                        }),
                        state: SagaStepState::Pending,
                    })
                    .collect(),
                compensation_order: vec![],
                current_step: 0,
            })
        } else {
            None
        };

        let transaction = Transaction {
            transaction_id: transaction_id.clone(),
            protocol: self.config.protocol,
            state: TransactionState::Initiated,
            participants,
            operations,
            saga_log,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
            timeout: self.config.timeout,
        };

        let mut txns = self.transactions.write().await;
        txns.insert(transaction_id.clone(), transaction);

        info!("Transaction {} initiated successfully", transaction_id);
        Ok(transaction_id)
    }

    /// Execute transaction using the configured protocol
    pub async fn execute_transaction(&self, transaction_id: &str) -> Result<TransactionResult> {
        let protocol = {
            let txns = self.transactions.read().await;
            let txn = txns
                .get(transaction_id)
                .ok_or_else(|| anyhow!("Transaction not found: {}", transaction_id))?;
            txn.protocol
        };

        match protocol {
            TransactionProtocol::TwoPhaseCommit => {
                self.execute_two_phase_commit(transaction_id).await
            }
            TransactionProtocol::Saga => self.execute_saga(transaction_id).await,
            TransactionProtocol::EventualConsistency => {
                self.execute_eventual_consistency(transaction_id).await
            }
        }
    }

    /// Execute Two-Phase Commit protocol
    async fn execute_two_phase_commit(&self, transaction_id: &str) -> Result<TransactionResult> {
        let start_time = SystemTime::now();
        info!("Executing 2PC for transaction {}", transaction_id);

        // Phase 1: Prepare
        self.update_transaction_state(transaction_id, TransactionState::Preparing)
            .await?;

        let prepare_result = self.prepare_phase(transaction_id).await;

        if !prepare_result {
            warn!("Prepare phase failed for transaction {}", transaction_id);
            return self.abort_transaction(transaction_id, start_time).await;
        }

        self.update_transaction_state(transaction_id, TransactionState::Prepared)
            .await?;

        // Phase 2: Commit
        self.update_transaction_state(transaction_id, TransactionState::Committing)
            .await?;

        let commit_result = self.commit_phase(transaction_id).await;

        if !commit_result {
            error!("Commit phase failed for transaction {}", transaction_id);
            return self.abort_transaction(transaction_id, start_time).await;
        }

        self.update_transaction_state(transaction_id, TransactionState::Committed)
            .await?;

        let execution_time = SystemTime::now().duration_since(start_time)?;

        let (results, errors) = self.collect_results(transaction_id).await?;

        Ok(TransactionResult {
            transaction_id: transaction_id.to_string(),
            success: true,
            final_state: TransactionState::Committed,
            results,
            errors,
            execution_time,
        })
    }

    /// Execute Saga pattern
    async fn execute_saga(&self, transaction_id: &str) -> Result<TransactionResult> {
        let start_time = SystemTime::now();
        info!("Executing Saga for transaction {}", transaction_id);

        let saga_steps = {
            let txns = self.transactions.read().await;
            let txn = txns
                .get(transaction_id)
                .ok_or_else(|| anyhow!("Transaction not found"))?;
            txn.saga_log
                .as_ref()
                .ok_or_else(|| anyhow!("No saga log found"))?
                .steps
                .len()
        };

        // Execute saga steps sequentially
        for step_idx in 0..saga_steps {
            let step_result = self.execute_saga_step(transaction_id, step_idx).await;

            if let Err(e) = step_result {
                warn!(
                    "Saga step {} failed for transaction {}: {}",
                    step_idx, transaction_id, e
                );

                // Begin compensation
                if self.config.enable_auto_compensation {
                    return self
                        .compensate_saga(transaction_id, step_idx, start_time)
                        .await;
                } else {
                    return self
                        .fail_transaction(transaction_id, start_time, e.to_string())
                        .await;
                }
            }
        }

        self.update_transaction_state(transaction_id, TransactionState::Committed)
            .await?;

        let execution_time = SystemTime::now().duration_since(start_time)?;
        let (results, errors) = self.collect_results(transaction_id).await?;

        Ok(TransactionResult {
            transaction_id: transaction_id.to_string(),
            success: true,
            final_state: TransactionState::Committed,
            results,
            errors,
            execution_time,
        })
    }

    /// Execute eventual consistency pattern
    async fn execute_eventual_consistency(
        &self,
        transaction_id: &str,
    ) -> Result<TransactionResult> {
        let start_time = SystemTime::now();
        info!(
            "Executing eventual consistency for transaction {}",
            transaction_id
        );

        // Execute all operations in parallel without coordination
        let operations = {
            let txns = self.transactions.read().await;
            let txn = txns
                .get(transaction_id)
                .ok_or_else(|| anyhow!("Transaction not found"))?;
            txn.operations.clone()
        };

        let mut results = Vec::new();
        let mut errors = Vec::new();

        for operation in operations {
            match self.execute_operation(&operation).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!("Operation {} failed: {}", operation.operation_id, e);
                    errors.push(e.to_string());
                }
            }
        }

        let success = !results.is_empty();
        let final_state = if success {
            TransactionState::Committed
        } else {
            TransactionState::Failed
        };

        self.update_transaction_state(transaction_id, final_state)
            .await?;

        let execution_time = SystemTime::now().duration_since(start_time)?;

        Ok(TransactionResult {
            transaction_id: transaction_id.to_string(),
            success,
            final_state,
            results,
            errors,
            execution_time,
        })
    }

    /// Prepare phase for 2PC
    async fn prepare_phase(&self, transaction_id: &str) -> bool {
        let participants = {
            let txns = self.transactions.read().await;
            let txn = txns.get(transaction_id).expect("transaction should exist");
            txn.participants.clone()
        };

        for participant in participants {
            let prepare_result = self.send_prepare(&participant).await;

            if !prepare_result {
                return false;
            }
        }

        true
    }

    /// Commit phase for 2PC
    async fn commit_phase(&self, transaction_id: &str) -> bool {
        let participants = {
            let txns = self.transactions.read().await;
            let txn = txns.get(transaction_id).expect("transaction should exist");
            txn.participants.clone()
        };

        for participant in participants {
            let commit_result = self.send_commit(&participant).await;

            if !commit_result {
                return false;
            }
        }

        true
    }

    /// Send prepare message to participant
    async fn send_prepare(&self, participant: &Participant) -> bool {
        debug!("Sending PREPARE to service {}", participant.service_id);

        // Simulate prepare request (in real implementation, send HTTP/gRPC request)
        // For now, randomly succeed/fail based on a probability
        let mut rng = rng();
        let success_probability = 0.95; // 95% success rate

        (rng.next_u64() as f64 / u64::MAX as f64) < success_probability
    }

    /// Send commit message to participant
    async fn send_commit(&self, participant: &Participant) -> bool {
        debug!("Sending COMMIT to service {}", participant.service_id);

        // Simulate commit request
        let mut rng = rng();
        let success_probability = 0.98; // 98% success rate for commit

        (rng.next_u64() as f64 / u64::MAX as f64) < success_probability
    }

    /// Send abort message to participant
    async fn send_abort(&self, participant: &Participant) -> bool {
        debug!("Sending ABORT to service {}", participant.service_id);

        // Abort should always succeed
        true
    }

    /// Execute a saga step
    async fn execute_saga_step(&self, transaction_id: &str, step_idx: usize) -> Result<()> {
        let operation = {
            let txns = self.transactions.read().await;
            let txn = txns.get(transaction_id).expect("transaction should exist");
            txn.saga_log
                .as_ref()
                .expect("saga log should exist")
                .steps
                .get(step_idx)
                .expect("saga step should exist")
                .operation
                .clone()
        };

        debug!(
            "Executing saga step {} for transaction {}",
            step_idx, transaction_id
        );

        self.execute_operation(&operation).await?;

        // Update step state
        let mut txns = self.transactions.write().await;
        let txn = txns
            .get_mut(transaction_id)
            .expect("transaction should exist");
        if let Some(saga_log) = &mut txn.saga_log {
            saga_log.steps[step_idx].state = SagaStepState::Completed;
            saga_log.current_step = step_idx + 1;
            saga_log
                .compensation_order
                .push(saga_log.steps[step_idx].step_id.clone());
        }

        Ok(())
    }

    /// Compensate saga on failure
    async fn compensate_saga(
        &self,
        transaction_id: &str,
        failed_step: usize,
        start_time: SystemTime,
    ) -> Result<TransactionResult> {
        info!(
            "Compensating saga for transaction {} from step {}",
            transaction_id, failed_step
        );

        self.update_transaction_state(transaction_id, TransactionState::Compensating)
            .await?;

        // Execute compensation in reverse order
        for step_idx in (0..failed_step).rev() {
            if let Err(e) = self.compensate_saga_step(transaction_id, step_idx).await {
                error!(
                    "Failed to compensate step {} for transaction {}: {}",
                    step_idx, transaction_id, e
                );
            }
        }

        self.update_transaction_state(transaction_id, TransactionState::Compensated)
            .await?;

        let execution_time = SystemTime::now().duration_since(start_time)?;
        let (results, errors) = self.collect_results(transaction_id).await?;

        Ok(TransactionResult {
            transaction_id: transaction_id.to_string(),
            success: false,
            final_state: TransactionState::Compensated,
            results,
            errors,
            execution_time,
        })
    }

    /// Compensate a saga step
    async fn compensate_saga_step(&self, transaction_id: &str, step_idx: usize) -> Result<()> {
        let compensation = {
            let txns = self.transactions.read().await;
            let txn = txns.get(transaction_id).expect("transaction should exist");
            txn.saga_log
                .as_ref()
                .expect("saga log should exist")
                .steps
                .get(step_idx)
                .expect("saga step should exist")
                .compensation
                .clone()
        };

        if let Some(comp_op) = compensation {
            debug!(
                "Compensating saga step {} for transaction {}",
                step_idx, transaction_id
            );

            self.execute_operation(&comp_op).await?;

            // Update step state
            let mut txns = self.transactions.write().await;
            let txn = txns
                .get_mut(transaction_id)
                .expect("transaction should exist");
            if let Some(saga_log) = &mut txn.saga_log {
                saga_log.steps[step_idx].state = SagaStepState::Compensated;
            }
        }

        Ok(())
    }

    /// Execute a single operation
    async fn execute_operation(&self, operation: &Operation) -> Result<OperationResult> {
        debug!("Executing operation {}", operation.operation_id);

        // Simulate operation execution
        let mut rng = rng();
        let success = (rng.next_u64() as f64 / u64::MAX as f64) < 0.9; // 90% success rate

        if success {
            Ok(OperationResult {
                operation_id: operation.operation_id.clone(),
                service_id: operation.service_id.clone(),
                success: true,
                data: Some(serde_json::json!({"status": "success"})),
                error: None,
            })
        } else {
            Err(anyhow!("Operation failed"))
        }
    }

    /// Abort transaction
    async fn abort_transaction(
        &self,
        transaction_id: &str,
        start_time: SystemTime,
    ) -> Result<TransactionResult> {
        info!("Aborting transaction {}", transaction_id);

        self.update_transaction_state(transaction_id, TransactionState::Aborting)
            .await?;

        let participants = {
            let txns = self.transactions.read().await;
            let txn = txns.get(transaction_id).expect("transaction should exist");
            txn.participants.clone()
        };

        for participant in participants {
            self.send_abort(&participant).await;
        }

        self.update_transaction_state(transaction_id, TransactionState::Aborted)
            .await?;

        let execution_time = SystemTime::now().duration_since(start_time)?;
        let (results, errors) = self.collect_results(transaction_id).await?;

        Ok(TransactionResult {
            transaction_id: transaction_id.to_string(),
            success: false,
            final_state: TransactionState::Aborted,
            results,
            errors,
            execution_time,
        })
    }

    /// Fail transaction
    async fn fail_transaction(
        &self,
        transaction_id: &str,
        start_time: SystemTime,
        error: String,
    ) -> Result<TransactionResult> {
        error!("Transaction {} failed: {}", transaction_id, error);

        self.update_transaction_state(transaction_id, TransactionState::Failed)
            .await?;

        let execution_time = SystemTime::now().duration_since(start_time)?;
        let (results, errors) = self.collect_results(transaction_id).await?;

        Ok(TransactionResult {
            transaction_id: transaction_id.to_string(),
            success: false,
            final_state: TransactionState::Failed,
            results,
            errors: vec![error].into_iter().chain(errors).collect(),
            execution_time,
        })
    }

    /// Update transaction state
    async fn update_transaction_state(
        &self,
        transaction_id: &str,
        new_state: TransactionState,
    ) -> Result<()> {
        let mut txns = self.transactions.write().await;
        let txn = txns
            .get_mut(transaction_id)
            .ok_or_else(|| anyhow!("Transaction not found"))?;

        txn.state = new_state;
        txn.updated_at = SystemTime::now();

        debug!(
            "Transaction {} state updated to {:?}",
            transaction_id, new_state
        );
        Ok(())
    }

    /// Collect results from transaction
    async fn collect_results(
        &self,
        transaction_id: &str,
    ) -> Result<(Vec<OperationResult>, Vec<String>)> {
        let txns = self.transactions.read().await;
        let txn = txns
            .get(transaction_id)
            .ok_or_else(|| anyhow!("Transaction not found"))?;

        let results: Vec<OperationResult> = txn
            .operations
            .iter()
            .filter_map(|op| {
                op.result.as_ref().map(|_| OperationResult {
                    operation_id: op.operation_id.clone(),
                    service_id: op.service_id.clone(),
                    success: op.state == OperationState::Completed,
                    data: op.result.clone(),
                    error: op.error.clone(),
                })
            })
            .collect();

        let errors: Vec<String> = txn
            .operations
            .iter()
            .filter_map(|op| op.error.clone())
            .collect();

        Ok((results, errors))
    }

    /// Get transaction status
    pub async fn get_transaction_status(&self, transaction_id: &str) -> Result<TransactionState> {
        let txns = self.transactions.read().await;
        let txn = txns
            .get(transaction_id)
            .ok_or_else(|| anyhow!("Transaction not found: {}", transaction_id))?;

        Ok(txn.state)
    }

    /// Get all transactions
    pub async fn list_transactions(&self) -> Vec<Transaction> {
        let txns = self.transactions.read().await;
        txns.values().cloned().collect()
    }

    /// Clean up completed transactions older than specified duration
    pub async fn cleanup_transactions(&self, older_than: Duration) -> Result<usize> {
        let mut txns = self.transactions.write().await;
        let cutoff_time = SystemTime::now() - older_than;

        let initial_count = txns.len();
        txns.retain(|_, txn| {
            let is_recent = txn.updated_at > cutoff_time;
            let is_active = !matches!(
                txn.state,
                TransactionState::Committed
                    | TransactionState::Aborted
                    | TransactionState::Compensated
                    | TransactionState::Failed
            );

            is_recent || is_active
        });

        let removed = initial_count - txns.len();
        info!("Cleaned up {} completed transactions", removed);

        Ok(removed)
    }
}

impl Default for DistributedTransactionCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_transaction_coordinator_creation() {
        let coordinator = DistributedTransactionCoordinator::new();
        let txns = coordinator.list_transactions().await;
        assert_eq!(txns.len(), 0);
    }

    #[tokio::test]
    async fn test_begin_transaction() {
        let coordinator = DistributedTransactionCoordinator::new();

        let participants = vec![Participant {
            service_id: "service1".to_string(),
            service_url: "http://localhost:8001".to_string(),
            state: ParticipantState::Idle,
            prepared_at: None,
            committed_at: None,
            aborted_at: None,
        }];

        let operations = vec![Operation {
            operation_id: "op1".to_string(),
            service_id: "service1".to_string(),
            operation_type: OperationType::Write,
            query: "INSERT DATA { ... }".to_string(),
            compensation_query: Some("DELETE DATA { ... }".to_string()),
            state: OperationState::Pending,
            result: None,
            error: None,
        }];

        let txn_id = coordinator
            .begin_transaction(participants, operations)
            .await
            .expect("operation should succeed");

        assert!(!txn_id.is_empty());

        let status = coordinator
            .get_transaction_status(&txn_id)
            .await
            .expect("async operation should succeed");
        assert_eq!(status, TransactionState::Initiated);
    }

    #[tokio::test]
    async fn test_two_phase_commit() {
        let config = TransactionConfig {
            protocol: TransactionProtocol::TwoPhaseCommit,
            ..Default::default()
        };

        let coordinator = DistributedTransactionCoordinator::with_config(config);

        let participants = vec![Participant {
            service_id: "service1".to_string(),
            service_url: "http://localhost:8001".to_string(),
            state: ParticipantState::Idle,
            prepared_at: None,
            committed_at: None,
            aborted_at: None,
        }];

        let operations = vec![Operation {
            operation_id: "op1".to_string(),
            service_id: "service1".to_string(),
            operation_type: OperationType::Write,
            query: "INSERT DATA { ... }".to_string(),
            compensation_query: None,
            state: OperationState::Pending,
            result: None,
            error: None,
        }];

        let txn_id = coordinator
            .begin_transaction(participants, operations)
            .await
            .expect("operation should succeed");

        let result = coordinator
            .execute_transaction(&txn_id)
            .await
            .expect("async operation should succeed");

        assert!(
            result.final_state == TransactionState::Committed
                || result.final_state == TransactionState::Aborted
        );
    }

    #[tokio::test]
    async fn test_saga_pattern() {
        let config = TransactionConfig {
            protocol: TransactionProtocol::Saga,
            ..Default::default()
        };

        let coordinator = DistributedTransactionCoordinator::with_config(config);

        let participants = vec![Participant {
            service_id: "service1".to_string(),
            service_url: "http://localhost:8001".to_string(),
            state: ParticipantState::Idle,
            prepared_at: None,
            committed_at: None,
            aborted_at: None,
        }];

        let operations = vec![
            Operation {
                operation_id: "op1".to_string(),
                service_id: "service1".to_string(),
                operation_type: OperationType::Write,
                query: "INSERT DATA { ... }".to_string(),
                compensation_query: Some("DELETE DATA { ... }".to_string()),
                state: OperationState::Pending,
                result: None,
                error: None,
            },
            Operation {
                operation_id: "op2".to_string(),
                service_id: "service1".to_string(),
                operation_type: OperationType::Update,
                query: "UPDATE DATA { ... }".to_string(),
                compensation_query: Some("UPDATE DATA { ... }".to_string()),
                state: OperationState::Pending,
                result: None,
                error: None,
            },
        ];

        let txn_id = coordinator
            .begin_transaction(participants, operations)
            .await
            .expect("operation should succeed");

        let result = coordinator
            .execute_transaction(&txn_id)
            .await
            .expect("async operation should succeed");

        assert!(
            result.final_state == TransactionState::Committed
                || result.final_state == TransactionState::Compensated
        );
    }

    #[tokio::test]
    async fn test_cleanup_transactions() {
        let coordinator = DistributedTransactionCoordinator::new();

        // Create a completed transaction
        let participants = vec![Participant {
            service_id: "service1".to_string(),
            service_url: "http://localhost:8001".to_string(),
            state: ParticipantState::Idle,
            prepared_at: None,
            committed_at: None,
            aborted_at: None,
        }];

        let operations = vec![Operation {
            operation_id: "op1".to_string(),
            service_id: "service1".to_string(),
            operation_type: OperationType::Write,
            query: "INSERT DATA { ... }".to_string(),
            compensation_query: None,
            state: OperationState::Pending,
            result: None,
            error: None,
        }];

        let _txn_id = coordinator
            .begin_transaction(participants, operations)
            .await
            .expect("operation should succeed");

        // Transactions should not be cleaned up immediately
        let removed = coordinator
            .cleanup_transactions(Duration::from_secs(3600))
            .await
            .expect("operation should succeed");

        assert_eq!(removed, 0);
    }
}
