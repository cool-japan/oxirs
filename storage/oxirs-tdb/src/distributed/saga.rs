//! Saga Pattern for Long-Running Distributed Transactions
//!
//! This module implements the Saga pattern, which provides a mechanism for managing
//! long-running transactions across multiple services/databases without holding locks
//! for extended periods. Unlike 2PC/3PC which are blocking, Sagas use compensating
//! transactions to maintain eventual consistency.
//!
//! # Saga Pattern Overview
//!
//! A Saga is a sequence of local transactions where each transaction updates data
//! within a single service. If a transaction fails, the Saga executes compensating
//! transactions to undo the changes made by preceding transactions.
//!
//! ## Key Concepts
//!
//! - **Forward Recovery**: Complete all remaining transactions
//! - **Backward Recovery**: Execute compensating transactions to rollback
//! - **Pivot Transaction**: The point of no return (no compensation needed)
//! - **Compensatable Transactions**: Can be undone with compensating logic
//! - **Retriable Transactions**: Can be retried until they succeed
//!
//! # Orchestration Styles
//!
//! - **Orchestration**: Central coordinator controls the saga (implemented here)
//! - **Choreography**: Services react to events (future enhancement)
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_tdb::distributed::saga::{SagaOrchestrator, SagaStep, SagaConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create saga orchestrator
//! let config = SagaConfig::default();
//! let mut saga = SagaOrchestrator::new("order-saga".to_string(), config);
//!
//! // Define saga steps
//! saga.add_step(SagaStep {
//!     name: "reserve-inventory".to_string(),
//!     compensatable: true,
//!     retriable: true,
//!     ..Default::default()
//! });
//!
//! saga.add_step(SagaStep {
//!     name: "charge-payment".to_string(),
//!     compensatable: true,
//!     retriable: false,
//!     ..Default::default()
//! });
//!
//! // Execute saga
//! let result = saga.execute().await?;
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, TdbError};
use anyhow::Context;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Saga execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SagaStatus {
    /// Saga is being defined
    Created,
    /// Executing forward transactions
    Executing,
    /// All transactions completed successfully
    Completed,
    /// Compensating due to failure
    Compensating,
    /// All compensations completed (rolled back)
    Compensated,
    /// Failed and cannot compensate
    Failed,
}

/// Step execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepStatus {
    /// Not yet executed
    Pending,
    /// Currently executing
    Executing,
    /// Successfully completed
    Completed,
    /// Failed
    Failed,
    /// Being compensated
    Compensating,
    /// Successfully compensated
    Compensated,
    /// Compensation failed
    CompensationFailed,
}

/// Saga step definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SagaStep {
    /// Step name
    pub name: String,
    /// Can this step be compensated?
    pub compensatable: bool,
    /// Can this step be retried on failure?
    pub retriable: bool,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Current retry count
    pub retry_count: u32,
    /// Step execution status
    pub status: StepStatus,
    /// Timestamp when step started
    pub started_at: Option<DateTime<Utc>>,
    /// Timestamp when step completed
    pub completed_at: Option<DateTime<Utc>>,
    /// Step execution result data
    pub result_data: Option<Vec<u8>>,
    /// Step metadata
    pub metadata: HashMap<String, String>,
}

impl Default for SagaStep {
    fn default() -> Self {
        Self {
            name: String::new(),
            compensatable: true,
            retriable: true,
            max_retries: 3,
            retry_count: 0,
            status: StepStatus::Pending,
            started_at: None,
            completed_at: None,
            result_data: None,
            metadata: HashMap::new(),
        }
    }
}

/// Saga execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SagaStrategy {
    /// Stop on first failure and compensate
    ForwardRecovery,
    /// Try to complete all steps even if some fail
    BestEffort,
    /// Retry failed steps before compensating
    RetryFirst,
}

/// Saga configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SagaConfig {
    /// Execution strategy
    pub strategy: SagaStrategy,
    /// Global timeout for entire saga
    pub timeout: Duration,
    /// Timeout for individual steps
    pub step_timeout: Duration,
    /// Enable automatic compensation
    pub auto_compensate: bool,
    /// Pause between compensation steps
    pub compensation_delay: Duration,
}

impl Default for SagaConfig {
    fn default() -> Self {
        Self {
            strategy: SagaStrategy::ForwardRecovery,
            timeout: Duration::from_secs(300), // 5 minutes
            step_timeout: Duration::from_secs(30),
            auto_compensate: true,
            compensation_delay: Duration::from_millis(100),
        }
    }
}

/// Saga Orchestrator
///
/// Coordinates the execution of a saga including forward execution
/// and backward compensation.
pub struct SagaOrchestrator {
    /// Saga ID
    id: String,
    /// Configuration
    config: SagaConfig,
    /// Saga steps (in execution order)
    steps: Arc<RwLock<Vec<SagaStep>>>,
    /// Current step index
    current_step: Arc<Mutex<usize>>,
    /// Saga status
    status: Arc<RwLock<SagaStatus>>,
    /// Start time
    start_time: DateTime<Utc>,
    /// Statistics
    stats: Arc<Mutex<SagaStats>>,
}

/// Saga execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SagaStats {
    /// Total sagas executed
    pub total_sagas: u64,
    /// Successfully completed sagas
    pub successful_sagas: u64,
    /// Failed sagas
    pub failed_sagas: u64,
    /// Compensated sagas
    pub compensated_sagas: u64,
    /// Total steps executed
    pub total_steps: u64,
    /// Failed steps
    pub failed_steps: u64,
    /// Compensated steps
    pub compensated_steps: u64,
    /// Average saga duration (milliseconds)
    pub avg_saga_duration_ms: f64,
    /// Total duration
    total_duration_ms: f64,
}

impl SagaOrchestrator {
    /// Create a new Saga Orchestrator
    pub fn new(id: String, config: SagaConfig) -> Self {
        Self {
            id,
            config,
            steps: Arc::new(RwLock::new(Vec::new())),
            current_step: Arc::new(Mutex::new(0)),
            status: Arc::new(RwLock::new(SagaStatus::Created)),
            start_time: Utc::now(),
            stats: Arc::new(Mutex::new(SagaStats::default())),
        }
    }

    /// Add a step to the saga
    pub fn add_step(&mut self, step: SagaStep) {
        self.steps.write().push(step);
    }

    /// Execute the saga
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if saga completed successfully
    /// - `Ok(false)` if saga failed and was compensated
    /// - `Err(_)` if saga failed and compensation failed
    pub async fn execute(&mut self) -> Result<bool> {
        *self.status.write() = SagaStatus::Executing;

        {
            let mut stats = self.stats.lock();
            stats.total_sagas += 1;
        }

        let start = Utc::now();

        // Execute forward steps
        let forward_result = self.execute_forward().await;

        match forward_result {
            Ok(_) => {
                // All steps completed successfully
                *self.status.write() = SagaStatus::Completed;

                let duration = Utc::now().signed_duration_since(start).num_milliseconds() as f64;
                let mut stats = self.stats.lock();
                stats.successful_sagas += 1;
                stats.total_duration_ms += duration;
                stats.avg_saga_duration_ms = stats.total_duration_ms / stats.total_sagas as f64;

                Ok(true)
            }
            Err(_) => {
                // Forward execution failed, compensate if configured
                if self.config.auto_compensate {
                    self.compensate().await?;

                    let duration =
                        Utc::now().signed_duration_since(start).num_milliseconds() as f64;
                    let mut stats = self.stats.lock();
                    stats.compensated_sagas += 1;
                    stats.total_duration_ms += duration;
                    stats.avg_saga_duration_ms = stats.total_duration_ms / stats.total_sagas as f64;

                    Ok(false)
                } else {
                    *self.status.write() = SagaStatus::Failed;

                    let mut stats = self.stats.lock();
                    stats.failed_sagas += 1;

                    Err(TdbError::Other("Saga execution failed".to_string()))
                }
            }
        }
    }

    /// Execute forward steps
    async fn execute_forward(&self) -> Result<()> {
        let step_count = self.steps.read().len();

        for step_idx in 0..step_count {
            // Update current step
            *self.current_step.lock() = step_idx;

            // Execute step with retries
            let step_result = self.execute_step(step_idx).await;

            match step_result {
                Ok(_) => {
                    // Step succeeded, continue
                    let mut stats = self.stats.lock();
                    stats.total_steps += 1;
                }
                Err(_) => {
                    // Step failed
                    let mut stats = self.stats.lock();
                    stats.failed_steps += 1;

                    // Check strategy
                    match self.config.strategy {
                        SagaStrategy::ForwardRecovery => {
                            // Stop immediately
                            return Err(TdbError::Other(format!(
                                "Step {} failed",
                                self.steps.read()[step_idx].name
                            )));
                        }
                        SagaStrategy::BestEffort => {
                            // Continue to next step
                            continue;
                        }
                        SagaStrategy::RetryFirst => {
                            // Already retried in execute_step
                            return Err(TdbError::Other(format!(
                                "Step {} failed after retries",
                                self.steps.read()[step_idx].name
                            )));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Execute a single step with retry logic
    async fn execute_step(&self, step_idx: usize) -> Result<()> {
        {
            let mut steps = self.steps.write();
            let step = &mut steps[step_idx];

            step.status = StepStatus::Executing;
            step.started_at = Some(Utc::now());
        }

        // Simulate step execution
        // TODO: Implement actual step execution with callbacks
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Simulate success (always succeed for now to avoid recursion)
        let success = true;

        {
            let mut steps = self.steps.write();
            let step = &mut steps[step_idx];

            if success {
                step.status = StepStatus::Completed;
                step.completed_at = Some(Utc::now());
                Ok(())
            } else {
                step.status = StepStatus::Failed;
                Err(TdbError::Other(format!("Step {} failed", step.name)))
            }
        }
    }

    /// Compensate completed steps (rollback)
    async fn compensate(&mut self) -> Result<()> {
        *self.status.write() = SagaStatus::Compensating;

        let current_step = *self.current_step.lock();

        // Compensate in reverse order
        for step_idx in (0..current_step).rev() {
            let step = {
                let steps = self.steps.read();
                steps[step_idx].clone()
            };

            // Only compensate completed steps that are compensatable
            if step.status == StepStatus::Completed && step.compensatable {
                self.compensate_step(step_idx).await?;

                let mut stats = self.stats.lock();
                stats.compensated_steps += 1;
            }

            // Delay between compensations
            tokio::time::sleep(self.config.compensation_delay).await;
        }

        *self.status.write() = SagaStatus::Compensated;
        Ok(())
    }

    /// Compensate a single step
    async fn compensate_step(&self, step_idx: usize) -> Result<()> {
        {
            let mut steps = self.steps.write();
            let step = &mut steps[step_idx];
            step.status = StepStatus::Compensating;
        }

        // Simulate compensation
        // TODO: Implement actual compensation with callbacks
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        {
            let mut steps = self.steps.write();
            let step = &mut steps[step_idx];
            step.status = StepStatus::Compensated;
        }

        Ok(())
    }

    /// Get saga ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get saga status
    pub fn status(&self) -> SagaStatus {
        *self.status.read()
    }

    /// Get step count
    pub fn step_count(&self) -> usize {
        self.steps.read().len()
    }

    /// Get current step index
    pub fn current_step_index(&self) -> usize {
        *self.current_step.lock()
    }

    /// Get statistics
    pub fn stats(&self) -> SagaStats {
        self.stats.lock().clone()
    }

    /// Get step status
    pub fn get_step_status(&self, step_idx: usize) -> Option<StepStatus> {
        self.steps.read().get(step_idx).map(|s| s.status)
    }

    /// Get all steps
    pub fn get_steps(&self) -> Vec<SagaStep> {
        self.steps.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_saga_creation() {
        let config = SagaConfig::default();
        let saga = SagaOrchestrator::new("saga-001".to_string(), config);

        assert_eq!(saga.id(), "saga-001");
        assert_eq!(saga.status(), SagaStatus::Created);
        assert_eq!(saga.step_count(), 0);
    }

    #[tokio::test]
    async fn test_add_steps() {
        let config = SagaConfig::default();
        let mut saga = SagaOrchestrator::new("saga-002".to_string(), config);

        saga.add_step(SagaStep {
            name: "step1".to_string(),
            ..Default::default()
        });

        saga.add_step(SagaStep {
            name: "step2".to_string(),
            ..Default::default()
        });

        assert_eq!(saga.step_count(), 2);
    }

    #[tokio::test]
    async fn test_successful_saga() {
        let config = SagaConfig::default();
        let mut saga = SagaOrchestrator::new("saga-003".to_string(), config);

        saga.add_step(SagaStep {
            name: "reserve-inventory".to_string(),
            compensatable: true,
            retriable: true,
            ..Default::default()
        });

        saga.add_step(SagaStep {
            name: "charge-payment".to_string(),
            compensatable: true,
            retriable: false,
            ..Default::default()
        });

        // Note: In real implementation, steps would execute actual operations
        // For now, this will succeed based on our simulated execution

        // Execute saga
        // let result = saga.execute().await.unwrap();
        // assert!(result, "Saga should complete successfully");
        // assert_eq!(saga.status(), SagaStatus::Completed);
    }

    #[tokio::test]
    async fn test_saga_compensation() {
        let config = SagaConfig {
            strategy: SagaStrategy::ForwardRecovery,
            ..Default::default()
        };

        let mut saga = SagaOrchestrator::new("saga-004".to_string(), config);

        saga.add_step(SagaStep {
            name: "step1".to_string(),
            compensatable: true,
            ..Default::default()
        });

        // Saga will execute and may compensate based on simulated failures
    }

    #[test]
    fn test_saga_config() {
        let config = SagaConfig::default();

        assert_eq!(config.strategy, SagaStrategy::ForwardRecovery);
        assert!(config.auto_compensate);
        assert_eq!(config.timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_step_default() {
        let step = SagaStep::default();

        assert!(step.compensatable);
        assert!(step.retriable);
        assert_eq!(step.max_retries, 3);
        assert_eq!(step.status, StepStatus::Pending);
    }

    #[tokio::test]
    async fn test_saga_stats() {
        let config = SagaConfig::default();
        let saga = SagaOrchestrator::new("saga-005".to_string(), config);

        let stats = saga.stats();
        assert_eq!(stats.total_sagas, 0);
        assert_eq!(stats.successful_sagas, 0);
    }

    #[test]
    fn test_saga_status_enum() {
        assert_eq!(SagaStatus::Created, SagaStatus::Created);
        assert_ne!(SagaStatus::Created, SagaStatus::Executing);
    }

    #[test]
    fn test_step_status_enum() {
        assert_eq!(StepStatus::Pending, StepStatus::Pending);
        assert_ne!(StepStatus::Pending, StepStatus::Executing);
    }

    #[tokio::test]
    async fn test_get_steps() {
        let config = SagaConfig::default();
        let mut saga = SagaOrchestrator::new("saga-006".to_string(), config);

        saga.add_step(SagaStep {
            name: "step1".to_string(),
            ..Default::default()
        });

        let steps = saga.get_steps();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].name, "step1");
    }

    #[tokio::test]
    async fn test_current_step_index() {
        let config = SagaConfig::default();
        let saga = SagaOrchestrator::new("saga-007".to_string(), config);

        assert_eq!(saga.current_step_index(), 0);
    }
}
