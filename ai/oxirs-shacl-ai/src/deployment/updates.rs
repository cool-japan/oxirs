//! Update and Rollback Management
//!
//! This module handles deployment updates, rollback strategies,
//! version management, and deployment history tracking.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Duration;

use super::config::{DeploymentStrategy, UpdateStrategy};
use super::types::{DeploymentRecord, DeploymentStatus, UpdateResult, UpdateSpec};
use crate::{Result, ShaclAiError};

/// Maximum number of deployment records retained in history.
const MAX_HISTORY: usize = 100;

/// Update manager
#[derive(Debug)]
pub struct UpdateManager {
    update_strategy: UpdateStrategy,
    rollback_config: RollbackConfig,
    deployment_history: VecDeque<DeploymentRecord>,
}

impl Default for UpdateManager {
    fn default() -> Self {
        Self::new()
    }
}

impl UpdateManager {
    pub fn new() -> Self {
        Self {
            update_strategy: UpdateStrategy::RollingUpdate,
            rollback_config: RollbackConfig {
                auto_rollback: true,
                rollback_triggers: vec![],
                max_rollback_attempts: 3,
            },
            deployment_history: VecDeque::new(),
        }
    }

    /// Perform a deployment update against the configured backend.
    ///
    /// # Fail-loud contract
    ///
    /// Rolling out a new version requires a live orchestration backend, which
    /// is not bundled with this crate. Rather than fabricating a successful
    /// version bump, this validates the spec (returning
    /// [`ShaclAiError::Configuration`] for an invalid target version), records
    /// the failed attempt in [`Self::deployment_history`], and returns
    /// [`ShaclAiError::Unsupported`].
    pub async fn perform_update(&mut self, spec: &UpdateSpec) -> Result<UpdateResult> {
        if spec.target_version.trim().is_empty() {
            return Err(ShaclAiError::Configuration(
                "update target_version must not be empty".to_string(),
            ));
        }

        let strategy = match self.update_strategy {
            UpdateStrategy::RollingUpdate => DeploymentStrategy::RollingUpdate,
            UpdateStrategy::BlueGreenUpdate => DeploymentStrategy::BlueGreen,
            UpdateStrategy::CanaryUpdate => DeploymentStrategy::Canary,
            UpdateStrategy::ImmediateUpdate => DeploymentStrategy::Recreation,
        };

        // Record the (failed) attempt so update/rollback history is real.
        self.record(DeploymentRecord {
            deployment_id: format!("update_{}", chrono::Utc::now().timestamp()),
            version: spec.target_version.clone(),
            timestamp: chrono::Utc::now(),
            strategy,
            status: DeploymentStatus::Failed,
            rollback_info: None,
        });

        Err(ShaclAiError::Unsupported(format!(
            "cannot roll out version '{}': no live orchestration backend is configured \
             (rollback_on_failure = {}, auto_rollback = {})",
            spec.target_version, spec.rollback_on_failure, self.rollback_config.auto_rollback
        )))
    }

    /// Append a record to the bounded deployment history.
    fn record(&mut self, record: DeploymentRecord) {
        if self.deployment_history.len() >= MAX_HISTORY {
            self.deployment_history.pop_front();
        }
        self.deployment_history.push_back(record);
    }

    /// Read-only view of the deployment/update history (most recent last).
    pub fn deployment_history(&self) -> &VecDeque<DeploymentRecord> {
        &self.deployment_history
    }
}

/// Rollback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackConfig {
    pub auto_rollback: bool,
    pub rollback_triggers: Vec<RollbackTrigger>,
    pub max_rollback_attempts: u32,
}

/// Rollback trigger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackTrigger {
    pub trigger_type: RollbackTriggerType,
    pub threshold: f64,
    pub duration: Duration,
}

/// Rollback trigger types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackTriggerType {
    ErrorRate,
    ResponseTime,
    HealthCheckFailure,
    CustomMetric(String),
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    fn spec(version: &str) -> UpdateSpec {
        UpdateSpec {
            target_version: version.to_string(),
            update_strategy: UpdateStrategy::RollingUpdate,
            rollback_on_failure: true,
            health_check_timeout: Duration::from_secs(60),
        }
    }

    /// Regression: `perform_update` must not fabricate a successful version bump
    /// and must record the attempt in history.
    #[tokio::test]
    async fn regression_perform_update_fails_loud_and_records_history() {
        let mut manager = UpdateManager::new();
        let result = manager.perform_update(&spec("2.0.0")).await;
        assert!(matches!(result, Err(ShaclAiError::Unsupported(_))));
        assert_eq!(manager.deployment_history().len(), 1);
        assert_eq!(manager.deployment_history()[0].version, "2.0.0");
    }

    /// Regression: an empty target version is rejected up-front.
    #[tokio::test]
    async fn regression_perform_update_rejects_empty_version() {
        let mut manager = UpdateManager::new();
        let result = manager.perform_update(&spec("")).await;
        assert!(matches!(result, Err(ShaclAiError::Configuration(_))));
        assert!(manager.deployment_history().is_empty());
    }
}
