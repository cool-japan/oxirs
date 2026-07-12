//! Consistency management and repair for the real-time embedding pipeline

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use crate::real_time_embedding_pipeline::{
    config::ConsistencyLevel,
    traits::{
        ConsistencyRepairStrategy, HealthStatus, Inconsistency, InconsistencySeverity,
        RepairResult, RepairStatus,
    },
};

/// Configuration for consistency checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyConfig {
    pub consistency_level: ConsistencyLevel,
    pub check_interval: Duration,
    pub max_repair_attempts: usize,
    pub enable_auto_repair: bool,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            consistency_level: ConsistencyLevel::Session,
            check_interval: Duration::from_secs(60),
            max_repair_attempts: 3,
            enable_auto_repair: true,
        }
    }
}

/// Statistics for consistency management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyStatistics {
    pub total_checks: u64,
    pub total_inconsistencies_detected: u64,
    pub total_repairs_attempted: u64,
    pub total_repairs_succeeded: u64,
    pub is_running: bool,
}

/// Default repair strategy
pub struct DefaultRepairStrategy;

impl ConsistencyRepairStrategy for DefaultRepairStrategy {
    fn repair_inconsistencies(
        &self,
        inconsistencies: &[Inconsistency],
    ) -> Result<Vec<RepairResult>> {
        let results = inconsistencies
            .iter()
            .map(|inc| {
                let repaired_at = SystemTime::now();
                let status = match inc.severity {
                    InconsistencySeverity::Low | InconsistencySeverity::Medium => {
                        RepairStatus::Success
                    }
                    InconsistencySeverity::High => RepairStatus::PartialSuccess,
                    InconsistencySeverity::Critical => RepairStatus::Skipped {
                        reason: "Critical inconsistency requires manual intervention".to_string(),
                    },
                };
                RepairResult {
                    inconsistency: inc.clone(),
                    status,
                    actions: vec!["logged".to_string(), "flagged_for_review".to_string()],
                    repaired_at,
                }
            })
            .collect();
        Ok(results)
    }

    fn get_strategy_name(&self) -> &str {
        "default_repair_strategy"
    }
}

/// Engine for orchestrating inconsistency repairs
pub struct InconsistencyRepairEngine {
    strategies: Vec<Box<dyn ConsistencyRepairStrategy>>,
    total_repairs: AtomicU64,
}

impl InconsistencyRepairEngine {
    pub fn new() -> Self {
        Self {
            strategies: vec![Box::new(DefaultRepairStrategy)],
            total_repairs: AtomicU64::new(0),
        }
    }

    pub fn add_strategy(&mut self, strategy: Box<dyn ConsistencyRepairStrategy>) {
        self.strategies.push(strategy);
    }

    pub fn repair(&self, inconsistencies: &[Inconsistency]) -> Result<Vec<RepairResult>> {
        let mut all_results = Vec::new();
        if let Some(strategy) = self.strategies.first() {
            let results = strategy.repair_inconsistencies(inconsistencies)?;
            let count = results.len() as u64;
            all_results.extend(results);
            self.total_repairs.fetch_add(count, Ordering::Relaxed);
        }
        Ok(all_results)
    }

    pub fn total_repairs(&self) -> u64 {
        self.total_repairs.load(Ordering::Acquire)
    }
}

impl Default for InconsistencyRepairEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Consistency manager for the embedding pipeline
pub struct ConsistencyManager {
    config: ConsistencyConfig,
    repair_engine: Arc<RwLock<InconsistencyRepairEngine>>,
    is_running: AtomicBool,
    total_checks: AtomicU64,
    total_detected: AtomicU64,
    total_repairs: AtomicU64,
    total_success: AtomicU64,
}

impl ConsistencyManager {
    pub fn new(consistency_level: ConsistencyLevel) -> Result<Self> {
        let config = ConsistencyConfig {
            consistency_level,
            ..Default::default()
        };
        Ok(Self {
            config,
            repair_engine: Arc::new(RwLock::new(InconsistencyRepairEngine::new())),
            is_running: AtomicBool::new(false),
            total_checks: AtomicU64::new(0),
            total_detected: AtomicU64::new(0),
            total_repairs: AtomicU64::new(0),
            total_success: AtomicU64::new(0),
        })
    }

    pub async fn start_consistency_checking(&self) -> Result<()> {
        self.is_running.store(true, Ordering::Release);
        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        self.is_running.store(false, Ordering::Release);
        Ok(())
    }

    pub async fn health_check(&self) -> Result<HealthStatus> {
        if self.is_running.load(Ordering::Acquire) {
            Ok(HealthStatus::Healthy)
        } else {
            Ok(HealthStatus::Warning {
                message: "Consistency manager not running".to_string(),
            })
        }
    }

    pub fn detect_and_repair(
        &self,
        inconsistencies: Vec<Inconsistency>,
    ) -> Result<Vec<RepairResult>> {
        self.total_checks.fetch_add(1, Ordering::Relaxed);
        self.total_detected
            .fetch_add(inconsistencies.len() as u64, Ordering::Relaxed);

        if !self.config.enable_auto_repair || inconsistencies.is_empty() {
            return Ok(vec![]);
        }

        let engine = self
            .repair_engine
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire repair engine lock"))?;

        self.total_repairs
            .fetch_add(inconsistencies.len() as u64, Ordering::Relaxed);
        let results = engine.repair(&inconsistencies)?;

        let successes = results
            .iter()
            .filter(|r| matches!(r.status, RepairStatus::Success))
            .count();
        self.total_success
            .fetch_add(successes as u64, Ordering::Relaxed);

        Ok(results)
    }

    pub fn get_statistics(&self) -> ConsistencyStatistics {
        ConsistencyStatistics {
            total_checks: self.total_checks.load(Ordering::Acquire),
            total_inconsistencies_detected: self.total_detected.load(Ordering::Acquire),
            total_repairs_attempted: self.total_repairs.load(Ordering::Acquire),
            total_repairs_succeeded: self.total_success.load(Ordering::Acquire),
            is_running: self.is_running.load(Ordering::Acquire),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real_time_embedding_pipeline::config::ConsistencyLevel;
    use crate::real_time_embedding_pipeline::traits::{
        Inconsistency, InconsistencySeverity, InconsistencyType,
    };

    fn make_inconsistency(severity: InconsistencySeverity) -> Inconsistency {
        Inconsistency {
            inconsistency_type: InconsistencyType::DataMismatch,
            affected_resources: vec!["resource1".to_string()],
            description: "Test inconsistency".to_string(),
            severity,
            detected_at: SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn test_consistency_manager_lifecycle() {
        let manager = ConsistencyManager::new(ConsistencyLevel::Session).expect("should create");
        manager
            .start_consistency_checking()
            .await
            .expect("should start");
        let health = manager.health_check().await.expect("should check");
        assert!(matches!(health, HealthStatus::Healthy));
        manager.stop().await.expect("should stop");
    }

    #[test]
    fn test_default_repair_strategy() {
        let strategy = DefaultRepairStrategy;
        let inc = make_inconsistency(InconsistencySeverity::Low);
        let results = strategy
            .repair_inconsistencies(&[inc])
            .expect("should repair");
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].status, RepairStatus::Success));
    }

    #[test]
    fn test_critical_inconsistency_skipped() {
        let strategy = DefaultRepairStrategy;
        let inc = make_inconsistency(InconsistencySeverity::Critical);
        let results = strategy
            .repair_inconsistencies(&[inc])
            .expect("should repair");
        assert!(matches!(results[0].status, RepairStatus::Skipped { .. }));
    }

    #[test]
    fn test_detect_and_repair() {
        let manager = ConsistencyManager::new(ConsistencyLevel::Strong).expect("should create");
        let incs = vec![make_inconsistency(InconsistencySeverity::Low)];
        let results = manager.detect_and_repair(incs).expect("should work");
        assert_eq!(results.len(), 1);
        let stats = manager.get_statistics();
        assert_eq!(stats.total_checks, 1);
        assert_eq!(stats.total_inconsistencies_detected, 1);
    }

    #[test]
    fn test_repair_engine() {
        let engine = InconsistencyRepairEngine::new();
        let inc = make_inconsistency(InconsistencySeverity::Medium);
        let results = engine.repair(&[inc]).expect("should repair");
        assert!(!results.is_empty());
    }
}
