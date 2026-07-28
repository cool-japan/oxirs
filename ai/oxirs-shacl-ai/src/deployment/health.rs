//! Health Monitoring and Circuit Breaking
//!
//! This module handles health checks, service discovery,
//! circuit breaking, and availability monitoring.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use super::load_balancing::HealthCheck;
use super::types::DeploymentInfo;
use crate::{Result, ShaclAiError};

/// Health monitor
#[derive(Debug)]
pub struct HealthMonitor {
    health_checks: Vec<HealthCheck>,
    service_discovery: ServiceDiscovery,
    circuit_breaker: CircuitBreaker,
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            health_checks: vec![],
            service_discovery: ServiceDiscovery::default(),
            circuit_breaker: CircuitBreaker::default(),
        }
    }

    /// Verify that a deployment is healthy.
    ///
    /// # Fail-loud contract
    ///
    /// This performs the structural validation it *can* do offline (a
    /// deployment with no services, no pods, or a zero replica count is
    /// unambiguously unhealthy and returns [`ShaclAiError::Validation`]).
    /// Actually probing liveness/readiness endpoints requires a live
    /// orchestration/network backend, which is not bundled; when no
    /// [`HealthCheck`] is configured this method therefore returns
    /// [`ShaclAiError::Unsupported`] rather than a false `Ok(())` that would
    /// mask a failed deployment.
    pub async fn verify_deployment_health(&self, deployment_info: &DeploymentInfo) -> Result<()> {
        if deployment_info.services.is_empty() {
            return Err(ShaclAiError::Validation(format!(
                "deployment '{}' exposes no services; cannot be healthy",
                deployment_info.deployment_id
            )));
        }
        if deployment_info.replicas == 0 || deployment_info.pods.is_empty() {
            return Err(ShaclAiError::Validation(format!(
                "deployment '{}' has no running pods (replicas = {})",
                deployment_info.deployment_id, deployment_info.replicas
            )));
        }
        if deployment_info.pods.len() != deployment_info.replicas as usize {
            return Err(ShaclAiError::Validation(format!(
                "deployment '{}' has {} pods but expected {} replicas",
                deployment_info.deployment_id,
                deployment_info.pods.len(),
                deployment_info.replicas
            )));
        }

        // No health checks configured means we have no way to actually probe
        // liveness — fail loud instead of pretending everything is fine.
        if self.health_checks.is_empty() {
            return Err(ShaclAiError::Unsupported(format!(
                "no health checks are configured, so the health of deployment '{}' cannot be \
                 verified; a live orchestration backend is required",
                deployment_info.deployment_id
            )));
        }

        // A live backend would probe each configured health check against the
        // deployment's services here. Without one, actual probing is
        // unsupported (the circuit breaker's failure_threshold of {} would gate
        // repeated failures once a backend is wired).
        Err(ShaclAiError::Unsupported(format!(
            "active health probing of deployment '{}' requires a live orchestration backend \
             (circuit-breaker failure threshold = {})",
            deployment_info.deployment_id, self.circuit_breaker.failure_threshold
        )))
    }
}

/// Service discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDiscovery {
    pub discovery_type: ServiceDiscoveryType,
    pub configuration: HashMap<String, String>,
}

impl Default for ServiceDiscovery {
    fn default() -> Self {
        Self {
            discovery_type: ServiceDiscoveryType::Kubernetes,
            configuration: HashMap::new(),
        }
    }
}

/// Service discovery types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceDiscoveryType {
    Kubernetes,
    Consul,
    Etcd,
    Eureka,
    Zookeeper,
}

/// Circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreaker {
    pub failure_threshold: u32,
    pub recovery_timeout: Duration,
    pub half_open_max_calls: u32,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            half_open_max_calls: 3,
        }
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    fn info(services: Vec<&str>, pods: Vec<&str>, replicas: u32) -> DeploymentInfo {
        DeploymentInfo {
            deployment_id: "d1".to_string(),
            namespace: "ns".to_string(),
            services: services.into_iter().map(str::to_string).collect(),
            pods: pods.into_iter().map(str::to_string).collect(),
            replicas,
        }
    }

    /// Regression: a deployment with no services is unambiguously unhealthy and
    /// must error, never `Ok(())`.
    #[tokio::test]
    async fn regression_verify_health_rejects_empty_deployment() {
        let monitor = HealthMonitor::new();
        let result = monitor
            .verify_deployment_health(&info(vec![], vec![], 0))
            .await;
        assert!(matches!(result, Err(crate::ShaclAiError::Validation(_))));
    }

    /// Regression: with no health checks configured, verification cannot succeed
    /// (it previously always returned Ok, masking failed deployments).
    #[tokio::test]
    async fn regression_verify_health_fails_loud_without_checks() {
        let monitor = HealthMonitor::new();
        let result = monitor
            .verify_deployment_health(&info(vec!["svc"], vec!["pod-0"], 1))
            .await;
        assert!(matches!(result, Err(crate::ShaclAiError::Unsupported(_))));
    }
}
