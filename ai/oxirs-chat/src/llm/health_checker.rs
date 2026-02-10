//! Health Checker for LLM Providers
//!
//! Monitors provider health status, latency, and error rates for intelligent routing.

use anyhow::Result;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Provider identifier
pub type ProviderId = String;

/// Health status of a provider
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Health check result for a provider
#[derive(Debug, Clone)]
pub struct ProviderHealth {
    pub provider_id: ProviderId,
    pub is_healthy: bool,
    pub status: HealthStatus,
    pub last_check: SystemTime,
    pub avg_latency_ms: f64,
    pub error_rate: f64,
    pub consecutive_failures: u32,
    pub uptime_percentage: f64,
}

impl ProviderHealth {
    pub fn new(provider_id: ProviderId) -> Self {
        Self {
            provider_id,
            is_healthy: true,
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            avg_latency_ms: 0.0,
            error_rate: 0.0,
            consecutive_failures: 0,
            uptime_percentage: 100.0,
        }
    }

    pub fn update_health(&mut self, config: &HealthCheckConfig) {
        // Determine health status based on metrics
        if self.error_rate >= 0.5 || self.consecutive_failures >= config.max_consecutive_failures {
            self.status = HealthStatus::Unhealthy;
            self.is_healthy = false;
        } else if self.error_rate >= config.error_rate_threshold
            || self.avg_latency_ms >= config.latency_threshold_ms as f64
        {
            self.status = HealthStatus::Degraded;
            self.is_healthy = true; // Still usable but degraded
        } else {
            self.status = HealthStatus::Healthy;
            self.is_healthy = true;
        }
    }
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    pub check_interval_secs: u64,
    pub latency_threshold_ms: u64,
    pub error_rate_threshold: f64,
    pub max_consecutive_failures: u32,
    pub health_check_timeout: Duration,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            check_interval_secs: 60,
            latency_threshold_ms: 5000,
            error_rate_threshold: 0.1, // 10%
            max_consecutive_failures: 3,
            health_check_timeout: Duration::from_secs(10),
        }
    }
}

/// Call result for tracking
#[derive(Debug, Clone)]
struct CallRecord {
    timestamp: SystemTime,
    success: bool,
    latency_ms: u64,
}

/// Health checker for LLM providers
pub struct HealthChecker {
    health_status: Arc<RwLock<HashMap<ProviderId, ProviderHealth>>>,
    call_history: Arc<RwLock<HashMap<ProviderId, Vec<CallRecord>>>>,
    config: HealthCheckConfig,
}

impl HealthChecker {
    pub fn new(config: HealthCheckConfig) -> Self {
        Self {
            health_status: Arc::new(RwLock::new(HashMap::new())),
            call_history: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Initialize health status for a provider
    pub async fn register_provider(&self, provider_id: ProviderId) {
        let mut health_status = self.health_status.write().await;
        health_status
            .entry(provider_id.clone())
            .or_insert_with(|| ProviderHealth::new(provider_id));
    }

    /// Record a call result
    pub async fn record_call(
        &self,
        provider_id: &ProviderId,
        success: bool,
        latency: Duration,
    ) -> Result<()> {
        // Register provider if not exists
        {
            let health_status = self.health_status.read().await;
            if !health_status.contains_key(provider_id) {
                drop(health_status);
                self.register_provider(provider_id.clone()).await;
            }
        }

        // Record call
        let mut call_history = self.call_history.write().await;
        let records = call_history
            .entry(provider_id.clone())
            .or_insert_with(Vec::new);

        records.push(CallRecord {
            timestamp: SystemTime::now(),
            success,
            latency_ms: latency.as_millis() as u64,
        });

        // Keep only recent records (last 100 calls or 1 hour)
        let cutoff = SystemTime::now()
            .checked_sub(Duration::from_secs(3600))
            .unwrap_or(SystemTime::UNIX_EPOCH);
        records.retain(|record| record.timestamp > cutoff);
        if records.len() > 100 {
            records.drain(0..records.len() - 100);
        }

        // Update health status
        self.update_provider_health(provider_id).await?;

        Ok(())
    }

    /// Update health status based on call history
    async fn update_provider_health(&self, provider_id: &ProviderId) -> Result<()> {
        let call_history = self.call_history.read().await;
        let records = call_history.get(provider_id);

        if records.is_none() || records.unwrap().is_empty() {
            return Ok(());
        }

        let records = records.unwrap();
        let total_calls = records.len() as f64;
        let failed_calls = records.iter().filter(|r| !r.success).count() as f64;
        let error_rate = failed_calls / total_calls;

        let avg_latency_ms = records.iter().map(|r| r.latency_ms).sum::<u64>() as f64 / total_calls;

        let consecutive_failures = records
            .iter()
            .rev()
            .take_while(|r| !r.success)
            .count() as u32;

        let successful_calls = total_calls - failed_calls;
        let uptime_percentage = (successful_calls / total_calls) * 100.0;

        // Update health status
        let mut health_status = self.health_status.write().await;
        if let Some(health) = health_status.get_mut(provider_id) {
            health.last_check = SystemTime::now();
            health.avg_latency_ms = avg_latency_ms;
            health.error_rate = error_rate;
            health.consecutive_failures = consecutive_failures;
            health.uptime_percentage = uptime_percentage;
            health.update_health(&self.config);

            if !health.is_healthy {
                warn!(
                    "Provider {} is unhealthy: error_rate={:.2}, latency={:.0}ms, failures={}",
                    provider_id, error_rate, avg_latency_ms, consecutive_failures
                );
            } else if health.status == HealthStatus::Degraded {
                debug!(
                    "Provider {} is degraded: error_rate={:.2}, latency={:.0}ms",
                    provider_id, error_rate, avg_latency_ms
                );
            }
        }

        Ok(())
    }

    /// Check if provider is healthy
    pub async fn is_provider_healthy(&self, provider_id: &ProviderId) -> bool {
        let health_status = self.health_status.read().await;
        health_status
            .get(provider_id)
            .map(|h| h.is_healthy)
            .unwrap_or(false)
    }

    /// Get provider health status
    pub async fn get_health_status(&self, provider_id: &ProviderId) -> Option<ProviderHealth> {
        let health_status = self.health_status.read().await;
        health_status.get(provider_id).cloned()
    }

    /// Get all provider health statuses
    pub async fn get_all_health_statuses(&self) -> HashMap<ProviderId, ProviderHealth> {
        let health_status = self.health_status.read().await;
        health_status.clone()
    }

    /// Get healthy providers sorted by performance
    pub async fn get_healthy_providers(&self) -> Vec<ProviderId> {
        let health_status = self.health_status.read().await;

        let mut providers: Vec<_> = health_status
            .iter()
            .filter(|(_, health)| health.is_healthy)
            .map(|(id, health)| (id.clone(), health.avg_latency_ms, health.error_rate))
            .collect();

        // Sort by error rate (ascending) then latency (ascending)
        providers.sort_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        providers.into_iter().map(|(id, _, _)| id).collect()
    }

    /// Start periodic health checks (background task)
    pub async fn start_periodic_checks(&self) -> Result<()> {
        let health_checker = Arc::new(self.health_status.clone());
        let _call_history = Arc::new(self.call_history.clone());
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(config.check_interval_secs));

            loop {
                interval.tick().await;

                // Update all provider health statuses
                let health_status = health_checker.read().await;
                for provider_id in health_status.keys() {
                    // Health is updated on each call, so we just log status here
                    if let Some(health) = health_status.get(provider_id) {
                        info!(
                            "Provider {} health: {:?}, latency={:.0}ms, error_rate={:.2}%",
                            provider_id,
                            health.status,
                            health.avg_latency_ms,
                            health.error_rate * 100.0
                        );
                    }
                }
            }
        });

        Ok(())
    }

    /// Reset health status for a provider
    pub async fn reset_provider(&self, provider_id: &ProviderId) -> Result<()> {
        let mut health_status = self.health_status.write().await;
        let mut call_history = self.call_history.write().await;

        health_status.insert(provider_id.clone(), ProviderHealth::new(provider_id.clone()));
        call_history.insert(provider_id.clone(), Vec::new());

        info!("Reset health status for provider: {}", provider_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_provider_registration() {
        let checker = HealthChecker::new(HealthCheckConfig::default());
        checker.register_provider("test-provider".to_string()).await;

        assert!(checker.is_provider_healthy(&"test-provider".to_string()).await);
    }

    #[tokio::test]
    async fn test_healthy_calls() {
        let checker = HealthChecker::new(HealthCheckConfig::default());
        let provider_id = "test-provider".to_string();

        // Record successful calls
        for _ in 0..10 {
            checker
                .record_call(&provider_id, true, Duration::from_millis(100))
                .await
                .unwrap();
        }

        assert!(checker.is_provider_healthy(&provider_id).await);

        let health = checker.get_health_status(&provider_id).await.unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert!((health.avg_latency_ms - 100.0).abs() < 1.0);
        assert_eq!(health.error_rate, 0.0);
    }

    #[tokio::test]
    async fn test_degraded_status() {
        let mut config = HealthCheckConfig::default();
        config.latency_threshold_ms = 100;

        let checker = HealthChecker::new(config);
        let provider_id = "test-provider".to_string();

        // Record slow calls (above threshold)
        for _ in 0..10 {
            checker
                .record_call(&provider_id, true, Duration::from_millis(200))
                .await
                .unwrap();
        }

        let health = checker.get_health_status(&provider_id).await.unwrap();
        assert_eq!(health.status, HealthStatus::Degraded);
        assert!(health.is_healthy); // Still healthy but degraded
    }

    #[tokio::test]
    async fn test_unhealthy_status() {
        let checker = HealthChecker::new(HealthCheckConfig::default());
        let provider_id = "test-provider".to_string();

        // Record failed calls
        for _ in 0..10 {
            checker
                .record_call(&provider_id, false, Duration::from_millis(100))
                .await
                .unwrap();
        }

        let health = checker.get_health_status(&provider_id).await.unwrap();
        assert_eq!(health.status, HealthStatus::Unhealthy);
        assert!(!health.is_healthy);
        assert_eq!(health.error_rate, 1.0);
    }

    #[tokio::test]
    async fn test_consecutive_failures() {
        let mut config = HealthCheckConfig::default();
        config.max_consecutive_failures = 3;

        let checker = HealthChecker::new(config);
        let provider_id = "test-provider".to_string();

        // Record some successful calls
        for _ in 0..5 {
            checker
                .record_call(&provider_id, true, Duration::from_millis(100))
                .await
                .unwrap();
        }

        // Record consecutive failures
        for _ in 0..3 {
            checker
                .record_call(&provider_id, false, Duration::from_millis(100))
                .await
                .unwrap();
        }

        let health = checker.get_health_status(&provider_id).await.unwrap();
        assert!(!health.is_healthy);
        assert_eq!(health.consecutive_failures, 3);
    }

    #[tokio::test]
    async fn test_get_healthy_providers() {
        let checker = HealthChecker::new(HealthCheckConfig::default());

        // Provider 1: healthy, low latency
        for _ in 0..10 {
            checker
                .record_call(&"provider1".to_string(), true, Duration::from_millis(100))
                .await
                .unwrap();
        }

        // Provider 2: healthy, high latency
        for _ in 0..10 {
            checker
                .record_call(&"provider2".to_string(), true, Duration::from_millis(500))
                .await
                .unwrap();
        }

        // Provider 3: unhealthy
        for _ in 0..10 {
            checker
                .record_call(&"provider3".to_string(), false, Duration::from_millis(100))
                .await
                .unwrap();
        }

        let healthy = checker.get_healthy_providers().await;
        assert_eq!(healthy.len(), 2);
        assert_eq!(healthy[0], "provider1"); // Lowest latency first
        assert_eq!(healthy[1], "provider2");
    }

    #[tokio::test]
    async fn test_reset_provider() {
        let checker = HealthChecker::new(HealthCheckConfig::default());
        let provider_id = "test-provider".to_string();

        // Record failed calls
        for _ in 0..10 {
            checker
                .record_call(&provider_id, false, Duration::from_millis(100))
                .await
                .unwrap();
        }

        assert!(!checker.is_provider_healthy(&provider_id).await);

        // Reset provider
        checker.reset_provider(&provider_id).await.unwrap();

        // Should be healthy again
        assert!(checker.is_provider_healthy(&provider_id).await);
    }
}
