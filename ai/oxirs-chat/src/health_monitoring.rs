//! Health monitoring module for comprehensive system health tracking
//!
//! This module provides:
//! - System health metrics collection
//! - Service availability monitoring
//! - Resource utilization tracking
//! - Error rate monitoring
//! - Performance baseline maintenance
//! - Health check endpoints
//! - Alerting and notification systems

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;

/// System health status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
}

/// Health monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitoringConfig {
    pub check_interval: Duration,
    pub metrics_retention_period: Duration,
    pub alert_thresholds: AlertThresholds,
    pub enabled_checks: Vec<HealthCheckType>,
    pub notification_config: NotificationConfig,
}

impl Default for HealthMonitoringConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            metrics_retention_period: Duration::from_secs(24 * 60 * 60), // 24 hours
            alert_thresholds: AlertThresholds::default(),
            enabled_checks: vec![
                HealthCheckType::SystemResources,
                HealthCheckType::ServiceAvailability,
                HealthCheckType::ErrorRates,
                HealthCheckType::ResponseTimes,
                HealthCheckType::DatabaseConnectivity,
                HealthCheckType::CacheHealth,
            ],
            notification_config: NotificationConfig::default(),
        }
    }
}

/// Types of health checks
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthCheckType {
    SystemResources,
    ServiceAvailability,
    ErrorRates,
    ResponseTimes,
    DatabaseConnectivity,
    CacheHealth,
    LLMConnectivity,
    VectorIndexHealth,
    StorageHealth,
}

/// Alert thresholds for various metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_usage_warning: f32,
    pub cpu_usage_critical: f32,
    pub memory_usage_warning: f32,
    pub memory_usage_critical: f32,
    pub disk_usage_warning: f32,
    pub disk_usage_critical: f32,
    pub error_rate_warning: f32,
    pub error_rate_critical: f32,
    pub response_time_warning: Duration,
    pub response_time_critical: Duration,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_warning: 70.0,
            cpu_usage_critical: 85.0,
            memory_usage_warning: 75.0,
            memory_usage_critical: 90.0,
            disk_usage_warning: 80.0,
            disk_usage_critical: 95.0,
            error_rate_warning: 5.0,
            error_rate_critical: 10.0,
            response_time_warning: Duration::from_millis(1000),
            response_time_critical: Duration::from_millis(3000),
        }
    }
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub enabled: bool,
    pub email_notifications: bool,
    pub webhook_notifications: bool,
    pub webhook_url: Option<String>,
    pub notification_cooldown: Duration,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            email_notifications: false,
            webhook_notifications: false,
            webhook_url: None,
            notification_cooldown: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Overall system health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthReport {
    pub overall_status: HealthStatus,
    pub timestamp: SystemTime,
    pub component_statuses: HashMap<String, ComponentHealth>,
    pub system_metrics: SystemMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub error_metrics: ErrorMetrics,
    pub health_checks: Vec<HealthCheckResult>,
    pub alerts: Vec<Alert>,
    pub uptime: Duration,
}

/// Health status of individual components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub last_check: SystemTime,
    pub metrics: ComponentMetrics,
    pub issues: Vec<HealthIssue>,
}

/// System resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub disk_usage: f32,
    pub network_io: NetworkMetrics,
    pub process_count: u32,
    pub thread_count: u32,
    pub open_files: u32,
}

/// Network I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub connections_active: u32,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub average_response_time: Duration,
    pub p50_response_time: Duration,
    pub p95_response_time: Duration,
    pub p99_response_time: Duration,
    pub requests_per_second: f32,
    pub concurrent_requests: u32,
    pub cache_hit_rate: f32,
}

/// Error tracking metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    pub total_errors: u64,
    pub error_rate: f32,
    pub errors_by_type: HashMap<String, u64>,
    pub recent_errors: Vec<ErrorOccurrence>,
}

/// Individual error occurrence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorOccurrence {
    pub timestamp: SystemTime,
    pub error_type: String,
    pub message: String,
    pub severity: ErrorSeverity,
    pub component: String,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Component-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetrics {
    pub availability_percentage: f32,
    pub response_time: Duration,
    pub error_count: u64,
    pub last_error: Option<SystemTime>,
    pub custom_metrics: HashMap<String, f64>,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub check_type: HealthCheckType,
    pub status: HealthStatus,
    pub timestamp: SystemTime,
    pub duration: Duration,
    pub message: String,
    pub details: Option<String>,
}

/// Health issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    pub issue_type: HealthIssueType,
    pub severity: HealthIssueSeverity,
    pub description: String,
    pub first_detected: SystemTime,
    pub last_seen: SystemTime,
    pub count: u32,
    pub resolved: bool,
}

/// Types of health issues
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthIssueType {
    HighResourceUsage,
    SlowResponse,
    HighErrorRate,
    ServiceUnavailable,
    DatabaseIssue,
    CacheIssue,
    ConfigurationError,
    DependencyFailure,
}

/// Severity of health issues
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthIssueSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert generated by health monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub title: String,
    pub description: String,
    pub component: String,
    pub created_at: SystemTime,
    pub resolved_at: Option<SystemTime>,
    pub actions_taken: Vec<String>,
}

/// Types of alerts
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertType {
    ThresholdExceeded,
    ServiceDown,
    PerformanceDegradation,
    ErrorSpike,
    ResourceExhaustion,
    SecurityConcern,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Health monitoring engine
pub struct HealthMonitor {
    config: HealthMonitoringConfig,
    start_time: SystemTime,
    metrics_history: Arc<RwLock<VecDeque<SystemHealthReport>>>,
    component_monitors: HashMap<String, Box<dyn ComponentMonitor + Send + Sync>>,
    alert_manager: AlertManager,
    notification_manager: NotificationManager,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(config: HealthMonitoringConfig) -> Self {
        let mut component_monitors: HashMap<String, Box<dyn ComponentMonitor + Send + Sync>> =
            HashMap::new();

        // Add default component monitors
        component_monitors.insert("system".to_string(), Box::new(SystemResourceMonitor::new()));
        component_monitors.insert("database".to_string(), Box::new(DatabaseMonitor::new()));
        component_monitors.insert("cache".to_string(), Box::new(CacheMonitor::new()));
        component_monitors.insert("llm".to_string(), Box::new(LLMMonitor::new()));

        Self {
            config: config.clone(),
            start_time: SystemTime::now(),
            metrics_history: Arc::new(RwLock::new(VecDeque::new())),
            component_monitors,
            alert_manager: AlertManager::new(config.alert_thresholds.clone()),
            notification_manager: NotificationManager::new(config.notification_config.clone()),
        }
    }

    /// Start the health monitoring loop
    pub async fn start_monitoring(&mut self) -> Result<()> {
        let mut interval = tokio::time::interval(self.config.check_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.perform_health_checks().await {
                eprintln!("Health check failed: {}", e);
            }
        }
    }

    /// Perform all configured health checks
    async fn perform_health_checks(&mut self) -> Result<()> {
        let health_report = self.generate_health_report().await?;

        // Check for alerts
        let alerts = self.alert_manager.check_for_alerts(&health_report).await?;

        // Send notifications if needed
        if !alerts.is_empty() {
            self.notification_manager.send_alerts(&alerts).await?;
        }

        // Store metrics
        self.store_metrics(health_report).await?;

        Ok(())
    }

    /// Generate comprehensive health report
    pub async fn generate_health_report(&self) -> Result<SystemHealthReport> {
        let timestamp = SystemTime::now();
        let uptime = timestamp
            .duration_since(self.start_time)
            .unwrap_or(Duration::ZERO);

        // Collect component health statuses
        let mut component_statuses = HashMap::new();
        for (name, monitor) in &self.component_monitors {
            let health = monitor.check_health().await?;
            component_statuses.insert(name.clone(), health);
        }

        // Collect system metrics
        let system_metrics = self.collect_system_metrics().await?;
        let performance_metrics = self.collect_performance_metrics().await?;
        let error_metrics = self.collect_error_metrics().await?;

        // Perform health checks
        let health_checks = self.run_health_checks().await?;

        // Determine overall status
        let overall_status = self.determine_overall_status(&component_statuses, &health_checks);

        Ok(SystemHealthReport {
            overall_status,
            timestamp,
            component_statuses,
            system_metrics,
            performance_metrics,
            error_metrics,
            health_checks,
            alerts: Vec::new(), // Will be populated by alert manager
            uptime,
        })
    }

    /// Get current health status
    pub async fn get_health_status(&self) -> Result<HealthStatus> {
        let report = self.generate_health_report().await?;
        Ok(report.overall_status)
    }

    /// Get detailed health report
    pub async fn get_detailed_health_report(&self) -> Result<SystemHealthReport> {
        self.generate_health_report().await
    }

    /// Get metrics history
    pub async fn get_metrics_history(&self, duration: Duration) -> Result<Vec<SystemHealthReport>> {
        let history = self.metrics_history.read().await;
        let cutoff = SystemTime::now() - duration;

        Ok(history
            .iter()
            .filter(|report| report.timestamp >= cutoff)
            .cloned()
            .collect())
    }

    /// Store metrics in history
    async fn store_metrics(&self, report: SystemHealthReport) -> Result<()> {
        let mut history = self.metrics_history.write().await;
        history.push_back(report);

        // Clean up old metrics
        let retention_cutoff = SystemTime::now() - self.config.metrics_retention_period;
        while let Some(front) = history.front() {
            if front.timestamp < retention_cutoff {
                history.pop_front();
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Collect system resource metrics
    async fn collect_system_metrics(&self) -> Result<SystemMetrics> {
        // Mock implementation - would use system APIs
        Ok(SystemMetrics {
            cpu_usage: 45.2,
            memory_usage: 62.8,
            disk_usage: 38.1,
            network_io: NetworkMetrics {
                bytes_sent: 1024768,
                bytes_received: 2048456,
                packets_sent: 512,
                packets_received: 1024,
                connections_active: 15,
            },
            process_count: 128,
            thread_count: 256,
            open_files: 64,
        })
    }

    /// Collect performance metrics
    async fn collect_performance_metrics(&self) -> Result<PerformanceMetrics> {
        // Mock implementation - would collect from performance monitor
        Ok(PerformanceMetrics {
            average_response_time: Duration::from_millis(245),
            p50_response_time: Duration::from_millis(180),
            p95_response_time: Duration::from_millis(520),
            p99_response_time: Duration::from_millis(890),
            requests_per_second: 42.5,
            concurrent_requests: 8,
            cache_hit_rate: 0.78,
        })
    }

    /// Collect error metrics
    async fn collect_error_metrics(&self) -> Result<ErrorMetrics> {
        // Mock implementation - would collect from error tracking
        Ok(ErrorMetrics {
            total_errors: 127,
            error_rate: 2.3,
            errors_by_type: HashMap::from([
                ("validation_error".to_string(), 45),
                ("timeout_error".to_string(), 32),
                ("connection_error".to_string(), 50),
            ]),
            recent_errors: Vec::new(),
        })
    }

    /// Run health checks
    async fn run_health_checks(&self) -> Result<Vec<HealthCheckResult>> {
        let mut results = Vec::new();

        for check_type in &self.config.enabled_checks {
            let start_time = SystemTime::now();
            let result = self.perform_health_check(check_type.clone()).await?;
            let duration = SystemTime::now()
                .duration_since(start_time)
                .unwrap_or(Duration::ZERO);

            results.push(HealthCheckResult {
                check_type: check_type.clone(),
                status: result.0,
                timestamp: SystemTime::now(),
                duration,
                message: result.1,
                details: result.2,
            });
        }

        Ok(results)
    }

    /// Perform individual health check
    async fn perform_health_check(
        &self,
        check_type: HealthCheckType,
    ) -> Result<(HealthStatus, String, Option<String>)> {
        match check_type {
            HealthCheckType::SystemResources => {
                let metrics = self.collect_system_metrics().await?;
                if metrics.cpu_usage > 90.0 || metrics.memory_usage > 95.0 {
                    Ok((
                        HealthStatus::Critical,
                        "High resource usage".to_string(),
                        None,
                    ))
                } else if metrics.cpu_usage > 70.0 || metrics.memory_usage > 80.0 {
                    Ok((
                        HealthStatus::Degraded,
                        "Elevated resource usage".to_string(),
                        None,
                    ))
                } else {
                    Ok((
                        HealthStatus::Healthy,
                        "Resource usage normal".to_string(),
                        None,
                    ))
                }
            }
            HealthCheckType::ServiceAvailability => {
                // Mock check - would test actual service endpoints
                Ok((
                    HealthStatus::Healthy,
                    "All services available".to_string(),
                    None,
                ))
            }
            HealthCheckType::ErrorRates => {
                let metrics = self.collect_error_metrics().await?;
                if metrics.error_rate > 10.0 {
                    Ok((HealthStatus::Critical, "High error rate".to_string(), None))
                } else if metrics.error_rate > 5.0 {
                    Ok((
                        HealthStatus::Degraded,
                        "Elevated error rate".to_string(),
                        None,
                    ))
                } else {
                    Ok((HealthStatus::Healthy, "Error rate normal".to_string(), None))
                }
            }
            HealthCheckType::ResponseTimes => {
                let metrics = self.collect_performance_metrics().await?;
                if metrics.p95_response_time > Duration::from_millis(2000) {
                    Ok((
                        HealthStatus::Degraded,
                        "Slow response times".to_string(),
                        None,
                    ))
                } else {
                    Ok((
                        HealthStatus::Healthy,
                        "Response times normal".to_string(),
                        None,
                    ))
                }
            }
            _ => Ok((HealthStatus::Healthy, "Check passed".to_string(), None)),
        }
    }

    /// Determine overall system status
    fn determine_overall_status(
        &self,
        component_statuses: &HashMap<String, ComponentHealth>,
        health_checks: &[HealthCheckResult],
    ) -> HealthStatus {
        let mut has_critical = false;
        let mut has_unhealthy = false;
        let mut has_degraded = false;

        // Check component statuses
        for component in component_statuses.values() {
            match component.status {
                HealthStatus::Critical => has_critical = true,
                HealthStatus::Unhealthy => has_unhealthy = true,
                HealthStatus::Degraded => has_degraded = true,
                HealthStatus::Healthy => {}
            }
        }

        // Check health check results
        for check in health_checks {
            match check.status {
                HealthStatus::Critical => has_critical = true,
                HealthStatus::Unhealthy => has_unhealthy = true,
                HealthStatus::Degraded => has_degraded = true,
                HealthStatus::Healthy => {}
            }
        }

        if has_critical {
            HealthStatus::Critical
        } else if has_unhealthy {
            HealthStatus::Unhealthy
        } else if has_degraded {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }
}

/// Trait for component-specific monitoring
#[async_trait::async_trait]
pub trait ComponentMonitor {
    async fn check_health(&self) -> Result<ComponentHealth>;
    async fn get_metrics(&self) -> Result<ComponentMetrics>;
    fn get_name(&self) -> &str;
}

/// System resource monitor
pub struct SystemResourceMonitor;

impl SystemResourceMonitor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl ComponentMonitor for SystemResourceMonitor {
    async fn check_health(&self) -> Result<ComponentHealth> {
        let metrics = self.get_metrics().await?;

        let status = if metrics.custom_metrics.get("cpu_usage").unwrap_or(&0.0) > &90.0 {
            HealthStatus::Critical
        } else if metrics.custom_metrics.get("cpu_usage").unwrap_or(&0.0) > &70.0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        Ok(ComponentHealth {
            name: "System Resources".to_string(),
            status,
            last_check: SystemTime::now(),
            metrics,
            issues: Vec::new(),
        })
    }

    async fn get_metrics(&self) -> Result<ComponentMetrics> {
        let mut custom_metrics = HashMap::new();
        custom_metrics.insert("cpu_usage".to_string(), 45.2);
        custom_metrics.insert("memory_usage".to_string(), 62.8);
        custom_metrics.insert("disk_usage".to_string(), 38.1);

        Ok(ComponentMetrics {
            availability_percentage: 99.9,
            response_time: Duration::from_millis(50),
            error_count: 0,
            last_error: None,
            custom_metrics,
        })
    }

    fn get_name(&self) -> &str {
        "system_resources"
    }
}

/// Database monitor
pub struct DatabaseMonitor;

impl DatabaseMonitor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl ComponentMonitor for DatabaseMonitor {
    async fn check_health(&self) -> Result<ComponentHealth> {
        let metrics = self.get_metrics().await?;

        Ok(ComponentHealth {
            name: "Database".to_string(),
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            metrics,
            issues: Vec::new(),
        })
    }

    async fn get_metrics(&self) -> Result<ComponentMetrics> {
        Ok(ComponentMetrics {
            availability_percentage: 99.95,
            response_time: Duration::from_millis(25),
            error_count: 3,
            last_error: Some(SystemTime::now() - Duration::from_secs(3600)),
            custom_metrics: HashMap::new(),
        })
    }

    fn get_name(&self) -> &str {
        "database"
    }
}

/// Cache monitor
pub struct CacheMonitor;

impl CacheMonitor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl ComponentMonitor for CacheMonitor {
    async fn check_health(&self) -> Result<ComponentHealth> {
        let metrics = self.get_metrics().await?;

        Ok(ComponentHealth {
            name: "Cache".to_string(),
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            metrics,
            issues: Vec::new(),
        })
    }

    async fn get_metrics(&self) -> Result<ComponentMetrics> {
        let mut custom_metrics = HashMap::new();
        custom_metrics.insert("hit_rate".to_string(), 0.82);
        custom_metrics.insert("memory_usage".to_string(), 45.6);

        Ok(ComponentMetrics {
            availability_percentage: 99.8,
            response_time: Duration::from_millis(5),
            error_count: 1,
            last_error: Some(SystemTime::now() - Duration::from_secs(7200)),
            custom_metrics,
        })
    }

    fn get_name(&self) -> &str {
        "cache"
    }
}

/// LLM service monitor
pub struct LLMMonitor;

impl LLMMonitor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl ComponentMonitor for LLMMonitor {
    async fn check_health(&self) -> Result<ComponentHealth> {
        let metrics = self.get_metrics().await?;

        Ok(ComponentHealth {
            name: "LLM Service".to_string(),
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            metrics,
            issues: Vec::new(),
        })
    }

    async fn get_metrics(&self) -> Result<ComponentMetrics> {
        let mut custom_metrics = HashMap::new();
        custom_metrics.insert("tokens_per_second".to_string(), 150.0);
        custom_metrics.insert("model_load_time".to_string(), 2.5);

        Ok(ComponentMetrics {
            availability_percentage: 98.5,
            response_time: Duration::from_millis(800),
            error_count: 12,
            last_error: Some(SystemTime::now() - Duration::from_secs(900)),
            custom_metrics,
        })
    }

    fn get_name(&self) -> &str {
        "llm_service"
    }
}

/// Alert manager for processing and managing alerts
pub struct AlertManager {
    thresholds: AlertThresholds,
    active_alerts: HashMap<String, Alert>,
}

impl AlertManager {
    pub fn new(thresholds: AlertThresholds) -> Self {
        Self {
            thresholds,
            active_alerts: HashMap::new(),
        }
    }

    pub async fn check_for_alerts(&mut self, report: &SystemHealthReport) -> Result<Vec<Alert>> {
        let mut new_alerts = Vec::new();

        // Check CPU usage
        if report.system_metrics.cpu_usage > self.thresholds.cpu_usage_critical {
            let alert = Alert {
                id: "cpu_critical".to_string(),
                alert_type: AlertType::ThresholdExceeded,
                severity: AlertSeverity::Critical,
                title: "Critical CPU Usage".to_string(),
                description: format!("CPU usage is {}%", report.system_metrics.cpu_usage),
                component: "system".to_string(),
                created_at: SystemTime::now(),
                resolved_at: None,
                actions_taken: Vec::new(),
            };
            new_alerts.push(alert);
        }

        // Check memory usage
        if report.system_metrics.memory_usage > self.thresholds.memory_usage_critical {
            let alert = Alert {
                id: "memory_critical".to_string(),
                alert_type: AlertType::ThresholdExceeded,
                severity: AlertSeverity::Critical,
                title: "Critical Memory Usage".to_string(),
                description: format!("Memory usage is {}%", report.system_metrics.memory_usage),
                component: "system".to_string(),
                created_at: SystemTime::now(),
                resolved_at: None,
                actions_taken: Vec::new(),
            };
            new_alerts.push(alert);
        }

        // Check error rate
        if report.error_metrics.error_rate > self.thresholds.error_rate_critical {
            let alert = Alert {
                id: "error_rate_critical".to_string(),
                alert_type: AlertType::ErrorSpike,
                severity: AlertSeverity::Critical,
                title: "High Error Rate".to_string(),
                description: format!("Error rate is {}%", report.error_metrics.error_rate),
                component: "application".to_string(),
                created_at: SystemTime::now(),
                resolved_at: None,
                actions_taken: Vec::new(),
            };
            new_alerts.push(alert);
        }

        // Store active alerts
        for alert in &new_alerts {
            self.active_alerts.insert(alert.id.clone(), alert.clone());
        }

        Ok(new_alerts)
    }

    pub fn get_active_alerts(&self) -> Vec<&Alert> {
        self.active_alerts.values().collect()
    }

    pub fn resolve_alert(&mut self, alert_id: &str) -> Result<()> {
        if let Some(alert) = self.active_alerts.get_mut(alert_id) {
            alert.resolved_at = Some(SystemTime::now());
        }
        Ok(())
    }
}

/// Notification manager for sending alerts
pub struct NotificationManager {
    config: NotificationConfig,
    last_notification: Option<SystemTime>,
}

impl NotificationManager {
    pub fn new(config: NotificationConfig) -> Self {
        Self {
            config,
            last_notification: None,
        }
    }

    pub async fn send_alerts(&mut self, alerts: &[Alert]) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check cooldown
        if let Some(last) = self.last_notification {
            if SystemTime::now()
                .duration_since(last)
                .unwrap_or(Duration::ZERO)
                < self.config.notification_cooldown
            {
                return Ok(());
            }
        }

        for alert in alerts {
            if alert.severity >= AlertSeverity::Warning {
                self.send_notification(alert).await?;
            }
        }

        self.last_notification = Some(SystemTime::now());
        Ok(())
    }

    async fn send_notification(&self, alert: &Alert) -> Result<()> {
        // Mock implementation - would send actual notifications
        println!("ALERT: {} - {}", alert.title, alert.description);

        if self.config.webhook_notifications {
            // Would send webhook notification
        }

        if self.config.email_notifications {
            // Would send email notification
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_monitor_creation() {
        let config = HealthMonitoringConfig::default();
        let monitor = HealthMonitor::new(config);

        let status = monitor.get_health_status().await.unwrap();
        // Should start healthy
        assert_eq!(status, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_system_resource_monitor() {
        let monitor = SystemResourceMonitor::new();
        let health = monitor.check_health().await.unwrap();

        assert_eq!(health.name, "System Resources");
        assert!(health.metrics.availability_percentage > 0.0);
    }

    #[tokio::test]
    async fn test_alert_manager() {
        let thresholds = AlertThresholds::default();
        let mut alert_manager = AlertManager::new(thresholds);

        let report = SystemHealthReport {
            overall_status: HealthStatus::Critical,
            timestamp: SystemTime::now(),
            component_statuses: HashMap::new(),
            system_metrics: SystemMetrics {
                cpu_usage: 95.0, // Above critical threshold
                memory_usage: 50.0,
                disk_usage: 30.0,
                network_io: NetworkMetrics {
                    bytes_sent: 0,
                    bytes_received: 0,
                    packets_sent: 0,
                    packets_received: 0,
                    connections_active: 0,
                },
                process_count: 0,
                thread_count: 0,
                open_files: 0,
            },
            performance_metrics: PerformanceMetrics {
                average_response_time: Duration::from_millis(100),
                p50_response_time: Duration::from_millis(80),
                p95_response_time: Duration::from_millis(200),
                p99_response_time: Duration::from_millis(300),
                requests_per_second: 10.0,
                concurrent_requests: 5,
                cache_hit_rate: 0.8,
            },
            error_metrics: ErrorMetrics {
                total_errors: 0,
                error_rate: 0.0,
                errors_by_type: HashMap::new(),
                recent_errors: Vec::new(),
            },
            health_checks: Vec::new(),
            alerts: Vec::new(),
            uptime: Duration::from_secs(3600),
        };

        let alerts = alert_manager.check_for_alerts(&report).await.unwrap();
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].alert_type, AlertType::ThresholdExceeded);
    }
}
