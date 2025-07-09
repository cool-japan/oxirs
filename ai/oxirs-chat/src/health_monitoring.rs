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
    time::{Duration, SystemTime},
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
                eprintln!("Health check failed: {e}");
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

impl Default for SystemResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

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

impl Default for DatabaseMonitor {
    fn default() -> Self {
        Self::new()
    }
}

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

impl Default for CacheMonitor {
    fn default() -> Self {
        Self::new()
    }
}

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

impl Default for LLMMonitor {
    fn default() -> Self {
        Self::new()
    }
}

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

/// Self-healing system for automated recovery from health issues
pub struct SelfHealingSystem {
    config: SelfHealingConfig,
    healing_actions: HashMap<String, HealingAction>,
    recovery_stats: RecoveryStats,
}

/// Self-healing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfHealingConfig {
    pub enabled: bool,
    pub max_recovery_attempts: u32,
    pub recovery_cooldown: Duration,
    pub auto_restart_threshold: f32,
    pub memory_cleanup_threshold: f32,
    pub circuit_breaker_reset_interval: Duration,
}

impl Default for SelfHealingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_recovery_attempts: 3,
            recovery_cooldown: Duration::from_secs(300), // 5 minutes
            auto_restart_threshold: 95.0,                // 95% resource usage
            memory_cleanup_threshold: 85.0,              // 85% memory usage
            circuit_breaker_reset_interval: Duration::from_secs(600), // 10 minutes
        }
    }
}

/// Healing action that can be taken to recover from issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealingAction {
    pub id: String,
    pub action_type: HealingActionType,
    pub target_component: String,
    pub description: String,
    pub attempts: u32,
    pub last_attempt: Option<SystemTime>,
    pub success_rate: f32,
}

/// Types of healing actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealingActionType {
    RestartComponent,
    ClearCache,
    MemoryCleanup,
    ResetCircuitBreaker,
    ScaleResources,
    RollbackConfiguration,
    FlushConnections,
    CompactDatabase,
}

/// Recovery statistics tracking
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RecoveryStats {
    pub total_recoveries: u64,
    pub successful_recoveries: u64,
    pub failed_recoveries: u64,
    pub average_recovery_time: Duration,
    pub recovery_by_type: HashMap<String, u32>,
}

impl SelfHealingSystem {
    /// Create a new self-healing system
    pub fn new(config: SelfHealingConfig) -> Self {
        Self {
            config,
            healing_actions: HashMap::new(),
            recovery_stats: RecoveryStats::default(),
        }
    }

    /// Analyze health issues and trigger appropriate healing actions
    pub async fn analyze_and_heal(
        &mut self,
        health_report: &SystemHealthReport,
    ) -> Result<Vec<HealingAction>> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        let mut triggered_actions = Vec::new();

        // High CPU usage healing
        if health_report.system_metrics.cpu_usage > self.config.auto_restart_threshold {
            if let Some(action) = self
                .create_healing_action(
                    "cpu_cleanup".to_string(),
                    HealingActionType::MemoryCleanup,
                    "system".to_string(),
                    "Clean up memory to reduce CPU pressure".to_string(),
                )
                .await
            {
                triggered_actions.push(action);
            }
        }

        // High memory usage healing
        if health_report.system_metrics.memory_usage > self.config.memory_cleanup_threshold {
            if let Some(action) = self
                .create_healing_action(
                    "memory_cleanup".to_string(),
                    HealingActionType::MemoryCleanup,
                    "system".to_string(),
                    "Trigger garbage collection and cache cleanup".to_string(),
                )
                .await
            {
                triggered_actions.push(action);
            }
        }

        // Service availability issues
        for component in &health_report.component_statuses {
            if component.1.status == HealthStatus::Critical {
                if let Some(action) = self
                    .create_healing_action(
                        format!("{}_restart", component.0),
                        HealingActionType::RestartComponent,
                        component.0.clone(),
                        format!("Restart {} due to critical status", component.0),
                    )
                    .await
                {
                    triggered_actions.push(action);
                }
            }
        }

        // High error rates
        if health_report.error_metrics.error_rate > 0.1 {
            // 10% error rate
            if let Some(action) = self
                .create_healing_action(
                    "circuit_breaker_reset".to_string(),
                    HealingActionType::ResetCircuitBreaker,
                    "llm_service".to_string(),
                    "Reset circuit breakers due to high error rate".to_string(),
                )
                .await
            {
                triggered_actions.push(action);
            }
        }

        // Execute healing actions
        for action in &triggered_actions {
            if let Err(e) = self.execute_healing_action(action).await {
                eprintln!("Failed to execute healing action {}: {}", action.id, e);
            }
        }

        Ok(triggered_actions)
    }

    /// Create a healing action if it hasn't been attempted too recently
    async fn create_healing_action(
        &mut self,
        id: String,
        action_type: HealingActionType,
        target_component: String,
        description: String,
    ) -> Option<HealingAction> {
        // Check if we already have this action and it's in cooldown
        if let Some(existing_action) = self.healing_actions.get(&id) {
            if let Some(last_attempt) = existing_action.last_attempt {
                if SystemTime::now()
                    .duration_since(last_attempt)
                    .unwrap_or(Duration::ZERO)
                    < self.config.recovery_cooldown
                {
                    return None; // Still in cooldown
                }
            }

            if existing_action.attempts >= self.config.max_recovery_attempts {
                return None; // Too many attempts
            }
        }

        let action = HealingAction {
            id: id.clone(),
            action_type,
            target_component,
            description,
            attempts: self
                .healing_actions
                .get(&id)
                .map(|a| a.attempts + 1)
                .unwrap_or(1),
            last_attempt: Some(SystemTime::now()),
            success_rate: self
                .healing_actions
                .get(&id)
                .map(|a| a.success_rate)
                .unwrap_or(0.0),
        };

        self.healing_actions.insert(id, action.clone());
        Some(action)
    }

    /// Execute a healing action
    async fn execute_healing_action(&mut self, action: &HealingAction) -> Result<()> {
        let start_time = SystemTime::now();
        let success = match &action.action_type {
            HealingActionType::MemoryCleanup => {
                self.perform_memory_cleanup().await?;
                true
            }
            HealingActionType::ClearCache => {
                self.clear_caches().await?;
                true
            }
            HealingActionType::RestartComponent => {
                self.restart_component(&action.target_component).await?;
                true
            }
            HealingActionType::ResetCircuitBreaker => {
                self.reset_circuit_breaker(&action.target_component).await?;
                true
            }
            HealingActionType::ScaleResources => {
                self.scale_resources(&action.target_component).await?;
                true
            }
            HealingActionType::RollbackConfiguration => {
                self.rollback_configuration(&action.target_component)
                    .await?;
                true
            }
            HealingActionType::FlushConnections => {
                self.flush_connections(&action.target_component).await?;
                true
            }
            HealingActionType::CompactDatabase => {
                self.compact_database().await?;
                true
            }
        };

        // Update statistics
        self.recovery_stats.total_recoveries += 1;
        if success {
            self.recovery_stats.successful_recoveries += 1;
        } else {
            self.recovery_stats.failed_recoveries += 1;
        }

        let recovery_time = SystemTime::now()
            .duration_since(start_time)
            .unwrap_or(Duration::ZERO);
        self.update_average_recovery_time(recovery_time);

        // Update action success rate
        if let Some(stored_action) = self.healing_actions.get_mut(&action.id) {
            stored_action.success_rate = self.recovery_stats.successful_recoveries as f32
                / self.recovery_stats.total_recoveries as f32;
        }

        Ok(())
    }

    /// Perform memory cleanup operations
    async fn perform_memory_cleanup(&self) -> Result<()> {
        // Force garbage collection in Rust is limited, but we can suggest cleanup
        // In a real implementation, this might clear caches, drop unused connections, etc.
        tokio::time::sleep(Duration::from_millis(100)).await; // Simulate cleanup
        Ok(())
    }

    /// Clear various system caches
    async fn clear_caches(&self) -> Result<()> {
        // Would clear application caches, connection pools, etc.
        tokio::time::sleep(Duration::from_millis(200)).await; // Simulate cache clearing
        Ok(())
    }

    /// Restart a specific component
    async fn restart_component(&self, _component: &str) -> Result<()> {
        // Would restart the specified component/service
        tokio::time::sleep(Duration::from_millis(500)).await; // Simulate restart
        Ok(())
    }

    /// Reset circuit breakers for a component
    async fn reset_circuit_breaker(&self, _component: &str) -> Result<()> {
        // Would reset circuit breakers to allow new requests
        tokio::time::sleep(Duration::from_millis(50)).await; // Simulate reset
        Ok(())
    }

    /// Scale resources for a component
    async fn scale_resources(&self, _component: &str) -> Result<()> {
        // Would scale up resources (CPU, memory, instances)
        tokio::time::sleep(Duration::from_millis(1000)).await; // Simulate scaling
        Ok(())
    }

    /// Rollback to a previous configuration
    async fn rollback_configuration(&self, _component: &str) -> Result<()> {
        // Would rollback to last known good configuration
        tokio::time::sleep(Duration::from_millis(300)).await; // Simulate rollback
        Ok(())
    }

    /// Flush network connections
    async fn flush_connections(&self, _component: &str) -> Result<()> {
        // Would flush stale network connections
        tokio::time::sleep(Duration::from_millis(100)).await; // Simulate flush
        Ok(())
    }

    /// Compact database to reclaim space
    async fn compact_database(&self) -> Result<()> {
        // Would perform database compaction/optimization
        tokio::time::sleep(Duration::from_millis(2000)).await; // Simulate compaction
        Ok(())
    }

    /// Update average recovery time
    fn update_average_recovery_time(&mut self, recovery_time: Duration) {
        if self.recovery_stats.total_recoveries == 1 {
            self.recovery_stats.average_recovery_time = recovery_time;
        } else {
            // Calculate weighted average
            let current_total = self.recovery_stats.average_recovery_time.as_millis()
                * (self.recovery_stats.total_recoveries - 1) as u128;
            let new_total = current_total + recovery_time.as_millis();
            self.recovery_stats.average_recovery_time = Duration::from_millis(
                (new_total / self.recovery_stats.total_recoveries as u128) as u64,
            );
        }
    }

    /// Get recovery statistics
    pub fn get_recovery_stats(&self) -> &RecoveryStats {
        &self.recovery_stats
    }

    /// Get active healing actions
    pub fn get_active_healing_actions(&self) -> Vec<&HealingAction> {
        self.healing_actions.values().collect()
    }

    /// Check if system is in recovery mode
    pub fn is_in_recovery_mode(&self) -> bool {
        self.healing_actions.iter().any(|(_, action)| {
            action.last_attempt.is_some_and(|last| {
                SystemTime::now()
                    .duration_since(last)
                    .unwrap_or(Duration::MAX)
                    < self.config.recovery_cooldown
            })
        })
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
