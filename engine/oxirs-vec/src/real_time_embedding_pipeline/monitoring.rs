//! Performance monitoring and alerting for the real-time embedding pipeline
//!
//! This module provides comprehensive monitoring capabilities including performance metrics,
//! alerting, health checks, and observability for the embedding pipeline.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::traits::{
    Alert, AlertCategory, AlertConfig, AlertHandler, AlertSeverity, AlertThrottling,
    MetricPoint, MetricsStorage, HealthStatus,
};
use super::types::{PerformanceMetrics, ResourceUtilization, StreamStatus};
use super::config::MonitoringConfig;
use super::PipelineError;

/// Performance monitoring manager for the pipeline
pub struct PipelinePerformanceMonitor {
    config: MonitoringConfig,
    metrics_storage: Arc<dyn MetricsStorage>,
    alert_manager: Arc<AlertManager>,
    health_checker: Arc<HealthChecker>,
    metrics_collector: Arc<MetricsCollector>,
    is_running: Arc<Mutex<bool>>,
}

impl PipelinePerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(
        config: MonitoringConfig,
        metrics_storage: Arc<dyn MetricsStorage>,
        alert_handler: Arc<dyn AlertHandler>,
    ) -> Self {
        let alert_manager = Arc::new(AlertManager::new(alert_handler));
        let health_checker = Arc::new(HealthChecker::new());
        let metrics_collector = Arc::new(MetricsCollector::new());

        Self {
            config,
            metrics_storage,
            alert_manager,
            health_checker,
            metrics_collector,
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    /// Start the monitoring system
    pub async fn start(&self) -> Result<(), PipelineError> {
        let mut running = self.is_running.lock().await;
        if *running {
            return Err(PipelineError::AlreadyRunning);
        }

        *running = true;

        // Start metrics collection
        self.start_metrics_collection().await;
        
        // Start health monitoring
        self.start_health_monitoring().await;
        
        // Start alert processing
        self.start_alert_processing().await;

        Ok(())
    }

    /// Stop the monitoring system
    pub async fn stop(&self) -> Result<(), PipelineError> {
        let mut running = self.is_running.lock().await;
        if !*running {
            return Err(PipelineError::NotRunning);
        }

        *running = false;
        Ok(())
    }

    /// Record a performance metric
    pub async fn record_metric(&self, metric: MetricPoint) -> Result<(), PipelineError> {
        self.metrics_storage.store_metric(metric).await
            .map_err(|e| PipelineError::MonitoringError {
                message: format!("Failed to store metric: {}", e),
            })
    }

    /// Get current performance metrics
    pub async fn get_current_metrics(&self) -> Result<PerformanceMetrics, PipelineError> {
        self.metrics_collector.get_current_metrics().await
    }

    /// Get health status
    pub async fn get_health_status(&self) -> Result<HealthStatus, PipelineError> {
        self.health_checker.get_overall_health().await
    }

    /// Register an alert
    pub async fn register_alert(&self, alert: Alert) -> Result<(), PipelineError> {
        self.alert_manager.register_alert(alert).await
    }

    // Private helper methods

    async fn start_metrics_collection(&self) {
        let metrics_collector = Arc::clone(&self.metrics_collector);
        let metrics_storage = Arc::clone(&self.metrics_storage);
        let is_running = Arc::clone(&self.is_running);
        let collection_interval = self.config.metrics_collection_interval;

        tokio::spawn(async move {
            while *is_running.lock().await {
                if let Ok(metrics) = metrics_collector.collect_metrics().await {
                    for metric in metrics {
                        let _ = metrics_storage.store_metric(metric).await;
                    }
                }
                
                tokio::time::sleep(collection_interval).await;
            }
        });
    }

    async fn start_health_monitoring(&self) {
        let health_checker = Arc::clone(&self.health_checker);
        let alert_manager = Arc::clone(&self.alert_manager);
        let is_running = Arc::clone(&self.is_running);
        let health_check_interval = self.config.health_check_interval;

        tokio::spawn(async move {
            while *is_running.lock().await {
                if let Ok(health_status) = health_checker.check_health().await {
                    if health_status != HealthStatus::Healthy {
                        let alert = Alert {
                            id: Uuid::new_v4(),
                            category: AlertCategory::Health,
                            severity: AlertSeverity::Warning,
                            message: format!("Health check failed: {:?}", health_status),
                            timestamp: SystemTime::now(),
                            source: "health_monitor".to_string(),
                            metadata: HashMap::new(),
                        };
                        
                        let _ = alert_manager.register_alert(alert).await;
                    }
                }
                
                tokio::time::sleep(health_check_interval).await;
            }
        });
    }

    async fn start_alert_processing(&self) {
        let alert_manager = Arc::clone(&self.alert_manager);
        let is_running = Arc::clone(&self.is_running);

        tokio::spawn(async move {
            while *is_running.lock().await {
                let _ = alert_manager.process_alerts().await;
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
    }
}

/// Alert manager for handling pipeline alerts
pub struct AlertManager {
    alert_handler: Arc<dyn AlertHandler>,
    active_alerts: Arc<RwLock<HashMap<Uuid, Alert>>>,
    alert_history: Arc<Mutex<VecDeque<Alert>>>,
    throttling_state: Arc<RwLock<HashMap<String, Instant>>>,
}

impl AlertManager {
    pub fn new(alert_handler: Arc<dyn AlertHandler>) -> Self {
        Self {
            alert_handler,
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(Mutex::new(VecDeque::new())),
            throttling_state: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register_alert(&self, alert: Alert) -> Result<(), PipelineError> {
        // Check throttling
        if self.should_throttle_alert(&alert).await {
            return Ok(());
        }

        // Store alert
        {
            let mut active_alerts = self.active_alerts.write().await;
            active_alerts.insert(alert.id, alert.clone());
        }

        // Add to history
        {
            let mut history = self.alert_history.lock().await;
            history.push_back(alert.clone());
            
            // Keep only recent alerts
            while history.len() > 1000 {
                history.pop_front();
            }
        }

        // Handle alert
        self.alert_handler.handle_alert(alert).await
            .map_err(|e| PipelineError::MonitoringError {
                message: format!("Alert handling failed: {}", e),
            })
    }

    pub async fn process_alerts(&self) -> Result<(), PipelineError> {
        // Process pending alerts
        let alerts = {
            self.active_alerts.read().await.values().cloned().collect::<Vec<_>>()
        };

        for alert in alerts {
            // Check if alert should be escalated or resolved
            self.check_alert_status(alert).await?;
        }

        Ok(())
    }

    pub async fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.read().await.values().cloned().collect()
    }

    async fn should_throttle_alert(&self, alert: &Alert) -> bool {
        let throttle_key = format!("{}:{}", alert.category, alert.source);
        let throttling_state = self.throttling_state.read().await;
        
        if let Some(last_sent) = throttling_state.get(&throttle_key) {
            let throttle_duration = match alert.severity {
                AlertSeverity::Critical => Duration::from_secs(60),
                AlertSeverity::Error => Duration::from_secs(300),
                AlertSeverity::Warning => Duration::from_secs(600),
                AlertSeverity::Info => Duration::from_secs(1200),
            };
            
            last_sent.elapsed() < throttle_duration
        } else {
            false
        }
    }

    async fn check_alert_status(&self, _alert: Alert) -> Result<(), PipelineError> {
        // Implementation for checking alert status and potential resolution
        Ok(())
    }
}

/// Health checker for monitoring system health
pub struct HealthChecker {
    component_health: Arc<RwLock<HashMap<String, HealthStatus>>>,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            component_health: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn check_health(&self) -> Result<HealthStatus, PipelineError> {
        // Implement comprehensive health checking
        let component_health = self.component_health.read().await;
        
        if component_health.is_empty() {
            return Ok(HealthStatus::Healthy);
        }

        // Check if any component is unhealthy
        for status in component_health.values() {
            match status {
                HealthStatus::Critical | HealthStatus::Unhealthy => return Ok(*status),
                _ => continue,
            }
        }

        // Check if any component is degraded
        for status in component_health.values() {
            if *status == HealthStatus::Degraded {
                return Ok(HealthStatus::Degraded);
            }
        }

        Ok(HealthStatus::Healthy)
    }

    pub async fn get_overall_health(&self) -> Result<HealthStatus, PipelineError> {
        self.check_health().await
    }

    pub async fn update_component_health(&self, component: String, status: HealthStatus) {
        let mut component_health = self.component_health.write().await;
        component_health.insert(component, status);
    }
}

/// Metrics collector for gathering performance data
pub struct MetricsCollector {
    current_metrics: Arc<RwLock<PerformanceMetrics>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            current_metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
        }
    }

    pub async fn collect_metrics(&self) -> Result<Vec<MetricPoint>, PipelineError> {
        let mut metrics = Vec::new();
        let timestamp = SystemTime::now();

        // Collect system metrics
        metrics.push(MetricPoint {
            name: "cpu_usage".to_string(),
            value: self.get_cpu_usage().await,
            timestamp,
            labels: HashMap::new(),
        });

        metrics.push(MetricPoint {
            name: "memory_usage".to_string(),
            value: self.get_memory_usage().await,
            timestamp,
            labels: HashMap::new(),
        });

        metrics.push(MetricPoint {
            name: "embedding_throughput".to_string(),
            value: self.get_embedding_throughput().await,
            timestamp,
            labels: HashMap::new(),
        });

        Ok(metrics)
    }

    pub async fn get_current_metrics(&self) -> Result<PerformanceMetrics, PipelineError> {
        let metrics = self.current_metrics.read().await;
        Ok(metrics.clone())
    }

    async fn get_cpu_usage(&self) -> f64 {
        // Implementation would use system APIs to get actual CPU usage
        // For now, return a simulated value
        50.0
    }

    async fn get_memory_usage(&self) -> f64 {
        // Implementation would use system APIs to get actual memory usage
        // For now, return a simulated value
        1024.0 * 1024.0 * 512.0 // 512 MB
    }

    async fn get_embedding_throughput(&self) -> f64 {
        // Implementation would track actual embedding processing throughput
        // For now, return a simulated value
        100.0 // embeddings per second
    }
}

/// In-memory metrics storage implementation
pub struct InMemoryMetricsStorage {
    metrics: Arc<RwLock<VecDeque<MetricPoint>>>,
    max_metrics: usize,
}

impl InMemoryMetricsStorage {
    pub fn new(max_metrics: usize) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(VecDeque::new())),
            max_metrics,
        }
    }
}

#[async_trait::async_trait]
impl MetricsStorage for InMemoryMetricsStorage {
    async fn store_metric(&self, metric: MetricPoint) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut metrics = self.metrics.write().await;
        metrics.push_back(metric);
        
        while metrics.len() > self.max_metrics {
            metrics.pop_front();
        }
        
        Ok(())
    }

    async fn get_metrics(
        &self,
        metric_name: &str,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> Result<Vec<MetricPoint>, Box<dyn std::error::Error + Send + Sync>> {
        let metrics = self.metrics.read().await;
        
        let filtered_metrics: Vec<MetricPoint> = metrics
            .iter()
            .filter(|m| {
                m.name == metric_name
                    && m.timestamp >= start_time
                    && m.timestamp <= end_time
            })
            .cloned()
            .collect();
        
        Ok(filtered_metrics)
    }
}

/// Console-based alert handler implementation
pub struct ConsoleAlertHandler;

#[async_trait::async_trait]
impl AlertHandler for ConsoleAlertHandler {
    async fn handle_alert(&self, alert: Alert) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!(
            "[ALERT] {} - {} - {} - {}",
            alert.severity,
            alert.category,
            alert.source,
            alert.message
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        let metrics = collector.collect_metrics().await.unwrap();
        assert!(!metrics.is_empty());
    }

    #[tokio::test]
    async fn test_health_checker() {
        let checker = HealthChecker::new();
        let health = checker.get_overall_health().await.unwrap();
        assert_eq!(health, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_alert_manager() {
        let handler = Arc::new(ConsoleAlertHandler);
        let manager = AlertManager::new(handler);
        
        let alert = Alert {
            id: Uuid::new_v4(),
            category: AlertCategory::Performance,
            severity: AlertSeverity::Warning,
            message: "Test alert".to_string(),
            timestamp: SystemTime::now(),
            source: "test".to_string(),
            metadata: HashMap::new(),
        };
        
        let result = manager.register_alert(alert).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_in_memory_metrics_storage() {
        let storage = InMemoryMetricsStorage::new(100);
        
        let metric = MetricPoint {
            name: "test_metric".to_string(),
            value: 42.0,
            timestamp: SystemTime::now(),
            labels: HashMap::new(),
        };
        
        let result = storage.store_metric(metric).await;
        assert!(result.is_ok());
    }
}