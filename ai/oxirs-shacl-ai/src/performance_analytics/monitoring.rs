//! Enhanced real-time monitoring functionality with advanced metrics and alerting

use crate::performance_analytics::config::MonitoringConfig;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Alert severity levels for monitoring events
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Performance alert with details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub id: String,
    pub severity: AlertSeverity,
    pub metric_name: String,
    pub current_value: f64,
    pub threshold: f64,
    pub message: String,
    pub timestamp: SystemTime,
    pub resolved: bool,
}

/// Real-time metrics collector with sliding window
#[derive(Debug)]
pub struct MetricsCollector {
    metrics: Arc<RwLock<HashMap<String, VecDeque<(Instant, f64)>>>>,
    window_size: Duration,
    max_samples: usize,
}

impl MetricsCollector {
    pub fn new(window_size: Duration, max_samples: usize) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            window_size,
            max_samples,
        }
    }

    /// Record a metric value
    pub fn record(&self, metric_name: &str, value: f64) -> crate::Result<()> {
        let mut metrics = self.metrics.write().map_err(|_| {
            crate::ShaclAiError::PerformanceAnalytics("Failed to acquire metrics write lock".into())
        })?;
        
        let entry = metrics.entry(metric_name.to_string()).or_insert_with(VecDeque::new);
        let now = Instant::now();
        
        // Add new sample
        entry.push_back((now, value));
        
        // Remove old samples outside window
        while let Some((timestamp, _)) = entry.front() {
            if now.duration_since(*timestamp) > self.window_size {
                entry.pop_front();
            } else {
                break;
            }
        }
        
        // Limit samples to max size
        while entry.len() > self.max_samples {
            entry.pop_front();
        }
        
        Ok(())
    }

    /// Get current average for a metric
    pub fn get_average(&self, metric_name: &str) -> crate::Result<Option<f64>> {
        let metrics = self.metrics.read().map_err(|_| {
            crate::ShaclAiError::PerformanceAnalytics("Failed to acquire metrics read lock".into())
        })?;
        
        if let Some(values) = metrics.get(metric_name) {
            if values.is_empty() {
                return Ok(None);
            }
            
            let sum: f64 = values.iter().map(|(_, value)| value).sum();
            Ok(Some(sum / values.len() as f64))
        } else {
            Ok(None)
        }
    }

    /// Get metric trend (positive = increasing, negative = decreasing)
    pub fn get_trend(&self, metric_name: &str) -> crate::Result<Option<f64>> {
        let metrics = self.metrics.read().map_err(|_| {
            crate::ShaclAiError::PerformanceAnalytics("Failed to acquire metrics read lock".into())
        })?;
        
        if let Some(values) = metrics.get(metric_name) {
            if values.len() < 2 {
                return Ok(None);
            }
            
            let recent_half = values.len() / 2;
            let older_avg: f64 = values.iter().take(recent_half).map(|(_, v)| v).sum::<f64>() / recent_half as f64;
            let recent_avg: f64 = values.iter().skip(recent_half).map(|(_, v)| v).sum::<f64>() / (values.len() - recent_half) as f64;
            
            Ok(Some(recent_avg - older_avg))
        } else {
            Ok(None)
        }
    }
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThreshold {
    pub metric_name: String,
    pub warning_threshold: f64,
    pub critical_threshold: f64,
    pub emergency_threshold: f64,
    pub comparison: ThresholdComparison,
}

/// Threshold comparison type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdComparison {
    GreaterThan,
    LessThan,
    Equals,
}

/// Enhanced real-time performance monitor with alerting and analytics
#[derive(Debug)]
pub struct RealTimeMonitor {
    config: MonitoringConfig,
    metrics_collector: MetricsCollector,
    alert_thresholds: Vec<AlertThreshold>,
    active_alerts: Arc<Mutex<HashMap<String, PerformanceAlert>>>,
    alert_sender: Option<mpsc::UnboundedSender<PerformanceAlert>>,
    is_running: bool,
    monitoring_handle: Option<tokio::task::JoinHandle<()>>,
}

impl RealTimeMonitor {
    pub fn new() -> Self {
        Self {
            config: MonitoringConfig::default(),
            metrics_collector: MetricsCollector::new(Duration::from_secs(300), 1000), // 5 min window, 1000 samples
            alert_thresholds: Vec::new(),
            active_alerts: Arc::new(Mutex::new(HashMap::new())),
            alert_sender: None,
            is_running: false,
            monitoring_handle: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: MonitoringConfig) -> Self {
        Self {
            config,
            metrics_collector: MetricsCollector::new(Duration::from_secs(300), 1000),
            alert_thresholds: Vec::new(),
            active_alerts: Arc::new(Mutex::new(HashMap::new())),
            alert_sender: None,
            is_running: false,
            monitoring_handle: None,
        }
    }

    /// Add alert threshold
    pub fn add_alert_threshold(&mut self, threshold: AlertThreshold) {
        self.alert_thresholds.push(threshold);
    }

    /// Record a performance metric
    pub fn record_metric(&self, metric_name: &str, value: f64) -> crate::Result<()> {
        self.metrics_collector.record(metric_name, value)?;
        
        // Check for threshold violations
        self.check_thresholds(metric_name, value)?;
        
        debug!("Recorded metric: {} = {}", metric_name, value);
        Ok(())
    }

    /// Get current metric average
    pub fn get_metric_average(&self, metric_name: &str) -> crate::Result<Option<f64>> {
        self.metrics_collector.get_average(metric_name)
    }

    /// Get metric trend
    pub fn get_metric_trend(&self, metric_name: &str) -> crate::Result<Option<f64>> {
        self.metrics_collector.get_trend(metric_name)
    }

    /// Start monitoring with alert handling
    pub async fn start(&mut self) -> crate::Result<mpsc::UnboundedReceiver<PerformanceAlert>> {
        if self.is_running {
            return Err(crate::ShaclAiError::PerformanceAnalytics("Monitor already running".into()));
        }

        let (tx, rx) = mpsc::unbounded_channel();
        self.alert_sender = Some(tx.clone());
        self.is_running = true;

        // Start background monitoring task
        let active_alerts = Arc::clone(&self.active_alerts);
        let alert_sender = tx;
        
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                // Perform periodic monitoring tasks
                if let Err(e) = Self::periodic_monitoring(&active_alerts, &alert_sender).await {
                    error!("Error in periodic monitoring: {}", e);
                }
            }
        });

        self.monitoring_handle = Some(handle);
        info!("Real-time monitor started");
        Ok(rx)
    }

    /// Stop monitoring
    pub async fn stop(&mut self) -> crate::Result<()> {
        if !self.is_running {
            return Ok(());
        }

        self.is_running = false;
        
        if let Some(handle) = self.monitoring_handle.take() {
            handle.abort();
        }
        
        self.alert_sender = None;
        info!("Real-time monitor stopped");
        Ok(())
    }

    /// Check thresholds for a metric value
    fn check_thresholds(&self, metric_name: &str, value: f64) -> crate::Result<()> {
        for threshold in &self.alert_thresholds {
            if threshold.metric_name != metric_name {
                continue;
            }

            let severity = self.evaluate_threshold(threshold, value);
            
            if let Some(severity) = severity {
                let alert = PerformanceAlert {
                    id: format!("{}_{}", metric_name, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()),
                    severity: severity.clone(),
                    metric_name: metric_name.to_string(),
                    current_value: value,
                    threshold: self.get_threshold_value(threshold, &severity),
                    message: format!("Metric '{}' exceeded {} threshold: {} > {}", 
                                   metric_name, 
                                   match severity {
                                       AlertSeverity::Warning => "warning",
                                       AlertSeverity::Critical => "critical", 
                                       AlertSeverity::Emergency => "emergency",
                                       _ => "info"
                                   },
                                   value, 
                                   self.get_threshold_value(threshold, &severity)),
                    timestamp: SystemTime::now(),
                    resolved: false,
                };

                // Store active alert
                if let Ok(mut alerts) = self.active_alerts.lock() {
                    alerts.insert(alert.id.clone(), alert.clone());
                }

                // Send alert if sender is available
                if let Some(sender) = &self.alert_sender {
                    if let Err(e) = sender.send(alert) {
                        warn!("Failed to send alert: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Evaluate threshold for a value
    fn evaluate_threshold(&self, threshold: &AlertThreshold, value: f64) -> Option<AlertSeverity> {
        let exceeds_threshold = |threshold_value: f64| -> bool {
            match threshold.comparison {
                ThresholdComparison::GreaterThan => value > threshold_value,
                ThresholdComparison::LessThan => value < threshold_value,
                ThresholdComparison::Equals => (value - threshold_value).abs() < f64::EPSILON,
            }
        };

        if exceeds_threshold(threshold.emergency_threshold) {
            Some(AlertSeverity::Emergency)
        } else if exceeds_threshold(threshold.critical_threshold) {
            Some(AlertSeverity::Critical)
        } else if exceeds_threshold(threshold.warning_threshold) {
            Some(AlertSeverity::Warning)
        } else {
            None
        }
    }

    /// Get threshold value for severity level
    fn get_threshold_value(&self, threshold: &AlertThreshold, severity: &AlertSeverity) -> f64 {
        match severity {
            AlertSeverity::Warning => threshold.warning_threshold,
            AlertSeverity::Critical => threshold.critical_threshold,
            AlertSeverity::Emergency => threshold.emergency_threshold,
            _ => threshold.warning_threshold,
        }
    }

    /// Periodic monitoring tasks
    async fn periodic_monitoring(
        active_alerts: &Arc<Mutex<HashMap<String, PerformanceAlert>>>,
        _alert_sender: &mpsc::UnboundedSender<PerformanceAlert>,
    ) -> crate::Result<()> {
        // Clean up old resolved alerts
        if let Ok(mut alerts) = active_alerts.lock() {
            alerts.retain(|_, alert| {
                SystemTime::now().duration_since(alert.timestamp).unwrap_or(Duration::ZERO) 
                    < Duration::from_secs(3600) // Keep alerts for 1 hour
            });
        }

        Ok(())
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> crate::Result<Vec<PerformanceAlert>> {
        let alerts = self.active_alerts.lock().map_err(|_| {
            crate::ShaclAiError::Performance("Failed to acquire alerts lock".into())
        })?;
        
        Ok(alerts.values().cloned().collect())
    }

    /// Get monitoring statistics
    pub fn get_monitoring_stats(&self) -> crate::Result<MonitoringStats> {
        let alerts = self.get_active_alerts()?;
        
        let alert_counts = alerts.iter().fold(HashMap::new(), |mut acc, alert| {
            *acc.entry(alert.severity.clone()).or_insert(0) += 1;
            acc
        });

        Ok(MonitoringStats {
            is_running: self.is_running,
            total_alerts: alerts.len(),
            alert_counts,
            threshold_count: self.alert_thresholds.len(),
        })
    }
}

impl Default for RealTimeMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Monitoring statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringStats {
    pub is_running: bool,
    pub total_alerts: usize,
    pub alert_counts: HashMap<AlertSeverity, usize>,
    pub threshold_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new(Duration::from_secs(60), 100);
        
        // Record some metrics
        collector.record("cpu_usage", 50.0).unwrap();
        collector.record("cpu_usage", 60.0).unwrap();
        collector.record("cpu_usage", 70.0).unwrap();
        
        // Check average
        let avg = collector.get_average("cpu_usage").unwrap().unwrap();
        assert!((avg - 60.0).abs() < f64::EPSILON);
        
        // Check trend (should be positive)
        let trend = collector.get_trend("cpu_usage").unwrap().unwrap();
        assert!(trend > 0.0);
    }

    #[test]
    fn test_alert_threshold() {
        let threshold = AlertThreshold {
            metric_name: "memory_usage".to_string(),
            warning_threshold: 80.0,
            critical_threshold: 90.0,
            emergency_threshold: 95.0,
            comparison: ThresholdComparison::GreaterThan,
        };

        let monitor = RealTimeMonitor::new();
        
        // Test threshold evaluation
        assert_eq!(monitor.evaluate_threshold(&threshold, 75.0), None);
        assert_eq!(monitor.evaluate_threshold(&threshold, 85.0), Some(AlertSeverity::Warning));
        assert_eq!(monitor.evaluate_threshold(&threshold, 92.0), Some(AlertSeverity::Critical));
        assert_eq!(monitor.evaluate_threshold(&threshold, 97.0), Some(AlertSeverity::Emergency));
    }

    #[tokio::test]
    async fn test_monitor_lifecycle() {
        let mut monitor = RealTimeMonitor::new();
        
        // Start monitoring
        let _alert_rx = monitor.start().await.unwrap();
        assert!(monitor.is_running);
        
        // Record some metrics
        monitor.record_metric("test_metric", 100.0).unwrap();
        
        // Stop monitoring
        monitor.stop().await.unwrap();
        assert!(!monitor.is_running);
    }
}
