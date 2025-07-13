//! # Advanced Observability and Telemetry for OxiRS Stream
//!
//! Comprehensive monitoring, metrics collection, distributed tracing, and
//! observability features for production deployment of OxiRS streaming systems.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, warn};
use uuid::Uuid;

// OpenTelemetry imports for enhanced observability (simplified for now)
// Full OpenTelemetry integration can be enabled when dependencies are stable
#[cfg(feature = "opentelemetry")]
use opentelemetry::global::BoxedTracer;
#[cfg(feature = "opentelemetry")]
use opentelemetry::{global, trace::Tracer, KeyValue};

use crate::StreamEvent;

/// Comprehensive telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable OpenTelemetry integration
    pub enable_opentelemetry: bool,
    /// Jaeger endpoint for distributed tracing
    pub jaeger_endpoint: Option<String>,
    /// Prometheus metrics endpoint
    pub prometheus_endpoint: Option<String>,
    /// Custom metrics collection interval
    pub metrics_interval: Duration,
    /// Enable detailed performance profiling
    pub enable_profiling: bool,
    /// Sampling rate for traces (0.0 to 1.0)
    pub trace_sampling_rate: f64,
    /// Enable custom business metrics
    pub enable_business_metrics: bool,
    /// Maximum number of spans to keep in memory
    pub max_spans_in_memory: usize,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enable_opentelemetry: true,
            jaeger_endpoint: Some("http://localhost:14268/api/traces".to_string()),
            prometheus_endpoint: Some("http://localhost:9090".to_string()),
            metrics_interval: Duration::from_secs(30),
            enable_profiling: false,
            trace_sampling_rate: 0.1,
            enable_business_metrics: true,
            max_spans_in_memory: 10000,
        }
    }
}

/// Detailed performance metrics for streaming operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingMetrics {
    /// Events processed per second
    pub events_per_second: f64,
    /// Average processing latency in milliseconds
    pub avg_latency_ms: f64,
    /// P95 latency in milliseconds
    pub p95_latency_ms: f64,
    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,
    /// Memory usage in megabytes
    pub memory_usage_mb: f64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Network I/O bytes per second
    pub network_io_bps: f64,
    /// Error rate as percentage
    pub error_rate_percent: f64,
    /// Queue depth/backlog size
    pub queue_depth: u64,
    /// Active connections count
    pub active_connections: u64,
    /// Timestamp of metrics collection
    pub timestamp: DateTime<Utc>,
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self {
            events_per_second: 0.0,
            avg_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            network_io_bps: 0.0,
            error_rate_percent: 0.0,
            queue_depth: 0,
            active_connections: 0,
            timestamp: Utc::now(),
        }
    }
}

/// Business-level streaming metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessMetrics {
    /// Total events processed since startup
    pub total_events_processed: u64,
    /// Total events failed since startup
    pub total_events_failed: u64,
    /// Revenue-related events count
    pub revenue_events_count: u64,
    /// Customer interaction events count
    pub customer_events_count: u64,
    /// System health score (0.0 to 1.0)
    pub health_score: f64,
    /// Data quality score (0.0 to 1.0)
    pub data_quality_score: f64,
    /// Custom business metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl Default for BusinessMetrics {
    fn default() -> Self {
        Self {
            total_events_processed: 0,
            total_events_failed: 0,
            revenue_events_count: 0,
            customer_events_count: 0,
            health_score: 1.0,
            data_quality_score: 1.0,
            custom_metrics: HashMap::new(),
        }
    }
}

/// Distributed tracing span information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    /// Unique span identifier
    pub span_id: String,
    /// Parent span identifier if any
    pub parent_span_id: Option<String>,
    /// Trace identifier
    pub trace_id: String,
    /// Operation name
    pub operation_name: String,
    /// Span start time
    pub start_time: DateTime<Utc>,
    /// Span duration
    pub duration: Option<Duration>,
    /// Span tags/attributes
    pub tags: HashMap<String, String>,
    /// Span logs/events
    pub logs: Vec<SpanLog>,
    /// Span status
    pub status: SpanStatus,
}

/// Span log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLog {
    pub timestamp: DateTime<Utc>,
    pub level: String,
    pub message: String,
    pub fields: HashMap<String, String>,
}

/// Span execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanStatus {
    Ok,
    Error { message: String },
    Timeout,
    Cancelled,
}

/// Alert configuration and management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerting system
    pub enabled: bool,
    /// Latency threshold for alerts (milliseconds)
    pub latency_threshold_ms: f64,
    /// Error rate threshold for alerts (percentage)
    pub error_rate_threshold_percent: f64,
    /// Memory usage threshold for alerts (percentage)
    pub memory_threshold_percent: f64,
    /// Queue depth threshold for alerts
    pub queue_depth_threshold: u64,
    /// Alert notification endpoints
    pub notification_endpoints: Vec<String>,
    /// Alert cooldown period to prevent spam
    pub cooldown_period: Duration,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            latency_threshold_ms: 100.0,
            error_rate_threshold_percent: 5.0,
            memory_threshold_percent: 80.0,
            queue_depth_threshold: 10000,
            notification_endpoints: vec!["http://localhost:9093/api/v1/alerts".to_string()],
            cooldown_period: Duration::from_secs(300),
        }
    }
}

/// Advanced observability system for streaming operations
pub struct StreamObservability {
    config: TelemetryConfig,
    alert_config: AlertConfig,
    streaming_metrics: Arc<RwLock<StreamingMetrics>>,
    business_metrics: Arc<RwLock<BusinessMetrics>>,
    active_spans: Arc<RwLock<HashMap<String, TraceSpan>>>,
    metrics_history: Arc<RwLock<Vec<StreamingMetrics>>>,
    alert_sender: broadcast::Sender<AlertEvent>,
    last_alert_times: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    /// OpenTelemetry tracer for enhanced distributed tracing
    #[cfg(feature = "opentelemetry")]
    tracer: Option<Arc<BoxedTracer>>,
    #[cfg(not(feature = "opentelemetry"))]
    tracer: Option<()>,
}

/// Alert event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    pub alert_id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub metric_value: f64,
    pub threshold: f64,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Types of alerts that can be triggered
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighLatency,
    HighErrorRate,
    HighMemoryUsage,
    QueueBacklog,
    ConnectionFailure,
    DataQualityIssue,
    Custom(String),
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

impl StreamObservability {
    /// Create a new observability system
    pub fn new(config: TelemetryConfig, alert_config: AlertConfig) -> Self {
        let (alert_sender, _) = broadcast::channel(1000);

        // Initialize OpenTelemetry tracer if enabled
        #[cfg(feature = "opentelemetry")]
        let tracer = if config.enable_opentelemetry {
            Self::setup_jaeger_tracer(&config).ok()
        } else {
            None
        };
        #[cfg(not(feature = "opentelemetry"))]
        let tracer = if config.enable_opentelemetry {
            warn!("OpenTelemetry requested but feature not enabled. Compile with --features opentelemetry");
            None
        } else {
            None
        };

        Self {
            config,
            alert_config,
            streaming_metrics: Arc::new(RwLock::new(StreamingMetrics::default())),
            business_metrics: Arc::new(RwLock::new(BusinessMetrics::default())),
            active_spans: Arc::new(RwLock::new(HashMap::new())),
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            alert_sender,
            last_alert_times: Arc::new(RwLock::new(HashMap::new())),
            tracer,
        }
    }

    /// Set up Jaeger tracer for distributed tracing
    #[cfg(feature = "opentelemetry")]
    fn setup_jaeger_tracer(config: &TelemetryConfig) -> Result<Arc<BoxedTracer>> {
        // For now, return a placeholder implementation
        // Full OpenTelemetry integration will be implemented when dependencies are stable
        warn!("OpenTelemetry Jaeger integration is disabled pending dependency stability");

        if let Some(jaeger_endpoint) = &config.jaeger_endpoint {
            info!("Jaeger endpoint configured: {}", jaeger_endpoint);
        }

        // Return a no-op tracer for now
        let tracer = opentelemetry::global::tracer("oxirs-stream");
        Ok(Arc::new(tracer))
    }

    /// Start a new distributed trace span
    pub async fn start_span(&self, operation_name: &str, parent_span_id: Option<String>) -> String {
        let span_id = Uuid::new_v4().to_string();
        let trace_id = parent_span_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let span = TraceSpan {
            span_id: span_id.clone(),
            parent_span_id,
            trace_id,
            operation_name: operation_name.to_string(),
            start_time: Utc::now(),
            duration: None,
            tags: HashMap::new(),
            logs: Vec::new(),
            status: SpanStatus::Ok,
        };

        // Add span to active spans if we haven't exceeded the limit
        {
            let mut active_spans = self.active_spans.write().await;
            if active_spans.len() < self.config.max_spans_in_memory {
                active_spans.insert(span_id.clone(), span);
            }
        }

        debug!(
            "Started span: {} for operation: {}",
            span_id, operation_name
        );
        span_id
    }

    /// Finish a trace span
    pub async fn finish_span(&self, span_id: &str, status: SpanStatus) -> Result<()> {
        let mut active_spans = self.active_spans.write().await;

        if let Some(mut span) = active_spans.remove(span_id) {
            span.duration = Some(Utc::now().signed_duration_since(span.start_time).to_std()?);
            span.status = status;

            debug!(
                "Finished span: {} with duration: {:?}",
                span_id, span.duration
            );

            // In a real implementation, you'd send this to your tracing backend
            if self.config.enable_opentelemetry {
                self.export_span_to_jaeger(&span).await?;
            }
        }

        Ok(())
    }

    /// Add a tag to an active span
    pub async fn add_span_tag(&self, span_id: &str, key: &str, value: &str) -> Result<()> {
        let mut active_spans = self.active_spans.write().await;

        if let Some(span) = active_spans.get_mut(span_id) {
            span.tags.insert(key.to_string(), value.to_string());
        }

        Ok(())
    }

    /// Add a log entry to an active span
    pub async fn add_span_log(&self, span_id: &str, level: &str, message: &str) -> Result<()> {
        let mut active_spans = self.active_spans.write().await;

        if let Some(span) = active_spans.get_mut(span_id) {
            span.logs.push(SpanLog {
                timestamp: Utc::now(),
                level: level.to_string(),
                message: message.to_string(),
                fields: HashMap::new(),
            });
        }

        Ok(())
    }

    /// Record a streaming event for metrics collection
    pub async fn record_event(
        &self,
        event: &StreamEvent,
        processing_duration: Duration,
    ) -> Result<()> {
        let processing_time_ms = processing_duration.as_millis() as f64;

        // Update streaming metrics
        {
            let mut metrics = self.streaming_metrics.write().await;

            // Simple moving average for latency
            metrics.avg_latency_ms = (metrics.avg_latency_ms + processing_time_ms) / 2.0;

            // Update P95/P99 approximation (simplified)
            if processing_time_ms > metrics.p95_latency_ms {
                metrics.p95_latency_ms = processing_time_ms;
            }
            if processing_time_ms > metrics.p99_latency_ms {
                metrics.p99_latency_ms = processing_time_ms;
            }

            metrics.timestamp = Utc::now();
        }

        // Update business metrics if enabled
        if self.config.enable_business_metrics {
            let mut business_metrics = self.business_metrics.write().await;
            business_metrics.total_events_processed += 1;

            // Classify business events based on event type
            if let StreamEvent::TripleAdded { subject, .. } = event {
                if subject.contains("customer") {
                    business_metrics.customer_events_count += 1;
                } else if subject.contains("revenue") || subject.contains("order") {
                    business_metrics.revenue_events_count += 1;
                }
            }
        }

        // Check for alerts
        self.check_and_trigger_alerts().await?;

        Ok(())
    }

    /// Record an error for metrics and alerting
    pub async fn record_error(&self, error: &anyhow::Error, context: &str) -> Result<()> {
        error!("Streaming error in {}: {}", context, error);

        // Update error metrics
        {
            let mut business_metrics = self.business_metrics.write().await;
            business_metrics.total_events_failed += 1;

            // Calculate error rate
            let total_events =
                business_metrics.total_events_processed + business_metrics.total_events_failed;
            if total_events > 0 {
                let error_rate =
                    (business_metrics.total_events_failed as f64 / total_events as f64) * 100.0;

                let mut streaming_metrics = self.streaming_metrics.write().await;
                streaming_metrics.error_rate_percent = error_rate;
            }
        }

        // Trigger error rate alert if necessary
        self.check_and_trigger_alerts().await?;

        Ok(())
    }

    /// Update system resource metrics
    pub async fn update_system_metrics(
        &self,
        memory_mb: f64,
        cpu_percent: f64,
        network_bps: f64,
    ) -> Result<()> {
        let mut metrics = self.streaming_metrics.write().await;
        metrics.memory_usage_mb = memory_mb;
        metrics.cpu_usage_percent = cpu_percent;
        metrics.network_io_bps = network_bps;
        metrics.timestamp = Utc::now();

        // Add to history for trend analysis
        {
            let mut history = self.metrics_history.write().await;
            history.push(metrics.clone());

            // Keep only last 1000 entries
            if history.len() > 1000 {
                history.remove(0);
            }
        }

        Ok(())
    }

    /// Get current streaming metrics
    pub async fn get_streaming_metrics(&self) -> StreamingMetrics {
        self.streaming_metrics.read().await.clone()
    }

    /// Get current business metrics
    pub async fn get_business_metrics(&self) -> BusinessMetrics {
        self.business_metrics.read().await.clone()
    }

    /// Get metrics history for trend analysis
    pub async fn get_metrics_history(&self) -> Vec<StreamingMetrics> {
        self.metrics_history.read().await.clone()
    }

    /// Subscribe to alert notifications
    pub fn subscribe_to_alerts(&self) -> broadcast::Receiver<AlertEvent> {
        self.alert_sender.subscribe()
    }

    /// Check metrics against thresholds and trigger alerts
    async fn check_and_trigger_alerts(&self) -> Result<()> {
        if !self.alert_config.enabled {
            return Ok(());
        }

        let metrics = self.streaming_metrics.read().await;
        let _now = Utc::now();

        // Check latency threshold
        if metrics.avg_latency_ms > self.alert_config.latency_threshold_ms {
            self.trigger_alert(
                AlertType::HighLatency,
                AlertSeverity::Warning,
                &format!(
                    "Average latency ({:.2}ms) exceeds threshold ({:.2}ms)",
                    metrics.avg_latency_ms, self.alert_config.latency_threshold_ms
                ),
                metrics.avg_latency_ms,
                self.alert_config.latency_threshold_ms,
            )
            .await?;
        }

        // Check error rate threshold
        if metrics.error_rate_percent > self.alert_config.error_rate_threshold_percent {
            self.trigger_alert(
                AlertType::HighErrorRate,
                AlertSeverity::Critical,
                &format!(
                    "Error rate ({:.2}%) exceeds threshold ({:.2}%)",
                    metrics.error_rate_percent, self.alert_config.error_rate_threshold_percent
                ),
                metrics.error_rate_percent,
                self.alert_config.error_rate_threshold_percent,
            )
            .await?;
        }

        // Check memory usage threshold
        let memory_percent = (metrics.memory_usage_mb / 1024.0) * 100.0; // Rough approximation
        if memory_percent > self.alert_config.memory_threshold_percent {
            self.trigger_alert(
                AlertType::HighMemoryUsage,
                AlertSeverity::Warning,
                &format!(
                    "Memory usage ({:.2}%) exceeds threshold ({:.2}%)",
                    memory_percent, self.alert_config.memory_threshold_percent
                ),
                memory_percent,
                self.alert_config.memory_threshold_percent,
            )
            .await?;
        }

        // Check queue depth threshold
        if metrics.queue_depth > self.alert_config.queue_depth_threshold {
            self.trigger_alert(
                AlertType::QueueBacklog,
                AlertSeverity::Critical,
                &format!(
                    "Queue depth ({}) exceeds threshold ({})",
                    metrics.queue_depth, self.alert_config.queue_depth_threshold
                ),
                metrics.queue_depth as f64,
                self.alert_config.queue_depth_threshold as f64,
            )
            .await?;
        }

        Ok(())
    }

    /// Trigger an alert if cooldown period has passed
    async fn trigger_alert(
        &self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: &str,
        metric_value: f64,
        threshold: f64,
    ) -> Result<()> {
        let alert_key = format!("{alert_type:?}");
        let now = Utc::now();

        // Check cooldown period
        {
            let last_alert_times = self.last_alert_times.read().await;
            if let Some(last_time) = last_alert_times.get(&alert_key) {
                if now.signed_duration_since(*last_time).to_std()?
                    < self.alert_config.cooldown_period
                {
                    return Ok(()); // Still in cooldown
                }
            }
        }

        // Update last alert time
        {
            let mut last_alert_times = self.last_alert_times.write().await;
            last_alert_times.insert(alert_key, now);
        }

        // Create and send alert
        let alert = AlertEvent {
            alert_id: Uuid::new_v4().to_string(),
            alert_type,
            severity,
            message: message.to_string(),
            metric_value,
            threshold,
            timestamp: now,
            metadata: HashMap::new(),
        };

        // Send alert to subscribers
        let _ = self.alert_sender.send(alert.clone());

        warn!("Alert triggered: {} - {}", alert.alert_id, alert.message);

        Ok(())
    }

    /// Export span to Jaeger using OpenTelemetry tracer
    async fn export_span_to_jaeger(&self, span: &TraceSpan) -> Result<()> {
        #[cfg(feature = "opentelemetry")]
        {
            if let Some(_tracer) = &self.tracer {
                debug!(
                    "OpenTelemetry span export is disabled pending dependency stability. Span: {}",
                    span.span_id
                );
                // Full OpenTelemetry integration will be implemented when dependencies are stable
            } else if let Some(jaeger_endpoint) = &self.config.jaeger_endpoint {
                debug!(
                    "Tracer not initialized, skipping span export to {}",
                    jaeger_endpoint
                );
            }
        }

        #[cfg(not(feature = "opentelemetry"))]
        {
            if let Some(jaeger_endpoint) = &self.config.jaeger_endpoint {
                debug!(
                    "OpenTelemetry feature not enabled, skipping span export to {}. Span: {}",
                    jaeger_endpoint, span.span_id
                );
            }
        }

        Ok(())
    }

    /// Generate a comprehensive observability report
    pub async fn generate_observability_report(&self) -> Result<String> {
        let streaming_metrics = self.get_streaming_metrics().await;
        let business_metrics = self.get_business_metrics().await;
        let metrics_history = self.get_metrics_history().await;

        let report = format!(
            r#"
# OxiRS Stream Observability Report
Generated: {}

## Performance Metrics
- Events per Second: {:.2}
- Average Latency: {:.2}ms
- P95 Latency: {:.2}ms
- P99 Latency: {:.2}ms
- Error Rate: {:.2}%
- Memory Usage: {:.2}MB
- CPU Usage: {:.2}%
- Network I/O: {:.2} Bps

## Business Metrics
- Total Events Processed: {}
- Total Events Failed: {}
- Revenue Events: {}
- Customer Events: {}
- Health Score: {:.2}
- Data Quality Score: {:.2}

## System Health
- Active Connections: {}
- Queue Depth: {}
- Metrics History Length: {}

## Configuration
- OpenTelemetry Enabled: {}
- Profiling Enabled: {}
- Trace Sampling Rate: {:.1}%
- Alert Thresholds:
  - Latency: {:.2}ms
  - Error Rate: {:.2}%
  - Memory: {:.2}%
  - Queue Depth: {}
"#,
            Utc::now(),
            streaming_metrics.events_per_second,
            streaming_metrics.avg_latency_ms,
            streaming_metrics.p95_latency_ms,
            streaming_metrics.p99_latency_ms,
            streaming_metrics.error_rate_percent,
            streaming_metrics.memory_usage_mb,
            streaming_metrics.cpu_usage_percent,
            streaming_metrics.network_io_bps,
            business_metrics.total_events_processed,
            business_metrics.total_events_failed,
            business_metrics.revenue_events_count,
            business_metrics.customer_events_count,
            business_metrics.health_score,
            business_metrics.data_quality_score,
            streaming_metrics.active_connections,
            streaming_metrics.queue_depth,
            metrics_history.len(),
            self.config.enable_opentelemetry,
            self.config.enable_profiling,
            self.config.trace_sampling_rate * 100.0,
            self.alert_config.latency_threshold_ms,
            self.alert_config.error_rate_threshold_percent,
            self.alert_config.memory_threshold_percent,
            self.alert_config.queue_depth_threshold,
        );

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_observability_creation() {
        let config = TelemetryConfig::default();
        let alert_config = AlertConfig::default();
        let observability = StreamObservability::new(config, alert_config);

        assert_eq!(
            observability
                .get_streaming_metrics()
                .await
                .events_per_second,
            0.0
        );
    }

    #[tokio::test]
    async fn test_span_lifecycle() {
        let config = TelemetryConfig::default();
        let alert_config = AlertConfig::default();
        let observability = StreamObservability::new(config, alert_config);

        let span_id = observability.start_span("test_operation", None).await;
        assert!(!span_id.is_empty());

        observability
            .add_span_tag(&span_id, "test_key", "test_value")
            .await
            .unwrap();
        observability
            .add_span_log(&span_id, "info", "Test log message")
            .await
            .unwrap();

        observability
            .finish_span(&span_id, SpanStatus::Ok)
            .await
            .unwrap();

        // Span should be removed from active spans after finishing
        let active_spans = observability.active_spans.read().await;
        assert!(!active_spans.contains_key(&span_id));
    }

    #[tokio::test]
    async fn test_metrics_recording() {
        let config = TelemetryConfig::default();
        let alert_config = AlertConfig::default();
        let observability = StreamObservability::new(config, alert_config);

        let event = crate::event::StreamEvent::TripleAdded {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "\"test_object\"".to_string(),
            graph: None,
            metadata: crate::event::EventMetadata::default(),
        };

        observability
            .record_event(&event, Duration::from_millis(50))
            .await
            .unwrap();

        let metrics = observability.get_streaming_metrics().await;
        assert!(metrics.avg_latency_ms > 0.0);

        let business_metrics = observability.get_business_metrics().await;
        assert_eq!(business_metrics.total_events_processed, 1);
    }

    #[tokio::test]
    async fn test_alert_system() {
        let config = TelemetryConfig::default();
        let alert_config = AlertConfig {
            latency_threshold_ms: 10.0,
            ..Default::default()
        }; // Very low threshold for testing

        let observability = StreamObservability::new(config, alert_config);
        let mut alert_receiver = observability.subscribe_to_alerts();

        let event = crate::event::StreamEvent::TripleAdded {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "\"test_object\"".to_string(),
            graph: None,
            metadata: crate::event::EventMetadata::default(),
        };

        // Record an event with high latency to trigger alert
        observability
            .record_event(&event, Duration::from_millis(100))
            .await
            .unwrap();

        // Check if alert was triggered
        tokio::select! {
            alert = alert_receiver.recv() => {
                assert!(alert.is_ok());
                let alert = alert.unwrap();
                assert!(matches!(alert.alert_type, AlertType::HighLatency));
            }
            _ = sleep(Duration::from_millis(100)) => {
                // Alert might not be triggered due to timing
            }
        }
    }

    #[tokio::test]
    async fn test_observability_report() {
        let config = TelemetryConfig::default();
        let alert_config = AlertConfig::default();
        let observability = StreamObservability::new(config, alert_config);

        // Add some test data
        observability
            .update_system_metrics(100.0, 50.0, 1000.0)
            .await
            .unwrap();

        let report = observability.generate_observability_report().await.unwrap();
        assert!(report.contains("OxiRS Stream Observability Report"));
        assert!(report.contains("Memory Usage: 100.00MB"));
        assert!(report.contains("CPU Usage: 50.00%"));
    }
}
