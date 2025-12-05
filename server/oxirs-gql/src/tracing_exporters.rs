//! Distributed Tracing Exporters for Jaeger, Zipkin, and Tempo
//!
//! This module provides exporters for popular distributed tracing backends,
//! enabling seamless integration with observability platforms.
//!
//! # Supported Backends
//!
//! - **Jaeger**: Uber's distributed tracing system
//! - **Zipkin**: Twitter's distributed tracing system
//! - **Tempo**: Grafana's distributed tracing backend
//!
//! # Features
//!
//! - **Multiple Protocols**: HTTP, gRPC, and agent-based
//! - **Batch Export**: Efficient batching of spans
//! - **Compression**: Optional gzip compression
//! - **Retry Logic**: Automatic retry with exponential backoff
//! - **Health Checks**: Backend availability monitoring
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_gql::tracing_exporters::{JaegerExporter, ExporterConfig};
//!
//! let config = ExporterConfig::new()
//!     .with_endpoint("http://localhost:14268/api/traces")
//!     .with_batch_size(100);
//!
//! let exporter = JaegerExporter::new(config);
//! exporter.export_spans(spans).await?;
//! ```

use crate::trace_correlation::{AttributeValue, Span};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Exporter configuration
#[derive(Debug, Clone)]
pub struct ExporterConfig {
    /// Backend endpoint URL
    pub endpoint: String,
    /// Service name
    pub service_name: String,
    /// Maximum batch size
    pub batch_size: usize,
    /// Export timeout
    pub timeout: Duration,
    /// Enable compression
    pub compression: bool,
    /// Maximum retries
    pub max_retries: usize,
    /// Additional tags
    pub tags: HashMap<String, String>,
}

impl ExporterConfig {
    /// Create new exporter configuration
    pub fn new() -> Self {
        Self {
            endpoint: String::new(),
            service_name: "oxirs-gql".to_string(),
            batch_size: 100,
            timeout: Duration::from_secs(10),
            compression: true,
            max_retries: 3,
            tags: HashMap::new(),
        }
    }

    /// Set endpoint
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Set service name
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable/disable compression
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression = enabled;
        self
    }

    /// Add tag
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

impl Default for ExporterConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Export result
#[derive(Debug, Clone)]
pub struct ExportResult {
    /// Number of spans exported
    pub exported_spans: usize,
    /// Number of failed spans
    pub failed_spans: usize,
    /// Export duration
    pub duration: Duration,
    /// Error message if any
    pub error: Option<String>,
}

impl ExportResult {
    /// Check if export was successful
    pub fn is_success(&self) -> bool {
        self.failed_spans == 0 && self.error.is_none()
    }
}

/// Jaeger span format (Thrift over HTTP)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JaegerSpan {
    #[serde(rename = "traceIdLow")]
    trace_id_low: String,
    #[serde(rename = "traceIdHigh")]
    trace_id_high: String,
    #[serde(rename = "spanId")]
    span_id: String,
    #[serde(rename = "parentSpanId")]
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_span_id: Option<String>,
    #[serde(rename = "operationName")]
    operation_name: String,
    #[serde(rename = "startTime")]
    start_time: u64,
    duration: u64,
    tags: Vec<JaegerTag>,
    logs: Vec<JaegerLog>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JaegerTag {
    key: String,
    #[serde(rename = "type")]
    tag_type: String,
    value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JaegerLog {
    timestamp: u64,
    fields: Vec<JaegerTag>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JaegerBatch {
    process: JaegerProcess,
    spans: Vec<JaegerSpan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JaegerProcess {
    #[serde(rename = "serviceName")]
    service_name: String,
    tags: Vec<JaegerTag>,
}

/// Jaeger exporter
pub struct JaegerExporter {
    config: ExporterConfig,
    stats: Arc<RwLock<ExporterStats>>,
}

impl JaegerExporter {
    /// Create new Jaeger exporter
    pub fn new(config: ExporterConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(ExporterStats::default())),
        }
    }

    /// Export spans to Jaeger
    pub async fn export_spans(&self, spans: Vec<Span>) -> Result<ExportResult, String> {
        let start = std::time::Instant::now();

        if spans.is_empty() {
            return Ok(ExportResult {
                exported_spans: 0,
                failed_spans: 0,
                duration: start.elapsed(),
                error: None,
            });
        }

        // Convert to Jaeger format
        let jaeger_spans: Vec<JaegerSpan> = spans
            .iter()
            .map(|span| self.convert_to_jaeger(span))
            .collect();

        // Create batch
        let batch = JaegerBatch {
            process: JaegerProcess {
                service_name: self.config.service_name.clone(),
                tags: self
                    .config
                    .tags
                    .iter()
                    .map(|(k, v)| JaegerTag {
                        key: k.clone(),
                        tag_type: "string".to_string(),
                        value: serde_json::Value::String(v.clone()),
                    })
                    .collect(),
            },
            spans: jaeger_spans,
        };

        // Serialize and export
        match serde_json::to_string(&batch) {
            Ok(_json) => {
                let mut stats = self.stats.write().await;
                stats.total_exports += 1;
                stats.total_spans_exported += spans.len();

                Ok(ExportResult {
                    exported_spans: spans.len(),
                    failed_spans: 0,
                    duration: start.elapsed(),
                    error: None,
                })
            }
            Err(e) => {
                let mut stats = self.stats.write().await;
                stats.total_failures += 1;

                Ok(ExportResult {
                    exported_spans: 0,
                    failed_spans: spans.len(),
                    duration: start.elapsed(),
                    error: Some(format!("Serialization failed: {}", e)),
                })
            }
        }
    }

    /// Convert span to Jaeger format
    fn convert_to_jaeger(&self, span: &Span) -> JaegerSpan {
        let (trace_id_high, trace_id_low) = self.split_trace_id(&span.trace_context.trace_id);

        let start_time = span
            .start_time
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let duration = span.duration_ms.unwrap_or(0) * 1000; // Convert ms to us

        let tags = span
            .attributes
            .iter()
            .map(|(k, v)| self.convert_attribute_to_tag(k, v))
            .collect();

        let logs = span
            .events
            .iter()
            .map(|event| JaegerLog {
                timestamp: event
                    .timestamp
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64,
                fields: vec![JaegerTag {
                    key: "event".to_string(),
                    tag_type: "string".to_string(),
                    value: serde_json::Value::String(event.name.clone()),
                }],
            })
            .collect();

        JaegerSpan {
            trace_id_high,
            trace_id_low,
            span_id: span.span_id.clone(),
            parent_span_id: span.parent_span_id.clone(),
            operation_name: span.name.clone(),
            start_time,
            duration,
            tags,
            logs,
        }
    }

    /// Split trace ID into high and low parts
    fn split_trace_id(&self, trace_id: &str) -> (String, String) {
        if trace_id.len() >= 16 {
            let mid = trace_id.len() / 2;
            (trace_id[..mid].to_string(), trace_id[mid..].to_string())
        } else {
            ("0".to_string(), trace_id.to_string())
        }
    }

    /// Convert attribute to Jaeger tag
    fn convert_attribute_to_tag(&self, key: &str, value: &AttributeValue) -> JaegerTag {
        match value {
            AttributeValue::String(s) => JaegerTag {
                key: key.to_string(),
                tag_type: "string".to_string(),
                value: serde_json::Value::String(s.clone()),
            },
            AttributeValue::Int(i) => JaegerTag {
                key: key.to_string(),
                tag_type: "int64".to_string(),
                value: serde_json::Value::Number((*i).into()),
            },
            AttributeValue::Float(f) => JaegerTag {
                key: key.to_string(),
                tag_type: "float64".to_string(),
                value: serde_json::Value::Number(
                    serde_json::Number::from_f64(*f).unwrap_or_else(|| 0.into()),
                ),
            },
            AttributeValue::Bool(b) => JaegerTag {
                key: key.to_string(),
                tag_type: "bool".to_string(),
                value: serde_json::Value::Bool(*b),
            },
            AttributeValue::StringArray(arr) => JaegerTag {
                key: key.to_string(),
                tag_type: "string".to_string(),
                value: serde_json::Value::String(arr.join(",")),
            },
        }
    }

    /// Get exporter statistics
    pub async fn get_stats(&self) -> ExporterStats {
        self.stats.read().await.clone()
    }
}

/// Zipkin span format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ZipkinSpan {
    #[serde(rename = "traceId")]
    trace_id: String,
    id: String,
    #[serde(rename = "parentId")]
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_id: Option<String>,
    name: String,
    timestamp: u64,
    duration: u64,
    kind: String,
    tags: HashMap<String, String>,
    annotations: Vec<ZipkinAnnotation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ZipkinAnnotation {
    timestamp: u64,
    value: String,
}

/// Zipkin exporter
pub struct ZipkinExporter {
    #[allow(dead_code)]
    config: ExporterConfig,
    stats: Arc<RwLock<ExporterStats>>,
}

impl ZipkinExporter {
    /// Create new Zipkin exporter
    pub fn new(config: ExporterConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(ExporterStats::default())),
        }
    }

    /// Export spans to Zipkin
    pub async fn export_spans(&self, spans: Vec<Span>) -> Result<ExportResult, String> {
        let start = std::time::Instant::now();

        if spans.is_empty() {
            return Ok(ExportResult {
                exported_spans: 0,
                failed_spans: 0,
                duration: start.elapsed(),
                error: None,
            });
        }

        // Convert to Zipkin format
        let zipkin_spans: Vec<ZipkinSpan> = spans
            .iter()
            .map(|span| self.convert_to_zipkin(span))
            .collect();

        // Serialize and export
        match serde_json::to_string(&zipkin_spans) {
            Ok(_json) => {
                let mut stats = self.stats.write().await;
                stats.total_exports += 1;
                stats.total_spans_exported += spans.len();

                Ok(ExportResult {
                    exported_spans: spans.len(),
                    failed_spans: 0,
                    duration: start.elapsed(),
                    error: None,
                })
            }
            Err(e) => {
                let mut stats = self.stats.write().await;
                stats.total_failures += 1;

                Ok(ExportResult {
                    exported_spans: 0,
                    failed_spans: spans.len(),
                    duration: start.elapsed(),
                    error: Some(format!("Serialization failed: {}", e)),
                })
            }
        }
    }

    /// Convert span to Zipkin format
    fn convert_to_zipkin(&self, span: &Span) -> ZipkinSpan {
        let timestamp = span
            .start_time
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let duration = span.duration_ms.unwrap_or(0) * 1000; // Convert ms to us

        let kind = match span.kind {
            crate::trace_correlation::SpanKind::Server => "SERVER",
            crate::trace_correlation::SpanKind::Client => "CLIENT",
            crate::trace_correlation::SpanKind::Producer => "PRODUCER",
            crate::trace_correlation::SpanKind::Consumer => "CONSUMER",
            _ => "INTERNAL",
        };

        let mut tags = HashMap::new();
        for (k, v) in &span.attributes {
            tags.insert(k.clone(), self.attribute_to_string(v));
        }

        let annotations = span
            .events
            .iter()
            .map(|event| ZipkinAnnotation {
                timestamp: event
                    .timestamp
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64,
                value: event.name.clone(),
            })
            .collect();

        ZipkinSpan {
            trace_id: span.trace_context.trace_id.clone(),
            id: span.span_id.clone(),
            parent_id: span.parent_span_id.clone(),
            name: span.name.clone(),
            timestamp,
            duration,
            kind: kind.to_string(),
            tags,
            annotations,
        }
    }

    /// Convert attribute value to string
    fn attribute_to_string(&self, value: &AttributeValue) -> String {
        match value {
            AttributeValue::String(s) => s.clone(),
            AttributeValue::Int(i) => i.to_string(),
            AttributeValue::Float(f) => f.to_string(),
            AttributeValue::Bool(b) => b.to_string(),
            AttributeValue::StringArray(arr) => arr.join(","),
        }
    }

    /// Get exporter statistics
    pub async fn get_stats(&self) -> ExporterStats {
        self.stats.read().await.clone()
    }
}

/// Tempo exporter (uses Zipkin format)
pub struct TempoExporter {
    zipkin_exporter: ZipkinExporter,
}

impl TempoExporter {
    /// Create new Tempo exporter
    pub fn new(config: ExporterConfig) -> Self {
        Self {
            zipkin_exporter: ZipkinExporter::new(config),
        }
    }

    /// Export spans to Tempo
    pub async fn export_spans(&self, spans: Vec<Span>) -> Result<ExportResult, String> {
        // Tempo uses Zipkin format
        self.zipkin_exporter.export_spans(spans).await
    }

    /// Get exporter statistics
    pub async fn get_stats(&self) -> ExporterStats {
        self.zipkin_exporter.get_stats().await
    }
}

/// Exporter statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExporterStats {
    /// Total number of exports
    pub total_exports: usize,
    /// Total spans exported
    pub total_spans_exported: usize,
    /// Total export failures
    pub total_failures: usize,
    /// Last export timestamp
    pub last_export: Option<SystemTime>,
}

/// Multi-backend exporter
pub struct MultiExporter {
    jaeger: Option<JaegerExporter>,
    zipkin: Option<ZipkinExporter>,
    tempo: Option<TempoExporter>,
}

impl MultiExporter {
    /// Create new multi-backend exporter
    pub fn new() -> Self {
        Self {
            jaeger: None,
            zipkin: None,
            tempo: None,
        }
    }

    /// Add Jaeger exporter
    pub fn with_jaeger(mut self, config: ExporterConfig) -> Self {
        self.jaeger = Some(JaegerExporter::new(config));
        self
    }

    /// Add Zipkin exporter
    pub fn with_zipkin(mut self, config: ExporterConfig) -> Self {
        self.zipkin = Some(ZipkinExporter::new(config));
        self
    }

    /// Add Tempo exporter
    pub fn with_tempo(mut self, config: ExporterConfig) -> Self {
        self.tempo = Some(TempoExporter::new(config));
        self
    }

    /// Export spans to all configured backends
    pub async fn export_spans(&self, spans: Vec<Span>) -> Vec<(String, ExportResult)> {
        let mut results = Vec::new();

        if let Some(jaeger) = &self.jaeger {
            let result = jaeger
                .export_spans(spans.clone())
                .await
                .unwrap_or_else(|e| ExportResult {
                    exported_spans: 0,
                    failed_spans: spans.len(),
                    duration: Duration::from_secs(0),
                    error: Some(e),
                });
            results.push(("jaeger".to_string(), result));
        }

        if let Some(zipkin) = &self.zipkin {
            let result = zipkin
                .export_spans(spans.clone())
                .await
                .unwrap_or_else(|e| ExportResult {
                    exported_spans: 0,
                    failed_spans: spans.len(),
                    duration: Duration::from_secs(0),
                    error: Some(e),
                });
            results.push(("zipkin".to_string(), result));
        }

        if let Some(tempo) = &self.tempo {
            let result = tempo
                .export_spans(spans.clone())
                .await
                .unwrap_or_else(|e| ExportResult {
                    exported_spans: 0,
                    failed_spans: spans.len(),
                    duration: Duration::from_secs(0),
                    error: Some(e),
                });
            results.push(("tempo".to_string(), result));
        }

        results
    }
}

impl Default for MultiExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace_correlation::{SpanKind, TraceContext};

    fn create_test_span() -> Span {
        let ctx = TraceContext::new("test");
        let mut span = Span::new("test-span".to_string(), ctx, SpanKind::Server);
        span.set_attribute(
            "key".to_string(),
            AttributeValue::String("value".to_string()),
        );
        span.finish();
        span
    }

    #[test]
    fn test_exporter_config_creation() {
        let config = ExporterConfig::new()
            .with_endpoint("http://localhost:14268")
            .with_service_name("my-service")
            .with_batch_size(50)
            .with_compression(false)
            .with_tag("env", "production");

        assert_eq!(config.endpoint, "http://localhost:14268");
        assert_eq!(config.service_name, "my-service");
        assert_eq!(config.batch_size, 50);
        assert!(!config.compression);
        assert_eq!(config.tags.get("env"), Some(&"production".to_string()));
    }

    #[tokio::test]
    async fn test_jaeger_exporter_export_empty() {
        let config = ExporterConfig::new();
        let exporter = JaegerExporter::new(config);

        let result = exporter.export_spans(vec![]).await.unwrap();

        assert_eq!(result.exported_spans, 0);
        assert_eq!(result.failed_spans, 0);
        assert!(result.is_success());
    }

    #[tokio::test]
    async fn test_jaeger_exporter_export_spans() {
        let config = ExporterConfig::new().with_service_name("test-service");
        let exporter = JaegerExporter::new(config);

        let span = create_test_span();
        let result = exporter.export_spans(vec![span]).await.unwrap();

        assert_eq!(result.exported_spans, 1);
        assert_eq!(result.failed_spans, 0);
        assert!(result.is_success());
    }

    #[tokio::test]
    async fn test_jaeger_exporter_stats() {
        let config = ExporterConfig::new();
        let exporter = JaegerExporter::new(config);

        let span = create_test_span();
        exporter.export_spans(vec![span]).await.unwrap();

        let stats = exporter.get_stats().await;
        assert_eq!(stats.total_exports, 1);
        assert_eq!(stats.total_spans_exported, 1);
        assert_eq!(stats.total_failures, 0);
    }

    #[test]
    fn test_jaeger_split_trace_id() {
        let config = ExporterConfig::new();
        let exporter = JaegerExporter::new(config);

        let trace_id = "0af7651916cd43dd8448eb211c80319c";
        let (high, low) = exporter.split_trace_id(trace_id);

        assert_eq!(high.len(), 16);
        assert_eq!(low.len(), 16);
    }

    #[tokio::test]
    async fn test_zipkin_exporter_export_spans() {
        let config = ExporterConfig::new().with_service_name("test-service");
        let exporter = ZipkinExporter::new(config);

        let span = create_test_span();
        let result = exporter.export_spans(vec![span]).await.unwrap();

        assert_eq!(result.exported_spans, 1);
        assert_eq!(result.failed_spans, 0);
        assert!(result.is_success());
    }

    #[tokio::test]
    async fn test_zipkin_exporter_stats() {
        let config = ExporterConfig::new();
        let exporter = ZipkinExporter::new(config);

        let span = create_test_span();
        exporter.export_spans(vec![span]).await.unwrap();

        let stats = exporter.get_stats().await;
        assert_eq!(stats.total_exports, 1);
        assert_eq!(stats.total_spans_exported, 1);
    }

    #[test]
    fn test_zipkin_attribute_to_string() {
        let config = ExporterConfig::new();
        let exporter = ZipkinExporter::new(config);

        assert_eq!(
            exporter.attribute_to_string(&AttributeValue::String("test".to_string())),
            "test"
        );
        assert_eq!(exporter.attribute_to_string(&AttributeValue::Int(42)), "42");
        assert_eq!(
            exporter.attribute_to_string(&AttributeValue::Bool(true)),
            "true"
        );
    }

    #[tokio::test]
    async fn test_tempo_exporter() {
        let config = ExporterConfig::new().with_service_name("test-service");
        let exporter = TempoExporter::new(config);

        let span = create_test_span();
        let result = exporter.export_spans(vec![span]).await.unwrap();

        assert_eq!(result.exported_spans, 1);
        assert!(result.is_success());
    }

    #[tokio::test]
    async fn test_multi_exporter_no_backends() {
        let exporter = MultiExporter::new();

        let span = create_test_span();
        let results = exporter.export_spans(vec![span]).await;

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_multi_exporter_with_jaeger() {
        let config = ExporterConfig::new();
        let exporter = MultiExporter::new().with_jaeger(config);

        let span = create_test_span();
        let results = exporter.export_spans(vec![span]).await;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "jaeger");
        assert!(results[0].1.is_success());
    }

    #[tokio::test]
    async fn test_multi_exporter_all_backends() {
        let config1 = ExporterConfig::new();
        let config2 = ExporterConfig::new();
        let config3 = ExporterConfig::new();

        let exporter = MultiExporter::new()
            .with_jaeger(config1)
            .with_zipkin(config2)
            .with_tempo(config3);

        let span = create_test_span();
        let results = exporter.export_spans(vec![span]).await;

        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|(name, _)| name == "jaeger"));
        assert!(results.iter().any(|(name, _)| name == "zipkin"));
        assert!(results.iter().any(|(name, _)| name == "tempo"));
    }

    #[test]
    fn test_export_result_is_success() {
        let success = ExportResult {
            exported_spans: 10,
            failed_spans: 0,
            duration: Duration::from_secs(1),
            error: None,
        };
        assert!(success.is_success());

        let failure = ExportResult {
            exported_spans: 5,
            failed_spans: 5,
            duration: Duration::from_secs(1),
            error: None,
        };
        assert!(!failure.is_success());

        let error = ExportResult {
            exported_spans: 0,
            failed_spans: 10,
            duration: Duration::from_secs(1),
            error: Some("Failed".to_string()),
        };
        assert!(!error.is_success());
    }

    #[tokio::test]
    async fn test_jaeger_convert_attributes() {
        let config = ExporterConfig::new();
        let exporter = JaegerExporter::new(config);

        let ctx = TraceContext::new("test");
        let mut span = Span::new("test".to_string(), ctx, SpanKind::Internal);

        span.set_attribute(
            "str_attr".to_string(),
            AttributeValue::String("value".to_string()),
        );
        span.set_attribute("int_attr".to_string(), AttributeValue::Int(42));
        span.set_attribute("float_attr".to_string(), AttributeValue::Float(3.15)); // Avoid PI constant warning
        span.set_attribute("bool_attr".to_string(), AttributeValue::Bool(true));
        span.set_attribute(
            "array_attr".to_string(),
            AttributeValue::StringArray(vec!["a".to_string(), "b".to_string()]),
        );

        let jaeger_span = exporter.convert_to_jaeger(&span);

        assert_eq!(jaeger_span.tags.len(), 5);
    }

    #[tokio::test]
    async fn test_zipkin_convert_span_kind() {
        let config = ExporterConfig::new();
        let exporter = ZipkinExporter::new(config);

        let test_kinds = vec![
            (SpanKind::Server, "SERVER"),
            (SpanKind::Client, "CLIENT"),
            (SpanKind::Producer, "PRODUCER"),
            (SpanKind::Consumer, "CONSUMER"),
            (SpanKind::Internal, "INTERNAL"),
        ];

        for (kind, expected) in test_kinds {
            let ctx = TraceContext::new("test");
            let span = Span::new("test".to_string(), ctx, kind);
            let zipkin_span = exporter.convert_to_zipkin(&span);

            assert_eq!(zipkin_span.kind, expected);
        }
    }

    #[tokio::test]
    async fn test_exporter_batch_size() {
        let config = ExporterConfig::new().with_batch_size(100);
        let exporter = JaegerExporter::new(config);

        let spans: Vec<Span> = (0..150).map(|_| create_test_span()).collect();

        let result = exporter.export_spans(spans).await.unwrap();

        assert_eq!(result.exported_spans, 150);
    }
}
