//! Distributed Tracing with OpenTelemetry Integration
//!
//! This module provides comprehensive distributed tracing capabilities for the OxiRS cluster,
//! enabling visibility into distributed operations, performance profiling, and debugging
//! across multiple nodes.
//!
//! # Features
//!
//! - **OpenTelemetry Integration**: Industry-standard distributed tracing
//! - **OTLP Export**: Export traces to Jaeger, Zipkin, or any OTLP-compatible backend
//! - **Automatic Context Propagation**: Trace contexts propagate across service boundaries
//! - **Performance Profiling**: Detailed timing information for operations
//! - **Custom Attributes**: Rich metadata for debugging and analysis
//! - **Sampling Strategies**: Configurable sampling to control overhead
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_cluster::distributed_tracing::{TracingConfig, TracingManager};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = TracingConfig::default()
//!     .with_service_name("oxirs-cluster")
//!     .with_otlp_endpoint("http://localhost:4317");
//!
//! let tracing_manager = TracingManager::new(config).await?;
//! tracing_manager.start().await?;
//!
//! // Traces are automatically collected and exported
//! # Ok(())
//! # }
//! ```

use opentelemetry::{
    global,
    trace::{Span, SpanKind, Status, TraceContextExt, Tracer},
    Context, KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    trace::{RandomIdGenerator, Sampler, SdkTracerProvider as TracerProvider},
    Resource,
};
use opentelemetry_semantic_conventions as semconv;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Errors that can occur during distributed tracing operations
#[derive(Debug, Error)]
pub enum TracingError {
    /// OpenTelemetry initialization error
    #[error("OpenTelemetry initialization failed: {0}")]
    InitializationError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Export error
    #[error("Failed to export traces: {0}")]
    ExportError(String),

    /// Other errors
    #[error("Tracing error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, TracingError>;

/// Sampling strategy for distributed tracing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SamplingStrategy {
    /// Always sample all traces
    AlwaysOn,
    /// Never sample any traces
    AlwaysOff,
    /// Sample based on a ratio (0.0 to 1.0)
    TraceIdRatioBased(f64),
    /// Parent-based sampling with fallback
    ParentBased {
        /// Root sampler when there is no parent
        root: Box<SamplingStrategy>,
    },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::TraceIdRatioBased(0.1) // 10% sampling by default
    }
}

/// Configuration for distributed tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Service name for tracing
    pub service_name: String,

    /// Service version
    pub service_version: String,

    /// OTLP endpoint for exporting traces
    pub otlp_endpoint: Option<String>,

    /// Enable tracing
    pub enabled: bool,

    /// Sampling strategy
    pub sampling_strategy: SamplingStrategy,

    /// Batch export timeout (milliseconds)
    pub batch_timeout_ms: u64,

    /// Maximum batch size
    pub max_batch_size: usize,

    /// Export timeout (milliseconds)
    pub export_timeout_ms: u64,

    /// Enable console output
    pub console_output: bool,

    /// Enable JSON formatting
    pub json_format: bool,

    /// Environment filter (e.g., "info,oxirs_cluster=debug")
    pub env_filter: String,

    /// Custom resource attributes
    pub resource_attributes: Vec<(String, String)>,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "oxirs-cluster".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            otlp_endpoint: Some("http://localhost:4317".to_string()),
            enabled: true,
            sampling_strategy: SamplingStrategy::default(),
            batch_timeout_ms: 5000,
            max_batch_size: 512,
            export_timeout_ms: 30000,
            console_output: true,
            json_format: false,
            env_filter: "info".to_string(),
            resource_attributes: Vec::new(),
        }
    }
}

impl TracingConfig {
    /// Set the service name
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Set the OTLP endpoint
    pub fn with_otlp_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.otlp_endpoint = Some(endpoint.into());
        self
    }

    /// Set the sampling strategy
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Enable or disable tracing
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Add a custom resource attribute
    pub fn add_resource_attribute(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.resource_attributes.push((key.into(), value.into()));
        self
    }
}

/// Distributed tracing manager
pub struct TracingManager {
    config: TracingConfig,
    tracer_provider: Arc<RwLock<Option<TracerProvider>>>,
    initialized: Arc<RwLock<bool>>,
}

impl TracingManager {
    /// Create a new tracing manager with the given configuration
    pub async fn new(config: TracingConfig) -> Result<Self> {
        if !config.enabled {
            return Ok(Self {
                config,
                tracer_provider: Arc::new(RwLock::new(None)),
                initialized: Arc::new(RwLock::new(false)),
            });
        }

        Ok(Self {
            config,
            tracer_provider: Arc::new(RwLock::new(None)),
            initialized: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the tracing manager and initialize OpenTelemetry
    #[allow(deprecated)] // Temporarily using deprecated API until version compatibility is resolved
    #[allow(unused_variables)] // batch_config will be used when OTLP export is re-enabled
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            tracing::info!("Distributed tracing is disabled");
            return Ok(());
        }

        let mut initialized = self.initialized.write().await;
        if *initialized {
            return Ok(());
        }

        // Build resource with all attributes
        let mut all_attrs = vec![KeyValue::new(
            semconv::resource::SERVICE_VERSION,
            self.config.service_version.clone(),
        )];

        // Add custom resource attributes
        for (key, value) in &self.config.resource_attributes {
            all_attrs.push(KeyValue::new(key.clone(), value.clone()));
        }

        let resource = Resource::builder_empty()
            .with_service_name(self.config.service_name.clone())
            .with_attributes(all_attrs)
            .build();

        // Create sampler based on strategy
        let sampler = self.create_sampler(&self.config.sampling_strategy);

        // Create tracer provider
        let provider = if let Some(otlp_endpoint) = &self.config.otlp_endpoint {
            // Create OTLP exporter using HTTP (tonic support removed in 0.31)
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_http()
                .with_endpoint(otlp_endpoint.clone())
                .with_timeout(Duration::from_millis(self.config.export_timeout_ms))
                .build()
                .map_err(|e| {
                    TracingError::InitializationError(format!(
                        "Failed to create OTLP exporter: {e}"
                    ))
                })?;

            TracerProvider::builder()
                .with_batch_exporter(exporter)
                .with_resource(resource)
                .with_sampler(sampler)
                .with_id_generator(RandomIdGenerator::default())
                .build()
        } else {
            // No OTLP endpoint, use simple provider
            TracerProvider::builder()
                .with_resource(resource)
                .with_sampler(sampler)
                .with_id_generator(RandomIdGenerator::default())
                .build()
        };

        // Set global tracer provider
        global::set_tracer_provider(provider.clone());

        // Store provider
        let mut tracer_provider = self.tracer_provider.write().await;
        *tracer_provider = Some(provider);

        // Initialize tracing subscriber
        self.init_tracing_subscriber()?;

        *initialized = true;

        tracing::info!(
            service_name = %self.config.service_name,
            otlp_endpoint = ?self.config.otlp_endpoint,
            "Distributed tracing initialized"
        );

        Ok(())
    }

    /// Stop the tracing manager and flush pending traces
    pub async fn stop(&self) -> Result<()> {
        if !*self.initialized.read().await {
            return Ok(());
        }

        tracing::info!("Shutting down distributed tracing");

        // Shutdown tracer provider
        let mut provider = self.tracer_provider.write().await;
        if let Some(provider) = provider.take() {
            if let Err(e) = provider.shutdown() {
                tracing::error!("Failed to shutdown tracer provider: {}", e);
            }
        }

        // Note: In OpenTelemetry 0.31, global shutdown is handled automatically
        // when the provider is dropped or explicitly shut down

        let mut initialized = self.initialized.write().await;
        *initialized = false;

        Ok(())
    }

    /// Check if tracing is initialized
    pub async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }

    /// Get the current configuration
    pub fn config(&self) -> &TracingConfig {
        &self.config
    }

    /// Create a sampler based on the strategy
    fn create_sampler(&self, strategy: &SamplingStrategy) -> Sampler {
        match strategy {
            SamplingStrategy::AlwaysOn => Sampler::AlwaysOn,
            SamplingStrategy::AlwaysOff => Sampler::AlwaysOff,
            SamplingStrategy::TraceIdRatioBased(ratio) => Sampler::TraceIdRatioBased(*ratio),
            SamplingStrategy::ParentBased { root } => {
                let root_sampler = self.create_sampler(root);
                Sampler::ParentBased(Box::new(root_sampler))
            }
        }
    }

    /// Initialize tracing subscriber
    fn init_tracing_subscriber(&self) -> Result<()> {
        // Create env filter
        let env_filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(&self.config.env_filter));

        // Build and initialize subscriber
        // Note: OpenTelemetry layer integration is temporarily disabled due to version compatibility issues
        // TODO: Re-enable when tracing-opentelemetry and opentelemetry versions are aligned
        if self.config.console_output {
            if self.config.json_format {
                let json_layer = tracing_subscriber::fmt::layer().json();
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(json_layer)
                    .try_init()
                    .map_err(|e| {
                        TracingError::InitializationError(format!(
                            "Failed to initialize subscriber: {e}"
                        ))
                    })?;
            } else {
                let fmt_layer = tracing_subscriber::fmt::layer();
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt_layer)
                    .try_init()
                    .map_err(|e| {
                        TracingError::InitializationError(format!(
                            "Failed to initialize subscriber: {e}"
                        ))
                    })?;
            }
        } else {
            tracing_subscriber::registry()
                .with(env_filter)
                .try_init()
                .map_err(|e| {
                    TracingError::InitializationError(format!(
                        "Failed to initialize subscriber: {e}"
                    ))
                })?;
        }

        tracing::warn!(
            "OpenTelemetry integration temporarily disabled due to version compatibility"
        );

        Ok(())
    }
}

/// Helper functions for creating spans with common attributes
pub mod span_helpers {
    use super::*;

    /// Create a span for a consensus operation
    pub fn consensus_span(operation: &str, node_id: u64, term: u64) -> impl Span {
        let tracer = global::tracer("oxirs-cluster");
        let mut span = tracer
            .span_builder(format!("consensus.{}", operation))
            .with_kind(SpanKind::Internal)
            .start(&tracer);

        span.set_attribute(KeyValue::new("node.id", node_id as i64));
        span.set_attribute(KeyValue::new("consensus.term", term as i64));
        span.set_attribute(KeyValue::new("operation.type", "consensus"));

        span
    }

    /// Create a span for a replication operation
    pub fn replication_span(operation: &str, source_node: u64, target_node: u64) -> impl Span {
        let tracer = global::tracer("oxirs-cluster");
        let mut span = tracer
            .span_builder(format!("replication.{}", operation))
            .with_kind(SpanKind::Internal)
            .start(&tracer);

        span.set_attribute(KeyValue::new("source.node.id", source_node as i64));
        span.set_attribute(KeyValue::new("target.node.id", target_node as i64));
        span.set_attribute(KeyValue::new("operation.type", "replication"));

        span
    }

    /// Create a span for a query operation
    pub fn query_span(query_type: &str, node_id: u64) -> impl Span {
        let tracer = global::tracer("oxirs-cluster");
        let mut span = tracer
            .span_builder(format!("query.{}", query_type))
            .with_kind(SpanKind::Server)
            .start(&tracer);

        span.set_attribute(KeyValue::new("node.id", node_id as i64));
        span.set_attribute(KeyValue::new("operation.type", "query"));

        span
    }

    /// Create a span for a storage operation
    pub fn storage_span(operation: &str, node_id: u64) -> impl Span {
        let tracer = global::tracer("oxirs-cluster");
        let mut span = tracer
            .span_builder(format!("storage.{}", operation))
            .with_kind(SpanKind::Internal)
            .start(&tracer);

        span.set_attribute(KeyValue::new("node.id", node_id as i64));
        span.set_attribute(KeyValue::new("operation.type", "storage"));

        span
    }

    /// Mark a span as successful
    pub fn mark_success(span: &mut impl Span) {
        span.set_status(Status::Ok);
    }

    /// Mark a span as failed with an error message
    pub fn mark_error(span: &mut impl Span, error: &str) {
        span.set_status(Status::error(error.to_string()));
        span.set_attribute(KeyValue::new("error", true));
        span.set_attribute(KeyValue::new("error.message", error.to_string()));
    }
}

/// Tracing context for distributed operations
#[derive(Debug, Clone)]
pub struct TracingContext {
    /// Trace ID
    pub trace_id: String,
    /// Span ID
    pub span_id: String,
    /// Parent span ID (if any)
    pub parent_span_id: Option<String>,
    /// Trace flags
    pub trace_flags: u8,
}

impl TracingContext {
    /// Extract tracing context from the current OpenTelemetry context
    pub fn current() -> Option<Self> {
        let context = Context::current();
        let span = context.span();
        let span_context = span.span_context();

        if span_context.is_valid() {
            Some(Self {
                trace_id: span_context.trace_id().to_string(),
                span_id: span_context.span_id().to_string(),
                parent_span_id: None,
                trace_flags: span_context.trace_flags().to_u8(),
            })
        } else {
            None
        }
    }

    /// Serialize to headers for propagation
    pub fn to_headers(&self) -> Vec<(String, String)> {
        vec![
            ("x-trace-id".to_string(), self.trace_id.clone()),
            ("x-span-id".to_string(), self.span_id.clone()),
            ("x-trace-flags".to_string(), self.trace_flags.to_string()),
        ]
    }
}

/// Statistics about distributed tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingStatistics {
    /// Total number of spans created
    pub total_spans_created: u64,
    /// Total number of spans exported
    pub total_spans_exported: u64,
    /// Total number of export errors
    pub total_export_errors: u64,
    /// Current sampling rate
    pub current_sampling_rate: f64,
    /// Average span duration (milliseconds)
    pub avg_span_duration_ms: f64,
    /// Is tracing active
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_config_default() {
        let config = TracingConfig::default();
        assert_eq!(config.service_name, "oxirs-cluster");
        assert!(config.enabled);
        assert_eq!(
            config.sampling_strategy,
            SamplingStrategy::TraceIdRatioBased(0.1)
        );
    }

    #[test]
    fn test_tracing_config_builder() {
        let config = TracingConfig::default()
            .with_service_name("test-service")
            .with_otlp_endpoint("http://test:4317")
            .with_sampling_strategy(SamplingStrategy::AlwaysOn)
            .with_enabled(false);

        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.otlp_endpoint, Some("http://test:4317".to_string()));
        assert_eq!(config.sampling_strategy, SamplingStrategy::AlwaysOn);
        assert!(!config.enabled);
    }

    #[test]
    fn test_sampling_strategies() {
        let always_on = SamplingStrategy::AlwaysOn;
        let always_off = SamplingStrategy::AlwaysOff;
        let ratio = SamplingStrategy::TraceIdRatioBased(0.5);

        assert!(matches!(always_on, SamplingStrategy::AlwaysOn));
        assert!(matches!(always_off, SamplingStrategy::AlwaysOff));
        if let SamplingStrategy::TraceIdRatioBased(r) = ratio {
            assert_eq!(r, 0.5);
        }
    }

    #[tokio::test]
    async fn test_tracing_manager_creation() {
        let config = TracingConfig::default().with_enabled(false);
        let manager = TracingManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_tracing_manager_disabled() {
        let config = TracingConfig::default().with_enabled(false);
        let manager = TracingManager::new(config).await.unwrap();

        let result = manager.start().await;
        assert!(result.is_ok());
        assert!(!manager.is_initialized().await);
    }
}
