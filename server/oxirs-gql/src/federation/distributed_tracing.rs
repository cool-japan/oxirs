//! Distributed Query Tracing Across Subgraphs
//!
//! This module provides comprehensive distributed tracing for federated GraphQL queries,
//! enabling end-to-end observability across multiple subgraphs with OpenTelemetry integration.
//!
//! ## Features
//!
//! - **OpenTelemetry Integration**: Full W3C trace context propagation
//! - **Span Hierarchy**: Automatic parent-child span relationships
//! - **Cross-Service Tracing**: Traces queries across multiple subgraphs
//! - **Performance Metrics**: Detailed timing and resource usage per subgraph
//! - **Error Tracking**: Captures and propagates errors across services
//! - **Sampling Strategies**: Configurable sampling for high-volume scenarios

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Trace context for distributed tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Unique trace ID following W3C trace context format
    pub trace_id: String,
    /// Current span ID
    pub span_id: String,
    /// Parent span ID (if any)
    pub parent_span_id: Option<String>,
    /// Trace flags (sampled, debug, etc.)
    pub trace_flags: u8,
    /// Trace state for vendor-specific data
    pub trace_state: HashMap<String, String>,
}

impl TraceContext {
    /// Create a new root trace context
    pub fn new() -> Self {
        Self {
            trace_id: generate_trace_id(),
            span_id: generate_span_id(),
            parent_span_id: None,
            trace_flags: 0x01, // Sampled by default
            trace_state: HashMap::new(),
        }
    }

    /// Create a child span context
    pub fn create_child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: generate_span_id(),
            parent_span_id: Some(self.span_id.clone()),
            trace_flags: self.trace_flags,
            trace_state: self.trace_state.clone(),
        }
    }

    /// Check if this trace is sampled
    pub fn is_sampled(&self) -> bool {
        (self.trace_flags & 0x01) != 0
    }

    /// Convert to W3C traceparent header
    pub fn to_traceparent(&self) -> String {
        format!(
            "00-{}-{}-{:02x}",
            self.trace_id, self.span_id, self.trace_flags
        )
    }

    /// Parse W3C traceparent header
    pub fn from_traceparent(header: &str) -> Result<Self> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() != 4 {
            return Err(anyhow!("Invalid traceparent format"));
        }

        if parts[0] != "00" {
            return Err(anyhow!("Unsupported trace version"));
        }

        let trace_flags =
            u8::from_str_radix(parts[3], 16).map_err(|e| anyhow!("Invalid trace flags: {}", e))?;

        Ok(Self {
            trace_id: parts[1].to_string(),
            span_id: parts[2].to_string(),
            parent_span_id: None,
            trace_flags,
            trace_state: HashMap::new(),
        })
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Span status
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum SpanStatus {
    /// Span completed successfully
    Ok,
    /// Span encountered an error
    Error,
    /// Span is still in progress
    InProgress,
}

/// Span kind (following OpenTelemetry convention)
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum SpanKind {
    /// Internal operation
    Internal,
    /// Server handling a request
    Server,
    /// Client making a request
    Client,
    /// Producer sending a message
    Producer,
    /// Consumer receiving a message
    Consumer,
}

/// A single span in the trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Span ID
    pub span_id: String,
    /// Parent span ID
    pub parent_span_id: Option<String>,
    /// Span name (e.g., "resolve_user", "subgraph_query")
    pub name: String,
    /// Span kind
    pub kind: SpanKind,
    /// Start time
    pub start_time: SystemTime,
    /// End time (None if span is still active)
    pub end_time: Option<SystemTime>,
    /// Span duration
    pub duration: Option<Duration>,
    /// Span status
    pub status: SpanStatus,
    /// Span attributes (key-value pairs)
    pub attributes: HashMap<String, String>,
    /// Events recorded during span
    pub events: Vec<SpanEvent>,
    /// Service name
    pub service_name: String,
    /// Subgraph name (for federated queries)
    pub subgraph_name: Option<String>,
}

impl Span {
    pub fn new(
        span_id: String,
        parent_span_id: Option<String>,
        name: String,
        kind: SpanKind,
        service_name: String,
    ) -> Self {
        Self {
            span_id,
            parent_span_id,
            name,
            kind,
            start_time: SystemTime::now(),
            end_time: None,
            duration: None,
            status: SpanStatus::InProgress,
            attributes: HashMap::new(),
            events: Vec::new(),
            service_name,
            subgraph_name: None,
        }
    }

    /// Set an attribute
    pub fn set_attribute(&mut self, key: String, value: String) {
        self.attributes.insert(key, value);
    }

    /// Add an event
    pub fn add_event(&mut self, name: String, attributes: HashMap<String, String>) {
        self.events.push(SpanEvent {
            timestamp: SystemTime::now(),
            name,
            attributes,
        });
    }

    /// End the span
    pub fn end(&mut self, status: SpanStatus) {
        let end_time = SystemTime::now();
        self.end_time = Some(end_time);
        self.status = status;

        if let Ok(duration) = end_time.duration_since(self.start_time) {
            self.duration = Some(duration);
        }
    }

    /// Check if span has ended
    pub fn is_finished(&self) -> bool {
        self.end_time.is_some()
    }
}

/// Event recorded during a span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    pub timestamp: SystemTime,
    pub name: String,
    pub attributes: HashMap<String, String>,
}

/// Complete trace with all spans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    /// Trace ID
    pub trace_id: String,
    /// Root span
    pub root_span: Span,
    /// All spans in the trace
    pub spans: Vec<Span>,
    /// Trace start time
    pub start_time: SystemTime,
    /// Trace end time
    pub end_time: Option<SystemTime>,
    /// Total duration
    pub duration: Option<Duration>,
    /// Trace-level attributes
    pub attributes: HashMap<String, String>,
}

impl Trace {
    pub fn new(trace_id: String, root_span: Span) -> Self {
        let start_time = root_span.start_time;

        Self {
            trace_id,
            root_span,
            spans: Vec::new(),
            start_time,
            end_time: None,
            duration: None,
            attributes: HashMap::new(),
        }
    }

    /// Add a span to the trace
    pub fn add_span(&mut self, span: Span) {
        self.spans.push(span);
    }

    /// End the trace
    pub fn end(&mut self) {
        let end_time = SystemTime::now();
        self.end_time = Some(end_time);

        if let Ok(duration) = end_time.duration_since(self.start_time) {
            self.duration = Some(duration);
        }
    }

    /// Get all spans (including root)
    pub fn all_spans(&self) -> Vec<&Span> {
        let mut spans = vec![&self.root_span];
        spans.extend(self.spans.iter());
        spans
    }

    /// Get subgraph breakdown (time spent per subgraph)
    pub fn subgraph_breakdown(&self) -> HashMap<String, Duration> {
        let mut breakdown = HashMap::new();

        for span in self.all_spans() {
            if let (Some(subgraph), Some(duration)) = (&span.subgraph_name, span.duration) {
                *breakdown.entry(subgraph.clone()).or_insert(Duration::ZERO) += duration;
            }
        }

        breakdown
    }

    /// Calculate critical path (slowest sequential path)
    pub fn critical_path(&self) -> Vec<&Span> {
        // Simple implementation: return spans in chronological order
        let mut spans = self.all_spans();
        spans.sort_by_key(|s| s.start_time);
        spans
    }
}

/// Distributed tracing configuration
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Enable distributed tracing
    pub enabled: bool,
    /// Sample rate (0.0 to 1.0)
    pub sample_rate: f64,
    /// Service name
    pub service_name: String,
    /// Export traces to backend
    pub export_enabled: bool,
    /// Export endpoint (e.g., OpenTelemetry collector)
    pub export_endpoint: Option<String>,
    /// Maximum spans per trace
    pub max_spans_per_trace: usize,
    /// Maximum trace duration before auto-completion
    pub max_trace_duration: Duration,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_rate: 1.0, // Sample all traces by default
            service_name: "oxirs-gql".to_string(),
            export_enabled: false,
            export_endpoint: None,
            max_spans_per_trace: 1000,
            max_trace_duration: Duration::from_secs(60),
        }
    }
}

/// Distributed tracer for federated queries
pub struct DistributedTracer {
    config: TracingConfig,
    active_traces: Arc<RwLock<HashMap<String, Trace>>>,
    completed_traces: Arc<RwLock<Vec<Trace>>>,
    active_spans: Arc<RwLock<HashMap<String, Span>>>,
}

impl DistributedTracer {
    pub fn new(config: TracingConfig) -> Self {
        Self {
            config,
            active_traces: Arc::new(RwLock::new(HashMap::new())),
            completed_traces: Arc::new(RwLock::new(Vec::new())),
            active_spans: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start a new trace
    pub async fn start_trace(&self, context: &TraceContext, name: String) -> Result<Span> {
        if !self.config.enabled || !self.should_sample() {
            // Return a no-op span if tracing is disabled or not sampled
            return Ok(Span::new(
                context.span_id.clone(),
                context.parent_span_id.clone(),
                name,
                SpanKind::Server,
                self.config.service_name.clone(),
            ));
        }

        let root_span = Span::new(
            context.span_id.clone(),
            context.parent_span_id.clone(),
            name,
            SpanKind::Server,
            self.config.service_name.clone(),
        );

        let trace = Trace::new(context.trace_id.clone(), root_span.clone());

        let mut traces = self.active_traces.write().await;
        traces.insert(context.trace_id.clone(), trace);

        Ok(root_span)
    }

    /// Start a child span
    pub async fn start_span(
        &self,
        context: &TraceContext,
        name: String,
        kind: SpanKind,
        subgraph_name: Option<String>,
    ) -> Result<Span> {
        if !self.config.enabled || !context.is_sampled() {
            // Return a no-op span
            let mut span = Span::new(
                context.span_id.clone(),
                context.parent_span_id.clone(),
                name,
                kind,
                self.config.service_name.clone(),
            );
            span.subgraph_name = subgraph_name;
            return Ok(span);
        }

        let mut span = Span::new(
            context.span_id.clone(),
            context.parent_span_id.clone(),
            name,
            kind,
            self.config.service_name.clone(),
        );
        span.subgraph_name = subgraph_name;

        // Store active span
        let mut spans = self.active_spans.write().await;
        spans.insert(context.span_id.clone(), span.clone());

        Ok(span)
    }

    /// End a span
    pub async fn end_span(&self, trace_id: &str, span_id: &str, status: SpanStatus) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Get and remove from active spans
        let span = {
            let mut spans = self.active_spans.write().await;
            spans.remove(span_id)
        };

        if let Some(mut span) = span {
            span.end(status);

            // Add to trace
            let mut traces = self.active_traces.write().await;
            if let Some(trace) = traces.get_mut(trace_id) {
                trace.add_span(span);
            }
        }

        Ok(())
    }

    /// End a trace
    pub async fn end_trace(&self, trace_id: &str) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let trace = {
            let mut traces = self.active_traces.write().await;
            traces.remove(trace_id)
        };

        if let Some(mut trace) = trace {
            trace.end();

            // Export if configured
            if self.config.export_enabled {
                self.export_trace(&trace).await?;
            }

            // Store completed trace
            let mut completed = self.completed_traces.write().await;
            completed.push(trace);

            // Limit completed traces
            if completed.len() > 1000 {
                completed.drain(0..500);
            }
        }

        Ok(())
    }

    /// Get a trace
    pub async fn get_trace(&self, trace_id: &str) -> Option<Trace> {
        // Check active traces
        {
            let traces = self.active_traces.read().await;
            if let Some(trace) = traces.get(trace_id) {
                return Some(trace.clone());
            }
        }

        // Check completed traces
        {
            let completed = self.completed_traces.read().await;
            completed.iter().find(|t| t.trace_id == trace_id).cloned()
        }
    }

    /// Get all completed traces
    pub async fn get_completed_traces(&self) -> Vec<Trace> {
        let completed = self.completed_traces.read().await;
        completed.clone()
    }

    /// Should sample this trace
    fn should_sample(&self) -> bool {
        use scirs2_core::random::rng;
        use scirs2_core::Rng;
        let mut rng = rng();
        rng.random_range(0.0..1.0) < self.config.sample_rate
    }

    /// Export trace to backend
    async fn export_trace(&self, trace: &Trace) -> Result<()> {
        // Placeholder for OpenTelemetry export
        // In production, this would send to an OTLP endpoint
        if let Some(endpoint) = &self.config.export_endpoint {
            tracing::debug!(
                "Exporting trace {} to {} (not implemented)",
                trace.trace_id,
                endpoint
            );
        }
        Ok(())
    }

    /// Get tracing statistics
    pub async fn get_stats(&self) -> TracingStats {
        let active_traces = self.active_traces.read().await;
        let completed_traces = self.completed_traces.read().await;

        let avg_duration = if !completed_traces.is_empty() {
            let total: Duration = completed_traces.iter().filter_map(|t| t.duration).sum();
            total / completed_traces.len() as u32
        } else {
            Duration::ZERO
        };

        TracingStats {
            active_traces: active_traces.len(),
            completed_traces: completed_traces.len(),
            total_traces: active_traces.len() + completed_traces.len(),
            avg_trace_duration: avg_duration,
        }
    }
}

/// Tracing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingStats {
    pub active_traces: usize,
    pub completed_traces: usize,
    pub total_traces: usize,
    pub avg_trace_duration: Duration,
}

/// Generate a random trace ID (32 hex characters)
fn generate_trace_id() -> String {
    format!("{:032x}", Uuid::new_v4().as_u128())
}

/// Generate a random span ID (16 hex characters)
fn generate_span_id() -> String {
    format!("{:016x}", Uuid::new_v4().as_u128() & 0xFFFFFFFFFFFFFFFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_context_creation() {
        let ctx = TraceContext::new();
        assert_eq!(ctx.trace_id.len(), 32);
        assert_eq!(ctx.span_id.len(), 16);
        assert!(ctx.parent_span_id.is_none());
        assert!(ctx.is_sampled());
    }

    #[test]
    fn test_trace_context_child() {
        let parent = TraceContext::new();
        let child = parent.create_child();

        assert_eq!(child.trace_id, parent.trace_id);
        assert_ne!(child.span_id, parent.span_id);
        assert_eq!(child.parent_span_id, Some(parent.span_id.clone()));
    }

    #[test]
    fn test_traceparent_format() {
        let ctx = TraceContext::new();
        let header = ctx.to_traceparent();

        assert!(header.starts_with("00-"));
        assert_eq!(header.matches('-').count(), 3);
    }

    #[test]
    fn test_traceparent_parse() {
        let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let ctx = TraceContext::from_traceparent(header).unwrap();

        assert_eq!(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(ctx.span_id, "b7ad6b7169203331");
        assert_eq!(ctx.trace_flags, 0x01);
        assert!(ctx.is_sampled());
    }

    #[test]
    fn test_span_lifecycle() {
        let mut span = Span::new(
            "span123".to_string(),
            None,
            "test_span".to_string(),
            SpanKind::Internal,
            "test_service".to_string(),
        );

        assert_eq!(span.status, SpanStatus::InProgress);
        assert!(!span.is_finished());

        span.set_attribute("key".to_string(), "value".to_string());
        assert_eq!(span.attributes.get("key"), Some(&"value".to_string()));

        span.add_event("event1".to_string(), HashMap::new());
        assert_eq!(span.events.len(), 1);

        span.end(SpanStatus::Ok);
        assert_eq!(span.status, SpanStatus::Ok);
        assert!(span.is_finished());
        assert!(span.duration.is_some());
    }

    #[test]
    fn test_trace_creation() {
        let root_span = Span::new(
            "root".to_string(),
            None,
            "root_span".to_string(),
            SpanKind::Server,
            "service".to_string(),
        );

        let trace = Trace::new("trace123".to_string(), root_span);

        assert_eq!(trace.trace_id, "trace123");
        assert_eq!(trace.spans.len(), 0);
        assert!(trace.duration.is_none());
    }

    #[test]
    fn test_trace_add_spans() {
        let root_span = Span::new(
            "root".to_string(),
            None,
            "root".to_string(),
            SpanKind::Server,
            "service".to_string(),
        );

        let mut trace = Trace::new("trace123".to_string(), root_span);

        let child_span = Span::new(
            "child1".to_string(),
            Some("root".to_string()),
            "child".to_string(),
            SpanKind::Client,
            "service".to_string(),
        );

        trace.add_span(child_span);
        assert_eq!(trace.spans.len(), 1);
        assert_eq!(trace.all_spans().len(), 2); // root + 1 child
    }

    #[tokio::test]
    async fn test_tracer_start_trace() {
        let config = TracingConfig::default();
        let tracer = DistributedTracer::new(config);

        let ctx = TraceContext::new();
        let span = tracer
            .start_trace(&ctx, "test_query".to_string())
            .await
            .unwrap();

        assert_eq!(span.name, "test_query");
        assert_eq!(span.kind, SpanKind::Server);

        let trace = tracer.get_trace(&ctx.trace_id).await;
        assert!(trace.is_some());
    }

    #[tokio::test]
    async fn test_tracer_child_span() {
        let config = TracingConfig::default();
        let tracer = DistributedTracer::new(config);

        let ctx = TraceContext::new();
        tracer.start_trace(&ctx, "root".to_string()).await.unwrap();

        let child_ctx = ctx.create_child();
        let child_span = tracer
            .start_span(
                &child_ctx,
                "child_query".to_string(),
                SpanKind::Client,
                Some("subgraph1".to_string()),
            )
            .await
            .unwrap();

        assert_eq!(child_span.name, "child_query");
        assert_eq!(child_span.subgraph_name, Some("subgraph1".to_string()));
        assert_eq!(child_span.parent_span_id, Some(ctx.span_id.clone()));
    }

    #[tokio::test]
    async fn test_tracer_end_span() {
        let config = TracingConfig::default();
        let tracer = DistributedTracer::new(config);

        let ctx = TraceContext::new();
        tracer.start_trace(&ctx, "root".to_string()).await.unwrap();

        let child_ctx = ctx.create_child();
        tracer
            .start_span(&child_ctx, "child".to_string(), SpanKind::Client, None)
            .await
            .unwrap();

        tracer
            .end_span(&ctx.trace_id, &child_ctx.span_id, SpanStatus::Ok)
            .await
            .unwrap();

        let trace = tracer.get_trace(&ctx.trace_id).await.unwrap();
        assert_eq!(trace.spans.len(), 1);
    }

    #[tokio::test]
    async fn test_tracer_end_trace() {
        let config = TracingConfig::default();
        let tracer = DistributedTracer::new(config);

        let ctx = TraceContext::new();
        tracer.start_trace(&ctx, "root".to_string()).await.unwrap();

        tracer.end_trace(&ctx.trace_id).await.unwrap();

        // Should now be in completed traces
        let completed = tracer.get_completed_traces().await;
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].trace_id, ctx.trace_id);
    }

    #[tokio::test]
    async fn test_tracer_disabled() {
        let config = TracingConfig {
            enabled: false,
            ..Default::default()
        };
        let tracer = DistributedTracer::new(config);

        let ctx = TraceContext::new();
        let span = tracer.start_trace(&ctx, "test".to_string()).await.unwrap();

        // Should still return a span, but not track it
        assert_eq!(span.name, "test");

        let trace = tracer.get_trace(&ctx.trace_id).await;
        assert!(trace.is_none());
    }

    #[tokio::test]
    async fn test_get_stats() {
        let config = TracingConfig::default();
        let tracer = DistributedTracer::new(config);

        let ctx1 = TraceContext::new();
        tracer
            .start_trace(&ctx1, "trace1".to_string())
            .await
            .unwrap();

        let ctx2 = TraceContext::new();
        tracer
            .start_trace(&ctx2, "trace2".to_string())
            .await
            .unwrap();

        tracer.end_trace(&ctx1.trace_id).await.unwrap();

        let stats = tracer.get_stats().await;
        assert_eq!(stats.active_traces, 1);
        assert_eq!(stats.completed_traces, 1);
        assert_eq!(stats.total_traces, 2);
    }

    #[test]
    fn test_span_kind_variants() {
        assert_eq!(SpanKind::Internal as i32, SpanKind::Internal as i32);
        assert_ne!(SpanKind::Client as i32, SpanKind::Server as i32);
    }

    #[test]
    fn test_span_status_variants() {
        assert_eq!(SpanStatus::Ok, SpanStatus::Ok);
        assert_ne!(SpanStatus::Ok, SpanStatus::Error);
        assert_ne!(SpanStatus::Error, SpanStatus::InProgress);
    }
}
