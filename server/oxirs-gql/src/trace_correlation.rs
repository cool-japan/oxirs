//! Trace Correlation for RDF Queries and SPARQL Execution
//!
//! This module provides distributed tracing correlation between GraphQL queries,
//! RDF operations, and SPARQL execution, enabling end-to-end observability.
//!
//! # Features
//!
//! - **W3C Trace Context**: Full W3C trace context propagation
//! - **Span Hierarchy**: Automatic parent-child span relationships
//! - **SPARQL Integration**: Trace SPARQL query execution with RDF operations
//! - **GraphQL Context**: Correlate GraphQL fields with underlying RDF queries
//! - **Performance Metrics**: Detailed timing for each operation
//! - **Error Tracking**: Capture and correlate errors across the stack
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_gql::trace_correlation::{TraceContext, SpanBuilder};
//!
//! // Create root trace
//! let trace = TraceContext::new("graphql-query");
//!
//! // Create child span for SPARQL execution
//! let sparql_span = trace.create_child_span("sparql-execution")
//!     .with_attribute("query", "SELECT * WHERE { ?s ?p ?o }")
//!     .build();
//!
//! // Execute and record timing
//! sparql_span.record_event("query_start");
//! // ... execute SPARQL ...
//! sparql_span.record_event("query_complete");
//! sparql_span.finish();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

/// W3C trace context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Trace ID (128-bit)
    pub trace_id: String,
    /// Parent span ID (64-bit)
    pub parent_span_id: Option<String>,
    /// Trace flags
    pub trace_flags: u8,
    /// Trace state
    pub trace_state: String,
}

impl TraceContext {
    /// Create a new root trace context
    pub fn new(operation: &str) -> Self {
        Self {
            trace_id: Self::generate_trace_id(),
            parent_span_id: None,
            trace_flags: 1, // Sampled
            trace_state: format!("operation={}", operation),
        }
    }

    /// Create from W3C traceparent header
    pub fn from_traceparent(traceparent: &str) -> Option<Self> {
        let parts: Vec<&str> = traceparent.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        Some(Self {
            trace_id: parts[1].to_string(),
            parent_span_id: Some(parts[2].to_string()),
            trace_flags: u8::from_str_radix(parts[3], 16).ok()?,
            trace_state: String::new(),
        })
    }

    /// Convert to W3C traceparent header
    pub fn to_traceparent(&self, span_id: &str) -> String {
        format!("00-{}-{}-{:02x}", self.trace_id, span_id, self.trace_flags)
    }

    /// Generate a new trace ID
    fn generate_trace_id() -> String {
        format!("{:032x}", fastrand::u128(..))
    }

    /// Generate a new span ID
    pub fn generate_span_id() -> String {
        format!("{:016x}", fastrand::u64(..))
    }

    /// Check if trace is sampled
    pub fn is_sampled(&self) -> bool {
        self.trace_flags & 0x01 != 0
    }
}

/// Span kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanKind {
    /// Internal operation
    Internal,
    /// Server (receiving request)
    Server,
    /// Client (making request)
    Client,
    /// Producer (sending to queue)
    Producer,
    /// Consumer (receiving from queue)
    Consumer,
}

/// Span status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanStatus {
    /// Unset (default)
    Unset,
    /// Ok (success)
    Ok,
    /// Error
    Error,
}

/// Span event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Event name
    pub name: String,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event attributes
    pub attributes: HashMap<String, AttributeValue>,
}

/// Attribute value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// Array of strings
    StringArray(Vec<String>),
}

/// Distributed tracing span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Span ID
    pub span_id: String,
    /// Trace context
    pub trace_context: TraceContext,
    /// Span name
    pub name: String,
    /// Span kind
    pub kind: SpanKind,
    /// Parent span ID
    pub parent_span_id: Option<String>,
    /// Start time
    pub start_time: SystemTime,
    /// End time
    pub end_time: Option<SystemTime>,
    /// Status
    pub status: SpanStatus,
    /// Attributes
    pub attributes: HashMap<String, AttributeValue>,
    /// Events
    pub events: Vec<SpanEvent>,
    /// Duration (computed when span ends)
    pub duration_ms: Option<u64>,
}

impl Span {
    /// Create a new span
    pub fn new(name: String, trace_context: TraceContext, kind: SpanKind) -> Self {
        Self {
            span_id: TraceContext::generate_span_id(),
            trace_context,
            name,
            kind,
            parent_span_id: None,
            start_time: SystemTime::now(),
            end_time: None,
            status: SpanStatus::Unset,
            attributes: HashMap::new(),
            events: Vec::new(),
            duration_ms: None,
        }
    }

    /// Set parent span ID
    pub fn with_parent(mut self, parent_span_id: String) -> Self {
        self.parent_span_id = Some(parent_span_id);
        self
    }

    /// Add attribute
    pub fn set_attribute(&mut self, key: String, value: AttributeValue) {
        self.attributes.insert(key, value);
    }

    /// Record an event
    pub fn record_event(&mut self, name: String) {
        self.events.push(SpanEvent {
            name,
            timestamp: SystemTime::now(),
            attributes: HashMap::new(),
        });
    }

    /// Record event with attributes
    pub fn record_event_with_attributes(
        &mut self,
        name: String,
        attributes: HashMap<String, AttributeValue>,
    ) {
        self.events.push(SpanEvent {
            name,
            timestamp: SystemTime::now(),
            attributes,
        });
    }

    /// Set status
    pub fn set_status(&mut self, status: SpanStatus) {
        self.status = status;
    }

    /// Finish the span
    pub fn finish(&mut self) {
        self.end_time = Some(SystemTime::now());
        if let Ok(duration) = self.end_time.unwrap().duration_since(self.start_time) {
            self.duration_ms = Some(duration.as_millis() as u64);
        }
    }

    /// Check if span is finished
    pub fn is_finished(&self) -> bool {
        self.end_time.is_some()
    }
}

/// Span builder for fluent API
pub struct SpanBuilder {
    name: String,
    trace_context: TraceContext,
    kind: SpanKind,
    parent_span_id: Option<String>,
    attributes: HashMap<String, AttributeValue>,
}

impl SpanBuilder {
    /// Create a new span builder
    pub fn new(name: String, trace_context: TraceContext) -> Self {
        Self {
            name,
            trace_context,
            kind: SpanKind::Internal,
            parent_span_id: None,
            attributes: HashMap::new(),
        }
    }

    /// Set span kind
    pub fn with_kind(mut self, kind: SpanKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set parent span
    pub fn with_parent(mut self, parent_span_id: String) -> Self {
        self.parent_span_id = Some(parent_span_id);
        self
    }

    /// Add attribute
    pub fn with_attribute(mut self, key: &str, value: AttributeValue) -> Self {
        self.attributes.insert(key.to_string(), value);
        self
    }

    /// Build the span
    pub fn build(self) -> Span {
        let mut span = Span::new(self.name, self.trace_context, self.kind);
        if let Some(parent) = self.parent_span_id {
            span = span.with_parent(parent);
        }
        for (key, value) in self.attributes {
            span.set_attribute(key, value);
        }
        span
    }
}

/// Trace correlation manager
pub struct TraceCorrelator {
    /// Active spans
    active_spans: Arc<RwLock<HashMap<String, Span>>>,
    /// Completed spans
    completed_spans: Arc<RwLock<Vec<Span>>>,
    /// Maximum completed spans to retain
    max_completed_spans: usize,
}

impl TraceCorrelator {
    /// Create a new trace correlator
    pub fn new() -> Self {
        Self {
            active_spans: Arc::new(RwLock::new(HashMap::new())),
            completed_spans: Arc::new(RwLock::new(Vec::new())),
            max_completed_spans: 10000,
        }
    }

    /// Create with custom max completed spans
    pub fn with_max_completed_spans(mut self, max: usize) -> Self {
        self.max_completed_spans = max;
        self
    }

    /// Start a new span
    pub async fn start_span(&self, span: Span) -> String {
        let span_id = span.span_id.clone();
        let mut spans = self.active_spans.write().await;
        spans.insert(span_id.clone(), span);
        span_id
    }

    /// Get active span
    pub async fn get_span(&self, span_id: &str) -> Option<Span> {
        let spans = self.active_spans.read().await;
        spans.get(span_id).cloned()
    }

    /// Update span
    pub async fn update_span<F>(&self, span_id: &str, update_fn: F)
    where
        F: FnOnce(&mut Span),
    {
        let mut spans = self.active_spans.write().await;
        if let Some(span) = spans.get_mut(span_id) {
            update_fn(span);
        }
    }

    /// Finish span
    pub async fn finish_span(&self, span_id: &str) {
        let mut active = self.active_spans.write().await;
        if let Some(mut span) = active.remove(span_id) {
            span.finish();

            let mut completed = self.completed_spans.write().await;
            completed.push(span);

            // Trim if exceeds max
            if completed.len() > self.max_completed_spans {
                let excess = completed.len() - self.max_completed_spans;
                completed.drain(0..excess);
            }
        }
    }

    /// Get all spans for a trace
    pub async fn get_trace_spans(&self, trace_id: &str) -> Vec<Span> {
        let completed = self.completed_spans.read().await;
        completed
            .iter()
            .filter(|span| span.trace_context.trace_id == trace_id)
            .cloned()
            .collect()
    }

    /// Get span hierarchy (parent-child relationships)
    pub async fn get_span_hierarchy(&self, trace_id: &str) -> HashMap<String, Vec<String>> {
        let spans = self.get_trace_spans(trace_id).await;
        let mut hierarchy: HashMap<String, Vec<String>> = HashMap::new();

        for span in &spans {
            if let Some(parent_id) = &span.parent_span_id {
                hierarchy
                    .entry(parent_id.clone())
                    .or_default()
                    .push(span.span_id.clone());
            }
        }

        hierarchy
    }

    /// Clear all completed spans
    pub async fn clear_completed_spans(&self) {
        let mut completed = self.completed_spans.write().await;
        completed.clear();
    }

    /// Get statistics
    pub async fn get_statistics(&self) -> CorrelationStatistics {
        let active = self.active_spans.read().await;
        let completed = self.completed_spans.read().await;

        CorrelationStatistics {
            active_spans: active.len(),
            completed_spans: completed.len(),
            total_traces: completed
                .iter()
                .map(|s| s.trace_context.trace_id.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len(),
        }
    }
}

impl Default for TraceCorrelator {
    fn default() -> Self {
        Self::new()
    }
}

/// Correlation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationStatistics {
    /// Number of active spans
    pub active_spans: usize,
    /// Number of completed spans
    pub completed_spans: usize,
    /// Number of unique traces
    pub total_traces: usize,
}

/// SPARQL trace context
#[derive(Debug, Clone)]
pub struct SparqlTraceContext {
    /// Parent trace context
    pub trace_context: TraceContext,
    /// GraphQL field path
    pub field_path: Vec<String>,
    /// SPARQL query
    pub query: String,
    /// Query type (SELECT, CONSTRUCT, ASK, DESCRIBE)
    pub query_type: String,
}

impl SparqlTraceContext {
    /// Create a new SPARQL trace context
    pub fn new(
        trace_context: TraceContext,
        field_path: Vec<String>,
        query: String,
        query_type: String,
    ) -> Self {
        Self {
            trace_context,
            field_path,
            query,
            query_type,
        }
    }

    /// Create span for SPARQL execution
    pub fn create_sparql_span(&self) -> Span {
        let mut span = Span::new(
            "sparql-execution".to_string(),
            self.trace_context.clone(),
            SpanKind::Internal,
        );

        span.set_attribute(
            "graphql.field_path".to_string(),
            AttributeValue::String(self.field_path.join(".")),
        );
        span.set_attribute(
            "sparql.query".to_string(),
            AttributeValue::String(self.query.clone()),
        );
        span.set_attribute(
            "sparql.query_type".to_string(),
            AttributeValue::String(self.query_type.clone()),
        );

        span
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_context_creation() {
        let ctx = TraceContext::new("test-operation");

        assert!(!ctx.trace_id.is_empty());
        assert!(ctx.parent_span_id.is_none());
        assert_eq!(ctx.trace_flags, 1);
        assert!(ctx.trace_state.contains("operation=test-operation"));
    }

    #[test]
    fn test_trace_context_from_traceparent() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-00f067aa0ba902b7-01";
        let ctx = TraceContext::from_traceparent(traceparent).unwrap();

        assert_eq!(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(ctx.parent_span_id.as_ref().unwrap(), "00f067aa0ba902b7");
        assert_eq!(ctx.trace_flags, 1);
        assert!(ctx.is_sampled());
    }

    #[test]
    fn test_trace_context_to_traceparent() {
        let ctx = TraceContext {
            trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
            parent_span_id: None,
            trace_flags: 1,
            trace_state: String::new(),
        };

        let span_id = "00f067aa0ba902b7";
        let traceparent = ctx.to_traceparent(span_id);

        assert_eq!(
            traceparent,
            "00-0af7651916cd43dd8448eb211c80319c-00f067aa0ba902b7-01"
        );
    }

    #[test]
    fn test_span_creation() {
        let ctx = TraceContext::new("test");
        let span = Span::new("test-span".to_string(), ctx, SpanKind::Server);

        assert_eq!(span.name, "test-span");
        assert_eq!(span.kind, SpanKind::Server);
        assert_eq!(span.status, SpanStatus::Unset);
        assert!(!span.is_finished());
    }

    #[test]
    fn test_span_with_parent() {
        let ctx = TraceContext::new("test");
        let parent_id = "parent123".to_string();
        let span = Span::new("child-span".to_string(), ctx, SpanKind::Internal)
            .with_parent(parent_id.clone());

        assert_eq!(span.parent_span_id, Some(parent_id));
    }

    #[test]
    fn test_span_attributes() {
        let ctx = TraceContext::new("test");
        let mut span = Span::new("test-span".to_string(), ctx, SpanKind::Internal);

        span.set_attribute(
            "key1".to_string(),
            AttributeValue::String("value1".to_string()),
        );
        span.set_attribute("key2".to_string(), AttributeValue::Int(42));
        span.set_attribute("key3".to_string(), AttributeValue::Bool(true));

        assert_eq!(span.attributes.len(), 3);
        match span.attributes.get("key2").unwrap() {
            AttributeValue::Int(v) => assert_eq!(*v, 42),
            _ => panic!("Wrong attribute type"),
        }
    }

    #[test]
    fn test_span_events() {
        let ctx = TraceContext::new("test");
        let mut span = Span::new("test-span".to_string(), ctx, SpanKind::Internal);

        span.record_event("event1".to_string());
        span.record_event("event2".to_string());

        assert_eq!(span.events.len(), 2);
        assert_eq!(span.events[0].name, "event1");
        assert_eq!(span.events[1].name, "event2");
    }

    #[test]
    fn test_span_finish() {
        let ctx = TraceContext::new("test");
        let mut span = Span::new("test-span".to_string(), ctx, SpanKind::Internal);

        assert!(!span.is_finished());
        assert!(span.duration_ms.is_none());

        span.finish();

        assert!(span.is_finished());
        assert!(span.duration_ms.is_some());
    }

    #[test]
    fn test_span_builder() {
        let ctx = TraceContext::new("test");
        let span = SpanBuilder::new("test-span".to_string(), ctx)
            .with_kind(SpanKind::Client)
            .with_parent("parent123".to_string())
            .with_attribute("key", AttributeValue::String("value".to_string()))
            .build();

        assert_eq!(span.kind, SpanKind::Client);
        assert_eq!(span.parent_span_id, Some("parent123".to_string()));
        assert_eq!(span.attributes.len(), 1);
    }

    #[tokio::test]
    async fn test_trace_correlator_start_span() {
        let correlator = TraceCorrelator::new();
        let ctx = TraceContext::new("test");
        let span = Span::new("test-span".to_string(), ctx, SpanKind::Internal);

        let span_id = correlator.start_span(span).await;

        let retrieved = correlator.get_span(&span_id).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "test-span");
    }

    #[tokio::test]
    async fn test_trace_correlator_update_span() {
        let correlator = TraceCorrelator::new();
        let ctx = TraceContext::new("test");
        let span = Span::new("test-span".to_string(), ctx, SpanKind::Internal);

        let span_id = correlator.start_span(span).await;

        correlator
            .update_span(&span_id, |s| {
                s.set_attribute("new_attr".to_string(), AttributeValue::Int(123));
            })
            .await;

        let updated = correlator.get_span(&span_id).await.unwrap();
        assert_eq!(updated.attributes.len(), 1);
    }

    #[tokio::test]
    async fn test_trace_correlator_finish_span() {
        let correlator = TraceCorrelator::new();
        let ctx = TraceContext::new("test");
        let span = Span::new("test-span".to_string(), ctx, SpanKind::Internal);

        let span_id = correlator.start_span(span).await;
        correlator.finish_span(&span_id).await;

        let stats = correlator.get_statistics().await;
        assert_eq!(stats.active_spans, 0);
        assert_eq!(stats.completed_spans, 1);
    }

    #[tokio::test]
    async fn test_trace_correlator_get_trace_spans() {
        let correlator = TraceCorrelator::new();
        let ctx = TraceContext::new("test");
        let trace_id = ctx.trace_id.clone();

        let span1 = Span::new("span1".to_string(), ctx.clone(), SpanKind::Internal);
        let span2 = Span::new("span2".to_string(), ctx, SpanKind::Internal);

        correlator.start_span(span1).await;
        correlator.start_span(span2).await;

        // Finish both spans
        let active_spans = correlator.active_spans.read().await;
        let span_ids: Vec<_> = active_spans.keys().cloned().collect();
        drop(active_spans);

        for span_id in span_ids {
            correlator.finish_span(&span_id).await;
        }

        let trace_spans = correlator.get_trace_spans(&trace_id).await;
        assert_eq!(trace_spans.len(), 2);
    }

    #[tokio::test]
    async fn test_trace_correlator_hierarchy() {
        let correlator = TraceCorrelator::new();
        let ctx = TraceContext::new("test");
        let trace_id = ctx.trace_id.clone();

        let parent_span = Span::new("parent".to_string(), ctx.clone(), SpanKind::Server);
        let parent_id = parent_span.span_id.clone();

        let child_span =
            Span::new("child".to_string(), ctx, SpanKind::Internal).with_parent(parent_id.clone());

        correlator.start_span(parent_span).await;
        correlator.start_span(child_span).await;

        // Finish spans
        let active_spans = correlator.active_spans.read().await;
        let span_ids: Vec<_> = active_spans.keys().cloned().collect();
        drop(active_spans);

        for span_id in span_ids {
            correlator.finish_span(&span_id).await;
        }

        let hierarchy = correlator.get_span_hierarchy(&trace_id).await;
        assert!(hierarchy.contains_key(&parent_id));
    }

    #[tokio::test]
    async fn test_trace_correlator_clear() {
        let correlator = TraceCorrelator::new();
        let ctx = TraceContext::new("test");
        let span = Span::new("test-span".to_string(), ctx, SpanKind::Internal);

        let span_id = correlator.start_span(span).await;
        correlator.finish_span(&span_id).await;

        let stats_before = correlator.get_statistics().await;
        assert_eq!(stats_before.completed_spans, 1);

        correlator.clear_completed_spans().await;

        let stats_after = correlator.get_statistics().await;
        assert_eq!(stats_after.completed_spans, 0);
    }

    #[tokio::test]
    async fn test_trace_correlator_max_completed_spans() {
        let correlator = TraceCorrelator::new().with_max_completed_spans(5);

        for i in 0..10 {
            let ctx = TraceContext::new("test");
            let span = Span::new(format!("span-{}", i), ctx, SpanKind::Internal);
            let span_id = correlator.start_span(span).await;
            correlator.finish_span(&span_id).await;
        }

        let stats = correlator.get_statistics().await;
        assert_eq!(stats.completed_spans, 5);
    }

    #[test]
    fn test_sparql_trace_context() {
        let ctx = TraceContext::new("graphql-query");
        let sparql_ctx = SparqlTraceContext::new(
            ctx,
            vec!["user".to_string(), "posts".to_string()],
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
            "SELECT".to_string(),
        );

        let span = sparql_ctx.create_sparql_span();

        assert_eq!(span.name, "sparql-execution");
        assert!(span.attributes.contains_key("graphql.field_path"));
        assert!(span.attributes.contains_key("sparql.query"));
    }

    #[test]
    fn test_span_status() {
        let ctx = TraceContext::new("test");
        let mut span = Span::new("test-span".to_string(), ctx, SpanKind::Internal);

        assert_eq!(span.status, SpanStatus::Unset);

        span.set_status(SpanStatus::Ok);
        assert_eq!(span.status, SpanStatus::Ok);

        span.set_status(SpanStatus::Error);
        assert_eq!(span.status, SpanStatus::Error);
    }

    #[test]
    fn test_attribute_value_types() {
        let string_attr = AttributeValue::String("test".to_string());
        let int_attr = AttributeValue::Int(42);
        let float_attr = AttributeValue::Float(1.5);
        let bool_attr = AttributeValue::Bool(true);
        let array_attr = AttributeValue::StringArray(vec!["a".to_string(), "b".to_string()]);

        match string_attr {
            AttributeValue::String(s) => assert_eq!(s, "test"),
            _ => panic!("Wrong type"),
        }

        match int_attr {
            AttributeValue::Int(i) => assert_eq!(i, 42),
            _ => panic!("Wrong type"),
        }

        match float_attr {
            AttributeValue::Float(f) => assert!((f - 1.5).abs() < 0.001),
            _ => panic!("Wrong type"),
        }

        match bool_attr {
            AttributeValue::Bool(b) => assert!(b),
            _ => panic!("Wrong type"),
        }

        match array_attr {
            AttributeValue::StringArray(arr) => assert_eq!(arr.len(), 2),
            _ => panic!("Wrong type"),
        }
    }
}
