//! Distributed Tracing for Federation Queries
//!
//! This module provides comprehensive OpenTelemetry-based distributed tracing
//! for federated query processing, including cross-service correlation and
//! performance analysis.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Distributed tracing manager for federation queries
#[derive(Debug)]
pub struct DistributedTracingManager {
    config: TracingConfig,
    active_traces: Arc<RwLock<HashMap<String, TraceContext>>>,
    span_storage: Arc<RwLock<HashMap<String, Vec<Span>>>>,
    metrics: Arc<RwLock<TracingMetrics>>,
}

/// Configuration for distributed tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Enable distributed tracing
    pub enabled: bool,
    /// OpenTelemetry service name
    pub service_name: String,
    /// OTLP exporter endpoint
    pub otlp_endpoint: Option<String>,
    /// Jaeger exporter endpoint  
    pub jaeger_endpoint: Option<String>,
    /// Sampling rate (0.0 - 1.0)
    pub sampling_rate: f64,
    /// Maximum trace duration before auto-completion
    pub max_trace_duration: Duration,
    /// Enable detailed span attributes
    pub enable_detailed_attributes: bool,
    /// Span attribute size limit
    pub max_attribute_size: usize,
    /// Number of recent traces to keep in memory
    pub max_recent_traces: usize,
    /// Enable performance analysis
    pub enable_performance_analysis: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            service_name: "oxirs-federate".to_string(),
            otlp_endpoint: None,
            jaeger_endpoint: None,
            sampling_rate: 0.1,                           // Sample 10% of traces
            max_trace_duration: Duration::from_secs(300), // 5 minutes
            enable_detailed_attributes: true,
            max_attribute_size: 1024,
            max_recent_traces: 1000,
            enable_performance_analysis: true,
        }
    }
}

/// Trace context for correlation across services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Unique trace identifier
    pub trace_id: String,
    /// Parent span identifier
    pub parent_span_id: Option<String>,
    /// Trace flags for sampling decisions
    pub trace_flags: u8,
    /// Baggage for cross-cutting concerns
    pub baggage: HashMap<String, String>,
    /// Trace start time
    pub start_time: SystemTime,
    /// Query context information
    pub query_context: QueryContext,
    /// Service path through federation
    pub service_path: Vec<String>,
}

/// Query context for tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    /// Original query string
    pub query: String,
    /// Query type (SPARQL, GraphQL, etc.)
    pub query_type: String,
    /// User/client identifier
    pub user_id: Option<String>,
    /// Session identifier
    pub session_id: Option<String>,
    /// Request priority
    pub priority: QueryPriority,
}

/// Query priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Distributed span information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Unique span identifier
    pub span_id: String,
    /// Parent span identifier
    pub parent_span_id: Option<String>,
    /// Trace identifier this span belongs to
    pub trace_id: String,
    /// Span operation name
    pub operation_name: String,
    /// Service that created this span
    pub service_name: String,
    /// Span start time
    pub start_time: SystemTime,
    /// Span end time
    pub end_time: Option<SystemTime>,
    /// Span duration
    pub duration: Option<Duration>,
    /// Span attributes/tags
    pub attributes: HashMap<String, String>,
    /// Span events (logs)
    pub events: Vec<SpanEvent>,
    /// Span status
    pub status: SpanStatus,
    /// Resource information
    pub resource: SpanResource,
}

/// Span event (log entry)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event name
    pub name: String,
    /// Event attributes
    pub attributes: HashMap<String, String>,
}

/// Span completion status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanStatus {
    Ok,
    Error(String),
    Timeout,
    Cancelled,
}

/// Resource information for spans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanResource {
    /// Service name
    pub service_name: String,
    /// Service version
    pub service_version: Option<String>,
    /// Service instance ID
    pub service_instance_id: Option<String>,
    /// Host information
    pub host_name: Option<String>,
    /// Process information
    pub process_id: Option<u32>,
}

/// Tracing metrics and statistics
#[derive(Debug, Default, Clone)]
pub struct TracingMetrics {
    /// Total traces created
    pub total_traces: u64,
    /// Total spans created
    pub total_spans: u64,
    /// Traces by status
    pub traces_by_status: HashMap<String, u64>,
    /// Average trace duration
    pub avg_trace_duration: Duration,
    /// Average spans per trace
    pub avg_spans_per_trace: f64,
    /// Service latency statistics
    pub service_latencies: HashMap<String, ServiceLatencyStats>,
    /// Error rate by service
    pub service_error_rates: HashMap<String, f64>,
}

/// Service-specific latency statistics
#[derive(Debug, Clone)]
pub struct ServiceLatencyStats {
    /// Service name
    pub service_name: String,
    /// Total requests
    pub total_requests: u64,
    /// Total duration
    pub total_duration: Duration,
    /// Average latency
    pub avg_latency: Duration,
    /// P50 latency
    pub p50_latency: Duration,
    /// P95 latency
    pub p95_latency: Duration,
    /// P99 latency
    pub p99_latency: Duration,
    /// Recent latency samples for percentile calculation
    pub recent_samples: Vec<Duration>,
}

impl Default for ServiceLatencyStats {
    fn default() -> Self {
        Self {
            service_name: String::new(),
            total_requests: 0,
            total_duration: Duration::from_secs(0),
            avg_latency: Duration::from_secs(0),
            p50_latency: Duration::from_secs(0),
            p95_latency: Duration::from_secs(0),
            p99_latency: Duration::from_secs(0),
            recent_samples: Vec::new(),
        }
    }
}

/// Trace analysis result
#[derive(Debug, Clone, Serialize)]
pub struct TraceAnalysis {
    /// Trace ID
    pub trace_id: String,
    /// Total trace duration
    pub total_duration: Duration,
    /// Critical path analysis
    pub critical_path: Vec<String>,
    /// Service breakdown
    pub service_breakdown: HashMap<String, Duration>,
    /// Parallel execution opportunities
    pub parallelization_opportunities: Vec<ParallelizationOpportunity>,
    /// Performance bottlenecks
    pub bottlenecks: Vec<TracingBottleneck>,
    /// Error propagation path
    pub error_path: Option<Vec<String>>,
}

/// Parallelization opportunity identified from trace
#[derive(Debug, Clone, Serialize)]
pub struct ParallelizationOpportunity {
    /// Description of the opportunity
    pub description: String,
    /// Potential time savings
    pub potential_savings: Duration,
    /// Services that could be parallelized
    pub services: Vec<String>,
    /// Confidence level of the recommendation
    pub confidence: f64,
}

/// Performance bottleneck identified from tracing
#[derive(Debug, Clone, Serialize)]
pub struct TracingBottleneck {
    /// Service or operation causing the bottleneck
    pub component: String,
    /// Duration of the bottleneck
    pub duration: Duration,
    /// Percentage of total trace time
    pub percentage_of_trace: f64,
    /// Bottleneck type
    pub bottleneck_type: TracingBottleneckType,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Types of bottlenecks identifiable through tracing
#[derive(Debug, Clone, Serialize)]
pub enum TracingBottleneckType {
    SlowService,
    NetworkLatency,
    SerialExecution,
    LargeDataTransfer,
    DatabaseQuery,
    Authentication,
    Caching,
}

impl DistributedTracingManager {
    /// Create a new distributed tracing manager
    pub fn new() -> Self {
        Self::with_config(TracingConfig::default())
    }

    /// Create a new distributed tracing manager with configuration
    pub fn with_config(config: TracingConfig) -> Self {
        Self {
            config,
            active_traces: Arc::new(RwLock::new(HashMap::new())),
            span_storage: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(TracingMetrics::default())),
        }
    }

    /// Start a new distributed trace
    pub async fn start_trace(&self, query_context: QueryContext) -> Result<TraceContext> {
        if !self.config.enabled {
            return Err(anyhow!("Distributed tracing is disabled"));
        }

        let trace_id = Uuid::new_v4().to_string();
        let trace_context = TraceContext {
            trace_id: trace_id.clone(),
            parent_span_id: None,
            trace_flags: if self.should_sample() { 1 } else { 0 },
            baggage: HashMap::new(),
            start_time: SystemTime::now(),
            query_context,
            service_path: vec![self.config.service_name.clone()],
        };

        // Store active trace
        self.active_traces
            .write()
            .await
            .insert(trace_id.clone(), trace_context.clone());

        // Initialize span storage for this trace
        self.span_storage
            .write()
            .await
            .insert(trace_id.clone(), Vec::new());

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_traces += 1;

        info!("Started distributed trace: {}", trace_id);
        Ok(trace_context)
    }

    /// Create a new span within a trace
    pub async fn create_span(
        &self,
        trace_context: &TraceContext,
        operation_name: &str,
        service_name: &str,
        parent_span_id: Option<String>,
    ) -> Result<Span> {
        let span_id = Uuid::new_v4().to_string();
        let mut span = Span {
            span_id: span_id.clone(),
            parent_span_id,
            trace_id: trace_context.trace_id.clone(),
            operation_name: operation_name.to_string(),
            service_name: service_name.to_string(),
            start_time: SystemTime::now(),
            end_time: None,
            duration: None,
            attributes: HashMap::new(),
            events: Vec::new(),
            status: SpanStatus::Ok,
            resource: SpanResource {
                service_name: service_name.to_string(),
                service_version: Some("1.0.0".to_string()),
                service_instance_id: Some(Uuid::new_v4().to_string()),
                host_name: Some(
                    std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
                ),
                process_id: Some(std::process::id()),
            },
        };

        // Add default attributes
        if self.config.enable_detailed_attributes {
            span.attributes.insert(
                "query.type".to_string(),
                trace_context.query_context.query_type.clone(),
            );
            span.attributes.insert(
                "trace.sampled".to_string(),
                (trace_context.trace_flags & 1 == 1).to_string(),
            );

            if let Some(user_id) = &trace_context.query_context.user_id {
                span.attributes
                    .insert("user.id".to_string(), user_id.clone());
            }

            if let Some(session_id) = &trace_context.query_context.session_id {
                span.attributes
                    .insert("session.id".to_string(), session_id.clone());
            }
        }

        // Store span
        if let Some(spans) = self
            .span_storage
            .write()
            .await
            .get_mut(&trace_context.trace_id)
        {
            spans.push(span.clone());
        }

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_spans += 1;

        debug!(
            "Created span: {} in trace: {}",
            span_id, trace_context.trace_id
        );
        Ok(span)
    }

    /// Finish a span
    pub async fn finish_span(&self, span_id: &str, status: SpanStatus) -> Result<()> {
        let mut span_storage = self.span_storage.write().await;

        for spans in span_storage.values_mut() {
            if let Some(span) = spans.iter_mut().find(|s| s.span_id == span_id) {
                span.end_time = Some(SystemTime::now());
                span.duration = span.end_time.map(|end| {
                    end.duration_since(span.start_time)
                        .unwrap_or(Duration::from_secs(0))
                });
                span.status = status;

                // Update service latency statistics
                if let Some(duration) = span.duration {
                    self.update_service_latency_stats(&span.service_name, duration)
                        .await;
                }

                debug!("Finished span: {} with status: {:?}", span_id, span.status);
                return Ok(());
            }
        }

        warn!("Span not found for finishing: {}", span_id);
        Ok(())
    }

    /// Add attributes to a span
    pub async fn add_span_attributes(
        &self,
        span_id: &str,
        attributes: HashMap<String, String>,
    ) -> Result<()> {
        let mut span_storage = self.span_storage.write().await;

        for spans in span_storage.values_mut() {
            if let Some(span) = spans.iter_mut().find(|s| s.span_id == span_id) {
                for (key, value) in attributes {
                    let truncated_value = if value.len() > self.config.max_attribute_size {
                        format!("{}...(truncated)", &value[..self.config.max_attribute_size])
                    } else {
                        value
                    };
                    span.attributes.insert(key, truncated_value);
                }
                return Ok(());
            }
        }

        warn!("Span not found for adding attributes: {}", span_id);
        Ok(())
    }

    /// Add an event to a span
    pub async fn add_span_event(
        &self,
        span_id: &str,
        event_name: &str,
        attributes: HashMap<String, String>,
    ) -> Result<()> {
        let mut span_storage = self.span_storage.write().await;

        for spans in span_storage.values_mut() {
            if let Some(span) = spans.iter_mut().find(|s| s.span_id == span_id) {
                let event = SpanEvent {
                    timestamp: SystemTime::now(),
                    name: event_name.to_string(),
                    attributes,
                };
                span.events.push(event);
                return Ok(());
            }
        }

        warn!("Span not found for adding event: {}", span_id);
        Ok(())
    }

    /// Complete a trace
    pub async fn complete_trace(&self, trace_id: &str) -> Result<TraceAnalysis> {
        // Remove from active traces
        let trace_context = self.active_traces.write().await.remove(trace_id);

        if let Some(_context) = trace_context {
            let analysis = self.analyze_trace(trace_id).await?;

            // Update trace duration metrics
            let mut metrics = self.metrics.write().await;
            let total_traces = metrics.total_traces;
            if total_traces > 0 {
                let current_avg = metrics.avg_trace_duration.as_millis() as f64;
                let new_avg = (current_avg * (total_traces - 1) as f64
                    + analysis.total_duration.as_millis() as f64)
                    / total_traces as f64;
                metrics.avg_trace_duration = Duration::from_millis(new_avg as u64);
            }

            // Update completion status
            let status = if analysis.error_path.is_some() {
                "error"
            } else {
                "success"
            };
            *metrics
                .traces_by_status
                .entry(status.to_string())
                .or_insert(0) += 1;

            info!(
                "Completed trace: {} with {} spans in {:?}",
                trace_id,
                analysis.service_breakdown.len(),
                analysis.total_duration
            );

            Ok(analysis)
        } else {
            Err(anyhow!("Trace not found: {}", trace_id))
        }
    }

    /// Analyze a completed trace for performance insights
    pub async fn analyze_trace(&self, trace_id: &str) -> Result<TraceAnalysis> {
        let span_storage = self.span_storage.read().await;
        let spans = span_storage
            .get(trace_id)
            .ok_or_else(|| anyhow!("Trace not found: {}", trace_id))?;

        if spans.is_empty() {
            return Err(anyhow!("No spans found for trace: {}", trace_id));
        }

        // Calculate total duration
        let start_time = spans.iter().map(|s| s.start_time).min().unwrap();
        let end_time = spans
            .iter()
            .filter_map(|s| s.end_time)
            .max()
            .unwrap_or(SystemTime::now());
        let total_duration = end_time
            .duration_since(start_time)
            .unwrap_or(Duration::from_secs(0));

        // Analyze critical path
        let critical_path = self.calculate_critical_path(spans);

        // Service breakdown
        let mut service_breakdown = HashMap::new();
        for span in spans {
            if let Some(duration) = span.duration {
                *service_breakdown
                    .entry(span.service_name.clone())
                    .or_insert(Duration::from_secs(0)) += duration;
            }
        }

        // Find parallelization opportunities
        let parallelization_opportunities = self.find_parallelization_opportunities(spans);

        // Identify bottlenecks
        let bottlenecks = self.identify_trace_bottlenecks(spans, total_duration);

        // Check for error propagation
        let error_path = self.trace_error_propagation(spans);

        Ok(TraceAnalysis {
            trace_id: trace_id.to_string(),
            total_duration,
            critical_path,
            service_breakdown,
            parallelization_opportunities,
            bottlenecks,
            error_path,
        })
    }

    /// Get current tracing metrics
    pub async fn get_metrics(&self) -> TracingMetrics {
        self.metrics.read().await.clone()
    }

    /// Export traces in OpenTelemetry format
    pub async fn export_traces(&self, trace_ids: Vec<String>) -> Result<String> {
        let span_storage = self.span_storage.read().await;
        let mut exported_traces = Vec::new();

        for trace_id in trace_ids {
            if let Some(spans) = span_storage.get(&trace_id) {
                exported_traces.push(serde_json::json!({
                    "traceId": trace_id,
                    "spans": spans
                }));
            }
        }

        let export_data = serde_json::json!({
            "traces": exported_traces,
            "exportedAt": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        });

        Ok(serde_json::to_string_pretty(&export_data)?)
    }

    // Private helper methods

    fn should_sample(&self) -> bool {
        use scirs2_core::random::{Rng, Random};
        let mut random = Random::default();
        random.gen::<f64>() < self.config.sampling_rate
    }

    async fn update_service_latency_stats(&self, service_name: &str, duration: Duration) {
        let mut metrics = self.metrics.write().await;
        let stats = metrics
            .service_latencies
            .entry(service_name.to_string())
            .or_insert_with(|| ServiceLatencyStats {
                service_name: service_name.to_string(),
                ..Default::default()
            });

        stats.total_requests += 1;
        stats.total_duration += duration;
        stats.avg_latency = stats.total_duration / stats.total_requests as u32;

        // Update recent samples for percentile calculation
        stats.recent_samples.push(duration);
        if stats.recent_samples.len() > 1000 {
            stats.recent_samples.remove(0);
        }

        // Calculate percentiles
        if !stats.recent_samples.is_empty() {
            let mut sorted_samples = stats.recent_samples.clone();
            sorted_samples.sort();

            let p50_idx = (sorted_samples.len() as f64 * 0.5) as usize;
            let p95_idx = (sorted_samples.len() as f64 * 0.95) as usize;
            let p99_idx = (sorted_samples.len() as f64 * 0.99) as usize;

            stats.p50_latency = sorted_samples
                .get(p50_idx)
                .copied()
                .unwrap_or(Duration::from_secs(0));
            stats.p95_latency = sorted_samples
                .get(p95_idx)
                .copied()
                .unwrap_or(Duration::from_secs(0));
            stats.p99_latency = sorted_samples
                .get(p99_idx)
                .copied()
                .unwrap_or(Duration::from_secs(0));
        }
    }

    fn calculate_critical_path(&self, spans: &[Span]) -> Vec<String> {
        // Simple critical path calculation - longest sequential chain
        let mut path = Vec::new();
        let mut current_span: Option<&Span> = spans
            .iter()
            .filter(|s| s.parent_span_id.is_none())
            .max_by_key(|s| s.duration.unwrap_or(Duration::from_secs(0)));

        while let Some(span) = current_span {
            path.push(format!("{}:{}", span.service_name, span.operation_name));

            // Find child span with longest duration
            current_span = spans
                .iter()
                .filter(|s| s.parent_span_id.as_ref() == Some(&span.span_id))
                .max_by_key(|s| s.duration.unwrap_or(Duration::from_secs(0)));
        }

        path
    }

    fn find_parallelization_opportunities(
        &self,
        spans: &[Span],
    ) -> Vec<ParallelizationOpportunity> {
        let mut opportunities = Vec::new();

        // Look for sequential spans that could be parallelized
        for span in spans {
            if let Some(span_id) = &span.parent_span_id {
                let siblings: Vec<&Span> = spans
                    .iter()
                    .filter(|s| {
                        s.parent_span_id.as_ref() == Some(span_id) && s.span_id != span.span_id
                    })
                    .collect();

                if siblings.len() > 1 {
                    let total_sequential_time: Duration = siblings
                        .iter()
                        .map(|s| s.duration.unwrap_or(Duration::from_secs(0)))
                        .sum();
                    let max_individual_time = siblings
                        .iter()
                        .map(|s| s.duration.unwrap_or(Duration::from_secs(0)))
                        .max()
                        .unwrap_or(Duration::from_secs(0));

                    if total_sequential_time > max_individual_time {
                        let potential_savings = total_sequential_time - max_individual_time;
                        opportunities.push(ParallelizationOpportunity {
                            description: format!(
                                "Parallelize {} operations under {}",
                                siblings.len(),
                                span.operation_name
                            ),
                            potential_savings,
                            services: siblings.iter().map(|s| s.service_name.clone()).collect(),
                            confidence: 0.8,
                        });
                    }
                }
            }
        }

        opportunities
    }

    fn identify_trace_bottlenecks(
        &self,
        spans: &[Span],
        total_duration: Duration,
    ) -> Vec<TracingBottleneck> {
        let mut bottlenecks = Vec::new();

        for span in spans {
            if let Some(duration) = span.duration {
                let percentage =
                    (duration.as_millis() as f64 / total_duration.as_millis() as f64) * 100.0;

                if percentage > 20.0 {
                    // Spans taking more than 20% of total time
                    let bottleneck_type =
                        self.classify_bottleneck_type(&span.operation_name, &span.attributes);
                    let recommendations =
                        self.generate_bottleneck_recommendations(&bottleneck_type, span);

                    bottlenecks.push(TracingBottleneck {
                        component: format!("{}:{}", span.service_name, span.operation_name),
                        duration,
                        percentage_of_trace: percentage,
                        bottleneck_type,
                        recommendations,
                    });
                }
            }
        }

        bottlenecks.sort_by(|a, b| {
            b.percentage_of_trace
                .partial_cmp(&a.percentage_of_trace)
                .unwrap()
        });
        bottlenecks
    }

    fn classify_bottleneck_type(
        &self,
        operation_name: &str,
        attributes: &HashMap<String, String>,
    ) -> TracingBottleneckType {
        let operation_lower = operation_name.to_lowercase();

        if operation_lower.contains("query") || operation_lower.contains("sparql") {
            TracingBottleneckType::DatabaseQuery
        } else if operation_lower.contains("auth") || operation_lower.contains("login") {
            TracingBottleneckType::Authentication
        } else if operation_lower.contains("cache") {
            TracingBottleneckType::Caching
        } else if operation_lower.contains("network") || operation_lower.contains("http") {
            TracingBottleneckType::NetworkLatency
        } else if attributes
            .values()
            .any(|v| v.contains("large") || v.contains("transfer"))
        {
            TracingBottleneckType::LargeDataTransfer
        } else {
            TracingBottleneckType::SlowService
        }
    }

    fn generate_bottleneck_recommendations(
        &self,
        bottleneck_type: &TracingBottleneckType,
        span: &Span,
    ) -> Vec<String> {
        match bottleneck_type {
            TracingBottleneckType::DatabaseQuery => vec![
                "Consider query optimization or indexing".to_string(),
                "Evaluate query caching strategies".to_string(),
                "Check database connection pooling".to_string(),
            ],
            TracingBottleneckType::NetworkLatency => vec![
                "Implement request batching".to_string(),
                "Consider service co-location".to_string(),
                "Optimize connection pooling".to_string(),
            ],
            TracingBottleneckType::SlowService => vec![
                format!("Investigate performance issues in {}", span.service_name),
                "Consider horizontal scaling".to_string(),
                "Profile service for optimization opportunities".to_string(),
            ],
            TracingBottleneckType::LargeDataTransfer => vec![
                "Implement result compression".to_string(),
                "Consider result streaming".to_string(),
                "Optimize data serialization".to_string(),
            ],
            TracingBottleneckType::Authentication => vec![
                "Implement authentication caching".to_string(),
                "Consider token-based authentication".to_string(),
                "Optimize authentication service performance".to_string(),
            ],
            TracingBottleneckType::Caching => vec![
                "Review cache hit rates and policies".to_string(),
                "Consider cache warming strategies".to_string(),
                "Optimize cache key design".to_string(),
            ],
            TracingBottleneckType::SerialExecution => vec![
                "Identify parallelization opportunities".to_string(),
                "Consider asynchronous processing".to_string(),
                "Optimize execution ordering".to_string(),
            ],
        }
    }

    fn trace_error_propagation(&self, spans: &[Span]) -> Option<Vec<String>> {
        let error_spans: Vec<&Span> = spans
            .iter()
            .filter(|s| matches!(s.status, SpanStatus::Error(_)))
            .collect();

        if error_spans.is_empty() {
            return None;
        }

        // Build error propagation path
        let mut error_path = Vec::new();
        for error_span in error_spans {
            error_path.push(format!(
                "{}:{}",
                error_span.service_name, error_span.operation_name
            ));
        }

        Some(error_path)
    }
}

impl Default for DistributedTracingManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_trace_creation() {
        let tracer = DistributedTracingManager::new();
        let query_context = QueryContext {
            query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            query_type: "SPARQL".to_string(),
            user_id: Some("user123".to_string()),
            session_id: Some("session456".to_string()),
            priority: QueryPriority::Normal,
        };

        let trace = tracer.start_trace(query_context).await.unwrap();
        assert!(!trace.trace_id.is_empty());
        assert_eq!(trace.query_context.query_type, "SPARQL");
    }

    #[tokio::test]
    async fn test_span_creation_and_completion() {
        let tracer = DistributedTracingManager::new();
        let query_context = QueryContext {
            query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            query_type: "SPARQL".to_string(),
            user_id: None,
            session_id: None,
            priority: QueryPriority::Normal,
        };

        let trace = tracer.start_trace(query_context).await.unwrap();
        let span = tracer
            .create_span(&trace, "test_operation", "test_service", None)
            .await
            .unwrap();

        assert!(!span.span_id.is_empty());
        assert_eq!(span.operation_name, "test_operation");
        assert_eq!(span.service_name, "test_service");

        let result = tracer.finish_span(&span.span_id, SpanStatus::Ok).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_trace_analysis() {
        let tracer = DistributedTracingManager::new();
        let query_context = QueryContext {
            query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            query_type: "SPARQL".to_string(),
            user_id: None,
            session_id: None,
            priority: QueryPriority::Normal,
        };

        let trace = tracer.start_trace(query_context).await.unwrap();
        let span1 = tracer
            .create_span(&trace, "query_parsing", "parser_service", None)
            .await
            .unwrap();
        let span2 = tracer
            .create_span(
                &trace,
                "query_execution",
                "executor_service",
                Some(span1.span_id.clone()),
            )
            .await
            .unwrap();

        tracer
            .finish_span(&span1.span_id, SpanStatus::Ok)
            .await
            .unwrap();
        tracer
            .finish_span(&span2.span_id, SpanStatus::Ok)
            .await
            .unwrap();

        let analysis = tracer.complete_trace(&trace.trace_id).await.unwrap();
        assert_eq!(analysis.trace_id, trace.trace_id);
        assert!(!analysis.service_breakdown.is_empty());
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let tracer = DistributedTracingManager::new();
        let initial_metrics = tracer.get_metrics().await;
        assert_eq!(initial_metrics.total_traces, 0);
        assert_eq!(initial_metrics.total_spans, 0);

        let query_context = QueryContext {
            query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            query_type: "SPARQL".to_string(),
            user_id: None,
            session_id: None,
            priority: QueryPriority::Normal,
        };

        let trace = tracer.start_trace(query_context).await.unwrap();
        let _span = tracer
            .create_span(&trace, "test_operation", "test_service", None)
            .await
            .unwrap();

        let metrics = tracer.get_metrics().await;
        assert_eq!(metrics.total_traces, 1);
        assert_eq!(metrics.total_spans, 1);
    }
}
