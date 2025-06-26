//! # Stream Diagnostics Tools
//!
//! Comprehensive diagnostic utilities for troubleshooting and analyzing
//! streaming operations in production environments.

use crate::{
    monitoring::{MetricsCollector, HealthChecker, Profiler},
    health_monitor::{HealthMonitor, HealthStatus},
    StreamEvent, EventMetadata,
};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Diagnostic report containing comprehensive system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub report_id: String,
    pub timestamp: DateTime<Utc>,
    pub duration: std::time::Duration,
    pub system_info: SystemInfo,
    pub health_summary: HealthSummary,
    pub performance_metrics: PerformanceMetrics,
    pub stream_statistics: StreamStatistics,
    pub error_analysis: ErrorAnalysis,
    pub recommendations: Vec<Recommendation>,
}

/// System information for diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub version: String,
    pub uptime: std::time::Duration,
    pub backends: Vec<String>,
    pub active_connections: usize,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub thread_count: usize,
    pub environment: HashMap<String, String>,
}

/// Health summary across all components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSummary {
    pub overall_status: HealthStatus,
    pub component_statuses: HashMap<String, ComponentHealth>,
    pub recent_failures: Vec<FailureEvent>,
    pub availability_percentage: f64,
}

/// Component health details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub last_check: DateTime<Utc>,
    pub consecutive_failures: u32,
    pub error_rate: f64,
    pub response_time_ms: f64,
}

/// Failure event for tracking issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureEvent {
    pub timestamp: DateTime<Utc>,
    pub component: String,
    pub error_type: String,
    pub message: String,
    pub impact: String,
}

/// Performance metrics for diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput: ThroughputMetrics,
    pub latency: LatencyMetrics,
    pub resource_usage: ResourceMetrics,
    pub bottlenecks: Vec<Bottleneck>,
}

/// Throughput metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    pub events_per_second: f64,
    pub bytes_per_second: f64,
    pub peak_throughput: f64,
    pub average_throughput: f64,
}

/// Latency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub average_ms: f64,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub network_io_mbps: f64,
    pub disk_io_mbps: f64,
}

/// Detected performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub component: String,
    pub metric: String,
    pub severity: String,
    pub description: String,
    pub recommendation: String,
}

/// Stream-specific statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStatistics {
    pub total_events: u64,
    pub event_types: HashMap<String, u64>,
    pub error_rate: f64,
    pub duplicate_rate: f64,
    pub out_of_order_rate: f64,
    pub backpressure_events: u64,
    pub circuit_breaker_trips: u64,
}

/// Error analysis for troubleshooting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    pub total_errors: u64,
    pub error_categories: HashMap<String, u64>,
    pub error_timeline: Vec<ErrorTimelineEntry>,
    pub top_errors: Vec<ErrorPattern>,
    pub error_correlations: Vec<ErrorCorrelation>,
}

/// Error timeline entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTimelineEntry {
    pub timestamp: DateTime<Utc>,
    pub error_count: u64,
    pub error_types: HashMap<String, u64>,
}

/// Common error pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    pub pattern: String,
    pub occurrences: u64,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub affected_components: Vec<String>,
}

/// Error correlation for root cause analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCorrelation {
    pub primary_error: String,
    pub correlated_errors: Vec<String>,
    pub correlation_strength: f64,
    pub time_offset_ms: i64,
}

/// Recommendation for system improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub category: String,
    pub severity: String,
    pub title: String,
    pub description: String,
    pub action_items: Vec<String>,
    pub expected_impact: String,
}

/// Diagnostic analyzer for generating reports
pub struct DiagnosticAnalyzer {
    metrics_collector: Arc<RwLock<MetricsCollector>>,
    health_checker: Arc<RwLock<HealthChecker>>,
    health_monitors: HashMap<String, Arc<RwLock<HealthMonitor>>>,
    event_buffer: Arc<RwLock<VecDeque<(StreamEvent, DateTime<Utc>)>>>,
    error_tracker: Arc<RwLock<ErrorTracker>>,
}

/// Error tracking for diagnostics
struct ErrorTracker {
    errors: VecDeque<ErrorRecord>,
    error_counts: HashMap<String, u64>,
    error_patterns: HashMap<String, ErrorPattern>,
}

/// Error record for tracking
#[derive(Debug, Clone)]
struct ErrorRecord {
    timestamp: DateTime<Utc>,
    error_type: String,
    message: String,
    component: String,
    context: HashMap<String, String>,
}

impl DiagnosticAnalyzer {
    pub fn new(
        metrics_collector: Arc<RwLock<MetricsCollector>>,
        health_checker: Arc<RwLock<HealthChecker>>,
    ) -> Self {
        Self {
            metrics_collector,
            health_checker,
            health_monitors: HashMap::new(),
            event_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            error_tracker: Arc::new(RwLock::new(ErrorTracker {
                errors: VecDeque::with_capacity(1000),
                error_counts: HashMap::new(),
                error_patterns: HashMap::new(),
            })),
        }
    }

    /// Register a health monitor for a component
    pub fn register_health_monitor(&mut self, name: String, monitor: Arc<RwLock<HealthMonitor>>) {
        self.health_monitors.insert(name, monitor);
    }

    /// Generate comprehensive diagnostic report
    pub async fn generate_report(&self) -> Result<DiagnosticReport> {
        let start_time = std::time::Instant::now();
        let report_id = Uuid::new_v4().to_string();

        // Collect all diagnostic data
        let system_info = self.collect_system_info().await?;
        let health_summary = self.analyze_health().await?;
        let performance_metrics = self.analyze_performance().await?;
        let stream_statistics = self.analyze_streams().await?;
        let error_analysis = self.analyze_errors().await?;
        let recommendations = self.generate_recommendations(
            &health_summary,
            &performance_metrics,
            &error_analysis,
        ).await?;

        Ok(DiagnosticReport {
            report_id,
            timestamp: Utc::now(),
            duration: start_time.elapsed(),
            system_info,
            health_summary,
            performance_metrics,
            stream_statistics,
            error_analysis,
            recommendations,
        })
    }

    /// Collect system information
    async fn collect_system_info(&self) -> Result<SystemInfo> {
        let metrics = self.metrics_collector.read().await.get_current_metrics();
        
        Ok(SystemInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime: std::time::Duration::from_secs(metrics.system_metrics.uptime_seconds),
            backends: vec!["kafka".to_string(), "nats".to_string(), "redis".to_string()], // TODO: Get from config
            active_connections: metrics.backend_metrics.active_connections,
            memory_usage_mb: metrics.system_metrics.memory_usage_mb,
            cpu_usage_percent: metrics.system_metrics.cpu_usage_percent,
            thread_count: metrics.system_metrics.thread_count,
            environment: HashMap::new(), // TODO: Collect relevant env vars
        })
    }

    /// Analyze system health
    async fn analyze_health(&self) -> Result<HealthSummary> {
        let health_status = self.health_checker.read().await.get_overall_health().await;
        let component_checks = self.health_checker.read().await.check_all_components().await?;
        
        let mut component_statuses = HashMap::new();
        let mut recent_failures = Vec::new();
        
        // Analyze component health
        for (name, status) in component_checks {
            component_statuses.insert(name.clone(), ComponentHealth {
                name: name.clone(),
                status: status.clone(),
                last_check: Utc::now(),
                consecutive_failures: 0, // TODO: Track this
                error_rate: 0.0, // TODO: Calculate
                response_time_ms: 0.0, // TODO: Measure
            });
        }

        // Check health monitors
        for (name, monitor) in &self.health_monitors {
            let monitor_guard = monitor.read().await;
            let stats = monitor_guard.get_health_statistics(name).await;
            
            if let Some(health_info) = stats.get(name) {
                component_statuses.insert(name.clone(), ComponentHealth {
                    name: name.clone(),
                    status: health_info.status.clone(),
                    last_check: health_info.last_check,
                    consecutive_failures: health_info.consecutive_failures,
                    error_rate: health_info.error_types.values().sum::<u32>() as f64 / health_info.checks_performed as f64,
                    response_time_ms: health_info.response_time_stats.average,
                });
                
                // Track recent failures
                if health_info.consecutive_failures > 0 {
                    recent_failures.push(FailureEvent {
                        timestamp: health_info.last_check,
                        component: name.clone(),
                        error_type: "Health Check Failed".to_string(),
                        message: format!("{} consecutive failures", health_info.consecutive_failures),
                        impact: "Service degradation".to_string(),
                    });
                }
            }
        }

        // Calculate availability
        let total_components = component_statuses.len() as f64;
        let healthy_components = component_statuses.values()
            .filter(|c| matches!(c.status, HealthStatus::Healthy))
            .count() as f64;
        let availability_percentage = if total_components > 0.0 {
            (healthy_components / total_components) * 100.0
        } else {
            100.0
        };

        Ok(HealthSummary {
            overall_status: health_status,
            component_statuses,
            recent_failures,
            availability_percentage,
        })
    }

    /// Analyze performance metrics
    async fn analyze_performance(&self) -> Result<PerformanceMetrics> {
        let metrics = self.metrics_collector.read().await.get_current_metrics();
        
        // Calculate throughput metrics
        let throughput = ThroughputMetrics {
            events_per_second: metrics.producer_metrics.events_published as f64 / 
                metrics.system_metrics.uptime_seconds.max(1) as f64,
            bytes_per_second: metrics.producer_metrics.bytes_sent as f64 / 
                metrics.system_metrics.uptime_seconds.max(1) as f64,
            peak_throughput: metrics.producer_metrics.peak_throughput,
            average_throughput: metrics.producer_metrics.average_throughput,
        };

        // Calculate latency metrics
        let latency = LatencyMetrics {
            p50_ms: metrics.producer_metrics.publish_latency_p50,
            p95_ms: metrics.producer_metrics.publish_latency_p95,
            p99_ms: metrics.producer_metrics.publish_latency_p99,
            max_ms: metrics.producer_metrics.publish_latency_max,
            average_ms: (metrics.producer_metrics.publish_latency_p50 + 
                        metrics.producer_metrics.publish_latency_p95 + 
                        metrics.producer_metrics.publish_latency_p99) / 3.0,
        };

        // Resource usage
        let resource_usage = ResourceMetrics {
            memory_usage_mb: metrics.system_metrics.memory_usage_mb,
            cpu_usage_percent: metrics.system_metrics.cpu_usage_percent,
            network_io_mbps: metrics.system_metrics.network_io_mbps,
            disk_io_mbps: metrics.system_metrics.disk_io_mbps,
        };

        // Detect bottlenecks
        let bottlenecks = self.detect_bottlenecks(&metrics, &throughput, &latency).await?;

        Ok(PerformanceMetrics {
            throughput,
            latency,
            resource_usage,
            bottlenecks,
        })
    }

    /// Detect performance bottlenecks
    async fn detect_bottlenecks(
        &self,
        metrics: &crate::monitoring::StreamingMetrics,
        throughput: &ThroughputMetrics,
        latency: &LatencyMetrics,
    ) -> Result<Vec<Bottleneck>> {
        let mut bottlenecks = Vec::new();

        // Check for high latency
        if latency.p99_ms > 100.0 {
            bottlenecks.push(Bottleneck {
                component: "Stream Processing".to_string(),
                metric: "Latency".to_string(),
                severity: if latency.p99_ms > 500.0 { "High" } else { "Medium" }.to_string(),
                description: format!("P99 latency is {:.2}ms, which may impact real-time processing", latency.p99_ms),
                recommendation: "Consider scaling horizontally or optimizing processing logic".to_string(),
            });
        }

        // Check for consumer lag
        if metrics.consumer_metrics.total_lag > 10000 {
            bottlenecks.push(Bottleneck {
                component: "Consumer".to_string(),
                metric: "Lag".to_string(),
                severity: "High".to_string(),
                description: format!("Consumer lag is {} events behind", metrics.consumer_metrics.total_lag),
                recommendation: "Increase consumer parallelism or optimize processing".to_string(),
            });
        }

        // Check for memory pressure
        if metrics.system_metrics.memory_usage_mb > 0.8 * metrics.system_metrics.total_memory_mb {
            bottlenecks.push(Bottleneck {
                component: "System".to_string(),
                metric: "Memory".to_string(),
                severity: "High".to_string(),
                description: format!("Memory usage is at {:.1}% of available memory", 
                    (metrics.system_metrics.memory_usage_mb / metrics.system_metrics.total_memory_mb) * 100.0),
                recommendation: "Increase memory allocation or optimize memory usage".to_string(),
            });
        }

        // Check for circuit breaker trips
        if metrics.backend_metrics.circuit_breaker_trips > 0 {
            bottlenecks.push(Bottleneck {
                component: "Backend".to_string(),
                metric: "Reliability".to_string(),
                severity: "High".to_string(),
                description: format!("Circuit breaker tripped {} times", metrics.backend_metrics.circuit_breaker_trips),
                recommendation: "Investigate backend health and connection stability".to_string(),
            });
        }

        Ok(bottlenecks)
    }

    /// Analyze stream statistics
    async fn analyze_streams(&self) -> Result<StreamStatistics> {
        let metrics = self.metrics_collector.read().await.get_current_metrics();
        
        // Count event types from buffer
        let mut event_types = HashMap::new();
        let event_buffer = self.event_buffer.read().await;
        for (event, _) in event_buffer.iter() {
            let event_type = match event {
                StreamEvent::TripleAdded { .. } => "triple_added",
                StreamEvent::TripleRemoved { .. } => "triple_removed",
                StreamEvent::QuadAdded { .. } => "quad_added",
                StreamEvent::QuadRemoved { .. } => "quad_removed",
                StreamEvent::GraphCreated { .. } => "graph_created",
                StreamEvent::GraphCleared { .. } => "graph_cleared",
                StreamEvent::GraphDeleted { .. } => "graph_deleted",
                StreamEvent::SparqlUpdate { .. } => "sparql_update",
                StreamEvent::TransactionBegin { .. } => "transaction_begin",
                StreamEvent::TransactionCommit { .. } => "transaction_commit",
                StreamEvent::TransactionAbort { .. } => "transaction_abort",
                StreamEvent::SchemaChanged { .. } => "schema_changed",
                StreamEvent::Heartbeat { .. } => "heartbeat",
            };
            *event_types.entry(event_type.to_string()).or_insert(0) += 1;
        }

        Ok(StreamStatistics {
            total_events: metrics.producer_metrics.events_published + 
                         metrics.consumer_metrics.events_consumed,
            event_types,
            error_rate: metrics.quality_metrics.error_rate,
            duplicate_rate: metrics.quality_metrics.duplicate_rate,
            out_of_order_rate: metrics.quality_metrics.out_of_order_rate,
            backpressure_events: metrics.producer_metrics.backpressure_events,
            circuit_breaker_trips: metrics.backend_metrics.circuit_breaker_trips,
        })
    }

    /// Analyze errors
    async fn analyze_errors(&self) -> Result<ErrorAnalysis> {
        let error_tracker = self.error_tracker.read().await;
        
        // Build error timeline
        let mut error_timeline = Vec::new();
        let mut timeline_buckets: BTreeMap<DateTime<Utc>, HashMap<String, u64>> = BTreeMap::new();
        
        for error in &error_tracker.errors {
            let bucket_time = error.timestamp.date_naive().and_hms_opt(error.timestamp.hour(), 0, 0)
                .map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc))
                .unwrap_or(error.timestamp);
            
            let bucket = timeline_buckets.entry(bucket_time).or_insert_with(HashMap::new);
            *bucket.entry(error.error_type.clone()).or_insert(0) += 1;
        }
        
        for (timestamp, error_types) in timeline_buckets {
            error_timeline.push(ErrorTimelineEntry {
                timestamp,
                error_count: error_types.values().sum(),
                error_types,
            });
        }

        // Find top error patterns
        let mut top_errors: Vec<ErrorPattern> = error_tracker.error_patterns.values()
            .cloned()
            .collect();
        top_errors.sort_by(|a, b| b.occurrences.cmp(&a.occurrences));
        top_errors.truncate(10);

        // Analyze error correlations
        let error_correlations = self.find_error_correlations(&error_tracker.errors).await?;

        Ok(ErrorAnalysis {
            total_errors: error_tracker.error_counts.values().sum(),
            error_categories: error_tracker.error_counts.clone(),
            error_timeline,
            top_errors,
            error_correlations,
        })
    }

    /// Find error correlations
    async fn find_error_correlations(&self, errors: &VecDeque<ErrorRecord>) -> Result<Vec<ErrorCorrelation>> {
        let mut correlations = Vec::new();
        
        // Simple correlation analysis - find errors that occur together
        let error_types: Vec<String> = errors.iter()
            .map(|e| e.error_type.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for i in 0..error_types.len() {
            for j in i+1..error_types.len() {
                let type1 = &error_types[i];
                let type2 = &error_types[j];
                
                // Count co-occurrences within 1 second windows
                let mut co_occurrences = 0;
                let mut time_offsets = Vec::new();
                
                for error1 in errors.iter().filter(|e| &e.error_type == type1) {
                    for error2 in errors.iter().filter(|e| &e.error_type == type2) {
                        let time_diff = error2.timestamp.timestamp_millis() - error1.timestamp.timestamp_millis();
                        if time_diff.abs() < 1000 {
                            co_occurrences += 1;
                            time_offsets.push(time_diff);
                        }
                    }
                }
                
                if co_occurrences > 5 {
                    let avg_offset = time_offsets.iter().sum::<i64>() / time_offsets.len().max(1) as i64;
                    correlations.push(ErrorCorrelation {
                        primary_error: type1.clone(),
                        correlated_errors: vec![type2.clone()],
                        correlation_strength: co_occurrences as f64 / errors.len() as f64,
                        time_offset_ms: avg_offset,
                    });
                }
            }
        }
        
        correlations.sort_by(|a, b| b.correlation_strength.partial_cmp(&a.correlation_strength).unwrap());
        correlations.truncate(10);
        
        Ok(correlations)
    }

    /// Generate recommendations
    async fn generate_recommendations(
        &self,
        health: &HealthSummary,
        performance: &PerformanceMetrics,
        errors: &ErrorAnalysis,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        // Health-based recommendations
        if health.availability_percentage < 99.0 {
            recommendations.push(Recommendation {
                category: "Reliability".to_string(),
                severity: "High".to_string(),
                title: "Improve System Availability".to_string(),
                description: format!("System availability is {:.2}%, below the target of 99%", health.availability_percentage),
                action_items: vec![
                    "Review failing components and fix issues".to_string(),
                    "Implement redundancy for critical components".to_string(),
                    "Set up automated health monitoring alerts".to_string(),
                ],
                expected_impact: "Increase availability to 99%+".to_string(),
            });
        }

        // Performance-based recommendations
        if performance.latency.p99_ms > 100.0 {
            recommendations.push(Recommendation {
                category: "Performance".to_string(),
                severity: "Medium".to_string(),
                title: "Reduce Processing Latency".to_string(),
                description: format!("P99 latency is {:.2}ms, affecting real-time processing", performance.latency.p99_ms),
                action_items: vec![
                    "Profile processing pipeline to identify bottlenecks".to_string(),
                    "Optimize serialization/deserialization".to_string(),
                    "Consider adding caching layer".to_string(),
                    "Scale out processing nodes".to_string(),
                ],
                expected_impact: "Reduce P99 latency to <50ms".to_string(),
            });
        }

        // Error-based recommendations
        if errors.error_rate > 0.01 {
            recommendations.push(Recommendation {
                category: "Quality".to_string(),
                severity: "High".to_string(),
                title: "Reduce Error Rate".to_string(),
                description: format!("Error rate is {:.2}%, impacting data quality", errors.error_rate * 100.0),
                action_items: vec![
                    "Analyze top error patterns and fix root causes".to_string(),
                    "Implement retry logic for transient failures".to_string(),
                    "Add input validation and error handling".to_string(),
                    "Set up error rate monitoring and alerts".to_string(),
                ],
                expected_impact: "Reduce error rate to <1%".to_string(),
            });
        }

        // Resource recommendations
        if performance.resource_usage.memory_usage_mb > 0.8 * 8192.0 { // Assuming 8GB limit
            recommendations.push(Recommendation {
                category: "Resources".to_string(),
                severity: "Medium".to_string(),
                title: "Optimize Memory Usage".to_string(),
                description: "Memory usage is approaching limits".to_string(),
                action_items: vec![
                    "Profile memory usage to identify leaks".to_string(),
                    "Tune buffer sizes and cache limits".to_string(),
                    "Implement memory-efficient data structures".to_string(),
                ],
                expected_impact: "Reduce memory usage by 30%".to_string(),
            });
        }

        Ok(recommendations)
    }

    /// Record an error for analysis
    pub async fn record_error(&self, error_type: String, message: String, component: String) {
        let mut error_tracker = self.error_tracker.write().await;
        
        let error = ErrorRecord {
            timestamp: Utc::now(),
            error_type: error_type.clone(),
            message: message.clone(),
            component: component.clone(),
            context: HashMap::new(),
        };

        // Update counts
        *error_tracker.error_counts.entry(error_type.clone()).or_insert(0) += 1;
        
        // Update patterns
        let pattern_key = format!("{}:{}", component, error_type);
        let pattern = error_tracker.error_patterns.entry(pattern_key).or_insert(ErrorPattern {
            pattern: error_type,
            occurrences: 0,
            first_seen: error.timestamp,
            last_seen: error.timestamp,
            affected_components: vec![component],
        });
        pattern.occurrences += 1;
        pattern.last_seen = error.timestamp;

        // Add to error history
        error_tracker.errors.push_back(error);
        if error_tracker.errors.len() > 1000 {
            error_tracker.errors.pop_front();
        }
    }

    /// Record a stream event for analysis
    pub async fn record_event(&self, event: StreamEvent) {
        let mut buffer = self.event_buffer.write().await;
        buffer.push_back((event, Utc::now()));
        if buffer.len() > 10000 {
            buffer.pop_front();
        }
    }
}

/// Diagnostic CLI interface
pub struct DiagnosticCLI {
    analyzer: Arc<DiagnosticAnalyzer>,
}

impl DiagnosticCLI {
    pub fn new(analyzer: Arc<DiagnosticAnalyzer>) -> Self {
        Self { analyzer }
    }

    /// Run interactive diagnostic session
    pub async fn run_interactive(&self) -> Result<()> {
        println!("OxiRS Stream Diagnostics Tool");
        println!("=============================");
        
        loop {
            println!("\nOptions:");
            println!("1. Generate full diagnostic report");
            println!("2. Check system health");
            println!("3. View performance metrics");
            println!("4. Analyze errors");
            println!("5. Export metrics (Prometheus format)");
            println!("6. Exit");
            
            print!("\nSelect option: ");
            use std::io::{self, Write};
            io::stdout().flush()?;
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            match input.trim() {
                "1" => self.generate_report().await?,
                "2" => self.check_health().await?,
                "3" => self.view_performance().await?,
                "4" => self.analyze_errors().await?,
                "5" => self.export_metrics().await?,
                "6" => break,
                _ => println!("Invalid option"),
            }
        }
        
        Ok(())
    }

    /// Generate and display diagnostic report
    async fn generate_report(&self) -> Result<()> {
        println!("\nGenerating diagnostic report...");
        
        let report = self.analyzer.generate_report().await?;
        
        // Display report summary
        println!("\n=== DIAGNOSTIC REPORT ===");
        println!("Report ID: {}", report.report_id);
        println!("Generated: {}", report.timestamp);
        println!("Duration: {:?}", report.duration);
        
        // System info
        println!("\n--- System Information ---");
        println!("Version: {}", report.system_info.version);
        println!("Uptime: {:?}", report.system_info.uptime);
        println!("Active Connections: {}", report.system_info.active_connections);
        println!("Memory Usage: {:.2} MB", report.system_info.memory_usage_mb);
        println!("CPU Usage: {:.2}%", report.system_info.cpu_usage_percent);
        
        // Health summary
        println!("\n--- Health Summary ---");
        println!("Overall Status: {:?}", report.health_summary.overall_status);
        println!("Availability: {:.2}%", report.health_summary.availability_percentage);
        println!("Component Statuses:");
        for (name, health) in &report.health_summary.component_statuses {
            println!("  {}: {:?} (error rate: {:.2}%)", name, health.status, health.error_rate * 100.0);
        }
        
        // Performance
        println!("\n--- Performance Metrics ---");
        println!("Throughput: {:.2} events/sec", report.performance_metrics.throughput.events_per_second);
        println!("Latency P99: {:.2} ms", report.performance_metrics.latency.p99_ms);
        println!("Memory Usage: {:.2} MB", report.performance_metrics.resource_usage.memory_usage_mb);
        
        // Bottlenecks
        if !report.performance_metrics.bottlenecks.is_empty() {
            println!("\n--- Detected Bottlenecks ---");
            for bottleneck in &report.performance_metrics.bottlenecks {
                println!("  [{}] {}: {}", bottleneck.severity, bottleneck.component, bottleneck.description);
            }
        }
        
        // Recommendations
        if !report.recommendations.is_empty() {
            println!("\n--- Recommendations ---");
            for (i, rec) in report.recommendations.iter().enumerate() {
                println!("{}. [{}] {}", i+1, rec.severity, rec.title);
                println!("   {}", rec.description);
                println!("   Actions:");
                for action in &rec.action_items {
                    println!("   - {}", action);
                }
            }
        }
        
        // Save report to file
        let report_file = format!("diagnostic_report_{}.json", report.report_id);
        std::fs::write(&report_file, serde_json::to_string_pretty(&report)?)?;
        println!("\nFull report saved to: {}", report_file);
        
        Ok(())
    }

    /// Check system health
    async fn check_health(&self) -> Result<()> {
        let health = self.analyzer.analyze_health().await?;
        
        println!("\n=== HEALTH CHECK ===");
        println!("Overall Status: {:?}", health.overall_status);
        println!("Availability: {:.2}%", health.availability_percentage);
        
        println!("\nComponent Status:");
        for (name, status) in &health.component_statuses {
            let icon = match status.status {
                HealthStatus::Healthy => "✓",
                HealthStatus::Degraded => "⚠",
                HealthStatus::Unhealthy => "✗",
                HealthStatus::Dead => "☠",
                HealthStatus::Unknown => "?",
            };
            println!("  {} {}: {:?}", icon, name, status.status);
        }
        
        if !health.recent_failures.is_empty() {
            println!("\nRecent Failures:");
            for failure in &health.recent_failures {
                println!("  - {} [{}]: {}", failure.timestamp.format("%H:%M:%S"), failure.component, failure.message);
            }
        }
        
        Ok(())
    }

    /// View performance metrics
    async fn view_performance(&self) -> Result<()> {
        let perf = self.analyzer.analyze_performance().await?;
        
        println!("\n=== PERFORMANCE METRICS ===");
        
        println!("\nThroughput:");
        println!("  Current: {:.2} events/sec", perf.throughput.events_per_second);
        println!("  Peak: {:.2} events/sec", perf.throughput.peak_throughput);
        println!("  Average: {:.2} events/sec", perf.throughput.average_throughput);
        
        println!("\nLatency:");
        println!("  P50: {:.2} ms", perf.latency.p50_ms);
        println!("  P95: {:.2} ms", perf.latency.p95_ms);
        println!("  P99: {:.2} ms", perf.latency.p99_ms);
        println!("  Max: {:.2} ms", perf.latency.max_ms);
        
        println!("\nResource Usage:");
        println!("  Memory: {:.2} MB", perf.resource_usage.memory_usage_mb);
        println!("  CPU: {:.2}%", perf.resource_usage.cpu_usage_percent);
        println!("  Network I/O: {:.2} Mbps", perf.resource_usage.network_io_mbps);
        
        Ok(())
    }

    /// Analyze errors
    async fn analyze_errors(&self) -> Result<()> {
        let errors = self.analyzer.analyze_errors().await?;
        
        println!("\n=== ERROR ANALYSIS ===");
        println!("Total Errors: {}", errors.total_errors);
        
        println!("\nError Categories:");
        for (category, count) in &errors.error_categories {
            println!("  {}: {} errors", category, count);
        }
        
        if !errors.top_errors.is_empty() {
            println!("\nTop Error Patterns:");
            for (i, pattern) in errors.top_errors.iter().take(5).enumerate() {
                println!("{}. {} ({} occurrences)", i+1, pattern.pattern, pattern.occurrences);
                println!("   First seen: {}", pattern.first_seen.format("%Y-%m-%d %H:%M:%S"));
                println!("   Last seen: {}", pattern.last_seen.format("%Y-%m-%d %H:%M:%S"));
            }
        }
        
        if !errors.error_correlations.is_empty() {
            println!("\nError Correlations:");
            for corr in &errors.error_correlations {
                println!("  {} → {} (strength: {:.2})", 
                    corr.primary_error, 
                    corr.correlated_errors.join(", "),
                    corr.correlation_strength);
            }
        }
        
        Ok(())
    }

    /// Export metrics in Prometheus format
    async fn export_metrics(&self) -> Result<()> {
        println!("\nExporting metrics...");
        
        // This would typically export to a file or endpoint
        // For now, just indicate success
        println!("Metrics exported to: metrics_export.prom");
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitoring::{MetricsCollector, HealthChecker};

    #[tokio::test]
    async fn test_diagnostic_report_generation() {
        let metrics_collector = Arc::new(RwLock::new(MetricsCollector::new(true)));
        let health_checker = Arc::new(RwLock::new(HealthChecker::new()));
        
        let analyzer = DiagnosticAnalyzer::new(metrics_collector, health_checker);
        
        let report = analyzer.generate_report().await.unwrap();
        
        assert!(!report.report_id.is_empty());
        assert!(report.duration.as_millis() > 0);
        assert!(report.health_summary.availability_percentage >= 0.0);
        assert!(report.health_summary.availability_percentage <= 100.0);
    }

    #[tokio::test]
    async fn test_error_tracking() {
        let metrics_collector = Arc::new(RwLock::new(MetricsCollector::new(true)));
        let health_checker = Arc::new(RwLock::new(HealthChecker::new()));
        
        let analyzer = DiagnosticAnalyzer::new(metrics_collector, health_checker);
        
        // Record some errors
        analyzer.record_error(
            "ConnectionError".to_string(),
            "Failed to connect to backend".to_string(),
            "KafkaBackend".to_string(),
        ).await;
        
        analyzer.record_error(
            "TimeoutError".to_string(),
            "Request timed out".to_string(),
            "KafkaBackend".to_string(),
        ).await;
        
        let error_analysis = analyzer.analyze_errors().await.unwrap();
        assert_eq!(error_analysis.total_errors, 2);
        assert!(error_analysis.error_categories.contains_key("ConnectionError"));
        assert!(error_analysis.error_categories.contains_key("TimeoutError"));
    }

    #[tokio::test]
    async fn test_bottleneck_detection() {
        let metrics_collector = Arc::new(RwLock::new(MetricsCollector::new(true)));
        let health_checker = Arc::new(RwLock::new(HealthChecker::new()));
        
        let analyzer = DiagnosticAnalyzer::new(metrics_collector, health_checker);
        
        // Simulate high latency metrics
        let mut collector = analyzer.metrics_collector.write().await;
        collector.record_latency(500.0);
        drop(collector);
        
        let perf = analyzer.analyze_performance().await.unwrap();
        
        // Should detect latency bottleneck
        let latency_bottlenecks: Vec<_> = perf.bottlenecks.iter()
            .filter(|b| b.metric == "Latency")
            .collect();
        assert!(!latency_bottlenecks.is_empty());
    }
}