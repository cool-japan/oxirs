//! Performance monitoring and Beta.2 statistics endpoints
//!
//! This module provides HTTP endpoints for monitoring Beta.2 performance features:
//! - Concurrency manager statistics
//! - Memory pool usage and efficiency
//! - Batch executor metrics
//! - Stream manager statistics
//! - Dataset manager status

use crate::server::AppState;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::instrument;

/// Combined performance statistics from all Beta.2 modules
#[derive(Debug, Serialize)]
pub struct PerformanceStats {
    pub concurrency: Option<ConcurrencyStats>,
    pub memory: Option<MemoryStats>,
    pub batching: Option<BatchingStats>,
    pub streaming: Option<StreamingStats>,
    pub datasets: Option<DatasetStats>,
    pub system: SystemStats,
}

/// Concurrency manager statistics
#[derive(Debug, Serialize)]
pub struct ConcurrencyStats {
    pub active_requests: usize,
    pub queued_requests: usize,
    pub total_requests: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub rejected_requests: u64,
    pub average_wait_time_ms: f64,
    pub average_execution_time_ms: f64,
    pub current_load: f64,
}

/// Memory pool statistics
#[derive(Debug, Serialize)]
pub struct MemoryStats {
    pub total_allocated: u64,
    pub total_deallocated: u64,
    pub current_usage: u64,
    pub peak_usage: u64,
    pub active_objects: usize,
    pub pooled_objects: usize,
    pub pool_hit_ratio: f64,
    pub memory_pressure: f64,
    pub gc_runs: u64,
    pub last_gc_duration_ms: u64,
}

/// Batch executor statistics
#[derive(Debug, Serialize)]
pub struct BatchingStats {
    pub total_batches: u64,
    pub total_queries: u64,
    pub average_batch_size: f64,
    pub average_wait_time_ms: f64,
    pub average_execution_time_ms: f64,
    pub parallel_efficiency: f64,
    pub queries_per_second: f64,
}

/// Stream manager statistics
#[derive(Debug, Serialize)]
pub struct StreamingStats {
    pub total_bytes: u64,
    pub total_chunks: u64,
    pub total_rows: u64,
    pub compression_ratio: f64,
    pub average_chunk_size: f64,
    pub throughput_mbps: f64,
    pub active_streams: usize,
    pub backpressure_events: u64,
}

/// Dataset manager statistics
#[derive(Debug, Serialize)]
pub struct DatasetStats {
    pub total_datasets: usize,
    pub total_snapshots: usize,
    pub active_operations: usize,
    pub pending_backups: usize,
}

/// System-level statistics
#[derive(Debug, Serialize)]
pub struct SystemStats {
    pub uptime_seconds: u64,
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub worker_threads: usize,
}

/// GET /performance/stats - Get comprehensive performance statistics
#[instrument(skip(state))]
pub async fn get_performance_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<PerformanceStats>, StatusCode> {
    // Collect concurrency stats
    let concurrency_stats = if let Some(ref manager) = state.concurrency_manager {
        let stats = manager.get_stats().await;
        Some(ConcurrencyStats {
            active_requests: stats.active_requests,
            queued_requests: stats.queued_requests,
            total_requests: stats.total_requests,
            completed_requests: stats.completed_requests,
            failed_requests: stats.failed_requests,
            rejected_requests: stats.rejected_requests,
            average_wait_time_ms: stats.average_wait_time_ms,
            average_execution_time_ms: stats.average_execution_time_ms,
            current_load: stats.current_load,
        })
    } else {
        None
    };

    // Collect memory stats
    let memory_stats = if let Some(ref manager) = state.memory_manager {
        let stats = manager.get_stats().await;
        Some(MemoryStats {
            total_allocated: stats.total_allocated,
            total_deallocated: stats.total_deallocated,
            current_usage: stats.current_usage,
            peak_usage: stats.peak_usage,
            active_objects: stats.active_objects,
            pooled_objects: stats.pooled_objects,
            pool_hit_ratio: stats.pool_hit_ratio,
            memory_pressure: stats.memory_pressure,
            gc_runs: stats.gc_runs,
            last_gc_duration_ms: stats.last_gc_duration_ms,
        })
    } else {
        None
    };

    // Collect batch executor stats
    let batching_stats = if let Some(ref executor) = state.batch_executor {
        let stats = executor.get_stats().await;
        Some(BatchingStats {
            total_batches: stats.total_batches,
            total_queries: stats.total_queries,
            average_batch_size: stats.average_batch_size,
            average_wait_time_ms: stats.average_wait_time_ms,
            average_execution_time_ms: stats.average_execution_time_ms,
            parallel_efficiency: stats.parallel_efficiency,
            queries_per_second: stats.queries_per_second,
        })
    } else {
        None
    };

    // Collect stream manager stats
    let streaming_stats = if let Some(ref manager) = state.stream_manager {
        let stats = manager.get_stats().await;
        Some(StreamingStats {
            total_bytes: stats.total_bytes,
            total_chunks: stats.total_chunks,
            total_rows: stats.total_rows,
            compression_ratio: stats.compression_ratio,
            average_chunk_size: stats.average_chunk_size,
            throughput_mbps: stats.throughput_mbps,
            active_streams: stats.active_streams,
            backpressure_events: stats.backpressure_events,
        })
    } else {
        None
    };

    // Collect dataset manager stats
    let dataset_stats = if let Some(ref manager) = state.dataset_manager {
        let stats = manager.get_stats().await;
        Some(DatasetStats {
            total_datasets: stats.total_datasets,
            total_snapshots: stats.total_snapshots,
            active_operations: stats.active_operations,
            pending_backups: stats.pending_backups,
        })
    } else {
        None
    };

    // Get system stats
    let mut system = state.system_monitor.lock();
    system.refresh_all();

    // Calculate CPU usage (average across all CPUs)
    let cpu_usage_percent = system.global_cpu_usage() as f64;

    // Calculate memory usage in MB
    let used_memory = system.used_memory() as f64 / (1024.0 * 1024.0);

    let system_stats = SystemStats {
        uptime_seconds: state.startup_time.elapsed().as_secs(),
        cpu_usage_percent,
        memory_usage_mb: used_memory,
        worker_threads: num_cpus::get(),
    };

    Ok(Json(PerformanceStats {
        concurrency: concurrency_stats,
        memory: memory_stats,
        batching: batching_stats,
        streaming: streaming_stats,
        datasets: dataset_stats,
        system: system_stats,
    }))
}

/// GET /performance/memory - Get detailed memory statistics
#[instrument(skip(state))]
pub async fn get_memory_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<MemoryStats>, StatusCode> {
    if let Some(ref manager) = state.memory_manager {
        let stats = manager.get_stats().await;
        Ok(Json(MemoryStats {
            total_allocated: stats.total_allocated,
            total_deallocated: stats.total_deallocated,
            current_usage: stats.current_usage,
            peak_usage: stats.peak_usage,
            active_objects: stats.active_objects,
            pooled_objects: stats.pooled_objects,
            pool_hit_ratio: stats.pool_hit_ratio,
            memory_pressure: stats.memory_pressure,
            gc_runs: stats.gc_runs,
            last_gc_duration_ms: stats.last_gc_duration_ms,
        }))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// GET /performance/concurrency - Get detailed concurrency statistics
#[instrument(skip(state))]
pub async fn get_concurrency_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ConcurrencyStats>, StatusCode> {
    if let Some(ref manager) = state.concurrency_manager {
        let stats = manager.get_stats().await;
        Ok(Json(ConcurrencyStats {
            active_requests: stats.active_requests,
            queued_requests: stats.queued_requests,
            total_requests: stats.total_requests,
            completed_requests: stats.completed_requests,
            failed_requests: stats.failed_requests,
            rejected_requests: stats.rejected_requests,
            average_wait_time_ms: stats.average_wait_time_ms,
            average_execution_time_ms: stats.average_execution_time_ms,
            current_load: stats.current_load,
        }))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// POST /performance/memory/gc - Trigger manual garbage collection
#[instrument(skip(state))]
pub async fn trigger_gc(State(state): State<Arc<AppState>>) -> Result<StatusCode, StatusCode> {
    if let Some(ref manager) = state.memory_manager {
        manager
            .force_gc()
            .await
            .map(|_| StatusCode::OK)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Health check response for Beta.2 features
#[derive(Debug, Serialize)]
pub struct Beta2HealthCheck {
    pub concurrency_manager: bool,
    pub memory_manager: bool,
    pub batch_executor: bool,
    pub stream_manager: bool,
    pub dataset_manager: bool,
    pub all_healthy: bool,
}

/// GET /performance/health - Check Beta.2 module health
#[instrument(skip(state))]
pub async fn beta2_health_check(State(state): State<Arc<AppState>>) -> Json<Beta2HealthCheck> {
    let concurrency_ok = state.concurrency_manager.is_some();
    let memory_ok = state.memory_manager.is_some();
    let batch_ok = state.batch_executor.is_some();
    let stream_ok = state.stream_manager.is_some();
    let dataset_ok = state.dataset_manager.is_some();

    let all_healthy = concurrency_ok && memory_ok && batch_ok && stream_ok && dataset_ok;

    Json(Beta2HealthCheck {
        concurrency_manager: concurrency_ok,
        memory_manager: memory_ok,
        batch_executor: batch_ok,
        stream_manager: stream_ok,
        dataset_manager: dataset_ok,
        all_healthy,
    })
}

// RC.1 Performance Profiler Endpoints

/// Performance report from the profiler
#[derive(Debug, Serialize)]
pub struct ProfilerReport {
    pub enabled: bool,
    pub total_profiles: usize,
    pub query_profiles: Vec<QueryProfile>,
    pub operation_stats: Vec<OperationStat>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct QueryProfile {
    pub query_id: String,
    pub execution_time_ms: f64,
    pub phases: Vec<String>,
    pub bottlenecks: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct OperationStat {
    pub operation: String,
    pub count: u64,
    pub avg_duration_ms: f64,
    pub min_duration_ms: f64,
    pub max_duration_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

/// GET /$/profiler/report - Generate comprehensive performance report
#[instrument(skip(state))]
pub async fn profiler_report_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ProfilerReport>, StatusCode> {
    if let Some(ref profiler) = state.performance_profiler {
        // Generate report for last hour (3600 seconds)
        match profiler.generate_report(3600).await {
            Ok(report) => {
                Ok(Json(ProfilerReport {
                    enabled: true,
                    total_profiles: report.slow_queries.len(),
                    query_profiles: report
                        .slow_queries
                        .into_iter()
                        .map(|p| {
                            let bottlenecks = detect_bottlenecks(&p.phases);
                            QueryProfile {
                                query_id: p.id.clone(),
                                execution_time_ms: p.execution_time_ms as f64,
                                phases: p.phases.iter().map(|ph| ph.name.clone()).collect(),
                                bottlenecks,
                            }
                        })
                        .collect(),
                    operation_stats: report
                        .top_operations
                        .into_iter()
                        .map(|op| OperationStat {
                            operation: op.operation,
                            count: op.execution_count,
                            avg_duration_ms: op.avg_time_ms,
                            min_duration_ms: op.min_time_ms as f64,
                            max_duration_ms: op.max_time_ms as f64,
                            p50_ms: op.avg_time_ms, // Approximate
                            p95_ms: op.max_time_ms as f64 * 0.95,
                            p99_ms: op.max_time_ms as f64 * 0.99,
                        })
                        .collect(),
                    recommendations: report.recommendations,
                }))
            }
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Query statistics response
#[derive(Debug, Serialize)]
pub struct QueryStats {
    pub total_queries: u64,
    pub avg_execution_time_ms: f64,
    pub slow_queries: Vec<SlowQuery>,
}

#[derive(Debug, Serialize)]
pub struct SlowQuery {
    pub query_id: String,
    pub execution_time_ms: f64,
    pub timestamp: String,
}

/// GET /$/profiler/query-stats - Get query execution statistics
#[instrument(skip(state))]
pub async fn profiler_query_stats_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<QueryStats>, StatusCode> {
    if let Some(ref profiler) = state.performance_profiler {
        let stats_map = profiler.get_query_statistics().await;

        let total_queries = stats_map.get("total_queries").copied().unwrap_or(0);
        let avg_execution_time_ms =
            stats_map.get("avg_execution_time_ms").copied().unwrap_or(0) as f64;

        // Get slow queries from recent report
        match profiler.generate_report(3600).await {
            Ok(report) => {
                let slow_queries = report
                    .slow_queries
                    .into_iter()
                    .map(|q| SlowQuery {
                        query_id: q.id,
                        execution_time_ms: q.execution_time_ms as f64,
                        timestamp: q.timestamp.to_rfc3339(),
                    })
                    .collect();

                Ok(Json(QueryStats {
                    total_queries,
                    avg_execution_time_ms,
                    slow_queries,
                }))
            }
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// POST /$/profiler/clear - Clear profiler historical data (placeholder)
#[instrument(skip(state))]
pub async fn profiler_reset_handler(
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, StatusCode> {
    if state.performance_profiler.is_some() {
        // Note: PerformanceProfiler doesn't have a reset method
        // Data is automatically cleaned up based on retention policy
        Ok(StatusCode::NOT_IMPLEMENTED)
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Detect bottlenecks in query execution phases
fn detect_bottlenecks(phases: &[crate::performance_profiler::ExecutionPhase]) -> Vec<String> {
    if phases.is_empty() {
        return Vec::new();
    }

    let mut bottlenecks = Vec::new();

    // Calculate total execution time
    let total_time: u64 = phases.iter().map(|p| p.duration_ms).sum();
    if total_time == 0 {
        return bottlenecks;
    }

    // Calculate average phase duration
    let avg_duration = total_time as f64 / phases.len() as f64;

    // Calculate total CPU time and memory
    let total_cpu: u64 = phases.iter().map(|p| p.cpu_time_ms).sum();
    let total_memory: u64 = phases.iter().map(|p| p.memory_bytes).sum();

    for phase in phases {
        let duration_pct = (phase.duration_ms as f64 / total_time as f64) * 100.0;
        let cpu_pct = if total_cpu > 0 {
            (phase.cpu_time_ms as f64 / total_cpu as f64) * 100.0
        } else {
            0.0
        };
        let memory_pct = if total_memory > 0 {
            (phase.memory_bytes as f64 / total_memory as f64) * 100.0
        } else {
            0.0
        };

        // Detect time bottlenecks (phases taking >30% of total time)
        if duration_pct > 30.0 {
            bottlenecks.push(format!(
                "Phase '{}' takes {:.1}% of total execution time ({} ms)",
                phase.name, duration_pct, phase.duration_ms
            ));
        }

        // Detect phases significantly slower than average
        if phase.duration_ms as f64 > avg_duration * 3.0 {
            bottlenecks.push(format!(
                "Phase '{}' is {:.1}x slower than average phase duration",
                phase.name,
                phase.duration_ms as f64 / avg_duration
            ));
        }

        // Detect CPU-intensive phases (>40% of total CPU time)
        if cpu_pct > 40.0 {
            bottlenecks.push(format!(
                "Phase '{}' is CPU-intensive ({:.1}% of total CPU time)",
                phase.name, cpu_pct
            ));
        }

        // Detect memory-intensive phases (>50% of total memory)
        if memory_pct > 50.0 {
            let memory_mb = phase.memory_bytes as f64 / (1024.0 * 1024.0);
            bottlenecks.push(format!(
                "Phase '{}' is memory-intensive ({:.1} MB, {:.1}% of total memory)",
                phase.name, memory_mb, memory_pct
            ));
        }

        // Detect phases with low CPU efficiency (wall time >> CPU time)
        if phase.duration_ms > 0 && phase.cpu_time_ms > 0 {
            let cpu_efficiency = (phase.cpu_time_ms as f64 / phase.duration_ms as f64) * 100.0;
            if cpu_efficiency < 20.0 && phase.duration_ms > 100 {
                bottlenecks.push(format!(
                    "Phase '{}' has low CPU efficiency ({:.1}%), likely I/O bound or waiting",
                    phase.name, cpu_efficiency
                ));
            }
        }
    }

    // Deduplicate and limit to top 5 bottlenecks
    bottlenecks.sort();
    bottlenecks.dedup();
    bottlenecks.truncate(5);

    bottlenecks
}
