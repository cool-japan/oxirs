//! Federation Monitor Implementation
//!
//! Main monitor that tracks federation performance, health, and issues.

use crate::monitoring::config::*;
use crate::monitoring::metrics::*;
use crate::monitoring::types::*;
use crate::monitoring::resilience::ResilienceManager;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Federation performance monitor
#[derive(Debug)]
pub struct FederationMonitor {
    metrics: Arc<RwLock<FederationMetrics>>,
    config: FederationMonitorConfig,
    start_time: Instant,
    resilience_manager: ResilienceManager,
}

impl FederationMonitor {
    /// Create a new federation monitor
    pub fn new() -> Self {
        let metrics = Arc::new(RwLock::new(FederationMetrics::new()));
        let config = FederationMonitorConfig::default();
        let resilience_manager = ResilienceManager::new(Arc::clone(&metrics), config.clone());
        
        Self {
            metrics,
            config,
            start_time: Instant::now(),
            resilience_manager,
        }
    }

    /// Create a new federation monitor with custom configuration
    pub fn with_config(config: FederationMonitorConfig) -> Self {
        let metrics = Arc::new(RwLock::new(FederationMetrics::new()));
        let resilience_manager = ResilienceManager::new(Arc::clone(&metrics), config.clone());
        
        Self {
            metrics,
            config,
            start_time: Instant::now(),
            resilience_manager,
        }
    }

    /// Record a query execution
    pub async fn record_query_execution(
        &self,
        query_type: &str,
        duration: Duration,
        success: bool,
    ) {
        let mut metrics = self.metrics.write().await;

        metrics.total_queries += 1;
        if success {
            metrics.successful_queries += 1;
        } else {
            metrics.failed_queries += 1;
        }

        // Update type-specific metrics
        let type_metrics = metrics
            .query_type_metrics
            .entry(query_type.to_string())
            .or_default();
        type_metrics.total_count += 1;
        type_metrics.total_duration += duration;
        type_metrics.avg_duration = type_metrics.total_duration / type_metrics.total_count as u32;

        if success {
            type_metrics.success_count += 1;
        } else {
            type_metrics.error_count += 1;
        }

        debug!(
            "Recorded {} query execution: {}ms, success: {}",
            query_type,
            duration.as_millis(),
            success
        );
    }

    /// Get comprehensive monitoring statistics
    pub async fn get_stats(&self) -> MonitorStats {
        let metrics = self.metrics.read().await;
        let uptime = self.start_time.elapsed();

        MonitorStats {
            uptime,
            total_queries: metrics.total_queries,
            successful_queries: metrics.successful_queries,
            failed_queries: metrics.failed_queries,
            success_rate: if metrics.total_queries > 0 {
                metrics.successful_queries as f64 / metrics.total_queries as f64
            } else {
                0.0
            },
            query_type_metrics: metrics.query_type_metrics.clone(),
            service_metrics: metrics.service_metrics.clone(),
            cache_metrics: metrics.cache_metrics.clone(),
            response_time_histogram: metrics.response_time_histogram.clone(),
            recent_events_count: metrics.federation_events.len(),
            event_type_counts: metrics.event_type_counts.clone(),
            avg_queries_per_second: if uptime.as_secs() > 0 {
                metrics.total_queries as f64 / uptime.as_secs() as f64
            } else {
                0.0
            },
        }
    }

    /// Record service interaction metrics
    pub async fn record_service_interaction(
        &self,
        service_id: &str,
        duration: Duration,
        success: bool,
        response_size: Option<usize>,
    ) {
        let mut metrics = self.metrics.write().await;

        let service_metrics = metrics
            .service_metrics
            .entry(service_id.to_string())
            .or_default();

        service_metrics.total_requests += 1;
        service_metrics.total_duration += duration;
        service_metrics.avg_duration =
            service_metrics.total_duration / service_metrics.total_requests as u32;

        if success {
            service_metrics.successful_requests += 1;
        } else {
            service_metrics.failed_requests += 1;
        }

        if let Some(size) = response_size {
            service_metrics.total_response_size += size as u64;
            service_metrics.avg_response_size = service_metrics.total_response_size
                / service_metrics.successful_requests.max(1);
        }

        // Update last seen timestamp
        service_metrics.last_seen = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        debug!(
            "Recorded service interaction for {}: {}ms, success: {}",
            service_id,
            duration.as_millis(),
            success
        );
    }

    /// Record federation-specific events
    pub async fn record_federation_event(&self, event_type: FederationEventType, details: &str) {
        let mut metrics = self.metrics.write().await;

        let event = FederationEvent {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            event_type,
            details: details.to_string(),
        };

        metrics.federation_events.push(event);

        // Keep only recent events
        if metrics.federation_events.len() > self.config.max_recent_events {
            metrics.federation_events.remove(0);
        }

        // Update event type counters
        *metrics.event_type_counts.entry(event_type).or_insert(0) += 1;

        info!("Federation event: {:?} - {}", event_type, details);
    }

    /// Record cache statistics
    pub async fn record_cache_hit(&self, cache_type: &str, hit: bool) {
        let mut metrics = self.metrics.write().await;

        let cache_metrics = metrics
            .cache_metrics
            .entry(cache_type.to_string())
            .or_default();

        cache_metrics.total_requests += 1;
        if hit {
            cache_metrics.hits += 1;
        } else {
            cache_metrics.misses += 1;
        }

        cache_metrics.hit_rate = if cache_metrics.total_requests > 0 {
            cache_metrics.hits as f64 / cache_metrics.total_requests as f64
        } else {
            0.0
        };

        debug!(
            "Cache {} for {}: hit rate {:.2}%",
            if hit { "hit" } else { "miss" },
            cache_type,
            cache_metrics.hit_rate * 100.0
        );
    }

    /// Get health metrics
    pub async fn get_health_metrics(&self) -> HealthMetrics {
        let metrics = self.metrics.read().await;
        let mut overall_health = HealthStatus::Healthy;
        let mut service_health = HashMap::new();
        
        // Calculate overall error rate
        let error_rate = if metrics.total_queries > 0 {
            metrics.failed_queries as f64 / metrics.total_queries as f64
        } else {
            0.0
        };
        
        // Calculate average response time
        let avg_response_time = if !metrics.query_type_metrics.is_empty() {
            let total_duration: Duration = metrics.query_type_metrics.values()
                .map(|m| m.avg_duration)
                .sum();
            total_duration / metrics.query_type_metrics.len() as u32
        } else {
            Duration::from_secs(0)
        };
        
        // Determine overall health based on error rate and response time
        if error_rate > 0.2 || avg_response_time > Duration::from_secs(5) {
            overall_health = HealthStatus::Unhealthy;
        } else if error_rate > 0.15 || avg_response_time > Duration::from_secs(2) {
            overall_health = HealthStatus::Degraded;
        }
        
        // Check individual service health
        for (service_id, service_metrics) in &metrics.service_metrics {
            let service_error_rate = if service_metrics.total_requests > 0 {
                service_metrics.failed_requests as f64 / service_metrics.total_requests as f64
            } else {
                0.0
            };
            
            let health = if service_error_rate > 0.2 || service_metrics.avg_duration > Duration::from_secs(3) {
                HealthStatus::Unhealthy
            } else if service_error_rate > 0.1 || service_metrics.avg_duration > Duration::from_secs(1) {
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            };
            
            service_health.insert(service_id.clone(), health);
        }
        
        // Calculate cache hit rate
        let cache_hit_rate = if !metrics.cache_metrics.is_empty() {
            let total_hits: u64 = metrics.cache_metrics.values().map(|m| m.hits).sum();
            let total_requests: u64 = metrics.cache_metrics.values().map(|m| m.total_requests).sum();
            if total_requests > 0 {
                total_hits as f64 / total_requests as f64
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        HealthMetrics {
            overall_health,
            service_health,
            error_rate,
            avg_response_time,
            active_services: metrics.service_metrics.len(),
            recent_error_count: metrics.failed_queries as usize,
            cache_hit_rate,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }
    
    /// Export metrics in Prometheus format
    pub async fn export_prometheus_metrics(&self) -> String {
        let metrics = self.metrics.read().await;
        let mut output = String::new();
        
        // Basic counters
        output.push_str(&"# HELP federation_queries_total Total number of federation queries\n".to_string());
        output.push_str(&"# TYPE federation_queries_total counter\n".to_string());
        output.push_str(&format!("federation_queries_total {}\n", metrics.total_queries));
        
        output.push_str(&"# HELP federation_queries_successful Total number of successful queries\n".to_string());
        output.push_str(&"# TYPE federation_queries_successful counter\n".to_string());
        output.push_str(&format!("federation_queries_successful {}\n", metrics.successful_queries));
        
        output.push_str(&"# HELP federation_queries_failed Total number of failed queries\n".to_string());
        output.push_str(&"# TYPE federation_queries_failed counter\n".to_string());
        output.push_str(&format!("federation_queries_failed {}\n", metrics.failed_queries));
        
        // Service metrics
        for (service_id, service_metrics) in &metrics.service_metrics {
            output.push_str(&format!("federation_service_requests_total{{service=\"{}\"}} {}\n", service_id, service_metrics.total_requests));
            output.push_str(&format!("federation_service_duration_seconds{{service=\"{}\"}} {:.3}\n", service_id, service_metrics.avg_duration.as_secs_f64()));
        }
        
        // Cache metrics
        for (cache_type, cache_metrics) in &metrics.cache_metrics {
            output.push_str(&format!("federation_cache_hit_rate{{cache=\"{}\"}} {:.3}\n", cache_type, cache_metrics.hit_rate));
            output.push_str(&format!("federation_cache_requests_total{{cache=\"{}\"}} {}\n", cache_type, cache_metrics.total_requests));
        }
        
        output
    }
    
    /// Get comprehensive performance report
    pub async fn get_performance_report(&self) -> PerformanceReport {
        let metrics = self.metrics.read().await;
        let uptime = self.start_time.elapsed();
        
        // Calculate query trends
        let mut query_trends = HashMap::new();
        for (query_type, type_metrics) in &metrics.query_type_metrics {
            let error_rate = if type_metrics.total_count > 0 {
                type_metrics.error_count as f64 / type_metrics.total_count as f64
            } else {
                0.0
            };
            
            let queries_per_second = if uptime.as_secs() > 0 {
                type_metrics.total_count as f64 / uptime.as_secs() as f64
            } else {
                0.0
            };
            
            query_trends.insert(query_type.clone(), QueryTrend {
                query_type: query_type.clone(),
                total_queries: type_metrics.total_count,
                avg_response_time: type_metrics.avg_duration,
                error_rate,
                queries_per_second,
            });
        }
        
        // Calculate performance summary
        let total_services = metrics.service_metrics.len();
        let healthy_services = metrics.service_metrics.iter()
            .filter(|(_, metrics)| {
                let error_rate = if metrics.total_requests > 0 {
                    metrics.failed_requests as f64 / metrics.total_requests as f64
                } else {
                    0.0
                };
                error_rate < 0.1 && metrics.avg_duration < Duration::from_secs(2)
            })
            .count();
        
        let avg_query_time = if !metrics.query_type_metrics.is_empty() {
            let total_duration: Duration = metrics.query_type_metrics.values()
                .map(|m| m.avg_duration)
                .sum();
            total_duration / metrics.query_type_metrics.len() as u32
        } else {
            Duration::from_secs(0)
        };
        
        let cache_efficiency = if !metrics.cache_metrics.is_empty() {
            let avg_hit_rate: f64 = metrics.cache_metrics.values()
                .map(|m| m.hit_rate)
                .sum::<f64>() / metrics.cache_metrics.len() as f64;
            avg_hit_rate
        } else {
            0.0
        };
        
        PerformanceReport {
            report_timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            uptime,
            total_queries: metrics.total_queries,
            overall_success_rate: if metrics.total_queries > 0 {
                metrics.successful_queries as f64 / metrics.total_queries as f64
            } else {
                0.0
            },
            query_trends,
            top_errors: vec![], // Would be populated with actual error analysis
            performance_summary: PerformanceSummary {
                total_services,
                healthy_services,
                avg_query_time,
                cache_efficiency,
            },
            bottlenecks: vec![], // Would be populated with bottleneck analysis
            performance_regressions: vec![], // Would be populated with regression analysis
            optimization_recommendations: vec![], // Would be populated with recommendations
        }
    }
    
    /// Delegate resilience methods to ResilienceManager
    pub async fn check_circuit_breaker(&self, service_id: &str) -> CircuitBreakerState {
        self.resilience_manager.check_circuit_breaker(service_id).await
    }
    
    pub async fn get_recovery_recommendations(&self) -> Vec<RecoveryRecommendation> {
        self.resilience_manager.get_recovery_recommendations().await
    }
    
    pub async fn predict_failures(&self) -> Vec<FailurePrediction> {
        self.resilience_manager.predict_failures().await
    }
    
    pub async fn attempt_auto_healing(&self, issue: &str) -> Result<AutoHealingAction> {
        self.resilience_manager.attempt_auto_healing(issue).await
    }
}

impl Default for FederationMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// Federation metrics and events are now defined in the metrics module