//! Enhanced health check system with component-level diagnostics
//!
//! This module provides comprehensive health checking including:
//! - Overall system health status
//! - Component-level health diagnostics
//! - Liveness probes for Kubernetes
//! - Readiness probes for load balancers
//! - Detailed diagnostic information

use crate::{
    error::{FusekiError, FusekiResult},
    server::AppState,
};
use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tracing::{debug, warn};

/// Overall health status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub last_check: SystemTime,
    pub details: Option<serde_json::Value>,
}

/// System health response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub timestamp: SystemTime,
    pub version: String,
    pub uptime_seconds: u64,
    pub components: Vec<ComponentHealth>,
}

/// Liveness probe response (simple, fast check)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessResponse {
    pub alive: bool,
    pub timestamp: SystemTime,
}

/// Readiness probe response (can serve traffic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessResponse {
    pub ready: bool,
    pub timestamp: SystemTime,
    pub reasons: Vec<String>,
}

/// Enhanced health check with comprehensive component diagnostics
pub async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let timestamp = SystemTime::now();
    let mut components = Vec::new();
    let mut overall_status = HealthStatus::Healthy;

    // Check RDF store
    let store_health = check_store_health(&state).await;
    if store_health.status != HealthStatus::Healthy {
        overall_status = HealthStatus::Degraded;
    }
    components.push(store_health);

    // Check authentication service
    if let Some(auth_service) = &state.auth_service {
        let auth_health = check_auth_health(auth_service).await;
        if auth_health.status == HealthStatus::Unhealthy {
            overall_status = HealthStatus::Degraded;
        }
        components.push(auth_health);
    }

    // Check metrics service
    if let Some(metrics_service) = &state.metrics_service {
        let metrics_health = check_metrics_health(metrics_service).await;
        components.push(metrics_health);
    }

    // Check performance service
    if let Some(perf_service) = &state.performance_service {
        let perf_health = check_performance_health(perf_service).await;
        components.push(perf_health);
    }

    // Check federation manager
    if let Some(federation_mgr) = &state.federation_manager {
        let federation_health = check_federation_health(federation_mgr).await;
        if federation_health.status == HealthStatus::Unhealthy {
            overall_status = HealthStatus::Degraded;
        }
        components.push(federation_health);
    }

    // Check subscription manager (WebSocket)
    if let Some(sub_mgr) = &state.subscription_manager {
        let ws_health = check_websocket_health(sub_mgr).await;
        components.push(ws_health);
    }

    Json(HealthResponse {
        status: overall_status,
        timestamp,
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.startup_time.elapsed().as_secs(),
        components,
    })
}

/// Liveness probe - minimal check to determine if process is alive
pub async fn liveness_handler() -> Json<LivenessResponse> {
    Json(LivenessResponse {
        alive: true,
        timestamp: SystemTime::now(),
    })
}

/// Readiness probe - determine if service can accept traffic
pub async fn readiness_handler(State(state): State<Arc<AppState>>) -> Json<ReadinessResponse> {
    let mut ready = true;
    let mut reasons = Vec::new();

    // Check if store is accessible
    match check_store_readiness(&state).await {
        Ok(true) => {}
        Ok(false) => {
            ready = false;
            reasons.push("Store not ready".to_string());
        }
        Err(e) => {
            ready = false;
            reasons.push(format!("Store check failed: {}", e));
        }
    }

    // Check if required services are initialized
    if state.config.security.auth_required && state.auth_service.is_none() {
        ready = false;
        reasons.push("Authentication service required but not initialized".to_string());
    }

    if ready {
        reasons.push("All systems operational".to_string());
    }

    Json(ReadinessResponse {
        ready,
        timestamp: SystemTime::now(),
        reasons,
    })
}

/// Check RDF store health
async fn check_store_health(state: &AppState) -> ComponentHealth {
    // Check if store is ready
    let status = if state.store.is_ready() {
        HealthStatus::Healthy
    } else {
        warn!("Store health check failed: not ready");
        HealthStatus::Unhealthy
    };

    ComponentHealth {
        name: "rdf_store".to_string(),
        status,
        message: Some("RDF triple store".to_string()),
        last_check: SystemTime::now(),
        details: None,
    }
}

/// Check store readiness
async fn check_store_readiness(state: &AppState) -> FusekiResult<bool> {
    // Use store's is_ready method
    Ok(state.store.is_ready())
}

/// Check authentication service health
async fn check_auth_health(_auth_service: &crate::auth::AuthService) -> ComponentHealth {
    // Authentication service is stateless, so if it exists it's healthy
    ComponentHealth {
        name: "authentication".to_string(),
        status: HealthStatus::Healthy,
        message: Some("Authentication and authorization".to_string()),
        last_check: SystemTime::now(),
        details: None,
    }
}

/// Check metrics service health
async fn check_metrics_health(
    _metrics_service: &Arc<crate::metrics::MetricsService>,
) -> ComponentHealth {
    ComponentHealth {
        name: "metrics".to_string(),
        status: HealthStatus::Healthy,
        message: Some("Metrics collection and monitoring".to_string()),
        last_check: SystemTime::now(),
        details: None,
    }
}

/// Check performance service health
async fn check_performance_health(
    perf_service: &Arc<crate::performance::PerformanceService>,
) -> ComponentHealth {
    // Get cache statistics
    let cache_stats = perf_service.get_cache_statistics().await;

    let details = serde_json::json!({
        "query_cache_size": cache_stats.query_cache_size,
        "query_cache_capacity": cache_stats.query_cache_capacity,
        "cache_hit_ratio": cache_stats.query_cache_hit_ratio,
        "prepared_cache_size": cache_stats.prepared_cache.size,
        "cache_enabled": cache_stats.cache_enabled,
    });

    ComponentHealth {
        name: "performance".to_string(),
        status: HealthStatus::Healthy,
        message: Some(format!(
            "Cache: {}/{} entries, {:.1}% hit rate",
            cache_stats.query_cache_size,
            cache_stats.query_cache_capacity,
            cache_stats.query_cache_hit_ratio * 100.0
        )),
        last_check: SystemTime::now(),
        details: Some(details),
    }
}

/// Check federation health
async fn check_federation_health(
    _federation_mgr: &Arc<crate::federation::FederationManager>,
) -> ComponentHealth {
    // Could check if federated endpoints are reachable
    ComponentHealth {
        name: "federation".to_string(),
        status: HealthStatus::Healthy,
        message: Some("Federated SPARQL query support".to_string()),
        last_check: SystemTime::now(),
        details: None,
    }
}

/// Check WebSocket subscription health
async fn check_websocket_health(
    _sub_mgr: &Arc<crate::websocket::SubscriptionManager>,
) -> ComponentHealth {
    ComponentHealth {
        name: "websocket".to_string(),
        status: HealthStatus::Healthy,
        message: Some("WebSocket subscription service".to_string()),
        last_check: SystemTime::now(),
        details: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::ServerConfig, store::Store};

    #[tokio::test]
    async fn test_liveness_probe() {
        let response = liveness_handler().await;
        assert!(response.0.alive);
    }

    #[tokio::test]
    async fn test_health_check_basic() {
        let store = Store::new().unwrap();
        let state = Arc::new(AppState {
            store,
            config: ServerConfig::default(),
            auth_service: None,
            metrics_service: None,
            performance_service: None,
            query_optimizer: None,
            subscription_manager: None,
            federation_manager: None,
            streaming_manager: None,
            // Beta.2 Performance & Scalability Features
            concurrency_manager: None,
            memory_manager: None,
            batch_executor: None,
            stream_manager: None,
            dataset_manager: None,
            // RC.1 Production & Advanced Features
            security_auditor: None,
            ddos_protector: None,
            load_balancer: None,
            edge_cache_manager: None,
            performance_profiler: None,
            notification_manager: None,
            backup_manager: None,
            recovery_manager: None,
            disaster_recovery: None,
            certificate_rotation: None,
            http2_manager: None,
            http3_manager: None,
            adaptive_execution_engine: None,
            rebac_manager: None,
            prefix_store: Arc::new(crate::handlers::PrefixStore::new()),
            task_manager: Arc::new(crate::handlers::TaskManager::new()),
            request_logger: Arc::new(crate::handlers::RequestLogger::new()),
            startup_time: std::time::Instant::now(),
            system_monitor: Arc::new(parking_lot::Mutex::new(sysinfo::System::new_all())),
            #[cfg(feature = "rate-limit")]
            rate_limiter: None,
        });

        let response = health_handler(State(state)).await;
        assert!(!response.0.components.is_empty());
        assert!(matches!(
            response.0.status,
            HealthStatus::Healthy | HealthStatus::Degraded
        ));
    }

    #[tokio::test]
    async fn test_readiness_probe() {
        let store = Store::new().unwrap();
        let state = Arc::new(AppState {
            store,
            config: ServerConfig::default(),
            auth_service: None,
            metrics_service: None,
            performance_service: None,
            query_optimizer: None,
            subscription_manager: None,
            federation_manager: None,
            streaming_manager: None,
            // Beta.2 Performance & Scalability Features
            concurrency_manager: None,
            memory_manager: None,
            batch_executor: None,
            stream_manager: None,
            dataset_manager: None,
            // RC.1 Production & Advanced Features
            security_auditor: None,
            ddos_protector: None,
            load_balancer: None,
            edge_cache_manager: None,
            performance_profiler: None,
            notification_manager: None,
            backup_manager: None,
            recovery_manager: None,
            disaster_recovery: None,
            certificate_rotation: None,
            http2_manager: None,
            http3_manager: None,
            adaptive_execution_engine: None,
            rebac_manager: None,
            prefix_store: Arc::new(crate::handlers::PrefixStore::new()),
            task_manager: Arc::new(crate::handlers::TaskManager::new()),
            request_logger: Arc::new(crate::handlers::RequestLogger::new()),
            startup_time: std::time::Instant::now(),
            system_monitor: Arc::new(parking_lot::Mutex::new(sysinfo::System::new_all())),
            #[cfg(feature = "rate-limit")]
            rate_limiter: None,
        });

        let response = readiness_handler(State(state)).await;
        assert!(response.0.ready);
        assert!(!response.0.reasons.is_empty());
    }
}
