//! Federation Topology Manager
//!
//! This module implements `FederationTopologyManager`, which provides:
//! - Dynamic endpoint discovery (register/unregister endpoints at runtime).
//! - Health monitoring with configurable intervals and thresholds.
//! - Capability negotiation (which SPARQL/GraphQL features each endpoint supports).
//! - Automatic failover: queries are re-routed away from failing endpoints.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// ─── SparqlFeature ────────────────────────────────────────────────────────────

/// A SPARQL / extension feature that an endpoint may support.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SparqlFeature {
    /// SPARQL 1.0 core SELECT/CONSTRUCT/ASK/DESCRIBE
    Sparql10,
    /// SPARQL 1.1 query (property paths, aggregates, sub-queries, …)
    Sparql11Query,
    /// SPARQL 1.1 UPDATE
    Sparql11Update,
    /// SPARQL 1.2 (draft features)
    Sparql12,
    /// RDF-star (embedded triples)
    RdfStar,
    /// Full-text search extension
    FullTextSearch,
    /// GeoSPARQL extension
    GeoSparql,
    /// Federated SERVICE clause
    FederatedService,
    /// Named graph management
    NamedGraphs,
    /// Bulk load endpoint
    BulkLoad,
    /// SPARQL-star / RDF-star query extensions
    SparqlStar,
    /// GraphQL query interface
    GraphQL,
    /// Custom named feature
    Custom(String),
}

// ─── EndpointStatus ──────────────────────────────────────────────────────────

/// Health status of a discovered endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndpointStatus {
    /// The endpoint responded successfully to the last health check.
    Healthy,
    /// The endpoint is temporarily degraded (slow / partial failures).
    Degraded,
    /// The endpoint is not responding.
    Unavailable,
    /// Status is not yet known (never probed).
    Unknown,
}

// ─── EndpointInfo ─────────────────────────────────────────────────────────────

/// All metadata for a single endpoint tracked by the topology manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointInfo {
    /// Unique identifier.
    pub id: String,
    /// URL of the SPARQL / GraphQL endpoint.
    pub url: String,
    /// Display name (optional).
    pub display_name: Option<String>,
    /// Current health status.
    pub status: EndpointStatus,
    /// Features negotiated / reported by this endpoint.
    pub capabilities: HashSet<SparqlFeature>,
    /// Number of consecutive health-check failures.
    pub consecutive_failures: u32,
    /// Timestamp (ms since UNIX epoch) of the last successful health check.
    pub last_healthy_ms: Option<u64>,
    /// Timestamp (ms since UNIX epoch) of the last health check attempt.
    pub last_checked_ms: Option<u64>,
    /// Average response time recorded during the last health check (ms).
    pub avg_response_ms: f64,
    /// Whether this endpoint is active (not soft-deleted).
    pub active: bool,
    /// Endpoint priority (higher = preferred). Default 100.
    pub priority: u32,
    /// Tags for grouping / filtering endpoints.
    pub tags: Vec<String>,
}

impl EndpointInfo {
    /// Create a new endpoint with unknown status and no negotiated capabilities.
    pub fn new(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
            display_name: None,
            status: EndpointStatus::Unknown,
            capabilities: HashSet::new(),
            consecutive_failures: 0,
            last_healthy_ms: None,
            last_checked_ms: None,
            avg_response_ms: 0.0,
            active: true,
            priority: 100,
            tags: Vec::new(),
        }
    }

    /// Return `true` if the endpoint can be used for routing (healthy/degraded and active).
    pub fn is_available(&self) -> bool {
        self.active
            && matches!(
                self.status,
                EndpointStatus::Healthy | EndpointStatus::Degraded
            )
    }

    /// Return `true` if the endpoint supports a given feature.
    pub fn supports(&self, feature: &SparqlFeature) -> bool {
        self.capabilities.contains(feature)
    }
}

// ─── TopologyConfig ──────────────────────────────────────────────────────────

/// Configuration for `FederationTopologyManager`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    /// Maximum consecutive health-check failures before marking unavailable.
    pub max_consecutive_failures: u32,
    /// Number of failures above which status is set to `Degraded` (before `max_consecutive_failures`).
    pub degraded_failure_threshold: u32,
    /// Interval between background health checks.
    pub health_check_interval: Duration,
    /// Timeout for each health check.
    pub health_check_timeout: Duration,
    /// Minimum interval between failover events per endpoint.
    pub failover_cooldown: Duration,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            max_consecutive_failures: 5,
            degraded_failure_threshold: 2,
            health_check_interval: Duration::from_secs(30),
            health_check_timeout: Duration::from_secs(10),
            failover_cooldown: Duration::from_secs(60),
        }
    }
}

// ─── HealthCheckResult ───────────────────────────────────────────────────────

/// Result of a simulated / real health check.
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Whether the check succeeded.
    pub success: bool,
    /// Response time (simulated or real).
    pub response_ms: f64,
    /// Features reported by the endpoint during the check.
    pub reported_features: HashSet<SparqlFeature>,
    /// Optional diagnostic message.
    pub message: Option<String>,
}

impl HealthCheckResult {
    /// Successful result with a known response time and features.
    pub fn success(response_ms: f64, features: HashSet<SparqlFeature>) -> Self {
        Self {
            success: true,
            response_ms,
            reported_features: features,
            message: None,
        }
    }

    /// Failed result.
    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            response_ms: 0.0,
            reported_features: HashSet::new(),
            message: Some(message.into()),
        }
    }
}

// ─── TopologyEvent ────────────────────────────────────────────────────────────

/// Events emitted by the topology manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopologyEvent {
    /// A new endpoint was registered.
    EndpointRegistered { endpoint_id: String },
    /// An endpoint was removed.
    EndpointRemoved { endpoint_id: String },
    /// An endpoint changed health status.
    StatusChanged {
        endpoint_id: String,
        old_status: EndpointStatus,
        new_status: EndpointStatus,
    },
    /// A failover occurred (queries redirected away from a failing endpoint).
    Failover {
        failed_endpoint: String,
        replacement_endpoint: Option<String>,
    },
    /// Capabilities were negotiated for an endpoint.
    CapabilitiesNegotiated {
        endpoint_id: String,
        feature_count: usize,
    },
}

// ─── FederationTopologyManager ───────────────────────────────────────────────

/// Manages the topology of federated endpoints: discovery, health, capabilities.
pub struct FederationTopologyManager {
    config: TopologyConfig,
    endpoints: Arc<RwLock<HashMap<String, EndpointInfo>>>,
    events: Arc<RwLock<Vec<TopologyEvent>>>,
    /// Last failover timestamp per endpoint.
    last_failover: Arc<RwLock<HashMap<String, Instant>>>,
}

impl FederationTopologyManager {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: TopologyConfig::default(),
            endpoints: Arc::new(RwLock::new(HashMap::new())),
            events: Arc::new(RwLock::new(Vec::new())),
            last_failover: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: TopologyConfig) -> Self {
        Self {
            config,
            endpoints: Arc::new(RwLock::new(HashMap::new())),
            events: Arc::new(RwLock::new(Vec::new())),
            last_failover: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // ── Discovery ─────────────────────────────────────────────────────────────

    /// Register a new endpoint.
    ///
    /// Returns an error if an endpoint with the same ID already exists.
    pub async fn register_endpoint(&self, info: EndpointInfo) -> Result<()> {
        let id = info.id.clone();
        let mut endpoints = self.endpoints.write().await;
        if endpoints.contains_key(&id) {
            return Err(anyhow!("Endpoint '{}' is already registered", id));
        }
        endpoints.insert(id.clone(), info);
        drop(endpoints);

        self.emit_event(TopologyEvent::EndpointRegistered {
            endpoint_id: id.clone(),
        })
        .await;
        info!("TopologyManager: registered endpoint '{}'", id);
        Ok(())
    }

    /// Register or update an endpoint (upsert semantics).
    pub async fn upsert_endpoint(&self, info: EndpointInfo) {
        let id = info.id.clone();
        let existing = {
            let endpoints = self.endpoints.read().await;
            endpoints.contains_key(&id)
        };
        let mut endpoints = self.endpoints.write().await;
        endpoints.insert(id.clone(), info);
        drop(endpoints);

        if !existing {
            self.emit_event(TopologyEvent::EndpointRegistered {
                endpoint_id: id.clone(),
            })
            .await;
            info!("TopologyManager: upserted (new) endpoint '{}'", id);
        } else {
            debug!("TopologyManager: updated endpoint '{}'", id);
        }
    }

    /// Remove an endpoint by ID.
    pub async fn remove_endpoint(&self, endpoint_id: &str) -> Result<()> {
        let mut endpoints = self.endpoints.write().await;
        if endpoints.remove(endpoint_id).is_none() {
            return Err(anyhow!("Endpoint '{}' not found", endpoint_id));
        }
        drop(endpoints);

        self.emit_event(TopologyEvent::EndpointRemoved {
            endpoint_id: endpoint_id.to_owned(),
        })
        .await;
        info!("TopologyManager: removed endpoint '{}'", endpoint_id);
        Ok(())
    }

    /// Soft-deactivate an endpoint (keeps it in the registry but routes around it).
    pub async fn deactivate_endpoint(&self, endpoint_id: &str) -> Result<()> {
        let mut endpoints = self.endpoints.write().await;
        let entry = endpoints
            .get_mut(endpoint_id)
            .ok_or_else(|| anyhow!("Endpoint '{}' not found", endpoint_id))?;
        entry.active = false;
        Ok(())
    }

    /// Re-activate a soft-deactivated endpoint.
    pub async fn activate_endpoint(&self, endpoint_id: &str) -> Result<()> {
        let mut endpoints = self.endpoints.write().await;
        let entry = endpoints
            .get_mut(endpoint_id)
            .ok_or_else(|| anyhow!("Endpoint '{}' not found", endpoint_id))?;
        entry.active = true;
        Ok(())
    }

    // ── Health monitoring ─────────────────────────────────────────────────────

    /// Apply a health-check result for an endpoint (update status + capabilities).
    pub async fn apply_health_check(
        &self,
        endpoint_id: &str,
        result: HealthCheckResult,
    ) -> Result<()> {
        let old_status;
        let new_status;

        {
            let mut endpoints = self.endpoints.write().await;
            let entry = endpoints
                .get_mut(endpoint_id)
                .ok_or_else(|| anyhow!("Endpoint '{}' not found", endpoint_id))?;

            old_status = entry.status.clone();
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            entry.last_checked_ms = Some(now_ms);

            if result.success {
                entry.consecutive_failures = 0;
                entry.last_healthy_ms = Some(now_ms);
                entry.avg_response_ms = result.response_ms;

                // Merge negotiated capabilities.
                if !result.reported_features.is_empty() {
                    let feature_count = result.reported_features.len();
                    entry.capabilities = result.reported_features;
                    drop(endpoints);
                    self.emit_event(TopologyEvent::CapabilitiesNegotiated {
                        endpoint_id: endpoint_id.to_owned(),
                        feature_count,
                    })
                    .await;
                    // Re-acquire to set status.
                    let mut endpoints = self.endpoints.write().await;
                    let entry = endpoints.get_mut(endpoint_id).unwrap();
                    entry.status = EndpointStatus::Healthy;
                    new_status = EndpointStatus::Healthy;
                    drop(endpoints);
                } else {
                    entry.status = EndpointStatus::Healthy;
                    new_status = EndpointStatus::Healthy;
                }
            } else {
                entry.consecutive_failures += 1;
                new_status = if entry.consecutive_failures >= self.config.max_consecutive_failures {
                    EndpointStatus::Unavailable
                } else if entry.consecutive_failures >= self.config.degraded_failure_threshold {
                    EndpointStatus::Degraded
                } else {
                    entry.status.clone()
                };
                entry.status = new_status.clone();
                if let Some(msg) = &result.message {
                    warn!(
                        "TopologyManager: health check failed for '{}': {}",
                        endpoint_id, msg
                    );
                }
            }
        }

        if old_status != new_status {
            self.emit_event(TopologyEvent::StatusChanged {
                endpoint_id: endpoint_id.to_owned(),
                old_status,
                new_status,
            })
            .await;
        }

        Ok(())
    }

    // ── Capability negotiation ─────────────────────────────────────────────────

    /// Manually set the capabilities for an endpoint.
    pub async fn set_capabilities(
        &self,
        endpoint_id: &str,
        features: HashSet<SparqlFeature>,
    ) -> Result<()> {
        let mut endpoints = self.endpoints.write().await;
        let entry = endpoints
            .get_mut(endpoint_id)
            .ok_or_else(|| anyhow!("Endpoint '{}' not found", endpoint_id))?;
        let feature_count = features.len();
        entry.capabilities = features;
        drop(endpoints);

        self.emit_event(TopologyEvent::CapabilitiesNegotiated {
            endpoint_id: endpoint_id.to_owned(),
            feature_count,
        })
        .await;
        Ok(())
    }

    /// Return all endpoints that support the given feature.
    pub async fn endpoints_with_feature(&self, feature: &SparqlFeature) -> Vec<EndpointInfo> {
        self.endpoints
            .read()
            .await
            .values()
            .filter(|e| e.is_available() && e.supports(feature))
            .cloned()
            .collect()
    }

    // ── Failover ──────────────────────────────────────────────────────────────

    /// Trigger a failover for a failing endpoint.
    ///
    /// Returns the ID of the replacement endpoint, or `None` if no healthy
    /// replacement is available.
    pub async fn trigger_failover(&self, failing_endpoint_id: &str) -> Option<String> {
        // Respect the cooldown period.
        {
            let last = self.last_failover.read().await;
            if let Some(ts) = last.get(failing_endpoint_id) {
                if ts.elapsed() < self.config.failover_cooldown {
                    debug!(
                        "TopologyManager: failover for '{}' is on cooldown",
                        failing_endpoint_id
                    );
                    return None;
                }
            }
        }

        let replacement = self.find_replacement(failing_endpoint_id).await;

        // Update cooldown.
        {
            let mut last = self.last_failover.write().await;
            last.insert(failing_endpoint_id.to_owned(), Instant::now());
        }

        let event = TopologyEvent::Failover {
            failed_endpoint: failing_endpoint_id.to_owned(),
            replacement_endpoint: replacement.clone(),
        };
        self.emit_event(event).await;

        if let Some(ref r) = replacement {
            info!(
                "TopologyManager: failover from '{}' to '{}'",
                failing_endpoint_id, r
            );
        } else {
            warn!(
                "TopologyManager: failover from '{}' – no replacement available",
                failing_endpoint_id
            );
        }

        replacement
    }

    /// Find a healthy replacement for the given endpoint.
    async fn find_replacement(&self, failed_id: &str) -> Option<String> {
        let endpoints = self.endpoints.read().await;
        endpoints
            .values()
            .filter(|e| e.id != failed_id && e.is_available())
            .max_by_key(|e| e.priority)
            .map(|e| e.id.clone())
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Return a snapshot of all registered endpoints.
    pub async fn all_endpoints(&self) -> Vec<EndpointInfo> {
        self.endpoints.read().await.values().cloned().collect()
    }

    /// Return all currently available endpoints.
    pub async fn available_endpoints(&self) -> Vec<EndpointInfo> {
        self.endpoints
            .read()
            .await
            .values()
            .filter(|e| e.is_available())
            .cloned()
            .collect()
    }

    /// Look up an endpoint by ID.
    pub async fn get_endpoint(&self, endpoint_id: &str) -> Option<EndpointInfo> {
        self.endpoints.read().await.get(endpoint_id).cloned()
    }

    /// Return the event log.
    pub async fn events(&self) -> Vec<TopologyEvent> {
        self.events.read().await.clone()
    }

    /// Clear the event log.
    pub async fn clear_events(&self) {
        self.events.write().await.clear();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    async fn emit_event(&self, event: TopologyEvent) {
        self.events.write().await.push(event);
    }
}

impl Default for FederationTopologyManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_endpoint(id: &str, url: &str) -> EndpointInfo {
        EndpointInfo::new(id, url)
    }

    fn sparql11_features() -> HashSet<SparqlFeature> {
        let mut s = HashSet::new();
        s.insert(SparqlFeature::Sparql11Query);
        s.insert(SparqlFeature::NamedGraphs);
        s
    }

    // ── Registration ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_register_endpoint() {
        let mgr = FederationTopologyManager::new();
        mgr.register_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await
            .unwrap();
        let eps = mgr.all_endpoints().await;
        assert_eq!(eps.len(), 1);
    }

    #[tokio::test]
    async fn test_register_duplicate_returns_error() {
        let mgr = FederationTopologyManager::new();
        mgr.register_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await
            .unwrap();
        let result = mgr
            .register_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_upsert_endpoint_creates_new() {
        let mgr = FederationTopologyManager::new();
        mgr.upsert_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await;
        assert_eq!(mgr.all_endpoints().await.len(), 1);
    }

    #[tokio::test]
    async fn test_upsert_endpoint_updates_existing() {
        let mgr = FederationTopologyManager::new();
        mgr.upsert_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await;
        let mut updated = make_endpoint("ep1", "http://a.example/sparql/v2");
        updated.priority = 200;
        mgr.upsert_endpoint(updated).await;
        let ep = mgr.get_endpoint("ep1").await.unwrap();
        assert_eq!(ep.priority, 200);
    }

    #[tokio::test]
    async fn test_remove_endpoint() {
        let mgr = FederationTopologyManager::new();
        mgr.register_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await
            .unwrap();
        mgr.remove_endpoint("ep1").await.unwrap();
        assert!(mgr.all_endpoints().await.is_empty());
    }

    #[tokio::test]
    async fn test_remove_nonexistent_endpoint_returns_error() {
        let mgr = FederationTopologyManager::new();
        let result = mgr.remove_endpoint("ghost").await;
        assert!(result.is_err());
    }

    // ── Activation ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_deactivate_and_activate_endpoint() {
        let mgr = FederationTopologyManager::new();
        let mut ep = make_endpoint("ep1", "http://a.example/sparql");
        ep.status = EndpointStatus::Healthy;
        mgr.register_endpoint(ep).await.unwrap();

        mgr.deactivate_endpoint("ep1").await.unwrap();
        assert!(mgr.available_endpoints().await.is_empty());

        mgr.activate_endpoint("ep1").await.unwrap();
        assert_eq!(mgr.available_endpoints().await.len(), 1);
    }

    // ── Health checks ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_successful_health_check_marks_healthy() {
        let mgr = FederationTopologyManager::new();
        mgr.register_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await
            .unwrap();

        let result = HealthCheckResult::success(50.0, HashSet::new());
        mgr.apply_health_check("ep1", result).await.unwrap();

        let ep = mgr.get_endpoint("ep1").await.unwrap();
        assert_eq!(ep.status, EndpointStatus::Healthy);
        assert_eq!(ep.consecutive_failures, 0);
    }

    #[tokio::test]
    async fn test_failed_health_check_increments_failures() {
        let mgr = FederationTopologyManager::new();
        mgr.register_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await
            .unwrap();

        let result = HealthCheckResult::failure("connection refused");
        mgr.apply_health_check("ep1", result).await.unwrap();

        let ep = mgr.get_endpoint("ep1").await.unwrap();
        assert_eq!(ep.consecutive_failures, 1);
    }

    #[tokio::test]
    async fn test_repeated_failures_mark_degraded_then_unavailable() {
        let config = TopologyConfig {
            max_consecutive_failures: 3,
            degraded_failure_threshold: 1,
            ..Default::default()
        };
        let mgr = FederationTopologyManager::with_config(config);
        mgr.register_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await
            .unwrap();

        // First failure → degraded threshold reached
        mgr.apply_health_check("ep1", HealthCheckResult::failure("err"))
            .await
            .unwrap();
        let ep = mgr.get_endpoint("ep1").await.unwrap();
        assert_eq!(ep.status, EndpointStatus::Degraded);

        // Second and third failures → unavailable
        mgr.apply_health_check("ep1", HealthCheckResult::failure("err"))
            .await
            .unwrap();
        mgr.apply_health_check("ep1", HealthCheckResult::failure("err"))
            .await
            .unwrap();
        let ep = mgr.get_endpoint("ep1").await.unwrap();
        assert_eq!(ep.status, EndpointStatus::Unavailable);
    }

    #[tokio::test]
    async fn test_health_check_recovery_resets_failures() {
        let mgr = FederationTopologyManager::new();
        mgr.register_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await
            .unwrap();

        mgr.apply_health_check("ep1", HealthCheckResult::failure("err"))
            .await
            .unwrap();
        mgr.apply_health_check("ep1", HealthCheckResult::success(30.0, HashSet::new()))
            .await
            .unwrap();

        let ep = mgr.get_endpoint("ep1").await.unwrap();
        assert_eq!(ep.status, EndpointStatus::Healthy);
        assert_eq!(ep.consecutive_failures, 0);
    }

    // ── Capabilities ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_set_capabilities() {
        let mgr = FederationTopologyManager::new();
        mgr.register_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await
            .unwrap();

        mgr.set_capabilities("ep1", sparql11_features())
            .await
            .unwrap();

        let ep = mgr.get_endpoint("ep1").await.unwrap();
        assert!(ep.supports(&SparqlFeature::Sparql11Query));
        assert!(ep.supports(&SparqlFeature::NamedGraphs));
        assert!(!ep.supports(&SparqlFeature::GeoSparql));
    }

    #[tokio::test]
    async fn test_health_check_with_features_updates_capabilities() {
        let mgr = FederationTopologyManager::new();
        mgr.register_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await
            .unwrap();

        let mut features = HashSet::new();
        features.insert(SparqlFeature::GeoSparql);
        mgr.apply_health_check("ep1", HealthCheckResult::success(40.0, features))
            .await
            .unwrap();

        let ep = mgr.get_endpoint("ep1").await.unwrap();
        assert!(ep.supports(&SparqlFeature::GeoSparql));
    }

    #[tokio::test]
    async fn test_endpoints_with_feature_filters_correctly() {
        let mgr = FederationTopologyManager::new();

        let mut ep1 = make_endpoint("ep1", "http://a.example/sparql");
        ep1.status = EndpointStatus::Healthy;
        ep1.capabilities.insert(SparqlFeature::GeoSparql);
        mgr.register_endpoint(ep1).await.unwrap();

        let mut ep2 = make_endpoint("ep2", "http://b.example/sparql");
        ep2.status = EndpointStatus::Healthy;
        ep2.capabilities.insert(SparqlFeature::FullTextSearch);
        mgr.register_endpoint(ep2).await.unwrap();

        let geo = mgr.endpoints_with_feature(&SparqlFeature::GeoSparql).await;
        assert_eq!(geo.len(), 1);
        assert_eq!(geo[0].id, "ep1");
    }

    // ── Failover ──────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_failover_picks_healthy_replacement() {
        let config = TopologyConfig {
            failover_cooldown: Duration::from_millis(0),
            ..Default::default()
        };
        let mgr = FederationTopologyManager::with_config(config);

        let mut good = make_endpoint("good", "http://good.example/sparql");
        good.status = EndpointStatus::Healthy;
        mgr.register_endpoint(good).await.unwrap();

        let bad = make_endpoint("bad", "http://bad.example/sparql");
        mgr.register_endpoint(bad).await.unwrap();

        let replacement = mgr.trigger_failover("bad").await;
        assert_eq!(replacement, Some("good".to_string()));
    }

    #[tokio::test]
    async fn test_failover_returns_none_when_no_replacement() {
        let config = TopologyConfig {
            failover_cooldown: Duration::from_millis(0),
            ..Default::default()
        };
        let mgr = FederationTopologyManager::with_config(config);

        let bad = make_endpoint("only", "http://only.example/sparql");
        mgr.register_endpoint(bad).await.unwrap();

        let replacement = mgr.trigger_failover("only").await;
        assert!(replacement.is_none());
    }

    #[tokio::test]
    async fn test_failover_prefers_higher_priority() {
        let config = TopologyConfig {
            failover_cooldown: Duration::from_millis(0),
            ..Default::default()
        };
        let mgr = FederationTopologyManager::with_config(config);

        let mut low = make_endpoint("low", "http://low.example/sparql");
        low.status = EndpointStatus::Healthy;
        low.priority = 50;
        mgr.register_endpoint(low).await.unwrap();

        let mut high = make_endpoint("high", "http://high.example/sparql");
        high.status = EndpointStatus::Healthy;
        high.priority = 200;
        mgr.register_endpoint(high).await.unwrap();

        let bad = make_endpoint("bad", "http://bad.example/sparql");
        mgr.register_endpoint(bad).await.unwrap();

        let replacement = mgr.trigger_failover("bad").await;
        assert_eq!(replacement, Some("high".to_string()));
    }

    // ── Event log ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_events_recorded() {
        let mgr = FederationTopologyManager::new();
        mgr.register_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await
            .unwrap();
        let events = mgr.events().await;
        assert!(!events.is_empty());
        assert!(matches!(
            events[0],
            TopologyEvent::EndpointRegistered { .. }
        ));
    }

    #[tokio::test]
    async fn test_clear_events() {
        let mgr = FederationTopologyManager::new();
        mgr.register_endpoint(make_endpoint("ep1", "http://a.example/sparql"))
            .await
            .unwrap();
        mgr.clear_events().await;
        assert!(mgr.events().await.is_empty());
    }

    #[tokio::test]
    async fn test_endpoint_info_is_available() {
        let mut ep = make_endpoint("ep1", "http://a.example/sparql");
        ep.status = EndpointStatus::Healthy;
        assert!(ep.is_available());

        ep.status = EndpointStatus::Unavailable;
        assert!(!ep.is_available());

        ep.status = EndpointStatus::Healthy;
        ep.active = false;
        assert!(!ep.is_available());
    }
}
