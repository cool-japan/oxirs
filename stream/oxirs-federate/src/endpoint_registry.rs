//! Federated SPARQL endpoint registry.
//!
//! Tracks available SPARQL endpoints, their health, capabilities, priority, and
//! tags.  Provides helpers for selecting the best active endpoint and detecting
//! stale health-check data.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Operational status of a federated endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndpointStatus {
    /// Endpoint is reachable and operating normally.
    Active,
    /// Endpoint is intentionally not serving requests.
    Inactive,
    /// Endpoint is reachable but experiencing degraded performance.
    Degraded,
    /// Status has never been checked or the check result is unavailable.
    Unknown,
}

/// Most-recent health observation for an endpoint.
#[derive(Debug, Clone)]
pub struct EndpointHealth {
    /// Current operational status.
    pub status: EndpointStatus,
    /// Unix epoch milliseconds of the last health check.
    pub last_check_ms: u64,
    /// Round-trip latency in ms observed during the last check, if available.
    pub latency_ms: Option<u32>,
    /// Fraction of recent requests that ended in errors (0.0 – 1.0).
    pub error_rate: f32,
}

impl Default for EndpointHealth {
    fn default() -> Self {
        Self {
            status: EndpointStatus::Unknown,
            last_check_ms: 0,
            latency_ms: None,
            error_rate: 0.0,
        }
    }
}

/// Feature flags reported by a federated endpoint.
#[derive(Debug, Clone, Default)]
pub struct EndpointCapabilities {
    /// Endpoint supports SPARQL 1.1 query and protocol.
    pub supports_sparql_11: bool,
    /// Endpoint accepts SPARQL Update (SPARQL 1.1 Update).
    pub supports_update: bool,
    /// Endpoint supports SPARQL federation via the SERVICE keyword.
    pub supports_federation: bool,
    /// Number of named graphs available, if reported.
    pub graph_count: Option<usize>,
    /// Estimated total triple count, if reported.
    pub triple_count_estimate: Option<u64>,
}

/// A single registered federated SPARQL endpoint.
#[derive(Debug, Clone)]
pub struct FederatedEndpoint {
    /// Unique registry identifier.
    pub id: String,
    /// SPARQL query endpoint URL.
    pub sparql_url: String,
    /// Human-readable label.
    pub label: Option<String>,
    /// Most-recent health observation.
    pub health: EndpointHealth,
    /// Reported capabilities.
    pub capabilities: EndpointCapabilities,
    /// Priority for selection (higher = preferred).
    pub priority: i32,
    /// Arbitrary categorisation tags.
    pub tags: Vec<String>,
}

/// Errors returned by registry operations.
#[derive(Debug)]
pub enum RegistryError {
    /// An endpoint with the same ID is already registered.
    DuplicateId(String),
    /// The provided SPARQL URL is invalid (empty for now).
    InvalidUrl(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::DuplicateId(id) => write!(f, "duplicate endpoint id: {id}"),
            RegistryError::InvalidUrl(url) => write!(f, "invalid endpoint URL: {url}"),
        }
    }
}

impl std::error::Error for RegistryError {}

// ─────────────────────────────────────────────────────────────────────────────
// EndpointRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory registry of federated SPARQL endpoints.
pub struct EndpointRegistry {
    endpoints: HashMap<String, FederatedEndpoint>,
}

impl Default for EndpointRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl EndpointRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
        }
    }

    /// Register a new endpoint.
    ///
    /// Fails with [`RegistryError::DuplicateId`] if an endpoint with the same
    /// `id` already exists, or [`RegistryError::InvalidUrl`] if the SPARQL URL
    /// is empty.
    pub fn register(&mut self, endpoint: FederatedEndpoint) -> Result<(), RegistryError> {
        if endpoint.sparql_url.is_empty() {
            return Err(RegistryError::InvalidUrl(endpoint.sparql_url));
        }
        if self.endpoints.contains_key(&endpoint.id) {
            return Err(RegistryError::DuplicateId(endpoint.id));
        }
        self.endpoints.insert(endpoint.id.clone(), endpoint);
        Ok(())
    }

    /// Remove an endpoint by ID.
    ///
    /// Returns `true` if the endpoint was present and has been removed.
    pub fn deregister(&mut self, id: &str) -> bool {
        self.endpoints.remove(id).is_some()
    }

    /// Look up an endpoint by ID.
    pub fn get(&self, id: &str) -> Option<&FederatedEndpoint> {
        self.endpoints.get(id)
    }

    /// Replace the health record for an endpoint.
    ///
    /// Returns `true` if the endpoint was found, `false` otherwise.
    pub fn update_health(&mut self, id: &str, health: EndpointHealth) -> bool {
        match self.endpoints.get_mut(id) {
            Some(ep) => {
                ep.health = health;
                true
            }
            None => false,
        }
    }

    /// Return all endpoints whose current status is [`EndpointStatus::Active`].
    pub fn active_endpoints(&self) -> Vec<&FederatedEndpoint> {
        self.endpoints
            .values()
            .filter(|ep| ep.health.status == EndpointStatus::Active)
            .collect()
    }

    /// Return all endpoints that carry the given `tag`.
    pub fn endpoints_by_tag(&self, tag: &str) -> Vec<&FederatedEndpoint> {
        self.endpoints
            .values()
            .filter(|ep| ep.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Return the highest-priority [`EndpointStatus::Active`] endpoint.
    ///
    /// When multiple active endpoints share the same priority, the one that
    /// was inserted first (arbitrary HashMap order) is chosen.
    pub fn best_endpoint(&self) -> Option<&FederatedEndpoint> {
        self.endpoints
            .values()
            .filter(|ep| ep.health.status == EndpointStatus::Active)
            .max_by_key(|ep| ep.priority)
    }

    /// Total number of registered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }

    /// Return endpoints whose `last_check_ms` is older than
    /// `current_time_ms - max_age_ms`.
    pub fn stale_endpoints(
        &self,
        current_time_ms: u64,
        max_age_ms: u64,
    ) -> Vec<&FederatedEndpoint> {
        let cutoff = current_time_ms.saturating_sub(max_age_ms);
        self.endpoints
            .values()
            .filter(|ep| ep.health.last_check_ms < cutoff)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_endpoint(id: &str, priority: i32, status: EndpointStatus) -> FederatedEndpoint {
        FederatedEndpoint {
            id: id.to_string(),
            sparql_url: format!("http://example.org/{id}/sparql"),
            label: Some(format!("Endpoint {id}")),
            health: EndpointHealth {
                status,
                last_check_ms: 1_000,
                latency_ms: Some(10),
                error_rate: 0.0,
            },
            capabilities: EndpointCapabilities {
                supports_sparql_11: true,
                supports_update: false,
                supports_federation: false,
                graph_count: None,
                triple_count_estimate: None,
            },
            priority,
            tags: Vec::new(),
        }
    }

    // ── register / get ────────────────────────────────────────────────────────

    #[test]
    fn test_register_and_get() {
        let mut reg = EndpointRegistry::new();
        let ep = make_endpoint("ep1", 0, EndpointStatus::Active);
        reg.register(ep).expect("should succeed");
        let found = reg.get("ep1");
        assert!(found.is_some());
        assert_eq!(found.expect("should succeed").id, "ep1");
    }

    #[test]
    fn test_get_nonexistent_returns_none() {
        let reg = EndpointRegistry::new();
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn test_register_duplicate_id_errors() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("ep1", 0, EndpointStatus::Active))
            .expect("should succeed");
        let result = reg.register(make_endpoint("ep1", 1, EndpointStatus::Active));
        assert!(matches!(result, Err(RegistryError::DuplicateId(_))));
    }

    #[test]
    fn test_register_empty_url_errors() {
        let mut reg = EndpointRegistry::new();
        let mut ep = make_endpoint("ep-bad", 0, EndpointStatus::Active);
        ep.sparql_url = String::new();
        let result = reg.register(ep);
        assert!(matches!(result, Err(RegistryError::InvalidUrl(_))));
    }

    // ── deregister ────────────────────────────────────────────────────────────

    #[test]
    fn test_deregister_existing() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("ep1", 0, EndpointStatus::Active))
            .expect("should succeed");
        assert!(reg.deregister("ep1"));
        assert!(reg.get("ep1").is_none());
    }

    #[test]
    fn test_deregister_nonexistent_returns_false() {
        let mut reg = EndpointRegistry::new();
        assert!(!reg.deregister("ghost"));
    }

    #[test]
    fn test_deregister_reduces_count() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("a", 0, EndpointStatus::Active))
            .expect("should succeed");
        reg.register(make_endpoint("b", 0, EndpointStatus::Active))
            .expect("should succeed");
        reg.deregister("a");
        assert_eq!(reg.endpoint_count(), 1);
    }

    // ── update_health ─────────────────────────────────────────────────────────

    #[test]
    fn test_update_health_returns_true() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("ep1", 0, EndpointStatus::Active))
            .expect("should succeed");
        let new_health = EndpointHealth {
            status: EndpointStatus::Degraded,
            last_check_ms: 5_000,
            latency_ms: Some(200),
            error_rate: 0.05,
        };
        assert!(reg.update_health("ep1", new_health));
        assert_eq!(
            reg.get("ep1").expect("should succeed").health.status,
            EndpointStatus::Degraded
        );
    }

    #[test]
    fn test_update_health_nonexistent_returns_false() {
        let mut reg = EndpointRegistry::new();
        assert!(!reg.update_health("ghost", EndpointHealth::default()));
    }

    #[test]
    fn test_update_health_stores_latency() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("ep1", 0, EndpointStatus::Active))
            .expect("should succeed");
        reg.update_health(
            "ep1",
            EndpointHealth {
                status: EndpointStatus::Active,
                last_check_ms: 2_000,
                latency_ms: Some(42),
                error_rate: 0.0,
            },
        );
        assert_eq!(reg.get("ep1").expect("should succeed").health.latency_ms, Some(42));
    }

    // ── active_endpoints ──────────────────────────────────────────────────────

    #[test]
    fn test_active_endpoints_filters_by_status() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("a", 0, EndpointStatus::Active))
            .expect("should succeed");
        reg.register(make_endpoint("b", 0, EndpointStatus::Inactive))
            .expect("should succeed");
        reg.register(make_endpoint("c", 0, EndpointStatus::Degraded))
            .expect("should succeed");
        let active = reg.active_endpoints();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "a");
    }

    #[test]
    fn test_active_endpoints_empty_registry() {
        let reg = EndpointRegistry::new();
        assert!(reg.active_endpoints().is_empty());
    }

    #[test]
    fn test_all_status_variants_recognised() {
        let mut reg = EndpointRegistry::new();
        for (id, status) in [
            ("a", EndpointStatus::Active),
            ("b", EndpointStatus::Inactive),
            ("c", EndpointStatus::Degraded),
            ("d", EndpointStatus::Unknown),
        ] {
            reg.register(make_endpoint(id, 0, status)).expect("should succeed");
        }
        assert_eq!(reg.active_endpoints().len(), 1);
    }

    // ── endpoints_by_tag ──────────────────────────────────────────────────────

    #[test]
    fn test_endpoints_by_tag_match() {
        let mut reg = EndpointRegistry::new();
        let mut ep1 = make_endpoint("ep1", 0, EndpointStatus::Active);
        ep1.tags = vec!["production".into(), "eu-west".into()];
        let mut ep2 = make_endpoint("ep2", 0, EndpointStatus::Active);
        ep2.tags = vec!["staging".into()];
        reg.register(ep1).expect("should succeed");
        reg.register(ep2).expect("should succeed");
        let tagged = reg.endpoints_by_tag("production");
        assert_eq!(tagged.len(), 1);
        assert_eq!(tagged[0].id, "ep1");
    }

    #[test]
    fn test_endpoints_by_tag_no_match() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("ep1", 0, EndpointStatus::Active))
            .expect("should succeed");
        assert!(reg.endpoints_by_tag("nonexistent-tag").is_empty());
    }

    #[test]
    fn test_endpoints_by_tag_multiple() {
        let mut reg = EndpointRegistry::new();
        for id in ["a", "b", "c"] {
            let mut ep = make_endpoint(id, 0, EndpointStatus::Active);
            ep.tags = vec!["shared-tag".into()];
            reg.register(ep).expect("should succeed");
        }
        assert_eq!(reg.endpoints_by_tag("shared-tag").len(), 3);
    }

    // ── best_endpoint ─────────────────────────────────────────────────────────

    #[test]
    fn test_best_endpoint_highest_priority() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("low", 1, EndpointStatus::Active))
            .expect("should succeed");
        reg.register(make_endpoint("high", 10, EndpointStatus::Active))
            .expect("should succeed");
        let best = reg.best_endpoint().expect("should succeed");
        assert_eq!(best.id, "high");
    }

    #[test]
    fn test_best_endpoint_ignores_inactive() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("top-inactive", 100, EndpointStatus::Inactive))
            .expect("should succeed");
        reg.register(make_endpoint("active-low", 1, EndpointStatus::Active))
            .expect("should succeed");
        let best = reg.best_endpoint().expect("should succeed");
        assert_eq!(best.id, "active-low");
    }

    #[test]
    fn test_best_endpoint_no_active_returns_none() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("ep1", 5, EndpointStatus::Inactive))
            .expect("should succeed");
        assert!(reg.best_endpoint().is_none());
    }

    #[test]
    fn test_best_endpoint_empty_registry() {
        let reg = EndpointRegistry::new();
        assert!(reg.best_endpoint().is_none());
    }

    // ── endpoint_count ────────────────────────────────────────────────────────

    #[test]
    fn test_endpoint_count_empty() {
        assert_eq!(EndpointRegistry::new().endpoint_count(), 0);
    }

    #[test]
    fn test_endpoint_count_after_inserts() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("a", 0, EndpointStatus::Active))
            .expect("should succeed");
        reg.register(make_endpoint("b", 0, EndpointStatus::Active))
            .expect("should succeed");
        assert_eq!(reg.endpoint_count(), 2);
    }

    // ── stale_endpoints ───────────────────────────────────────────────────────

    #[test]
    fn test_stale_endpoints_old_check() {
        let mut reg = EndpointRegistry::new();
        let mut ep = make_endpoint("old", 0, EndpointStatus::Active);
        ep.health.last_check_ms = 100;
        reg.register(ep).expect("should succeed");
        // current_time=10_000, max_age=1_000 → cutoff=9_000 → 100 < 9_000 → stale
        let stale = reg.stale_endpoints(10_000, 1_000);
        assert_eq!(stale.len(), 1);
    }

    #[test]
    fn test_stale_endpoints_fresh_check_not_stale() {
        let mut reg = EndpointRegistry::new();
        let mut ep = make_endpoint("fresh", 0, EndpointStatus::Active);
        ep.health.last_check_ms = 9_500; // within max_age
        reg.register(ep).expect("should succeed");
        let stale = reg.stale_endpoints(10_000, 1_000);
        assert!(stale.is_empty());
    }

    #[test]
    fn test_stale_endpoints_multiple() {
        let mut reg = EndpointRegistry::new();
        let times = [100u64, 500, 9_800];
        for (i, &t) in times.iter().enumerate() {
            let mut ep = make_endpoint(&i.to_string(), 0, EndpointStatus::Active);
            ep.health.last_check_ms = t;
            reg.register(ep).expect("should succeed");
        }
        // cutoff = 10_000 - 1_000 = 9_000
        let stale = reg.stale_endpoints(10_000, 1_000);
        assert_eq!(stale.len(), 2); // 100 and 500 are stale; 9_800 is not
    }

    // ── capabilities stored ───────────────────────────────────────────────────

    #[test]
    fn test_capabilities_stored() {
        let mut reg = EndpointRegistry::new();
        let mut ep = make_endpoint("ep-caps", 0, EndpointStatus::Active);
        ep.capabilities = EndpointCapabilities {
            supports_sparql_11: true,
            supports_update: true,
            supports_federation: true,
            graph_count: Some(42),
            triple_count_estimate: Some(1_000_000),
        };
        reg.register(ep).expect("should succeed");
        let caps = &reg.get("ep-caps").expect("should succeed").capabilities;
        assert!(caps.supports_sparql_11);
        assert!(caps.supports_update);
        assert!(caps.supports_federation);
        assert_eq!(caps.graph_count, Some(42));
        assert_eq!(caps.triple_count_estimate, Some(1_000_000));
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_default_health_unknown_status() {
        let h = EndpointHealth::default();
        assert_eq!(h.status, EndpointStatus::Unknown);
    }

    #[test]
    fn test_default_health_zero_last_check() {
        let h = EndpointHealth::default();
        assert_eq!(h.last_check_ms, 0);
    }

    #[test]
    fn test_default_health_no_latency() {
        let h = EndpointHealth::default();
        assert!(h.latency_ms.is_none());
    }

    #[test]
    fn test_update_health_to_inactive() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("ep1", 0, EndpointStatus::Active))
            .expect("should succeed");
        reg.update_health("ep1", EndpointHealth {
            status: EndpointStatus::Inactive,
            last_check_ms: 1_000,
            latency_ms: None,
            error_rate: 0.0,
        });
        assert_eq!(reg.active_endpoints().len(), 0);
    }

    #[test]
    fn test_update_health_to_active() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("ep1", 0, EndpointStatus::Unknown))
            .expect("should succeed");
        reg.update_health("ep1", EndpointHealth {
            status: EndpointStatus::Active,
            last_check_ms: 1_000,
            latency_ms: Some(5),
            error_rate: 0.0,
        });
        assert_eq!(reg.active_endpoints().len(), 1);
    }

    #[test]
    fn test_label_stored() {
        let mut reg = EndpointRegistry::new();
        let mut ep = make_endpoint("ep1", 0, EndpointStatus::Active);
        ep.label = Some("My Label".into());
        reg.register(ep).expect("should succeed");
        assert_eq!(reg.get("ep1").expect("should succeed").label.as_deref(), Some("My Label"));
    }

    #[test]
    fn test_priority_negative() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("neg", -10, EndpointStatus::Active))
            .expect("should succeed");
        let ep = reg.get("neg").expect("should succeed");
        assert_eq!(ep.priority, -10);
    }

    #[test]
    fn test_multiple_tags_per_endpoint() {
        let mut reg = EndpointRegistry::new();
        let mut ep = make_endpoint("ep1", 0, EndpointStatus::Active);
        ep.tags = vec!["tag1".into(), "tag2".into(), "tag3".into()];
        reg.register(ep).expect("should succeed");
        assert_eq!(reg.endpoints_by_tag("tag2").len(), 1);
        assert_eq!(reg.endpoints_by_tag("tag3").len(), 1);
    }

    #[test]
    fn test_endpoint_url_stored() {
        let mut reg = EndpointRegistry::new();
        let mut ep = make_endpoint("ep1", 0, EndpointStatus::Active);
        ep.sparql_url = "https://sparql.example.org/query".into();
        reg.register(ep).expect("should succeed");
        assert_eq!(
            reg.get("ep1").expect("should succeed").sparql_url,
            "https://sparql.example.org/query"
        );
    }

    #[test]
    fn test_stale_endpoints_exact_cutoff_boundary() {
        let mut reg = EndpointRegistry::new();
        let mut ep = make_endpoint("ep1", 0, EndpointStatus::Active);
        // last_check_ms = cutoff exactly (9_000) → not stale (strict <)
        ep.health.last_check_ms = 9_000;
        reg.register(ep).expect("should succeed");
        let stale = reg.stale_endpoints(10_000, 1_000);
        assert!(stale.is_empty());
    }

    #[test]
    fn test_register_five_endpoints() {
        let mut reg = EndpointRegistry::new();
        for i in 0..5 {
            reg.register(make_endpoint(&i.to_string(), i as i32, EndpointStatus::Active))
                .expect("should succeed");
        }
        assert_eq!(reg.endpoint_count(), 5);
    }

    #[test]
    fn test_active_endpoints_count_multiple() {
        let mut reg = EndpointRegistry::new();
        for i in 0..4 {
            reg.register(make_endpoint(&i.to_string(), 0, EndpointStatus::Active))
                .expect("should succeed");
        }
        reg.register(make_endpoint("x", 0, EndpointStatus::Degraded))
            .expect("should succeed");
        assert_eq!(reg.active_endpoints().len(), 4);
    }

    #[test]
    fn test_best_endpoint_degraded_ignored() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("d", 100, EndpointStatus::Degraded))
            .expect("should succeed");
        reg.register(make_endpoint("a", 1, EndpointStatus::Active))
            .expect("should succeed");
        let best = reg.best_endpoint().expect("should succeed");
        assert_eq!(best.id, "a");
    }

    #[test]
    fn test_registry_error_display_duplicate() {
        let e = RegistryError::DuplicateId("my-ep".into());
        assert!(e.to_string().contains("my-ep"));
    }

    #[test]
    fn test_registry_error_display_invalid_url() {
        let e = RegistryError::InvalidUrl(String::new());
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_default_registry_is_empty() {
        let reg = EndpointRegistry::default();
        assert_eq!(reg.endpoint_count(), 0);
    }

    #[test]
    fn test_capabilities_default() {
        let caps = EndpointCapabilities::default();
        assert!(!caps.supports_sparql_11);
        assert!(caps.graph_count.is_none());
    }

    #[test]
    fn test_endpoint_health_error_rate_stored() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("ep1", 0, EndpointStatus::Active))
            .expect("should succeed");
        reg.update_health("ep1", EndpointHealth {
            status: EndpointStatus::Degraded,
            last_check_ms: 1_000,
            latency_ms: None,
            error_rate: 0.25,
        });
        assert!((reg.get("ep1").expect("should succeed").health.error_rate - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_register_then_deregister_then_register_again() {
        let mut reg = EndpointRegistry::new();
        reg.register(make_endpoint("ep1", 0, EndpointStatus::Active))
            .expect("should succeed");
        reg.deregister("ep1");
        // Should be able to register same ID again
        assert!(reg.register(make_endpoint("ep1", 0, EndpointStatus::Active)).is_ok());
    }
}
