//! SPARQL endpoint discovery and capability registry.
//!
//! Maintains a registry of known SPARQL endpoints with their capabilities,
//! supported graphs and predicates, freshness tracking, and stale eviction.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Metadata describing a discovered SPARQL endpoint.
#[derive(Debug, Clone)]
pub struct EndpointInfo {
    /// The base URL of the endpoint (used as the registry key).
    pub url: String,
    /// Human-readable name for the endpoint.
    pub name: String,
    /// Named graphs exposed by this endpoint.
    pub graphs: Vec<String>,
    /// Predicates known to be present in the endpoint's data.
    pub predicates: Vec<String>,
    /// Whether the endpoint accepts SPARQL Update (write) operations.
    pub supports_update: bool,
    /// Optional cap on the number of results the endpoint will return.
    pub max_result_size: Option<usize>,
    /// Millisecond timestamp of the last successful contact.
    pub last_seen_ms: u64,
}

/// Configuration for an [`EndpointRegistry`].
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// After how many milliseconds without an update an endpoint is considered
    /// stale and eligible for eviction.
    pub ttl_ms: u64,
    /// Maximum number of endpoints the registry may hold.
    pub max_endpoints: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            ttl_ms: 300_000, // 5 minutes
            max_endpoints: 1024,
        }
    }
}

/// Lifecycle status of a registered endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndpointStatus {
    /// The endpoint is active and was recently seen.
    Active,
    /// The endpoint has not been seen for longer than the TTL.
    Stale,
    /// The endpoint has been explicitly removed.
    Removed,
}

/// A registry of SPARQL endpoints with capability-based lookup.
pub struct EndpointRegistry {
    endpoints: HashMap<String, (EndpointInfo, EndpointStatus)>,
    config: DiscoveryConfig,
}

impl EndpointRegistry {
    /// Create a new, empty registry.
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            endpoints: HashMap::new(),
            config,
        }
    }

    /// Register or update an endpoint.
    ///
    /// Returns `true` when the endpoint is newly registered, `false` when it
    /// is an update of an existing entry.
    ///
    /// If the registry has reached `max_endpoints` and the URL is not already
    /// known, the registration is silently ignored and `false` is returned.
    pub fn register(&mut self, info: EndpointInfo) -> bool {
        let url = info.url.clone();
        if let Some(existing) = self.endpoints.get_mut(&url) {
            existing.0 = info;
            existing.1 = EndpointStatus::Active;
            false
        } else {
            if self.endpoints.len() >= self.config.max_endpoints {
                return false;
            }
            self.endpoints.insert(url, (info, EndpointStatus::Active));
            true
        }
    }

    /// Retrieve an endpoint's info by URL (regardless of status).
    pub fn get(&self, url: &str) -> Option<&EndpointInfo> {
        self.endpoints.get(url).map(|(info, _)| info)
    }

    /// Return all endpoints that are currently `Active`.
    pub fn active_endpoints(&self) -> Vec<&EndpointInfo> {
        self.endpoints
            .values()
            .filter(|(_, s)| *s == EndpointStatus::Active)
            .map(|(info, _)| info)
            .collect()
    }

    /// Return all active endpoints that expose the given named graph.
    pub fn endpoints_for_graph(&self, graph: &str) -> Vec<&EndpointInfo> {
        self.endpoints
            .values()
            .filter(|(info, status)| {
                *status == EndpointStatus::Active && info.graphs.iter().any(|g| g == graph)
            })
            .map(|(info, _)| info)
            .collect()
    }

    /// Return all active endpoints that contain the given predicate.
    pub fn endpoints_for_predicate(&self, predicate: &str) -> Vec<&EndpointInfo> {
        self.endpoints
            .values()
            .filter(|(info, status)| {
                *status == EndpointStatus::Active && info.predicates.iter().any(|p| p == predicate)
            })
            .map(|(info, _)| info)
            .collect()
    }

    /// Mark an endpoint as `Stale`.
    ///
    /// Returns `true` if the endpoint was found and updated, `false` otherwise.
    pub fn mark_stale(&mut self, url: &str) -> bool {
        if let Some(entry) = self.endpoints.get_mut(url) {
            entry.1 = EndpointStatus::Stale;
            true
        } else {
            false
        }
    }

    /// Remove all endpoints whose `last_seen_ms` is older than
    /// `now_ms - config.ttl_ms`, i.e. entries that have gone stale by time.
    ///
    /// Returns the number of entries evicted.
    pub fn evict_stale(&mut self, now_ms: u64) -> usize {
        let ttl = self.config.ttl_ms;
        let before = self.endpoints.len();
        self.endpoints.retain(|_, (info, status)| {
            let age = now_ms.saturating_sub(info.last_seen_ms);
            if age >= ttl {
                *status = EndpointStatus::Stale;
            }
            // Keep Active; remove anything that became Stale or was Removed
            *status == EndpointStatus::Active
        });
        before - self.endpoints.len()
    }

    /// Total number of entries in the registry (active, stale, or removed).
    pub fn count(&self) -> usize {
        self.endpoints.len()
    }

    /// Update the `last_seen_ms` timestamp for an endpoint.
    ///
    /// If the endpoint is currently `Stale` this also reactivates it.
    pub fn update_seen(&mut self, url: &str, now_ms: u64) {
        if let Some((info, status)) = self.endpoints.get_mut(url) {
            info.last_seen_ms = now_ms;
            *status = EndpointStatus::Active;
        }
    }

    /// Access the registry's configuration.
    pub fn config(&self) -> &DiscoveryConfig {
        &self.config
    }

    /// Return the status of a specific endpoint, or `None` if unknown.
    pub fn status_of(&self, url: &str) -> Option<&EndpointStatus> {
        self.endpoints.get(url).map(|(_, s)| s)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Construct a minimal `EndpointInfo` for use in tests.
#[cfg(test)]
fn endpoint(url: &str, name: &str, last_seen_ms: u64) -> EndpointInfo {
    EndpointInfo {
        url: url.to_string(),
        name: name.to_string(),
        graphs: Vec::new(),
        predicates: Vec::new(),
        supports_update: false,
        max_result_size: None,
        last_seen_ms,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(ttl_ms: u64, max_endpoints: usize) -> DiscoveryConfig {
        DiscoveryConfig {
            ttl_ms,
            max_endpoints,
        }
    }

    fn ep_with_graphs(url: &str, graphs: &[&str], now: u64) -> EndpointInfo {
        EndpointInfo {
            url: url.to_string(),
            name: url.to_string(),
            graphs: graphs.iter().map(|s| s.to_string()).collect(),
            predicates: Vec::new(),
            supports_update: false,
            max_result_size: None,
            last_seen_ms: now,
        }
    }

    fn ep_with_predicates(url: &str, preds: &[&str], now: u64) -> EndpointInfo {
        EndpointInfo {
            url: url.to_string(),
            name: url.to_string(),
            graphs: Vec::new(),
            predicates: preds.iter().map(|s| s.to_string()).collect(),
            supports_update: false,
            max_result_size: None,
            last_seen_ms: now,
        }
    }

    // --- register ---

    #[test]
    fn test_register_new_returns_true() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        let result = reg.register(endpoint("http://a.example/sparql", "A", 1000));
        assert!(result);
    }

    #[test]
    fn test_register_update_returns_false() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a.example/sparql", "A", 1000));
        let result = reg.register(endpoint("http://a.example/sparql", "A-updated", 2000));
        assert!(!result);
    }

    #[test]
    fn test_register_stores_info() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a.example/sparql", "Alpha", 1000));
        let info = reg.get("http://a.example/sparql").expect("should exist");
        assert_eq!(info.name, "Alpha");
    }

    #[test]
    fn test_register_update_overwrites_name() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a.example/sparql", "old", 1000));
        reg.register(endpoint("http://a.example/sparql", "new", 2000));
        assert_eq!(reg.get("http://a.example/sparql").unwrap().name, "new");
    }

    #[test]
    fn test_register_at_capacity_ignored() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 2));
        reg.register(endpoint("http://a/s", "A", 0));
        reg.register(endpoint("http://b/s", "B", 0));
        // At capacity — third registration should be dropped
        let result = reg.register(endpoint("http://c/s", "C", 0));
        assert!(!result);
        assert!(reg.get("http://c/s").is_none());
    }

    // --- get ---

    #[test]
    fn test_get_unknown_url_returns_none() {
        let reg = EndpointRegistry::new(cfg(60_000, 100));
        assert!(reg.get("http://unknown/sparql").is_none());
    }

    #[test]
    fn test_get_returns_correct_info() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://e1/sparql", "E1", 999));
        let info = reg.get("http://e1/sparql").expect("exists");
        assert_eq!(info.url, "http://e1/sparql");
        assert_eq!(info.last_seen_ms, 999);
    }

    // --- active_endpoints ---

    #[test]
    fn test_active_endpoints_initially_all() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        reg.register(endpoint("http://b/s", "B", 0));
        assert_eq!(reg.active_endpoints().len(), 2);
    }

    #[test]
    fn test_active_endpoints_excludes_stale() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        reg.register(endpoint("http://b/s", "B", 0));
        reg.mark_stale("http://a/s");
        assert_eq!(reg.active_endpoints().len(), 1);
    }

    #[test]
    fn test_active_endpoints_empty_registry() {
        let reg = EndpointRegistry::new(cfg(60_000, 100));
        assert!(reg.active_endpoints().is_empty());
    }

    // --- endpoints_for_graph ---

    #[test]
    fn test_endpoints_for_graph_match() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(ep_with_graphs("http://a/s", &["urn:g1", "urn:g2"], 0));
        reg.register(ep_with_graphs("http://b/s", &["urn:g2"], 0));
        let result = reg.endpoints_for_graph("urn:g1");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].url, "http://a/s");
    }

    #[test]
    fn test_endpoints_for_graph_no_match() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(ep_with_graphs("http://a/s", &["urn:g1"], 0));
        assert!(reg.endpoints_for_graph("urn:g99").is_empty());
    }

    #[test]
    fn test_endpoints_for_graph_excludes_stale() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(ep_with_graphs("http://a/s", &["urn:g1"], 0));
        reg.mark_stale("http://a/s");
        assert!(reg.endpoints_for_graph("urn:g1").is_empty());
    }

    #[test]
    fn test_endpoints_for_graph_multiple_matches() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(ep_with_graphs("http://a/s", &["urn:shared"], 0));
        reg.register(ep_with_graphs("http://b/s", &["urn:shared"], 0));
        reg.register(ep_with_graphs("http://c/s", &["urn:other"], 0));
        let result = reg.endpoints_for_graph("urn:shared");
        assert_eq!(result.len(), 2);
    }

    // --- endpoints_for_predicate ---

    #[test]
    fn test_endpoints_for_predicate_match() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(ep_with_predicates(
            "http://a/s",
            &["http://schema.org/name"],
            0,
        ));
        let result = reg.endpoints_for_predicate("http://schema.org/name");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_endpoints_for_predicate_no_match() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(ep_with_predicates(
            "http://a/s",
            &["http://schema.org/age"],
            0,
        ));
        assert!(reg
            .endpoints_for_predicate("http://schema.org/name")
            .is_empty());
    }

    #[test]
    fn test_endpoints_for_predicate_excludes_stale() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(ep_with_predicates("http://a/s", &["http://p/x"], 0));
        reg.mark_stale("http://a/s");
        assert!(reg.endpoints_for_predicate("http://p/x").is_empty());
    }

    // --- mark_stale ---

    #[test]
    fn test_mark_stale_returns_true_for_existing() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        assert!(reg.mark_stale("http://a/s"));
    }

    #[test]
    fn test_mark_stale_returns_false_for_unknown() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        assert!(!reg.mark_stale("http://unknown/sparql"));
    }

    #[test]
    fn test_mark_stale_changes_status() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        reg.mark_stale("http://a/s");
        assert_eq!(reg.status_of("http://a/s"), Some(&EndpointStatus::Stale));
    }

    // --- evict_stale ---

    #[test]
    fn test_evict_stale_removes_old_entries() {
        let mut reg = EndpointRegistry::new(cfg(1_000, 100));
        reg.register(endpoint("http://old/s", "old", 0)); // age = 5000 > ttl 1000
        reg.register(endpoint("http://fresh/s", "fresh", 4_500)); // age = 500 < ttl
        let removed = reg.evict_stale(5_000);
        assert_eq!(removed, 1);
        assert!(reg.get("http://old/s").is_none());
        assert!(reg.get("http://fresh/s").is_some());
    }

    #[test]
    fn test_evict_stale_empty_registry() {
        let mut reg = EndpointRegistry::new(cfg(1_000, 100));
        assert_eq!(reg.evict_stale(9999), 0);
    }

    #[test]
    fn test_evict_stale_all_fresh() {
        let mut reg = EndpointRegistry::new(cfg(100_000, 100));
        reg.register(endpoint("http://a/s", "A", 9_000));
        reg.register(endpoint("http://b/s", "B", 9_001));
        assert_eq!(reg.evict_stale(9_500), 0);
        assert_eq!(reg.count(), 2);
    }

    #[test]
    fn test_evict_stale_decrements_count() {
        let mut reg = EndpointRegistry::new(cfg(500, 100));
        for i in 0..5u64 {
            reg.register(endpoint(&format!("http://e{i}/s"), "x", i * 10));
        }
        reg.evict_stale(10_000);
        assert_eq!(reg.count(), 0);
    }

    // --- count ---

    #[test]
    fn test_count_starts_at_zero() {
        let reg = EndpointRegistry::new(cfg(60_000, 100));
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_count_increments_on_register() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        assert_eq!(reg.count(), 1);
        reg.register(endpoint("http://b/s", "B", 0));
        assert_eq!(reg.count(), 2);
    }

    #[test]
    fn test_count_not_incremented_on_update() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        reg.register(endpoint("http://a/s", "A-updated", 1));
        assert_eq!(reg.count(), 1);
    }

    // --- update_seen ---

    #[test]
    fn test_update_seen_refreshes_timestamp() {
        let mut reg = EndpointRegistry::new(cfg(1_000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        reg.update_seen("http://a/s", 9_000);
        let info = reg.get("http://a/s").expect("exists");
        assert_eq!(info.last_seen_ms, 9_000);
    }

    #[test]
    fn test_update_seen_reactivates_stale() {
        let mut reg = EndpointRegistry::new(cfg(1_000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        reg.mark_stale("http://a/s");
        reg.update_seen("http://a/s", 9_000);
        assert_eq!(reg.status_of("http://a/s"), Some(&EndpointStatus::Active));
    }

    #[test]
    fn test_update_seen_unknown_url_no_panic() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        // Should not panic when the URL is not registered
        reg.update_seen("http://unknown/sparql", 1000);
    }

    // --- supports_update / max_result_size ---

    #[test]
    fn test_supports_update_field_stored() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(EndpointInfo {
            url: "http://writable/sparql".to_string(),
            name: "writable".to_string(),
            graphs: Vec::new(),
            predicates: Vec::new(),
            supports_update: true,
            max_result_size: Some(10_000),
            last_seen_ms: 0,
        });
        let info = reg.get("http://writable/sparql").expect("exists");
        assert!(info.supports_update);
        assert_eq!(info.max_result_size, Some(10_000));
    }

    // --- status_of ---

    #[test]
    fn test_status_of_active() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        assert_eq!(reg.status_of("http://a/s"), Some(&EndpointStatus::Active));
    }

    #[test]
    fn test_status_of_stale() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        reg.mark_stale("http://a/s");
        assert_eq!(reg.status_of("http://a/s"), Some(&EndpointStatus::Stale));
    }

    #[test]
    fn test_status_of_unknown() {
        let reg = EndpointRegistry::new(cfg(60_000, 100));
        assert!(reg.status_of("http://unknown/sparql").is_none());
    }

    // --- config ---

    #[test]
    fn test_config_accessor() {
        let config = cfg(1234, 42);
        let reg = EndpointRegistry::new(config);
        assert_eq!(reg.config().ttl_ms, 1234);
        assert_eq!(reg.config().max_endpoints, 42);
    }

    #[test]
    fn test_default_config() {
        let dc = DiscoveryConfig::default();
        assert!(dc.ttl_ms > 0);
        assert!(dc.max_endpoints > 0);
    }

    // --- combined scenarios ---

    #[test]
    fn test_reregister_stale_reactivates() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        reg.mark_stale("http://a/s");
        reg.register(endpoint("http://a/s", "A-fresh", 5000));
        assert_eq!(reg.status_of("http://a/s"), Some(&EndpointStatus::Active));
    }

    #[test]
    fn test_evict_then_register_new() {
        let mut reg = EndpointRegistry::new(cfg(500, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        reg.evict_stale(1000); // age = 1000 >= ttl 500 → evicted
        assert_eq!(reg.count(), 0);
        reg.register(endpoint("http://b/s", "B", 1000));
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_endpoint_status_equality() {
        assert_eq!(EndpointStatus::Active, EndpointStatus::Active);
        assert_ne!(EndpointStatus::Active, EndpointStatus::Stale);
        assert_ne!(EndpointStatus::Stale, EndpointStatus::Removed);
    }

    #[test]
    fn test_endpoint_info_predicates_and_graphs() {
        let info = EndpointInfo {
            url: "http://x/s".to_string(),
            name: "X".to_string(),
            graphs: vec!["urn:g1".to_string(), "urn:g2".to_string()],
            predicates: vec!["http://p/a".to_string()],
            supports_update: false,
            max_result_size: None,
            last_seen_ms: 0,
        };
        assert_eq!(info.graphs.len(), 2);
        assert_eq!(info.predicates.len(), 1);
    }

    // --- additional coverage ---

    #[test]
    fn test_update_seen_prevents_eviction() {
        let mut reg = EndpointRegistry::new(cfg(500, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        // Refresh before evict
        reg.update_seen("http://a/s", 800); // age = 1000-800=200 < ttl=500 → survives
        let evicted = reg.evict_stale(1000);
        assert_eq!(evicted, 0);
        assert!(reg.get("http://a/s").is_some());
    }

    #[test]
    fn test_register_at_exact_capacity_updates_existing() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 2));
        reg.register(endpoint("http://a/s", "A", 0));
        reg.register(endpoint("http://b/s", "B", 0));
        // Registry full — but updating an existing URL should succeed
        let updated = reg.register(endpoint("http://a/s", "A-new", 1000));
        assert!(!updated); // update returns false
        assert_eq!(reg.get("http://a/s").unwrap().name, "A-new");
        assert_eq!(reg.count(), 2);
    }

    #[test]
    fn test_endpoints_for_predicate_multiple_matches() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(ep_with_predicates(
            "http://a/s",
            &["http://p/x", "http://p/y"],
            0,
        ));
        reg.register(ep_with_predicates("http://b/s", &["http://p/x"], 0));
        let result = reg.endpoints_for_predicate("http://p/x");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_evict_stale_at_exact_ttl_boundary() {
        // age == ttl → should be evicted (>= ttl)
        let mut reg = EndpointRegistry::new(cfg(1000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        // age = 1000 == ttl=1000 → evicted
        let evicted = reg.evict_stale(1000);
        assert_eq!(evicted, 1);
        assert!(reg.get("http://a/s").is_none());
    }

    #[test]
    fn test_mark_stale_does_not_remove_from_get() {
        let mut reg = EndpointRegistry::new(cfg(60_000, 100));
        reg.register(endpoint("http://a/s", "A", 0));
        reg.mark_stale("http://a/s");
        // Stale entries still accessible via get()
        assert!(reg.get("http://a/s").is_some());
        // But not in active list
        assert!(reg.active_endpoints().is_empty());
    }
}
