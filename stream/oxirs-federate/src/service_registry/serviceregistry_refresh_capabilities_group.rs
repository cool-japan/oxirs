//! # ServiceRegistry - refresh_capabilities_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! Provides the real service-discovery refresh used by
//! `FederationEngine::discover_services()`: it re-runs live capability
//! detection (SPARQL Service Description probing, GraphQL introspection)
//! against every currently registered service and updates the registry in
//! place, instead of being a disabled no-op.

use super::serviceregistry_type::ServiceRegistry;
use anyhow::Result;
use tracing::{debug, warn};

/// Summary of a `ServiceRegistry::refresh_capabilities()` run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilityRefreshStats {
    /// Total number of registered services (SPARQL + GraphQL) considered.
    pub total_services: usize,
    /// Number of services whose capabilities were successfully re-detected.
    pub refreshed: usize,
    /// Number of services for which re-detection failed (they keep their
    /// last-known capabilities rather than being dropped).
    pub failed: usize,
}

impl ServiceRegistry {
    /// Refresh capability information for every registered service by
    /// re-running live discovery against it.
    ///
    /// This is the real implementation backing `FederationEngine::discover_services()`.
    /// For each registered SPARQL endpoint it re-runs `detect_sparql_capabilities`;
    /// for each registered GraphQL service it re-runs `introspect_graphql_service`.
    /// A single unreachable service does not abort the whole refresh — its
    /// failure is logged and counted, and its previously known capabilities
    /// are left untouched, mirroring how `health_check` tolerates individual
    /// service outages without failing the whole registry.
    pub async fn refresh_capabilities(&self) -> Result<CapabilityRefreshStats> {
        let mut stats = CapabilityRefreshStats::default();

        // Snapshot the current SPARQL endpoints so we don't hold the DashMap
        // shard locks across `.await` points.
        let sparql_ids: Vec<String> = self
            .sparql_endpoints
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        for service_id in sparql_ids {
            stats.total_services += 1;
            let endpoint = match self.sparql_endpoints.get(&service_id) {
                Some(entry) => entry.value().clone(),
                None => continue, // Removed concurrently; nothing to refresh.
            };
            match self.detect_sparql_capabilities(&endpoint).await {
                Ok(capabilities) => {
                    if let Some(mut entry) = self.sparql_endpoints.get_mut(&service_id) {
                        entry.value_mut().capabilities = capabilities;
                    }
                    stats.refreshed += 1;
                    debug!("Refreshed SPARQL endpoint capabilities: {}", service_id);
                }
                Err(e) => {
                    stats.failed += 1;
                    warn!(
                        "Failed to refresh capabilities for SPARQL endpoint '{}': {}",
                        service_id, e
                    );
                }
            }
        }

        // Same pattern for GraphQL services.
        let graphql_ids: Vec<String> = self
            .graphql_services
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        for service_id in graphql_ids {
            stats.total_services += 1;
            let service = match self.graphql_services.get(&service_id) {
                Some(entry) => entry.value().clone(),
                None => continue,
            };
            match self.introspect_graphql_service(&service).await {
                Ok((capabilities, schema)) => {
                    if let Some(mut entry) = self.graphql_services.get_mut(&service_id) {
                        let value = entry.value_mut();
                        value.capabilities = capabilities;
                        value.schema = schema;
                    }
                    stats.refreshed += 1;
                    debug!("Refreshed GraphQL service capabilities: {}", service_id);
                }
                Err(e) => {
                    stats.failed += 1;
                    warn!(
                        "Failed to refresh capabilities for GraphQL service '{}': {}",
                        service_id, e
                    );
                }
            }
        }

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{
        ConnectionConfig, PerformanceStats, RegistryConfig, SparqlCapabilities, SparqlEndpoint,
        SparqlVersion,
    };
    use super::super::ServiceRegistry;
    use chrono::Utc;
    use std::collections::{HashMap, HashSet};
    use url::Url;

    /// Reserve an ephemeral port, then immediately release it without ever
    /// accepting a connection: this deterministically guarantees nothing is
    /// listening there when the caller dials out.
    fn unused_port() -> u16 {
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("binding an ephemeral port");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);
        port
    }

    fn unreachable_endpoint(id: &str, port: u16) -> SparqlEndpoint {
        SparqlEndpoint {
            id: id.to_string(),
            name: "Test endpoint".to_string(),
            // "localhost" is on register_sparql_endpoint's bypass list, so
            // registration succeeds without needing a live server; the
            // refresh path below has no such bypass, so it genuinely
            // attempts (and fails) a real probe.
            url: Url::parse(&format!("http://localhost:{port}/sparql")).expect("valid URL"),
            auth: None,
            capabilities: SparqlCapabilities {
                sparql_version: SparqlVersion::V11,
                result_formats: HashSet::new(),
                graph_formats: HashSet::new(),
                custom_functions: HashSet::new(),
                max_query_complexity: None,
                supports_federation: false,
                supports_update: false,
                supports_named_graphs: false,
                supports_full_text_search: false,
                supports_geospatial: false,
                supports_rdf_star: false,
                service_description: None,
            },
            statistics: PerformanceStats::default(),
            registered_at: Utc::now(),
            last_access: None,
            metadata: HashMap::new(),
            connection_config: ConnectionConfig::default(),
        }
    }

    /// Regression test for stream/oxirs-federate/src/lib.rs:841 —
    /// `discover_services()` used to be a disabled no-op (`warn!` + `Ok(())`)
    /// that never touched the registry at all. This verifies the real
    /// refresh path actually runs per-service capability detection: every
    /// registered service is genuinely considered (`total_services` reflects
    /// registrations, not just a constant `Ok(())`), and the (unreachable)
    /// endpoint's capabilities are actually re-detected in place rather than
    /// left untouched.
    ///
    /// Note: `detect_sparql_capabilities`'s individual probes are themselves
    /// designed to degrade gracefully on network failure (each sub-probe
    /// treats "can't connect" as "capability not supported" rather than a
    /// hard error), so an unreachable endpoint here is counted as
    /// successfully *assessed* (with defaulted/negative capabilities), not as
    /// a `refresh_capabilities`-level failure.
    #[tokio::test]
    async fn test_refresh_capabilities_actually_probes_registered_services() {
        let registry = ServiceRegistry::with_config(RegistryConfig::default());
        registry
            .register_sparql_endpoint(unreachable_endpoint(
                "regression-test-endpoint",
                unused_port(),
            ))
            .await
            .expect("registration against the bypassed 'localhost' host should succeed");

        let stats = registry
            .refresh_capabilities()
            .await
            .expect("refresh_capabilities should complete even against an unreachable endpoint");

        assert_eq!(
            stats.total_services, 1,
            "the registered endpoint must actually be considered for refresh"
        );
        assert_eq!(
            stats.refreshed, 1,
            "capability detection must actually run (and complete) for the registered endpoint"
        );
        assert_eq!(stats.failed, 0);

        // The endpoint's SPARQL version must have been genuinely
        // (re-)detected — a V10 fallback, since nothing answered any of the
        // version-probing queries — proving real detection ran rather than
        // the call being a no-op.
        let refreshed_endpoint = registry
            .get_sparql_endpoints()
            .into_iter()
            .find(|e| e.id == "regression-test-endpoint")
            .expect("endpoint should still be registered");
        assert!(matches!(
            refreshed_endpoint.capabilities.sparql_version,
            SparqlVersion::V10
        ));
    }
}
