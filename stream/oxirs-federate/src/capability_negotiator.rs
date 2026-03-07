//! SPARQL endpoint capability negotiation.
//!
//! Provides types for declaring which SPARQL protocol features an endpoint supports,
//! and a negotiator that determines whether a client's capability requirements are
//! satisfied by a given endpoint.

use std::collections::HashSet;

// ────────────────────────────────────────────────────────────────────────────
// SparqlCapability
// ────────────────────────────────────────────────────────────────────────────

/// SPARQL protocol features an endpoint may support.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SparqlCapability {
    /// SPARQL 1.1 SELECT queries.
    Select11,
    /// SPARQL 1.1 UPDATE (graph management, insert/delete).
    Update11,
    /// SPARQL 1.2 features (RDF-star, annotation syntax, etc.).
    Sparql12,
    /// Named graph support (GRAPH keyword in WHERE and FROM NAMED).
    NamedGraphs,
    /// SERVICE federation clause for delegating sub-queries.
    FederatedQuery,
    /// SPARQL Graph Store HTTP Protocol (GET/PUT/POST/DELETE on graphs).
    GraphStoreProtocol,
    /// Full-text search extension (e.g. Jena Text or Fuseki text:query).
    TextSearch,
    /// GeoSPARQL geometric functions (spatial query extensions).
    GeoSparql,
}

// ────────────────────────────────────────────────────────────────────────────
// EndpointCapabilities
// ────────────────────────────────────────────────────────────────────────────

/// Declared capabilities and constraints of a SPARQL endpoint.
#[derive(Debug, Clone)]
pub struct EndpointCapabilities {
    /// The base URL of the SPARQL endpoint.
    pub endpoint_url: String,
    /// Set of supported capability flags.
    pub capabilities: HashSet<SparqlCapability>,
    /// Optional maximum query execution timeout in milliseconds.
    pub max_query_timeout_ms: Option<u64>,
    /// Optional maximum number of results returned per query.
    pub max_result_limit: Option<usize>,
}

impl EndpointCapabilities {
    /// Create a new `EndpointCapabilities` with no declared capabilities or constraints.
    pub fn new(endpoint_url: &str) -> Self {
        Self {
            endpoint_url: endpoint_url.to_string(),
            capabilities: HashSet::new(),
            max_query_timeout_ms: None,
            max_result_limit: None,
        }
    }

    /// Add a capability flag and return `self` (builder pattern).
    pub fn with_capability(mut self, cap: SparqlCapability) -> Self {
        self.capabilities.insert(cap);
        self
    }

    /// Set the maximum query timeout and return `self` (builder pattern).
    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.max_query_timeout_ms = Some(ms);
        self
    }

    /// Set the maximum result limit and return `self` (builder pattern).
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.max_result_limit = Some(limit);
        self
    }

    /// Return `true` if this endpoint supports the given capability.
    pub fn supports(&self, cap: &SparqlCapability) -> bool {
        self.capabilities.contains(cap)
    }

    /// Return `true` if this endpoint supports **all** of the given capabilities.
    ///
    /// Returns `true` for an empty slice (vacuous satisfaction).
    pub fn supports_all(&self, caps: &[SparqlCapability]) -> bool {
        caps.iter().all(|c| self.capabilities.contains(c))
    }

    /// Return the number of declared capabilities.
    pub fn capability_count(&self) -> usize {
        self.capabilities.len()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// NegotiationResult
// ────────────────────────────────────────────────────────────────────────────

/// The outcome of a capability negotiation between a client's requirements and an endpoint.
#[derive(Debug, Clone)]
pub struct NegotiationResult {
    /// Capabilities that were both required and available on the endpoint.
    pub satisfied: Vec<SparqlCapability>,
    /// Capabilities that were required but not available on the endpoint.
    pub unsatisfied: Vec<SparqlCapability>,
    /// `true` when every required capability is satisfied.
    pub compatible: bool,
}

// ────────────────────────────────────────────────────────────────────────────
// CapabilityNegotiator
// ────────────────────────────────────────────────────────────────────────────

/// Negotiates compatible capabilities between a client's requirements and a SPARQL endpoint.
#[derive(Debug, Clone, Default)]
pub struct CapabilityNegotiator;

impl CapabilityNegotiator {
    /// Create a new `CapabilityNegotiator`.
    pub fn new() -> Self {
        Self
    }

    /// Determine which of the required capabilities are satisfied by `endpoint`.
    ///
    /// The returned `NegotiationResult` lists all satisfied and unsatisfied capabilities,
    /// and sets `compatible` to `true` only when every required capability is available.
    pub fn negotiate(
        &self,
        required: &[SparqlCapability],
        endpoint: &EndpointCapabilities,
    ) -> NegotiationResult {
        let mut satisfied = Vec::new();
        let mut unsatisfied = Vec::new();

        for cap in required {
            if endpoint.supports(cap) {
                satisfied.push(cap.clone());
            } else {
                unsatisfied.push(cap.clone());
            }
        }

        let compatible = unsatisfied.is_empty();
        NegotiationResult {
            satisfied,
            unsatisfied,
            compatible,
        }
    }

    /// Return `true` if the endpoint satisfies all required capabilities.
    pub fn is_compatible(
        &self,
        required: &[SparqlCapability],
        endpoint: &EndpointCapabilities,
    ) -> bool {
        endpoint.supports_all(required)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────

    fn basic_endpoint() -> EndpointCapabilities {
        EndpointCapabilities::new("http://example.org/sparql")
            .with_capability(SparqlCapability::Select11)
            .with_capability(SparqlCapability::NamedGraphs)
            .with_capability(SparqlCapability::GraphStoreProtocol)
    }

    fn full_endpoint() -> EndpointCapabilities {
        EndpointCapabilities::new("http://full.example.org/sparql")
            .with_capability(SparqlCapability::Select11)
            .with_capability(SparqlCapability::Update11)
            .with_capability(SparqlCapability::Sparql12)
            .with_capability(SparqlCapability::NamedGraphs)
            .with_capability(SparqlCapability::FederatedQuery)
            .with_capability(SparqlCapability::GraphStoreProtocol)
            .with_capability(SparqlCapability::TextSearch)
            .with_capability(SparqlCapability::GeoSparql)
    }

    // ── EndpointCapabilities construction ────────────────────────────────

    #[test]
    fn test_new_empty_capabilities() {
        let ep = EndpointCapabilities::new("http://test.org/sparql");
        assert_eq!(ep.endpoint_url, "http://test.org/sparql");
        assert_eq!(ep.capability_count(), 0);
        assert!(ep.max_query_timeout_ms.is_none());
        assert!(ep.max_result_limit.is_none());
    }

    #[test]
    fn test_with_capability_adds_flag() {
        let ep = EndpointCapabilities::new("http://test.org/sparql")
            .with_capability(SparqlCapability::Select11);
        assert_eq!(ep.capability_count(), 1);
        assert!(ep.supports(&SparqlCapability::Select11));
    }

    #[test]
    fn test_with_capability_duplicate_ignored() {
        let ep = EndpointCapabilities::new("http://test.org/sparql")
            .with_capability(SparqlCapability::Select11)
            .with_capability(SparqlCapability::Select11);
        // HashSet deduplicates
        assert_eq!(ep.capability_count(), 1);
    }

    #[test]
    fn test_with_timeout() {
        let ep = EndpointCapabilities::new("http://test.org/sparql").with_timeout(30_000);
        assert_eq!(ep.max_query_timeout_ms, Some(30_000));
    }

    #[test]
    fn test_with_limit() {
        let ep = EndpointCapabilities::new("http://test.org/sparql").with_limit(10_000);
        assert_eq!(ep.max_result_limit, Some(10_000));
    }

    #[test]
    fn test_with_timeout_and_limit() {
        let ep = EndpointCapabilities::new("http://test.org/sparql")
            .with_timeout(5_000)
            .with_limit(500);
        assert_eq!(ep.max_query_timeout_ms, Some(5_000));
        assert_eq!(ep.max_result_limit, Some(500));
    }

    // ── supports / supports_all ───────────────────────────────────────────

    #[test]
    fn test_supports_present_capability() {
        let ep = basic_endpoint();
        assert!(ep.supports(&SparqlCapability::Select11));
        assert!(ep.supports(&SparqlCapability::NamedGraphs));
    }

    #[test]
    fn test_supports_absent_capability() {
        let ep = basic_endpoint();
        assert!(!ep.supports(&SparqlCapability::GeoSparql));
        assert!(!ep.supports(&SparqlCapability::TextSearch));
    }

    #[test]
    fn test_supports_all_empty_slice() {
        let ep = basic_endpoint();
        assert!(ep.supports_all(&[]));
    }

    #[test]
    fn test_supports_all_all_present() {
        let ep = basic_endpoint();
        let caps = [SparqlCapability::Select11, SparqlCapability::NamedGraphs];
        assert!(ep.supports_all(&caps));
    }

    #[test]
    fn test_supports_all_one_missing() {
        let ep = basic_endpoint();
        let caps = [SparqlCapability::Select11, SparqlCapability::GeoSparql];
        assert!(!ep.supports_all(&caps));
    }

    #[test]
    fn test_supports_all_on_full_endpoint() {
        let ep = full_endpoint();
        let caps = [
            SparqlCapability::Select11,
            SparqlCapability::Update11,
            SparqlCapability::Sparql12,
            SparqlCapability::NamedGraphs,
            SparqlCapability::FederatedQuery,
            SparqlCapability::GraphStoreProtocol,
            SparqlCapability::TextSearch,
            SparqlCapability::GeoSparql,
        ];
        assert!(ep.supports_all(&caps));
    }

    #[test]
    fn test_capability_count_basic() {
        let ep = basic_endpoint();
        assert_eq!(ep.capability_count(), 3);
    }

    #[test]
    fn test_capability_count_full() {
        let ep = full_endpoint();
        assert_eq!(ep.capability_count(), 8);
    }

    // ── CapabilityNegotiator::negotiate ───────────────────────────────────

    #[test]
    fn test_negotiate_all_satisfied() {
        let neg = CapabilityNegotiator::new();
        let required = [SparqlCapability::Select11, SparqlCapability::NamedGraphs];
        let result = neg.negotiate(&required, &basic_endpoint());
        assert!(result.compatible);
        assert_eq!(result.satisfied.len(), 2);
        assert!(result.unsatisfied.is_empty());
    }

    #[test]
    fn test_negotiate_none_satisfied() {
        let neg = CapabilityNegotiator::new();
        let required = [SparqlCapability::GeoSparql, SparqlCapability::TextSearch];
        let result = neg.negotiate(&required, &basic_endpoint());
        assert!(!result.compatible);
        assert!(result.satisfied.is_empty());
        assert_eq!(result.unsatisfied.len(), 2);
    }

    #[test]
    fn test_negotiate_partial_satisfaction() {
        let neg = CapabilityNegotiator::new();
        let required = [SparqlCapability::Select11, SparqlCapability::GeoSparql];
        let result = neg.negotiate(&required, &basic_endpoint());
        assert!(!result.compatible);
        assert_eq!(result.satisfied.len(), 1);
        assert_eq!(result.unsatisfied.len(), 1);
        assert!(result.satisfied.contains(&SparqlCapability::Select11));
        assert!(result.unsatisfied.contains(&SparqlCapability::GeoSparql));
    }

    #[test]
    fn test_negotiate_empty_requirements() {
        let neg = CapabilityNegotiator::new();
        let result = neg.negotiate(&[], &basic_endpoint());
        assert!(result.compatible);
        assert!(result.satisfied.is_empty());
        assert!(result.unsatisfied.is_empty());
    }

    #[test]
    fn test_negotiate_all_caps_on_full_endpoint() {
        let neg = CapabilityNegotiator::new();
        let required = [
            SparqlCapability::Select11,
            SparqlCapability::Update11,
            SparqlCapability::Sparql12,
            SparqlCapability::GeoSparql,
            SparqlCapability::TextSearch,
        ];
        let result = neg.negotiate(&required, &full_endpoint());
        assert!(result.compatible);
        assert_eq!(result.satisfied.len(), 5);
        assert!(result.unsatisfied.is_empty());
    }

    // ── CapabilityNegotiator::is_compatible ───────────────────────────────

    #[test]
    fn test_is_compatible_all_present() {
        let neg = CapabilityNegotiator::new();
        let required = [SparqlCapability::Select11, SparqlCapability::NamedGraphs];
        assert!(neg.is_compatible(&required, &basic_endpoint()));
    }

    #[test]
    fn test_is_compatible_missing_cap() {
        let neg = CapabilityNegotiator::new();
        let required = [SparqlCapability::Select11, SparqlCapability::GeoSparql];
        assert!(!neg.is_compatible(&required, &basic_endpoint()));
    }

    #[test]
    fn test_is_compatible_empty_requirements() {
        let neg = CapabilityNegotiator::new();
        assert!(neg.is_compatible(&[], &basic_endpoint()));
    }

    #[test]
    fn test_is_compatible_full_endpoint() {
        let neg = CapabilityNegotiator::new();
        let required = [
            SparqlCapability::GeoSparql,
            SparqlCapability::TextSearch,
            SparqlCapability::Sparql12,
        ];
        assert!(neg.is_compatible(&required, &full_endpoint()));
    }

    // ── Multiple endpoints comparison ─────────────────────────────────────

    #[test]
    fn test_multiple_endpoints_select_compatible_one() {
        let neg = CapabilityNegotiator::new();
        let required = [SparqlCapability::GeoSparql];
        let ep_basic = basic_endpoint();
        let ep_full = full_endpoint();

        assert!(!neg.is_compatible(&required, &ep_basic));
        assert!(neg.is_compatible(&required, &ep_full));
    }

    #[test]
    fn test_negotiate_select11_only_endpoint_vs_update_requirement() {
        let neg = CapabilityNegotiator::new();
        let minimal_ep = EndpointCapabilities::new("http://minimal.org/sparql")
            .with_capability(SparqlCapability::Select11);

        let result = neg.negotiate(&[SparqlCapability::Update11], &minimal_ep);
        assert!(!result.compatible);
        assert!(result.unsatisfied.contains(&SparqlCapability::Update11));
    }

    // ── Capability enum variants ──────────────────────────────────────────

    #[test]
    fn test_all_capability_variants_can_be_added() {
        let ep = EndpointCapabilities::new("http://test.org/sparql")
            .with_capability(SparqlCapability::Select11)
            .with_capability(SparqlCapability::Update11)
            .with_capability(SparqlCapability::Sparql12)
            .with_capability(SparqlCapability::NamedGraphs)
            .with_capability(SparqlCapability::FederatedQuery)
            .with_capability(SparqlCapability::GraphStoreProtocol)
            .with_capability(SparqlCapability::TextSearch)
            .with_capability(SparqlCapability::GeoSparql);
        assert_eq!(ep.capability_count(), 8);
    }

    #[test]
    fn test_capability_equality() {
        assert_eq!(SparqlCapability::Select11, SparqlCapability::Select11);
        assert_ne!(SparqlCapability::Select11, SparqlCapability::Update11);
    }

    // ── Default impl ──────────────────────────────────────────────────────

    #[test]
    fn test_capability_negotiator_default() {
        let neg = CapabilityNegotiator;
        let result = neg.negotiate(&[SparqlCapability::Select11], &basic_endpoint());
        assert!(result.compatible);
    }

    // ── Satisfied/unsatisfied ordering preserved ──────────────────────────

    #[test]
    fn test_negotiate_preserves_order_in_satisfied() {
        let neg = CapabilityNegotiator::new();
        let required = [
            SparqlCapability::Select11,
            SparqlCapability::NamedGraphs,
            SparqlCapability::GraphStoreProtocol,
        ];
        let result = neg.negotiate(&required, &basic_endpoint());
        // All should be satisfied since basic_endpoint has these three
        assert_eq!(result.satisfied.len(), 3);
        assert_eq!(result.satisfied[0], SparqlCapability::Select11);
        assert_eq!(result.satisfied[1], SparqlCapability::NamedGraphs);
        assert_eq!(result.satisfied[2], SparqlCapability::GraphStoreProtocol);
    }

    #[test]
    fn test_negotiate_preserves_order_in_unsatisfied() {
        let neg = CapabilityNegotiator::new();
        let required = [SparqlCapability::GeoSparql, SparqlCapability::TextSearch];
        let result = neg.negotiate(&required, &basic_endpoint());
        assert_eq!(result.unsatisfied[0], SparqlCapability::GeoSparql);
        assert_eq!(result.unsatisfied[1], SparqlCapability::TextSearch);
    }

    // ── Endpoint with only timeout/limit (no caps) ─────────────────────

    #[test]
    fn test_endpoint_with_only_constraints() {
        let ep = EndpointCapabilities::new("http://constrained.org/sparql")
            .with_timeout(1000)
            .with_limit(100);
        assert_eq!(ep.capability_count(), 0);
        assert!(!ep.supports(&SparqlCapability::Select11));
    }

    #[test]
    fn test_negotiate_against_empty_endpoint() {
        let neg = CapabilityNegotiator::new();
        let ep = EndpointCapabilities::new("http://empty.org/sparql");
        let result = neg.negotiate(&[SparqlCapability::Select11], &ep);
        assert!(!result.compatible);
        assert_eq!(result.unsatisfied.len(), 1);
    }

    // ── NegotiationResult fields ───────────────────────────────────────────

    #[test]
    fn test_negotiation_result_compatible_true_when_all_satisfied() {
        let neg = CapabilityNegotiator::new();
        let result = neg.negotiate(&[SparqlCapability::Select11], &basic_endpoint());
        assert!(result.compatible);
        assert!(result.unsatisfied.is_empty());
    }

    #[test]
    fn test_negotiation_result_compatible_false_when_unsatisfied() {
        let neg = CapabilityNegotiator::new();
        let result = neg.negotiate(&[SparqlCapability::GeoSparql], &basic_endpoint());
        assert!(!result.compatible);
    }

    // ── is_compatible on multiple endpoint scenarios ───────────────────────

    #[test]
    fn test_is_compatible_with_no_caps_required() {
        let neg = CapabilityNegotiator::new();
        let ep = EndpointCapabilities::new("http://x.org/sparql");
        assert!(neg.is_compatible(&[], &ep));
    }

    #[test]
    fn test_is_compatible_full_endpoint_with_all_caps() {
        let neg = CapabilityNegotiator::new();
        let ep = full_endpoint();
        assert!(neg.is_compatible(
            &[
                SparqlCapability::Select11,
                SparqlCapability::Update11,
                SparqlCapability::Sparql12,
                SparqlCapability::NamedGraphs,
                SparqlCapability::FederatedQuery,
                SparqlCapability::GraphStoreProtocol,
                SparqlCapability::TextSearch,
                SparqlCapability::GeoSparql,
            ],
            &ep
        ));
    }

    // ── Endpoint URL stored correctly ─────────────────────────────────────

    #[test]
    fn test_endpoint_url_stored() {
        let ep = EndpointCapabilities::new("https://dbpedia.org/sparql");
        assert_eq!(ep.endpoint_url, "https://dbpedia.org/sparql");
    }

    #[test]
    fn test_endpoint_url_with_path() {
        let ep = EndpointCapabilities::new("https://api.endpoint.io/v2/sparql/query");
        assert_eq!(ep.endpoint_url, "https://api.endpoint.io/v2/sparql/query");
    }

    // ── Adding same capability multiple times ─────────────────────────────

    #[test]
    fn test_adding_all_caps_individually_correct_count() {
        let mut ep = EndpointCapabilities::new("http://x.org/sparql");
        ep = ep.with_capability(SparqlCapability::Select11);
        ep = ep.with_capability(SparqlCapability::Update11);
        ep = ep.with_capability(SparqlCapability::GeoSparql);
        assert_eq!(ep.capability_count(), 3);
    }

    // ── negotiate: single cap partially satisfied ─────────────────────────

    #[test]
    fn test_negotiate_single_required_satisfied() {
        let neg = CapabilityNegotiator::new();
        let result = neg.negotiate(&[SparqlCapability::Select11], &basic_endpoint());
        assert_eq!(result.satisfied.len(), 1);
        assert!(result.unsatisfied.is_empty());
    }

    #[test]
    fn test_negotiate_single_required_unsatisfied() {
        let neg = CapabilityNegotiator::new();
        let result = neg.negotiate(&[SparqlCapability::FederatedQuery], &basic_endpoint());
        assert_eq!(result.unsatisfied.len(), 1);
        assert!(result.satisfied.is_empty());
    }

    // ── Clone of EndpointCapabilities ────────────────────────────────────

    #[test]
    fn test_endpoint_capabilities_clone() {
        let ep = basic_endpoint();
        let ep2 = ep.clone();
        assert_eq!(ep.endpoint_url, ep2.endpoint_url);
        assert_eq!(ep.capability_count(), ep2.capability_count());
    }

    // ── negotiate against basic with FederatedQuery requirement ───────────

    #[test]
    fn test_negotiate_federated_query_not_in_basic() {
        let neg = CapabilityNegotiator::new();
        let result = neg.negotiate(&[SparqlCapability::FederatedQuery], &basic_endpoint());
        assert!(!result.compatible);
    }

    #[test]
    fn test_negotiate_federated_query_in_full() {
        let neg = CapabilityNegotiator::new();
        let result = neg.negotiate(&[SparqlCapability::FederatedQuery], &full_endpoint());
        assert!(result.compatible);
    }

    // ── GraphStoreProtocol ────────────────────────────────────────────────

    #[test]
    fn test_basic_endpoint_has_graph_store_protocol() {
        let ep = basic_endpoint();
        assert!(ep.supports(&SparqlCapability::GraphStoreProtocol));
    }

    // ── TextSearch only on full endpoint ─────────────────────────────────

    #[test]
    fn test_text_search_not_in_basic_endpoint() {
        let ep = basic_endpoint();
        assert!(!ep.supports(&SparqlCapability::TextSearch));
    }
}
