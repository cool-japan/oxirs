//! Geo-aware region router for federated SPARQL queries.
//!
//! Routes sub-queries to the nearest available SPARQL region, maintains a
//! latency matrix across (source-region, target-region) pairs, respects
//! region-pinned queries, and falls back to alternative regions when the
//! primary is unavailable.
//!
//! # Design
//!
//! - `RegionRouter` owns a `LatencyMatrix` and a list of `RegionEndpoint`s.
//! - Routing decisions are deterministic given a fixed matrix (useful for tests).
//! - The latency matrix is updated via recorded probes; it converges with an
//!   exponential-moving-average so it is not noisy with a single bad sample.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

// ─── Region ───────────────────────────────────────────────────────────────────

/// A geographic / logical region identifier (e.g. `"us-east-1"`, `"eu-west-1"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Region(pub String);

impl Region {
    /// Create a new region from any string-like value.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Return the region name as a `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ─── RegionEndpoint ───────────────────────────────────────────────────────────

/// A SPARQL endpoint that belongs to a specific region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionEndpoint {
    /// Unique identifier across all endpoints.
    pub endpoint_id: String,
    /// SPARQL endpoint URL.
    pub url: String,
    /// Region this endpoint belongs to.
    pub region: Region,
    /// Whether this endpoint is currently available.
    pub available: bool,
    /// Optional endpoint-level priority (higher is preferred within a region).
    pub priority: u32,
}

impl RegionEndpoint {
    /// Create a new available endpoint in the given region.
    pub fn new(endpoint_id: impl Into<String>, url: impl Into<String>, region: Region) -> Self {
        Self {
            endpoint_id: endpoint_id.into(),
            url: url.into(),
            region,
            available: true,
            priority: 100,
        }
    }
}

// ─── LatencyMatrix ────────────────────────────────────────────────────────────

/// Exponentially-smoothed inter-region latency matrix.
///
/// Rows are source regions, columns are target regions.  Missing entries
/// default to `DEFAULT_LATENCY_MS`.
#[derive(Debug, Clone, Default)]
pub struct LatencyMatrix {
    /// Smoothed latencies in milliseconds: (source, target) → ms.
    data: HashMap<(Region, Region), f64>,
    /// EMA smoothing factor α ∈ (0, 1].  Higher = faster adaptation.
    alpha: f64,
}

const DEFAULT_LATENCY_MS: f64 = 1_000.0; // assume 1 s for unknown pairs
const DEFAULT_ALPHA: f64 = 0.2;

impl LatencyMatrix {
    /// Create a matrix with the default EMA smoothing factor.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            alpha: DEFAULT_ALPHA,
        }
    }

    /// Create a matrix with a custom EMA smoothing factor.
    pub fn with_alpha(alpha: f64) -> Self {
        let alpha = alpha.clamp(1e-6, 1.0);
        Self {
            data: HashMap::new(),
            alpha,
        }
    }

    /// Record a new observation and update the smoothed estimate.
    pub fn record(&mut self, source: Region, target: Region, observed: Duration) {
        let obs_ms = observed.as_secs_f64() * 1_000.0;
        let key = (source, target);
        let smoothed = self.data.entry(key).or_insert(obs_ms);
        // EMA: new = alpha * obs + (1 - alpha) * old
        *smoothed = self.alpha * obs_ms + (1.0 - self.alpha) * *smoothed;
    }

    /// Get the current smoothed latency estimate in milliseconds.
    ///
    /// Returns `DEFAULT_LATENCY_MS` for unknown (source, target) pairs.
    pub fn get_ms(&self, source: &Region, target: &Region) -> f64 {
        self.data
            .get(&(source.clone(), target.clone()))
            .copied()
            .unwrap_or(DEFAULT_LATENCY_MS)
    }

    /// Return all known (source, target) pairs and their smoothed latencies.
    pub fn entries(&self) -> impl Iterator<Item = (&Region, &Region, f64)> {
        self.data.iter().map(|((src, tgt), &ms)| (src, tgt, ms))
    }
}

// ─── RouteRequest ─────────────────────────────────────────────────────────────

/// Routing request that optionally pins a query to a specific region.
#[derive(Debug, Clone)]
pub struct RouteRequest {
    /// The SPARQL query to route (used for logging / tracing only here).
    pub query: String,
    /// The region from which the request originates.
    pub source_region: Region,
    /// Optional region-pin: if set, the router must use this region.
    pub pinned_region: Option<Region>,
    /// Subset of endpoint IDs that are candidates (empty = all endpoints).
    pub candidate_endpoints: Vec<String>,
}

impl RouteRequest {
    /// Create an unpinned routing request.
    pub fn new(query: impl Into<String>, source_region: Region) -> Self {
        Self {
            query: query.into(),
            source_region,
            pinned_region: None,
            candidate_endpoints: Vec::new(),
        }
    }

    /// Pin the request to a specific region.
    pub fn with_pin(mut self, region: Region) -> Self {
        self.pinned_region = Some(region);
        self
    }

    /// Restrict candidates to the given endpoint IDs.
    pub fn with_candidates(mut self, candidates: Vec<String>) -> Self {
        self.candidate_endpoints = candidates;
        self
    }
}

// ─── RouteDecision ────────────────────────────────────────────────────────────

/// The outcome of a routing decision.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    /// The chosen endpoint.
    pub endpoint: RegionEndpoint,
    /// Estimated latency from source to the chosen endpoint's region.
    pub estimated_latency_ms: f64,
    /// Whether the primary region was used or a fallback.
    pub is_fallback: bool,
    /// Ordered list of all fallback endpoints that were considered.
    pub fallbacks: Vec<RegionEndpoint>,
}

// ─── RouterConfig ─────────────────────────────────────────────────────────────

/// Configuration for `RegionRouter`.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Maximum number of fallback hops before returning an error.
    pub max_fallback_hops: usize,
    /// EMA alpha for the latency matrix.
    pub latency_alpha: f64,
    /// Whether to allow cross-region fallback when all local endpoints fail.
    pub allow_cross_region_fallback: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            max_fallback_hops: 3,
            latency_alpha: DEFAULT_ALPHA,
            allow_cross_region_fallback: true,
        }
    }
}

// ─── RouterError ──────────────────────────────────────────────────────────────

/// Errors produced by the region router.
#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("No available endpoints for region {region}")]
    NoEndpointsForRegion { region: String },

    #[error("Pinned region {region} has no available endpoints")]
    PinnedRegionUnavailable { region: String },

    #[error("All endpoints exhausted after {hops} fallback hop(s)")]
    AllEndpointsExhausted { hops: usize },

    #[error("No endpoints registered")]
    NoEndpoints,
}

// ─── RegionRouter ─────────────────────────────────────────────────────────────

/// Geo-aware router that selects the best endpoint for a sub-query.
///
/// # Thread safety
///
/// `RegionRouter` is designed to be wrapped in `Arc<Mutex<RegionRouter>>` or
/// `Arc<RwLock<RegionRouter>>` for concurrent use.  All mutating operations
/// (`register_endpoint`, `mark_unavailable`, `record_latency`) take `&mut self`.
pub struct RegionRouter {
    endpoints: Vec<RegionEndpoint>,
    latency: LatencyMatrix,
    config: RouterConfig,
}

impl RegionRouter {
    /// Create a new router with default configuration.
    pub fn new() -> Self {
        Self {
            endpoints: Vec::new(),
            latency: LatencyMatrix::new(),
            config: RouterConfig::default(),
        }
    }

    /// Create a new router with custom configuration.
    pub fn with_config(config: RouterConfig) -> Self {
        let latency = LatencyMatrix::with_alpha(config.latency_alpha);
        Self {
            endpoints: Vec::new(),
            latency,
            config,
        }
    }

    /// Register an endpoint.  Duplicate IDs are replaced.
    pub fn register_endpoint(&mut self, endpoint: RegionEndpoint) {
        if let Some(existing) = self
            .endpoints
            .iter_mut()
            .find(|e| e.endpoint_id == endpoint.endpoint_id)
        {
            *existing = endpoint;
        } else {
            self.endpoints.push(endpoint);
        }
    }

    /// Mark an endpoint as unavailable by ID.
    pub fn mark_unavailable(&mut self, endpoint_id: &str) {
        for ep in &mut self.endpoints {
            if ep.endpoint_id == endpoint_id {
                ep.available = false;
                break;
            }
        }
    }

    /// Mark an endpoint as available again by ID.
    pub fn mark_available(&mut self, endpoint_id: &str) {
        for ep in &mut self.endpoints {
            if ep.endpoint_id == endpoint_id {
                ep.available = true;
                break;
            }
        }
    }

    /// Record an observed latency probe for the given (source, target) pair.
    pub fn record_latency(&mut self, source: Region, target: Region, observed: Duration) {
        self.latency.record(source, target, observed);
    }

    /// Route a request to the best available endpoint.
    ///
    /// Algorithm:
    ///
    /// 1. If the request is pinned to a region, restrict candidates to that
    ///    region only.
    /// 2. Among the candidate endpoints (filtered by availability), rank by
    ///    a. Endpoints in the same region as `source_region` come first and
    ///    b. Within a region, sort by estimated latency ascending, then by priority descending.
    /// 3. Return the top-ranked endpoint; populate `fallbacks` with the rest
    ///    (up to `max_fallback_hops`).
    pub fn route(&self, request: &RouteRequest) -> Result<RouteDecision, RouterError> {
        if self.endpoints.is_empty() {
            return Err(RouterError::NoEndpoints);
        }

        // --- build candidate pool -------------------------------------------
        let mut candidates: Vec<&RegionEndpoint> = self
            .endpoints
            .iter()
            .filter(|ep| ep.available)
            .filter(|ep| {
                request.candidate_endpoints.is_empty()
                    || request.candidate_endpoints.contains(&ep.endpoint_id)
            })
            .collect();

        // --- region-pin enforcement -----------------------------------------
        if let Some(ref pinned) = request.pinned_region {
            let pinned_cands: Vec<&RegionEndpoint> = candidates
                .iter()
                .copied()
                .filter(|ep| &ep.region == pinned)
                .collect();

            if pinned_cands.is_empty() {
                return Err(RouterError::PinnedRegionUnavailable {
                    region: pinned.to_string(),
                });
            }
            candidates = pinned_cands;
        }

        if candidates.is_empty() {
            return Err(RouterError::AllEndpointsExhausted { hops: 0 });
        }

        // --- scoring / ranking ----------------------------------------------
        // Score = (is_local: bool, estimated_latency_ms, -priority)
        // We want: local first, then low latency, then high priority.
        let source = &request.source_region;

        let mut scored: Vec<(&RegionEndpoint, bool, f64)> = candidates
            .iter()
            .map(|ep| {
                let is_local = ep.region == *source;
                let lat = self.latency.get_ms(source, &ep.region);
                (*ep, is_local, lat)
            })
            .collect();

        // Sort: local first (true > false), then latency ascending, then
        // priority descending.
        scored.sort_by(|a, b| {
            b.1.cmp(&a.1) // local descending (true before false)
                .then(a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)) // latency asc
                .then(b.0.priority.cmp(&a.0.priority)) // priority desc
        });

        let best = scored[0];
        let is_fallback = !best.1 && request.pinned_region.is_none();

        let fallbacks: Vec<RegionEndpoint> = scored
            .iter()
            .skip(1)
            .take(self.config.max_fallback_hops)
            .map(|(ep, _, _)| (*ep).clone())
            .collect();

        Ok(RouteDecision {
            endpoint: best.0.clone(),
            estimated_latency_ms: best.2,
            is_fallback,
            fallbacks,
        })
    }

    /// Return all registered endpoints.
    pub fn endpoints(&self) -> &[RegionEndpoint] {
        &self.endpoints
    }

    /// Return a reference to the latency matrix.
    pub fn latency_matrix(&self) -> &LatencyMatrix {
        &self.latency
    }

    /// Return available endpoints for a given region.
    pub fn endpoints_for_region(&self, region: &Region) -> Vec<&RegionEndpoint> {
        self.endpoints
            .iter()
            .filter(|ep| &ep.region == region && ep.available)
            .collect()
    }

    /// List all known regions (including unavailable endpoints).
    pub fn known_regions(&self) -> Vec<Region> {
        let mut seen = std::collections::HashSet::new();
        let mut regions = Vec::new();
        for ep in &self.endpoints {
            if seen.insert(ep.region.clone()) {
                regions.push(ep.region.clone());
            }
        }
        regions
    }
}

impl Default for RegionRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_ep(id: &str, region: &str, available: bool) -> RegionEndpoint {
        let mut ep = RegionEndpoint::new(
            id,
            format!("http://{}.example.com/sparql", id),
            Region::new(region),
        );
        ep.available = available;
        ep
    }

    #[test]
    fn test_region_new_and_display() {
        let r = Region::new("us-east-1");
        assert_eq!(r.as_str(), "us-east-1");
        assert_eq!(format!("{}", r), "us-east-1");
    }

    #[test]
    fn test_region_equality_and_hash() {
        let r1 = Region::new("eu-west-1");
        let r2 = Region::new("eu-west-1");
        let r3 = Region::new("ap-northeast-1");
        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
        let mut set = std::collections::HashSet::new();
        set.insert(r1.clone());
        assert!(set.contains(&r2));
        assert!(!set.contains(&r3));
    }

    #[test]
    fn test_latency_matrix_default() {
        let m = LatencyMatrix::new();
        let src = Region::new("us-east-1");
        let tgt = Region::new("eu-west-1");
        assert!((m.get_ms(&src, &tgt) - DEFAULT_LATENCY_MS).abs() < 1e-6);
    }

    #[test]
    fn test_latency_matrix_record_and_retrieve() {
        let mut m = LatencyMatrix::with_alpha(1.0); // alpha=1 means obs replaces estimate
        let src = Region::new("us-east-1");
        let tgt = Region::new("eu-west-1");
        m.record(src.clone(), tgt.clone(), Duration::from_millis(120));
        assert!((m.get_ms(&src, &tgt) - 120.0).abs() < 1e-6);
    }

    #[test]
    fn test_latency_matrix_ema_smoothing() {
        let mut m = LatencyMatrix::with_alpha(0.5);
        let src = Region::new("us-east-1");
        let tgt = Region::new("eu-west-1");
        // First observation: initial value = obs_ms (no prior)
        m.record(src.clone(), tgt.clone(), Duration::from_millis(200));
        // Second observation with alpha=0.5: 0.5*100 + 0.5*200 = 150
        m.record(src.clone(), tgt.clone(), Duration::from_millis(100));
        let v = m.get_ms(&src, &tgt);
        assert!((v - 150.0).abs() < 1e-6);
    }

    #[test]
    fn test_latency_matrix_entries() {
        let mut m = LatencyMatrix::new();
        let src = Region::new("A");
        let tgt = Region::new("B");
        m.record(src.clone(), tgt.clone(), Duration::from_millis(50));
        let entries: Vec<_> = m.entries().collect();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_router_no_endpoints() {
        let router = RegionRouter::new();
        let req = RouteRequest::new("SELECT * WHERE { ?s ?p ?o }", Region::new("us-east-1"));
        match router.route(&req) {
            Err(RouterError::NoEndpoints) => {}
            other => panic!("Expected NoEndpoints, got {:?}", other),
        }
    }

    #[test]
    fn test_router_prefers_local_endpoint() {
        let mut router = RegionRouter::new();
        router.register_endpoint(make_ep("remote-ep", "eu-west-1", true));
        router.register_endpoint(make_ep("local-ep", "us-east-1", true));

        let req = RouteRequest::new("SELECT * WHERE { ?s ?p ?o }", Region::new("us-east-1"));
        let decision = router.route(&req).expect("route should succeed");
        assert_eq!(decision.endpoint.endpoint_id, "local-ep");
        assert!(!decision.is_fallback);
    }

    #[test]
    fn test_router_falls_back_when_local_unavailable() {
        let mut router = RegionRouter::new();
        router.register_endpoint(make_ep("local-ep", "us-east-1", false)); // unavailable
        router.register_endpoint(make_ep("remote-ep", "eu-west-1", true));

        let req = RouteRequest::new("SELECT * WHERE { ?s ?p ?o }", Region::new("us-east-1"));
        let decision = router
            .route(&req)
            .expect("route should succeed with fallback");
        assert_eq!(decision.endpoint.endpoint_id, "remote-ep");
        assert!(decision.is_fallback);
    }

    #[test]
    fn test_router_pinned_region_success() {
        let mut router = RegionRouter::new();
        router.register_endpoint(make_ep("us-ep", "us-east-1", true));
        router.register_endpoint(make_ep("eu-ep", "eu-west-1", true));

        let req = RouteRequest::new("SELECT * WHERE { ?s ?p ?o }", Region::new("us-east-1"))
            .with_pin(Region::new("eu-west-1"));
        let decision = router.route(&req).expect("route should succeed");
        assert_eq!(decision.endpoint.endpoint_id, "eu-ep");
    }

    #[test]
    fn test_router_pinned_region_unavailable() {
        let mut router = RegionRouter::new();
        router.register_endpoint(make_ep("us-ep", "us-east-1", true));
        router.register_endpoint(make_ep("eu-ep", "eu-west-1", false));

        let req = RouteRequest::new("SELECT * WHERE { ?s ?p ?o }", Region::new("us-east-1"))
            .with_pin(Region::new("eu-west-1"));
        match router.route(&req) {
            Err(RouterError::PinnedRegionUnavailable { region }) => {
                assert_eq!(region, "eu-west-1");
            }
            other => panic!("Expected PinnedRegionUnavailable, got {:?}", other),
        }
    }

    #[test]
    fn test_router_candidate_filter() {
        let mut router = RegionRouter::new();
        router.register_endpoint(make_ep("ep-a", "us-east-1", true));
        router.register_endpoint(make_ep("ep-b", "us-east-1", true));

        let req = RouteRequest::new("SELECT * WHERE { ?s ?p ?o }", Region::new("us-east-1"))
            .with_candidates(vec!["ep-b".to_string()]);
        let decision = router.route(&req).expect("route should succeed");
        assert_eq!(decision.endpoint.endpoint_id, "ep-b");
    }

    #[test]
    fn test_router_mark_available_unavailable() {
        let mut router = RegionRouter::new();
        router.register_endpoint(make_ep("ep-a", "us-east-1", true));

        router.mark_unavailable("ep-a");
        let req = RouteRequest::new("q", Region::new("us-east-1"));
        assert!(matches!(
            router.route(&req),
            Err(RouterError::AllEndpointsExhausted { .. })
        ));

        router.mark_available("ep-a");
        assert!(router.route(&req).is_ok());
    }

    #[test]
    fn test_router_record_latency_affects_routing() {
        let mut router = RegionRouter::new();
        // Two remote endpoints in different regions
        router.register_endpoint(make_ep("ap-ep", "ap-northeast-1", true));
        router.register_endpoint(make_ep("eu-ep", "eu-west-1", true));

        let source = Region::new("us-east-1");
        // Teach the router that EU is closer
        router.record_latency(
            source.clone(),
            Region::new("eu-west-1"),
            Duration::from_millis(80),
        );
        router.record_latency(
            source.clone(),
            Region::new("ap-northeast-1"),
            Duration::from_millis(200),
        );

        let req = RouteRequest::new("SELECT * WHERE { ?s ?p ?o }", source);
        let decision = router.route(&req).expect("route should succeed");
        assert_eq!(decision.endpoint.endpoint_id, "eu-ep");
    }

    #[test]
    fn test_router_priority_within_region() {
        let mut router = RegionRouter::new();
        let mut low = make_ep("low-ep", "us-east-1", true);
        low.priority = 50;
        let mut high = make_ep("high-ep", "us-east-1", true);
        high.priority = 200;
        router.register_endpoint(low);
        router.register_endpoint(high);

        let req = RouteRequest::new("q", Region::new("us-east-1"));
        let decision = router.route(&req).expect("route should succeed");
        assert_eq!(decision.endpoint.endpoint_id, "high-ep");
    }

    #[test]
    fn test_router_fallbacks_populated() {
        let mut router = RegionRouter::new();
        router.register_endpoint(make_ep("ep-1", "us-east-1", true));
        router.register_endpoint(make_ep("ep-2", "us-east-1", true));
        router.register_endpoint(make_ep("ep-3", "eu-west-1", true));

        let req = RouteRequest::new("q", Region::new("us-east-1"));
        let decision = router.route(&req).expect("route should succeed");
        // There should be at least 1 fallback
        assert!(!decision.fallbacks.is_empty());
    }

    #[test]
    fn test_router_known_regions() {
        let mut router = RegionRouter::new();
        router.register_endpoint(make_ep("a", "us-east-1", true));
        router.register_endpoint(make_ep("b", "eu-west-1", false));

        let mut regions: Vec<String> = router.known_regions().into_iter().map(|r| r.0).collect();
        regions.sort();
        assert_eq!(regions, vec!["eu-west-1", "us-east-1"]);
    }

    #[test]
    fn test_router_endpoints_for_region() {
        let mut router = RegionRouter::new();
        router.register_endpoint(make_ep("us-a", "us-east-1", true));
        router.register_endpoint(make_ep("us-b", "us-east-1", false));
        router.register_endpoint(make_ep("eu-a", "eu-west-1", true));

        let r = Region::new("us-east-1");
        let eps = router.endpoints_for_region(&r);
        assert_eq!(eps.len(), 1);
        assert_eq!(eps[0].endpoint_id, "us-a");
    }

    #[test]
    fn test_router_register_duplicate_replaces() {
        let mut router = RegionRouter::new();
        router.register_endpoint(make_ep("ep", "us-east-1", true));
        let mut updated = make_ep("ep", "eu-west-1", true);
        updated.priority = 999;
        router.register_endpoint(updated);

        // Only one endpoint should exist
        assert_eq!(router.endpoints().len(), 1);
        assert_eq!(router.endpoints()[0].region.as_str(), "eu-west-1");
        assert_eq!(router.endpoints()[0].priority, 999);
    }

    #[test]
    fn test_route_request_builder() {
        let req = RouteRequest::new(
            "SELECT ?s WHERE { ?s a <T> }",
            Region::new("ap-southeast-1"),
        )
        .with_pin(Region::new("eu-central-1"))
        .with_candidates(vec!["ep-x".to_string(), "ep-y".to_string()]);
        assert!(req.pinned_region.is_some());
        assert_eq!(req.candidate_endpoints.len(), 2);
    }

    #[test]
    fn test_latency_matrix_alpha_clamp() {
        // alpha should be clamped to [1e-6, 1.0]
        let m_low = LatencyMatrix::with_alpha(-5.0);
        assert!(m_low.alpha > 0.0);
        let m_high = LatencyMatrix::with_alpha(5.0);
        assert!((m_high.alpha - 1.0).abs() < 1e-9);
    }
}
