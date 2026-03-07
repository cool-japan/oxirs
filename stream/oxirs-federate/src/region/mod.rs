//! Multi-region federation support for federated SPARQL queries.
//!
//! This module provides:
//! - [`RegionInfo`] – metadata for a geographic/logical region.
//! - [`RegionRegistry`] – manages known regions; supports Haversine-nearest lookup.
//! - [`MultiRegionFederator`] – routes and merges queries across multiple regions.
//! - [`LoadBalancer`] – distributes queries across region replicas with pluggable strategies.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by the multi-region federation subsystem.
#[derive(Debug, Error)]
pub enum RegionError {
    #[error("Region '{id}' already exists")]
    DuplicateRegion { id: String },

    #[error("No region found with id '{id}'")]
    RegionNotFound { id: String },

    #[error("No regions available for routing")]
    NoRegionsAvailable,

    #[error("All region queries failed")]
    AllRegionsFailed,

    #[error("Load balancer has no endpoints to select from")]
    NoEndpoints,

    #[error("Merge error: {0}")]
    MergeError(String),
}

// ─── Binding ──────────────────────────────────────────────────────────────────

/// A single SPARQL variable binding (variable name → RDF value string).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Binding {
    /// Variable name (without leading `?`).
    pub variable: String,
    /// Serialised value (IRI, literal, blank-node label, etc.).
    pub value: String,
}

impl Binding {
    /// Create a new binding.
    pub fn new(variable: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            variable: variable.into(),
            value: value.into(),
        }
    }
}

// ─── RegionInfo ───────────────────────────────────────────────────────────────

/// Metadata for a geographic / logical region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionInfo {
    /// Unique stable identifier (e.g. `"us-east-1"`).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// SPARQL endpoint URL for this region.
    pub endpoint: String,
    /// Routing priority — higher = preferred (0 = lowest).
    pub priority: u8,
    /// Optional geographic coordinates `(latitude°, longitude°)`.
    pub lat_lon: Option<(f64, f64)>,
}

impl RegionInfo {
    /// Construct a new [`RegionInfo`].
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        endpoint: impl Into<String>,
        priority: u8,
        lat_lon: Option<(f64, f64)>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            endpoint: endpoint.into(),
            priority,
            lat_lon,
        }
    }

    /// Haversine distance in kilometres from this region to the given coordinates.
    /// Returns `None` when this region has no coordinates.
    pub fn distance_km(&self, lat: f64, lon: f64) -> Option<f64> {
        let (rlat, rlon) = self.lat_lon?;
        Some(haversine_km(rlat, rlon, lat, lon))
    }
}

// ─── Haversine helper ─────────────────────────────────────────────────────────

/// Haversine great-circle distance between two WGS-84 coordinate pairs.
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6_371.0; // Earth radius in km
    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    let a = (d_lat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (d_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    R * c
}

// ─── RegionRegistry ───────────────────────────────────────────────────────────

/// Manages known regions; supports CRUD and geo-nearest lookup.
#[derive(Debug, Default)]
pub struct RegionRegistry {
    regions: HashMap<String, RegionInfo>,
}

impl RegionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            regions: HashMap::new(),
        }
    }

    /// Add a region.  Returns an error if the id is already registered.
    pub fn add_region(&mut self, region: RegionInfo) -> Result<(), RegionError> {
        if self.regions.contains_key(&region.id) {
            return Err(RegionError::DuplicateRegion { id: region.id });
        }
        self.regions.insert(region.id.clone(), region);
        Ok(())
    }

    /// Remove a region by id.
    pub fn remove_region(&mut self, id: &str) -> Result<RegionInfo, RegionError> {
        self.regions
            .remove(id)
            .ok_or_else(|| RegionError::RegionNotFound { id: id.to_owned() })
    }

    /// Return references to all registered regions (order is unspecified).
    pub fn list_regions(&self) -> Vec<&RegionInfo> {
        self.regions.values().collect()
    }

    /// Look up a region by id.
    pub fn get_region(&self, id: &str) -> Option<&RegionInfo> {
        self.regions.get(id)
    }

    /// Return the region whose coordinates are closest to `(lat, lon)`.
    ///
    /// Only considers regions that have coordinates. Returns `None` when no
    /// region has coordinates.
    pub fn nearest_region(&self, lat: f64, lon: f64) -> Option<&RegionInfo> {
        self.regions
            .values()
            .filter_map(|r| r.distance_km(lat, lon).map(|d| (d, r)))
            .min_by(|(d1, _), (d2, _)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, r)| r)
    }

    /// Number of registered regions.
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
}

// ─── RegionResult ─────────────────────────────────────────────────────────────

/// Result from a single region's SPARQL query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionResult {
    /// The region that produced this result.
    pub region_id: String,
    /// Rows of bindings returned by the query.
    pub bindings: Vec<Vec<Binding>>,
    /// Observed round-trip latency in milliseconds.
    pub latency_ms: u64,
    /// Non-fatal error message if the region partially failed.
    pub error: Option<String>,
}

impl RegionResult {
    /// Create a successful region result.
    pub fn success(
        region_id: impl Into<String>,
        bindings: Vec<Vec<Binding>>,
        latency_ms: u64,
    ) -> Self {
        Self {
            region_id: region_id.into(),
            bindings,
            latency_ms,
            error: None,
        }
    }

    /// Create a failed region result.
    pub fn failure(region_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            region_id: region_id.into(),
            bindings: vec![],
            latency_ms: 0,
            error: Some(error.into()),
        }
    }

    /// Whether this result contains any data.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

// ─── FederatedResult ──────────────────────────────────────────────────────────

/// Merged result from a multi-region federated query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedResult {
    /// Deduplicated, merged binding rows.
    pub bindings: Vec<Vec<Binding>>,
    /// Per-region breakdown.
    pub region_results: Vec<RegionResult>,
    /// Total execution time (wall-clock from dispatch to last result).
    pub total_latency_ms: u64,
    /// Regions that reported errors.
    pub failed_regions: Vec<String>,
}

impl FederatedResult {
    /// Whether all queried regions succeeded.
    pub fn all_succeeded(&self) -> bool {
        self.failed_regions.is_empty()
    }

    /// Number of merged result rows.
    pub fn row_count(&self) -> usize {
        self.bindings.len()
    }
}

// ─── MultiRegionFederator ─────────────────────────────────────────────────────

/// Configuration for the multi-region federator.
#[derive(Debug, Clone)]
pub struct MultiRegionFederatorConfig {
    /// Milliseconds to wait for each region before considering it timed out.
    pub region_timeout_ms: u64,
    /// If true, partial results from failed regions are still returned.
    pub allow_partial_results: bool,
    /// Maximum number of duplicate rows to keep (0 = keep all).
    pub max_rows: usize,
}

impl Default for MultiRegionFederatorConfig {
    fn default() -> Self {
        Self {
            region_timeout_ms: 5_000,
            allow_partial_results: true,
            max_rows: 0,
        }
    }
}

/// Routes and merges queries across multiple regions.
///
/// `MultiRegionFederator` is deliberately sync; callers can wrap the
/// region-specific HTTP calls in their own async runtime.  The `execute_with`
/// closure lets tests inject a mock HTTP back-end.
#[derive(Debug)]
pub struct MultiRegionFederator {
    registry: Arc<Mutex<RegionRegistry>>,
    config: MultiRegionFederatorConfig,
}

impl MultiRegionFederator {
    /// Create a new federator backed by the given registry.
    pub fn new(registry: Arc<Mutex<RegionRegistry>>) -> Self {
        Self {
            registry,
            config: MultiRegionFederatorConfig::default(),
        }
    }

    /// Create with explicit configuration.
    pub fn with_config(
        registry: Arc<Mutex<RegionRegistry>>,
        config: MultiRegionFederatorConfig,
    ) -> Self {
        Self { registry, config }
    }

    /// Execute a federated query against the named regions.
    ///
    /// `execute_fn` is called once per region; it receives the region info and
    /// the query string and should return the raw bindings or an error string.
    pub fn execute_federated<F>(
        &self,
        query: &str,
        region_ids: &[String],
        execute_fn: F,
    ) -> Result<FederatedResult, RegionError>
    where
        F: Fn(&RegionInfo, &str) -> Result<Vec<Vec<Binding>>, String>,
    {
        let registry = self
            .registry
            .lock()
            .map_err(|_| RegionError::MergeError("registry lock poisoned".to_owned()))?;

        if region_ids.is_empty() {
            return Err(RegionError::NoRegionsAvailable);
        }

        let start = Instant::now();
        let mut region_results: Vec<RegionResult> = Vec::with_capacity(region_ids.len());

        for id in region_ids {
            let region = registry
                .get_region(id)
                .ok_or_else(|| RegionError::RegionNotFound { id: id.clone() })?;

            let t0 = Instant::now();
            match execute_fn(region, query) {
                Ok(bindings) => {
                    let latency_ms = t0.elapsed().as_millis() as u64;
                    region_results.push(RegionResult::success(id, bindings, latency_ms));
                }
                Err(err) => {
                    if !self.config.allow_partial_results {
                        return Err(RegionError::AllRegionsFailed);
                    }
                    region_results.push(RegionResult::failure(id, err));
                }
            }
        }

        let total_latency_ms = start.elapsed().as_millis() as u64;
        let mut result = self.merge_results(region_results);
        result.total_latency_ms = total_latency_ms;
        Ok(result)
    }

    /// Broadcast a query to **all** registered regions.
    ///
    /// Results are collected and merged regardless of individual region failures
    /// (when `allow_partial_results` is enabled).
    pub fn broadcast_query<F>(
        &self,
        query: &str,
        execute_fn: F,
    ) -> Result<FederatedResult, RegionError>
    where
        F: Fn(&RegionInfo, &str) -> Result<Vec<Vec<Binding>>, String>,
    {
        let region_ids: Vec<String> = {
            let registry = self
                .registry
                .lock()
                .map_err(|_| RegionError::MergeError("registry lock poisoned".to_owned()))?;
            registry
                .list_regions()
                .iter()
                .map(|r| r.id.clone())
                .collect()
        };

        if region_ids.is_empty() {
            return Err(RegionError::NoRegionsAvailable);
        }

        self.execute_federated(query, &region_ids, execute_fn)
    }

    /// Merge a list of [`RegionResult`]s into a single deduplicated [`FederatedResult`].
    pub fn merge_results(&self, results: Vec<RegionResult>) -> FederatedResult {
        let mut seen: std::collections::HashSet<Vec<String>> = std::collections::HashSet::new();
        let mut merged_bindings: Vec<Vec<Binding>> = Vec::new();
        let mut failed_regions: Vec<String> = Vec::new();

        for r in &results {
            if r.error.is_some() {
                failed_regions.push(r.region_id.clone());
                continue;
            }
            for row in &r.bindings {
                // Build a canonical key for dedup: sorted variable=value pairs
                let key: Vec<String> = {
                    let mut pairs: Vec<String> = row
                        .iter()
                        .map(|b| format!("{}={}", b.variable, b.value))
                        .collect();
                    pairs.sort();
                    pairs
                };
                if seen.insert(key) {
                    merged_bindings.push(row.clone());
                    if self.config.max_rows > 0 && merged_bindings.len() >= self.config.max_rows {
                        return FederatedResult {
                            bindings: merged_bindings,
                            region_results: results,
                            total_latency_ms: 0,
                            failed_regions,
                        };
                    }
                }
            }
        }

        FederatedResult {
            bindings: merged_bindings,
            region_results: results,
            total_latency_ms: 0,
            failed_regions,
        }
    }
}

// ─── LoadBalancer ─────────────────────────────────────────────────────────────

/// Strategy used by [`LoadBalancer`] when selecting an endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LoadBalancingStrategy {
    /// Cycle through endpoints in order.
    #[default]
    RoundRobin,
    /// Always pick the endpoint with the lowest recorded latency.
    LeastLatency,
    /// Pick randomly, weighted by inverse priority (higher priority = more weight).
    WeightedRandom,
}

/// Distributes queries across region replicas according to a pluggable strategy.
#[derive(Debug)]
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    /// Round-robin cursor.
    rr_index: Mutex<usize>,
    /// Per-region smoothed latency in milliseconds.
    latencies: Mutex<HashMap<String, f64>>,
}

impl LoadBalancer {
    /// Create a new load balancer with the given strategy.
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            rr_index: Mutex::new(0),
            latencies: Mutex::new(HashMap::new()),
        }
    }

    /// Record an observed latency for a region (used by `LeastLatency`).
    pub fn record_latency(&self, region_id: &str, observed_ms: f64) {
        const ALPHA: f64 = 0.2;
        if let Ok(mut map) = self.latencies.lock() {
            let entry = map.entry(region_id.to_owned()).or_insert(observed_ms);
            *entry = ALPHA * observed_ms + (1.0 - ALPHA) * *entry;
        }
    }

    /// Select an endpoint from a non-empty slice of [`RegionInfo`]s.
    ///
    /// Returns `Err(RegionError::NoEndpoints)` when `regions` is empty.
    pub fn select_endpoint<'a>(
        &self,
        regions: &'a [RegionInfo],
    ) -> Result<&'a RegionInfo, RegionError> {
        if regions.is_empty() {
            return Err(RegionError::NoEndpoints);
        }

        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(regions),
            LoadBalancingStrategy::LeastLatency => self.select_least_latency(regions),
            LoadBalancingStrategy::WeightedRandom => self.select_weighted_random(regions),
        }
    }

    fn select_round_robin<'a>(
        &self,
        regions: &'a [RegionInfo],
    ) -> Result<&'a RegionInfo, RegionError> {
        let mut idx = self
            .rr_index
            .lock()
            .map_err(|_| RegionError::MergeError("rr_index lock poisoned".to_owned()))?;
        let chosen = &regions[*idx % regions.len()];
        *idx = idx.wrapping_add(1);
        Ok(chosen)
    }

    fn select_least_latency<'a>(
        &self,
        regions: &'a [RegionInfo],
    ) -> Result<&'a RegionInfo, RegionError> {
        const DEFAULT_LATENCY: f64 = 1_000.0;
        let latencies = self
            .latencies
            .lock()
            .map_err(|_| RegionError::MergeError("latencies lock poisoned".to_owned()))?;
        let chosen = regions
            .iter()
            .min_by(|a, b| {
                let la = latencies.get(&a.id).copied().unwrap_or(DEFAULT_LATENCY);
                let lb = latencies.get(&b.id).copied().unwrap_or(DEFAULT_LATENCY);
                la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(RegionError::NoEndpoints)?;
        Ok(chosen)
    }

    fn select_weighted_random<'a>(
        &self,
        regions: &'a [RegionInfo],
    ) -> Result<&'a RegionInfo, RegionError> {
        // Weights = priority value; use a simple deterministic hash of current
        // nanoseconds as a seed to avoid pulling in `rand` directly.
        let total_weight: u64 = regions.iter().map(|r| r.priority as u64 + 1).sum();
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .subsec_nanos() as u64;
        let pick = seed % total_weight;
        let mut cumulative: u64 = 0;
        for r in regions {
            cumulative += r.priority as u64 + 1;
            if pick < cumulative {
                return Ok(r);
            }
        }
        // Fallback — should be unreachable
        Ok(&regions[0])
    }

    /// Current strategy.
    pub fn strategy(&self) -> LoadBalancingStrategy {
        self.strategy
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // ── RegionInfo & haversine ─────────────────────────────────────────────

    #[test]
    fn test_region_info_new() {
        let r = RegionInfo::new("us-east-1", "US East", "http://us.example/sparql", 10, None);
        assert_eq!(r.id, "us-east-1");
        assert_eq!(r.priority, 10);
        assert!(r.lat_lon.is_none());
    }

    #[test]
    fn test_haversine_same_point_is_zero() {
        let d = haversine_km(51.5, -0.1, 51.5, -0.1);
        assert!(d < 1e-6, "same point distance should be ~0, got {d}");
    }

    #[test]
    fn test_haversine_london_to_paris_approx() {
        // London (51.5, -0.1) → Paris (48.85, 2.35) ≈ 340 km
        let d = haversine_km(51.5, -0.1, 48.85, 2.35);
        assert!(d > 300.0 && d < 400.0, "London-Paris distance = {d}");
    }

    #[test]
    fn test_region_distance_km_some() {
        let r = RegionInfo::new(
            "eu-west-1",
            "EU West",
            "http://eu.example/sparql",
            5,
            Some((53.33, -6.25)),
        ); // Dublin
        let d = r.distance_km(51.5, -0.1); // London
        assert!(d.is_some());
        assert!(d.unwrap() > 400.0 && d.unwrap() < 600.0);
    }

    #[test]
    fn test_region_distance_km_none_when_no_coords() {
        let r = RegionInfo::new("x", "X", "http://x.example/sparql", 5, None);
        assert!(r.distance_km(0.0, 0.0).is_none());
    }

    // ── RegionRegistry ────────────────────────────────────────────────────

    #[test]
    fn test_registry_add_and_get() {
        let mut reg = RegionRegistry::new();
        let r = RegionInfo::new("r1", "R1", "http://r1/sparql", 5, None);
        reg.add_region(r).expect("first add should succeed");
        assert!(reg.get_region("r1").is_some());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_registry_duplicate_returns_error() {
        let mut reg = RegionRegistry::new();
        let r1 = RegionInfo::new("dup", "Dup", "http://a/sparql", 5, None);
        let r2 = RegionInfo::new("dup", "Dup2", "http://b/sparql", 5, None);
        reg.add_region(r1).expect("first add");
        let err = reg.add_region(r2).unwrap_err();
        assert!(matches!(err, RegionError::DuplicateRegion { .. }));
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = RegionRegistry::new();
        reg.add_region(RegionInfo::new("rm", "Remove", "http://rm/sparql", 5, None))
            .expect("add");
        reg.remove_region("rm").expect("remove");
        assert!(reg.get_region("rm").is_none());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_remove_nonexistent_errors() {
        let mut reg = RegionRegistry::new();
        assert!(reg.remove_region("ghost").is_err());
    }

    #[test]
    fn test_registry_list_regions() {
        let mut reg = RegionRegistry::new();
        for i in 0..3 {
            reg.add_region(RegionInfo::new(
                format!("r{i}"),
                format!("R{i}"),
                format!("http://r{i}/sparql"),
                5,
                None,
            ))
            .expect("add");
        }
        assert_eq!(reg.list_regions().len(), 3);
    }

    #[test]
    fn test_registry_nearest_region() {
        let mut reg = RegionRegistry::new();
        // London
        reg.add_region(RegionInfo::new(
            "london",
            "London",
            "http://lon/sparql",
            5,
            Some((51.5, -0.1)),
        ))
        .expect("add");
        // Tokyo
        reg.add_region(RegionInfo::new(
            "tokyo",
            "Tokyo",
            "http://tok/sparql",
            5,
            Some((35.68, 139.69)),
        ))
        .expect("add");
        // Frankfurt
        reg.add_region(RegionInfo::new(
            "frankfurt",
            "Frankfurt",
            "http://fra/sparql",
            5,
            Some((50.11, 8.68)),
        ))
        .expect("add");

        // Query from Paris — nearest should be Frankfurt or London
        let nearest = reg
            .nearest_region(48.85, 2.35)
            .expect("should find nearest");
        assert!(nearest.id == "london" || nearest.id == "frankfurt");
    }

    #[test]
    fn test_registry_nearest_returns_none_when_no_coords() {
        let mut reg = RegionRegistry::new();
        reg.add_region(RegionInfo::new("r1", "R1", "http://r1/sparql", 5, None))
            .expect("add");
        assert!(reg.nearest_region(0.0, 0.0).is_none());
    }

    #[test]
    fn test_registry_is_empty() {
        let reg = RegionRegistry::new();
        assert!(reg.is_empty());
    }

    // ── RegionResult ──────────────────────────────────────────────────────

    #[test]
    fn test_region_result_success() {
        let r = RegionResult::success("r1", vec![vec![Binding::new("x", "1")]], 42);
        assert!(r.error.is_none());
        assert_eq!(r.latency_ms, 42);
        assert!(!r.is_empty());
    }

    #[test]
    fn test_region_result_failure() {
        let r = RegionResult::failure("r2", "timeout");
        assert!(r.error.is_some());
        assert!(r.is_empty());
    }

    // ── MultiRegionFederator ──────────────────────────────────────────────

    fn make_registry_with_regions(ids: &[&str]) -> Arc<Mutex<RegionRegistry>> {
        let mut reg = RegionRegistry::new();
        for id in ids {
            reg.add_region(RegionInfo::new(
                *id,
                *id,
                format!("http://{id}/sparql"),
                5,
                None,
            ))
            .expect("add");
        }
        Arc::new(Mutex::new(reg))
    }

    #[test]
    fn test_federator_execute_federated_basic() {
        let reg = make_registry_with_regions(&["r1", "r2"]);
        let fed = MultiRegionFederator::new(reg);

        let result = fed
            .execute_federated(
                "SELECT * WHERE { ?s ?p ?o }",
                &["r1".to_owned(), "r2".to_owned()],
                |region, _q| {
                    Ok(vec![vec![Binding::new(
                        "s",
                        format!("http://{}", region.id),
                    )]])
                },
            )
            .expect("federated query should succeed");

        assert_eq!(result.row_count(), 2);
        assert!(result.all_succeeded());
    }

    #[test]
    fn test_federator_dedup_identical_rows() {
        let reg = make_registry_with_regions(&["r1", "r2"]);
        let fed = MultiRegionFederator::new(reg);
        let same_row = vec![Binding::new("x", "same")];

        let result = fed
            .execute_federated(
                "SELECT ?x WHERE { ?x a <C> }",
                &["r1".to_owned(), "r2".to_owned()],
                |_region, _q| Ok(vec![same_row.clone()]),
            )
            .expect("should succeed");

        // Both regions return the same row → dedup → 1 row
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn test_federator_partial_failure_allowed() {
        let reg = make_registry_with_regions(&["ok", "fail"]);
        let config = MultiRegionFederatorConfig {
            allow_partial_results: true,
            ..Default::default()
        };
        let fed = MultiRegionFederator::with_config(reg, config);

        let result = fed
            .execute_federated(
                "SELECT * WHERE { ?s ?p ?o }",
                &["ok".to_owned(), "fail".to_owned()],
                |region, _q| {
                    if region.id == "fail" {
                        Err("endpoint down".to_owned())
                    } else {
                        Ok(vec![vec![Binding::new("s", "http://ok/node")]])
                    }
                },
            )
            .expect("should succeed with partial results");

        assert_eq!(result.failed_regions.len(), 1);
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn test_federator_no_regions_error() {
        let reg = make_registry_with_regions(&["r1"]);
        let fed = MultiRegionFederator::new(reg);
        let err = fed
            .execute_federated("SELECT ?x WHERE {}", &[], |_, _| Ok(vec![]))
            .unwrap_err();
        assert!(matches!(err, RegionError::NoRegionsAvailable));
    }

    #[test]
    fn test_federator_broadcast_query() {
        let reg = make_registry_with_regions(&["a", "b", "c"]);
        let fed = MultiRegionFederator::new(reg);

        let result = fed
            .broadcast_query("SELECT * WHERE { ?s ?p ?o }", |region, _q| {
                Ok(vec![vec![Binding::new("region", region.id.clone())]])
            })
            .expect("broadcast should succeed");

        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_federator_broadcast_empty_registry() {
        let reg = Arc::new(Mutex::new(RegionRegistry::new()));
        let fed = MultiRegionFederator::new(reg);
        let err = fed
            .broadcast_query("SELECT * WHERE {}", |_, _| Ok(vec![]))
            .unwrap_err();
        assert!(matches!(err, RegionError::NoRegionsAvailable));
    }

    #[test]
    fn test_federator_max_rows_limit() {
        let reg = make_registry_with_regions(&["r1", "r2", "r3"]);
        let config = MultiRegionFederatorConfig {
            max_rows: 2,
            allow_partial_results: true,
            ..Default::default()
        };
        let fed = MultiRegionFederator::with_config(reg, config);

        let result = fed
            .execute_federated(
                "SELECT ?x WHERE {}",
                &["r1".to_owned(), "r2".to_owned(), "r3".to_owned()],
                |region, _q| Ok(vec![vec![Binding::new("x", region.id.clone())]]),
            )
            .expect("should succeed");

        assert!(result.row_count() <= 2);
    }

    #[test]
    fn test_federator_region_not_found_error() {
        let reg = make_registry_with_regions(&["r1"]);
        let fed = MultiRegionFederator::new(reg);
        let err = fed
            .execute_federated("SELECT ?x WHERE {}", &["ghost".to_owned()], |_, _| {
                Ok(vec![])
            })
            .unwrap_err();
        assert!(matches!(err, RegionError::RegionNotFound { .. }));
    }

    // ── LoadBalancer ──────────────────────────────────────────────────────

    fn sample_regions(n: usize) -> Vec<RegionInfo> {
        (0..n)
            .map(|i| {
                RegionInfo::new(
                    format!("r{i}"),
                    format!("R{i}"),
                    format!("http://r{i}/sparql"),
                    i as u8 + 1,
                    None,
                )
            })
            .collect()
    }

    #[test]
    fn test_lb_round_robin_cycles() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        let regions = sample_regions(3);
        let id0 = lb
            .select_endpoint(&regions)
            .expect("should pick")
            .id
            .clone();
        let id1 = lb
            .select_endpoint(&regions)
            .expect("should pick")
            .id
            .clone();
        let id2 = lb
            .select_endpoint(&regions)
            .expect("should pick")
            .id
            .clone();
        // Must be three different selections in sequence
        assert_eq!(id0, "r0");
        assert_eq!(id1, "r1");
        assert_eq!(id2, "r2");
    }

    #[test]
    fn test_lb_round_robin_wraps() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        let regions = sample_regions(2);
        for _ in 0..6 {
            lb.select_endpoint(&regions).expect("should pick");
        }
        // Should not panic or error
    }

    #[test]
    fn test_lb_least_latency_picks_lowest() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::LeastLatency);
        let regions = sample_regions(3);
        lb.record_latency("r0", 500.0);
        lb.record_latency("r1", 50.0); // lowest
        lb.record_latency("r2", 200.0);
        let chosen = lb.select_endpoint(&regions).expect("should pick");
        assert_eq!(chosen.id, "r1");
    }

    #[test]
    fn test_lb_least_latency_defaults_equal() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::LeastLatency);
        let regions = sample_regions(3);
        // No latencies recorded → all equal → first alphabetically or first in vec
        let chosen = lb.select_endpoint(&regions).expect("should pick");
        assert!(!chosen.id.is_empty());
    }

    #[test]
    fn test_lb_weighted_random_returns_valid() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::WeightedRandom);
        let regions = sample_regions(5);
        for _ in 0..20 {
            let r = lb.select_endpoint(&regions).expect("should pick");
            assert!(regions.iter().any(|x| x.id == r.id));
        }
    }

    #[test]
    fn test_lb_empty_regions_error() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        let err = lb.select_endpoint(&[]).unwrap_err();
        assert!(matches!(err, RegionError::NoEndpoints));
    }

    #[test]
    fn test_lb_strategy_accessor() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::LeastLatency);
        assert_eq!(lb.strategy(), LoadBalancingStrategy::LeastLatency);
    }

    #[test]
    fn test_lb_record_latency_ema() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::LeastLatency);
        lb.record_latency("r0", 100.0);
        lb.record_latency("r0", 200.0);
        // After two observations the EMA should be between 100 and 200
        let latencies = lb.latencies.lock().expect("lock");
        let v = latencies.get("r0").copied().expect("should exist");
        assert!(v > 100.0 && v < 200.0, "EMA = {v}");
    }

    #[test]
    fn test_federated_result_row_count() {
        let res = FederatedResult {
            bindings: vec![vec![Binding::new("x", "1")], vec![Binding::new("x", "2")]],
            region_results: vec![],
            total_latency_ms: 10,
            failed_regions: vec![],
        };
        assert_eq!(res.row_count(), 2);
        assert!(res.all_succeeded());
    }

    #[test]
    fn test_federated_result_all_succeeded_false() {
        let res = FederatedResult {
            bindings: vec![],
            region_results: vec![],
            total_latency_ms: 0,
            failed_regions: vec!["r1".to_owned()],
        };
        assert!(!res.all_succeeded());
    }
}
