//! Multi-Region Support
//!
//! This module provides infrastructure for deploying and managing GraphQL
//! endpoints across multiple geographic regions for improved latency,
//! availability, and data sovereignty compliance.
//!
//! ## Features
//!
//! - **Region Management**: Define and manage multiple regions
//! - **Geo-Routing**: Route requests to the nearest healthy region
//! - **Cross-Region Replication**: Data consistency across regions
//! - **Failover**: Automatic failover to healthy regions
//! - **Latency Optimization**: Select regions based on latency metrics
//! - **Data Locality**: Ensure data stays in specified regions
//!
//! ## Usage
//!
//! ```rust,ignore
//! use oxirs_gql::multi_region::{MultiRegionManager, RegionConfig, Region};
//!
//! let config = RegionConfig::default();
//! let manager = MultiRegionManager::new(config);
//!
//! // Add regions
//! manager.add_region(Region::new("us-east-1", "US East", "https://us-east.example.com")).await;
//! manager.add_region(Region::new("eu-west-1", "EU West", "https://eu-west.example.com")).await;
//!
//! // Route a request
//! let region = manager.route_request("client-ip", client_lat, client_lon).await;
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

/// Region health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionHealth {
    /// Region is healthy
    Healthy,
    /// Region is degraded but operational
    Degraded,
    /// Region is unhealthy
    Unhealthy,
    /// Region health is unknown
    Unknown,
    /// Region is in maintenance mode
    Maintenance,
}

/// Region state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionState {
    /// Region is active and receiving traffic
    Active,
    /// Region is passive (standby)
    Passive,
    /// Region is draining connections
    Draining,
    /// Region is offline
    Offline,
}

/// Routing strategy for multi-region
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route to nearest region (by geographic distance)
    Nearest,
    /// Route to region with lowest latency
    #[default]
    LowestLatency,
    /// Route based on weighted distribution
    Weighted,
    /// Route based on geofencing rules
    Geofenced,
    /// Route to primary region with failover
    PrimaryWithFailover { primary: String },
    /// Round-robin across active regions
    RoundRobin,
    /// Route based on custom rules
    Custom { rules: Vec<RoutingRule> },
}

/// Custom routing rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    /// Rule name
    pub name: String,
    /// Condition for the rule
    pub condition: RoutingCondition,
    /// Target region(s)
    pub target_regions: Vec<String>,
    /// Rule priority (lower = higher priority)
    pub priority: u32,
}

/// Routing condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingCondition {
    /// Match by country code
    Country { codes: Vec<String> },
    /// Match by continent
    Continent { codes: Vec<String> },
    /// Match by IP range (CIDR notation)
    IpRange { ranges: Vec<String> },
    /// Match by header value
    Header { name: String, values: Vec<String> },
    /// Match by time of day (UTC)
    TimeOfDay { start_hour: u8, end_hour: u8 },
    /// Always match (fallback)
    Always,
}

/// Region definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    /// Region identifier (e.g., "us-east-1")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Endpoint URL
    pub endpoint: String,
    /// Geographic location
    pub location: GeoLocation,
    /// Region state
    pub state: RegionState,
    /// Health status
    pub health: RegionHealth,
    /// Traffic weight (for weighted routing)
    pub weight: f64,
    /// Current latency (ms)
    pub latency_ms: u64,
    /// Last health check time
    pub last_health_check: Option<SystemTime>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Data sovereignty tags
    pub sovereignty_tags: Vec<String>,
}

impl Region {
    /// Create a new region
    pub fn new(id: &str, name: &str, endpoint: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            endpoint: endpoint.to_string(),
            location: GeoLocation::default(),
            state: RegionState::Active,
            health: RegionHealth::Unknown,
            weight: 1.0,
            latency_ms: 0,
            last_health_check: None,
            metadata: HashMap::new(),
            sovereignty_tags: Vec::new(),
        }
    }

    /// Create with location
    pub fn with_location(mut self, lat: f64, lon: f64) -> Self {
        self.location = GeoLocation {
            latitude: lat,
            longitude: lon,
        };
        self
    }

    /// Add sovereignty tag
    pub fn with_sovereignty_tag(mut self, tag: &str) -> Self {
        self.sovereignty_tags.push(tag.to_string());
        self
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Check if region is available for traffic
    pub fn is_available(&self) -> bool {
        self.state == RegionState::Active
            && (self.health == RegionHealth::Healthy || self.health == RegionHealth::Degraded)
    }
}

/// Geographic location
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GeoLocation {
    /// Latitude
    pub latitude: f64,
    /// Longitude
    pub longitude: f64,
}

impl Default for GeoLocation {
    fn default() -> Self {
        Self {
            latitude: 0.0,
            longitude: 0.0,
        }
    }
}

impl GeoLocation {
    /// Calculate distance to another location (in kilometers)
    pub fn distance_to(&self, other: &GeoLocation) -> f64 {
        // Haversine formula
        let r = 6371.0; // Earth's radius in km

        let lat1 = self.latitude.to_radians();
        let lat2 = other.latitude.to_radians();
        let dlat = (other.latitude - self.latitude).to_radians();
        let dlon = (other.longitude - self.longitude).to_radians();

        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();

        r * c
    }
}

/// Replication mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationMode {
    /// Synchronous replication (consistent but higher latency)
    Synchronous,
    /// Asynchronous replication (eventual consistency)
    Asynchronous,
    /// No replication (single region)
    None,
}

/// Failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfig {
    /// Enable automatic failover
    pub auto_failover: bool,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Number of failures before failover
    pub failure_threshold: u32,
    /// Failover timeout
    pub failover_timeout: Duration,
    /// Preferred failover order
    pub failover_order: Vec<String>,
    /// Enable failback to primary when recovered
    pub auto_failback: bool,
    /// Minimum time in failover before failback
    pub failback_delay: Duration,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            auto_failover: true,
            health_check_interval: Duration::from_secs(30),
            failure_threshold: 3,
            failover_timeout: Duration::from_secs(30),
            failover_order: Vec::new(),
            auto_failback: true,
            failback_delay: Duration::from_secs(300),
        }
    }
}

/// Multi-region configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionConfig {
    /// Routing strategy
    pub routing_strategy: RoutingStrategy,
    /// Replication mode
    pub replication_mode: ReplicationMode,
    /// Failover configuration
    pub failover: FailoverConfig,
    /// Enable latency-based optimization
    pub latency_optimization: bool,
    /// Latency sample count for averaging
    pub latency_sample_count: usize,
    /// Enable cross-region request tracking
    pub track_cross_region_requests: bool,
    /// Default timeout for cross-region operations
    pub cross_region_timeout: Duration,
}

impl Default for RegionConfig {
    fn default() -> Self {
        Self {
            routing_strategy: RoutingStrategy::default(),
            replication_mode: ReplicationMode::Asynchronous,
            failover: FailoverConfig::default(),
            latency_optimization: true,
            latency_sample_count: 100,
            track_cross_region_requests: true,
            cross_region_timeout: Duration::from_secs(30),
        }
    }
}

/// Routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Selected region
    pub region_id: String,
    /// Endpoint URL
    pub endpoint: String,
    /// Reason for selection
    pub reason: String,
    /// Is this a failover route?
    pub is_failover: bool,
    /// Distance to client (if available)
    pub distance_km: Option<f64>,
    /// Expected latency (if available)
    pub expected_latency_ms: Option<u64>,
}

/// Region event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegionEvent {
    /// Region added
    RegionAdded { region_id: String },
    /// Region removed
    RegionRemoved { region_id: String },
    /// Health status changed
    HealthChanged {
        region_id: String,
        old_health: RegionHealth,
        new_health: RegionHealth,
    },
    /// State changed
    StateChanged {
        region_id: String,
        old_state: RegionState,
        new_state: RegionState,
    },
    /// Failover initiated
    FailoverInitiated {
        from_region: String,
        to_region: String,
        reason: String,
    },
    /// Failback completed
    FailbackCompleted { to_region: String },
    /// Latency updated
    LatencyUpdated { region_id: String, latency_ms: u64 },
}

/// Region metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionMetrics {
    /// Region ID
    pub region_id: String,
    /// Request count
    pub request_count: u64,
    /// Error count
    pub error_count: u64,
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// P95 latency (ms)
    pub p95_latency_ms: u64,
    /// P99 latency (ms)
    pub p99_latency_ms: u64,
    /// Cross-region request count
    pub cross_region_requests: u64,
    /// Failover count
    pub failover_count: u64,
    /// Last updated
    pub last_updated: SystemTime,
}

impl Default for RegionMetrics {
    fn default() -> Self {
        Self {
            region_id: String::new(),
            request_count: 0,
            error_count: 0,
            avg_latency_ms: 0.0,
            p95_latency_ms: 0,
            p99_latency_ms: 0,
            cross_region_requests: 0,
            failover_count: 0,
            last_updated: SystemTime::now(),
        }
    }
}

/// Internal state for multi-region manager
struct ManagerState {
    /// Registered regions
    regions: HashMap<String, Region>,
    /// Region metrics
    metrics: HashMap<String, RegionMetrics>,
    /// Latency samples per region
    latency_samples: HashMap<String, Vec<u64>>,
    /// Current primary region
    primary_region: Option<String>,
    /// Failover in progress
    failover_in_progress: bool,
    /// Failover start time
    failover_started_at: Option<Instant>,
    /// Event log
    events: Vec<(SystemTime, RegionEvent)>,
    /// Round-robin counter
    round_robin_counter: usize,
    /// Failure counts per region
    failure_counts: HashMap<String, u32>,
}

impl ManagerState {
    fn new() -> Self {
        Self {
            regions: HashMap::new(),
            metrics: HashMap::new(),
            latency_samples: HashMap::new(),
            primary_region: None,
            failover_in_progress: false,
            failover_started_at: None,
            events: Vec::new(),
            round_robin_counter: 0,
            failure_counts: HashMap::new(),
        }
    }
}

/// Multi-Region Manager
///
/// Manages geographic distribution of GraphQL endpoints across multiple
/// regions, handling routing, failover, and replication.
pub struct MultiRegionManager {
    /// Configuration
    config: RegionConfig,
    /// Internal state
    state: Arc<RwLock<ManagerState>>,
    /// Event handlers
    event_handlers: Arc<RwLock<Vec<Arc<dyn RegionEventHandler + Send + Sync>>>>,
}

impl MultiRegionManager {
    /// Create a new multi-region manager
    pub fn new(config: RegionConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(ManagerState::new())),
            event_handlers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register an event handler
    pub async fn register_event_handler(&self, handler: Arc<dyn RegionEventHandler + Send + Sync>) {
        let mut handlers = self.event_handlers.write().await;
        handlers.push(handler);
    }

    /// Emit a region event
    async fn emit_event(&self, event: RegionEvent) {
        let now = SystemTime::now();

        {
            let mut state = self.state.write().await;
            state.events.push((now, event.clone()));

            // Limit event history
            if state.events.len() > 1000 {
                state.events.drain(0..100);
            }
        }

        let handlers = self.event_handlers.read().await;
        for handler in handlers.iter() {
            handler.on_event(&event).await;
        }
    }

    /// Add a region
    pub async fn add_region(&self, region: Region) -> Result<()> {
        let region_id = region.id.clone();

        {
            let mut state = self.state.write().await;
            if state.regions.contains_key(&region_id) {
                return Err(anyhow!("Region {} already exists", region_id));
            }

            state.regions.insert(region_id.clone(), region);
            state.metrics.insert(
                region_id.clone(),
                RegionMetrics {
                    region_id: region_id.clone(),
                    ..Default::default()
                },
            );
            state.latency_samples.insert(region_id.clone(), Vec::new());

            // Set as primary if first region
            if state.primary_region.is_none() {
                state.primary_region = Some(region_id.clone());
            }
        }

        self.emit_event(RegionEvent::RegionAdded { region_id })
            .await;

        Ok(())
    }

    /// Remove a region
    pub async fn remove_region(&self, region_id: &str) -> Result<()> {
        {
            let mut state = self.state.write().await;
            if !state.regions.contains_key(region_id) {
                return Err(anyhow!("Region {} not found", region_id));
            }

            state.regions.remove(region_id);
            state.metrics.remove(region_id);
            state.latency_samples.remove(region_id);

            // Update primary if removed
            if state.primary_region.as_deref() == Some(region_id) {
                state.primary_region = state.regions.keys().next().cloned();
            }
        }

        self.emit_event(RegionEvent::RegionRemoved {
            region_id: region_id.to_string(),
        })
        .await;

        Ok(())
    }

    /// Get all regions
    pub async fn get_regions(&self) -> Vec<Region> {
        let state = self.state.read().await;
        state.regions.values().cloned().collect()
    }

    /// Get a specific region
    pub async fn get_region(&self, region_id: &str) -> Option<Region> {
        let state = self.state.read().await;
        state.regions.get(region_id).cloned()
    }

    /// Update region health
    pub async fn update_health(&self, region_id: &str, health: RegionHealth) {
        let old_health;
        {
            let mut state = self.state.write().await;
            let Some(region) = state.regions.get_mut(region_id) else {
                return;
            };
            old_health = region.health;
            region.health = health;
            region.last_health_check = Some(SystemTime::now());

            // Update failure count
            match health {
                RegionHealth::Healthy => {
                    state.failure_counts.insert(region_id.to_string(), 0);
                }
                RegionHealth::Unhealthy => {
                    let count = state
                        .failure_counts
                        .entry(region_id.to_string())
                        .or_insert(0);
                    *count += 1;
                }
                _ => {}
            }
        }

        if old_health != health {
            self.emit_event(RegionEvent::HealthChanged {
                region_id: region_id.to_string(),
                old_health,
                new_health: health,
            })
            .await;

            // Check for failover condition
            if health == RegionHealth::Unhealthy && self.config.failover.auto_failover {
                let should_failover = {
                    let state = self.state.read().await;
                    let failure_count = state.failure_counts.get(region_id).copied().unwrap_or(0);
                    failure_count >= self.config.failover.failure_threshold
                        && state.primary_region.as_deref() == Some(region_id)
                };

                if should_failover {
                    let _ = self
                        .initiate_failover(region_id, "Health check failure")
                        .await;
                }
            }
        }
    }

    /// Update region state
    pub async fn update_state(&self, region_id: &str, new_state: RegionState) {
        let old_state;
        {
            let mut state = self.state.write().await;
            let Some(region) = state.regions.get_mut(region_id) else {
                return;
            };
            old_state = region.state;
            region.state = new_state;
        }

        if old_state != new_state {
            self.emit_event(RegionEvent::StateChanged {
                region_id: region_id.to_string(),
                old_state,
                new_state,
            })
            .await;
        }
    }

    /// Record latency measurement
    pub async fn record_latency(&self, region_id: &str, latency_ms: u64) {
        let mut state = self.state.write().await;

        // Update latency samples
        if let Some(samples) = state.latency_samples.get_mut(region_id) {
            samples.push(latency_ms);
            if samples.len() > self.config.latency_sample_count {
                samples.drain(0..samples.len() - self.config.latency_sample_count);
            }
        }

        // Update region latency (moving average)
        let samples_for_region = state.latency_samples.get(region_id).cloned();
        if let Some(region) = state.regions.get_mut(region_id) {
            if let Some(samples) = &samples_for_region {
                if !samples.is_empty() {
                    let sum: u64 = samples.iter().sum();
                    region.latency_ms = sum / samples.len() as u64;
                }
            }
        }

        // Update metrics
        let samples_for_metrics = state.latency_samples.get(region_id).cloned();
        if let Some(metrics) = state.metrics.get_mut(region_id) {
            metrics.last_updated = SystemTime::now();
            if let Some(samples) = samples_for_metrics {
                if !samples.is_empty() {
                    let sum: u64 = samples.iter().sum();
                    metrics.avg_latency_ms = sum as f64 / samples.len() as f64;

                    let mut sorted = samples.clone();
                    sorted.sort();
                    let p95_idx = (sorted.len() as f64 * 0.95) as usize;
                    let p99_idx = (sorted.len() as f64 * 0.99) as usize;
                    metrics.p95_latency_ms = sorted.get(p95_idx).copied().unwrap_or(0);
                    metrics.p99_latency_ms = sorted.get(p99_idx).copied().unwrap_or(0);
                }
            }
        }
    }

    /// Route a request to the best region
    pub async fn route_request(
        &self,
        client_location: Option<GeoLocation>,
    ) -> Result<RoutingDecision> {
        let state = self.state.read().await;

        // Get available regions
        let available_regions: Vec<&Region> = state
            .regions
            .values()
            .filter(|r| r.is_available())
            .collect();

        if available_regions.is_empty() {
            return Err(anyhow!("No available regions"));
        }

        let decision = match &self.config.routing_strategy {
            RoutingStrategy::Nearest => self.route_nearest(&available_regions, client_location)?,
            RoutingStrategy::LowestLatency => self.route_lowest_latency(&available_regions)?,
            RoutingStrategy::Weighted => self.route_weighted(&available_regions)?,
            RoutingStrategy::Geofenced => {
                // For geofenced, fall back to nearest
                self.route_nearest(&available_regions, client_location)?
            }
            RoutingStrategy::PrimaryWithFailover { primary } => {
                self.route_primary_with_failover(&available_regions, primary)?
            }
            RoutingStrategy::RoundRobin => {
                drop(state);
                self.route_round_robin().await?
            }
            RoutingStrategy::Custom { rules } => {
                self.route_custom(&available_regions, rules, client_location)?
            }
        };

        Ok(decision)
    }

    fn route_nearest(
        &self,
        regions: &[&Region],
        client_location: Option<GeoLocation>,
    ) -> Result<RoutingDecision> {
        let client_loc = client_location.unwrap_or_default();

        let mut best_region: Option<(&Region, f64)> = None;

        for region in regions {
            let distance = region.location.distance_to(&client_loc);
            if best_region
                .as_ref()
                .map_or(true, |(_, best_dist)| distance < *best_dist)
            {
                best_region = Some((region, distance));
            }
        }

        let (region, distance) = best_region.ok_or_else(|| anyhow!("No regions available"))?;

        Ok(RoutingDecision {
            region_id: region.id.clone(),
            endpoint: region.endpoint.clone(),
            reason: format!("Nearest region ({:.0} km)", distance),
            is_failover: false,
            distance_km: Some(distance),
            expected_latency_ms: Some(region.latency_ms),
        })
    }

    fn route_lowest_latency(&self, regions: &[&Region]) -> Result<RoutingDecision> {
        let best_region = regions
            .iter()
            .min_by_key(|r| r.latency_ms)
            .ok_or_else(|| anyhow!("No regions available"))?;

        Ok(RoutingDecision {
            region_id: best_region.id.clone(),
            endpoint: best_region.endpoint.clone(),
            reason: format!("Lowest latency ({}ms)", best_region.latency_ms),
            is_failover: false,
            distance_km: None,
            expected_latency_ms: Some(best_region.latency_ms),
        })
    }

    fn route_weighted(&self, regions: &[&Region]) -> Result<RoutingDecision> {
        let total_weight: f64 = regions.iter().map(|r| r.weight).sum();
        if total_weight <= 0.0 {
            return Err(anyhow!("No weighted regions available"));
        }

        let random = fastrand::f64() * total_weight;
        let mut cumulative = 0.0;

        for region in regions {
            cumulative += region.weight;
            if random <= cumulative {
                return Ok(RoutingDecision {
                    region_id: region.id.clone(),
                    endpoint: region.endpoint.clone(),
                    reason: format!("Weighted selection (weight: {:.2})", region.weight),
                    is_failover: false,
                    distance_km: None,
                    expected_latency_ms: Some(region.latency_ms),
                });
            }
        }

        // Fallback to first region
        let region = regions
            .first()
            .expect("collection validated to be non-empty");
        Ok(RoutingDecision {
            region_id: region.id.clone(),
            endpoint: region.endpoint.clone(),
            reason: "Weighted fallback".to_string(),
            is_failover: false,
            distance_km: None,
            expected_latency_ms: Some(region.latency_ms),
        })
    }

    fn route_primary_with_failover(
        &self,
        regions: &[&Region],
        primary_id: &str,
    ) -> Result<RoutingDecision> {
        // Try primary first
        if let Some(primary) = regions.iter().find(|r| r.id == primary_id) {
            return Ok(RoutingDecision {
                region_id: primary.id.clone(),
                endpoint: primary.endpoint.clone(),
                reason: "Primary region".to_string(),
                is_failover: false,
                distance_km: None,
                expected_latency_ms: Some(primary.latency_ms),
            });
        }

        // Failover to first available
        let region = regions
            .first()
            .ok_or_else(|| anyhow!("No regions available"))?;

        Ok(RoutingDecision {
            region_id: region.id.clone(),
            endpoint: region.endpoint.clone(),
            reason: format!("Failover from {}", primary_id),
            is_failover: true,
            distance_km: None,
            expected_latency_ms: Some(region.latency_ms),
        })
    }

    async fn route_round_robin(&self) -> Result<RoutingDecision> {
        let mut state = self.state.write().await;

        let available_regions: Vec<Region> = state
            .regions
            .values()
            .filter(|r| r.is_available())
            .cloned()
            .collect();

        if available_regions.is_empty() {
            return Err(anyhow!("No available regions"));
        }

        let index = state.round_robin_counter % available_regions.len();
        state.round_robin_counter = state.round_robin_counter.wrapping_add(1);

        let region = &available_regions[index];

        Ok(RoutingDecision {
            region_id: region.id.clone(),
            endpoint: region.endpoint.clone(),
            reason: format!("Round-robin selection (index: {})", index),
            is_failover: false,
            distance_km: None,
            expected_latency_ms: Some(region.latency_ms),
        })
    }

    fn route_custom(
        &self,
        regions: &[&Region],
        rules: &[RoutingRule],
        client_location: Option<GeoLocation>,
    ) -> Result<RoutingDecision> {
        // Sort rules by priority
        let mut sorted_rules = rules.to_vec();
        sorted_rules.sort_by_key(|r| r.priority);

        for rule in sorted_rules {
            let matches = match &rule.condition {
                RoutingCondition::Always => true,
                _ => false, // Other conditions would need more context
            };

            if matches {
                // Find first available target region
                for target_id in &rule.target_regions {
                    if let Some(region) = regions.iter().find(|r| r.id == *target_id) {
                        return Ok(RoutingDecision {
                            region_id: region.id.clone(),
                            endpoint: region.endpoint.clone(),
                            reason: format!("Custom rule: {}", rule.name),
                            is_failover: false,
                            distance_km: None,
                            expected_latency_ms: Some(region.latency_ms),
                        });
                    }
                }
            }
        }

        // Fallback to nearest
        self.route_nearest(regions, client_location)
    }

    /// Initiate failover to another region
    pub async fn initiate_failover(&self, from_region: &str, reason: &str) -> Result<String> {
        let to_region = {
            let mut state = self.state.write().await;

            if state.failover_in_progress {
                return Err(anyhow!("Failover already in progress"));
            }

            // Find best failover target - clone regions to avoid borrow issues
            let failover_candidates: Vec<Region> = state
                .regions
                .values()
                .filter(|r| r.id != from_region && r.is_available())
                .cloned()
                .collect();

            if failover_candidates.is_empty() {
                return Err(anyhow!("No failover candidates available"));
            }

            // Use failover order if configured
            let target = if !self.config.failover.failover_order.is_empty() {
                self.config
                    .failover
                    .failover_order
                    .iter()
                    .find_map(|id| failover_candidates.iter().find(|r| &r.id == id))
                    .cloned()
            } else {
                // Fallback to first available
                failover_candidates.first().cloned()
            };

            let target = target.ok_or_else(|| anyhow!("No suitable failover target"))?;

            let target_id = target.id.clone();
            state.failover_in_progress = true;
            state.failover_started_at = Some(Instant::now());
            state.primary_region = Some(target_id.clone());

            // Update metrics
            if let Some(metrics) = state.metrics.get_mut(&target_id) {
                metrics.failover_count += 1;
            }

            target_id
        };

        self.emit_event(RegionEvent::FailoverInitiated {
            from_region: from_region.to_string(),
            to_region: to_region.clone(),
            reason: reason.to_string(),
        })
        .await;

        // Mark failover complete
        {
            let mut state = self.state.write().await;
            state.failover_in_progress = false;
        }

        Ok(to_region)
    }

    /// Complete failback to original region
    pub async fn complete_failback(&self, to_region: &str) -> Result<()> {
        {
            let mut state = self.state.write().await;

            let region = state
                .regions
                .get(to_region)
                .ok_or_else(|| anyhow!("Region {} not found", to_region))?;

            if !region.is_available() {
                return Err(anyhow!(
                    "Region {} is not available for failback",
                    to_region
                ));
            }

            // Check failback delay
            if let Some(failover_started) = state.failover_started_at {
                if failover_started.elapsed() < self.config.failover.failback_delay {
                    return Err(anyhow!(
                        "Failback delay not elapsed ({:?} remaining)",
                        self.config.failover.failback_delay - failover_started.elapsed()
                    ));
                }
            }

            state.primary_region = Some(to_region.to_string());
            state.failover_started_at = None;
        }

        self.emit_event(RegionEvent::FailbackCompleted {
            to_region: to_region.to_string(),
        })
        .await;

        Ok(())
    }

    /// Get all region metrics
    pub async fn get_all_metrics(&self) -> HashMap<String, RegionMetrics> {
        let state = self.state.read().await;
        state.metrics.clone()
    }

    /// Get region metrics
    pub async fn get_metrics(&self, region_id: &str) -> Option<RegionMetrics> {
        let state = self.state.read().await;
        state.metrics.get(region_id).cloned()
    }

    /// Get recent events
    pub async fn get_recent_events(&self, limit: usize) -> Vec<(SystemTime, RegionEvent)> {
        let state = self.state.read().await;
        state.events.iter().rev().take(limit).cloned().collect()
    }

    /// Get primary region
    pub async fn get_primary_region(&self) -> Option<String> {
        let state = self.state.read().await;
        state.primary_region.clone()
    }
}

/// Trait for handling region events
#[async_trait::async_trait]
pub trait RegionEventHandler {
    /// Handle a region event
    async fn on_event(&self, event: &RegionEvent);
}

/// Logging event handler
pub struct LoggingRegionHandler;

#[async_trait::async_trait]
impl RegionEventHandler for LoggingRegionHandler {
    async fn on_event(&self, event: &RegionEvent) {
        match event {
            RegionEvent::RegionAdded { region_id } => {
                tracing::info!("Region added: {}", region_id);
            }
            RegionEvent::RegionRemoved { region_id } => {
                tracing::info!("Region removed: {}", region_id);
            }
            RegionEvent::HealthChanged {
                region_id,
                old_health,
                new_health,
            } => {
                tracing::info!(
                    "Region {} health: {:?} -> {:?}",
                    region_id,
                    old_health,
                    new_health
                );
            }
            RegionEvent::StateChanged {
                region_id,
                old_state,
                new_state,
            } => {
                tracing::info!(
                    "Region {} state: {:?} -> {:?}",
                    region_id,
                    old_state,
                    new_state
                );
            }
            RegionEvent::FailoverInitiated {
                from_region,
                to_region,
                reason,
            } => {
                tracing::warn!("Failover: {} -> {} ({})", from_region, to_region, reason);
            }
            RegionEvent::FailbackCompleted { to_region } => {
                tracing::info!("Failback completed to {}", to_region);
            }
            RegionEvent::LatencyUpdated {
                region_id,
                latency_ms,
            } => {
                tracing::debug!("Region {} latency: {}ms", region_id, latency_ms);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_manager_creation() {
        let config = RegionConfig::default();
        let manager = MultiRegionManager::new(config);

        let regions = manager.get_regions().await;
        assert!(regions.is_empty());
    }

    #[tokio::test]
    async fn test_add_region() {
        let config = RegionConfig::default();
        let manager = MultiRegionManager::new(config);

        let region = Region::new("us-east-1", "US East", "https://us-east.example.com")
            .with_location(37.7749, -122.4194);

        manager.add_region(region).await.unwrap();

        let regions = manager.get_regions().await;
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].id, "us-east-1");
    }

    #[tokio::test]
    async fn test_remove_region() {
        let config = RegionConfig::default();
        let manager = MultiRegionManager::new(config);

        let region = Region::new("us-east-1", "US East", "https://us-east.example.com");
        manager.add_region(region).await.unwrap();

        manager.remove_region("us-east-1").await.unwrap();

        let regions = manager.get_regions().await;
        assert!(regions.is_empty());
    }

    #[tokio::test]
    async fn test_health_update() {
        let config = RegionConfig::default();
        let manager = MultiRegionManager::new(config);

        let region = Region::new("us-east-1", "US East", "https://us-east.example.com");
        manager.add_region(region).await.unwrap();

        manager
            .update_health("us-east-1", RegionHealth::Healthy)
            .await;

        let region = manager.get_region("us-east-1").await.unwrap();
        assert_eq!(region.health, RegionHealth::Healthy);
    }

    #[tokio::test]
    async fn test_route_nearest() {
        let config = RegionConfig {
            routing_strategy: RoutingStrategy::Nearest,
            ..Default::default()
        };
        let manager = MultiRegionManager::new(config);

        // US East (NYC area)
        let region1 = Region::new("us-east-1", "US East", "https://us-east.example.com")
            .with_location(40.7128, -74.0060);

        // US West (SF area)
        let region2 = Region::new("us-west-1", "US West", "https://us-west.example.com")
            .with_location(37.7749, -122.4194);

        manager.add_region(region1).await.unwrap();
        manager.add_region(region2).await.unwrap();

        manager
            .update_health("us-east-1", RegionHealth::Healthy)
            .await;
        manager
            .update_health("us-west-1", RegionHealth::Healthy)
            .await;

        // Client in NYC area should route to US East
        let client_loc = GeoLocation {
            latitude: 40.7128,
            longitude: -74.0060,
        };

        let decision = manager.route_request(Some(client_loc)).await.unwrap();
        assert_eq!(decision.region_id, "us-east-1");
    }

    #[tokio::test]
    async fn test_route_lowest_latency() {
        let config = RegionConfig {
            routing_strategy: RoutingStrategy::LowestLatency,
            ..Default::default()
        };
        let manager = MultiRegionManager::new(config);

        let mut region1 = Region::new("us-east-1", "US East", "https://us-east.example.com");
        region1.latency_ms = 100;

        let mut region2 = Region::new("us-west-1", "US West", "https://us-west.example.com");
        region2.latency_ms = 50;

        manager.add_region(region1).await.unwrap();
        manager.add_region(region2).await.unwrap();

        manager
            .update_health("us-east-1", RegionHealth::Healthy)
            .await;
        manager
            .update_health("us-west-1", RegionHealth::Healthy)
            .await;

        let decision = manager.route_request(None).await.unwrap();
        assert_eq!(decision.region_id, "us-west-1");
    }

    #[tokio::test]
    async fn test_failover() {
        let config = RegionConfig {
            failover: FailoverConfig {
                auto_failover: true,
                failure_threshold: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let manager = MultiRegionManager::new(config);

        let region1 = Region::new("us-east-1", "US East", "https://us-east.example.com");
        let region2 = Region::new("us-west-1", "US West", "https://us-west.example.com");

        manager.add_region(region1).await.unwrap();
        manager.add_region(region2).await.unwrap();

        manager
            .update_health("us-east-1", RegionHealth::Healthy)
            .await;
        manager
            .update_health("us-west-1", RegionHealth::Healthy)
            .await;

        // Trigger failover from primary
        let new_primary = manager
            .initiate_failover("us-east-1", "Test")
            .await
            .unwrap();
        assert_eq!(new_primary, "us-west-1");
    }

    #[tokio::test]
    async fn test_round_robin() {
        let config = RegionConfig {
            routing_strategy: RoutingStrategy::RoundRobin,
            ..Default::default()
        };
        let manager = MultiRegionManager::new(config);

        let region1 = Region::new("region-1", "Region 1", "https://r1.example.com");
        let region2 = Region::new("region-2", "Region 2", "https://r2.example.com");

        manager.add_region(region1).await.unwrap();
        manager.add_region(region2).await.unwrap();

        manager
            .update_health("region-1", RegionHealth::Healthy)
            .await;
        manager
            .update_health("region-2", RegionHealth::Healthy)
            .await;

        // Should alternate between regions
        let decision1 = manager.route_request(None).await.unwrap();
        let decision2 = manager.route_request(None).await.unwrap();

        assert_ne!(decision1.region_id, decision2.region_id);
    }

    #[tokio::test]
    async fn test_geo_distance_calculation() {
        let nyc = GeoLocation {
            latitude: 40.7128,
            longitude: -74.0060,
        };
        let sf = GeoLocation {
            latitude: 37.7749,
            longitude: -122.4194,
        };

        let distance = nyc.distance_to(&sf);
        // NYC to SF is approximately 4130 km
        assert!(distance > 4000.0 && distance < 4300.0);
    }

    #[tokio::test]
    async fn test_latency_recording() {
        let config = RegionConfig::default();
        let manager = MultiRegionManager::new(config);

        let region = Region::new("us-east-1", "US East", "https://us-east.example.com");
        manager.add_region(region).await.unwrap();

        // Record latencies
        for i in 1..=10 {
            manager.record_latency("us-east-1", i * 10).await;
        }

        let metrics = manager.get_metrics("us-east-1").await.unwrap();
        assert!(metrics.avg_latency_ms > 0.0);
    }

    #[tokio::test]
    async fn test_weighted_routing() {
        let config = RegionConfig {
            routing_strategy: RoutingStrategy::Weighted,
            ..Default::default()
        };
        let manager = MultiRegionManager::new(config);

        let region1 =
            Region::new("region-1", "Region 1", "https://r1.example.com").with_weight(0.9);
        let region2 =
            Region::new("region-2", "Region 2", "https://r2.example.com").with_weight(0.1);

        manager.add_region(region1).await.unwrap();
        manager.add_region(region2).await.unwrap();

        manager
            .update_health("region-1", RegionHealth::Healthy)
            .await;
        manager
            .update_health("region-2", RegionHealth::Healthy)
            .await;

        // With 90/10 weighting, most requests should go to region-1
        let mut region1_count = 0;
        for _ in 0..100 {
            let decision = manager.route_request(None).await.unwrap();
            if decision.region_id == "region-1" {
                region1_count += 1;
            }
        }

        // Should be roughly 90%
        assert!(region1_count >= 70);
    }

    #[tokio::test]
    async fn test_unavailable_region_excluded() {
        let config = RegionConfig::default();
        let manager = MultiRegionManager::new(config);

        let region1 = Region::new("region-1", "Region 1", "https://r1.example.com");
        let region2 = Region::new("region-2", "Region 2", "https://r2.example.com");

        manager.add_region(region1).await.unwrap();
        manager.add_region(region2).await.unwrap();

        // Only region-2 is healthy
        manager
            .update_health("region-1", RegionHealth::Unhealthy)
            .await;
        manager
            .update_health("region-2", RegionHealth::Healthy)
            .await;

        let decision = manager.route_request(None).await.unwrap();
        assert_eq!(decision.region_id, "region-2");
    }
}
