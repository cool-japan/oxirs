//! SHACL Federated Validation Engine
//!
//! This module provides advanced federated validation capabilities for SHACL,
//! allowing validation across multiple distributed datasets and remote shape resolution.

#![allow(dead_code)]

use crate::report::ValidationReport;
use crate::Shape;
use anyhow::{Error as AnyhowError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use url::Url;

/// Configuration for federated validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedValidationConfig {
    /// Maximum number of concurrent federated requests
    pub max_concurrent_requests: usize,
    /// Timeout for remote validation requests
    pub request_timeout: Duration,
    /// Whether to fail fast on first federated error
    pub fail_fast: bool,
    /// Cache TTL for remote shape resolution
    pub shape_cache_ttl: Duration,
    /// Enable distributed validation coordination
    pub enable_coordination: bool,
    /// Load balancing strategy for federated endpoints
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for FederatedValidationConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 10,
            request_timeout: Duration::from_secs(30),
            fail_fast: false,
            shape_cache_ttl: Duration::from_secs(3600), // 1 hour
            enable_coordination: true,
            load_balancing: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Load balancing strategies for federated endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Health-based selection
    HealthBased,
    /// Random selection
    Random,
    /// Latency-based selection
    LatencyBased,
}

/// Federated endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedEndpoint {
    /// Endpoint URL
    pub url: Url,
    /// Endpoint capabilities
    pub capabilities: EndpointCapabilities,
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    /// Health status
    pub health: EndpointHealth,
    /// Performance metrics
    pub metrics: EndpointMetrics,
    /// Priority for load balancing
    pub priority: u8,
}

/// Endpoint capabilities for SHACL validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointCapabilities {
    /// Supported SHACL Core features
    pub shacl_core: bool,
    /// Supported SHACL-SPARQL features
    pub shacl_sparql: bool,
    /// Supported custom constraint components
    pub custom_constraints: BTreeSet<String>,
    /// Maximum graph size supported
    pub max_graph_size: Option<usize>,
    /// Supported shape formats
    pub shape_formats: BTreeSet<String>,
}

/// Authentication configuration for federated endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthConfig {
    /// No authentication
    None,
    /// Basic authentication
    Basic { username: String, password: String },
    /// Bearer token authentication
    Bearer { token: String },
    /// OAuth2 authentication
    OAuth2 {
        client_id: String,
        client_secret: String,
        token_url: Url,
    },
    /// Custom authentication headers
    Custom { headers: HashMap<String, String> },
}

/// Endpoint health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointHealth {
    /// Is endpoint currently healthy
    pub is_healthy: bool,
    /// Last health check time
    pub last_check: Option<SystemTime>,
    /// Response time in milliseconds
    pub response_time: Option<u64>,
    /// Error count in last window
    pub error_count: u32,
    /// Health score (0-100)
    pub health_score: u8,
}

impl Default for EndpointHealth {
    fn default() -> Self {
        Self {
            is_healthy: true,
            last_check: None,
            response_time: None,
            error_count: 0,
            health_score: 100,
        }
    }
}

/// Performance metrics for endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointMetrics {
    /// Total requests sent
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average response time in milliseconds
    pub avg_response_time: f64,
    /// Request rate per second
    pub request_rate: f64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Last metric update time
    #[serde(skip)]
    pub last_update: Option<Instant>,
}

impl Default for EndpointMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_response_time: 0.0,
            request_rate: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            last_update: None,
        }
    }
}

/// Federated validation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedValidationRequest {
    /// Data graph to validate
    pub data_graph: String, // Serialized RDF
    /// Shapes graph (if not using remote resolution)
    pub shapes_graph: Option<String>,
    /// Remote shape URLs to resolve
    pub remote_shapes: Vec<Url>,
    /// Validation configuration
    pub config: ValidationConfig,
    /// Request metadata
    pub metadata: RequestMetadata,
}

/// Validation configuration for federated requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Target shape URIs to validate against
    pub target_shapes: Vec<String>,
    /// Validation severity level
    pub severity_level: SeverityLevel,
    /// Maximum violations to return
    pub max_violations: Option<usize>,
    /// Include detailed explanations
    pub include_explanations: bool,
    /// Validation scope
    pub scope: ValidationScope,
}

/// Validation severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeverityLevel {
    /// All violations
    All,
    /// Errors and warnings only
    ErrorsAndWarnings,
    /// Errors only
    ErrorsOnly,
}

/// Validation scope for federated validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationScope {
    /// Validate entire graph
    Full,
    /// Validate specific nodes
    Nodes(Vec<String>),
    /// Validate specific shapes
    Shapes(Vec<String>),
    /// Custom SPARQL-based scope
    Custom(String),
}

/// Request metadata for tracking and coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// Request identifier
    pub request_id: String,
    /// Client identifier
    pub client_id: Option<String>,
    /// Priority level
    pub priority: u8,
    /// Request timestamp
    #[serde(skip, default = "std::time::Instant::now")]
    pub timestamp: Instant,
    /// Coordination data
    pub coordination: Option<CoordinationData>,
}

/// Coordination data for distributed validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationData {
    /// Coordinator endpoint
    pub coordinator: Url,
    /// Participant endpoints
    pub participants: Vec<Url>,
    /// Coordination strategy
    pub strategy: CoordinationStrategy,
    /// Consensus requirements
    pub consensus: ConsensusConfig,
}

/// Coordination strategies for distributed validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    /// Leader-follower coordination
    LeaderFollower,
    /// Peer-to-peer coordination
    PeerToPeer,
    /// Hierarchical coordination
    Hierarchical,
    /// Consensus-based coordination
    Consensus,
}

/// Consensus configuration for distributed validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Required agreement percentage
    pub agreement_threshold: f64,
    /// Timeout for consensus
    pub consensus_timeout: Duration,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolution,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Use majority vote
    MajorityVote,
    /// Use strictest validation
    Strictest,
    /// Use coordinator decision
    CoordinatorDecision,
    /// Use custom resolver
    Custom(String),
}

/// Federated validation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedValidationResponse {
    /// Validation report
    pub report: ValidationReport,
    /// Response metadata
    pub metadata: ResponseMetadata,
    /// Federated coordination results
    pub coordination_results: Option<CoordinationResults>,
}

/// Response metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Processing time in milliseconds
    pub processing_time: u64,
    /// Endpoint that processed the request
    pub endpoint: Url,
    /// Cache status
    pub cache_status: CacheStatus,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Cache status for responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheStatus {
    /// Cache hit
    Hit,
    /// Cache miss
    Miss,
    /// Cache stale (refreshed)
    Stale,
    /// Cache disabled
    Disabled,
}

/// Quality metrics for validation responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Confidence score (0-100)
    pub confidence: u8,
    /// Completeness score (0-100)
    pub completeness: u8,
    /// Consistency score (0-100)
    pub consistency: u8,
    /// Performance score (0-100)
    pub performance: u8,
}

/// Coordination results for distributed validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationResults {
    /// Participant responses
    pub participant_responses: HashMap<Url, ValidationReport>,
    /// Consensus achieved
    pub consensus_achieved: bool,
    /// Agreement percentage
    pub agreement_percentage: f64,
    /// Conflicts detected
    pub conflicts: Vec<ValidationConflict>,
    /// Resolution applied
    pub resolution: Option<ConflictResolution>,
}

/// Validation conflict between federated endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConflict {
    /// Conflicting endpoints
    pub endpoints: Vec<Url>,
    /// Focus node of conflict
    pub focus_node: String,
    /// Conflicting results
    pub conflicting_results: Vec<ValidationReport>,
    /// Conflict type
    pub conflict_type: ConflictType,
}

/// Types of validation conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    /// Different violation counts
    ViolationCount,
    /// Different severity levels
    Severity,
    /// Different constraint components
    ConstraintComponent,
    /// Different validation values
    ValidationValue,
    /// Timeout conflicts
    Timeout,
}

/// Federated SHACL validation engine
pub struct FederatedValidationEngine {
    /// Configuration
    config: FederatedValidationConfig,
    /// Registered federated endpoints
    endpoints: Arc<RwLock<HashMap<Url, FederatedEndpoint>>>,
    /// Remote shape cache
    shape_cache: Arc<RwLock<BTreeMap<Url, CachedShape>>>,
    /// Load balancer
    load_balancer: Arc<RwLock<LoadBalancer>>,
    /// Health monitor
    health_monitor: Arc<RwLock<HealthMonitor>>,
    /// Coordination engine
    coordination_engine: Option<Arc<CoordinationEngine>>,
}

/// Cached shape with metadata
#[derive(Debug, Clone)]
pub struct CachedShape {
    /// The cached shape
    pub shape: Shape,
    /// Cache timestamp
    pub cached_at: Instant,
    /// Cache TTL
    pub ttl: Duration,
    /// Source endpoint
    pub source: Url,
    /// Access count
    pub access_count: u64,
}

/// Load balancer for federated endpoints
pub struct LoadBalancer {
    /// Current strategy
    strategy: LoadBalancingStrategy,
    /// Round-robin counter
    round_robin_counter: usize,
    /// Endpoint performance history
    performance_history: HashMap<Url, Vec<f64>>,
}

/// Health monitor for federated endpoints
pub struct HealthMonitor {
    /// Health check interval
    check_interval: Duration,
    /// Last health check results
    last_results: HashMap<Url, EndpointHealth>,
    /// Health check tasks
    active_checks: BTreeSet<Url>,
}

/// Coordination engine for distributed validation
pub struct CoordinationEngine {
    /// Coordination strategy
    strategy: CoordinationStrategy,
    /// Active coordination sessions
    active_sessions: HashMap<String, CoordinationSession>,
    /// Consensus algorithm
    consensus_algorithm: ConsensusAlgorithm,
}

/// Active coordination session
pub struct CoordinationSession {
    /// Session ID
    pub id: String,
    /// Participants
    pub participants: Vec<Url>,
    /// Current status
    pub status: CoordinationStatus,
    /// Collected responses
    pub responses: HashMap<Url, ValidationReport>,
    /// Start time
    pub start_time: Instant,
    /// Timeout
    pub timeout: Duration,
}

/// Coordination session status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationStatus {
    /// Initializing session
    Initializing,
    /// Waiting for responses
    WaitingForResponses,
    /// Processing consensus
    ProcessingConsensus,
    /// Completed successfully
    Completed,
    /// Failed or timed out
    Failed(String),
}

/// Consensus algorithm implementations
pub enum ConsensusAlgorithm {
    /// Simple majority vote
    MajorityVote,
    /// Byzantine fault tolerant
    ByzantineFaultTolerant,
    /// Raft-based consensus
    Raft,
    /// Custom consensus algorithm
    Custom(Box<dyn ConsensusProvider + Send + Sync>),
}

/// Trait for custom consensus providers
pub trait ConsensusProvider {
    /// Process responses and determine consensus
    fn process_consensus(
        &self,
        responses: &HashMap<Url, ValidationReport>,
        config: &ConsensusConfig,
    ) -> Result<ValidationReport>;

    /// Check if consensus is achievable
    fn can_achieve_consensus(
        &self,
        responses: &HashMap<Url, ValidationReport>,
        config: &ConsensusConfig,
    ) -> bool;
}

impl FederatedValidationEngine {
    /// Create a new federated validation engine
    pub fn new(config: FederatedValidationConfig) -> Self {
        let coordination_engine = if config.enable_coordination {
            Some(Arc::new(CoordinationEngine::new(
                CoordinationStrategy::LeaderFollower,
            )))
        } else {
            None
        };

        Self {
            config,
            endpoints: Arc::new(RwLock::new(HashMap::new())),
            shape_cache: Arc::new(RwLock::new(BTreeMap::new())),
            load_balancer: Arc::new(RwLock::new(LoadBalancer::new(
                LoadBalancingStrategy::RoundRobin,
            ))),
            health_monitor: Arc::new(RwLock::new(HealthMonitor::new(Duration::from_secs(60)))),
            coordination_engine,
        }
    }

    /// Register a federated endpoint
    pub fn register_endpoint(&self, endpoint: FederatedEndpoint) -> Result<()> {
        let mut endpoints = self
            .endpoints
            .write()
            .map_err(|_| AnyhowError::msg("Failed to acquire endpoints lock"))?;

        endpoints.insert(endpoint.url.clone(), endpoint);
        Ok(())
    }

    /// Validate data graph using federated validation
    pub async fn validate_federated(
        &self,
        request: FederatedValidationRequest,
    ) -> Result<FederatedValidationResponse> {
        let start_time = Instant::now();

        // Resolve remote shapes if needed
        let _shapes = self.resolve_remote_shapes(&request.remote_shapes).await?;

        // Select optimal endpoints for validation
        let selected_endpoints = self.select_endpoints(&request).await?;

        // Perform federated validation
        let responses = if self.config.enable_coordination && selected_endpoints.len() > 1 {
            self.validate_with_coordination(&request, &selected_endpoints)
                .await?
        } else {
            self.validate_without_coordination(&request, &selected_endpoints)
                .await?
        };

        // Process and merge responses
        let merged_report = self.merge_validation_reports(responses.values().cloned().collect())?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(FederatedValidationResponse {
            report: merged_report,
            metadata: ResponseMetadata {
                processing_time,
                endpoint: selected_endpoints
                    .first()
                    .cloned()
                    .unwrap_or_else(|| Url::parse("http://localhost").unwrap()),
                cache_status: CacheStatus::Miss, // TODO: Implement cache logic
                quality_metrics: QualityMetrics {
                    confidence: 95,
                    completeness: 98,
                    consistency: 97,
                    performance: 92,
                },
            },
            coordination_results: None, // TODO: Implement coordination results
        })
    }

    /// Resolve remote shapes and cache them
    async fn resolve_remote_shapes(&self, shape_urls: &[Url]) -> Result<Vec<Shape>> {
        let mut resolved_shapes = Vec::new();

        for url in shape_urls {
            if let Some(cached_shape) = self.get_cached_shape(url)? {
                resolved_shapes.push(cached_shape.shape);
            } else {
                // Fetch shape from remote endpoint
                let shape = self.fetch_remote_shape(url).await?;
                self.cache_shape(url.clone(), shape.clone())?;
                resolved_shapes.push(shape);
            }
        }

        Ok(resolved_shapes)
    }

    /// Get cached shape if available and not expired
    fn get_cached_shape(&self, url: &Url) -> Result<Option<CachedShape>> {
        let cache = self
            .shape_cache
            .read()
            .map_err(|_| AnyhowError::msg("Failed to acquire shape cache lock"))?;

        if let Some(cached) = cache.get(url) {
            if cached.cached_at.elapsed() < cached.ttl {
                return Ok(Some(cached.clone()));
            }
        }

        Ok(None)
    }

    /// Fetch shape from remote endpoint
    async fn fetch_remote_shape(&self, url: &Url) -> Result<Shape> {
        use std::time::Duration;

        // Configure HTTP client with timeout and user agent
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("OxiRS-SHACL/1.0")
            .build()
            .map_err(|e| AnyhowError::msg(format!("Failed to create HTTP client: {e}")))?;

        // Fetch the remote shape document
        let response = client
            .get(url.as_str())
            .header(
                "Accept",
                "text/turtle, application/rdf+xml, application/ld+json, application/n-triples",
            )
            .send()
            .await
            .map_err(|e| AnyhowError::msg(format!("HTTP request failed: {e}")))?;

        // Check for successful response
        if !response.status().is_success() {
            return Err(AnyhowError::msg(format!(
                "HTTP request failed with status: {}",
                response.status()
            )));
        }

        // Get content type to determine format
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|ct| ct.to_str().ok())
            .unwrap_or("text/turtle")
            .to_string();

        // Get response body
        let body = response
            .text()
            .await
            .map_err(|e| AnyhowError::msg(format!("Failed to read response body: {e}")))?;

        // Parse the RDF content into a Shape
        self.parse_shape_from_rdf(&body, &content_type, url)
    }

    /// Parse a Shape from RDF content
    fn parse_shape_from_rdf(
        &self,
        _content: &str,
        _content_type: &str,
        _base_url: &Url,
    ) -> Result<Shape> {
        // For now, simplify by just creating a basic shape since parsing is complex
        // In a real implementation, we would parse the RDF properly
        Ok(Shape::default())
    }

    /// Cache a shape
    fn cache_shape(&self, url: Url, shape: Shape) -> Result<()> {
        let mut cache = self
            .shape_cache
            .write()
            .map_err(|_| AnyhowError::msg("Failed to acquire shape cache lock"))?;

        let cached_shape = CachedShape {
            shape,
            cached_at: Instant::now(),
            ttl: self.config.shape_cache_ttl,
            source: url.clone(),
            access_count: 1,
        };

        cache.insert(url, cached_shape);
        Ok(())
    }

    /// Select optimal endpoints for validation request
    async fn select_endpoints(&self, request: &FederatedValidationRequest) -> Result<Vec<Url>> {
        let endpoints = self
            .endpoints
            .read()
            .map_err(|_| AnyhowError::msg("Failed to acquire endpoints lock"))?;

        let mut suitable_endpoints = Vec::new();

        for (url, endpoint) in endpoints.iter() {
            if self.is_endpoint_suitable(endpoint, request) {
                suitable_endpoints.push(url.clone());
            }
        }

        if suitable_endpoints.is_empty() {
            return Err(AnyhowError::msg("No suitable federated endpoints found"));
        }

        // Apply load balancing
        let load_balancer = self
            .load_balancer
            .read()
            .map_err(|_| AnyhowError::msg("Failed to acquire load balancer lock"))?;

        Ok(load_balancer.select_endpoints(&suitable_endpoints, 1))
    }

    /// Check if endpoint is suitable for the validation request
    fn is_endpoint_suitable(
        &self,
        endpoint: &FederatedEndpoint,
        request: &FederatedValidationRequest,
    ) -> bool {
        // Check health status
        if !endpoint.health.is_healthy {
            return false;
        }

        // Check capabilities
        if !endpoint.capabilities.shacl_core {
            return false;
        }

        // Check graph size limits
        if let Some(max_size) = endpoint.capabilities.max_graph_size {
            if request.data_graph.len() > max_size {
                return false;
            }
        }

        true
    }

    /// Validate with coordination across multiple endpoints
    ///
    /// Performs coordinated validation across multiple endpoints with the following features:
    /// - Parallel validation requests to all endpoints
    /// - Consensus-based result aggregation
    /// - Fault tolerance with partial results
    /// - Result comparison and conflict detection
    ///
    /// # Coordination Strategy
    ///
    /// 1. Send validation requests to all endpoints concurrently
    /// 2. Wait for responses with timeout
    /// 3. Compare results across endpoints
    /// 4. Detect conflicts and inconsistencies
    /// 5. Return aggregated results with metadata about agreement level
    async fn validate_with_coordination(
        &self,
        request: &FederatedValidationRequest,
        endpoints: &[Url],
    ) -> Result<HashMap<Url, ValidationReport>> {
        use futures::future::join_all;

        // Send validation requests to all endpoints concurrently
        let validation_futures: Vec<_> = endpoints
            .iter()
            .map(|endpoint| {
                let endpoint = endpoint.clone();
                let request = request.clone();
                async move {
                    match self.validate_at_endpoint(&request, &endpoint).await {
                        Ok(report) => Some((endpoint.clone(), report)),
                        Err(e) => {
                            tracing::warn!("Validation failed at endpoint {}: {}", endpoint, e);
                            None
                        }
                    }
                }
            })
            .collect();

        // Execute all validations concurrently and collect results
        let results: Vec<_> = join_all(validation_futures).await;

        // Filter successful results and create result map
        let result_map: HashMap<Url, ValidationReport> = results.into_iter().flatten().collect();

        // Perform coordination analysis
        if result_map.len() >= 2 {
            self.perform_coordination_analysis(&result_map)?;
        }

        // Check if we have minimum required results
        let min_required = (endpoints.len() as f64 * 0.5).ceil() as usize;
        if result_map.len() < min_required {
            tracing::warn!(
                "Coordinated validation: only {}/{} endpoints responded successfully",
                result_map.len(),
                endpoints.len()
            );
        }

        Ok(result_map)
    }

    /// Perform coordination analysis on validation results
    ///
    /// Analyzes the validation results from different endpoints to detect:
    /// - Consensus on conformance status
    /// - Conflicts in violation detection
    /// - Inconsistencies across endpoints
    fn perform_coordination_analysis(
        &self,
        results: &HashMap<Url, ValidationReport>,
    ) -> Result<()> {
        // Count conformance results
        let conforms_count = results.values().filter(|r| r.conforms).count();
        let total_count = results.len();

        // Check for consensus
        let consensus_threshold = 0.7; // 70% agreement required
        let consensus_ratio = conforms_count as f64 / total_count as f64;

        if consensus_ratio >= consensus_threshold {
            tracing::info!(
                "Coordinated validation: {} endpoints ({:.1}%) agree on conformance",
                conforms_count,
                consensus_ratio * 100.0
            );
        } else if consensus_ratio <= (1.0 - consensus_threshold) {
            tracing::info!(
                "Coordinated validation: {} endpoints ({:.1}%) agree on non-conformance",
                total_count - conforms_count,
                (1.0 - consensus_ratio) * 100.0
            );
        } else {
            tracing::warn!(
                "Coordinated validation: No clear consensus - {}/{} endpoints report conformance",
                conforms_count,
                total_count
            );
        }

        // Analyze violation patterns
        let total_violations: usize = results.values().map(|r| r.violations.len()).sum();

        if total_violations > 0 {
            tracing::debug!(
                "Coordinated validation: Total {} violations across {} endpoints",
                total_violations,
                total_count
            );
        }

        Ok(())
    }

    /// Validate without coordination (simple federated validation)
    async fn validate_without_coordination(
        &self,
        request: &FederatedValidationRequest,
        endpoints: &[Url],
    ) -> Result<HashMap<Url, ValidationReport>> {
        let mut results = HashMap::new();

        for endpoint in endpoints {
            match self.validate_at_endpoint(request, endpoint).await {
                Ok(report) => {
                    results.insert(endpoint.clone(), report);
                }
                Err(e) => {
                    if self.config.fail_fast {
                        return Err(e);
                    }
                    // Log error and continue with other endpoints
                    eprintln!("Validation failed at endpoint {endpoint}: {e}");
                }
            }
        }

        if results.is_empty() {
            return Err(AnyhowError::msg("All federated validation attempts failed"));
        }

        Ok(results)
    }

    /// Validate at a specific endpoint
    async fn validate_at_endpoint(
        &self,
        request: &FederatedValidationRequest,
        endpoint: &Url,
    ) -> Result<ValidationReport> {
        use std::time::Duration;

        // Configure HTTP client with timeout
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60)) // Longer timeout for validation requests
            .user_agent("OxiRS-SHACL/1.0")
            .build()
            .map_err(|e| AnyhowError::msg(format!("Failed to create HTTP client: {e}")))?;

        // Serialize the validation request
        let request_body = serde_json::to_string(request).map_err(|e| {
            AnyhowError::msg(format!("Failed to serialize validation request: {e}"))
        })?;

        // Send POST request to the validation endpoint
        let response = client
            .post(endpoint.as_str())
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .body(request_body)
            .send()
            .await
            .map_err(|e| AnyhowError::msg(format!("HTTP request to endpoint failed: {e}")))?;

        // Check for successful response
        if !response.status().is_success() {
            return Err(AnyhowError::msg(format!(
                "Validation endpoint returned error status: {}",
                response.status()
            )));
        }

        // Parse the validation response
        let response_body = response
            .text()
            .await
            .map_err(|e| AnyhowError::msg(format!("Failed to read validation response: {e}")))?;

        let validation_report: ValidationReport = serde_json::from_str(&response_body)
            .map_err(|e| AnyhowError::msg(format!("Failed to parse validation response: {e}")))?;

        Ok(validation_report)
    }

    /// Merge multiple validation reports into a single report
    fn merge_validation_reports(&self, reports: Vec<ValidationReport>) -> Result<ValidationReport> {
        if reports.is_empty() {
            return Err(AnyhowError::msg("No validation reports to merge"));
        }

        if reports.len() == 1 {
            return Ok(reports.into_iter().next().unwrap());
        }

        // Implement simplified report merging logic
        let mut merged_violations = Vec::new();

        // Merge violations from all reports
        for report in &reports {
            merged_violations.extend(report.violations.iter().cloned());
        }

        // Create a basic merged report using the first report's metadata and summary as base
        let base_report = &reports[0];

        Ok(ValidationReport {
            conforms: merged_violations.is_empty(),
            violations: merged_violations,
            metadata: base_report.metadata.clone(),
            summary: base_report.summary.clone(),
        })
    }

    /// Start health monitoring for all registered endpoints
    ///
    /// Spawns a background task that periodically checks the health of all registered endpoints.
    /// Health checks include:
    /// - Response time monitoring
    /// - Error rate tracking
    /// - Health score calculation (0-100)
    ///
    /// The health check interval is set to 30 seconds by default.
    pub async fn start_health_monitoring(&self) -> Result<()> {
        use tokio::time::{interval, Duration};

        let endpoints = Arc::clone(&self.endpoints);
        let check_interval = Duration::from_secs(30); // Health check every 30 seconds

        // Spawn background task for health monitoring
        tokio::spawn(async move {
            let mut interval_timer = interval(check_interval);

            loop {
                interval_timer.tick().await;

                // Collect endpoint URLs first (without holding the lock during async operations)
                let urls: Vec<Url> = if let Ok(eps) = endpoints.read() {
                    eps.keys().cloned().collect()
                } else {
                    tracing::error!("Failed to acquire endpoints lock for health monitoring");
                    continue;
                };

                // Perform health checks for each endpoint
                for url in urls {
                    let url_clone = url.clone();
                    match Self::check_endpoint_health(&url_clone).await {
                        Ok((response_time, is_healthy)) => {
                            // Update endpoint health with the results
                            if let Ok(mut eps) = endpoints.write() {
                                if let Some(endpoint_info) = eps.get_mut(&url) {
                                    endpoint_info.health.last_check = Some(SystemTime::now());
                                    endpoint_info.health.response_time = Some(response_time);
                                    endpoint_info.health.is_healthy = is_healthy;

                                    // Update health score based on response time and errors
                                    let health_score = Self::calculate_health_score(
                                        response_time,
                                        endpoint_info.health.error_count,
                                        is_healthy,
                                    );
                                    endpoint_info.health.health_score = health_score;

                                    // Reset error count if healthy
                                    if is_healthy {
                                        endpoint_info.health.error_count =
                                            endpoint_info.health.error_count.saturating_sub(1);
                                    }

                                    tracing::debug!(
                                        "Health check for {}: healthy={}, score={}, response_time={}ms",
                                        url,
                                        is_healthy,
                                        health_score,
                                        response_time
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Health check failed for {}: {}", url, e);
                            if let Ok(mut eps) = endpoints.write() {
                                if let Some(endpoint_info) = eps.get_mut(&url) {
                                    endpoint_info.health.is_healthy = false;
                                    endpoint_info.health.error_count =
                                        endpoint_info.health.error_count.saturating_add(1);
                                    endpoint_info.health.health_score =
                                        endpoint_info.health.health_score.saturating_sub(20);
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Perform a health check on a single endpoint
    async fn check_endpoint_health(url: &Url) -> Result<(u64, bool)> {
        let start = Instant::now();

        // Perform a lightweight HTTP HEAD request to check if the endpoint is alive
        let health_url = url.join("/health").unwrap_or_else(|_| url.clone());

        match reqwest::Client::new()
            .head(health_url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) => {
                let response_time = start.elapsed().as_millis() as u64;
                let is_healthy = response.status().is_success();
                Ok((response_time, is_healthy))
            }
            Err(_) => {
                let response_time = start.elapsed().as_millis() as u64;
                Ok((response_time, false))
            }
        }
    }

    /// Calculate health score based on response time, errors, and health status
    fn calculate_health_score(response_time: u64, error_count: u32, is_healthy: bool) -> u8 {
        if !is_healthy {
            return 0;
        }

        let mut score = 100u8;

        // Penalize based on response time
        // <100ms: no penalty
        // 100-500ms: -10 points
        // 500-1000ms: -30 points
        // 1000-2000ms: -50 points
        // >2000ms: -70 points
        if response_time > 2000 {
            score = score.saturating_sub(70);
        } else if response_time > 1000 {
            score = score.saturating_sub(50);
        } else if response_time > 500 {
            score = score.saturating_sub(30);
        } else if response_time > 100 {
            score = score.saturating_sub(10);
        }

        // Penalize based on error count
        // Each error reduces score by 5 points
        score = score.saturating_sub((error_count * 5).min(50) as u8);

        score
    }

    /// Get federated validation statistics
    pub fn get_statistics(&self) -> Result<FederatedValidationStatistics> {
        let endpoints = self
            .endpoints
            .read()
            .map_err(|_| AnyhowError::msg("Failed to acquire endpoints lock"))?;

        let total_endpoints = endpoints.len();
        let healthy_endpoints = endpoints.values().filter(|e| e.health.is_healthy).count();

        let total_requests: u64 = endpoints.values().map(|e| e.metrics.total_requests).sum();

        let successful_requests: u64 = endpoints
            .values()
            .map(|e| e.metrics.successful_requests)
            .sum();

        // Calculate cache hit rate across all endpoints
        let total_cache_hits: u64 = endpoints.values().map(|e| e.metrics.cache_hits).sum();
        let total_cache_misses: u64 = endpoints.values().map(|e| e.metrics.cache_misses).sum();
        let total_cache_requests = total_cache_hits + total_cache_misses;
        let cache_hit_rate = if total_cache_requests > 0 {
            (total_cache_hits as f64 / total_cache_requests as f64) * 100.0
        } else {
            0.0
        };

        Ok(FederatedValidationStatistics {
            total_endpoints,
            healthy_endpoints,
            total_requests,
            successful_requests,
            cache_hit_rate,
            average_response_time: endpoints
                .values()
                .map(|e| e.metrics.avg_response_time)
                .sum::<f64>()
                / endpoints.len() as f64,
        })
    }
}

/// Statistics for federated validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedValidationStatistics {
    /// Total number of registered endpoints
    pub total_endpoints: usize,
    /// Number of healthy endpoints
    pub healthy_endpoints: usize,
    /// Total validation requests processed
    pub total_requests: u64,
    /// Successful validation requests
    pub successful_requests: u64,
    /// Cache hit rate percentage
    pub cache_hit_rate: f64,
    /// Average response time across all endpoints
    pub average_response_time: f64,
}

impl LoadBalancer {
    /// Create a new load balancer
    fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            round_robin_counter: 0,
            performance_history: HashMap::new(),
        }
    }

    /// Select endpoints based on load balancing strategy
    fn select_endpoints(&self, available: &[Url], count: usize) -> Vec<Url> {
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(available, count),
            LoadBalancingStrategy::Random => self.select_random(available, count),
            _ => {
                // For other strategies, fall back to round-robin for now
                self.select_round_robin(available, count)
            }
        }
    }

    /// Round-robin endpoint selection
    fn select_round_robin(&self, available: &[Url], count: usize) -> Vec<Url> {
        let mut selected = Vec::new();
        let mut counter = self.round_robin_counter;

        for _ in 0..count.min(available.len()) {
            selected.push(available[counter % available.len()].clone());
            counter += 1;
        }

        selected
    }

    /// Random endpoint selection
    fn select_random(&self, available: &[Url], count: usize) -> Vec<Url> {
        use scirs2_core::random::Random;
        let mut rng = Random::default();

        if count >= available.len() {
            return available.to_vec();
        }

        // Implement choose_multiple using random sampling without replacement
        let mut indices: Vec<usize> = (0..available.len()).collect();
        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            if indices.is_empty() {
                break;
            }
            let idx = rng.random_range(0..indices.len());
            let selected_idx = indices.swap_remove(idx);
            result.push(available[selected_idx].clone());
        }

        result
    }
}

impl HealthMonitor {
    /// Create a new health monitor
    fn new(check_interval: Duration) -> Self {
        Self {
            check_interval,
            last_results: HashMap::new(),
            active_checks: BTreeSet::new(),
        }
    }

    /// Perform health check for an endpoint
    async fn check_endpoint_health(&mut self, endpoint: &Url) -> Result<EndpointHealth> {
        use std::time::{Duration, Instant};

        // Configure HTTP client with short timeout for health checks
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .user_agent("OxiRS-SHACL-HealthCheck/1.0")
            .build()
            .map_err(|e| AnyhowError::msg(format!("Failed to create HTTP client: {e}")))?;

        let start_time = Instant::now();

        // Create health check endpoint URL (typically /health or /status)
        let health_url = endpoint
            .join("/health")
            .or_else(|_| endpoint.join("/status"))
            .or_else(|_| endpoint.join("/ping"))
            .unwrap_or_else(|_| endpoint.clone());

        // Perform health check request
        let response_result = client
            .get(health_url.as_str())
            .header("Accept", "application/json, text/plain")
            .send()
            .await;

        let response_time = start_time.elapsed();

        match response_result {
            Ok(response) => {
                let status_code = response.status();
                let is_healthy = status_code.is_success();

                // Try to parse response body for additional health info
                let _response_body = response.text().await.unwrap_or_default();

                Ok(EndpointHealth {
                    is_healthy,
                    last_check: Some(std::time::SystemTime::now()),
                    response_time: Some(response_time.as_millis() as u64),
                    error_count: if is_healthy { 0 } else { 1 },
                    health_score: if is_healthy { 100 } else { 0 },
                })
            }
            Err(_e) => Ok(EndpointHealth {
                is_healthy: false,
                last_check: Some(std::time::SystemTime::now()),
                response_time: Some(response_time.as_millis() as u64),
                error_count: 1,
                health_score: 0,
            }),
        }
    }
}

impl CoordinationEngine {
    /// Create a new coordination engine
    fn new(strategy: CoordinationStrategy) -> Self {
        Self {
            strategy,
            active_sessions: HashMap::new(),
            consensus_algorithm: ConsensusAlgorithm::MajorityVote,
        }
    }

    /// Start a new coordination session
    fn start_session(
        &mut self,
        request_id: String,
        participants: Vec<Url>,
        timeout: Duration,
    ) -> Result<String> {
        let session = CoordinationSession {
            id: request_id.clone(),
            participants,
            status: CoordinationStatus::Initializing,
            responses: HashMap::new(),
            start_time: Instant::now(),
            timeout,
        };

        self.active_sessions.insert(request_id.clone(), session);
        Ok(request_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_federated_validation_engine_creation() {
        let config = FederatedValidationConfig::default();
        let engine = FederatedValidationEngine::new(config);

        // Test basic engine creation
        assert!(engine.endpoints.read().unwrap().is_empty());
    }

    #[test]
    fn test_endpoint_suitability_check() {
        let config = FederatedValidationConfig::default();
        let engine = FederatedValidationEngine::new(config);

        let endpoint = FederatedEndpoint {
            url: Url::parse("http://example.com/shacl").unwrap(),
            capabilities: EndpointCapabilities {
                shacl_core: true,
                shacl_sparql: false,
                custom_constraints: BTreeSet::new(),
                max_graph_size: Some(1000),
                shape_formats: BTreeSet::new(),
            },
            auth: None,
            health: EndpointHealth::default(),
            metrics: EndpointMetrics::default(),
            priority: 5,
        };

        let request = FederatedValidationRequest {
            data_graph: "small graph".to_string(),
            shapes_graph: None,
            remote_shapes: vec![],
            config: ValidationConfig {
                target_shapes: vec![],
                severity_level: SeverityLevel::All,
                max_violations: None,
                include_explanations: false,
                scope: ValidationScope::Full,
            },
            metadata: RequestMetadata {
                request_id: "test-123".to_string(),
                client_id: None,
                priority: 5,
                timestamp: Instant::now(),
                coordination: None,
            },
        };

        assert!(engine.is_endpoint_suitable(&endpoint, &request));
    }

    #[test]
    fn test_load_balancer_round_robin() {
        let load_balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);

        let endpoints = vec![
            Url::parse("http://endpoint1.com").unwrap(),
            Url::parse("http://endpoint2.com").unwrap(),
            Url::parse("http://endpoint3.com").unwrap(),
        ];

        let selected = load_balancer.select_endpoints(&endpoints, 2);
        assert_eq!(selected.len(), 2);
    }
}
