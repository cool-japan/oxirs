//! Federated SPARQL query planning: endpoint registry, discovery, query
//! decomposition, join-order optimization and cost estimation.
//!
//! Implementations split out from `federated_query_optimizer` for
//! maintainability.

use crate::{
    error::{FusekiError, FusekiResult},
    federated_query_types::*,
};
use dashmap::DashMap;
use reqwest::{Client, ClientBuilder, StatusCode};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;

impl Default for EndpointRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl EndpointRegistry {
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
            health_cache: DashMap::new(),
            discovery: Arc::new(EndpointDiscovery::new()),
        }
    }

    /// Register a new endpoint
    pub fn register_endpoint(&mut self, endpoint: EndpointInfo) {
        self.endpoints.insert(endpoint.url.clone(), endpoint);
    }

    /// Accessor for the underlying endpoint map.
    pub fn endpoints(&self) -> &HashMap<String, EndpointInfo> {
        &self.endpoints
    }

    /// Check endpoint health
    pub async fn check_endpoint_health(&self, endpoint: EndpointInfo) -> FusekiResult<()> {
        let client = ClientBuilder::new()
            .timeout(Duration::from_millis(5000))
            .build()
            .map_err(|e| FusekiError::internal(format!("Client error: {e}")))?;

        let start = Instant::now();
        let response = client
            .get(&endpoint.url)
            .header("Accept", "application/sparql-results+json")
            .query(&[("query", "ASK { ?s ?p ?o } LIMIT 1")])
            .send()
            .await;

        let response_time = start.elapsed().as_millis() as u64;

        match response {
            Ok(resp) if resp.status() == StatusCode::OK => {
                self.health_cache.insert(
                    endpoint.url.clone(),
                    HealthStatus {
                        is_healthy: true,
                        last_check: Instant::now(),
                        response_time_ms: response_time,
                        error_count: 0,
                        success_count: 1,
                    },
                );
                Ok(())
            }
            Ok(resp) => Err(FusekiError::bad_request(format!(
                "Endpoint returned status: {}",
                resp.status()
            ))),
            Err(e) => {
                self.health_cache.insert(
                    endpoint.url.clone(),
                    HealthStatus {
                        is_healthy: false,
                        last_check: Instant::now(),
                        response_time_ms: response_time,
                        error_count: 1,
                        success_count: 0,
                    },
                );
                Err(FusekiError::internal(format!("Health check failed: {e}")))
            }
        }
    }

    /// Discover endpoints from catalogs
    pub async fn discover_endpoints(&self) -> FusekiResult<Vec<EndpointInfo>> {
        self.discovery.discover_from_catalogs().await
    }
}

impl Default for EndpointDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl EndpointDiscovery {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            catalogs: vec![
                "https://www.w3.org/wiki/SparqlEndpoints".to_string(),
                "https://lod-cloud.net/endpoints".to_string(),
            ],
        }
    }

    /// Discover endpoints from known catalogs
    pub async fn discover_from_catalogs(&self) -> FusekiResult<Vec<EndpointInfo>> {
        // Placeholder for actual discovery implementation
        // Would parse VOID descriptions, SPARQL Service Descriptions, etc.
        Ok(vec![])
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryPlanner {
    pub fn new() -> Self {
        Self {
            decomposition_rules: Self::create_decomposition_rules(),
            join_optimizer: Arc::new(JoinOrderOptimizer::new()),
            statistics: Arc::new(RwLock::new(FederationStatistics::new())),
        }
    }

    /// Create standard decomposition rules
    fn create_decomposition_rules() -> Vec<DecompositionRule> {
        vec![
            // Triple pattern decomposition
            DecompositionRule {
                name: "TriplePatternDecomposition".to_string(),
                pattern: "triple_pattern".to_string(),
                applicability_check: Box::new(|query| {
                    query.contains("?s") && query.contains("?p") && query.contains("?o")
                }),
                decompose: Box::new(|_query| {
                    // Decompose triple patterns across endpoints
                    vec![]
                }),
            },
            // UNION decomposition
            DecompositionRule {
                name: "UnionDecomposition".to_string(),
                pattern: "union".to_string(),
                applicability_check: Box::new(|query| query.to_uppercase().contains("UNION")),
                decompose: Box::new(|_query| {
                    // Split UNION branches for parallel execution
                    vec![]
                }),
            },
            // OPTIONAL decomposition
            DecompositionRule {
                name: "OptionalDecomposition".to_string(),
                pattern: "optional".to_string(),
                applicability_check: Box::new(|query| query.to_uppercase().contains("OPTIONAL")),
                decompose: Box::new(|_query| {
                    // Handle OPTIONAL patterns
                    vec![]
                }),
            },
        ]
    }

    /// Create execution plan for federated query
    pub async fn create_execution_plan(
        &self,
        query: &str,
        service_patterns: &[ServicePattern],
    ) -> FusekiResult<ExecutionPlan> {
        // Decompose query into fragments
        let fragments = self.decompose_query(query, service_patterns)?;

        // Optimize join order
        let join_plan = self.join_optimizer.optimize_joins(&fragments).await?;

        // Create execution plan
        let required_endpoints: Vec<String> = service_patterns
            .iter()
            .map(|p| p.service_url.clone())
            .collect();

        let execution_steps: Vec<String> = fragments
            .iter()
            .enumerate()
            .map(|(i, f)| {
                format!(
                    "Execute fragment {} ({}) at {:?}",
                    i, f.sparql, f.target_endpoints
                )
            })
            .collect();

        let estimated_cost = fragments.iter().map(|f| f.estimated_cost).sum::<f64>(); // Sum of fragment costs

        Ok(ExecutionPlan {
            query_id: crate::federated_query_executor::new_query_id(),
            fragments: fragments.clone(),
            join_plan,
            timeout_ms: 30000,
            optimization_hints: HashMap::new(),
            execution_steps,
            estimated_cost,
            resource_requirements: ResourceRequirements {
                required_endpoints,
                estimated_memory_mb: fragments.len() as f64 * 5.0,
                estimated_cpu_cores: (fragments.len() as f64 / 2.0).max(1.0),
            },
        })
    }

    /// Decompose query into executable fragments
    fn decompose_query(
        &self,
        query: &str,
        service_patterns: &[ServicePattern],
    ) -> FusekiResult<Vec<QueryFragment>> {
        let mut fragments = Vec::new();

        // Create fragments for each SERVICE pattern
        for (idx, pattern) in service_patterns.iter().enumerate() {
            fragments.push(QueryFragment {
                fragment_id: format!("service_{idx}"),
                sparql: pattern.pattern.clone(),
                target_endpoints: vec![pattern.service_url.clone()],
                dependencies: vec![],
                estimated_cost: 1.0,
                is_optional: pattern.is_optional,
            });
        }

        // Apply decomposition rules
        for rule in &self.decomposition_rules {
            if (rule.applicability_check)(query) {
                let decomposed = (rule.decompose)(query);
                fragments.extend(decomposed);
            }
        }

        Ok(fragments)
    }
}

impl Default for JoinOrderOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl JoinOrderOptimizer {
    pub fn new() -> Self {
        Self {
            cost_model: Arc::new(JoinCostModel::new()),
            dp_cache: DashMap::new(),
        }
    }

    /// Optimize join order for fragments
    pub async fn optimize_joins(&self, fragments: &[QueryFragment]) -> FusekiResult<JoinPlan> {
        // Use dynamic programming to find optimal join order
        let cache_key = self.compute_cache_key(fragments);

        if let Some(cached_plan) = self.dp_cache.get(&cache_key) {
            return Ok(cached_plan.clone());
        }

        // Compute optimal plan
        let plan = self.compute_optimal_plan(fragments).await?;

        // Cache the result
        self.dp_cache.insert(cache_key, plan.clone());

        Ok(plan)
    }

    /// Compute cache key for fragments
    fn compute_cache_key(&self, fragments: &[QueryFragment]) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for fragment in fragments {
            std::hash::Hash::hash(&fragment.fragment_id, &mut hasher);
        }
        format!("{:x}", std::hash::Hasher::finish(&hasher))
    }

    /// Compute optimal join plan
    async fn compute_optimal_plan(&self, fragments: &[QueryFragment]) -> FusekiResult<JoinPlan> {
        // Simplified join planning
        let mut steps = Vec::new();

        if fragments.len() > 1 {
            // Create pairwise joins
            for i in 0..fragments.len() - 1 {
                steps.push(JoinStep {
                    operation: JoinOperation::HashJoin,
                    left_source: fragments[i].fragment_id.clone(),
                    right_source: fragments[i + 1].fragment_id.clone(),
                    output_destination: format!("join_{i}"),
                });
            }
        }

        Ok(JoinPlan {
            steps,
            estimated_cost: fragments.len() as f64,
            estimated_time_ms: fragments.len() as u64 * 100,
        })
    }
}

impl Default for JoinCostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl JoinCostModel {
    pub fn new() -> Self {
        Self {
            latency_map: DashMap::new(),
            bandwidth_map: DashMap::new(),
        }
    }
}

impl Default for CostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl CostEstimator {
    pub fn new() -> Self {
        Self {
            history: Arc::new(RwLock::new(QueryHistory::new())),
            ml_model: None,
            cardinality: Arc::new(CardinalityEstimator::new()),
        }
    }

    /// Estimate cost of execution plan
    pub async fn estimate_cost(&self, plan: &ExecutionPlan) -> FusekiResult<f64> {
        let mut total_cost = 0.0;

        // Estimate fragment costs
        for fragment in &plan.fragments {
            let fragment_cost = self.estimate_fragment_cost(fragment).await?;
            total_cost += fragment_cost;
        }

        // Add join costs
        for step in &plan.join_plan.steps {
            let join_cost = self.estimate_join_cost(step).await?;
            total_cost += join_cost;
        }

        Ok(total_cost)
    }

    /// Estimate cost of a query fragment
    async fn estimate_fragment_cost(&self, fragment: &QueryFragment) -> FusekiResult<f64> {
        // Use historical data if available
        let history = self.history.read().await;
        if let Some(stats) = history.patterns.get(&fragment.fragment_id) {
            return Ok(stats.avg_execution_time);
        }

        // Otherwise use cardinality estimation
        let cardinality = self
            .cardinality
            .estimate_cardinality(&fragment.sparql)
            .await?;
        Ok(cardinality as f64 * 0.001) // 1ms per 1000 results
    }

    /// Estimate cost of a join operation
    async fn estimate_join_cost(&self, step: &JoinStep) -> FusekiResult<f64> {
        match step.operation {
            JoinOperation::HashJoin => Ok(10.0),
            JoinOperation::SortMergeJoin => Ok(20.0),
            JoinOperation::NestedLoopJoin => Ok(100.0),
            JoinOperation::BroadcastJoin => Ok(5.0),
            JoinOperation::IndexJoin => Ok(2.0),
        }
    }
}

impl Default for QueryHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryHistory {
    pub fn new() -> Self {
        Self {
            executions: Vec::new(),
            patterns: HashMap::new(),
        }
    }
}

impl Default for CardinalityEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl CardinalityEstimator {
    pub fn new() -> Self {
        Self {
            endpoint_stats: DashMap::new(),
            histograms: DashMap::new(),
        }
    }

    /// Estimate result cardinality
    pub async fn estimate_cardinality(&self, query: &str) -> FusekiResult<u64> {
        // Simplified cardinality estimation
        if query.contains("LIMIT") {
            if let Some(limit_pos) = query.find("LIMIT") {
                let limit_str = &query[limit_pos + 5..].trim();

                // Extract the limit value (handle both cases: with space after or at end of query)
                let limit_val = if let Some(space_pos) = limit_str.find(' ') {
                    &limit_str[..space_pos]
                } else {
                    // No space found, use the entire remaining string
                    limit_str
                };

                if let Ok(limit) = limit_val.parse::<u64>() {
                    return Ok(limit);
                }
            }
        }

        // Default estimate
        Ok(1000)
    }
}

impl Default for FederationStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl FederationStatistics {
    pub fn new() -> Self {
        Self {
            query_stats: HashMap::new(),
            endpoint_stats: HashMap::new(),
        }
    }
}
