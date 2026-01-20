//! Federated query executor for parallel service execution

use futures::future::join_all;
use reqwest::Client;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{sync::Semaphore, time::timeout};

use oxirs_arq::query::{Query, QueryType};
use oxirs_core::query::QueryResults;

use crate::{
    error::{FusekiError, FusekiResult},
    federation::{
        health::HealthMonitor,
        planner::{
            ExecutionStep, ExecutionStrategy, FederatedQueryPlan, QueryPlanner, ServiceSelection,
        },
        FederationConfig,
    },
};

/// Query execution result
#[derive(Debug)]
pub struct QueryResult {
    /// Query results
    pub results: QueryResults,
    /// Execution metadata
    pub metadata: QueryMetadata,
}

/// Query execution metadata
#[derive(Debug, Clone, Default)]
pub struct QueryMetadata {
    /// Execution time
    pub execution_time: Option<Duration>,
    /// Service that executed the query
    pub service_id: Option<String>,
    /// Number of results
    pub result_count: usize,
}

impl QueryResult {
    /// Create a new empty result
    pub fn new_empty() -> Self {
        Self {
            results: QueryResults::Boolean(false),
            metadata: QueryMetadata::default(),
        }
    }

    /// Get size hint for result
    pub fn size_hint(&self) -> usize {
        self.metadata.result_count
    }
}

/// Federated query executor
pub struct FederatedExecutor {
    config: FederationConfig,
    http_client: Client,
    semaphore: Arc<Semaphore>,
    planner: Arc<QueryPlanner>,
    health_monitor: Arc<HealthMonitor>,
}

/// Execution context for a federated query
#[derive(Debug)]
struct ExecutionContext {
    /// Query plan
    plan: FederatedQueryPlan,
    /// Intermediate results
    results: HashMap<String, QueryResult>,
    /// Completed steps
    completed_steps: HashSet<String>,
    /// Execution metrics
    metrics: ExecutionMetrics,
}

/// Metrics for query execution
#[derive(Debug, Default)]
struct ExecutionMetrics {
    /// Total execution time
    total_time: Option<Duration>,
    /// Time per step
    step_times: HashMap<String, Duration>,
    /// Service calls made
    service_calls: u32,
    /// Failed service calls
    failed_calls: u32,
    /// Bytes transferred
    bytes_transferred: u64,
}

impl FederatedExecutor {
    /// Create a new federated executor
    pub fn new(
        config: FederationConfig,
        planner: Arc<QueryPlanner>,
        health_monitor: Arc<HealthMonitor>,
    ) -> Self {
        let max_concurrent = config.max_concurrent_requests;

        Self {
            http_client: Client::builder()
                .timeout(config.request_timeout)
                .pool_max_idle_per_host(max_concurrent)
                .build()
                .expect("HTTP client builder should succeed"),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            config,
            planner,
            health_monitor,
        }
    }

    /// Execute a federated query plan
    pub async fn execute(&self, plan: FederatedQueryPlan) -> FusekiResult<QueryResult> {
        let start = Instant::now();

        let mut context = ExecutionContext {
            plan: plan.clone(),
            results: HashMap::new(),
            completed_steps: HashSet::new(),
            metrics: ExecutionMetrics::default(),
        };

        // Execute based on strategy
        let result = match plan.strategy {
            ExecutionStrategy::Sequential => self.execute_sequential(&mut context).await?,
            ExecutionStrategy::Parallel => self.execute_parallel(&mut context).await?,
            ExecutionStrategy::Adaptive => self.execute_adaptive(&mut context).await?,
        };

        // Update metrics
        context.metrics.total_time = Some(start.elapsed());
        self.report_metrics(&context.metrics);

        Ok(result)
    }

    /// Execute steps sequentially
    async fn execute_sequential(
        &self,
        context: &mut ExecutionContext,
    ) -> FusekiResult<QueryResult> {
        let mut final_result = QueryResult::new_empty();

        // Clone the steps to avoid borrow checker issues
        let steps = context.plan.steps.clone();
        for step in &steps {
            let result = self.execute_step(step, context).await?;
            // Store step completion status instead of cloning results
            context.completed_steps.insert(step.id.clone());
            final_result = result;
        }

        Ok(final_result)
    }

    /// Execute steps in parallel
    async fn execute_parallel(&self, context: &mut ExecutionContext) -> FusekiResult<QueryResult> {
        // Group steps by dependencies
        let step_groups = self.group_steps_by_dependencies(&context.plan.steps);

        let mut final_result = QueryResult::new_empty();

        // Execute each group in sequence, but steps within group in parallel
        for group in step_groups {
            let mut group_results = Vec::new();

            // Execute steps in this group concurrently
            let futures = group.into_iter().map(|step| {
                let step = step.clone();
                async move {
                    // Create a temporary context for this step
                    let _start = std::time::Instant::now();
                    let primary_service =
                        step.services.iter().find(|s| s.is_primary).ok_or_else(|| {
                            FusekiError::Configuration {
                                message: "No primary service for step".to_string(),
                            }
                        })?;

                    self.execute_on_service_standalone(primary_service, &step.sub_query)
                        .await
                        .map(|result| (step.id.clone(), result))
                }
            });

            let results = join_all(futures).await;

            // Process results and update main context
            for result in results {
                match result {
                    Ok((step_id, res)) => {
                        context.completed_steps.insert(step_id);
                        group_results.push(res);
                    }
                    Err(e) => return Err(e),
                }
            }

            // Use the last result as final result (or merge if needed)
            if let Some(last_result) = group_results.pop() {
                final_result = last_result;
            }
        }

        Ok(final_result)
    }

    /// Execute with adaptive strategy
    async fn execute_adaptive(&self, context: &mut ExecutionContext) -> FusekiResult<QueryResult> {
        // Start with parallel, fall back to sequential on errors
        match self.execute_parallel(context).await {
            Ok(result) => Ok(result),
            Err(_) => {
                tracing::warn!("Parallel execution failed, falling back to sequential");
                context.results.clear();
                context.metrics = ExecutionMetrics::default();
                self.execute_sequential(context).await
            }
        }
    }

    /// Execute a single step
    async fn execute_step(
        &self,
        step: &ExecutionStep,
        context: &mut ExecutionContext,
    ) -> FusekiResult<QueryResult> {
        let start = Instant::now();

        // Check if we should use circuit breaker
        let primary_service = step.services.iter().find(|s| s.is_primary).ok_or_else(|| {
            FusekiError::Configuration {
                message: "No primary service for step".to_string(),
            }
        })?;

        if !self
            .health_monitor
            .should_use_service(&primary_service.service_id)
            .await
        {
            // Try fallback services
            for service in &step.services {
                if !service.is_primary
                    && self
                        .health_monitor
                        .should_use_service(&service.service_id)
                        .await
                {
                    return self
                        .execute_on_service(service, &step.sub_query, context)
                        .await;
                }
            }

            return Err(FusekiError::ServiceUnavailable {
                message: format!("All services unavailable for step {}", step.id),
            });
        }

        // Execute on primary service
        let result = self
            .execute_on_service(primary_service, &step.sub_query, context)
            .await;

        // Update metrics
        context
            .metrics
            .step_times
            .insert(step.id.clone(), start.elapsed());

        // Update planner statistics
        if let Ok(ref res) = result {
            self.planner
                .update_statistics(
                    &primary_service.service_id,
                    format!("step_{}", step.id),
                    res.size_hint(),
                    start.elapsed(),
                    true,
                )
                .await;
        }

        result
    }

    /// Execute query on a specific service (standalone version for parallel execution)
    async fn execute_on_service_standalone(
        &self,
        service: &ServiceSelection,
        query: &Query,
    ) -> FusekiResult<QueryResult> {
        // Acquire semaphore permit
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| FusekiError::QueryExecution {
                message: "Failed to acquire semaphore".to_string(),
            })?;

        // Prepare request
        let query_string = self.serialize_query(query)?;

        let response = match timeout(
            self.config.request_timeout,
            self.http_client
                .post(service.service_url.as_str())
                .header("Content-Type", "application/sparql-query")
                .header("Accept", self.get_accept_header(&query.query_type))
                .body(query_string)
                .send(),
        )
        .await
        {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                return Err(FusekiError::QueryExecution {
                    message: format!("Service request failed: {e}"),
                });
            }
            Err(_) => {
                return Err(FusekiError::QueryExecution {
                    message: "Service request timed out".to_string(),
                });
            }
        };

        if !response.status().is_success() {
            return Err(FusekiError::QueryExecution {
                message: format!("Service returned error: {}", response.status()),
            });
        }

        // Parse response based on query type
        self.parse_response(response, &query.query_type).await
    }

    /// Execute query on a specific service
    async fn execute_on_service(
        &self,
        service: &ServiceSelection,
        query: &Query,
        context: &mut ExecutionContext,
    ) -> FusekiResult<QueryResult> {
        // Acquire semaphore permit
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| FusekiError::QueryExecution {
                message: "Failed to acquire semaphore".to_string(),
            })?;

        context.metrics.service_calls += 1;

        // Prepare request
        let query_string = self.serialize_query(query)?;

        let response = match timeout(
            self.config.request_timeout,
            self.http_client
                .post(service.service_url.as_str())
                .header("Content-Type", "application/sparql-query")
                .header("Accept", self.get_accept_header(&query.query_type))
                .body(query_string)
                .send(),
        )
        .await
        {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                context.metrics.failed_calls += 1;
                return Err(FusekiError::QueryExecution {
                    message: format!("Service request failed: {e}"),
                });
            }
            Err(_) => {
                context.metrics.failed_calls += 1;
                return Err(FusekiError::QueryExecution {
                    message: "Service request timed out".to_string(),
                });
            }
        };

        if !response.status().is_success() {
            context.metrics.failed_calls += 1;
            return Err(FusekiError::QueryExecution {
                message: format!("Service returned error: {}", response.status()),
            });
        }

        // Track bytes transferred
        if let Some(len) = response.content_length() {
            context.metrics.bytes_transferred += len;
        }

        // Parse response based on query type
        self.parse_response(response, &query.query_type).await
    }

    /// Parse HTTP response into QueryResult
    async fn parse_response(
        &self,
        response: reqwest::Response,
        query_type: &QueryType,
    ) -> FusekiResult<QueryResult> {
        let response_text = response
            .text()
            .await
            .map_err(|e| FusekiError::QueryExecution {
                message: format!("Failed to read response: {e}"),
            })?;

        // Parse based on query type
        let results = match query_type {
            QueryType::Select => {
                // Parse SPARQL JSON results
                let json: serde_json::Value =
                    serde_json::from_str(&response_text).map_err(|e| {
                        FusekiError::QueryExecution {
                            message: format!("Invalid JSON response: {e}"),
                        }
                    })?;

                // Parse actual bindings from JSON
                let variable_bindings = self.parse_sparql_json_bindings(&json)?;
                // Convert VariableBinding to Solution
                let solutions: Vec<oxirs_core::query::Solution> = variable_bindings
                    .into_iter()
                    .map(|vb| {
                        let mut solution = oxirs_core::query::Solution::new();
                        for var_name in vb.variables() {
                            if let Some(term) = vb.get(var_name) {
                                if let Ok(var) = oxirs_core::model::Variable::new(var_name) {
                                    solution.bind(var, term.clone());
                                }
                            }
                        }
                        solution
                    })
                    .collect();
                QueryResults::Solutions(solutions)
            }
            QueryType::Ask => {
                let json: serde_json::Value =
                    serde_json::from_str(&response_text).map_err(|e| {
                        FusekiError::QueryExecution {
                            message: format!("Invalid JSON response: {e}"),
                        }
                    })?;

                let boolean_result = json
                    .get("boolean")
                    .and_then(|b| b.as_bool())
                    .unwrap_or(false);

                QueryResults::Boolean(boolean_result)
            }
            QueryType::Construct | QueryType::Describe => {
                // Parse N-Triples or Turtle response
                let quads = self.parse_graph_response(&response_text)?;
                // Convert Quads to Triples (drop graph information)
                let triples: Vec<oxirs_core::model::Triple> = quads
                    .into_iter()
                    .map(|quad| {
                        oxirs_core::model::Triple::new(
                            quad.subject().clone(),
                            quad.predicate().clone(),
                            quad.object().clone(),
                        )
                    })
                    .collect();
                QueryResults::Graph(triples)
            }
        };

        let result_count = match &results {
            QueryResults::Solutions(solutions) => solutions.len(),
            QueryResults::Boolean(_) => 1,
            QueryResults::Graph(graph) => graph.len(),
        };

        Ok(QueryResult {
            results,
            metadata: QueryMetadata {
                execution_time: None, // Will be set by caller
                service_id: None,
                result_count,
            },
        })
    }

    /// Serialize query to SPARQL string
    fn serialize_query(&self, query: &Query) -> FusekiResult<String> {
        // Use oxirs-arq's built-in query serialization
        match query.to_string() {
            query_str if !query_str.is_empty() => Ok(query_str),
            _ => Err(FusekiError::QueryExecution {
                message: "Failed to serialize query".to_string(),
            }),
        }
    }

    /// Get appropriate Accept header for query type
    fn get_accept_header(&self, query_type: &QueryType) -> &'static str {
        match query_type {
            QueryType::Select | QueryType::Ask => "application/sparql-results+json",
            QueryType::Construct | QueryType::Describe => "application/n-triples",
        }
    }

    /// Group execution steps by their dependencies
    fn group_steps_by_dependencies(&self, steps: &[ExecutionStep]) -> Vec<Vec<ExecutionStep>> {
        let mut groups = Vec::new();
        let mut remaining_steps: HashMap<String, ExecutionStep> = steps
            .iter()
            .map(|step| (step.id.clone(), step.clone()))
            .collect();
        let mut processed = std::collections::HashSet::new();

        while !remaining_steps.is_empty() {
            let mut current_group = Vec::new();

            // Find steps with no unresolved dependencies
            let ready_steps: Vec<String> = remaining_steps
                .keys()
                .filter(|step_id| {
                    remaining_steps
                        .get(*step_id)
                        .map(|step| step.dependencies.iter().all(|dep| processed.contains(dep)))
                        .unwrap_or(false)
                })
                .cloned()
                .collect();

            if ready_steps.is_empty() {
                // No more resolvable dependencies - break potential cycles
                // by taking first remaining step
                if let Some((first_id, _)) = remaining_steps.iter().next() {
                    let first_id = first_id.clone();
                    if let Some(step) = remaining_steps.remove(&first_id) {
                        current_group.push(step);
                        processed.insert(first_id);
                    }
                }
            } else {
                // Add all ready steps to current group
                for step_id in ready_steps {
                    if let Some(step) = remaining_steps.remove(&step_id) {
                        current_group.push(step);
                        processed.insert(step_id);
                    }
                }
            }

            if !current_group.is_empty() {
                groups.push(current_group);
            } else {
                // Safety break to prevent infinite loop
                break;
            }
        }

        groups
    }

    /// Parse SPARQL JSON bindings into Solutions
    fn parse_sparql_json_bindings(
        &self,
        json: &serde_json::Value,
    ) -> FusekiResult<Vec<oxirs_core::rdf_store::VariableBinding>> {
        let bindings_array = json
            .get("results")
            .and_then(|r| r.get("bindings"))
            .and_then(|b| b.as_array())
            .ok_or_else(|| FusekiError::QueryExecution {
                message: "Invalid SPARQL JSON format: missing results.bindings".to_string(),
            })?;

        let mut solutions = Vec::new();

        for binding_obj in bindings_array {
            let mut variable_binding = oxirs_core::rdf_store::VariableBinding::new();

            if let Some(binding_map) = binding_obj.as_object() {
                for (var_name, term_obj) in binding_map {
                    if let Some(term) = self.parse_sparql_json_term(term_obj)? {
                        variable_binding.bind(var_name.clone(), term);
                    }
                }
            }

            solutions.push(variable_binding);
        }

        Ok(solutions)
    }

    /// Parse a single SPARQL JSON term
    fn parse_sparql_json_term(
        &self,
        term_obj: &serde_json::Value,
    ) -> FusekiResult<Option<oxirs_core::model::Term>> {
        let term_type = term_obj
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| FusekiError::QueryExecution {
                message: "Invalid SPARQL JSON term: missing type".to_string(),
            })?;

        let value = term_obj
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FusekiError::QueryExecution {
                message: "Invalid SPARQL JSON term: missing value".to_string(),
            })?;

        let term = match term_type {
            "uri" => oxirs_core::model::Term::NamedNode(
                oxirs_core::model::NamedNode::new(value)
                    .expect("IRI from remote endpoint should be valid"),
            ),
            "bnode" => oxirs_core::model::Term::BlankNode(
                oxirs_core::model::BlankNode::new(value)
                    .expect("blank node from remote endpoint should be valid"),
            ),
            "literal" => {
                let language = term_obj
                    .get("xml:lang")
                    .and_then(|l| l.as_str())
                    .map(|s| s.to_string());

                let datatype = term_obj.get("datatype").and_then(|d| d.as_str()).map(|s| {
                    oxirs_core::model::NamedNode::new(s).expect("graph IRI should be valid")
                });

                oxirs_core::model::Term::Literal(if let Some(lang) = language {
                    oxirs_core::model::Literal::new_language_tagged_literal(value, lang)
                        .expect("language-tagged literal should be valid")
                } else if let Some(dt) = datatype {
                    oxirs_core::model::Literal::new_typed_literal(value, dt)
                } else {
                    oxirs_core::model::Literal::new_simple_literal(value)
                })
            }
            _ => {
                return Err(FusekiError::QueryExecution {
                    message: format!("Unknown SPARQL JSON term type: {}", term_type),
                });
            }
        };

        Ok(Some(term))
    }

    /// Parse graph response from N-Triples or Turtle
    fn parse_graph_response(
        &self,
        response_text: &str,
    ) -> FusekiResult<Vec<oxirs_core::model::Quad>> {
        let mut quads = Vec::new();

        // Simple N-Triples parsing (lines ending with '.')
        for line in response_text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.ends_with('.') {
                if let Some(quad) = self.parse_ntriples_line(line)? {
                    quads.push(quad);
                }
            }
        }

        Ok(quads)
    }

    /// Parse a single N-Triples line into a Quad
    fn parse_ntriples_line(&self, line: &str) -> FusekiResult<Option<oxirs_core::model::Quad>> {
        let line = line.trim_end_matches('.');
        let parts = self.split_ntriples_terms(line)?;

        if parts.len() < 3 {
            return Ok(None);
        }

        let subject = self.parse_ntriples_term(&parts[0])?;
        let predicate = self.parse_ntriples_term(&parts[1])?;
        let object = self.parse_ntriples_term(&parts[2])?;

        // Default graph for federation results
        let graph = oxirs_core::model::GraphName::NamedNode(
            oxirs_core::model::NamedNode::new("http://default-graph")
                .expect("hardcoded default graph IRI should be valid"),
        );

        let subject_pos: oxirs_core::model::Subject =
            subject
                .try_into()
                .map_err(|_| FusekiError::QueryExecution {
                    message: "Invalid subject term".to_string(),
                })?;
        let predicate_pos: oxirs_core::model::Predicate =
            predicate
                .try_into()
                .map_err(|_| FusekiError::QueryExecution {
                    message: "Invalid predicate term".to_string(),
                })?;
        let object_pos: oxirs_core::model::Object = object.into();

        Ok(Some(oxirs_core::model::Quad::new(
            subject_pos,
            predicate_pos,
            object_pos,
            graph,
        )))
    }

    /// Split N-Triples line into terms
    fn split_ntriples_terms(&self, line: &str) -> FusekiResult<Vec<String>> {
        let mut terms = Vec::new();
        let mut current_term = String::new();
        let mut in_quotes = false;
        let chars = line.chars();

        for ch in chars {
            match ch {
                '"' if !in_quotes => {
                    in_quotes = true;
                    current_term.push(ch);
                }
                '"' if in_quotes => {
                    in_quotes = false;
                    current_term.push(ch);
                }
                ' ' | '\t' if !in_quotes => {
                    if !current_term.is_empty() {
                        terms.push(current_term.trim().to_string());
                        current_term.clear();
                    }
                }
                _ => {
                    current_term.push(ch);
                }
            }
        }

        if !current_term.is_empty() {
            terms.push(current_term.trim().to_string());
        }

        Ok(terms)
    }

    /// Parse N-Triples term
    fn parse_ntriples_term(&self, term_str: &str) -> FusekiResult<oxirs_core::model::Term> {
        let term_str = term_str.trim();

        if term_str.starts_with('<') && term_str.ends_with('>') {
            // IRI
            let iri = &term_str[1..term_str.len() - 1];
            Ok(oxirs_core::model::Term::NamedNode(
                oxirs_core::model::NamedNode::new(iri).expect("IRI from N-Triples should be valid"),
            ))
        } else if let Some(bnode) = term_str.strip_prefix("_:") {
            // Blank node
            Ok(oxirs_core::model::Term::BlankNode(
                oxirs_core::model::BlankNode::new(bnode)
                    .expect("blank node from N-Triples should be valid"),
            ))
        } else if term_str.starts_with('"') {
            // Literal
            self.parse_ntriples_literal(term_str)
        } else {
            Err(FusekiError::QueryExecution {
                message: format!("Invalid N-Triples term: {}", term_str),
            })
        }
    }

    /// Parse N-Triples literal
    fn parse_ntriples_literal(&self, literal_str: &str) -> FusekiResult<oxirs_core::model::Term> {
        if let Some(end_quote) = literal_str[1..].find('"') {
            let value = &literal_str[1..end_quote + 1];
            let rest = &literal_str[end_quote + 2..];

            if let Some(lang) = rest.strip_prefix('@') {
                // Language tag
                Ok(oxirs_core::model::Term::Literal(
                    oxirs_core::model::Literal::new_language_tagged_literal(value, lang)
                        .expect("language-tagged literal should be valid"),
                ))
            } else if let Some(datatype) = rest.strip_prefix("^^") {
                // Datatype
                let datatype = if datatype.starts_with('<') && datatype.ends_with('>') {
                    &datatype[1..datatype.len() - 1]
                } else {
                    datatype
                };
                Ok(oxirs_core::model::Term::Literal(
                    oxirs_core::model::Literal::new_typed_literal(
                        value,
                        oxirs_core::model::NamedNode::new(datatype)
                            .expect("datatype IRI from remote endpoint should be valid"),
                    ),
                ))
            } else {
                // Plain literal
                Ok(oxirs_core::model::Term::Literal(
                    oxirs_core::model::Literal::new_simple_literal(value),
                ))
            }
        } else {
            Err(FusekiError::QueryExecution {
                message: format!("Invalid N-Triples literal: {}", literal_str),
            })
        }
    }

    /// Report execution metrics
    fn report_metrics(&self, metrics: &ExecutionMetrics) {
        tracing::info!(
            "Federated query executed in {:?} with {} service calls ({} failed)",
            metrics.total_time.unwrap_or_default(),
            metrics.service_calls,
            metrics.failed_calls
        );

        if !metrics.step_times.is_empty() {
            tracing::debug!("Step execution times: {:?}", metrics.step_times);
        }

        if metrics.bytes_transferred > 0 {
            tracing::debug!("Bytes transferred: {}", metrics.bytes_transferred);
        }
    }
}

/// Parallel result merger for combining results from multiple services
pub struct FederatedResultMerger {
    merge_strategy: FederatedMergeStrategy,
}

#[derive(Debug, Clone)]
pub enum FederatedMergeStrategy {
    /// Union all results (default)
    Union,
    /// Intersection of results
    Intersection,
    /// Join on specific variables
    Join(Vec<String>),
    /// Custom merge function
    Custom,
}

impl FederatedResultMerger {
    /// Create a new result merger
    pub fn new(strategy: FederatedMergeStrategy) -> Self {
        Self {
            merge_strategy: strategy,
        }
    }

    /// Merge multiple query results
    pub async fn merge(&self, results: Vec<QueryResult>) -> FusekiResult<QueryResult> {
        if results.is_empty() {
            return Ok(QueryResult::new_empty());
        }

        if results.len() == 1 {
            return Ok(results
                .into_iter()
                .next()
                .expect("results should not be empty after non_empty check"));
        }

        match &self.merge_strategy {
            FederatedMergeStrategy::Union => self.merge_union(results).await,
            FederatedMergeStrategy::Intersection => self.merge_intersection(results).await,
            FederatedMergeStrategy::Join(vars) => self.merge_join(results, vars).await,
            FederatedMergeStrategy::Custom => Err(FusekiError::QueryExecution {
                message: "Custom merge not implemented".to_string(),
            }),
        }
    }

    /// Merge results using union
    async fn merge_union(&self, mut results: Vec<QueryResult>) -> FusekiResult<QueryResult> {
        use oxirs_core::query::Solution;
        use std::collections::HashSet;

        if results.is_empty() {
            return Ok(QueryResult::new_empty());
        }

        if results.len() == 1 {
            return Ok(results
                .pop()
                .expect("results should not be empty after non_empty check"));
        }

        // Extract the first result as base
        let mut base = results.remove(0);

        // Merge each additional result
        for result in results {
            match (&mut base.results, &result.results) {
                // Boolean: OR operation
                (QueryResults::Boolean(ref mut b1), QueryResults::Boolean(b2)) => {
                    *b1 = *b1 || *b2;
                }
                // Solutions: Union of all solutions (remove duplicates)
                (QueryResults::Solutions(ref mut sols1), QueryResults::Solutions(sols2)) => {
                    // Use HashSet to track unique solutions
                    let mut seen = HashSet::new();
                    for sol in sols1.iter() {
                        seen.insert(format!("{:?}", sol)); // Simple string-based dedup
                    }

                    for sol in sols2 {
                        let key = format!("{:?}", sol);
                        if !seen.contains(&key) {
                            sols1.push(sol.clone());
                            seen.insert(key);
                        }
                    }

                    base.metadata.result_count = sols1.len();
                }
                // Graph: Union of triples (remove duplicates)
                (QueryResults::Graph(ref mut triples1), QueryResults::Graph(triples2)) => {
                    let mut seen = HashSet::new();
                    for triple in triples1.iter() {
                        seen.insert(format!("{:?}", triple));
                    }

                    for triple in triples2 {
                        let key = format!("{:?}", triple);
                        if !seen.contains(&key) {
                            triples1.push(triple.clone());
                            seen.insert(key);
                        }
                    }

                    base.metadata.result_count = triples1.len();
                }
                // Incompatible types
                _ => {
                    return Err(FusekiError::QueryExecution {
                        message: "Cannot union incompatible result types".to_string(),
                    });
                }
            }

            // Update execution time
            if let (Some(t1), Some(t2)) =
                (base.metadata.execution_time, result.metadata.execution_time)
            {
                base.metadata.execution_time = Some(t1 + t2);
            }
        }

        Ok(base)
    }

    /// Merge results using intersection
    async fn merge_intersection(&self, mut results: Vec<QueryResult>) -> FusekiResult<QueryResult> {
        use std::collections::HashSet;

        if results.is_empty() {
            return Ok(QueryResult::new_empty());
        }

        if results.len() == 1 {
            return Ok(results
                .pop()
                .expect("results should not be empty after non_empty check"));
        }

        // Extract the first result as base
        let mut base = results.remove(0);

        // Compute intersection with each additional result
        for result in results {
            match (&mut base.results, &result.results) {
                // Boolean: AND operation
                (QueryResults::Boolean(ref mut b1), QueryResults::Boolean(b2)) => {
                    *b1 = *b1 && *b2;
                }
                // Solutions: Intersection of solutions
                (QueryResults::Solutions(ref mut sols1), QueryResults::Solutions(sols2)) => {
                    // Build set of solutions from sols2
                    let set2: HashSet<String> =
                        sols2.iter().map(|sol| format!("{:?}", sol)).collect();

                    // Keep only solutions that exist in both
                    sols1.retain(|sol| {
                        let key = format!("{:?}", sol);
                        set2.contains(&key)
                    });

                    base.metadata.result_count = sols1.len();
                }
                // Graph: Intersection of triples
                (QueryResults::Graph(ref mut triples1), QueryResults::Graph(triples2)) => {
                    // Build set of triples from triples2
                    let set2: HashSet<String> = triples2
                        .iter()
                        .map(|triple| format!("{:?}", triple))
                        .collect();

                    // Keep only triples that exist in both
                    triples1.retain(|triple| {
                        let key = format!("{:?}", triple);
                        set2.contains(&key)
                    });

                    base.metadata.result_count = triples1.len();
                }
                // Incompatible types
                _ => {
                    return Err(FusekiError::QueryExecution {
                        message: "Cannot intersect incompatible result types".to_string(),
                    });
                }
            }

            // Update execution time
            if let (Some(t1), Some(t2)) =
                (base.metadata.execution_time, result.metadata.execution_time)
            {
                base.metadata.execution_time = Some(t1 + t2);
            }
        }

        Ok(base)
    }

    /// Merge results using join
    async fn merge_join(
        &self,
        mut results: Vec<QueryResult>,
        join_vars: &[String],
    ) -> FusekiResult<QueryResult> {
        use oxirs_core::model::Variable;
        use std::collections::HashMap;

        if results.is_empty() {
            return Ok(QueryResult::new_empty());
        }

        if results.len() == 1 {
            return Ok(results
                .pop()
                .expect("results should not be empty after non_empty check"));
        }

        // Only solutions can be joined (Boolean and Graph don't support join)
        let mut base = results.remove(0);

        if let QueryResults::Solutions(ref mut sols1) = base.results {
            for result in results {
                if let QueryResults::Solutions(sols2) = result.results {
                    // Perform natural join on specified variables
                    let mut joined_solutions = Vec::new();

                    // Convert join_vars to Variable objects for comparison
                    let join_variables: Vec<Variable> = join_vars
                        .iter()
                        .map(|v| {
                            Variable::new(v).unwrap_or_else(|_| {
                                Variable::new("_").expect("underscore variable should be valid")
                            })
                        })
                        .collect();

                    // If no join variables specified, use all common variables
                    let actual_join_vars = if join_variables.is_empty() {
                        // Find common variables between solutions (simplified)
                        Vec::new() // Fallback to Cartesian product if no common vars
                    } else {
                        join_variables
                    };

                    // Perform join: for each solution in sols1, find matching solutions in sols2
                    for sol1 in sols1.iter() {
                        for sol2 in &sols2 {
                            // Check if join variables match
                            let mut matches = true;

                            if !actual_join_vars.is_empty() {
                                for join_var in &actual_join_vars {
                                    // Both solutions must have the variable with the same value
                                    let val1_opt = sol1.get(join_var);
                                    let val2_opt = sol2.get(join_var);

                                    match (val1_opt, val2_opt) {
                                        (Some(v1), Some(v2)) if v1 == v2 => {
                                            // Values match, continue checking
                                        }
                                        (None, None) => {
                                            // Neither has the variable, that's ok
                                        }
                                        _ => {
                                            // Mismatch or only one has the variable
                                            matches = false;
                                            break;
                                        }
                                    }
                                }
                            }

                            if matches {
                                // Merge the two solutions
                                if let Some(merged) = sol1.merge(sol2) {
                                    joined_solutions.push(merged);
                                }
                            }
                        }
                    }

                    *sols1 = joined_solutions;
                    base.metadata.result_count = sols1.len();

                    // Update execution time
                    if let (Some(t1), Some(t2)) =
                        (base.metadata.execution_time, result.metadata.execution_time)
                    {
                        base.metadata.execution_time = Some(t1 + t2);
                    }
                } else {
                    return Err(FusekiError::QueryExecution {
                        message: "Can only join Solutions result types".to_string(),
                    });
                }
            }

            Ok(base)
        } else {
            Err(FusekiError::QueryExecution {
                message: "Join operation requires Solutions result type".to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_merger() {
        let _merger = FederatedResultMerger::new(FederatedMergeStrategy::Union);
        // Test would go here with proper QueryResult implementation
    }
}
