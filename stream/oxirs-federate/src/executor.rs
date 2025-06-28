//! Federated Query Execution Engine
//!
//! This module handles the execution of federated queries across multiple services,
//! including parallel execution, timeout handling, and fault tolerance.

use anyhow::{anyhow, Result};
use futures::{stream, StreamExt, TryStreamExt};
use reqwest::{
    header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE},
    Client,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, error, info, instrument, warn};

use crate::{
    cache::{CacheConfig, FederationCache},
    service_executor::{JoinExecutor, ServiceExecutor, ServiceExecutorConfig},
    service_optimizer::{OptimizedServiceClause, ServiceExecutionStrategy},
    ExecutionPlan, ExecutionStep, FederatedService, FederationError, ServiceRegistry, StepType,
};

/// Federated query executor
#[derive(Debug)]
pub struct FederatedExecutor {
    client: Client,
    config: FederatedExecutorConfig,
    service_executor: Arc<ServiceExecutor>,
    join_executor: Arc<JoinExecutor>,
    cache: Arc<FederationCache>,
}

impl FederatedExecutor {
    /// Create a new federated executor with default configuration
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("oxirs-federate/1.0")
            .build()
            .expect("Failed to create HTTP client");

        let cache = Arc::new(FederationCache::new());
        let service_executor = Arc::new(ServiceExecutor::new(cache.clone()));
        let join_executor = Arc::new(JoinExecutor::new());

        Self {
            client,
            config: FederatedExecutorConfig::default(),
            service_executor,
            join_executor,
            cache,
        }
    }

    /// Create a new federated executor with custom configuration
    pub fn with_config(config: FederatedExecutorConfig) -> Self {
        let client = Client::builder()
            .timeout(config.request_timeout)
            .user_agent(&config.user_agent)
            .build()
            .expect("Failed to create HTTP client");

        let cache = Arc::new(FederationCache::with_config(config.cache_config.clone()));
        let service_executor = Arc::new(ServiceExecutor::with_config(
            config.service_executor_config.clone(),
            cache.clone(),
        ));
        let join_executor = Arc::new(JoinExecutor::new());

        Self {
            client,
            config,
            service_executor,
            join_executor,
            cache,
        }
    }

    /// Execute a federated query plan
    #[instrument(skip(self, plan))]
    pub async fn execute_plan(&self, plan: &ExecutionPlan) -> Result<Vec<StepResult>> {
        info!("Executing federated plan with {} steps", plan.steps.len());

        let start_time = Instant::now();
        let mut results = Vec::new();
        let mut completed_steps = HashMap::new();

        // Execute steps according to dependencies and parallelization
        for parallel_group in &plan.parallelizable_steps {
            let group_results = self
                .execute_parallel_group(parallel_group, plan, &completed_steps)
                .await?;

            for result in group_results {
                completed_steps.insert(result.step_id.clone(), result.clone());
                results.push(result);
            }
        }

        // Execute remaining sequential steps
        for step in &plan.steps {
            if !completed_steps.contains_key(&step.step_id) {
                let result = self.execute_step(step, &completed_steps).await?;
                completed_steps.insert(result.step_id.clone(), result.clone());
                results.push(result);
            }
        }

        let execution_time = start_time.elapsed();
        info!("Plan execution completed in {:?}", execution_time);

        Ok(results)
    }

    /// Execute a group of steps in parallel
    async fn execute_parallel_group(
        &self,
        step_ids: &[String],
        plan: &ExecutionPlan,
        completed_steps: &HashMap<String, StepResult>,
    ) -> Result<Vec<StepResult>> {
        let steps: Vec<_> = step_ids
            .iter()
            .filter_map(|id| plan.steps.iter().find(|s| &s.step_id == id))
            .collect();

        if steps.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Executing {} steps in parallel", steps.len());

        // Execute steps concurrently
        let futures: Vec<_> = steps
            .into_iter()
            .map(|step| self.execute_step(step, completed_steps))
            .collect();

        // Wait for all steps to complete or timeout
        let timeout_duration = self.config.parallel_timeout;
        match timeout(timeout_duration, futures::future::try_join_all(futures)).await {
            Ok(Ok(results)) => Ok(results),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(anyhow!(
                "Parallel execution timed out after {:?}",
                timeout_duration
            )),
        }
    }

    /// Execute a single step
    #[instrument(skip(self, step, completed_steps))]
    async fn execute_step(
        &self,
        step: &ExecutionStep,
        completed_steps: &HashMap<String, StepResult>,
    ) -> Result<StepResult> {
        debug!("Executing step: {} ({})", step.step_id, step.step_type);

        // Check dependencies
        for dep_id in &step.dependencies {
            if !completed_steps.contains_key(dep_id) {
                return Err(anyhow!(
                    "Dependency {} not completed for step {}",
                    dep_id,
                    step.step_id
                ));
            }
        }

        let start_time = Instant::now();

        let result = match step.step_type {
            StepType::ServiceQuery => self.execute_service_query(step).await,
            StepType::GraphQLQuery => self.execute_graphql_query(step).await,
            StepType::Join => self.execute_join(step, completed_steps).await,
            StepType::Union => self.execute_union(step, completed_steps).await,
            StepType::Filter => self.execute_filter(step, completed_steps).await,
            StepType::SchemaStitch => self.execute_schema_stitch(step, completed_steps).await,
            StepType::Aggregate => self.execute_aggregate(step, completed_steps).await,
            StepType::Sort => self.execute_sort(step, completed_steps).await,
        };

        let execution_time = start_time.elapsed();

        match result {
            Ok(data) => {
                debug!("Step {} completed in {:?}", step.step_id, execution_time);
                Ok(StepResult {
                    step_id: step.step_id.clone(),
                    step_type: step.step_type,
                    status: ExecutionStatus::Success,
                    data: Some(data),
                    error: None,
                    execution_time,
                    service_id: step.service_id.clone(),
                })
            }
            Err(e) => {
                error!("Step {} failed: {}", step.step_id, e);
                Ok(StepResult {
                    step_id: step.step_id.clone(),
                    step_type: step.step_type,
                    status: ExecutionStatus::Failed,
                    data: None,
                    error: Some(e.to_string()),
                    execution_time,
                    service_id: step.service_id.clone(),
                })
            }
        }
    }

    /// Execute a SPARQL service query
    async fn execute_service_query(&self, step: &ExecutionStep) -> Result<QueryResultData> {
        let service_id = step
            .service_id
            .as_ref()
            .ok_or_else(|| anyhow!("Service ID required for service query"))?;

        // TODO: Get service details from registry
        let endpoint = format!("http://localhost:8080/sparql"); // Placeholder

        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-query"),
        );
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/sparql-results+json"),
        );

        let response = self
            .client
            .post(&endpoint)
            .headers(headers)
            .body(step.query_fragment.clone())
            .send()
            .await
            .map_err(|e| anyhow!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(anyhow!("Service returned error: {}", response.status()));
        }

        let response_text = response
            .text()
            .await
            .map_err(|e| anyhow!("Failed to read response: {}", e))?;

        // Parse SPARQL results JSON
        let sparql_results: SparqlResults = serde_json::from_str(&response_text)
            .map_err(|e| anyhow!("Failed to parse SPARQL results: {}", e))?;

        Ok(QueryResultData::Sparql(sparql_results))
    }

    /// Execute a GraphQL query
    async fn execute_graphql_query(&self, step: &ExecutionStep) -> Result<QueryResultData> {
        let service_id = step
            .service_id
            .as_ref()
            .ok_or_else(|| anyhow!("Service ID required for GraphQL query"))?;

        // TODO: Get service details from registry
        let endpoint = format!("http://localhost:8080/graphql"); // Placeholder

        let graphql_request = GraphQLRequest {
            query: step.query_fragment.clone(),
            variables: None,
            operation_name: None,
        };

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let response = self
            .client
            .post(&endpoint)
            .headers(headers)
            .json(&graphql_request)
            .send()
            .await
            .map_err(|e| anyhow!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(anyhow!("Service returned error: {}", response.status()));
        }

        let graphql_response: GraphQLResponse = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse GraphQL response: {}", e))?;

        Ok(QueryResultData::GraphQL(graphql_response))
    }

    /// Execute a join operation with enhanced parallel processing
    async fn execute_join(
        &self,
        step: &ExecutionStep,
        completed_steps: &HashMap<String, StepResult>,
    ) -> Result<QueryResultData> {
        debug!("Executing join step: {}", step.step_id);

        // Get results from dependency steps
        let mut input_results = Vec::new();
        for dep_id in &step.dependencies {
            if let Some(dep_result) = completed_steps.get(dep_id) {
                if let Some(data) = &dep_result.data {
                    input_results.push(data);
                }
            }
        }

        if input_results.len() < 2 {
            return Err(anyhow!("Join requires at least 2 input results"));
        }

        // Use the advanced join executor for optimized joins
        self.join_executor
            .execute_advanced_join(&input_results)
            .await
    }

    /// Execute a union operation
    async fn execute_union(
        &self,
        step: &ExecutionStep,
        completed_steps: &HashMap<String, StepResult>,
    ) -> Result<QueryResultData> {
        debug!("Executing union step: {}", step.step_id);

        let mut all_bindings = Vec::new();
        let mut variables = Vec::new();

        for dep_id in &step.dependencies {
            if let Some(dep_result) = completed_steps.get(dep_id) {
                if let Some(QueryResultData::Sparql(sparql_result)) = &dep_result.data {
                    if variables.is_empty() {
                        variables = sparql_result.head.vars.clone();
                    }
                    all_bindings.extend(sparql_result.results.bindings.clone());
                }
            }
        }

        let union_result = SparqlResults {
            head: SparqlHead { vars: variables },
            results: SparqlResultSet {
                bindings: all_bindings,
            },
        };

        Ok(QueryResultData::Sparql(union_result))
    }

    /// Execute a filter operation
    async fn execute_filter(
        &self,
        step: &ExecutionStep,
        completed_steps: &HashMap<String, StepResult>,
    ) -> Result<QueryResultData> {
        debug!("Executing filter step: {}", step.step_id);

        // Get the input data from dependencies
        let mut input_data = None;
        for dep_id in &step.dependencies {
            if let Some(dep_result) = completed_steps.get(dep_id) {
                if let Some(data) = &dep_result.data {
                    input_data = Some(data.clone());
                    break;
                }
            }
        }

        let input_data = input_data.ok_or_else(|| anyhow!("No input data for filter operation"))?;

        match input_data {
            QueryResultData::Sparql(sparql_results) => {
                let filtered_results =
                    self.filter_sparql_results(&sparql_results, &step.query_fragment)?;
                Ok(QueryResultData::Sparql(filtered_results))
            }
            QueryResultData::GraphQL(graphql_response) => {
                // For GraphQL, filters are usually applied at the field level
                // For now, we'll pass through the data as GraphQL filtering is more complex
                warn!("GraphQL filter execution not fully implemented, passing through data");
                Ok(QueryResultData::GraphQL(graphql_response))
            }
        }
    }

    /// Filter SPARQL results based on filter expression
    fn filter_sparql_results(
        &self,
        results: &SparqlResults,
        filter_expr: &str,
    ) -> Result<SparqlResults> {
        let filtered_bindings = results
            .results
            .bindings
            .iter()
            .filter(|binding| self.evaluate_filter_expression(binding, filter_expr))
            .cloned()
            .collect();

        Ok(SparqlResults {
            head: results.head.clone(),
            results: SparqlResultSet {
                bindings: filtered_bindings,
            },
        })
    }

    /// Evaluate a filter expression against a binding
    fn evaluate_filter_expression(&self, binding: &SparqlBinding, filter_expr: &str) -> bool {
        // Parse and evaluate SPARQL filter expressions
        // This is a simplified implementation - full SPARQL filter evaluation is complex
        let filter_expr = filter_expr.trim();

        // Handle common filter patterns
        if filter_expr.contains("REGEX") {
            return self.evaluate_regex_filter(binding, filter_expr);
        }

        if filter_expr.contains("langMatches") {
            return self.evaluate_lang_matches_filter(binding, filter_expr);
        }

        if filter_expr.contains("=") || filter_expr.contains("!=") {
            return self.evaluate_comparison_filter(binding, filter_expr);
        }

        if filter_expr.contains("BOUND") {
            return self.evaluate_bound_filter(binding, filter_expr);
        }

        // For complex expressions, default to true for now
        // In a full implementation, we'd need a proper SPARQL expression parser
        warn!(
            "Complex filter expression not fully supported: {}",
            filter_expr
        );
        true
    }

    /// Evaluate REGEX filter expressions
    fn evaluate_regex_filter(&self, binding: &SparqlBinding, filter_expr: &str) -> bool {
        // Extract variable and regex pattern from REGEX(?var, "pattern")
        if let Some(start) = filter_expr.find("REGEX(") {
            let substr = &filter_expr[start + 6..];
            if let Some(end) = substr.find(')') {
                let args = &substr[..end];
                let parts: Vec<&str> = args.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 2 {
                    let var_name = parts[0].trim_start_matches('?');
                    let pattern = parts[1].trim_matches('"');

                    if let Some(value) = binding.get(var_name) {
                        if let Ok(regex) = regex::Regex::new(pattern) {
                            return regex.is_match(&value.value);
                        }
                    }
                }
            }
        }
        false
    }

    /// Evaluate langMatches filter expressions
    fn evaluate_lang_matches_filter(&self, binding: &SparqlBinding, filter_expr: &str) -> bool {
        // Extract variable and language pattern from langMatches(lang(?var), "lang")
        if let Some(start) = filter_expr.find("langMatches(") {
            let substr = &filter_expr[start + 12..];
            if let Some(end) = substr.find(')') {
                let args = &substr[..end];
                let parts: Vec<&str> = args.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 2 {
                    // Extract variable from lang(?var)
                    let lang_part = parts[0];
                    if let Some(var_start) = lang_part.find("lang(") {
                        let var_part = &lang_part[var_start + 5..];
                        if let Some(var_end) = var_part.find(')') {
                            let var_name = var_part[..var_end].trim_start_matches('?');
                            let lang_pattern = parts[1].trim_matches('"');

                            if let Some(value) = binding.get(var_name) {
                                if let Some(lang) = &value.lang {
                                    return lang == lang_pattern || lang_pattern == "*";
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Evaluate comparison filter expressions (=, !=, <, >, <=, >=)
    fn evaluate_comparison_filter(&self, binding: &SparqlBinding, filter_expr: &str) -> bool {
        let operators = ["!=", "<=", ">=", "=", "<", ">"];

        for op in &operators {
            if let Some(pos) = filter_expr.find(op) {
                let left = filter_expr[..pos].trim();
                let right = filter_expr[pos + op.len()..].trim();

                let left_value = self.resolve_filter_value(binding, left);
                let right_value = self.resolve_filter_value(binding, right);

                return match op {
                    "=" => left_value == right_value,
                    "!=" => left_value != right_value,
                    "<" => left_value < right_value,
                    ">" => left_value > right_value,
                    "<=" => left_value <= right_value,
                    ">=" => left_value >= right_value,
                    _ => false,
                };
            }
        }
        false
    }

    /// Evaluate BOUND filter expressions
    fn evaluate_bound_filter(&self, binding: &SparqlBinding, filter_expr: &str) -> bool {
        if let Some(start) = filter_expr.find("BOUND(") {
            let substr = &filter_expr[start + 6..];
            if let Some(end) = substr.find(')') {
                let var_name = substr[..end].trim().trim_start_matches('?');
                return binding.contains_key(var_name);
            }
        }
        false
    }

    /// Resolve a filter value (variable or literal)
    fn resolve_filter_value(&self, binding: &SparqlBinding, value_expr: &str) -> String {
        let value_expr = value_expr.trim();

        if value_expr.starts_with('?') {
            // Variable reference
            let var_name = &value_expr[1..];
            binding
                .get(var_name)
                .map(|v| v.value.clone())
                .unwrap_or_default()
        } else if value_expr.starts_with('"') && value_expr.ends_with('"') {
            // String literal
            value_expr[1..value_expr.len() - 1].to_string()
        } else {
            // Literal value
            value_expr.to_string()
        }
    }

    /// Execute schema stitching for GraphQL
    async fn execute_schema_stitch(
        &self,
        step: &ExecutionStep,
        completed_steps: &HashMap<String, StepResult>,
    ) -> Result<QueryResultData> {
        debug!("Executing schema stitch step: {}", step.step_id);

        // Combine GraphQL results from multiple services
        let mut combined_data = serde_json::Map::new();

        for dep_id in &step.dependencies {
            if let Some(dep_result) = completed_steps.get(dep_id) {
                if let Some(QueryResultData::GraphQL(gql_result)) = &dep_result.data {
                    if let Some(data_obj) = gql_result.data.as_object() {
                        for (key, value) in data_obj {
                            combined_data.insert(key.clone(), value.clone());
                        }
                    }
                }
            }
        }

        let stitched_result = GraphQLResponse {
            data: serde_json::Value::Object(combined_data),
            errors: Vec::new(),
            extensions: None,
        };

        Ok(QueryResultData::GraphQL(stitched_result))
    }

    /// Execute aggregation
    async fn execute_aggregate(
        &self,
        step: &ExecutionStep,
        completed_steps: &HashMap<String, StepResult>,
    ) -> Result<QueryResultData> {
        debug!("Executing aggregate step: {}", step.step_id);

        // Get the input data from dependencies
        let mut input_data = None;
        for dep_id in &step.dependencies {
            if let Some(dep_result) = completed_steps.get(dep_id) {
                if let Some(data) = &dep_result.data {
                    input_data = Some(data.clone());
                    break;
                }
            }
        }

        let input_data =
            input_data.ok_or_else(|| anyhow!("No input data for aggregate operation"))?;

        match input_data {
            QueryResultData::Sparql(sparql_results) => {
                let aggregated_results =
                    self.aggregate_sparql_results(&sparql_results, &step.query_fragment)?;
                Ok(QueryResultData::Sparql(aggregated_results))
            }
            QueryResultData::GraphQL(_) => {
                // GraphQL aggregation is typically handled by the underlying GraphQL engine
                warn!("GraphQL aggregation not implemented, passing through data");
                Ok(input_data)
            }
        }
    }

    /// Aggregate SPARQL results based on aggregate expression
    fn aggregate_sparql_results(
        &self,
        results: &SparqlResults,
        aggregate_expr: &str,
    ) -> Result<SparqlResults> {
        let aggregate_expr = aggregate_expr.trim();

        // Parse the aggregate expression to identify the operation
        if aggregate_expr.contains("GROUP BY") {
            self.perform_group_by_aggregation(results, aggregate_expr)
        } else if aggregate_expr.contains("COUNT") {
            self.perform_count_aggregation(results, aggregate_expr)
        } else if aggregate_expr.contains("SUM") {
            self.perform_sum_aggregation(results, aggregate_expr)
        } else if aggregate_expr.contains("AVG") {
            self.perform_avg_aggregation(results, aggregate_expr)
        } else if aggregate_expr.contains("MIN") {
            self.perform_min_aggregation(results, aggregate_expr)
        } else if aggregate_expr.contains("MAX") {
            self.perform_max_aggregation(results, aggregate_expr)
        } else {
            warn!("Unknown aggregation type: {}", aggregate_expr);
            Ok(results.clone())
        }
    }

    /// Perform GROUP BY aggregation
    fn perform_group_by_aggregation(
        &self,
        results: &SparqlResults,
        expr: &str,
    ) -> Result<SparqlResults> {
        // Extract GROUP BY variables
        let group_vars = self.extract_group_by_variables(expr);

        // Group bindings by the GROUP BY variables
        let mut groups: HashMap<String, Vec<SparqlBinding>> = HashMap::new();

        for binding in &results.results.bindings {
            let group_key = self.create_group_key(binding, &group_vars);
            groups.entry(group_key).or_default().push(binding.clone());
        }

        // Create aggregated results
        let mut aggregated_bindings = Vec::new();

        for (group_key, group_bindings) in groups {
            if let Some(first_binding) = group_bindings.first() {
                // Start with the group variables from the first binding
                let mut agg_binding = HashMap::new();
                for var in &group_vars {
                    if let Some(value) = first_binding.get(var) {
                        agg_binding.insert(var.clone(), value.clone());
                    }
                }

                // Apply aggregation functions within the group
                if expr.contains("COUNT(") {
                    let count_value = SparqlValue {
                        value_type: "literal".to_string(),
                        value: group_bindings.len().to_string(),
                        datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                        lang: None,
                    };
                    agg_binding.insert("count".to_string(), count_value);
                }

                // Add more aggregation functions as needed
                aggregated_bindings.push(agg_binding);
            }
        }

        // Create new variable list including group variables and aggregate variables
        let mut new_vars = group_vars;
        if expr.contains("COUNT(") {
            new_vars.push("count".to_string());
        }

        Ok(SparqlResults {
            head: SparqlHead { vars: new_vars },
            results: SparqlResultSet {
                bindings: aggregated_bindings,
            },
        })
    }

    /// Perform COUNT aggregation
    fn perform_count_aggregation(
        &self,
        results: &SparqlResults,
        expr: &str,
    ) -> Result<SparqlResults> {
        let count = if expr.contains("COUNT(DISTINCT") {
            // Count distinct values
            let var_name = self.extract_count_variable(expr);
            let mut distinct_values = HashSet::new();

            for binding in &results.results.bindings {
                if let Some(var_name) = &var_name {
                    if let Some(value) = binding.get(var_name) {
                        distinct_values.insert(value.value.clone());
                    }
                }
            }
            distinct_values.len()
        } else {
            // Count all rows
            results.results.bindings.len()
        };

        let count_binding = {
            let mut binding = HashMap::new();
            binding.insert(
                "count".to_string(),
                SparqlValue {
                    value_type: "literal".to_string(),
                    value: count.to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                    lang: None,
                },
            );
            binding
        };

        Ok(SparqlResults {
            head: SparqlHead {
                vars: vec!["count".to_string()],
            },
            results: SparqlResultSet {
                bindings: vec![count_binding],
            },
        })
    }

    /// Perform SUM aggregation
    fn perform_sum_aggregation(
        &self,
        results: &SparqlResults,
        expr: &str,
    ) -> Result<SparqlResults> {
        let var_name = self.extract_aggregate_variable(expr, "SUM");
        let mut sum: f64 = 0.0;
        let mut count = 0;

        if let Some(var_name) = var_name {
            for binding in &results.results.bindings {
                if let Some(value) = binding.get(&var_name) {
                    if let Ok(num) = value.value.parse::<f64>() {
                        sum += num;
                        count += 1;
                    }
                }
            }
        }

        let sum_binding = {
            let mut binding = HashMap::new();
            binding.insert(
                "sum".to_string(),
                SparqlValue {
                    value_type: "literal".to_string(),
                    value: sum.to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
                    lang: None,
                },
            );
            binding
        };

        Ok(SparqlResults {
            head: SparqlHead {
                vars: vec!["sum".to_string()],
            },
            results: SparqlResultSet {
                bindings: vec![sum_binding],
            },
        })
    }

    /// Perform AVG aggregation
    fn perform_avg_aggregation(
        &self,
        results: &SparqlResults,
        expr: &str,
    ) -> Result<SparqlResults> {
        let var_name = self.extract_aggregate_variable(expr, "AVG");
        let mut sum: f64 = 0.0;
        let mut count = 0;

        if let Some(var_name) = var_name {
            for binding in &results.results.bindings {
                if let Some(value) = binding.get(&var_name) {
                    if let Ok(num) = value.value.parse::<f64>() {
                        sum += num;
                        count += 1;
                    }
                }
            }
        }

        let avg = if count > 0 { sum / count as f64 } else { 0.0 };

        let avg_binding = {
            let mut binding = HashMap::new();
            binding.insert(
                "avg".to_string(),
                SparqlValue {
                    value_type: "literal".to_string(),
                    value: avg.to_string(),
                    datatype: Some("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
                    lang: None,
                },
            );
            binding
        };

        Ok(SparqlResults {
            head: SparqlHead {
                vars: vec!["avg".to_string()],
            },
            results: SparqlResultSet {
                bindings: vec![avg_binding],
            },
        })
    }

    /// Perform MIN aggregation
    fn perform_min_aggregation(
        &self,
        results: &SparqlResults,
        expr: &str,
    ) -> Result<SparqlResults> {
        let var_name = self.extract_aggregate_variable(expr, "MIN");
        let mut min_value: Option<String> = None;

        if let Some(var_name) = var_name {
            for binding in &results.results.bindings {
                if let Some(value) = binding.get(&var_name) {
                    match &min_value {
                        None => min_value = Some(value.value.clone()),
                        Some(current_min) => {
                            if value.value < *current_min {
                                min_value = Some(value.value.clone());
                            }
                        }
                    }
                }
            }
        }

        let min_binding = {
            let mut binding = HashMap::new();
            binding.insert(
                "min".to_string(),
                SparqlValue {
                    value_type: "literal".to_string(),
                    value: min_value.unwrap_or_default(),
                    datatype: None,
                    lang: None,
                },
            );
            binding
        };

        Ok(SparqlResults {
            head: SparqlHead {
                vars: vec!["min".to_string()],
            },
            results: SparqlResultSet {
                bindings: vec![min_binding],
            },
        })
    }

    /// Perform MAX aggregation
    fn perform_max_aggregation(
        &self,
        results: &SparqlResults,
        expr: &str,
    ) -> Result<SparqlResults> {
        let var_name = self.extract_aggregate_variable(expr, "MAX");
        let mut max_value: Option<String> = None;

        if let Some(var_name) = var_name {
            for binding in &results.results.bindings {
                if let Some(value) = binding.get(&var_name) {
                    match &max_value {
                        None => max_value = Some(value.value.clone()),
                        Some(current_max) => {
                            if value.value > *current_max {
                                max_value = Some(value.value.clone());
                            }
                        }
                    }
                }
            }
        }

        let max_binding = {
            let mut binding = HashMap::new();
            binding.insert(
                "max".to_string(),
                SparqlValue {
                    value_type: "literal".to_string(),
                    value: max_value.unwrap_or_default(),
                    datatype: None,
                    lang: None,
                },
            );
            binding
        };

        Ok(SparqlResults {
            head: SparqlHead {
                vars: vec!["max".to_string()],
            },
            results: SparqlResultSet {
                bindings: vec![max_binding],
            },
        })
    }

    /// Extract GROUP BY variables from expression
    fn extract_group_by_variables(&self, expr: &str) -> Vec<String> {
        if let Some(group_start) = expr.find("GROUP BY") {
            let group_part = &expr[group_start + 8..];
            group_part
                .split_whitespace()
                .filter(|s| s.starts_with('?'))
                .map(|s| s.trim_start_matches('?').to_string())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Extract variable from COUNT expression
    fn extract_count_variable(&self, expr: &str) -> Option<String> {
        if let Some(start) = expr.find("COUNT(") {
            let substr = &expr[start + 6..];
            if let Some(end) = substr.find(')') {
                let var_part = &substr[..end];
                if var_part.contains("DISTINCT") {
                    return var_part
                        .split_whitespace()
                        .find(|s| s.starts_with('?'))
                        .map(|s| s.trim_start_matches('?').to_string());
                } else if var_part.starts_with('?') {
                    return Some(var_part.trim_start_matches('?').to_string());
                }
            }
        }
        None
    }

    /// Extract variable from aggregate expression (SUM, AVG, MIN, MAX)
    fn extract_aggregate_variable(&self, expr: &str, agg_type: &str) -> Option<String> {
        let pattern = format!("{}(", agg_type);
        if let Some(start) = expr.find(&pattern) {
            let substr = &expr[start + pattern.len()..];
            if let Some(end) = substr.find(')') {
                let var_part = &substr[..end].trim();
                if var_part.starts_with('?') {
                    return Some(var_part.trim_start_matches('?').to_string());
                }
            }
        }
        None
    }

    /// Create a group key for GROUP BY operations
    fn create_group_key(&self, binding: &SparqlBinding, group_vars: &[String]) -> String {
        let mut key_parts = Vec::new();
        for var in group_vars {
            if let Some(value) = binding.get(var) {
                key_parts.push(format!("{}:{}", var, value.value));
            } else {
                key_parts.push(format!("{}:NULL", var));
            }
        }
        key_parts.join("|")
    }

    /// Execute sorting
    async fn execute_sort(
        &self,
        step: &ExecutionStep,
        completed_steps: &HashMap<String, StepResult>,
    ) -> Result<QueryResultData> {
        debug!("Executing sort step: {}", step.step_id);

        // Get the input data from dependencies
        let mut input_data = None;
        for dep_id in &step.dependencies {
            if let Some(dep_result) = completed_steps.get(dep_id) {
                if let Some(data) = &dep_result.data {
                    input_data = Some(data.clone());
                    break;
                }
            }
        }

        let input_data = input_data.ok_or_else(|| anyhow!("No input data for sort operation"))?;

        match input_data {
            QueryResultData::Sparql(sparql_results) => {
                let sorted_results =
                    self.sort_sparql_results(&sparql_results, &step.query_fragment)?;
                Ok(QueryResultData::Sparql(sorted_results))
            }
            QueryResultData::GraphQL(graphql_response) => {
                // GraphQL sorting is typically handled at the field level
                warn!("GraphQL sort execution not fully implemented, passing through data");
                Ok(QueryResultData::GraphQL(graphql_response))
            }
        }
    }

    /// Sort SPARQL results based on ORDER BY expression
    fn sort_sparql_results(
        &self,
        results: &SparqlResults,
        order_expr: &str,
    ) -> Result<SparqlResults> {
        let order_clauses = self.parse_order_by_expression(order_expr);

        let mut sorted_bindings = results.results.bindings.clone();

        // Sort the bindings based on the ORDER BY clauses
        sorted_bindings.sort_by(|a, b| {
            for order_clause in &order_clauses {
                let comparison = self.compare_bindings(a, b, &order_clause.variable);

                let result = if order_clause.descending {
                    comparison.reverse()
                } else {
                    comparison
                };

                if result != std::cmp::Ordering::Equal {
                    return result;
                }
            }
            std::cmp::Ordering::Equal
        });

        Ok(SparqlResults {
            head: results.head.clone(),
            results: SparqlResultSet {
                bindings: sorted_bindings,
            },
        })
    }

    /// Parse ORDER BY expression into order clauses
    fn parse_order_by_expression(&self, expr: &str) -> Vec<OrderClause> {
        let mut clauses = Vec::new();

        if let Some(order_start) = expr.find("ORDER BY") {
            let order_part = &expr[order_start + 8..];

            // Split by comma and parse each order expression
            for order_expr in order_part.split(',') {
                let order_expr = order_expr.trim();

                let (variable, descending) = if order_expr.starts_with("DESC(") {
                    // DESC(?variable)
                    if let Some(start) = order_expr.find('(') {
                        if let Some(end) = order_expr.find(')') {
                            let var_part = &order_expr[start + 1..end];
                            (var_part.trim_start_matches('?').to_string(), true)
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                } else if order_expr.starts_with("ASC(") {
                    // ASC(?variable)
                    if let Some(start) = order_expr.find('(') {
                        if let Some(end) = order_expr.find(')') {
                            let var_part = &order_expr[start + 1..end];
                            (var_part.trim_start_matches('?').to_string(), false)
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                } else if order_expr.starts_with('?') {
                    // Simple variable (defaults to ASC)
                    (order_expr.trim_start_matches('?').to_string(), false)
                } else {
                    // Try to extract variable from complex expressions
                    if let Some(var_match) =
                        order_expr.split_whitespace().find(|s| s.starts_with('?'))
                    {
                        let descending = order_expr.to_lowercase().contains("desc");
                        (var_match.trim_start_matches('?').to_string(), descending)
                    } else {
                        continue;
                    }
                };

                clauses.push(OrderClause {
                    variable,
                    descending,
                });
            }
        }

        clauses
    }

    /// Compare two bindings for a specific variable
    fn compare_bindings(
        &self,
        a: &SparqlBinding,
        b: &SparqlBinding,
        variable: &str,
    ) -> std::cmp::Ordering {
        let a_value = a.get(variable);
        let b_value = b.get(variable);

        match (a_value, b_value) {
            (Some(a_val), Some(b_val)) => self.compare_sparql_values(a_val, b_val),
            (Some(_), None) => std::cmp::Ordering::Greater, // Non-null values come after null
            (None, Some(_)) => std::cmp::Ordering::Less,
            (None, None) => std::cmp::Ordering::Equal,
        }
    }

    /// Compare two SPARQL values with type-aware comparison
    fn compare_sparql_values(&self, a: &SparqlValue, b: &SparqlValue) -> std::cmp::Ordering {
        // Compare by type first (URIs < literals < blank nodes)
        let type_order = |value_type: &str| match value_type {
            "uri" => 0,
            "literal" => 1,
            "bnode" => 2,
            _ => 3,
        };

        let a_type_order = type_order(&a.value_type);
        let b_type_order = type_order(&b.value_type);

        match a_type_order.cmp(&b_type_order) {
            std::cmp::Ordering::Equal => {
                // Same type, compare values
                match a.value_type.as_str() {
                    "literal" => self.compare_literal_values(a, b),
                    _ => a.value.cmp(&b.value), // String comparison for URIs and blank nodes
                }
            }
            other => other,
        }
    }

    /// Compare literal values with datatype-aware comparison
    fn compare_literal_values(&self, a: &SparqlValue, b: &SparqlValue) -> std::cmp::Ordering {
        // If both have the same datatype, try type-specific comparison
        if let (Some(a_dt), Some(b_dt)) = (&a.datatype, &b.datatype) {
            if a_dt == b_dt {
                return self.compare_typed_literals(a, b, a_dt);
            }
        }

        // Fall back to string comparison
        a.value.cmp(&b.value)
    }

    /// Compare typed literal values
    fn compare_typed_literals(
        &self,
        a: &SparqlValue,
        b: &SparqlValue,
        datatype: &str,
    ) -> std::cmp::Ordering {
        match datatype {
            "http://www.w3.org/2001/XMLSchema#integer"
            | "http://www.w3.org/2001/XMLSchema#int"
            | "http://www.w3.org/2001/XMLSchema#long" => {
                match (a.value.parse::<i64>(), b.value.parse::<i64>()) {
                    (Ok(a_num), Ok(b_num)) => a_num.cmp(&b_num),
                    _ => a.value.cmp(&b.value),
                }
            }
            "http://www.w3.org/2001/XMLSchema#decimal"
            | "http://www.w3.org/2001/XMLSchema#double"
            | "http://www.w3.org/2001/XMLSchema#float" => {
                match (a.value.parse::<f64>(), b.value.parse::<f64>()) {
                    (Ok(a_num), Ok(b_num)) => a_num
                        .partial_cmp(&b_num)
                        .unwrap_or(std::cmp::Ordering::Equal),
                    _ => a.value.cmp(&b.value),
                }
            }
            "http://www.w3.org/2001/XMLSchema#dateTime"
            | "http://www.w3.org/2001/XMLSchema#date" => {
                // For dates, ISO format string comparison usually works
                a.value.cmp(&b.value)
            }
            "http://www.w3.org/2001/XMLSchema#boolean" => {
                match (a.value.parse::<bool>(), b.value.parse::<bool>()) {
                    (Ok(a_bool), Ok(b_bool)) => a_bool.cmp(&b_bool),
                    _ => a.value.cmp(&b.value),
                }
            }
            _ => a.value.cmp(&b.value), // Default to string comparison
        }
    }

    /// Join two SPARQL result sets
    fn join_sparql_results(
        &self,
        left: &SparqlResults,
        right: &SparqlResults,
    ) -> Result<SparqlResults> {
        // Find common variables
        let common_vars: Vec<_> = left
            .head
            .vars
            .iter()
            .filter(|var| right.head.vars.contains(var))
            .cloned()
            .collect();

        if common_vars.is_empty() {
            // Cartesian product if no common variables
            return self.cartesian_product_sparql(left, right);
        }

        // Perform hash join
        let mut joined_bindings = Vec::new();

        // Build hash table for right side
        let mut right_index: HashMap<String, Vec<&SparqlBinding>> = HashMap::new();
        for binding in &right.results.bindings {
            let key = self.create_join_key(binding, &common_vars);
            right_index.entry(key).or_default().push(binding);
        }

        // Probe with left side
        for left_binding in &left.results.bindings {
            let key = self.create_join_key(left_binding, &common_vars);
            if let Some(right_bindings) = right_index.get(&key) {
                for right_binding in right_bindings {
                    let mut merged = left_binding.clone();
                    for (var, value) in right_binding.iter() {
                        if !merged.contains_key(var) {
                            merged.insert(var.clone(), value.clone());
                        }
                    }
                    joined_bindings.push(merged);
                }
            }
        }

        // Combine variable lists
        let mut all_vars = left.head.vars.clone();
        for var in &right.head.vars {
            if !all_vars.contains(var) {
                all_vars.push(var.clone());
            }
        }

        Ok(SparqlResults {
            head: SparqlHead { vars: all_vars },
            results: SparqlResultSet {
                bindings: joined_bindings,
            },
        })
    }

    /// Create a join key from bindings and common variables
    fn create_join_key(&self, binding: &SparqlBinding, common_vars: &[String]) -> String {
        let mut key_parts = Vec::new();
        for var in common_vars {
            if let Some(value) = binding.get(var) {
                key_parts.push(format!(
                    "{}:{}",
                    var,
                    serde_json::to_string(value).unwrap_or_default()
                ));
            }
        }
        key_parts.join("|")
    }

    /// Cartesian product of two SPARQL result sets
    fn cartesian_product_sparql(
        &self,
        left: &SparqlResults,
        right: &SparqlResults,
    ) -> Result<SparqlResults> {
        let mut product_bindings = Vec::new();

        for left_binding in &left.results.bindings {
            for right_binding in &right.results.bindings {
                let mut merged = left_binding.clone();
                merged.extend(right_binding.clone());
                product_bindings.push(merged);
            }
        }

        let mut all_vars = left.head.vars.clone();
        all_vars.extend(right.head.vars.clone());

        Ok(SparqlResults {
            head: SparqlHead { vars: all_vars },
            results: SparqlResultSet {
                bindings: product_bindings,
            },
        })
    }

    /// Join two GraphQL results
    fn join_graphql_results(
        &self,
        left: &GraphQLResponse,
        right: &GraphQLResponse,
    ) -> Result<GraphQLResponse> {
        // Simple merge of GraphQL objects
        let mut merged_data = serde_json::Map::new();

        if let Some(left_obj) = left.data.as_object() {
            merged_data.extend(left_obj.clone());
        }

        if let Some(right_obj) = right.data.as_object() {
            merged_data.extend(right_obj.clone());
        }

        let mut all_errors = left.errors.clone();
        all_errors.extend(right.errors.clone());

        Ok(GraphQLResponse {
            data: serde_json::Value::Object(merged_data),
            errors: all_errors,
            extensions: None,
        })
    }
}

impl Default for FederatedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the federated executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedExecutorConfig {
    pub request_timeout: Duration,
    pub parallel_timeout: Duration,
    pub max_concurrent_requests: usize,
    pub retry_attempts: usize,
    pub retry_delay: Duration,
    pub user_agent: String,
    pub enable_compression: bool,
}

impl Default for FederatedExecutorConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            parallel_timeout: Duration::from_secs(60),
            max_concurrent_requests: 10,
            retry_attempts: 3,
            retry_delay: Duration::from_millis(500),
            user_agent: "oxirs-federate/1.0".to_string(),
            enable_compression: true,
        }
    }
}

/// Result of executing a single step
#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_id: String,
    pub step_type: StepType,
    pub status: ExecutionStatus,
    pub data: Option<QueryResultData>,
    pub error: Option<String>,
    pub execution_time: Duration,
    pub service_id: Option<String>,
}

/// Status of step execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    Success,
    Failed,
    Timeout,
    Cancelled,
}

/// Data returned from query execution
#[derive(Debug, Clone)]
pub enum QueryResultData {
    Sparql(SparqlResults),
    GraphQL(GraphQLResponse),
}

/// SPARQL query results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlResults {
    pub head: SparqlHead,
    pub results: SparqlResultSet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlHead {
    pub vars: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlResultSet {
    pub bindings: Vec<SparqlBinding>,
}

pub type SparqlBinding = HashMap<String, SparqlValue>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlValue {
    #[serde(rename = "type")]
    pub value_type: String,
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub datatype: Option<String>,
    #[serde(rename = "xml:lang", skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
}

/// GraphQL request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<serde_json::Value>,
    #[serde(rename = "operationName", skip_serializing_if = "Option::is_none")]
    pub operation_name: Option<String>,
}

/// GraphQL response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLResponse {
    pub data: serde_json::Value,
    #[serde(default)]
    pub errors: Vec<GraphQLError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locations: Option<Vec<GraphQLLocation>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLLocation {
    pub line: u32,
    pub column: u32,
}

/// Order clause for sorting
#[derive(Debug, Clone)]
struct OrderClause {
    variable: String,
    descending: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::QueryType;
    use crate::{ExecutionPlan, ExecutionStep};

    #[tokio::test]
    async fn test_executor_creation() {
        let executor = FederatedExecutor::new();
        assert_eq!(executor.config.max_concurrent_requests, 10);
    }

    #[tokio::test]
    async fn test_step_result_creation() {
        let result = StepResult {
            step_id: "test-step".to_string(),
            step_type: StepType::ServiceQuery,
            status: ExecutionStatus::Success,
            data: None,
            error: None,
            execution_time: Duration::from_millis(100),
            service_id: Some("test-service".to_string()),
        };

        assert_eq!(result.status, ExecutionStatus::Success);
        assert!(result.data.is_none());
    }

    #[tokio::test]
    async fn test_sparql_results_join() {
        let executor = FederatedExecutor::new();

        let left = SparqlResults {
            head: SparqlHead {
                vars: vec!["s".to_string(), "p".to_string()],
            },
            results: SparqlResultSet { bindings: vec![] },
        };

        let right = SparqlResults {
            head: SparqlHead {
                vars: vec!["p".to_string(), "o".to_string()],
            },
            results: SparqlResultSet { bindings: vec![] },
        };

        let result = executor.join_sparql_results(&left, &right);
        assert!(result.is_ok());

        let joined = result.unwrap();
        assert_eq!(joined.head.vars.len(), 3); // s, p, o
    }

    #[test]
    fn test_join_key_creation() {
        let executor = FederatedExecutor::new();
        let mut binding = HashMap::new();
        binding.insert(
            "x".to_string(),
            SparqlValue {
                value_type: "uri".to_string(),
                value: "http://example.org".to_string(),
                datatype: None,
                lang: None,
            },
        );

        let common_vars = vec!["x".to_string()];
        let key = executor.create_join_key(&binding, &common_vars);
        assert!(key.contains("x:"));
    }
}
