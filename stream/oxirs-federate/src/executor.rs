//! Federated Query Execution Engine
//!
//! This module handles the execution of federated queries across multiple services,
//! including parallel execution, timeout handling, and fault tolerance.

use anyhow::{anyhow, Result};
use futures::{stream, StreamExt, TryStreamExt};
use reqwest::{Client, header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, ACCEPT}};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, error, info, warn, instrument};

use crate::{ExecutionPlan, ExecutionStep, StepType, FederationError, FederatedService, ServiceRegistry};

/// Federated query executor
#[derive(Debug)]
pub struct FederatedExecutor {
    client: Client,
    config: FederatedExecutorConfig,
}

impl FederatedExecutor {
    /// Create a new federated executor with default configuration
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("oxirs-federate/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            config: FederatedExecutorConfig::default(),
        }
    }

    /// Create a new federated executor with custom configuration
    pub fn with_config(config: FederatedExecutorConfig) -> Self {
        let client = Client::builder()
            .timeout(config.request_timeout)
            .user_agent(&config.user_agent)
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
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
            let group_results = self.execute_parallel_group(parallel_group, plan, &completed_steps).await?;
            
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
            Err(_) => Err(anyhow!("Parallel execution timed out after {:?}", timeout_duration)),
        }
    }

    /// Execute a single step
    #[instrument(skip(self, step, completed_steps))]
    async fn execute_step(&self, step: &ExecutionStep, completed_steps: &HashMap<String, StepResult>) -> Result<StepResult> {
        debug!("Executing step: {} ({})", step.step_id, step.step_type);

        // Check dependencies
        for dep_id in &step.dependencies {
            if !completed_steps.contains_key(dep_id) {
                return Err(anyhow!("Dependency {} not completed for step {}", dep_id, step.step_id));
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
        let service_id = step.service_id.as_ref()
            .ok_or_else(|| anyhow!("Service ID required for service query"))?;

        // TODO: Get service details from registry
        let endpoint = format!("http://localhost:8080/sparql"); // Placeholder

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/sparql-query"));
        headers.insert(ACCEPT, HeaderValue::from_static("application/sparql-results+json"));

        let response = self.client
            .post(&endpoint)
            .headers(headers)
            .body(step.query_fragment.clone())
            .send()
            .await
            .map_err(|e| anyhow!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(anyhow!("Service returned error: {}", response.status()));
        }

        let response_text = response.text().await
            .map_err(|e| anyhow!("Failed to read response: {}", e))?;

        // Parse SPARQL results JSON
        let sparql_results: SparqlResults = serde_json::from_str(&response_text)
            .map_err(|e| anyhow!("Failed to parse SPARQL results: {}", e))?;

        Ok(QueryResultData::Sparql(sparql_results))
    }

    /// Execute a GraphQL query
    async fn execute_graphql_query(&self, step: &ExecutionStep) -> Result<QueryResultData> {
        let service_id = step.service_id.as_ref()
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

        let response = self.client
            .post(&endpoint)
            .headers(headers)
            .json(&graphql_request)
            .send()
            .await
            .map_err(|e| anyhow!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(anyhow!("Service returned error: {}", response.status()));
        }

        let graphql_response: GraphQLResponse = response.json().await
            .map_err(|e| anyhow!("Failed to parse GraphQL response: {}", e))?;

        Ok(QueryResultData::GraphQL(graphql_response))
    }

    /// Execute a join operation
    async fn execute_join(&self, step: &ExecutionStep, completed_steps: &HashMap<String, StepResult>) -> Result<QueryResultData> {
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

        // Perform join operation (simplified)
        match (&input_results[0], &input_results[1]) {
            (QueryResultData::Sparql(left), QueryResultData::Sparql(right)) => {
                let joined = self.join_sparql_results(left, right)?;
                Ok(QueryResultData::Sparql(joined))
            }
            (QueryResultData::GraphQL(left), QueryResultData::GraphQL(right)) => {
                let joined = self.join_graphql_results(left, right)?;
                Ok(QueryResultData::GraphQL(joined))
            }
            _ => Err(anyhow!("Cannot join results of different types")),
        }
    }

    /// Execute a union operation
    async fn execute_union(&self, step: &ExecutionStep, completed_steps: &HashMap<String, StepResult>) -> Result<QueryResultData> {
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
            results: SparqlResultSet { bindings: all_bindings },
        };

        Ok(QueryResultData::Sparql(union_result))
    }

    /// Execute a filter operation
    async fn execute_filter(&self, _step: &ExecutionStep, _completed_steps: &HashMap<String, StepResult>) -> Result<QueryResultData> {
        // TODO: Implement filter execution
        Err(anyhow!("Filter execution not yet implemented"))
    }

    /// Execute schema stitching for GraphQL
    async fn execute_schema_stitch(&self, step: &ExecutionStep, completed_steps: &HashMap<String, StepResult>) -> Result<QueryResultData> {
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
    async fn execute_aggregate(&self, _step: &ExecutionStep, _completed_steps: &HashMap<String, StepResult>) -> Result<QueryResultData> {
        // TODO: Implement aggregation
        Err(anyhow!("Aggregation execution not yet implemented"))
    }

    /// Execute sorting
    async fn execute_sort(&self, _step: &ExecutionStep, _completed_steps: &HashMap<String, StepResult>) -> Result<QueryResultData> {
        // TODO: Implement sorting
        Err(anyhow!("Sort execution not yet implemented"))
    }

    /// Join two SPARQL result sets
    fn join_sparql_results(&self, left: &SparqlResults, right: &SparqlResults) -> Result<SparqlResults> {
        // Find common variables
        let common_vars: Vec<_> = left.head.vars.iter()
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
            results: SparqlResultSet { bindings: joined_bindings },
        })
    }

    /// Create a join key from bindings and common variables
    fn create_join_key(&self, binding: &SparqlBinding, common_vars: &[String]) -> String {
        let mut key_parts = Vec::new();
        for var in common_vars {
            if let Some(value) = binding.get(var) {
                key_parts.push(format!("{}:{}", var, serde_json::to_string(value).unwrap_or_default()));
            }
        }
        key_parts.join("|")
    }

    /// Cartesian product of two SPARQL result sets
    fn cartesian_product_sparql(&self, left: &SparqlResults, right: &SparqlResults) -> Result<SparqlResults> {
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
            results: SparqlResultSet { bindings: product_bindings },
        })
    }

    /// Join two GraphQL results
    fn join_graphql_results(&self, left: &GraphQLResponse, right: &GraphQLResponse) -> Result<GraphQLResponse> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionPlan, ExecutionStep, QueryType};

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
            head: SparqlHead { vars: vec!["s".to_string(), "p".to_string()] },
            results: SparqlResultSet { bindings: vec![] },
        };

        let right = SparqlResults {
            head: SparqlHead { vars: vec!["p".to_string(), "o".to_string()] },
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
        binding.insert("x".to_string(), SparqlValue {
            value_type: "uri".to_string(),
            value: "http://example.org".to_string(),
            datatype: None,
            lang: None,
        });

        let common_vars = vec!["x".to_string()];
        let key = executor.create_join_key(&binding, &common_vars);
        assert!(key.contains("x:"));
    }
}