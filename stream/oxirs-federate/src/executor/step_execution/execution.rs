//! Core step execution functions for federated queries

use anyhow::{anyhow, Result};
use reqwest::{
    header::{HeaderMap, HeaderValue, ACCEPT, CONTENT_TYPE},
    Client,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, error, instrument, warn};

use crate::{
    executor::types::{QueryResultData, StepResult},
    planner::{ExecutionPlan, ExecutionStep, StepType},
    service_client::GraphQLRequest,
};

use super::super::types::*;
use super::result_processing::{
    perform_sparql_join,
    aggregate_graphql_response,
    aggregate_service_result,
    sort_service_result,
    apply_sparql_filters,
};
use super::joins::perform_graphql_join;
use super::aggregation::apply_service_result_filters;
use super::stitching::perform_intelligent_result_stitching;
use super::entity_resolution::{
    perform_sparql_entity_resolution,
    perform_graphql_entity_resolution,
    perform_service_entity_resolution,
};
use super::sorting::{sort_sparql_results, aggregate_sparql_results};

pub async fn execute_parallel_group(
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
        .map(|step| execute_step(step, completed_steps))
        .collect();

    // Wait for all steps to complete or timeout
    let timeout_duration = Duration::from_secs(60); // Default timeout
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
#[instrument(skip(step, completed_steps))]
pub async fn execute_step(
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
        StepType::ServiceQuery => execute_service_query(step).await,
        StepType::GraphQLQuery => execute_graphql_query(step).await,
        StepType::Join => execute_join(step, completed_steps).await,
        StepType::Union => execute_union(step, completed_steps).await,
        StepType::Filter => execute_filter(step, completed_steps).await,
        StepType::SchemaStitch => execute_schema_stitch(step, completed_steps).await,
        StepType::Aggregate => execute_aggregate(step, completed_steps).await,
        StepType::Sort => execute_sort(step, completed_steps).await,
        StepType::EntityResolution => execute_entity_resolution(step, completed_steps).await,
        StepType::ResultStitching => execute_result_stitching(step, completed_steps).await,
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
                memory_used: 0,
                result_size: 0,
                success: true,
                error_message: None,
                service_response_time: execution_time,
                cache_hit: false,
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
                memory_used: 0,
                result_size: 0,
                success: false,
                error_message: Some(e.to_string()),
                service_response_time: execution_time,
                cache_hit: false,
            })
        }
    }
}

/// Execute a SPARQL service query
pub async fn execute_service_query(step: &ExecutionStep) -> Result<QueryResultData> {
    let _service_id = step
        .service_id
        .as_ref()
        .ok_or_else(|| anyhow!("Service ID required for service query"))?;

    // TODO: Get service details from registry
    let endpoint = "http://localhost:8080/sparql".to_string(); // Placeholder

    let mut headers = HeaderMap::new();
    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/sparql-query"),
    );
    headers.insert(
        ACCEPT,
        HeaderValue::from_static("application/sparql-results+json"),
    );

    let client = Client::new();
    let response = client
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
pub async fn execute_graphql_query(step: &ExecutionStep) -> Result<QueryResultData> {
    let _service_id = step
        .service_id
        .as_ref()
        .ok_or_else(|| anyhow!("Service ID required for GraphQL query"))?;

    // TODO: Get service details from registry
    let endpoint = "http://localhost:8080/graphql".to_string(); // Placeholder

    let graphql_request = GraphQLRequest {
        query: step.query_fragment.clone(),
        variables: None,
        operation_name: None,
    };

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let client = Client::new();
    let response = client
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
pub async fn execute_join(
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

    // Implement advanced join logic
    match input_results.first() {
        Some(QueryResultData::Sparql(first_result)) => {
            let mut joined_result = first_result.clone();

            // Join with additional results based on common variables
            for result_data in input_results.iter().skip(1) {
                if let QueryResultData::Sparql(other_result) = result_data {
                    joined_result = perform_sparql_join(&joined_result, other_result)?;
                }
            }

            Ok(QueryResultData::Sparql(joined_result))
        }
        Some(QueryResultData::GraphQL(first_response)) => {
            let mut joined_response = first_response.clone();

            // Join GraphQL responses by merging their data fields
            for result_data in input_results.iter().skip(1) {
                if let QueryResultData::GraphQL(other_response) = result_data {
                    joined_response = perform_graphql_join(&joined_response, other_response)?;
                }
            }

            Ok(QueryResultData::GraphQL(joined_response))
        }
        _ => Err(anyhow!("No valid results to join")),
    }
}

/// Execute a union operation
pub async fn execute_union(
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
        results: SparqlResultsData {
            bindings: all_bindings,
        },
    };

    Ok(QueryResultData::Sparql(union_result))
}

/// Execute a filter operation
pub async fn execute_filter(
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
            // Implement SPARQL result filtering based on filter expressions
            let filtered_results = apply_sparql_filters(&sparql_results, &step.query_fragment)?;
            Ok(QueryResultData::Sparql(filtered_results))
        }
        QueryResultData::GraphQL(graphql_response) => {
            // For GraphQL, filters are usually applied at the field level
            // For now, we'll pass through the data as GraphQL filtering is more complex
            warn!("GraphQL filter execution not fully implemented, passing through data");
            Ok(QueryResultData::GraphQL(graphql_response))
        }
        QueryResultData::ServiceResult(service_result) => {
            // Apply filters to service results (JSON data)
            let filtered_result =
                apply_service_result_filters(&service_result, &step.query_fragment)?;
            Ok(QueryResultData::ServiceResult(filtered_result))
        }
    }
}

/// Execute schema stitching for GraphQL
pub async fn execute_schema_stitch(
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
pub async fn execute_aggregate(
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

    let input_data = input_data.ok_or_else(|| anyhow!("No input data for aggregate operation"))?;

    match input_data {
        QueryResultData::Sparql(sparql_results) => {
            let aggregated_results =
                aggregate_sparql_results(&sparql_results, &step.query_fragment)?;
            Ok(QueryResultData::Sparql(aggregated_results))
        }
        QueryResultData::GraphQL(graphql_response) => {
            // Perform GraphQL aggregation on the response data
            let aggregated_response =
                aggregate_graphql_response(&graphql_response, &step.query_fragment)?;
            Ok(QueryResultData::GraphQL(aggregated_response))
        }
        QueryResultData::ServiceResult(service_result) => {
            // Perform aggregation on service results (JSON data)
            let aggregated_result =
                aggregate_service_result(&service_result, &step.query_fragment)?;
            Ok(QueryResultData::ServiceResult(aggregated_result))
        }
    }
}

/// Execute sorting
pub async fn execute_sort(
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
            let sorted_results = sort_sparql_results(&sparql_results, &step.query_fragment)?;
            Ok(QueryResultData::Sparql(sorted_results))
        }
        QueryResultData::GraphQL(graphql_response) => {
            // GraphQL sorting is typically handled at the field level
            warn!("GraphQL sort execution not fully implemented, passing through data");
            Ok(QueryResultData::GraphQL(graphql_response))
        }
        QueryResultData::ServiceResult(service_result) => {
            // Perform sorting on service results (JSON data)
            let sorted_result = sort_service_result(&service_result, &step.query_fragment)?;
            Ok(QueryResultData::ServiceResult(sorted_result))
        }
    }
}

/// Execute entity resolution
pub async fn execute_entity_resolution(
    step: &ExecutionStep,
    completed_steps: &HashMap<String, StepResult>,
) -> Result<QueryResultData> {
    debug!("Executing entity resolution step: {}", step.step_id);

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

    let input_data = input_data.ok_or_else(|| anyhow!("No input data for entity resolution"))?;

    // Perform entity resolution based on the data type
    match input_data {
        QueryResultData::Sparql(sparql_results) => {
            let resolved_results =
                perform_sparql_entity_resolution(&sparql_results, &step.query_fragment)?;
            Ok(QueryResultData::Sparql(resolved_results))
        }
        QueryResultData::GraphQL(graphql_response) => {
            let resolved_response =
                perform_graphql_entity_resolution(&graphql_response, &step.query_fragment)?;
            Ok(QueryResultData::GraphQL(resolved_response))
        }
        QueryResultData::ServiceResult(service_result) => {
            let resolved_result =
                perform_service_entity_resolution(&service_result, &step.query_fragment)?;
            Ok(QueryResultData::ServiceResult(resolved_result))
        }
    }
}

/// Execute result stitching
pub async fn execute_result_stitching(
    step: &ExecutionStep,
    completed_steps: &HashMap<String, StepResult>,
) -> Result<QueryResultData> {
    debug!("Executing result stitching step: {}", step.step_id);

    // Collect all input data from dependencies
    let mut input_data_list = Vec::new();
    for dep_id in &step.dependencies {
        if let Some(dep_result) = completed_steps.get(dep_id) {
            if let Some(data) = &dep_result.data {
                input_data_list.push(data.clone());
            }
        }
    }

    if input_data_list.is_empty() {
        return Err(anyhow!("No input data for result stitching"));
    }

    // If only one result, return it
    if input_data_list.len() == 1 {
        return Ok(input_data_list.into_iter().next().unwrap());
    }

    // Perform intelligent result stitching based on data types
    let stitched_result =
        perform_intelligent_result_stitching(&input_data_list, &step.query_fragment)?;

    debug!("Successfully stitched {} results", input_data_list.len());
    Ok(stitched_result)
}
