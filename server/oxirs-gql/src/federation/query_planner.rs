//! Query planning for federated GraphQL queries

use anyhow::{Context, Result};
use futures_util::future;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use super::config::FederationConfig;
use super::schema_stitcher::SchemaStitcher;
use crate::ast::{Document, Field, OperationDefinition, Selection, SelectionSet};
use crate::types::Schema;

/// Query planner for federated queries
pub struct QueryPlanner {
    #[allow(dead_code)]
    schema_stitcher: Arc<SchemaStitcher>,
    config: FederationConfig,
}

/// Execution plan for a federated query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    /// List of execution steps
    pub steps: Vec<QueryStep>,
    /// Estimated execution cost
    pub estimated_cost: f64,
    /// Whether parallel execution is possible
    pub can_execute_parallel: bool,
}

/// Individual execution step in a query plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStep {
    /// Target endpoint for this step
    pub endpoint_id: String,
    /// GraphQL query fragment for this step
    pub query_fragment: String,
    /// Dependencies on other steps
    pub dependencies: Vec<usize>,
    /// Expected result variables
    pub result_variables: Vec<String>,
}

impl QueryPlanner {
    pub fn new(schema_stitcher: Arc<SchemaStitcher>, config: FederationConfig) -> Self {
        Self {
            schema_stitcher,
            config,
        }
    }

    /// Plan execution for a federated query
    pub async fn plan_query(&self, query: &Document, merged_schema: &Schema) -> Result<QueryPlan> {
        let mut steps = Vec::new();

        // Analyze the query to identify which services need to be involved
        for definition in &query.definitions {
            if let crate::ast::Definition::Operation(op) = definition {
                let service_steps = self.analyze_operation(op, merged_schema).await?;
                steps.extend(service_steps);
            }
        }

        // Optimize step ordering
        self.optimize_step_order(&mut steps);

        // Calculate estimated cost
        let estimated_cost = self.calculate_execution_cost(&steps);

        // Determine if parallel execution is possible
        let can_execute_parallel = self.can_parallelize(&steps);

        Ok(QueryPlan {
            steps,
            estimated_cost,
            can_execute_parallel,
        })
    }

    /// Analyze an operation to determine required services
    async fn analyze_operation(
        &self,
        operation: &OperationDefinition,
        schema: &Schema,
    ) -> Result<Vec<QueryStep>> {
        let mut steps = Vec::new();

        match operation.operation_type {
            crate::ast::OperationType::Query => {
                steps.extend(
                    self.analyze_selection_set(&operation.selection_set, schema)
                        .await?,
                );
            }
            crate::ast::OperationType::Mutation => {
                steps.extend(
                    self.analyze_selection_set(&operation.selection_set, schema)
                        .await?,
                );
            }
            crate::ast::OperationType::Subscription => {
                steps.extend(
                    self.analyze_selection_set(&operation.selection_set, schema)
                        .await?,
                );
            }
        }

        Ok(steps)
    }

    /// Analyze a selection set to determine required services
    async fn analyze_selection_set(
        &self,
        selection_set: &SelectionSet,
        schema: &Schema,
    ) -> Result<Vec<QueryStep>> {
        let mut steps = Vec::new();
        let mut service_queries: HashMap<String, Vec<String>> = HashMap::new();

        for selection in &selection_set.selections {
            match selection {
                Selection::Field(field) => {
                    let service_id = self.determine_field_service(field, schema).await?;
                    service_queries
                        .entry(service_id.clone())
                        .or_default()
                        .push(field.name.clone());

                    // Recursively analyze nested selections
                    if let Some(nested_selection_set) = &field.selection_set {
                        let nested_steps =
                            Box::pin(self.analyze_selection_set(nested_selection_set, schema))
                                .await?;
                        steps.extend(nested_steps);
                    }
                }
                Selection::InlineFragment(fragment) => {
                    let nested_steps =
                        Box::pin(self.analyze_selection_set(&fragment.selection_set, schema))
                            .await?;
                    steps.extend(nested_steps);
                }
                Selection::FragmentSpread(_) => {
                    // Fragment spreads would need additional processing
                }
            }
        }

        // Create query steps for each service
        for (service_id, fields) in service_queries {
            let query_fragment = self.build_query_fragment(&fields);
            steps.push(QueryStep {
                endpoint_id: service_id,
                query_fragment,
                dependencies: Vec::new(),
                result_variables: fields,
            });
        }

        Ok(steps)
    }

    /// Determine which service owns a particular field
    async fn determine_field_service(&self, field: &Field, _schema: &Schema) -> Result<String> {
        // This is a simplified implementation
        // In practice, you'd need to analyze the field's type and ownership

        // For now, look for service prefix in field type
        for endpoint in &self.config.endpoints {
            let namespace = endpoint.namespace.as_deref().unwrap_or(&endpoint.id);
            if field.name.starts_with(&format!("{namespace}_")) {
                return Ok(endpoint.id.clone());
            }
        }

        // Default to local service
        Ok("local".to_string())
    }

    /// Build a GraphQL query fragment for specific fields
    fn build_query_fragment(&self, fields: &[String]) -> String {
        let field_list = fields.join(" ");
        format!("{{ {field_list} }}")
    }

    /// Optimize the ordering of execution steps
    fn optimize_step_order(&self, steps: &mut Vec<QueryStep>) {
        // Topological sort based on dependencies
        let mut sorted_steps = Vec::new();
        let mut remaining_steps: VecDeque<_> = steps.iter().enumerate().collect();
        let mut resolved_indices = HashSet::new();

        while !remaining_steps.is_empty() {
            let mut made_progress = false;

            for i in 0..remaining_steps.len() {
                let (idx, step) = remaining_steps[i];

                // Check if all dependencies are resolved
                let dependencies_resolved = step
                    .dependencies
                    .iter()
                    .all(|dep_idx| resolved_indices.contains(dep_idx));

                if dependencies_resolved {
                    sorted_steps.push(step.clone());
                    resolved_indices.insert(idx);
                    remaining_steps.remove(i);
                    made_progress = true;
                    break;
                }
            }

            if !made_progress {
                // Circular dependency or other issue
                tracing::warn!("Unable to resolve all step dependencies, using original order");
                break;
            }
        }

        if sorted_steps.len() == steps.len() {
            *steps = sorted_steps;
        }
    }

    /// Calculate estimated execution cost
    fn calculate_execution_cost(&self, steps: &[QueryStep]) -> f64 {
        let mut total_cost = 0.0;

        for step in steps {
            // Base cost per step
            total_cost += 1.0;

            // Add cost based on query complexity
            total_cost += step.result_variables.len() as f64 * 0.1;

            // Add cost for dependencies (serialization overhead)
            total_cost += step.dependencies.len() as f64 * 0.5;
        }

        total_cost
    }

    /// Determine if steps can be executed in parallel
    fn can_parallelize(&self, steps: &[QueryStep]) -> bool {
        // Check if any step has dependencies
        for step in steps {
            if !step.dependencies.is_empty() {
                return false;
            }
        }

        true
    }

    /// Execute a query plan
    pub async fn execute_plan(&self, plan: &QueryPlan) -> Result<serde_json::Value> {
        let mut results: HashMap<usize, serde_json::Value> = HashMap::new();

        if plan.can_execute_parallel {
            // Execute all steps in parallel
            let futures: Vec<_> = plan
                .steps
                .iter()
                .enumerate()
                .map(|(idx, step)| {
                    let step = step.clone();
                    async move {
                        let result = self.execute_step(&step).await?;
                        Ok::<(usize, serde_json::Value), anyhow::Error>((idx, result))
                    }
                })
                .collect();

            let results_vec = future::try_join_all(futures).await?;
            results.extend(results_vec);
        } else {
            // Execute steps sequentially based on dependencies
            for (idx, step) in plan.steps.iter().enumerate() {
                let result = self.execute_step(step).await?;
                results.insert(idx, result);
            }
        }

        // Merge results
        self.merge_execution_results(&results)
    }

    /// Execute a single query step
    async fn execute_step(&self, step: &QueryStep) -> Result<serde_json::Value> {
        // Find the endpoint
        let endpoint = self
            .config
            .endpoints
            .iter()
            .find(|ep| ep.id == step.endpoint_id)
            .ok_or_else(|| anyhow::anyhow!("Endpoint not found: {}", step.endpoint_id))?;

        // Create HTTP client
        let client = reqwest::Client::new();

        // Build request
        let mut request = client.post(&endpoint.url).json(&serde_json::json!({
            "query": step.query_fragment,
            "variables": {}
        }));

        // Add authentication if provided
        if let Some(auth) = &endpoint.auth_header {
            request = request.header("Authorization", auth);
        }

        // Execute request
        let response = request
            .timeout(std::time::Duration::from_secs(endpoint.timeout_secs))
            .send()
            .await
            .context("Failed to execute federated query step")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Query step failed with status: {}",
                response.status()
            ));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse query step response")?;

        // Check for GraphQL errors
        if let Some(errors) = result.get("errors") {
            return Err(anyhow::anyhow!(
                "GraphQL errors in federated step: {}",
                serde_json::to_string_pretty(errors)?
            ));
        }

        Ok(result
            .get("data")
            .unwrap_or(&serde_json::Value::Null)
            .clone())
    }

    /// Merge execution results from multiple steps
    fn merge_execution_results(
        &self,
        results: &HashMap<usize, serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let mut merged = serde_json::Map::new();

        for result in results.values() {
            if let serde_json::Value::Object(obj) = result {
                for (key, value) in obj {
                    merged.insert(key.clone(), value.clone());
                }
            }
        }

        Ok(serde_json::Value::Object(merged))
    }
}
