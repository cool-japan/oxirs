//! Result Integration and Processing
//!
//! This module handles the integration of results from multiple federated services,
//! including merging, deduplication, sorting, and error handling.

use anyhow::{anyhow, Result};
use oxirs_core::{BlankNode, Literal, NamedNode, Term};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::{
    ExecutionMetadata, ExecutionStatus, FederatedResult, FederationError, GraphQLResponse,
    QueryResult, QueryResultData, SparqlBinding, SparqlResults, SparqlValue, StepResult,
};

/// Result integrator for federated queries
#[derive(Debug)]
pub struct ResultIntegrator {
    config: ResultIntegratorConfig,
}

impl ResultIntegrator {
    /// Create a new result integrator with default configuration
    pub fn new() -> Self {
        Self {
            config: ResultIntegratorConfig::default(),
        }
    }

    /// Create a new result integrator with custom configuration
    pub fn with_config(config: ResultIntegratorConfig) -> Self {
        Self { config }
    }

    /// Integrate SPARQL query results from multiple services
    pub async fn integrate_sparql_results(
        &self,
        step_results: Vec<StepResult>,
    ) -> Result<FederatedResult> {
        info!("Integrating {} SPARQL result steps", step_results.len());

        let start_time = std::time::Instant::now();
        let mut all_bindings = Vec::new();
        let mut all_variables = HashSet::new();
        let mut errors = Vec::new();
        let mut services_used = HashSet::new();
        let mut successful_services = 0;
        let total_services = step_results.len();

        // Process each step result
        for step_result in &step_results {
            if let Some(service_id) = &step_result.service_id {
                services_used.insert(service_id.clone());
            }

            match step_result.status {
                ExecutionStatus::Success => {
                    successful_services += 1;
                    if let Some(QueryResultData::Sparql(sparql_result)) = &step_result.data {
                        // Collect variables
                        for var in &sparql_result.head.vars {
                            all_variables.insert(var.clone());
                        }

                        // Collect bindings
                        all_bindings.extend(sparql_result.results.bindings.clone());
                    }
                }
                ExecutionStatus::Failed => {
                    let error = FederationError::ServiceUnavailable {
                        service_id: step_result
                            .service_id
                            .clone()
                            .unwrap_or("unknown".to_string()),
                    };
                    errors.push(error);
                }
                ExecutionStatus::Timeout => {
                    let error = FederationError::ExecutionTimeout {
                        timeout: step_result.execution_time,
                    };
                    errors.push(error);
                }
                ExecutionStatus::Cancelled => {
                    let error = FederationError::ServiceUnavailable {
                        service_id: step_result
                            .service_id
                            .clone()
                            .unwrap_or("cancelled".to_string()),
                    };
                    errors.push(error);
                }
            }
        }

        // Add partial results warning if some services failed
        if successful_services < total_services && successful_services > 0 {
            errors.push(FederationError::PartialResults {
                successful_services,
                total_services,
            });
        }

        // Deduplicate if enabled
        if self.config.enable_deduplication {
            all_bindings = self.deduplicate_sparql_bindings(all_bindings);
        }

        // Sort if requested
        if let Some(ref sort_config) = self.config.sort_config {
            all_bindings = self.sort_sparql_bindings(all_bindings, sort_config)?;
        }

        // Apply limit if specified
        if let Some(limit) = self.config.result_limit {
            all_bindings.truncate(limit);
        }

        // Create integrated SPARQL results
        let sparql_results = SparqlResults {
            head: crate::executor::SparqlHead {
                vars: all_variables.into_iter().collect(),
            },
            results: crate::executor::SparqlResultsData {
                bindings: all_bindings,
            },
        };

        let execution_time = start_time.elapsed();
        let metadata = ExecutionMetadata {
            execution_time,
            services_used: services_used.len(),
            subqueries_executed: step_results.len(),
            cache_hit: false, // TODO: Implement cache hit detection
            plan_summary: format!("Integrated {} services", services_used.len()),
        };

        // Convert to HashMap format for FederatedResult
        let result_bindings: Vec<HashMap<String, oxirs_core::Term>> = sparql_results
            .results
            .bindings
            .into_iter()
            .map(|binding| self.convert_sparql_binding_to_terms(binding))
            .collect::<Result<Vec<_>>>()?;

        Ok(FederatedResult {
            data: QueryResult::Sparql(result_bindings),
            metadata,
            errors,
        })
    }

    /// Integrate GraphQL query results from multiple services
    pub async fn integrate_graphql_results(
        &self,
        step_results: Vec<StepResult>,
    ) -> Result<FederatedResult> {
        info!("Integrating {} GraphQL result steps", step_results.len());

        let start_time = std::time::Instant::now();
        let mut merged_data = serde_json::Map::new();
        let mut all_errors = Vec::new();
        let mut federation_errors = Vec::new();
        let mut services_used = HashSet::new();
        let mut successful_services = 0;
        let total_services = step_results.len();

        // Process each step result
        for step_result in &step_results {
            if let Some(service_id) = &step_result.service_id {
                services_used.insert(service_id.clone());
            }

            match step_result.status {
                ExecutionStatus::Success => {
                    successful_services += 1;
                    if let Some(QueryResultData::GraphQL(graphql_result)) = &step_result.data {
                        // Merge data objects
                        if let Some(data_obj) = graphql_result.data.as_object() {
                            for (key, value) in data_obj {
                                if merged_data.contains_key(key) {
                                    // Handle field conflicts
                                    match self.config.conflict_resolution {
                                        ConflictResolution::FirstWins => {
                                            // Keep existing value
                                        }
                                        ConflictResolution::LastWins => {
                                            merged_data.insert(key.clone(), value.clone());
                                        }
                                        ConflictResolution::Merge => {
                                            // Attempt to merge values
                                            if let Some(merged_value) = self.merge_graphql_values(
                                                merged_data.get(key).unwrap(),
                                                value,
                                            ) {
                                                merged_data.insert(key.clone(), merged_value);
                                            } else {
                                                federation_errors.push(FederationError::SchemaConflict {
                                                    conflict: format!("Cannot merge field '{}' from multiple services", key),
                                                });
                                            }
                                        }
                                        ConflictResolution::Error => {
                                            federation_errors.push(
                                                FederationError::SchemaConflict {
                                                    conflict: format!(
                                                        "Field '{}' exists in multiple services",
                                                        key
                                                    ),
                                                },
                                            );
                                        }
                                    }
                                } else {
                                    merged_data.insert(key.clone(), value.clone());
                                }
                            }
                        }

                        // Collect GraphQL errors
                        all_errors.extend(graphql_result.errors.clone());
                    }
                }
                ExecutionStatus::Failed => {
                    federation_errors.push(FederationError::ServiceUnavailable {
                        service_id: step_result
                            .service_id
                            .clone()
                            .unwrap_or("unknown".to_string()),
                    });
                }
                ExecutionStatus::Timeout => {
                    federation_errors.push(FederationError::ExecutionTimeout {
                        timeout: step_result.execution_time,
                    });
                }
                ExecutionStatus::Cancelled => {
                    federation_errors.push(FederationError::ServiceUnavailable {
                        service_id: step_result
                            .service_id
                            .clone()
                            .unwrap_or("cancelled".to_string()),
                    });
                }
            }
        }

        // Add partial results warning if some services failed
        if successful_services < total_services && successful_services > 0 {
            federation_errors.push(FederationError::PartialResults {
                successful_services,
                total_services,
            });
        }

        let execution_time = start_time.elapsed();
        let metadata = ExecutionMetadata {
            execution_time,
            services_used: services_used.len(),
            subqueries_executed: step_results.len(),
            cache_hit: false,
            plan_summary: format!("Stitched {} GraphQL services", services_used.len()),
        };

        let final_graphql_response = serde_json::Value::Object(merged_data);

        Ok(FederatedResult {
            data: QueryResult::GraphQL(final_graphql_response),
            metadata,
            errors: federation_errors,
        })
    }

    /// Deduplicate SPARQL bindings
    fn deduplicate_sparql_bindings(&self, bindings: Vec<SparqlBinding>) -> Vec<SparqlBinding> {
        let mut seen = HashSet::new();
        let mut deduplicated = Vec::new();

        for binding in bindings {
            let binding_key = self.create_binding_hash(&binding);
            if seen.insert(binding_key) {
                deduplicated.push(binding);
            }
        }

        debug!(
            "Deduplicated {} bindings to {}",
            seen.len() + deduplicated.len() - seen.len(),
            deduplicated.len()
        );
        deduplicated
    }

    /// Sort SPARQL bindings according to configuration
    fn sort_sparql_bindings(
        &self,
        mut bindings: Vec<SparqlBinding>,
        sort_config: &SortConfig,
    ) -> Result<Vec<SparqlBinding>> {
        bindings.sort_by(|a, b| {
            for sort_key in &sort_config.sort_keys {
                let a_value = a.get(&sort_key.variable);
                let b_value = b.get(&sort_key.variable);

                let ordering = match (a_value, b_value) {
                    (Some(a_val), Some(b_val)) => self.compare_sparql_values(a_val, b_val),
                    (Some(_), None) => std::cmp::Ordering::Greater,
                    (None, Some(_)) => std::cmp::Ordering::Less,
                    (None, None) => std::cmp::Ordering::Equal,
                };

                let final_ordering = if sort_key.descending {
                    ordering.reverse()
                } else {
                    ordering
                };

                if final_ordering != std::cmp::Ordering::Equal {
                    return final_ordering;
                }
            }
            std::cmp::Ordering::Equal
        });

        Ok(bindings)
    }

    /// Compare two SPARQL values for sorting
    fn compare_sparql_values(&self, a: &SparqlValue, b: &SparqlValue) -> std::cmp::Ordering {
        // First compare by type
        let type_order = self
            .get_type_order(&a.value_type)
            .cmp(&self.get_type_order(&b.value_type));
        if type_order != std::cmp::Ordering::Equal {
            return type_order;
        }

        // Then compare by value
        match a.value_type.as_str() {
            "uri" => a.value.cmp(&b.value),
            "literal" => {
                // Consider datatype for literals
                match (&a.datatype, &b.datatype) {
                    (Some(a_dt), Some(b_dt)) if a_dt == b_dt => {
                        // Same datatype, try typed comparison
                        self.compare_typed_literals(a, b)
                    }
                    _ => a.value.cmp(&b.value), // Fallback to string comparison
                }
            }
            "bnode" => a.value.cmp(&b.value),
            _ => a.value.cmp(&b.value),
        }
    }

    /// Compare typed literals
    fn compare_typed_literals(&self, a: &SparqlValue, b: &SparqlValue) -> std::cmp::Ordering {
        if let Some(datatype) = &a.datatype {
            match datatype.as_str() {
                "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#int"
                | "http://www.w3.org/2001/XMLSchema#long" => {
                    match (a.value.parse::<i64>(), b.value.parse::<i64>()) {
                        (Ok(a_int), Ok(b_int)) => a_int.cmp(&b_int),
                        _ => a.value.cmp(&b.value),
                    }
                }
                "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#double"
                | "http://www.w3.org/2001/XMLSchema#float" => {
                    match (a.value.parse::<f64>(), b.value.parse::<f64>()) {
                        (Ok(a_float), Ok(b_float)) => a_float
                            .partial_cmp(&b_float)
                            .unwrap_or(std::cmp::Ordering::Equal),
                        _ => a.value.cmp(&b.value),
                    }
                }
                "http://www.w3.org/2001/XMLSchema#dateTime" => {
                    // Simple string comparison for now - could use proper date parsing
                    a.value.cmp(&b.value)
                }
                _ => a.value.cmp(&b.value),
            }
        } else {
            a.value.cmp(&b.value)
        }
    }

    /// Get sorting order for RDF term types
    fn get_type_order(&self, term_type: &str) -> u8 {
        match term_type {
            "bnode" => 0,
            "uri" => 1,
            "literal" => 2,
            _ => 3,
        }
    }

    /// Create a hash key for a SPARQL binding for deduplication
    fn create_binding_hash(&self, binding: &SparqlBinding) -> String {
        let mut pairs: Vec<_> = binding.iter().collect();
        pairs.sort_by_key(|(var, _)| *var);

        pairs
            .into_iter()
            .map(|(var, value)| {
                format!(
                    "{}:{}",
                    var,
                    serde_json::to_string(value).unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    /// Merge two GraphQL values
    fn merge_graphql_values(
        &self,
        a: &serde_json::Value,
        b: &serde_json::Value,
    ) -> Option<serde_json::Value> {
        match (a, b) {
            (serde_json::Value::Object(a_obj), serde_json::Value::Object(b_obj)) => {
                let mut merged = a_obj.clone();
                for (key, value) in b_obj {
                    if let Some(existing) = merged.get(key) {
                        if let Some(merged_value) = self.merge_graphql_values(existing, value) {
                            merged.insert(key.clone(), merged_value);
                        } else {
                            return None; // Cannot merge
                        }
                    } else {
                        merged.insert(key.clone(), value.clone());
                    }
                }
                Some(serde_json::Value::Object(merged))
            }
            (serde_json::Value::Array(a_arr), serde_json::Value::Array(b_arr)) => {
                let mut merged = a_arr.clone();
                merged.extend(b_arr.clone());
                Some(serde_json::Value::Array(merged))
            }
            _ => {
                // Cannot merge different types or primitive values
                None
            }
        }
    }

    /// Convert SPARQL binding to oxirs-core Terms
    fn convert_sparql_binding_to_terms(
        &self,
        binding: SparqlBinding,
    ) -> Result<HashMap<String, Term>> {
        let mut term_binding = HashMap::new();

        for (var, sparql_value) in binding {
            let term = match sparql_value.value_type.as_str() {
                "uri" => {
                    let iri = NamedNode::new(&sparql_value.value)
                        .map_err(|e| anyhow!("Invalid IRI '{}': {}", sparql_value.value, e))?;
                    Term::NamedNode(iri)
                }
                "literal" => {
                    if let Some(datatype_str) = sparql_value.datatype {
                        let datatype = NamedNode::new(&datatype_str).map_err(|e| {
                            anyhow!("Invalid datatype IRI '{}': {}", datatype_str, e)
                        })?;
                        Term::Literal(Literal::new_typed(&sparql_value.value, datatype))
                    } else if let Some(lang) = sparql_value.lang {
                        Term::Literal(Literal::new_lang(&sparql_value.value, &lang)?)
                    } else {
                        Term::Literal(Literal::new(&sparql_value.value))
                    }
                }
                "bnode" => Term::BlankNode(BlankNode::new(&sparql_value.value)?),
                _ => {
                    return Err(anyhow!(
                        "Unknown SPARQL value type: {}",
                        sparql_value.value_type
                    ));
                }
            };

            term_binding.insert(var, term);
        }

        Ok(term_binding)
    }
}

impl Default for ResultIntegrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for result integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultIntegratorConfig {
    pub enable_deduplication: bool,
    pub result_limit: Option<usize>,
    pub sort_config: Option<SortConfig>,
    pub conflict_resolution: ConflictResolution,
    pub enable_partial_results: bool,
    pub timeout: Duration,
}

impl Default for ResultIntegratorConfig {
    fn default() -> Self {
        Self {
            enable_deduplication: true,
            result_limit: None,
            sort_config: None,
            conflict_resolution: ConflictResolution::Merge,
            enable_partial_results: true,
            timeout: Duration::from_secs(10),
        }
    }
}

/// Configuration for sorting results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortConfig {
    pub sort_keys: Vec<SortKey>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortKey {
    pub variable: String,
    pub descending: bool,
}

/// Conflict resolution strategies for merging results
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConflictResolution {
    FirstWins,
    LastWins,
    Merge,
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        executor::{SparqlHead, SparqlResults, SparqlResultsData},
        QueryResultData, StepResult, StepType,
    };
    use std::time::Duration;

    #[tokio::test]
    async fn test_integrator_creation() {
        let integrator = ResultIntegrator::new();
        assert!(integrator.config.enable_deduplication);
    }

    #[tokio::test]
    async fn test_sparql_integration() {
        let integrator = ResultIntegrator::new();

        let sparql_result = SparqlResults {
            head: SparqlHead {
                vars: vec!["s".to_string()],
            },
            results: SparqlResultsData { bindings: vec![] },
        };

        let step_result = StepResult {
            step_id: "test-step".to_string(),
            step_type: StepType::ServiceQuery,
            status: ExecutionStatus::Success,
            data: Some(QueryResultData::Sparql(sparql_result)),
            error: None,
            execution_time: Duration::from_millis(100),
            service_id: Some("test-service".to_string()),
            memory_used: 1024,
            result_size: 100,
            success: true,
            error_message: None,
            service_response_time: Duration::from_millis(50),
            cache_hit: false,
        };

        let result = integrator.integrate_sparql_results(vec![step_result]).await;
        assert!(result.is_ok());

        let federated_result = result.unwrap();
        assert!(federated_result.is_success());
        assert_eq!(federated_result.metadata.services_used, 1);
    }

    #[tokio::test]
    async fn test_graphql_integration() {
        let integrator = ResultIntegrator::new();

        let graphql_response = crate::executor::GraphQLResponse {
            data: serde_json::json!({"user": {"name": "Alice"}}),
            errors: vec![],
            extensions: None,
        };

        let step_result = StepResult {
            step_id: "test-step".to_string(),
            step_type: StepType::GraphQLQuery,
            status: ExecutionStatus::Success,
            data: Some(QueryResultData::GraphQL(graphql_response)),
            error: None,
            execution_time: Duration::from_millis(150),
            service_id: Some("test-service".to_string()),
            memory_used: 1024,
            result_size: 256,
            success: true,
            error_message: None,
            service_response_time: Duration::from_millis(100),
            cache_hit: false,
        };

        let result = integrator
            .integrate_graphql_results(vec![step_result])
            .await;
        assert!(result.is_ok());

        let federated_result = result.unwrap();
        assert!(federated_result.is_success());
    }

    #[test]
    fn test_binding_hash() {
        let integrator = ResultIntegrator::new();
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

        let hash = integrator.create_binding_hash(&binding);
        assert!(!hash.is_empty());
    }

    #[test]
    fn test_conflict_resolution() {
        let integrator = ResultIntegrator::new();

        let a = serde_json::json!({"field": "value1"});
        let b = serde_json::json!({"field": "value2"});

        let result = integrator.merge_graphql_values(&a, &b);
        assert!(result.is_none()); // Cannot merge conflicting primitives

        let a_obj = serde_json::json!({"field1": "value1"});
        let b_obj = serde_json::json!({"field2": "value2"});

        let merged = integrator.merge_graphql_values(&a_obj, &b_obj);
        assert!(merged.is_some());
    }
}
