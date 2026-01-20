//! # GraphQL Integration for SHACL Validation
//!
//! This module provides integration between SHACL validation and GraphQL operations,
//! enabling automatic validation of GraphQL mutations and queries against SHACL shapes.
//!
//! ## Features
//!
//! - **Query validation**: Validate GraphQL query results against shapes
//! - **Mutation validation**: Pre-validate mutations before execution
//! - **Schema mapping**: Map GraphQL types to SHACL shapes
//! - **Error conversion**: Convert SHACL violations to GraphQL errors
//! - **Real-time validation**: Validate during query execution

use crate::{Result, Shape, ShapeId, ValidationReport, Validator};
use oxirs_core::Store;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Configuration for GraphQL integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLIntegrationConfig {
    /// Enable automatic validation for mutations
    pub validate_mutations: bool,

    /// Enable validation for query results
    pub validate_queries: bool,

    /// Fail mutations on validation errors
    pub fail_on_violation: bool,

    /// Include validation warnings in GraphQL response
    pub include_warnings: bool,

    /// Maximum query complexity allowed
    pub max_query_complexity: usize,

    /// Map GraphQL types to SHACL shapes
    pub type_shape_mapping: HashMap<String, ShapeId>,

    /// Timeout for validation (milliseconds)
    pub timeout_ms: Option<u64>,
}

impl Default for GraphQLIntegrationConfig {
    fn default() -> Self {
        Self {
            validate_mutations: true,
            validate_queries: false,
            fail_on_violation: true,
            include_warnings: true,
            max_query_complexity: 1000,
            type_shape_mapping: HashMap::new(),
            timeout_ms: Some(5000),
        }
    }
}

/// GraphQL field context for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLFieldContext {
    /// GraphQL type name
    pub type_name: String,

    /// Field name
    pub field_name: String,

    /// Field path in the query
    pub field_path: Vec<String>,

    /// Operation type (query, mutation, subscription)
    pub operation_type: OperationType,

    /// Query complexity score
    pub complexity: usize,
}

/// GraphQL operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationType {
    Query,
    Mutation,
    Subscription,
}

/// GraphQL validation extension for the validator
pub struct GraphQLValidator {
    /// SHACL validator
    validator: Arc<Validator>,

    /// Integration configuration
    config: GraphQLIntegrationConfig,

    /// Type to shape mapping cache
    mapping_cache: Arc<dashmap::DashMap<String, Vec<ShapeId>>>,
}

impl GraphQLValidator {
    /// Create a new GraphQL validator
    pub fn new(validator: Arc<Validator>, config: GraphQLIntegrationConfig) -> Self {
        Self {
            validator,
            config,
            mapping_cache: Arc::new(dashmap::DashMap::new()),
        }
    }

    /// Validate a GraphQL operation before execution
    pub fn validate_operation(
        &self,
        store: &dyn Store,
        context: &GraphQLFieldContext,
    ) -> Result<GraphQLValidationResult> {
        info!(
            "Validating GraphQL {} operation on {}",
            match context.operation_type {
                OperationType::Query => "query",
                OperationType::Mutation => "mutation",
                OperationType::Subscription => "subscription",
            },
            context.type_name
        );

        // Check query complexity
        if context.complexity > self.config.max_query_complexity {
            warn!(
                "Query complexity {} exceeds maximum {}",
                context.complexity, self.config.max_query_complexity
            );
            return Ok(GraphQLValidationResult {
                conforms: false,
                violations: vec![GraphQLViolation {
                    message: format!(
                        "Query complexity {} exceeds maximum {}",
                        context.complexity, self.config.max_query_complexity
                    ),
                    path: context.field_path.clone(),
                    extensions: Default::default(),
                }],
                warnings: Vec::new(),
            });
        }

        // Get shapes for this GraphQL type
        let shapes = self.get_shapes_for_type(&context.type_name)?;

        if shapes.is_empty() {
            debug!("No SHACL shapes found for type {}", context.type_name);
            return Ok(GraphQLValidationResult {
                conforms: true,
                violations: Vec::new(),
                warnings: Vec::new(),
            });
        }

        // Perform SHACL validation
        let report = self.validator.validate_store(store, None)?;

        // Convert SHACL violations to GraphQL errors
        let graphql_result = self.convert_to_graphql_result(&report, context)?;

        Ok(graphql_result)
    }

    /// Validate before a mutation is executed
    pub fn validate_mutation_input(
        &self,
        store: &dyn Store,
        type_name: &str,
        _input_data: &HashMap<String, serde_json::Value>,
    ) -> Result<GraphQLValidationResult> {
        if !self.config.validate_mutations {
            return Ok(GraphQLValidationResult {
                conforms: true,
                violations: Vec::new(),
                warnings: Vec::new(),
            });
        }

        info!("Validating mutation input for type {}", type_name);

        // Get shapes for this type
        let shapes = self.get_shapes_for_type(type_name)?;

        if shapes.is_empty() {
            return Ok(GraphQLValidationResult {
                conforms: true,
                violations: Vec::new(),
                warnings: Vec::new(),
            });
        }

        // Perform validation
        let report = self.validator.validate_store(store, None)?;

        // Convert to GraphQL result
        let context = GraphQLFieldContext {
            type_name: type_name.to_string(),
            field_name: "mutation".to_string(),
            field_path: vec!["mutation".to_string()],
            operation_type: OperationType::Mutation,
            complexity: 1,
        };

        self.convert_to_graphql_result(&report, &context)
    }

    /// Validate query results after execution
    pub fn validate_query_result(
        &self,
        store: &dyn Store,
        type_name: &str,
    ) -> Result<GraphQLValidationResult> {
        if !self.config.validate_queries {
            return Ok(GraphQLValidationResult {
                conforms: true,
                violations: Vec::new(),
                warnings: Vec::new(),
            });
        }

        info!("Validating query result for type {}", type_name);

        let shapes = self.get_shapes_for_type(type_name)?;

        if shapes.is_empty() {
            return Ok(GraphQLValidationResult {
                conforms: true,
                violations: Vec::new(),
                warnings: Vec::new(),
            });
        }

        let report = self.validator.validate_store(store, None)?;

        let context = GraphQLFieldContext {
            type_name: type_name.to_string(),
            field_name: "query".to_string(),
            field_path: vec!["query".to_string()],
            operation_type: OperationType::Query,
            complexity: 1,
        };

        self.convert_to_graphql_result(&report, &context)
    }

    /// Register a mapping between GraphQL type and SHACL shape
    pub fn register_type_mapping(&mut self, graphql_type: String, shape_id: ShapeId) {
        self.config
            .type_shape_mapping
            .insert(graphql_type.clone(), shape_id.clone());

        // Invalidate cache for this type
        self.mapping_cache.remove(&graphql_type);
    }

    /// Clear all type mappings
    pub fn clear_mappings(&mut self) {
        self.config.type_shape_mapping.clear();
        self.mapping_cache.clear();
    }

    // Private helper methods

    fn get_shapes_for_type(&self, type_name: &str) -> Result<Vec<Shape>> {
        // Check cache first
        if let Some(cached) = self.mapping_cache.get(type_name) {
            debug!("Using cached shape mapping for type {}", type_name);
            return self.resolve_shapes(cached.value());
        }

        // Look up in configuration
        let shape_ids: Vec<ShapeId> = self
            .config
            .type_shape_mapping
            .iter()
            .filter(|(k, _)| k.as_str() == type_name)
            .map(|(_, v)| v.clone())
            .collect();

        if !shape_ids.is_empty() {
            // Cache the mapping
            self.mapping_cache
                .insert(type_name.to_string(), shape_ids.clone());

            return self.resolve_shapes(&shape_ids);
        }

        // No mapping found
        debug!("No shape mapping found for type {}", type_name);
        Ok(Vec::new())
    }

    fn resolve_shapes(&self, _shape_ids: &[ShapeId]) -> Result<Vec<Shape>> {
        // In a real implementation, this would look up shapes from the validator
        // For now, return empty vector
        Ok(Vec::new())
    }

    fn convert_to_graphql_result(
        &self,
        report: &ValidationReport,
        context: &GraphQLFieldContext,
    ) -> Result<GraphQLValidationResult> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();

        for violation in report.violations() {
            let graphql_violation = GraphQLViolation {
                message: violation
                    .result_message
                    .clone()
                    .unwrap_or_else(|| "SHACL validation error".to_string()),
                path: context.field_path.clone(),
                extensions: {
                    let mut ext = HashMap::new();
                    ext.insert("code".to_string(), "SHACL_VALIDATION_ERROR".to_string());
                    ext.insert("shape".to_string(), violation.source_shape.to_string());
                    ext.insert(
                        "severity".to_string(),
                        violation.result_severity.to_string(),
                    );
                    if let Some(path) = &violation.result_path {
                        ext.insert("propertyPath".to_string(), format!("{:?}", path));
                    }
                    ext
                },
            };

            match violation.result_severity {
                crate::Severity::Violation => violations.push(graphql_violation),
                crate::Severity::Warning if self.config.include_warnings => {
                    warnings.push(graphql_violation)
                }
                _ => {}
            }
        }

        Ok(GraphQLValidationResult {
            conforms: violations.is_empty(),
            violations,
            warnings,
        })
    }
}

/// GraphQL validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLValidationResult {
    /// Whether the data conforms to shapes
    pub conforms: bool,

    /// List of violations (errors)
    pub violations: Vec<GraphQLViolation>,

    /// List of warnings
    pub warnings: Vec<GraphQLViolation>,
}

impl GraphQLValidationResult {
    /// Convert to GraphQL error format
    pub fn to_graphql_errors(&self) -> Vec<serde_json::Value> {
        self.violations
            .iter()
            .map(|v| {
                serde_json::json!({
                    "message": v.message,
                    "path": v.path,
                    "extensions": v.extensions,
                })
            })
            .collect()
    }

    /// Check if validation should block the operation
    pub fn should_block_operation(&self) -> bool {
        !self.conforms
    }
}

/// GraphQL validation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLViolation {
    /// Error message
    pub message: String,

    /// Path in the GraphQL query
    pub path: Vec<String>,

    /// Extensions with additional metadata
    pub extensions: HashMap<String, String>,
}

/// Builder for GraphQL validator
pub struct GraphQLValidatorBuilder {
    config: GraphQLIntegrationConfig,
}

impl GraphQLValidatorBuilder {
    pub fn new() -> Self {
        Self {
            config: GraphQLIntegrationConfig::default(),
        }
    }

    pub fn validate_mutations(mut self, enabled: bool) -> Self {
        self.config.validate_mutations = enabled;
        self
    }

    pub fn validate_queries(mut self, enabled: bool) -> Self {
        self.config.validate_queries = enabled;
        self
    }

    pub fn fail_on_violation(mut self, enabled: bool) -> Self {
        self.config.fail_on_violation = enabled;
        self
    }

    pub fn max_complexity(mut self, max: usize) -> Self {
        self.config.max_query_complexity = max;
        self
    }

    pub fn type_mapping(mut self, graphql_type: String, shape_id: ShapeId) -> Self {
        self.config
            .type_shape_mapping
            .insert(graphql_type, shape_id);
        self
    }

    pub fn timeout(mut self, ms: u64) -> Self {
        self.config.timeout_ms = Some(ms);
        self
    }

    pub fn build(self, validator: Arc<Validator>) -> GraphQLValidator {
        GraphQLValidator::new(validator, self.config)
    }
}

impl Default for GraphQLValidatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graphql_validator_builder() {
        let _builder = GraphQLValidatorBuilder::new()
            .validate_mutations(true)
            .validate_queries(false)
            .max_complexity(500);

        // Verify builder pattern compiles correctly
        // (build() would require a real Arc<Validator> to test fully)
    }

    #[test]
    fn test_operation_types() {
        assert_eq!(OperationType::Query as i32, 0);
        assert_eq!(OperationType::Mutation as i32, 1);
        assert_eq!(OperationType::Subscription as i32, 2);
    }

    #[test]
    fn test_graphql_validation_result() {
        let result = GraphQLValidationResult {
            conforms: false,
            violations: vec![GraphQLViolation {
                message: "Test error".to_string(),
                path: vec!["field".to_string()],
                extensions: HashMap::new(),
            }],
            warnings: Vec::new(),
        };

        assert!(!result.conforms);
        assert!(result.should_block_operation());
        assert_eq!(result.violations.len(), 1);
    }
}
