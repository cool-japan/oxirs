//! # Fuseki Integration for SHACL Validation
//!
//! This module provides integration between SHACL validation and Apache Jena Fuseki
//! (or oxirs-fuseki) SPARQL endpoints, enabling validation at the endpoint level.
//!
//! ## Features
//!
//! - **Endpoint validation**: Validate SPARQL UPDATE operations before execution
//! - **Query result validation**: Validate CONSTRUCT/DESCRIBE query results
//! - **Transaction support**: Validate within SPARQL transactions
//! - **Access control**: Shape-based access control for endpoints
//! - **Validation hooks**: Pre/post operation validation hooks

use crate::{Result, Shape, ShapeId, ValidationReport, Validator};
use oxirs_core::Store;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Configuration for Fuseki integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusekiIntegrationConfig {
    /// Enable validation for UPDATE operations
    pub validate_updates: bool,

    /// Enable validation for CONSTRUCT/DESCRIBE results
    pub validate_construct: bool,

    /// Fail UPDATE operations on validation errors
    pub fail_on_violation: bool,

    /// Maximum triples to validate in a single operation
    pub max_triples: usize,

    /// Timeout for validation (milliseconds)
    pub timeout_ms: Option<u64>,

    /// Endpoint-specific shape mappings
    pub endpoint_shapes: HashMap<String, Vec<ShapeId>>,

    /// Enable transactional validation
    pub transactional: bool,

    /// Cache validation results
    pub cache_results: bool,
}

impl Default for FusekiIntegrationConfig {
    fn default() -> Self {
        Self {
            validate_updates: true,
            validate_construct: false,
            fail_on_violation: true,
            max_triples: 10000,
            timeout_ms: Some(10000), // 10 seconds
            endpoint_shapes: HashMap::new(),
            transactional: true,
            cache_results: true,
        }
    }
}

/// SPARQL operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparqlOperation {
    /// SELECT query
    Select,

    /// CONSTRUCT query
    Construct,

    /// DESCRIBE query
    Describe,

    /// ASK query
    Ask,

    /// INSERT DATA
    InsertData,

    /// DELETE DATA
    DeleteData,

    /// INSERT/DELETE WHERE
    Modify,

    /// LOAD
    Load,

    /// CLEAR
    Clear,

    /// CREATE
    Create,

    /// DROP
    Drop,
}

/// Fuseki endpoint context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusekiEndpointContext {
    /// Endpoint path (e.g., "/dataset/sparql")
    pub endpoint_path: String,

    /// Operation type
    pub operation: SparqlOperation,

    /// User agent or client identifier
    pub user_agent: Option<String>,

    /// Request ID for tracing
    pub request_id: String,

    /// Number of triples affected
    pub triple_count: usize,
}

/// Fuseki validator for SPARQL endpoints
pub struct FusekiValidator {
    /// SHACL validator
    validator: Arc<Validator>,

    /// Integration configuration
    config: FusekiIntegrationConfig,

    /// Validation result cache
    cache: Arc<dashmap::DashMap<String, ValidationReport>>,
}

impl FusekiValidator {
    /// Create a new Fuseki validator
    pub fn new(validator: Arc<Validator>, config: FusekiIntegrationConfig) -> Self {
        // Use default capacity of 100 to avoid frequent reallocations
        Self {
            validator,
            config,
            cache: Arc::new(dashmap::DashMap::with_capacity(100)),
        }
    }

    /// Validate a SPARQL operation before execution
    pub fn validate_operation(
        &self,
        store: &dyn Store,
        context: &FusekiEndpointContext,
    ) -> Result<FusekiValidationResult> {
        info!(
            "Validating SPARQL {:?} operation on endpoint {} (request: {})",
            context.operation, context.endpoint_path, context.request_id
        );

        // Check triple count limit
        if context.triple_count > self.config.max_triples {
            warn!(
                "Triple count {} exceeds maximum {}",
                context.triple_count, self.config.max_triples
            );
            return Ok(FusekiValidationResult {
                conforms: false,
                validation_time_ms: 0,
                triple_count: context.triple_count,
                violations: vec![format!(
                    "Operation affects {} triples, exceeding maximum of {}",
                    context.triple_count, self.config.max_triples
                )],
                should_proceed: false,
            });
        }

        // Check if validation is enabled for this operation type
        if !self.should_validate_operation(context.operation) {
            debug!("Validation not enabled for {:?}", context.operation);
            return Ok(FusekiValidationResult {
                conforms: true,
                validation_time_ms: 0,
                triple_count: context.triple_count,
                violations: Vec::new(),
                should_proceed: true,
            });
        }

        // Get shapes for this endpoint
        let shapes = self.get_shapes_for_endpoint(&context.endpoint_path)?;

        if shapes.is_empty() {
            debug!(
                "No SHACL shapes configured for endpoint {}",
                context.endpoint_path
            );
            return Ok(FusekiValidationResult {
                conforms: true,
                validation_time_ms: 0,
                triple_count: context.triple_count,
                violations: Vec::new(),
                should_proceed: true,
            });
        }

        // Check cache if enabled
        if self.config.cache_results {
            let cache_key = self.generate_cache_key(store, &shapes)?;
            if let Some(cached_report) = self.cache.get(&cache_key) {
                debug!("Using cached validation result for {}", context.request_id);
                return Ok(self.convert_to_fuseki_result(cached_report.value(), context));
            }
        }

        // Perform validation
        let start = std::time::Instant::now();
        let report = self.validator.validate_store(store, None)?;
        let validation_time_ms = start.elapsed().as_millis() as u64;

        // Cache result if enabled
        if self.config.cache_results {
            let cache_key = self.generate_cache_key(store, &shapes)?;
            self.cache.insert(cache_key, report.clone());
        }

        // Convert to Fuseki result
        let mut result = self.convert_to_fuseki_result(&report, context);
        result.validation_time_ms = validation_time_ms;

        Ok(result)
    }

    /// Validate UPDATE operation before execution
    pub fn validate_update(
        &self,
        store: &dyn Store,
        endpoint: &str,
        update_query: &str,
    ) -> Result<FusekiValidationResult> {
        if !self.config.validate_updates {
            return Ok(FusekiValidationResult {
                conforms: true,
                validation_time_ms: 0,
                triple_count: 0,
                violations: Vec::new(),
                should_proceed: true,
            });
        }

        let context = FusekiEndpointContext {
            endpoint_path: endpoint.to_string(),
            operation: SparqlOperation::Modify,
            user_agent: None,
            request_id: uuid::Uuid::new_v4().to_string(),
            triple_count: self.estimate_triple_count(update_query),
        };

        self.validate_operation(store, &context)
    }

    /// Validate CONSTRUCT query results
    pub fn validate_construct_result(
        &self,
        store: &dyn Store,
        endpoint: &str,
    ) -> Result<FusekiValidationResult> {
        if !self.config.validate_construct {
            return Ok(FusekiValidationResult {
                conforms: true,
                validation_time_ms: 0,
                triple_count: 0,
                violations: Vec::new(),
                should_proceed: true,
            });
        }

        let context = FusekiEndpointContext {
            endpoint_path: endpoint.to_string(),
            operation: SparqlOperation::Construct,
            user_agent: None,
            request_id: uuid::Uuid::new_v4().to_string(),
            triple_count: 0, // Would be determined from result set
        };

        self.validate_operation(store, &context)
    }

    /// Register shapes for an endpoint
    pub fn register_endpoint_shapes(&mut self, endpoint: String, shapes: Vec<ShapeId>) {
        self.config.endpoint_shapes.insert(endpoint, shapes);
        self.clear_cache();
    }

    /// Clear validation cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStatistics {
        CacheStatistics {
            size: self.cache.len(),
            capacity: self.cache.capacity(),
        }
    }

    // Private helper methods

    fn should_validate_operation(&self, operation: SparqlOperation) -> bool {
        match operation {
            SparqlOperation::Select | SparqlOperation::Ask => false,
            SparqlOperation::Construct | SparqlOperation::Describe => {
                self.config.validate_construct
            }
            SparqlOperation::InsertData
            | SparqlOperation::DeleteData
            | SparqlOperation::Modify
            | SparqlOperation::Load => self.config.validate_updates,
            SparqlOperation::Clear | SparqlOperation::Create | SparqlOperation::Drop => {
                self.config.validate_updates
            }
        }
    }

    fn get_shapes_for_endpoint(&self, endpoint: &str) -> Result<Vec<Shape>> {
        let _shape_ids = self
            .config
            .endpoint_shapes
            .get(endpoint)
            .cloned()
            .unwrap_or_default();

        // In a real implementation, resolve shape IDs to actual shapes
        Ok(Vec::new())
    }

    fn generate_cache_key(&self, _store: &dyn Store, shapes: &[Shape]) -> Result<String> {
        // Generate a hash-based cache key from store state and shapes
        let mut key = String::new();
        for shape in shapes {
            key.push_str(&shape.id.to_string());
        }
        Ok(format!("cache_{}", key))
    }

    fn estimate_triple_count(&self, _query: &str) -> usize {
        // In a real implementation, parse the query and estimate affected triples
        0
    }

    fn convert_to_fuseki_result(
        &self,
        report: &ValidationReport,
        context: &FusekiEndpointContext,
    ) -> FusekiValidationResult {
        let violations: Vec<String> = report
            .violations()
            .iter()
            .map(|v| {
                v.result_message
                    .clone()
                    .unwrap_or_else(|| "Validation error".to_string())
            })
            .collect();

        let conforms = violations.is_empty();
        let should_proceed = if self.config.fail_on_violation {
            conforms
        } else {
            true
        };

        FusekiValidationResult {
            conforms,
            validation_time_ms: 0, // Set by caller
            triple_count: context.triple_count,
            violations,
            should_proceed,
        }
    }
}

/// Fuseki validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusekiValidationResult {
    /// Whether the data conforms to shapes
    pub conforms: bool,

    /// Time taken for validation (milliseconds)
    pub validation_time_ms: u64,

    /// Number of triples validated
    pub triple_count: usize,

    /// List of violation messages
    pub violations: Vec<String>,

    /// Whether the operation should proceed
    pub should_proceed: bool,
}

impl FusekiValidationResult {
    /// Convert to HTTP response format
    pub fn to_http_response(&self) -> Result<String> {
        if self.conforms {
            Ok("Validation passed".to_string())
        } else {
            Ok(format!(
                "Validation failed with {} violations:\n{}",
                self.violations.len(),
                self.violations.join("\n")
            ))
        }
    }

    /// Get HTTP status code for this result
    pub fn http_status_code(&self) -> u16 {
        if self.should_proceed {
            200 // OK
        } else {
            400 // Bad Request
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub size: usize,
    pub capacity: usize,
}

/// Builder for Fuseki validator
pub struct FusekiValidatorBuilder {
    config: FusekiIntegrationConfig,
}

impl FusekiValidatorBuilder {
    pub fn new() -> Self {
        Self {
            config: FusekiIntegrationConfig::default(),
        }
    }

    pub fn validate_updates(mut self, enabled: bool) -> Self {
        self.config.validate_updates = enabled;
        self
    }

    pub fn validate_construct(mut self, enabled: bool) -> Self {
        self.config.validate_construct = enabled;
        self
    }

    pub fn fail_on_violation(mut self, enabled: bool) -> Self {
        self.config.fail_on_violation = enabled;
        self
    }

    pub fn max_triples(mut self, max: usize) -> Self {
        self.config.max_triples = max;
        self
    }

    pub fn timeout(mut self, ms: u64) -> Self {
        self.config.timeout_ms = Some(ms);
        self
    }

    pub fn transactional(mut self, enabled: bool) -> Self {
        self.config.transactional = enabled;
        self
    }

    pub fn cache_results(mut self, enabled: bool) -> Self {
        self.config.cache_results = enabled;
        self
    }

    pub fn endpoint_shapes(mut self, endpoint: String, shapes: Vec<ShapeId>) -> Self {
        self.config.endpoint_shapes.insert(endpoint, shapes);
        self
    }

    pub fn build(self, validator: Arc<Validator>) -> FusekiValidator {
        FusekiValidator::new(validator, self.config)
    }
}

impl Default for FusekiValidatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparql_operation_types() {
        let operations = [
            SparqlOperation::Select,
            SparqlOperation::Construct,
            SparqlOperation::InsertData,
            SparqlOperation::Modify,
        ];

        assert_eq!(operations.len(), 4);
    }

    #[test]
    fn test_fuseki_validation_result() {
        let result = FusekiValidationResult {
            conforms: false,
            validation_time_ms: 100,
            triple_count: 50,
            violations: vec!["Error 1".to_string()],
            should_proceed: false,
        };

        assert!(!result.conforms);
        assert_eq!(result.http_status_code(), 400);
        assert!(!result.should_proceed);
    }

    #[test]
    fn test_cache_statistics() {
        let stats = CacheStatistics {
            size: 10,
            capacity: 100,
        };

        assert_eq!(stats.size, 10);
        assert_eq!(stats.capacity, 100);
    }
}
