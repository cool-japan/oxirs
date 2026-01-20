//! Cross-module integration capabilities for OxiRS SHACL
//!
//! This module provides enhanced integration between oxirs-shacl and other
//! modules in the OxiRS ecosystem, enabling seamless data validation across
//! different interfaces and protocols.

#![allow(dead_code)]

pub mod ai_integration;
pub mod cicd;
pub mod fuseki_integration;
pub mod graphql_integration;
pub mod performance;
pub mod rule_engine;
pub mod shex_migration;
pub mod stream_integration;

// Re-export public API
pub use ai_integration::*;
pub use cicd::{
    CiCdConfig, CiCdEngine, CiCdResult, CiSystem, EnvironmentConfig, ExitCodeConfig, OutputFormat,
    RegressionAnalysis, ThresholdConfig, ThresholdResults,
};
pub use fuseki_integration::*;
pub use graphql_integration::*;
pub use performance::{
    BottleneckType, LatencyPrediction, LoadBalancingRecommendation, PerformanceBottleneck,
    PerformanceConfig, PerformanceMetrics, PerformanceOptimizer, PerformanceSummary, Severity,
};
pub use rule_engine::{
    CacheStats, ConstraintGenerator, ConstraintPattern, ConstraintRule, ReasoningBridge,
    ReasoningResult, ReasoningStrategy, RefinementAction, RuleBasedValidator,
    RuleBasedValidatorBuilder, RuleEngineConfig, ShapeRefinementRule, ViolationPattern,
};
pub use shex_migration::{
    AnnotationHandling, ClosedHandling, ExtraHandling, ImportHandling, MigrationConfig,
    MigrationConfigBuilder, MigrationError, MigrationReport, MigrationResult, MigrationStatistics,
    MigrationWarning, NodeConstraint, NumericFacets, SemanticAction, SemanticMappingConfig,
    ShexAnnotation, ShexExpression, ShexMigrationEngine, ShexSchema, ShexShape, StringFacets,
    TripleConstraint, UnmappedFeature, ValueSetHandling, ValueSetValue,
};
pub use stream_integration::*;

use crate::{Result, ShaclError, ValidationReport, Validator};
use oxirs_core::Store;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for cross-module integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enable GraphQL integration
    pub enable_graphql_integration: bool,
    /// Enable Fuseki server integration
    pub enable_fuseki_integration: bool,
    /// Enable streaming integration
    pub enable_stream_integration: bool,
    /// Enable AI integration
    pub enable_ai_integration: bool,
    /// Custom integration endpoints
    pub custom_endpoints: HashMap<String, String>,
    /// Integration timeout in milliseconds
    pub timeout_ms: Option<u64>,
    /// Maximum concurrent validations
    pub max_concurrent_validations: usize,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_graphql_integration: true,
            enable_fuseki_integration: true,
            enable_stream_integration: false,
            enable_ai_integration: false,
            custom_endpoints: HashMap::new(),
            timeout_ms: Some(30000), // 30 seconds
            max_concurrent_validations: 10,
        }
    }
}

/// Integration manager for coordinating validation across modules
#[derive(Debug)]
pub struct IntegrationManager {
    validator: Arc<Validator>,
    config: IntegrationConfig,
    #[cfg(feature = "async")]
    runtime: Option<tokio::runtime::Handle>,
}

impl IntegrationManager {
    /// Create a new integration manager
    pub fn new(validator: Arc<Validator>, config: IntegrationConfig) -> Self {
        Self {
            validator,
            config,
            #[cfg(feature = "async")]
            runtime: tokio::runtime::Handle::try_current().ok(),
        }
    }

    /// Get the validator reference
    pub fn validator(&self) -> &Arc<Validator> {
        &self.validator
    }

    /// Get the integration configuration
    pub fn config(&self) -> &IntegrationConfig {
        &self.config
    }

    /// Update integration configuration
    pub fn update_config(&mut self, config: IntegrationConfig) {
        self.config = config;
    }

    /// Validate data for a specific integration context
    pub fn validate_for_context(
        &self,
        store: &dyn Store,
        context: IntegrationContext,
    ) -> Result<IntegratedValidationReport> {
        let validation_report = self.validator.validate_store(store, None)?;

        Ok(IntegratedValidationReport {
            base_report: validation_report,
            context,
            integration_metadata: self.create_integration_metadata(),
        })
    }

    /// Create integration metadata
    fn create_integration_metadata(&self) -> IntegrationMetadata {
        IntegrationMetadata {
            validator_version: crate::VERSION.to_string(),
            integration_config: self.config.clone(),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Context for different integration scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationContext {
    /// GraphQL query execution context
    GraphQL {
        operation_name: Option<String>,
        query_complexity: usize,
        client_info: Option<String>,
    },
    /// SPARQL endpoint context
    Fuseki {
        endpoint_path: String,
        operation_type: String,
        user_agent: Option<String>,
    },
    /// Streaming data context
    Stream {
        stream_id: String,
        event_type: String,
        batch_size: usize,
    },
    /// AI/ML context
    AI {
        model_id: String,
        confidence_threshold: f64,
        suggestion_type: String,
    },
    /// Custom integration context
    Custom {
        context_type: String,
        properties: HashMap<String, String>,
    },
}

/// Enhanced validation report with integration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedValidationReport {
    /// Base SHACL validation report
    pub base_report: ValidationReport,
    /// Integration context
    pub context: IntegrationContext,
    /// Integration-specific metadata
    pub integration_metadata: IntegrationMetadata,
}

impl IntegratedValidationReport {
    /// Check if validation passed
    pub fn conforms(&self) -> bool {
        self.base_report.conforms()
    }

    /// Get violations from the base report
    pub fn violations(&self) -> &[crate::validation::ValidationViolation] {
        self.base_report.violations()
    }

    /// Get context-specific recommendations
    pub fn get_context_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        match &self.context {
            IntegrationContext::GraphQL {
                query_complexity, ..
            } => {
                if *query_complexity > 500 {
                    recommendations
                        .push("Consider simplifying GraphQL query or using pagination".to_string());
                }
                if !self.conforms() {
                    recommendations.push(
                        "GraphQL mutations may fail due to SHACL constraint violations".to_string(),
                    );
                }
            }
            IntegrationContext::Fuseki { operation_type, .. } => {
                if operation_type == "UPDATE" && !self.conforms() {
                    recommendations.push(
                        "SPARQL UPDATE operations should be reviewed due to constraint violations"
                            .to_string(),
                    );
                }
            }
            IntegrationContext::Stream { batch_size, .. } => {
                if *batch_size > 1000 && !self.conforms() {
                    recommendations.push(
                        "Consider reducing batch size to isolate validation errors".to_string(),
                    );
                }
            }
            IntegrationContext::AI {
                confidence_threshold,
                ..
            } => {
                if *confidence_threshold < 0.8 && !self.conforms() {
                    recommendations.push(
                        "AI suggestions may be unreliable due to low confidence and validation errors".to_string()
                    );
                }
            }
            IntegrationContext::Custom { .. } => {
                if !self.conforms() {
                    recommendations.push(
                        "Custom integration should handle validation errors appropriately"
                            .to_string(),
                    );
                }
            }
        }

        recommendations
    }

    /// Export report in context-appropriate format
    pub fn export_for_context(&self) -> Result<String> {
        match &self.context {
            IntegrationContext::GraphQL { .. } => {
                // Export as GraphQL-compatible JSON
                let graphql_format = serde_json::json!({
                    "data": null,
                    "errors": self.violations().iter().map(|v| {
                        serde_json::json!({
                            "message": v.result_message.as_deref().unwrap_or("Validation error"),
                            "path": ["validation"],
                            "extensions": {
                                "code": "SHACL_VALIDATION_ERROR",
                                "shape": v.source_shape.to_string(),
                                "severity": v.result_severity.to_string()
                            }
                        })
                    }).collect::<Vec<_>>()
                });
                Ok(serde_json::to_string_pretty(&graphql_format)?)
            }
            IntegrationContext::Fuseki { .. } => {
                // Export as SPARQL result format
                Ok(format!(
                    "# SHACL Validation Report\n# Conforms: {}\n# Violations: {}",
                    self.conforms(),
                    self.violations().len()
                ))
            }
            _ => {
                // Default JSON export
                Ok(serde_json::to_string_pretty(&self)?)
            }
        }
    }
}

/// Integration-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMetadata {
    /// Version of the SHACL validator
    pub validator_version: String,
    /// Integration configuration used
    pub integration_config: IntegrationConfig,
    /// Timestamp of validation
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Error types for integration operations
#[derive(Debug, thiserror::Error)]
pub enum IntegrationError {
    #[error("GraphQL integration error: {0}")]
    GraphQL(String),

    #[error("Fuseki integration error: {0}")]
    Fuseki(String),

    #[error("Stream integration error: {0}")]
    Stream(String),

    #[error("AI integration error: {0}")]
    AI(String),

    #[error("Custom integration error: {0}")]
    Custom(String),

    #[error("SHACL validation error: {0}")]
    Validation(#[from] ShaclError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for integration operations
pub type IntegrationResult<T> = std::result::Result<T, IntegrationError>;
