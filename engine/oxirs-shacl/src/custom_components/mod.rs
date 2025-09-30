//! Custom SHACL Constraint Components
//!
//! This module provides comprehensive support for user-defined SHACL constraint components,
//! allowing users to extend SHACL with domain-specific validation logic, parameter validation,
//! component inheritance, performance optimization, and security features.

#![allow(dead_code)]

pub mod constraints;
pub mod metadata;
pub mod performance;
pub mod registry;
pub mod security;
pub mod standard;

// Re-export key types
pub use constraints::{
    ComponentExecutionResult, CompositeConstraint, CompositionType, CustomConstraint,
};
pub use metadata::{
    ComponentLibrary, ComponentMetadata, ParameterConstraint, ParameterDefinition,
    ValidationCondition, ValidationRule, ValidationRuleType,
};
pub use performance::{
    ComponentExecutionContext, ComponentPerformanceStats, ErrorStats, ErrorTrend, ExecutionMetrics,
    MemoryUsageStats,
};
pub use registry::CustomConstraintRegistry;
pub use security::{
    ResourceQuotas, SandboxingLevel, SecurityPolicy, SecurityViolation, SecurityViolationType,
    SparqlOperation,
};
pub use standard::{
    EmailValidationComponent, RangeConstraintComponent, RegexConstraintComponent,
    SparqlConstraintComponent, UrlValidationComponent,
};

use crate::{ConstraintComponentId, Result};
use oxirs_core::model::Term;
use std::collections::HashMap;

/// Trait for custom constraint components
pub trait CustomConstraintComponent: Send + Sync + std::fmt::Debug {
    /// Get the component identifier
    fn component_id(&self) -> &ConstraintComponentId;

    /// Get component metadata
    fn metadata(&self) -> &ComponentMetadata;

    /// Validate the component configuration
    fn validate_configuration(&self, parameters: &HashMap<String, Term>) -> Result<()>;

    /// Create a constraint instance from parameters
    fn create_constraint(&self, parameters: HashMap<String, Term>) -> Result<CustomConstraint>;

    /// Get the SPARQL query template for this component (if SPARQL-based)
    fn sparql_template(&self) -> Option<&str> {
        None
    }

    /// Get additional prefixes needed for SPARQL queries
    fn sparql_prefixes(&self) -> Option<&str> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::Literal;

    #[test]
    fn test_registry_registration() {
        let mut registry = CustomConstraintRegistry::new();
        let component = std::sync::Arc::new(RegexConstraintComponent::new());
        let component_id = component.component_id().clone();

        assert!(registry.register_component(component).is_ok());
        assert!(registry.get_component(&component_id).is_some());
        assert!(registry.get_metadata(&component_id).is_some());
    }

    #[test]
    fn test_regex_constraint_creation() {
        let component = RegexConstraintComponent::new();
        let mut parameters = HashMap::new();
        parameters.insert(
            "pattern".to_string(),
            Term::Literal(Literal::new(r"^test\d+$")),
        );

        let constraint = component.create_constraint(parameters).unwrap();
        assert_eq!(
            constraint.component_id.as_str(),
            "ex:RegexConstraintComponent"
        );
        assert!(constraint.parameters.contains_key("pattern"));
    }

    #[test]
    fn test_range_constraint_validation() {
        let component = RangeConstraintComponent::new();

        // Should fail without min or max
        let empty_params = HashMap::new();
        assert!(component.validate_configuration(&empty_params).is_err());

        // Should succeed with min value
        let mut params = HashMap::new();
        params.insert("minValue".to_string(), Term::Literal(Literal::new("0")));
        assert!(component.validate_configuration(&params).is_ok());
    }

    #[test]
    fn test_standard_extensions_registration() {
        let mut registry = CustomConstraintRegistry::new();
        assert!(registry.register_standard_extensions().is_ok());

        // Check that standard components are registered
        assert!(registry
            .get_component(&ConstraintComponentId(
                "ex:RegexConstraintComponent".to_string()
            ))
            .is_some());
        assert!(registry
            .get_component(&ConstraintComponentId(
                "ex:RangeConstraintComponent".to_string()
            ))
            .is_some());
        assert!(registry
            .get_component(&ConstraintComponentId(
                "ex:UrlValidationComponent".to_string()
            ))
            .is_some());
        assert!(registry
            .get_component(&ConstraintComponentId(
                "ex:EmailValidationComponent".to_string()
            ))
            .is_some());
        assert!(registry
            .get_component(&ConstraintComponentId(
                "ex:SparqlConstraintComponent".to_string()
            ))
            .is_some());
    }

    #[test]
    fn test_composite_constraint_creation() {
        let mut registry = CustomConstraintRegistry::new();
        registry.register_standard_extensions().unwrap();

        let regex_id = ConstraintComponentId("ex:RegexConstraintComponent".to_string());
        let range_id = ConstraintComponentId("ex:RangeConstraintComponent".to_string());

        let mut parameters = HashMap::new();
        parameters.insert("pattern".to_string(), Term::Literal(Literal::new(r"^\d+$")));
        parameters.insert("minValue".to_string(), Term::Literal(Literal::new("0")));

        let composite = registry
            .create_composite_constraint(&[regex_id, range_id], parameters)
            .unwrap();

        assert_eq!(composite.component_ids.len(), 2);
        assert_eq!(composite.constraints.len(), 2);
        assert!(matches!(composite.composition_type, CompositionType::And));
    }
}
