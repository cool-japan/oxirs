//! SHACL validation engine implementation
//!
//! This module implements the core validation engine that orchestrates SHACL validation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use oxirs_core::model::{BlankNode, Literal, NamedNode, Object, Subject, Term, Triple};

use crate::{ConstraintComponentId, PropertyPath, Severity, ShapeId, SHACL_NS};

// Re-export submodules
#[cfg(feature = "async")]
pub mod async_engine;
pub mod batch;
pub mod cache;
pub mod constraint_validators;
#[cfg(feature = "async")]
pub mod distributed;
pub mod engine;
pub mod error_recovery;
pub mod ml_integration;
pub mod multi_graph;
pub mod stats;
#[cfg(feature = "async")]
pub mod streaming;
pub mod utils;

#[cfg(test)]
pub mod tests;

// Re-export main types
#[cfg(feature = "async")]
pub use async_engine::{
    AsyncValidationConfig, AsyncValidationEngine, AsyncValidationEngineBuilder,
    AsyncValidationResult, AsyncValidationStats, ValidationEvent,
};
pub use batch::*;
pub use cache::*;
pub use constraint_validators::*;
#[cfg(feature = "async")]
pub use distributed::{
    ConsistencyLevel, DistributedError, DistributedStats, DistributedValidationConfig,
    DistributedValidator, DistributedValidatorBuilder, LoadBalancingStrategy, PartitionResult,
    ValidationPartition, WorkerInfo,
};
pub use engine::ValidationEngine;
pub use error_recovery::*;
pub use ml_integration::*;
pub use multi_graph::{
    CrossGraphViolation, GraphSelectionStrategy, MultiGraphStats, MultiGraphValidationConfig,
    MultiGraphValidationEngine, MultiGraphValidationResult,
};
pub use stats::*;
#[cfg(feature = "async")]
pub use streaming::{
    StreamEvent, StreamingStats, StreamingValidationConfig, StreamingValidationEngine,
    StreamingValidationResult, ValidationAlert,
};
pub use utils::*;

// Re-export core validation types - defined in this module

/// Cache key for constraint results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstraintCacheKey {
    pub focus_node: Term,
    pub shape_id: ShapeId,
    pub constraint_component_id: ConstraintComponentId,
    pub property_path: Option<PropertyPath>,
}

/// Result of evaluating a constraint
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintEvaluationResult {
    /// Constraint is satisfied
    Satisfied,
    /// Constraint is satisfied but with a note (e.g., due to error recovery)
    SatisfiedWithNote { note: String },
    /// Constraint is violated with optional violating value and message
    Violated {
        violating_value: Option<Term>,
        message: Option<String>,
    },
}

impl ConstraintEvaluationResult {
    /// Create a satisfied result
    pub fn satisfied() -> Self {
        ConstraintEvaluationResult::Satisfied
    }

    /// Create a satisfied result with a note
    pub fn satisfied_with_note(note: String) -> Self {
        ConstraintEvaluationResult::SatisfiedWithNote { note }
    }

    /// Create a violated result
    pub fn violated(violating_value: Option<Term>, message: Option<String>) -> Self {
        ConstraintEvaluationResult::Violated {
            violating_value,
            message,
        }
    }

    /// Check if the result is satisfied (including satisfied with note)
    pub fn is_satisfied(&self) -> bool {
        matches!(
            self,
            ConstraintEvaluationResult::Satisfied
                | ConstraintEvaluationResult::SatisfiedWithNote { .. }
        )
    }

    /// Check if the result is violated
    pub fn is_violated(&self) -> bool {
        matches!(self, ConstraintEvaluationResult::Violated { .. })
    }

    /// Get the violating value if any
    pub fn violating_value(&self) -> Option<&Term> {
        match self {
            ConstraintEvaluationResult::Satisfied => None,
            ConstraintEvaluationResult::SatisfiedWithNote { .. } => None,
            ConstraintEvaluationResult::Violated {
                violating_value, ..
            } => violating_value.as_ref(),
        }
    }

    /// Get the violation message if any
    pub fn message(&self) -> Option<&str> {
        match self {
            ConstraintEvaluationResult::Satisfied => None,
            ConstraintEvaluationResult::SatisfiedWithNote { .. } => None,
            ConstraintEvaluationResult::Violated { message, .. } => message.as_deref(),
        }
    }

    /// Get the note for satisfied with note results
    pub fn note(&self) -> Option<&str> {
        match self {
            ConstraintEvaluationResult::SatisfiedWithNote { note } => Some(note),
            _ => None,
        }
    }
}

/// Validation violation details
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationViolation {
    /// The focus node that failed validation
    pub focus_node: Term,
    /// The path to the value that failed validation (if applicable)
    pub result_path: Option<PropertyPath>,
    /// The specific value that failed validation (if applicable)
    pub value: Option<Term>,
    /// The shape that was being validated
    pub source_shape: ShapeId,
    /// The constraint component that was violated
    pub source_constraint_component: ConstraintComponentId,
    /// The severity of the violation
    pub result_severity: Severity,
    /// Human-readable error message
    pub result_message: Option<String>,
    /// Additional violation details
    pub details: HashMap<String, String>,
    /// Nested validation results for complex constraints
    pub nested_results: Vec<ValidationViolation>,
}

impl ValidationViolation {
    /// Create a new validation violation
    pub fn new(
        focus_node: Term,
        source_shape: ShapeId,
        source_constraint_component: ConstraintComponentId,
        result_severity: Severity,
    ) -> Self {
        Self {
            focus_node,
            result_path: None,
            value: None,
            source_shape,
            source_constraint_component,
            result_severity,
            result_message: None,
            details: HashMap::new(),
            nested_results: Vec::new(),
        }
    }

    /// Set the result path
    pub fn with_path(mut self, path: PropertyPath) -> Self {
        self.result_path = Some(path);
        self
    }

    /// Set the violating value
    pub fn with_value(mut self, value: Term) -> Self {
        self.value = Some(value);
        self
    }

    /// Set the result message
    pub fn with_message(mut self, message: String) -> Self {
        self.result_message = Some(message);
        self
    }

    /// Add a detail to the violation
    pub fn with_detail(mut self, key: String, value: String) -> Self {
        self.details.insert(key, value);
        self
    }

    /// Add a nested validation result
    pub fn with_nested_result(mut self, nested: ValidationViolation) -> Self {
        self.nested_results.push(nested);
        self
    }

    /// Add multiple nested validation results
    pub fn with_nested_results(mut self, nested: Vec<ValidationViolation>) -> Self {
        self.nested_results.extend(nested);
        self
    }

    /// Check if this violation has nested results
    pub fn has_nested_results(&self) -> bool {
        !self.nested_results.is_empty()
    }

    /// Get the total number of violations (including nested)
    pub fn total_violation_count(&self) -> usize {
        1 + self
            .nested_results
            .iter()
            .map(|v| v.total_violation_count())
            .sum::<usize>()
    }

    /// Get all violations flattened (including nested)
    pub fn flatten_violations(&self) -> Vec<&ValidationViolation> {
        let mut violations = vec![self];
        for nested in &self.nested_results {
            violations.extend(nested.flatten_violations());
        }
        violations
    }

    /// Get the message (convenience method for result_message)
    pub fn message(&self) -> &Option<String> {
        &self.result_message
    }

    /// Get the violation as an RDF graph representation
    ///
    /// Serializes this validation violation as RDF triples according to the SHACL specification.
    /// Each violation is represented as a sh:ValidationResult with properties describing
    /// the violation details.
    pub fn to_rdf(&self) -> Vec<Triple> {
        self.to_rdf_with_subject(None)
    }

    /// Internal method to serialize violation with a specific subject or generate a blank node
    fn to_rdf_with_subject(&self, subject: Option<Subject>) -> Vec<Triple> {
        let mut triples = Vec::new();

        // Create or use provided subject for the validation result
        let result_subject = subject.unwrap_or_else(|| {
            Subject::BlankNode(BlankNode::new_unchecked(format!(
                "result_{}",
                uuid::Uuid::new_v4()
            )))
        });

        // Add rdf:type sh:ValidationResult
        let rdf_type = NamedNode::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        let validation_result = NamedNode::new_unchecked(format!("{}ValidationResult", SHACL_NS));
        triples.push(Triple::new(
            result_subject.clone(),
            rdf_type,
            validation_result,
        ));

        // Add sh:focusNode
        let focus_node_pred = NamedNode::new_unchecked(format!("{}focusNode", SHACL_NS));
        triples.push(Triple::new(
            result_subject.clone(),
            focus_node_pred,
            self.focus_node.clone(),
        ));

        // Add sh:resultPath if present
        if let Some(ref path) = self.result_path {
            let result_path_pred = NamedNode::new_unchecked(format!("{}resultPath", SHACL_NS));
            // For simple single-predicate paths, serialize as the predicate IRI
            // For complex paths, we would need to serialize the path structure
            match path {
                PropertyPath::Predicate(node) => {
                    triples.push(Triple::new(
                        result_subject.clone(),
                        result_path_pred,
                        node.clone(),
                    ));
                }
                // Complex paths would require more sophisticated serialization
                // For now, we skip complex paths in RDF serialization
                _ => {
                    tracing::debug!("Skipping complex property path in RDF serialization");
                }
            }
        }

        // Add sh:value if present
        if let Some(ref value) = self.value {
            let value_pred = NamedNode::new_unchecked(format!("{}value", SHACL_NS));
            triples.push(Triple::new(
                result_subject.clone(),
                value_pred,
                value.clone(),
            ));
        }

        // Add sh:sourceShape
        let source_shape_pred = NamedNode::new_unchecked(format!("{}sourceShape", SHACL_NS));
        if let Ok(shape_node) = NamedNode::new(self.source_shape.as_str()) {
            triples.push(Triple::new(
                result_subject.clone(),
                source_shape_pred,
                shape_node,
            ));
        }

        // Add sh:sourceConstraintComponent
        let source_component_pred =
            NamedNode::new_unchecked(format!("{}sourceConstraintComponent", SHACL_NS));
        if let Ok(component_node) = NamedNode::new(self.source_constraint_component.as_str()) {
            triples.push(Triple::new(
                result_subject.clone(),
                source_component_pred,
                component_node,
            ));
        }

        // Add sh:resultSeverity
        let severity_pred = NamedNode::new_unchecked(format!("{}resultSeverity", SHACL_NS));
        let severity_iri = match self.result_severity {
            Severity::Violation => format!("{}Violation", SHACL_NS),
            Severity::Warning => format!("{}Warning", SHACL_NS),
            Severity::Info => format!("{}Info", SHACL_NS),
        };
        triples.push(Triple::new(
            result_subject.clone(),
            severity_pred,
            NamedNode::new_unchecked(severity_iri),
        ));

        // Add sh:resultMessage if present
        if let Some(ref message) = self.result_message {
            let message_pred = NamedNode::new_unchecked(format!("{}resultMessage", SHACL_NS));
            triples.push(Triple::new(
                result_subject.clone(),
                message_pred,
                Literal::new(message),
            ));
        }

        // Add details as sh:detail properties
        let detail_pred = NamedNode::new_unchecked(format!("{}detail", SHACL_NS));
        for (key, value) in &self.details {
            let detail_message = format!("{}: {}", key, value);
            triples.push(Triple::new(
                result_subject.clone(),
                detail_pred.clone(),
                Literal::new(detail_message),
            ));
        }

        // Add nested results as sh:detail with separate ValidationResult resources
        for nested in &self.nested_results {
            let nested_result_subject = Subject::BlankNode(BlankNode::new_unchecked(format!(
                "nested_result_{}",
                uuid::Uuid::new_v4()
            )));
            triples.push(Triple::new(
                result_subject.clone(),
                detail_pred.clone(),
                Object::from(nested_result_subject.clone()),
            ));
            triples.extend(nested.to_rdf_with_subject(Some(nested_result_subject)));
        }

        triples
    }
}

// Custom Hash implementation for ValidationViolation
// We exclude the HashMap field from hashing to make this hashable
impl std::hash::Hash for ValidationViolation {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.focus_node.hash(state);
        self.result_path.hash(state);
        self.value.hash(state);
        self.source_shape.hash(state);
        self.source_constraint_component.hash(state);
        self.result_severity.hash(state);
        self.result_message.hash(state);
        // Skip details HashMap as it doesn't implement Hash
        self.nested_results.hash(state);
    }
}
