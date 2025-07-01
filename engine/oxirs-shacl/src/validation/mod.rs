//! SHACL validation engine implementation
//!
//! This module implements the core validation engine that orchestrates SHACL validation.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use oxirs_core::{
    model::{BlankNode, Literal, NamedNode, RdfTerm, Term, Triple},
    OxirsError, Store,
};

use crate::{
    constraints::*, iri_resolver::*, optimization::*, paths::*, report::*, sparql::*, targets::*,
    Constraint, ConstraintComponentId, PropertyPath, Result, Severity, ShaclError, Shape, ShapeId,
    Target, ValidationConfig, ValidationReport,
};

// Re-export submodules
#[cfg(feature = "async")]
pub mod async_engine;
pub mod batch;
pub mod cache;
pub mod constraint_validators;
pub mod engine;
pub mod error_recovery;
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
pub use engine::ValidationEngine;
pub use error_recovery::*;
pub use multi_graph::{
    MultiGraphValidationConfig, MultiGraphValidationEngine, MultiGraphValidationResult,
    GraphSelectionStrategy, CrossGraphViolation, MultiGraphStats,
};
pub use stats::*;
#[cfg(feature = "async")]
pub use streaming::{
    StreamingValidationConfig, StreamingValidationEngine, StreamingValidationResult,
    StreamEvent, StreamingStats, ValidationAlert,
};
pub use utils::*;

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
        matches!(self, ConstraintEvaluationResult::Satisfied | ConstraintEvaluationResult::SatisfiedWithNote { .. })
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        1 + self.nested_results.iter()
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

    /// Get the violation as an RDF graph representation
    pub fn to_rdf(&self) -> Vec<Triple> {
        // TODO: Implement RDF serialization
        vec![]
    }
}
