//! Constraint evaluation context and results

use indexmap::IndexMap;
use std::collections::HashMap;

use oxirs_core::model::{NamedNode, Term};

use crate::{ConstraintComponentId, PropertyPath, Severity, ShapeId, ValidationViolation};

/// Constraint evaluation context
#[derive(Debug, Clone)]
pub struct ConstraintContext {
    /// Current focus node being validated
    pub focus_node: Term,

    /// Current property path (for property shapes)
    pub path: Option<PropertyPath>,

    /// Values at the current path
    pub values: Vec<Term>,

    /// Shape being validated
    pub shape_id: ShapeId,

    /// Validation depth (for recursion control)
    pub depth: usize,

    /// Custom validation context
    pub custom_context: HashMap<String, String>,

    /// Allowed properties for closed shape validation
    /// Contains all property paths defined in the shape
    pub allowed_properties: Vec<PropertyPath>,

    /// Reference to shapes registry for qualified constraint validation
    pub shapes_registry: Option<std::sync::Arc<IndexMap<ShapeId, crate::Shape>>>,
}

impl ConstraintContext {
    pub fn new(focus_node: Term, shape_id: ShapeId) -> Self {
        Self {
            focus_node,
            path: None,
            values: Vec::new(),
            shape_id,
            depth: 0,
            custom_context: HashMap::new(),
            allowed_properties: Vec::new(),
            shapes_registry: None,
        }
    }

    pub fn with_path(mut self, path: PropertyPath) -> Self {
        self.path = Some(path);
        self
    }

    pub fn with_values(mut self, values: Vec<Term>) -> Self {
        self.values = values;
        self
    }

    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_allowed_properties(mut self, properties: Vec<PropertyPath>) -> Self {
        self.allowed_properties = properties;
        self
    }

    pub fn with_shapes_registry(
        mut self,
        shapes_registry: std::sync::Arc<IndexMap<ShapeId, crate::Shape>>,
    ) -> Self {
        self.shapes_registry = Some(shapes_registry);
        self
    }
}

/// Constraint evaluation result
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintEvaluationResult {
    /// Constraint is satisfied
    Satisfied,

    /// Constraint is violated
    Violated {
        /// Specific value that caused the violation (if applicable)
        violating_value: Option<Term>,

        /// Custom violation message
        message: Option<String>,

        /// Additional details about the violation
        details: HashMap<String, String>,
    },

    /// Constraint evaluation failed due to error
    Error {
        /// Error message
        message: String,

        /// Error details
        details: HashMap<String, String>,
    },
}

impl ConstraintEvaluationResult {
    pub fn satisfied() -> Self {
        ConstraintEvaluationResult::Satisfied
    }

    pub fn violated(violating_value: Option<Term>, message: Option<String>) -> Self {
        ConstraintEvaluationResult::Violated {
            violating_value,
            message,
            details: HashMap::new(),
        }
    }

    pub fn violated_with_details(
        violating_value: Option<Term>,
        message: Option<String>,
        details: HashMap<String, String>,
    ) -> Self {
        ConstraintEvaluationResult::Violated {
            violating_value,
            message,
            details,
        }
    }

    pub fn error(message: String) -> Self {
        ConstraintEvaluationResult::Error {
            message,
            details: HashMap::new(),
        }
    }

    pub fn error_with_details(message: String, details: HashMap<String, String>) -> Self {
        ConstraintEvaluationResult::Error { message, details }
    }

    pub fn is_satisfied(&self) -> bool {
        matches!(self, ConstraintEvaluationResult::Satisfied)
    }

    pub fn is_violated(&self) -> bool {
        matches!(self, ConstraintEvaluationResult::Violated { .. })
    }

    pub fn is_error(&self) -> bool {
        matches!(self, ConstraintEvaluationResult::Error { .. })
    }

    pub fn into_violation(self) -> Option<ValidationViolation> {
        match self {
            ConstraintEvaluationResult::Violated {
                violating_value,
                message,
                details,
            } => Some(ValidationViolation {
                value: violating_value,
                source_constraint_component: ConstraintComponentId::new("http://example.org/constraint"),
                source_shape: ShapeId::new("http://example.org/shape"),
                focus_node: Term::NamedNode(NamedNode::new("http://example.org/focus").unwrap()),
                result_path: None,
                result_severity: Severity::Violation,
                result_message: message,
                details: details,
            }),
            _ => None,
        }
    }
}
