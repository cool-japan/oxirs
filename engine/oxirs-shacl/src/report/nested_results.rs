//! Nested validation result support for complex constraints
//!
//! This module provides comprehensive support for nested validation results,
//! allowing detailed hierarchical reporting of complex constraint violations
//! such as logical constraints (AND, OR, NOT) and shape-based constraints.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{validation::ValidationViolation, ConstraintComponentId, PropertyPath, ShapeId};

use oxirs_core::model::Term;

/// Enhanced validation violation with nested result support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedValidationViolation {
    /// Basic violation information
    pub violation: ValidationViolation,

    /// Nested validation results for complex constraints
    pub nested_results: Option<NestedValidationResults>,

    /// Violation hierarchy level (0 = root, 1 = nested, etc.)
    pub hierarchy_level: u32,

    /// Parent violation ID (for building tree structure)
    pub parent_violation_id: Option<String>,

    /// Unique violation ID
    pub violation_id: String,

    /// Additional context for nested violations
    pub context: NestedViolationContext,
}

/// Nested validation results container
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NestedValidationResults {
    /// Child violations from nested constraint evaluation
    pub child_violations: Vec<NestedValidationViolation>,

    /// Logical constraint context (for AND, OR, NOT, etc.)
    pub logical_context: Option<LogicalConstraintContext>,

    /// Shape constraint context (for node/property shapes)
    pub shape_context: Option<ShapeConstraintContext>,

    /// Qualified constraint context (for qualified value shapes)
    pub qualified_context: Option<QualifiedConstraintContext>,

    /// Summary of nested results
    pub nested_summary: NestedResultSummary,
}

/// Context for logical constraint violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalConstraintContext {
    /// Type of logical constraint
    pub constraint_type: LogicalConstraintType,

    /// Total sub-constraints evaluated
    pub total_sub_constraints: usize,

    /// Sub-constraints that passed
    pub passed_sub_constraints: usize,

    /// Sub-constraints that failed
    pub failed_sub_constraints: usize,

    /// Required number of passing constraints
    pub required_passing: Option<usize>,

    /// Detailed results for each sub-constraint
    pub sub_constraint_results: Vec<SubConstraintResult>,
}

/// Types of logical constraints
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LogicalConstraintType {
    And,
    Or,
    Not,
    ExactlyOne,
    Custom(String),
}

/// Result for individual sub-constraint in logical constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubConstraintResult {
    /// Sub-constraint identifier
    pub constraint_id: String,

    /// Whether this sub-constraint passed
    pub passed: bool,

    /// Sub-constraint description
    pub description: String,

    /// Nested violations from this sub-constraint
    pub nested_violations: Vec<NestedValidationViolation>,

    /// Evaluation order/priority
    pub evaluation_order: u32,
}

/// Context for shape constraint violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeConstraintContext {
    /// Target shape being validated
    pub target_shape: ShapeId,

    /// Type of shape validation
    pub shape_type: ShapeValidationType,

    /// Property constraints that were evaluated
    pub property_constraints: Vec<PropertyConstraintResult>,

    /// Node constraints that were evaluated
    pub node_constraints: Vec<NodeConstraintResult>,

    /// Inheritance chain if applicable
    pub inheritance_chain: Vec<ShapeId>,
}

/// Types of shape validation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ShapeValidationType {
    NodeShape,
    PropertyShape,
    QualifiedValueShape,
    NestedShape,
}

/// Result for property constraint evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyConstraintResult {
    /// Property path
    pub property_path: Option<PropertyPath>,

    /// Constraint type
    pub constraint_type: ConstraintComponentId,

    /// Whether constraint passed
    pub passed: bool,

    /// Values that were validated
    pub validated_values: Vec<Term>,

    /// Specific violations for this property
    pub property_violations: Vec<NestedValidationViolation>,
}

/// Result for node constraint evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConstraintResult {
    /// Constraint type
    pub constraint_type: ConstraintComponentId,

    /// Whether constraint passed
    pub passed: bool,

    /// Target node that was validated
    pub target_node: Term,

    /// Specific violations for this node
    pub node_violations: Vec<NestedValidationViolation>,
}

/// Context for qualified constraint violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualifiedConstraintContext {
    /// Qualified shape being validated
    pub qualified_shape: ShapeId,

    /// Required minimum count
    pub min_count: Option<u32>,

    /// Required maximum count
    pub max_count: Option<u32>,

    /// Actual conforming count found
    pub actual_conforming_count: u32,

    /// Values that conformed to qualified shape
    pub conforming_values: Vec<Term>,

    /// Values that did not conform to qualified shape
    pub non_conforming_values: Vec<Term>,

    /// Whether disjoint checking was required
    pub disjoint_required: bool,

    /// Disjoint violations if any
    pub disjoint_violations: Vec<DisjointViolation>,
}

/// Disjoint violation for qualified value shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisjointViolation {
    /// Value that violates disjoint requirement
    pub violating_value: Term,

    /// Shapes that the value conforms to (causing disjoint violation)
    pub conflicting_shapes: Vec<ShapeId>,
}

/// Summary of nested validation results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NestedResultSummary {
    /// Total nested violations
    pub total_nested_violations: usize,

    /// Maximum nesting depth
    pub max_nesting_depth: u32,

    /// Violations by hierarchy level
    pub violations_by_level: HashMap<u32, usize>,

    /// Violations by constraint type
    pub violations_by_constraint_type: HashMap<ConstraintComponentId, usize>,

    /// Most deeply nested violation
    pub deepest_violation_level: u32,

    /// Root cause analysis
    pub root_causes: Vec<RootCause>,
}

/// Root cause analysis for nested violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCause {
    /// Description of root cause
    pub description: String,

    /// Constraint component that is the root cause
    pub root_constraint: ConstraintComponentId,

    /// Shape that contains the root cause
    pub root_shape: ShapeId,

    /// Number of violations caused by this root issue
    pub affected_violations: usize,

    /// Suggested remediation
    pub suggested_remediation: String,
}

/// Additional context for nested violations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NestedViolationContext {
    /// Evaluation path that led to this violation
    pub evaluation_path: Vec<EvaluationStep>,

    /// Constraint evaluation order
    pub evaluation_order: u32,

    /// Whether this violation caused early termination
    pub caused_early_termination: bool,

    /// Performance metrics for this violation
    pub performance_metrics: NestedViolationMetrics,

    /// Dependencies on other constraints
    pub constraint_dependencies: Vec<ConstraintDependency>,
}

/// Step in constraint evaluation path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationStep {
    /// Step description
    pub description: String,

    /// Constraint being evaluated at this step
    pub constraint: ConstraintComponentId,

    /// Shape context at this step
    pub shape_context: ShapeId,

    /// Step number in evaluation sequence
    pub step_number: u32,
}

/// Performance metrics for nested violation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NestedViolationMetrics {
    /// Time spent evaluating this constraint
    pub evaluation_time_ms: u64,

    /// Memory used during evaluation
    pub memory_usage_bytes: Option<u64>,

    /// Number of sub-evaluations performed
    pub sub_evaluations: u32,

    /// Cache hits during evaluation
    pub cache_hits: u32,

    /// Cache misses during evaluation
    pub cache_misses: u32,
}

/// Dependency between constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDependency {
    /// Dependent constraint
    pub dependent_constraint: ConstraintComponentId,

    /// Dependency type
    pub dependency_type: DependencyType,

    /// Whether dependency was satisfied
    pub satisfied: bool,
}

/// Types of constraint dependencies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DependencyType {
    /// Must evaluate before this constraint
    Prerequisite,

    /// Must evaluate after this constraint
    Subsequent,

    /// Must evaluate together
    Concurrent,

    /// Mutually exclusive
    Exclusive,
}

/// Builder for creating nested validation violations
#[derive(Debug)]
pub struct NestedViolationBuilder {
    violation: ValidationViolation,
    hierarchy_level: u32,
    parent_violation_id: Option<String>,
    nested_results: Option<NestedValidationResults>,
    context: NestedViolationContext,
}

impl NestedViolationBuilder {
    /// Create a new builder from a basic violation
    pub fn from_violation(violation: ValidationViolation) -> Self {
        Self {
            violation,
            hierarchy_level: 0,
            parent_violation_id: None,
            nested_results: None,
            context: NestedViolationContext::default(),
        }
    }

    /// Set hierarchy level
    pub fn with_hierarchy_level(mut self, level: u32) -> Self {
        self.hierarchy_level = level;
        self
    }

    /// Set parent violation ID
    pub fn with_parent(mut self, parent_id: String) -> Self {
        self.parent_violation_id = Some(parent_id);
        self
    }

    /// Add nested results
    pub fn with_nested_results(mut self, results: NestedValidationResults) -> Self {
        self.nested_results = Some(results);
        self
    }

    /// Add logical constraint context
    pub fn with_logical_context(mut self, context: LogicalConstraintContext) -> Self {
        if let Some(ref mut nested_results) = self.nested_results {
            nested_results.logical_context = Some(context);
        } else {
            let nested_results = NestedValidationResults {
                logical_context: Some(context),
                ..Default::default()
            };
            self.nested_results = Some(nested_results);
        }
        self
    }

    /// Add shape constraint context
    pub fn with_shape_context(mut self, context: ShapeConstraintContext) -> Self {
        if let Some(ref mut nested_results) = self.nested_results {
            nested_results.shape_context = Some(context);
        } else {
            let nested_results = NestedValidationResults {
                shape_context: Some(context),
                ..Default::default()
            };
            self.nested_results = Some(nested_results);
        }
        self
    }

    /// Add qualified constraint context
    pub fn with_qualified_context(mut self, context: QualifiedConstraintContext) -> Self {
        if let Some(ref mut nested_results) = self.nested_results {
            nested_results.qualified_context = Some(context);
        } else {
            let nested_results = NestedValidationResults {
                qualified_context: Some(context),
                ..Default::default()
            };
            self.nested_results = Some(nested_results);
        }
        self
    }

    /// Add evaluation step to context
    pub fn add_evaluation_step(mut self, step: EvaluationStep) -> Self {
        self.context.evaluation_path.push(step);
        self
    }

    /// Set evaluation order
    pub fn with_evaluation_order(mut self, order: u32) -> Self {
        self.context.evaluation_order = order;
        self
    }

    /// Build the nested validation violation
    pub fn build(self) -> NestedValidationViolation {
        let violation_id = format!("violation_{}", uuid::Uuid::new_v4());

        NestedValidationViolation {
            violation: self.violation,
            nested_results: self.nested_results,
            hierarchy_level: self.hierarchy_level,
            parent_violation_id: self.parent_violation_id,
            violation_id,
            context: self.context,
        }
    }
}

impl NestedValidationViolation {
    /// Get all violations in the nested hierarchy (flattened)
    pub fn get_all_violations(&self) -> Vec<&NestedValidationViolation> {
        let mut all_violations = vec![self];

        if let Some(ref nested_results) = self.nested_results {
            for child in &nested_results.child_violations {
                all_violations.extend(child.get_all_violations());
            }
        }

        all_violations
    }

    /// Get violations at a specific hierarchy level
    pub fn get_violations_at_level(&self, level: u32) -> Vec<&NestedValidationViolation> {
        let mut violations = Vec::new();

        if self.hierarchy_level == level {
            violations.push(self);
        }

        if let Some(ref nested_results) = self.nested_results {
            for child in &nested_results.child_violations {
                violations.extend(child.get_violations_at_level(level));
            }
        }

        violations
    }

    /// Get the maximum nesting depth
    pub fn get_max_depth(&self) -> u32 {
        let mut max_depth = self.hierarchy_level;

        if let Some(ref nested_results) = self.nested_results {
            for child in &nested_results.child_violations {
                max_depth = max_depth.max(child.get_max_depth());
            }
        }

        max_depth
    }

    /// Get total count of nested violations
    pub fn get_total_violation_count(&self) -> usize {
        let mut count = 1; // Count self

        if let Some(ref nested_results) = self.nested_results {
            for child in &nested_results.child_violations {
                count += child.get_total_violation_count();
            }
        }

        count
    }

    /// Analyze root causes in the violation hierarchy
    pub fn analyze_root_causes(&self) -> Vec<RootCause> {
        let mut root_causes = Vec::new();

        // If this is a leaf violation (no children), it might be a root cause
        if self
            .nested_results
            .as_ref()
            .map_or(true, |nr| nr.child_violations.is_empty())
        {
            root_causes.push(RootCause {
                description: format!(
                    "Root constraint violation: {}",
                    self.violation.source_constraint_component
                ),
                root_constraint: self.violation.source_constraint_component.clone(),
                root_shape: self.violation.source_shape.clone(),
                affected_violations: 1,
                suggested_remediation: self.suggest_remediation(),
            });
        }

        // Recursively analyze children
        if let Some(ref nested_results) = self.nested_results {
            for child in &nested_results.child_violations {
                root_causes.extend(child.analyze_root_causes());
            }
        }

        root_causes
    }

    /// Suggest remediation for this violation
    fn suggest_remediation(&self) -> String {
        match self.violation.source_constraint_component.as_str() {
            "MinCountConstraintComponent" => {
                "Add more values to satisfy minimum cardinality".to_string()
            }
            "MaxCountConstraintComponent" => {
                "Remove excess values to satisfy maximum cardinality".to_string()
            }
            "DatatypeConstraintComponent" => {
                "Ensure values match the required datatype".to_string()
            }
            "ClassConstraintComponent" => {
                "Ensure resources are instances of the required class".to_string()
            }
            "NodeKindConstraintComponent" => {
                "Ensure values are of the correct node kind (IRI, Literal, etc.)".to_string()
            }
            _ => "Review constraint requirements and adjust data accordingly".to_string(),
        }
    }

    /// Generate a detailed tree representation of the violation hierarchy
    pub fn to_tree_string(&self) -> String {
        self.to_tree_string_with_prefix("", true)
    }

    fn to_tree_string_with_prefix(&self, prefix: &str, is_last: bool) -> String {
        let mut result = String::new();

        let connector = if is_last { "└── " } else { "├── " };
        result.push_str(&format!(
            "{}{}{}\n",
            prefix,
            connector,
            self.format_violation_summary()
        ));

        if let Some(ref nested_results) = self.nested_results {
            let child_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

            for (i, child) in nested_results.child_violations.iter().enumerate() {
                let is_last_child = i == nested_results.child_violations.len() - 1;
                result.push_str(&child.to_tree_string_with_prefix(&child_prefix, is_last_child));
            }
        }

        result
    }

    fn format_violation_summary(&self) -> String {
        format!(
            "[{}] {} - {} ({})",
            self.violation.result_severity,
            self.violation.source_constraint_component,
            self.violation.focus_node,
            self.violation.source_shape
        )
    }
}

/// Extension trait for converting basic violations to nested violations
pub trait ToNestedViolation {
    fn to_nested(self) -> NestedViolationBuilder;
}

impl ToNestedViolation for ValidationViolation {
    fn to_nested(self) -> NestedViolationBuilder {
        NestedViolationBuilder::from_violation(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstraintComponentId, Severity, ShapeId};
    use oxirs_core::model::NamedNode;

    #[test]
    fn test_nested_violation_builder() {
        let violation = ValidationViolation {
            focus_node: oxirs_core::model::Term::NamedNode(
                NamedNode::new("http://example.org/test").expect("valid IRI"),
            ),
            result_severity: Severity::Violation,
            source_shape: ShapeId::new("TestShape"),
            source_constraint_component: ConstraintComponentId::new("TestConstraint"),
            result_path: None,
            value: None,
            result_message: Some("Test violation".to_string()),
            details: HashMap::new(),
            nested_results: Vec::new(),
        };

        let nested_violation = violation
            .to_nested()
            .with_hierarchy_level(1)
            .with_evaluation_order(5)
            .build();

        assert_eq!(nested_violation.hierarchy_level, 1);
        assert_eq!(nested_violation.context.evaluation_order, 5);
        assert!(nested_violation.violation_id.starts_with("violation_"));
    }

    #[test]
    fn test_logical_constraint_context() {
        let context = LogicalConstraintContext {
            constraint_type: LogicalConstraintType::And,
            total_sub_constraints: 3,
            passed_sub_constraints: 1,
            failed_sub_constraints: 2,
            required_passing: Some(3),
            sub_constraint_results: Vec::new(),
        };

        assert_eq!(context.constraint_type, LogicalConstraintType::And);
        assert_eq!(context.total_sub_constraints, 3);
        assert_eq!(context.failed_sub_constraints, 2);
    }

    #[test]
    fn test_nested_violation_hierarchy() {
        let root_violation = ValidationViolation {
            focus_node: oxirs_core::model::Term::NamedNode(
                NamedNode::new("http://example.org/root").expect("valid IRI"),
            ),
            result_severity: Severity::Violation,
            source_shape: ShapeId::new("RootShape"),
            source_constraint_component: ConstraintComponentId::new("AndConstraint"),
            result_path: None,
            value: None,
            result_message: Some("AND constraint violation".to_string()),
            details: HashMap::new(),
            nested_results: Vec::new(),
        };

        let child_violation = ValidationViolation {
            focus_node: oxirs_core::model::Term::NamedNode(
                NamedNode::new("http://example.org/child").expect("valid IRI"),
            ),
            result_severity: Severity::Violation,
            source_shape: ShapeId::new("ChildShape"),
            source_constraint_component: ConstraintComponentId::new("MinCountConstraint"),
            result_path: None,
            value: None,
            result_message: Some("Min count violation".to_string()),
            details: HashMap::new(),
            nested_results: Vec::new(),
        };

        let nested_child = child_violation.to_nested().with_hierarchy_level(1).build();

        let mut nested_results = NestedValidationResults::default();
        nested_results.child_violations.push(nested_child);

        let nested_root = root_violation
            .to_nested()
            .with_hierarchy_level(0)
            .with_nested_results(nested_results)
            .build();

        assert_eq!(nested_root.get_max_depth(), 1);
        assert_eq!(nested_root.get_total_violation_count(), 2);

        let all_violations = nested_root.get_all_violations();
        assert_eq!(all_violations.len(), 2);
    }

    #[test]
    fn test_tree_string_generation() {
        let root_violation = ValidationViolation {
            focus_node: oxirs_core::model::Term::NamedNode(
                NamedNode::new("http://example.org/root").expect("valid IRI"),
            ),
            result_severity: Severity::Violation,
            source_shape: ShapeId::new("RootShape"),
            source_constraint_component: ConstraintComponentId::new("AndConstraint"),
            result_path: None,
            value: None,
            result_message: Some("AND constraint violation".to_string()),
            details: HashMap::new(),
            nested_results: Vec::new(),
        };

        let nested_root = root_violation.to_nested().build();
        let tree_string = nested_root.to_tree_string();

        assert!(tree_string.contains("└── "));
        assert!(tree_string.contains("[Violation]"));
        assert!(tree_string.contains("AndConstraint"));
    }
}
