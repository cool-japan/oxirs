//! Core constraint types and traits

use serde::{Deserialize, Serialize};

use oxirs_core::Store;

use crate::constraints::constraint_context::{ConstraintContext, ConstraintEvaluationResult};
use crate::{sparql::SparqlConstraint, ConstraintComponentId, Result, Severity};

/// SHACL constraint types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    // Core Value Constraints
    Class(crate::constraints::value_constraints::ClassConstraint),
    Datatype(crate::constraints::value_constraints::DatatypeConstraint),
    NodeKind(crate::constraints::value_constraints::NodeKindConstraint),

    // Cardinality Constraints
    MinCount(crate::constraints::cardinality_constraints::MinCountConstraint),
    MaxCount(crate::constraints::cardinality_constraints::MaxCountConstraint),

    // Range Constraints
    MinExclusive(crate::constraints::range_constraints::MinExclusiveConstraint),
    MaxExclusive(crate::constraints::range_constraints::MaxExclusiveConstraint),
    MinInclusive(crate::constraints::range_constraints::MinInclusiveConstraint),
    MaxInclusive(crate::constraints::range_constraints::MaxInclusiveConstraint),

    // String Constraints
    MinLength(crate::constraints::string_constraints::MinLengthConstraint),
    MaxLength(crate::constraints::string_constraints::MaxLengthConstraint),
    Pattern(crate::constraints::string_constraints::PatternConstraint),
    LanguageIn(crate::constraints::string_constraints::LanguageInConstraint),
    UniqueLang(crate::constraints::string_constraints::UniqueLangConstraint),

    // Value Constraints
    Equals(crate::constraints::comparison_constraints::EqualsConstraint),
    Disjoint(crate::constraints::comparison_constraints::DisjointConstraint),
    LessThan(crate::constraints::comparison_constraints::LessThanConstraint),
    LessThanOrEquals(crate::constraints::comparison_constraints::LessThanOrEqualsConstraint),
    In(crate::constraints::comparison_constraints::InConstraint),
    HasValue(crate::constraints::comparison_constraints::HasValueConstraint),

    // Logical Constraints
    Not(crate::constraints::logical_constraints::NotConstraint),
    And(crate::constraints::logical_constraints::AndConstraint),
    Or(crate::constraints::logical_constraints::OrConstraint),
    Xone(crate::constraints::logical_constraints::XoneConstraint),

    // Shape-based Constraints
    Node(crate::constraints::shape_constraints::NodeConstraint),
    QualifiedValueShape(crate::constraints::shape_constraints::QualifiedValueShapeConstraint),

    // Closed Shape Constraints
    Closed(crate::constraints::shape_constraints::ClosedConstraint),

    // SPARQL Constraints
    Sparql(SparqlConstraint),
}

impl Constraint {
    /// Validate the constraint itself (check for validity)
    pub fn validate(&self) -> Result<()> {
        match self {
            Constraint::Class(c) => c.validate(),
            Constraint::Datatype(c) => c.validate(),
            Constraint::NodeKind(c) => c.validate(),
            Constraint::MinCount(c) => c.validate(),
            Constraint::MaxCount(c) => c.validate(),
            Constraint::MinExclusive(c) => c.validate(),
            Constraint::MaxExclusive(c) => c.validate(),
            Constraint::MinInclusive(c) => c.validate(),
            Constraint::MaxInclusive(c) => c.validate(),
            Constraint::MinLength(c) => c.validate(),
            Constraint::MaxLength(c) => c.validate(),
            Constraint::Pattern(c) => c.validate(),
            Constraint::LanguageIn(c) => c.validate(),
            Constraint::UniqueLang(c) => c.validate(),
            Constraint::Equals(c) => c.validate(),
            Constraint::Disjoint(c) => c.validate(),
            Constraint::LessThan(c) => c.validate(),
            Constraint::LessThanOrEquals(c) => c.validate(),
            Constraint::In(c) => c.validate(),
            Constraint::HasValue(c) => c.validate(),
            Constraint::Not(c) => c.validate(),
            Constraint::And(c) => c.validate(),
            Constraint::Or(c) => c.validate(),
            Constraint::Xone(c) => c.validate(),
            Constraint::Node(c) => c.validate(),
            Constraint::QualifiedValueShape(c) => c.validate(),
            Constraint::Closed(c) => c.validate(),
            Constraint::Sparql(c) => c.validate(),
        }
    }

    /// Get the constraint component ID for this constraint
    pub fn component_id(&self) -> ConstraintComponentId {
        match self {
            Constraint::Class(_) => {
                ConstraintComponentId("sh:ClassConstraintComponent".to_string())
            }
            Constraint::Datatype(_) => {
                ConstraintComponentId("sh:DatatypeConstraintComponent".to_string())
            }
            Constraint::NodeKind(_) => {
                ConstraintComponentId("sh:NodeKindConstraintComponent".to_string())
            }
            Constraint::MinCount(_) => {
                ConstraintComponentId("sh:MinCountConstraintComponent".to_string())
            }
            Constraint::MaxCount(_) => {
                ConstraintComponentId("sh:MaxCountConstraintComponent".to_string())
            }
            Constraint::MinExclusive(_) => {
                ConstraintComponentId("sh:MinExclusiveConstraintComponent".to_string())
            }
            Constraint::MaxExclusive(_) => {
                ConstraintComponentId("sh:MaxExclusiveConstraintComponent".to_string())
            }
            Constraint::MinInclusive(_) => {
                ConstraintComponentId("sh:MinInclusiveConstraintComponent".to_string())
            }
            Constraint::MaxInclusive(_) => {
                ConstraintComponentId("sh:MaxInclusiveConstraintComponent".to_string())
            }
            Constraint::MinLength(_) => {
                ConstraintComponentId("sh:MinLengthConstraintComponent".to_string())
            }
            Constraint::MaxLength(_) => {
                ConstraintComponentId("sh:MaxLengthConstraintComponent".to_string())
            }
            Constraint::Pattern(_) => {
                ConstraintComponentId("sh:PatternConstraintComponent".to_string())
            }
            Constraint::LanguageIn(_) => {
                ConstraintComponentId("sh:LanguageInConstraintComponent".to_string())
            }
            Constraint::UniqueLang(_) => {
                ConstraintComponentId("sh:UniqueLangConstraintComponent".to_string())
            }
            Constraint::Equals(_) => {
                ConstraintComponentId("sh:EqualsConstraintComponent".to_string())
            }
            Constraint::Disjoint(_) => {
                ConstraintComponentId("sh:DisjointConstraintComponent".to_string())
            }
            Constraint::LessThan(_) => {
                ConstraintComponentId("sh:LessThanConstraintComponent".to_string())
            }
            Constraint::LessThanOrEquals(_) => {
                ConstraintComponentId("sh:LessThanOrEqualsConstraintComponent".to_string())
            }
            Constraint::In(_) => ConstraintComponentId("sh:InConstraintComponent".to_string()),
            Constraint::HasValue(_) => {
                ConstraintComponentId("sh:HasValueConstraintComponent".to_string())
            }
            Constraint::Not(_) => ConstraintComponentId("sh:NotConstraintComponent".to_string()),
            Constraint::And(_) => ConstraintComponentId("sh:AndConstraintComponent".to_string()),
            Constraint::Or(_) => ConstraintComponentId("sh:OrConstraintComponent".to_string()),
            Constraint::Xone(_) => ConstraintComponentId("sh:XoneConstraintComponent".to_string()),
            Constraint::Node(_) => ConstraintComponentId("sh:NodeConstraintComponent".to_string()),
            Constraint::QualifiedValueShape(_) => {
                ConstraintComponentId("sh:QualifiedValueShapeConstraintComponent".to_string())
            }
            Constraint::Closed(_) => {
                ConstraintComponentId("sh:ClosedConstraintComponent".to_string())
            }
            Constraint::Sparql(_) => {
                ConstraintComponentId("sh:SPARQLConstraintComponent".to_string())
            }
        }
    }

    /// Get severity for this constraint (if specified)
    pub fn severity(&self) -> Option<Severity> {
        // Most constraints don't specify their own severity
        None
    }

    /// Get custom message for this constraint (if specified)
    pub fn message(&self) -> Option<&str> {
        match self {
            Constraint::Pattern(c) => c.message.as_deref(),
            Constraint::Sparql(c) => c.message.as_deref(),
            _ => None,
        }
    }

    /// Evaluate this constraint against the given context
    pub fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        match self {
            Constraint::Class(c) => c.evaluate(store, context),
            Constraint::Datatype(c) => c.evaluate(store, context),
            Constraint::NodeKind(c) => c.evaluate(store, context),
            Constraint::MinCount(c) => c.evaluate(store, context),
            Constraint::MaxCount(c) => c.evaluate(store, context),
            Constraint::MinLength(c) => c.evaluate(store, context),
            Constraint::MaxLength(c) => c.evaluate(store, context),
            Constraint::Pattern(c) => c.evaluate(store, context),
            Constraint::LanguageIn(c) => c.evaluate(store, context),
            Constraint::UniqueLang(c) => c.evaluate(store, context),
            Constraint::MinInclusive(c) => c.evaluate(store, context),
            Constraint::MaxInclusive(c) => c.evaluate(store, context),
            Constraint::MinExclusive(c) => c.evaluate(store, context),
            Constraint::MaxExclusive(c) => c.evaluate(store, context),
            Constraint::LessThan(c) => c.evaluate(context, store),
            Constraint::LessThanOrEquals(c) => c.evaluate(context, store),
            Constraint::Equals(c) => c.evaluate(context, store),
            Constraint::Disjoint(c) => c.evaluate(context, store),
            Constraint::In(c) => c.evaluate(context, store),
            Constraint::HasValue(c) => c.evaluate(context, store),
            Constraint::Not(c) => c.evaluate(context, store),
            Constraint::And(c) => c.evaluate(context, store),
            Constraint::Or(c) => c.evaluate(context, store),
            Constraint::Xone(c) => c.evaluate(context, store),
            Constraint::Node(c) => c.evaluate(context, store),
            Constraint::QualifiedValueShape(c) => c.evaluate(context, store),
            Constraint::Closed(c) => c.evaluate(context, store),
            Constraint::Sparql(c) => c.evaluate(store, context),
        }
    }
}

/// Trait for validating constraint definitions
pub trait ConstraintValidator {
    fn validate(&self) -> Result<()>;
}

/// Trait for evaluating constraints against data
pub trait ConstraintEvaluator {
    /// Evaluate the constraint against the given context
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult>;
}
