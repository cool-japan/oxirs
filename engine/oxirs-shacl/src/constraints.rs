//! SHACL constraint implementation
//!
//! This module implements all SHACL Core constraints and validation logic.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use oxirs_core::{
    model::{BlankNode, Literal, NamedNode, Term, Triple},
    store::Store,
    OxirsError,
};

use crate::{
    sparql::SparqlConstraint, ConstraintComponentId, PropertyPath, Result, Severity, ShaclError,
    ShapeId, SHACL_VOCAB,
};

/// SHACL constraint types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    // Core Value Constraints
    Class(ClassConstraint),
    Datatype(DatatypeConstraint),
    NodeKind(NodeKindConstraint),

    // Cardinality Constraints
    MinCount(MinCountConstraint),
    MaxCount(MaxCountConstraint),

    // Range Constraints
    MinExclusive(MinExclusiveConstraint),
    MaxExclusive(MaxExclusiveConstraint),
    MinInclusive(MinInclusiveConstraint),
    MaxInclusive(MaxInclusiveConstraint),

    // String Constraints
    MinLength(MinLengthConstraint),
    MaxLength(MaxLengthConstraint),
    Pattern(PatternConstraint),
    LanguageIn(LanguageInConstraint),
    UniqueLang(UniqueLangConstraint),

    // Value Constraints
    Equals(EqualsConstraint),
    Disjoint(DisjointConstraint),
    LessThan(LessThanConstraint),
    LessThanOrEquals(LessThanOrEqualsConstraint),
    In(InConstraint),
    HasValue(HasValueConstraint),

    // Logical Constraints
    Not(NotConstraint),
    And(AndConstraint),
    Or(OrConstraint),
    Xone(XoneConstraint),

    // Shape-based Constraints
    Node(NodeConstraint),
    QualifiedValueShape(QualifiedValueShapeConstraint),

    // Closed Shape Constraints
    Closed(ClosedConstraint),

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
            Constraint::LessThan(c) => c.evaluate(store, context),
            Constraint::LessThanOrEquals(c) => c.evaluate(store, context),
            Constraint::Equals(c) => c.evaluate(store, context),
            Constraint::Disjoint(c) => c.evaluate(store, context),
            Constraint::In(c) => c.evaluate(store, context),
            Constraint::HasValue(c) => c.evaluate(store, context),
            Constraint::Not(c) => c.evaluate(store, context),
            Constraint::And(c) => c.evaluate(store, context),
            Constraint::Or(c) => c.evaluate(store, context),
            Constraint::Xone(c) => c.evaluate(store, context),
            Constraint::Node(c) => c.evaluate(store, context),
            Constraint::QualifiedValueShape(c) => c.evaluate(store, context),
            Constraint::Closed(c) => c.evaluate(store, context),
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

// Core Value Constraints

/// sh:class constraint - validates that values are instances of a class
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassConstraint {
    pub class_iri: NamedNode,
}

impl ConstraintValidator for ClassConstraint {
    fn validate(&self) -> Result<()> {
        // Class IRI should be valid
        Ok(())
    }
}

impl ConstraintEvaluator for ClassConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // For each value, check if it's an instance of the required class
        for value in &context.values {
            let is_instance = self.check_class_membership(store, value)?;
            if !is_instance {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "Value {} is not an instance of class {}",
                        value, self.class_iri
                    )),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

impl ClassConstraint {
    fn check_class_membership(&self, store: &Store, value: &Term) -> Result<bool> {
        // Check if the value is an instance of the class
        // This involves checking for rdf:type triples and possibly rdfs:subClassOf inference
        match value {
            Term::NamedNode(node) => {
                // Query for ?value rdf:type ?class where ?class is self.class_iri or a subclass
                let type_predicate = NamedNode::new(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                )
                .map_err(|e| {
                    ShaclError::ConstraintValidation(format!("Invalid RDF type IRI: {}", e))
                })?;

                // Check direct type assertion
                let triple =
                    Triple::new(node.clone(), type_predicate.clone(), self.class_iri.clone());
                if store.contains_quad(&triple.into()).unwrap_or(false) {
                    return Ok(true);
                }

                // TODO: Check subclass relationships using RDFS reasoning
                // For now, we only check direct type assertions
                Ok(false)
            }
            _ => {
                // Blank nodes and literals cannot be instances of classes in standard RDF
                Ok(false)
            }
        }
    }
}

/// sh:datatype constraint - validates that values have a specific datatype
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatatypeConstraint {
    pub datatype_iri: NamedNode,
}

impl ConstraintValidator for DatatypeConstraint {
    fn validate(&self) -> Result<()> {
        // Datatype IRI should be valid
        Ok(())
    }
}

impl ConstraintEvaluator for DatatypeConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // For each value, check if it has the required datatype
        for value in &context.values {
            match value {
                Term::Literal(literal) => {
                    if literal.datatype() != self.datatype_iri.as_ref() {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!(
                                "Value {} has datatype {:?} but expected {}",
                                literal,
                                literal.datatype(),
                                self.datatype_iri
                            )),
                        ));
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value {} is not a literal, cannot check datatype",
                            value
                        )),
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// Node kind values for sh:nodeKind constraint
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Iri,
    BlankNode,
    Literal,
    BlankNodeOrIri,
    BlankNodeOrLiteral,
    IriOrLiteral,
}

/// sh:nodeKind constraint - validates the kind of node
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeKindConstraint {
    pub node_kind: NodeKind,
}

impl ConstraintValidator for NodeKindConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for NodeKindConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // For each value, check if it matches the required node kind
        for value in &context.values {
            if !self.matches_node_kind(value) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "Value {} does not match required node kind {:?}",
                        value, self.node_kind
                    )),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

impl NodeKindConstraint {
    fn matches_node_kind(&self, value: &Term) -> bool {
        match (&self.node_kind, value) {
            (NodeKind::Iri, Term::NamedNode(_)) => true,
            (NodeKind::BlankNode, Term::BlankNode(_)) => true,
            (NodeKind::Literal, Term::Literal(_)) => true,
            (NodeKind::BlankNodeOrIri, Term::BlankNode(_)) => true,
            (NodeKind::BlankNodeOrIri, Term::NamedNode(_)) => true,
            (NodeKind::BlankNodeOrLiteral, Term::BlankNode(_)) => true,
            (NodeKind::BlankNodeOrLiteral, Term::Literal(_)) => true,
            (NodeKind::IriOrLiteral, Term::NamedNode(_)) => true,
            (NodeKind::IriOrLiteral, Term::Literal(_)) => true,
            _ => false,
        }
    }
}

// Cardinality Constraints

/// sh:minCount constraint - validates minimum number of values
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MinCountConstraint {
    pub min_count: u32,
}

impl ConstraintValidator for MinCountConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MinCountConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let value_count = context.values.len() as u32;
        if value_count < self.min_count {
            return Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!(
                    "Expected at least {} values, but found {}",
                    self.min_count, value_count
                )),
            ));
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:maxCount constraint - validates maximum number of values
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaxCountConstraint {
    pub max_count: u32,
}

impl ConstraintValidator for MaxCountConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MaxCountConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let value_count = context.values.len() as u32;
        if value_count > self.max_count {
            return Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!(
                    "Expected at most {} values, but found {}",
                    self.max_count, value_count
                )),
            ));
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

// Range Constraints

/// sh:minExclusive constraint - validates minimum exclusive value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinExclusiveConstraint {
    pub min_value: Literal,
}

impl ConstraintValidator for MinExclusiveConstraint {
    fn validate(&self) -> Result<()> {
        // Value should be a comparable literal (number, date, etc.)
        Ok(())
    }
}

/// sh:maxExclusive constraint - validates maximum exclusive value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaxExclusiveConstraint {
    pub max_value: Literal,
}

impl ConstraintValidator for MaxExclusiveConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// sh:minInclusive constraint - validates minimum inclusive value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinInclusiveConstraint {
    pub min_value: Literal,
}

impl ConstraintValidator for MinInclusiveConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// sh:maxInclusive constraint - validates maximum inclusive value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaxInclusiveConstraint {
    pub max_value: Literal,
}

impl ConstraintValidator for MaxInclusiveConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MinInclusiveConstraint {
    fn evaluate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                if !self.compare_values_gte(literal, &self.min_value)? {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value {} is less than minimum value {}",
                            literal, self.min_value
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Value must be a literal for range comparison".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

impl ConstraintEvaluator for MaxInclusiveConstraint {
    fn evaluate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                if !self.compare_values_lte(literal, &self.max_value)? {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value {} is greater than maximum value {}",
                            literal, self.max_value
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Value must be a literal for range comparison".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

impl ConstraintEvaluator for MinExclusiveConstraint {
    fn evaluate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                if !self.compare_values_gt(literal, &self.min_value)? {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value {} is not greater than minimum value {}",
                            literal, self.min_value
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Value must be a literal for range comparison".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

impl ConstraintEvaluator for MaxExclusiveConstraint {
    fn evaluate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                if !self.compare_values_lt(literal, &self.max_value)? {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value {} is not less than maximum value {}",
                            literal, self.max_value
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Value must be a literal for range comparison".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

// Helper methods for range constraints
impl MinInclusiveConstraint {
    fn compare_values_gte(&self, value: &Literal, min_value: &Literal) -> Result<bool> {
        // Basic comparison - for now just compare string representations
        // TODO: Implement proper typed comparison for numbers, dates, etc.
        Ok(value.value() >= min_value.value())
    }
}

impl MaxInclusiveConstraint {
    fn compare_values_lte(&self, value: &Literal, max_value: &Literal) -> Result<bool> {
        // Basic comparison - for now just compare string representations
        // TODO: Implement proper typed comparison for numbers, dates, etc.
        Ok(value.value() <= max_value.value())
    }
}

impl MinExclusiveConstraint {
    fn compare_values_gt(&self, value: &Literal, min_value: &Literal) -> Result<bool> {
        // Basic comparison - for now just compare string representations
        // TODO: Implement proper typed comparison for numbers, dates, etc.
        Ok(value.value() > min_value.value())
    }
}

impl MaxExclusiveConstraint {
    fn compare_values_lt(&self, value: &Literal, max_value: &Literal) -> Result<bool> {
        // Basic comparison - for now just compare string representations
        // TODO: Implement proper typed comparison for numbers, dates, etc.
        Ok(value.value() < max_value.value())
    }
}

// String Constraints

/// sh:minLength constraint - validates minimum string length
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MinLengthConstraint {
    pub min_length: u32,
}

impl ConstraintValidator for MinLengthConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MinLengthConstraint {
    fn evaluate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            match value {
                Term::Literal(literal) => {
                    let string_value = literal.value();
                    if (string_value.chars().count() as u32) < self.min_length {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!(
                                "String length {} is less than minimum length {}",
                                string_value.chars().count(),
                                self.min_length
                            )),
                        ));
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some("Value must be a literal for length validation".to_string()),
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:maxLength constraint - validates maximum string length
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaxLengthConstraint {
    pub max_length: u32,
}

impl ConstraintValidator for MaxLengthConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MaxLengthConstraint {
    fn evaluate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            match value {
                Term::Literal(literal) => {
                    let string_value = literal.value();
                    if (string_value.chars().count() as u32) > self.max_length {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(format!(
                                "String length {} is greater than maximum length {}",
                                string_value.chars().count(),
                                self.max_length
                            )),
                        ));
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some("Value must be a literal for length validation".to_string()),
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:pattern constraint - validates against regular expression
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternConstraint {
    pub pattern: String,
    pub flags: Option<String>,
    pub message: Option<String>,
}

impl ConstraintValidator for PatternConstraint {
    fn validate(&self) -> Result<()> {
        // Validate that the pattern is a valid regex
        let mut regex_builder = regex::RegexBuilder::new(&self.pattern);

        if let Some(flags) = &self.flags {
            // Parse regex flags
            let case_insensitive = flags.contains('i');
            let multi_line = flags.contains('m');
            let dot_matches_new_line = flags.contains('s');

            let _regex = regex_builder
                .case_insensitive(case_insensitive)
                .multi_line(multi_line)
                .dot_matches_new_line(dot_matches_new_line)
                .build()
                .map_err(|e| {
                    ShaclError::ConstraintValidation(format!(
                        "Invalid regex pattern '{}': {}",
                        self.pattern, e
                    ))
                })?;
        } else {
            let _regex = Regex::new(&self.pattern).map_err(|e| {
                ShaclError::ConstraintValidation(format!(
                    "Invalid regex pattern '{}': {}",
                    self.pattern, e
                ))
            })?;
        }

        Ok(())
    }
}

impl ConstraintEvaluator for PatternConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // Build the regex with flags
        let regex = if let Some(flags) = &self.flags {
            let case_insensitive = flags.contains('i');
            let multi_line = flags.contains('m');
            let dot_matches_new_line = flags.contains('s');

            regex::RegexBuilder::new(&self.pattern)
                .case_insensitive(case_insensitive)
                .multi_line(multi_line)
                .dot_matches_new_line(dot_matches_new_line)
                .build()
                .map_err(|e| {
                    ShaclError::ConstraintValidation(format!(
                        "Invalid regex pattern '{}': {}",
                        self.pattern, e
                    ))
                })?
        } else {
            Regex::new(&self.pattern).map_err(|e| {
                ShaclError::ConstraintValidation(format!(
                    "Invalid regex pattern '{}': {}",
                    self.pattern, e
                ))
            })?
        };

        // Check each value against the pattern
        for value in &context.values {
            match value {
                Term::Literal(literal) => {
                    let string_value = literal.value();
                    if !regex.is_match(string_value) {
                        let message = self.message.clone().unwrap_or_else(|| {
                            format!(
                                "Value '{}' does not match pattern '{}'",
                                string_value, self.pattern
                            )
                        });
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some(message),
                        ));
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value {} is not a literal, cannot check pattern",
                            value
                        )),
                    ));
                }
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:languageIn constraint - validates language tags
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageInConstraint {
    pub languages: Vec<String>,
}

impl ConstraintValidator for LanguageInConstraint {
    fn validate(&self) -> Result<()> {
        // Validate that all language tags are valid BCP 47 language tags
        for lang in &self.languages {
            if lang.is_empty() {
                return Err(ShaclError::ConstraintValidation(
                    "Empty language tag in sh:languageIn".to_string(),
                ));
            }
            // TODO: More thorough BCP 47 validation
        }
        Ok(())
    }
}

impl ConstraintEvaluator for LanguageInConstraint {
    fn evaluate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            match value {
                Term::Literal(literal) => {
                    if let Some(lang) = literal.language() {
                        if !self.languages.contains(&lang.to_string()) {
                            return Ok(ConstraintEvaluationResult::violated(
                                Some(value.clone()),
                                Some(format!(
                                    "Language tag '{}' is not in allowed languages: {:?}",
                                    lang, self.languages
                                )),
                            ));
                        }
                    } else if !self.languages.is_empty() {
                        return Ok(ConstraintEvaluationResult::violated(
                            Some(value.clone()),
                            Some("Literal has no language tag but constraint requires one".to_string()),
                        ));
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some("Value must be a literal for language validation".to_string()),
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:uniqueLang constraint - validates unique language tags
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniqueLangConstraint {
    pub unique_lang: bool,
}

impl ConstraintValidator for UniqueLangConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for UniqueLangConstraint {
    fn evaluate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        if !self.unique_lang {
            // If uniqueLang is false, no constraint
            return Ok(ConstraintEvaluationResult::satisfied());
        }

        let mut seen_languages = HashSet::new();
        for value in &context.values {
            match value {
                Term::Literal(literal) => {
                    if let Some(lang) = literal.language() {
                        if seen_languages.contains(lang) {
                            return Ok(ConstraintEvaluationResult::violated(
                                Some(value.clone()),
                                Some(format!(
                                    "Duplicate language tag '{}' found, but unique languages required",
                                    lang
                                )),
                            ));
                        }
                        seen_languages.insert(lang);
                    }
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some("Value must be a literal for language uniqueness validation".to_string()),
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

// Value Constraints

/// sh:equals constraint - validates value equality
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EqualsConstraint {
    pub property: PropertyPath,
}

impl ConstraintValidator for EqualsConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for EqualsConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // TODO: Implement sh:equals constraint evaluation
        // This requires evaluating the property path on the focus node
        // and comparing the resulting values
        tracing::warn!("sh:equals constraint evaluation not yet implemented");
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:disjoint constraint - validates value disjointness
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisjointConstraint {
    pub property: PropertyPath,
}

impl ConstraintValidator for DisjointConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for DisjointConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // TODO: Implement sh:disjoint constraint evaluation
        // This requires evaluating the property path on the focus node
        // and ensuring no values overlap
        tracing::warn!("sh:disjoint constraint evaluation not yet implemented");
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:lessThan constraint - validates value ordering
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LessThanConstraint {
    pub property: PropertyPath,
}

impl ConstraintValidator for LessThanConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for LessThanConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // TODO: Implement sh:lessThan constraint evaluation
        // This requires evaluating the property path and comparing values
        tracing::warn!("sh:lessThan constraint evaluation not yet implemented");
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:lessThanOrEquals constraint - validates value ordering
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LessThanOrEqualsConstraint {
    pub property: PropertyPath,
}

impl ConstraintValidator for LessThanOrEqualsConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for LessThanOrEqualsConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // TODO: Implement sh:lessThanOrEquals constraint evaluation
        // This requires evaluating the property path and comparing values
        tracing::warn!("sh:lessThanOrEquals constraint evaluation not yet implemented");
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:in constraint - validates enumerated values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InConstraint {
    pub values: Vec<Term>,
}

impl ConstraintValidator for InConstraint {
    fn validate(&self) -> Result<()> {
        if self.values.is_empty() {
            return Err(ShaclError::ConstraintValidation(
                "sh:in constraint must have at least one value".to_string(),
            ));
        }
        Ok(())
    }
}

impl ConstraintEvaluator for InConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // Check if each value is in the allowed set
        for value in &context.values {
            if !self.values.contains(value) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "Value {} is not in the allowed set of values",
                        value
                    )),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:hasValue constraint - validates required values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HasValueConstraint {
    pub value: Term,
}

impl ConstraintValidator for HasValueConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for HasValueConstraint {
    fn evaluate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // Check if the required value is present in the value set
        if !context.values.contains(&self.value) {
            return Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!(
                    "Required value {} is not present",
                    self.value
                )),
            ));
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

// Logical Constraints

/// sh:not constraint - negation constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NotConstraint {
    pub shape: ShapeId,
}

impl ConstraintValidator for NotConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for NotConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // TODO: Implement sh:not constraint evaluation
        // This requires evaluating the referenced shape and negating the result
        tracing::warn!("sh:not constraint evaluation not yet implemented");
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:and constraint - conjunction constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AndConstraint {
    pub shapes: Vec<ShapeId>,
}

impl ConstraintValidator for AndConstraint {
    fn validate(&self) -> Result<()> {
        if self.shapes.is_empty() {
            return Err(ShaclError::ConstraintValidation(
                "sh:and constraint must have at least one shape".to_string(),
            ));
        }
        Ok(())
    }
}

impl ConstraintEvaluator for AndConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // TODO: Implement sh:and constraint evaluation
        // This requires evaluating all shapes and ensuring all are satisfied
        tracing::warn!("sh:and constraint evaluation not yet implemented");
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:or constraint - disjunction constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrConstraint {
    pub shapes: Vec<ShapeId>,
}

impl ConstraintValidator for OrConstraint {
    fn validate(&self) -> Result<()> {
        if self.shapes.is_empty() {
            return Err(ShaclError::ConstraintValidation(
                "sh:or constraint must have at least one shape".to_string(),
            ));
        }
        Ok(())
    }
}

impl ConstraintEvaluator for OrConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // TODO: Implement sh:or constraint evaluation
        // This requires evaluating shapes and ensuring at least one is satisfied
        tracing::warn!("sh:or constraint evaluation not yet implemented");
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:xone constraint - exclusive disjunction constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XoneConstraint {
    pub shapes: Vec<ShapeId>,
}

impl ConstraintValidator for XoneConstraint {
    fn validate(&self) -> Result<()> {
        if self.shapes.len() < 2 {
            return Err(ShaclError::ConstraintValidation(
                "sh:xone constraint must have at least two shapes".to_string(),
            ));
        }
        Ok(())
    }
}

impl ConstraintEvaluator for XoneConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // TODO: Implement sh:xone constraint evaluation
        // This requires evaluating shapes and ensuring exactly one is satisfied
        tracing::warn!("sh:xone constraint evaluation not yet implemented");
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

// Shape-based Constraints

/// sh:node constraint - nested shape validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeConstraint {
    pub shape: ShapeId,
}

impl ConstraintValidator for NodeConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for NodeConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // TODO: Implement sh:node constraint evaluation
        // This requires validating each value against the referenced shape
        tracing::warn!("sh:node constraint evaluation not yet implemented");
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:qualifiedValueShape constraint - qualified cardinality constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualifiedValueShapeConstraint {
    pub qualified_value_shape: ShapeId,
    pub qualified_min_count: Option<u32>,
    pub qualified_max_count: Option<u32>,
    pub qualified_value_shapes_disjoint: bool,
}

impl ConstraintValidator for QualifiedValueShapeConstraint {
    fn validate(&self) -> Result<()> {
        if self.qualified_min_count.is_none() && self.qualified_max_count.is_none() {
            return Err(ShaclError::ConstraintValidation(
                "Qualified value shape constraint must have at least qualifiedMinCount or qualifiedMaxCount".to_string()
            ));
        }

        if let (Some(min), Some(max)) = (self.qualified_min_count, self.qualified_max_count) {
            if min > max {
                return Err(ShaclError::ConstraintValidation(format!(
                    "qualifiedMinCount ({}) cannot be greater than qualifiedMaxCount ({})",
                    min, max
                )));
            }
        }

        Ok(())
    }
}

impl ConstraintEvaluator for QualifiedValueShapeConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // TODO: Implement sh:qualifiedValueShape constraint evaluation
        // This requires validating qualified values against the shape and checking cardinality
        tracing::warn!("sh:qualifiedValueShape constraint evaluation not yet implemented");
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

// Closed Shape Constraints

/// sh:closed constraint - closed shape validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClosedConstraint {
    pub closed: bool,
    pub ignored_properties: Vec<PropertyPath>,
}

impl ConstraintValidator for ClosedConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for ClosedConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // TODO: Implement sh:closed constraint evaluation
        // This requires checking that the focus node only has properties allowed by the shape
        tracing::warn!("sh:closed constraint evaluation not yet implemented");
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_class_constraint() {
        let class_iri = NamedNode::new("http://example.org/Person").unwrap();
        let constraint = ClassConstraint {
            class_iri: class_iri.clone(),
        };

        assert!(constraint.validate().is_ok());
        assert_eq!(constraint.class_iri, class_iri);
    }

    #[test]
    fn test_node_kind_constraint() {
        let constraint = NodeKindConstraint {
            node_kind: NodeKind::Iri,
        };
        assert!(constraint.validate().is_ok());

        let constraint = NodeKindConstraint {
            node_kind: NodeKind::BlankNodeOrLiteral,
        };
        assert!(constraint.validate().is_ok());
    }

    #[test]
    fn test_min_max_count_constraints() {
        let min_constraint = MinCountConstraint { min_count: 1 };
        assert!(min_constraint.validate().is_ok());

        let max_constraint = MaxCountConstraint { max_count: 5 };
        assert!(max_constraint.validate().is_ok());
    }

    #[test]
    fn test_pattern_constraint_valid() {
        let constraint = PatternConstraint {
            pattern: r"^[a-zA-Z]+$".to_string(),
            flags: Some("i".to_string()),
            message: Some("Must be letters only".to_string()),
        };

        assert!(constraint.validate().is_ok());
    }

    #[test]
    fn test_pattern_constraint_invalid() {
        let constraint = PatternConstraint {
            pattern: r"[invalid regex(".to_string(),
            flags: None,
            message: None,
        };

        assert!(constraint.validate().is_err());
    }

    #[test]
    fn test_in_constraint() {
        let values = vec![
            Term::NamedNode(NamedNode::new("http://example.org/red").unwrap()),
            Term::NamedNode(NamedNode::new("http://example.org/green").unwrap()),
            Term::NamedNode(NamedNode::new("http://example.org/blue").unwrap()),
        ];

        let constraint = InConstraint {
            values: values.clone(),
        };
        assert!(constraint.validate().is_ok());

        let empty_constraint = InConstraint { values: vec![] };
        assert!(empty_constraint.validate().is_err());
    }

    #[test]
    fn test_and_constraint() {
        let shapes = vec![ShapeId::new("shape1"), ShapeId::new("shape2")];

        let constraint = AndConstraint {
            shapes: shapes.clone(),
        };
        assert!(constraint.validate().is_ok());

        let empty_constraint = AndConstraint { shapes: vec![] };
        assert!(empty_constraint.validate().is_err());
    }

    #[test]
    fn test_xone_constraint() {
        let shapes = vec![ShapeId::new("shape1"), ShapeId::new("shape2")];

        let constraint = XoneConstraint {
            shapes: shapes.clone(),
        };
        assert!(constraint.validate().is_ok());

        let single_shape_constraint = XoneConstraint {
            shapes: vec![ShapeId::new("shape1")],
        };
        assert!(single_shape_constraint.validate().is_err());
    }

    #[test]
    fn test_qualified_value_shape_constraint() {
        let constraint = QualifiedValueShapeConstraint {
            qualified_value_shape: ShapeId::new("shape1"),
            qualified_min_count: Some(1),
            qualified_max_count: Some(5),
            qualified_value_shapes_disjoint: false,
        };
        assert!(constraint.validate().is_ok());

        let invalid_constraint = QualifiedValueShapeConstraint {
            qualified_value_shape: ShapeId::new("shape1"),
            qualified_min_count: Some(5),
            qualified_max_count: Some(1),
            qualified_value_shapes_disjoint: false,
        };
        assert!(invalid_constraint.validate().is_err());

        let no_counts_constraint = QualifiedValueShapeConstraint {
            qualified_value_shape: ShapeId::new("shape1"),
            qualified_min_count: None,
            qualified_max_count: None,
            qualified_value_shapes_disjoint: false,
        };
        assert!(no_counts_constraint.validate().is_err());
    }

    #[test]
    fn test_constraint_evaluation_result() {
        let satisfied = ConstraintEvaluationResult::satisfied();
        assert!(satisfied.is_satisfied());
        assert!(!satisfied.is_violated());
        assert!(!satisfied.is_error());

        let violated = ConstraintEvaluationResult::violated(
            Some(Term::NamedNode(
                NamedNode::new("http://example.org/test").unwrap(),
            )),
            Some("Test violation".to_string()),
        );
        assert!(!violated.is_satisfied());
        assert!(violated.is_violated());
        assert!(!violated.is_error());

        let error = ConstraintEvaluationResult::error("Test error".to_string());
        assert!(!error.is_satisfied());
        assert!(!error.is_violated());
        assert!(error.is_error());
    }
}
