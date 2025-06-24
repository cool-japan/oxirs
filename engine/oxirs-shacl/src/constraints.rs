//! SHACL constraint implementation
//! 
//! This module implements all SHACL Core constraints and validation logic.

use std::collections::{HashMap, HashSet};
use regex::Regex;
use serde::{Deserialize, Serialize};

use oxirs_core::{
    model::{NamedNode, Term, Triple, BlankNode, Literal},
    store::Store,
    OxirsError,
};

use crate::{
    ShaclError, Result, PropertyPath, Severity, ShapeId, ConstraintComponentId,
    SHACL_VOCAB, sparql::SparqlConstraint,
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
            Constraint::Class(_) => ConstraintComponentId("sh:ClassConstraintComponent".to_string()),
            Constraint::Datatype(_) => ConstraintComponentId("sh:DatatypeConstraintComponent".to_string()),
            Constraint::NodeKind(_) => ConstraintComponentId("sh:NodeKindConstraintComponent".to_string()),
            Constraint::MinCount(_) => ConstraintComponentId("sh:MinCountConstraintComponent".to_string()),
            Constraint::MaxCount(_) => ConstraintComponentId("sh:MaxCountConstraintComponent".to_string()),
            Constraint::MinExclusive(_) => ConstraintComponentId("sh:MinExclusiveConstraintComponent".to_string()),
            Constraint::MaxExclusive(_) => ConstraintComponentId("sh:MaxExclusiveConstraintComponent".to_string()),
            Constraint::MinInclusive(_) => ConstraintComponentId("sh:MinInclusiveConstraintComponent".to_string()),
            Constraint::MaxInclusive(_) => ConstraintComponentId("sh:MaxInclusiveConstraintComponent".to_string()),
            Constraint::MinLength(_) => ConstraintComponentId("sh:MinLengthConstraintComponent".to_string()),
            Constraint::MaxLength(_) => ConstraintComponentId("sh:MaxLengthConstraintComponent".to_string()),
            Constraint::Pattern(_) => ConstraintComponentId("sh:PatternConstraintComponent".to_string()),
            Constraint::LanguageIn(_) => ConstraintComponentId("sh:LanguageInConstraintComponent".to_string()),
            Constraint::UniqueLang(_) => ConstraintComponentId("sh:UniqueLangConstraintComponent".to_string()),
            Constraint::Equals(_) => ConstraintComponentId("sh:EqualsConstraintComponent".to_string()),
            Constraint::Disjoint(_) => ConstraintComponentId("sh:DisjointConstraintComponent".to_string()),
            Constraint::LessThan(_) => ConstraintComponentId("sh:LessThanConstraintComponent".to_string()),
            Constraint::LessThanOrEquals(_) => ConstraintComponentId("sh:LessThanOrEqualsConstraintComponent".to_string()),
            Constraint::In(_) => ConstraintComponentId("sh:InConstraintComponent".to_string()),
            Constraint::HasValue(_) => ConstraintComponentId("sh:HasValueConstraintComponent".to_string()),
            Constraint::Not(_) => ConstraintComponentId("sh:NotConstraintComponent".to_string()),
            Constraint::And(_) => ConstraintComponentId("sh:AndConstraintComponent".to_string()),
            Constraint::Or(_) => ConstraintComponentId("sh:OrConstraintComponent".to_string()),
            Constraint::Xone(_) => ConstraintComponentId("sh:XoneConstraintComponent".to_string()),
            Constraint::Node(_) => ConstraintComponentId("sh:NodeConstraintComponent".to_string()),
            Constraint::QualifiedValueShape(_) => ConstraintComponentId("sh:QualifiedValueShapeConstraintComponent".to_string()),
            Constraint::Closed(_) => ConstraintComponentId("sh:ClosedConstraintComponent".to_string()),
            Constraint::Sparql(_) => ConstraintComponentId("sh:SPARQLConstraintComponent".to_string()),
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
}

/// Trait for validating constraint values
pub trait ConstraintValidator {
    fn validate(&self) -> Result<()>;
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
            
            let regex = regex_builder
                .case_insensitive(case_insensitive)
                .multi_line(multi_line)
                .dot_matches_new_line(dot_matches_new_line)
                .build()
                .map_err(|e| ShaclError::ConstraintValidation(
                    format!("Invalid regex pattern '{}': {}", self.pattern, e)
                ))?;
        } else {
            let _regex = Regex::new(&self.pattern)
                .map_err(|e| ShaclError::ConstraintValidation(
                    format!("Invalid regex pattern '{}': {}", self.pattern, e)
                ))?;
        }
        
        Ok(())
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
                    "Empty language tag in sh:languageIn".to_string()
                ));
            }
            // TODO: More thorough BCP 47 validation
        }
        Ok(())
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

/// sh:in constraint - validates enumerated values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InConstraint {
    pub values: Vec<Term>,
}

impl ConstraintValidator for InConstraint {
    fn validate(&self) -> Result<()> {
        if self.values.is_empty() {
            return Err(ShaclError::ConstraintValidation(
                "sh:in constraint must have at least one value".to_string()
            ));
        }
        Ok(())
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

/// sh:and constraint - conjunction constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AndConstraint {
    pub shapes: Vec<ShapeId>,
}

impl ConstraintValidator for AndConstraint {
    fn validate(&self) -> Result<()> {
        if self.shapes.is_empty() {
            return Err(ShaclError::ConstraintValidation(
                "sh:and constraint must have at least one shape".to_string()
            ));
        }
        Ok(())
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
                "sh:or constraint must have at least one shape".to_string()
            ));
        }
        Ok(())
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
                "sh:xone constraint must have at least two shapes".to_string()
            ));
        }
        Ok(())
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
                return Err(ShaclError::ConstraintValidation(
                    format!("qualifiedMinCount ({}) cannot be greater than qualifiedMaxCount ({})", min, max)
                ));
            }
        }
        
        Ok(())
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
        details: HashMap<String, String>
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
        ConstraintEvaluationResult::Error {
            message,
            details,
        }
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
        let constraint = ClassConstraint { class_iri: class_iri.clone() };
        
        assert!(constraint.validate().is_ok());
        assert_eq!(constraint.class_iri, class_iri);
    }
    
    #[test]
    fn test_node_kind_constraint() {
        let constraint = NodeKindConstraint { node_kind: NodeKind::Iri };
        assert!(constraint.validate().is_ok());
        
        let constraint = NodeKindConstraint { node_kind: NodeKind::BlankNodeOrLiteral };
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
        
        let constraint = InConstraint { values: values.clone() };
        assert!(constraint.validate().is_ok());
        
        let empty_constraint = InConstraint { values: vec![] };
        assert!(empty_constraint.validate().is_err());
    }
    
    #[test]
    fn test_and_constraint() {
        let shapes = vec![
            ShapeId::new("shape1"),
            ShapeId::new("shape2"),
        ];
        
        let constraint = AndConstraint { shapes: shapes.clone() };
        assert!(constraint.validate().is_ok());
        
        let empty_constraint = AndConstraint { shapes: vec![] };
        assert!(empty_constraint.validate().is_err());
    }
    
    #[test]
    fn test_xone_constraint() {
        let shapes = vec![
            ShapeId::new("shape1"),
            ShapeId::new("shape2"),
        ];
        
        let constraint = XoneConstraint { shapes: shapes.clone() };
        assert!(constraint.validate().is_ok());
        
        let single_shape_constraint = XoneConstraint { 
            shapes: vec![ShapeId::new("shape1")] 
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
            Some(Term::NamedNode(NamedNode::new("http://example.org/test").unwrap())),
            Some("Test violation".to_string())
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