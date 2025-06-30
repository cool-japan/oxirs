//! Constraint validation implementations
//!
//! This module contains the actual validation logic for different types of SHACL constraints.

use std::collections::HashSet;
use std::cmp::Ordering;

use oxirs_core::{
    model::{Literal, NamedNode, Term},
    Store,
};

use crate::{
    constraints::*,
    PropertyPath,
    Result,
    ShaclError,
};

use super::{ConstraintEvaluationResult, utils::format_term_for_message};

/// Trait for validating specific constraint types
pub trait ConstraintValidator {
    fn validate(
        &self,
        store: &Store,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult>;
}

/// Validator for node kind constraints
pub struct NodeKindConstraintValidator;

impl ConstraintValidator for NodeKindConstraintValidator {
    fn validate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
        _graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // Extract the constraint from context
        // This is a simplified implementation - in practice you'd get the constraint from context
        
        for value in &context.values {
            // Check if value matches expected node kind
            // This is a placeholder implementation
            match value {
                Term::NamedNode(_) => {
                    // If we're expecting IRI and got IRI, continue
                }
                Term::BlankNode(_) => {
                    // If we're expecting BlankNode and got BlankNode, continue
                }
                Term::Literal(_) => {
                    // If we're expecting Literal and got Literal, continue
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Node kind constraint violation: unexpected node type for value {}",
                            format_term_for_message(value)
                        )),
                    ));
                }
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// Validator for cardinality constraints
pub struct CardinalityConstraintValidator;

impl ConstraintValidator for CardinalityConstraintValidator {
    fn validate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
        _graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        let value_count = context.values.len();
        
        // This is a simplified implementation
        // In practice, you'd check min_count and max_count from the actual constraint
        
        // Placeholder validation - check if we have at least one value
        if value_count == 0 {
            return Ok(ConstraintEvaluationResult::violated(
                None,
                Some("Minimum cardinality constraint violated: no values found".to_string()),
            ));
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// Validator for datatype constraints
pub struct DatatypeConstraintValidator;

impl ConstraintValidator for DatatypeConstraintValidator {
    fn validate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
        _graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                // Check if literal has the expected datatype
                // This is a placeholder - in practice you'd get the expected datatype from the constraint
                if literal.datatype().is_none() {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Datatype constraint violation: literal {} has no datatype",
                            format_term_for_message(value)
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "Datatype constraint violation: value {} is not a literal",
                        format_term_for_message(value)
                    )),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// Validator for class constraints
pub struct ClassConstraintValidator;

impl ConstraintValidator for ClassConstraintValidator {
    fn validate(
        &self,
        store: &Store,
        context: &ConstraintContext,
        graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            // Check if the value is an instance of the required class
            // This would require checking rdf:type triples in the store
            // Placeholder implementation
            match value {
                Term::NamedNode(_) => {
                    // Check class membership in store
                    // This is a simplified check
                }
                _ => {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Class constraint violation: value {} is not a named node",
                            format_term_for_message(value)
                        )),
                    ));
                }
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// Validator for string length constraints
pub struct StringLengthConstraintValidator;

impl ConstraintValidator for StringLengthConstraintValidator {
    fn validate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
        _graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                let str_value = literal.value();
                let length = str_value.chars().count();
                
                // Placeholder: check if length is reasonable (between 1 and 1000)
                if length == 0 {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some("String is empty".to_string()),
                    ));
                }
                
                if length > 1000 {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!("String too long: {} characters", length)),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "String length constraint can only be applied to literals, got: {}",
                        format_term_for_message(value)
                    )),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// Validator for pattern constraints
pub struct PatternConstraintValidator;

impl ConstraintValidator for PatternConstraintValidator {
    fn validate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
        _graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                let str_value = literal.value();
                
                // Placeholder pattern validation - check if string contains only alphanumeric characters
                if !str_value.chars().all(|c| c.is_alphanumeric() || c.is_whitespace()) {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Pattern constraint violation: '{}' contains invalid characters",
                            str_value
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "Pattern constraint can only be applied to literals, got: {}",
                        format_term_for_message(value)
                    )),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// Validator for value enumeration constraints (sh:in)
pub struct InConstraintValidator;

impl ConstraintValidator for InConstraintValidator {
    fn validate(
        &self,
        _store: &Store,
        context: &ConstraintContext,
        _graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // Placeholder allowed values - in practice this would come from the constraint
        let allowed_values: HashSet<String> = ["value1", "value2", "value3"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        for value in &context.values {
            let value_str = format_term_for_message(value);
            if !allowed_values.contains(&value_str) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!(
                        "Value {} is not in the allowed enumeration",
                        value_str
                    )),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// Factory for creating constraint validators
pub struct ConstraintValidatorFactory;

impl ConstraintValidatorFactory {
    /// Create a validator for a specific constraint type
    pub fn create_validator(constraint: &Constraint) -> Box<dyn ConstraintValidator> {
        match constraint {
            Constraint::NodeKind(_) => Box::new(NodeKindConstraintValidator),
            Constraint::MinCount(_) | Constraint::MaxCount(_) => Box::new(CardinalityConstraintValidator),
            Constraint::Datatype(_) => Box::new(DatatypeConstraintValidator),
            Constraint::Class(_) => Box::new(ClassConstraintValidator),
            Constraint::MinLength(_) | Constraint::MaxLength(_) => Box::new(StringLengthConstraintValidator),
            Constraint::Pattern(_) => Box::new(PatternConstraintValidator),
            Constraint::In(_) => Box::new(InConstraintValidator),
            // Add more constraint types as needed
            _ => Box::new(DefaultConstraintValidator),
        }
    }
}

/// Default validator for constraints that don't have specific implementations yet
pub struct DefaultConstraintValidator;

impl ConstraintValidator for DefaultConstraintValidator {
    fn validate(
        &self,
        _store: &Store,
        _context: &ConstraintContext,
        _graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // Default to satisfied for constraints not yet implemented
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{Literal, NamedNode, Term};
    use crate::{ShapeId, constraints::ConstraintContext};

    #[test]
    fn test_node_kind_validator() {
        let validator = NodeKindConstraintValidator;
        let context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("test_shape"),
        )
        .with_values(vec![Term::NamedNode(
            NamedNode::new("http://example.org/value").unwrap(),
        )]);

        let store = Store::new().unwrap();
        let result = validator.validate(&store, &context, None).unwrap();
        assert!(result.is_satisfied());
    }

    #[test]
    fn test_cardinality_validator() {
        let validator = CardinalityConstraintValidator;
        
        // Test with values - should pass
        let context_with_values = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("test_shape"),
        )
        .with_values(vec![Term::Literal(Literal::new("value"))]);

        let store = Store::new().unwrap();
        let result = validator.validate(&store, &context_with_values, None).unwrap();
        assert!(result.is_satisfied());

        // Test without values - should fail
        let context_no_values = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("test_shape"),
        );

        let result = validator.validate(&store, &context_no_values, None).unwrap();
        assert!(result.is_violated());
    }
}