//! Comparison constraint implementations

use super::constraint_context::{ConstraintContext, ConstraintEvaluationResult};
use crate::Result;
use oxirs_core::{model::Term, rdf_store::Store};
use serde::{Deserialize, Serialize};

/// Equals constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EqualsConstraint {
    pub property: Term,
}

impl EqualsConstraint {
    pub fn new(property: Term) -> Self {
        Self { property }
    }

    pub fn validate(&self) -> Result<()> {
        // Basic validation of the constraint structure
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        use oxirs_core::model::{Predicate, Subject};

        // Get the property values for the constraint property
        let mut constraint_property_values = Vec::new();

        if let (Term::NamedNode(focus_node), Term::NamedNode(property_node)) =
            (&context.focus_node, &self.property)
        {
            let subject = Subject::from(focus_node.clone());
            let predicate = Predicate::from(property_node.clone());

            let quads = store.find_quads(Some(&subject), Some(&predicate), None, None)?;
            for quad in quads {
                constraint_property_values.push(quad.object().clone().into());
            }
        }

        // Check if current values equal constraint property values
        for current_value in &context.values {
            if !constraint_property_values.contains(current_value) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(current_value.clone()),
                    Some(format!(
                        "Value {current_value} does not equal any value of property {}",
                        self.property
                    )),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

/// Disjoint constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisjointConstraint {
    pub property: Term,
}

impl DisjointConstraint {
    pub fn new(property: Term) -> Self {
        Self { property }
    }

    pub fn validate(&self) -> Result<()> {
        // Basic validation of the constraint structure
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        use oxirs_core::model::{Predicate, Subject};

        // Get the property values for the constraint property
        let mut constraint_property_values = Vec::new();

        if let (Term::NamedNode(focus_node), Term::NamedNode(property_node)) =
            (&context.focus_node, &self.property)
        {
            let subject = Subject::from(focus_node.clone());
            let predicate = Predicate::from(property_node.clone());

            let quads = store.find_quads(Some(&subject), Some(&predicate), None, None)?;
            for quad in quads {
                constraint_property_values.push(quad.object().clone().into());
            }
        }

        // Check if current values are disjoint from constraint property values
        for current_value in &context.values {
            if constraint_property_values.contains(current_value) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(current_value.clone()),
                    Some(format!(
                        "Value {current_value} is not disjoint from values of property {}",
                        self.property
                    )),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

/// Less than constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LessThanConstraint {
    pub property: Term,
}

impl LessThanConstraint {
    pub fn new(property: Term) -> Self {
        Self { property }
    }

    pub fn validate(&self) -> Result<()> {
        // Basic validation of the constraint structure
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        use crate::validation::utils::{is_numeric_term, parse_numeric_value};
        use oxirs_core::model::{Predicate, Subject};

        // Get the property values for the constraint property
        let mut constraint_property_values = Vec::new();

        if let (Term::NamedNode(focus_node), Term::NamedNode(property_node)) =
            (&context.focus_node, &self.property)
        {
            let subject = Subject::from(focus_node.clone());
            let predicate = Predicate::from(property_node.clone());

            let quads = store.find_quads(Some(&subject), Some(&predicate), None, None)?;
            for quad in quads {
                constraint_property_values.push(quad.object().clone().into());
            }
        }

        // Check if current values are less than constraint property values
        for current_value in &context.values {
            if !is_numeric_term(current_value) {
                continue; // Skip non-numeric values
            }

            let current_num = parse_numeric_value(current_value)?;
            let mut satisfied = false;

            for constraint_value in &constraint_property_values {
                if is_numeric_term(constraint_value) {
                    let constraint_num = parse_numeric_value(constraint_value)?;
                    if current_num < constraint_num {
                        satisfied = true;
                        break;
                    }
                }
            }

            if !satisfied {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(current_value.clone()),
                    Some(format!(
                        "Value {current_value} is not less than any value of property {}",
                        self.property
                    )),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

/// Less than or equals constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LessThanOrEqualsConstraint {
    pub property: Term,
}

impl LessThanOrEqualsConstraint {
    pub fn new(property: Term) -> Self {
        Self { property }
    }

    pub fn validate(&self) -> Result<()> {
        // Basic validation of the constraint structure
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        use crate::validation::utils::{is_numeric_term, parse_numeric_value};
        use oxirs_core::model::{Predicate, Subject};

        // Get the property values for the constraint property
        let mut constraint_property_values = Vec::new();

        if let (Term::NamedNode(focus_node), Term::NamedNode(property_node)) =
            (&context.focus_node, &self.property)
        {
            let subject = Subject::from(focus_node.clone());
            let predicate = Predicate::from(property_node.clone());

            let quads = store.find_quads(Some(&subject), Some(&predicate), None, None)?;
            for quad in quads {
                constraint_property_values.push(quad.object().clone().into());
            }
        }

        // Check if current values are less than or equal to constraint property values
        for current_value in &context.values {
            if !is_numeric_term(current_value) {
                continue; // Skip non-numeric values
            }

            let current_num = parse_numeric_value(current_value)?;
            let mut satisfied = false;

            for constraint_value in &constraint_property_values {
                if is_numeric_term(constraint_value) {
                    let constraint_num = parse_numeric_value(constraint_value)?;
                    if current_num <= constraint_num {
                        satisfied = true;
                        break;
                    }
                }
            }

            if !satisfied {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(current_value.clone()),
                    Some(format!("Value {current_value} is not less than or equal to any value of property {}", self.property)),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

/// In constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InConstraint {
    pub values: Vec<Term>,
}

impl InConstraint {
    pub fn new(values: Vec<Term>) -> Self {
        Self { values }
    }

    pub fn validate(&self) -> Result<()> {
        // Basic validation of the constraint structure
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        _store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        // Check if all values in the context are in the allowed set
        for value in &context.values {
            if !self.values.contains(value) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!("Value {value} is not in the allowed set of values")),
                ));
            }
        }

        Ok(ConstraintEvaluationResult::Satisfied)
    }
}

/// Has value constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HasValueConstraint {
    pub value: Term,
}

impl HasValueConstraint {
    pub fn new(value: Term) -> Self {
        Self { value }
    }

    pub fn validate(&self) -> Result<()> {
        // Basic validation of the constraint structure
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        _store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        // Check if any value in the context matches the required value
        for value in &context.values {
            if value == &self.value {
                return Ok(ConstraintEvaluationResult::Satisfied);
            }
        }

        // If no matching value found, constraint is violated
        Ok(ConstraintEvaluationResult::violated(
            None,
            Some(format!(
                "No value matches required value: {value}",
                value = self.value
            )),
        ))
    }
}
