//! Core value constraints (Class, Datatype, NodeKind)

use serde::{Deserialize, Serialize};

use oxirs_core::{
    model::{BlankNode, Literal, NamedNode, Term, Triple},
    Store,
};

use crate::{Result, ShaclError};
use super::{ConstraintValidator, ConstraintEvaluator, ConstraintContext, ConstraintEvaluationResult};

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
            println!(
                "DEBUG ClassConstraint: Checking if value {:?} is instance of class {}",
                value, self.class_iri
            );
            let is_instance = self.check_class_membership(store, value)?;
            println!(
                "DEBUG ClassConstraint: Value {:?} is_instance: {}",
                value, is_instance
            );
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