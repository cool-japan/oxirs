//! Conversion utilities for StarTerm to core RDF terms.
//!
//! This module provides conversion functions to bridge between RDF-star terms
//! and the core OxiRS RDF terms.

use crate::model::StarTerm;
use crate::{StarError, StarResult};
use oxirs_core::model::{
    BlankNode as CoreBlankNode, NamedNode as CoreNamedNode, Object, Predicate, Subject,
};

/// Convert StarTerm to a core RDF Subject
pub(super) fn star_term_to_subject(term: &StarTerm) -> StarResult<Subject> {
    match term {
        StarTerm::NamedNode(nn) => {
            let named_node = CoreNamedNode::new(&nn.iri).map_err(StarError::CoreError)?;
            Ok(Subject::NamedNode(named_node))
        }
        StarTerm::BlankNode(bn) => {
            let blank_node = CoreBlankNode::new(&bn.id).map_err(StarError::CoreError)?;
            Ok(Subject::BlankNode(blank_node))
        }
        StarTerm::Literal(_) => Err(StarError::invalid_term_type(
            "Literal cannot be used as subject".to_string(),
        )),
        StarTerm::QuotedTriple(_) => Err(StarError::invalid_term_type(
            "Quoted triple cannot be converted to core RDF subject".to_string(),
        )),
        StarTerm::Variable(_) => Err(StarError::invalid_term_type(
            "Variable cannot be converted to core RDF subject".to_string(),
        )),
    }
}

/// Convert StarTerm to a core RDF Predicate
pub(super) fn star_term_to_predicate(term: &StarTerm) -> StarResult<Predicate> {
    match term {
        StarTerm::NamedNode(nn) => {
            let named_node = CoreNamedNode::new(&nn.iri).map_err(StarError::CoreError)?;
            Ok(Predicate::NamedNode(named_node))
        }
        _ => Err(StarError::invalid_term_type(
            "Only IRIs can be used as predicates".to_string(),
        )),
    }
}

/// Convert StarTerm to a core RDF Object
pub(super) fn star_term_to_object(term: &StarTerm) -> StarResult<Object> {
    match term {
        StarTerm::NamedNode(nn) => {
            let named_node = CoreNamedNode::new(&nn.iri).map_err(StarError::CoreError)?;
            Ok(Object::NamedNode(named_node))
        }
        StarTerm::BlankNode(bn) => {
            let blank_node = CoreBlankNode::new(&bn.id).map_err(StarError::CoreError)?;
            Ok(Object::BlankNode(blank_node))
        }
        StarTerm::Literal(lit) => {
            let core_literal = if let Some(ref language) = lit.language {
                // Language-tagged literal
                oxirs_core::model::Literal::new_language_tagged_literal(&lit.value, language)
                    .map_err(|e| StarError::parse_error(format!("Invalid language tag: {e}")))?
            } else if let Some(ref datatype) = lit.datatype {
                // Typed literal
                let core_datatype =
                    oxirs_core::model::NamedNode::new(&datatype.iri).map_err(|e| {
                        StarError::invalid_term_type(format!("Invalid datatype IRI: {e}"))
                    })?;
                oxirs_core::model::Literal::new_typed_literal(&lit.value, core_datatype)
            } else {
                // Simple literal
                oxirs_core::model::Literal::new_simple_literal(&lit.value)
            };
            Ok(Object::Literal(core_literal))
        }
        StarTerm::QuotedTriple(_) => Err(StarError::invalid_term_type(
            "Quoted triple cannot be converted to core RDF object".to_string(),
        )),
        StarTerm::Variable(_) => Err(StarError::invalid_term_type(
            "Variable cannot be converted to core RDF object".to_string(),
        )),
    }
}
