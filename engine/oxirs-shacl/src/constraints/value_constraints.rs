//! Core value constraints for SHACL validation
//!
//! This module implements the fundamental SHACL value constraints that validate
//! the types and characteristics of RDF values:
//!
//! - [`ClassConstraint`] - Validates that values are instances of a specific class (`sh:class`)
//! - [`DatatypeConstraint`] - Validates that values have a specific datatype (`sh:datatype`)
//! - [`NodeKindConstraint`] - Validates the kind of RDF node (`sh:nodeKind`)
//!
//! # Usage
//!
//! ```rust
//! use oxirs_shacl::constraints::value_constraints::*;
//! use oxirs_core::{Store, model::*};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a class constraint requiring values to be instances of foaf:Person
//! let person_class = NamedNode::new("http://xmlns.com/foaf/0.1/Person")?;
//! let class_constraint = ClassConstraint {
//!     class_iri: person_class,
//! };
//!
//! // Create a datatype constraint requiring string values
//! let string_datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#string")?;
//! let datatype_constraint = DatatypeConstraint {
//!     datatype_iri: string_datatype,
//! };
//!
//! // Create a node kind constraint requiring IRI values
//! let nodekind_constraint = NodeKindConstraint {
//!     node_kind: NodeKind::Iri,
//! };
//! # Ok(())
//! # }
//! ```
//!
//! # SHACL Specification
//!
//! These constraints implement the core value constraint components from the
//! [SHACL specification](https://www.w3.org/TR/shacl/#core-components-value-type):
//!
//! - `sh:class` - Specifies that each value node is an instance of a class
//! - `sh:datatype` - Specifies the datatype that each value node must have
//! - `sh:nodeKind` - Specifies the type of node (IRI, blank node, or literal)

use serde::{Deserialize, Serialize};

use oxirs_core::{
    model::{BlankNode, Literal, NamedNode, Quad, Term, Triple},
    Store,
};

use super::{
    ConstraintContext, ConstraintEvaluationResult, ConstraintEvaluator, ConstraintValidator,
};
use crate::{Result, ShaclError};

/// SHACL `sh:class` constraint that validates values are instances of a specific class.
///
/// This constraint checks that each value node is an instance of the specified class,
/// meaning there exists a triple `?value rdf:type ?class` where `?class` is either
/// the specified class or a subclass of it (when RDFS reasoning is enabled).
///
/// # SHACL Specification
///
/// From [SHACL Core Components - Class Constraint Component](https://www.w3.org/TR/shacl/#ClassConstraintComponent):
/// "Specifies that each value node is an instance of a given type."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::value_constraints::ClassConstraint;
/// use oxirs_core::model::NamedNode;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a constraint requiring values to be instances of foaf:Person
/// let person_class = NamedNode::new("http://xmlns.com/foaf/0.1/Person")?;
/// let constraint = ClassConstraint {
///     class_iri: person_class,
/// };
///
/// // This would validate that any focus node has:
/// // ?focusNode rdf:type foaf:Person
/// # Ok(())
/// # }
/// ```
///
/// # Validation Behavior
///
/// - **Passes**: When the value node is an instance of the specified class
/// - **Fails**: When the value node is not an instance of the specified class
/// - **N/A**: For blank nodes and literals (cannot be class instances in standard RDF)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassConstraint {
    /// The IRI of the class that values must be instances of
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
        store: &dyn Store,
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
    fn check_class_membership(&self, store: &dyn Store, value: &Term) -> Result<bool> {
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
                let quad: Quad = triple.into();
                let subject = quad.subject();
                let predicate = quad.predicate();
                let object = quad.object();
                let graph_name = quad.graph_name();
                if !store
                    .find_quads(
                        Some(subject),
                        Some(predicate),
                        Some(object),
                        Some(graph_name),
                    )
                    .unwrap_or_default()
                    .is_empty()
                {
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

/// SHACL `sh:datatype` constraint that validates values have a specific datatype.
///
/// This constraint checks that each value node is a literal with the specified datatype.
/// Only literal values are considered valid; IRIs and blank nodes will cause violations.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - Datatype Constraint Component](https://www.w3.org/TR/shacl/#DatatypeConstraintComponent):
/// "Specifies that each value node is a literal with a given datatype."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::value_constraints::DatatypeConstraint;
/// use oxirs_core::model::NamedNode;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a constraint requiring string values
/// let string_datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#string")?;
/// let constraint = DatatypeConstraint {
///     datatype_iri: string_datatype,
/// };
///
/// // This would validate that any focus node has literal values with xsd:string datatype:
/// // "Hello World"^^xsd:string (passes)
/// // "123"^^xsd:integer (fails - wrong datatype)
/// // <http://example.org/person> (fails - not a literal)
/// # Ok(())
/// # }
/// ```
///
/// # Validation Behavior
///
/// - **Passes**: When the value node is a literal with the specified datatype
/// - **Fails**: When the value node is a literal with a different datatype
/// - **Fails**: When the value node is an IRI or blank node (not a literal)
///
/// # Common Datatypes
///
/// - `xsd:string` - Text strings
/// - `xsd:integer` - Integer numbers
/// - `xsd:decimal` - Decimal numbers
/// - `xsd:boolean` - Boolean values (true/false)
/// - `xsd:date` - Date values
/// - `xsd:dateTime` - Date and time values
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatatypeConstraint {
    /// The IRI of the datatype that literal values must have
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
        store: &dyn Store,
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

/// Node kind values for SHACL `sh:nodeKind` constraint.
///
/// These values specify the allowed types of RDF nodes according to the
/// [SHACL specification](https://www.w3.org/TR/shacl/#node-kind).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    /// Only IRI nodes are allowed (`sh:IRI`)
    Iri,
    /// Only blank nodes are allowed (`sh:BlankNode`)
    BlankNode,
    /// Only literal nodes are allowed (`sh:Literal`)
    Literal,
    /// Either blank nodes or IRIs are allowed (`sh:BlankNodeOrIRI`)
    BlankNodeOrIri,
    /// Either blank nodes or literals are allowed (`sh:BlankNodeOrLiteral`)
    BlankNodeOrLiteral,
    /// Either IRIs or literals are allowed (`sh:IRIOrLiteral`)
    IriOrLiteral,
}

/// SHACL `sh:nodeKind` constraint that validates the kind of RDF node.
///
/// This constraint restricts the type of RDF node that can appear as a value.
/// It provides fine-grained control over whether values can be IRIs, blank nodes,
/// literals, or combinations thereof.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - Node Kind Constraint Component](https://www.w3.org/TR/shacl/#NodeKindConstraintComponent):
/// "Specifies that each value node is of a given node kind."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::value_constraints::{NodeKindConstraint, NodeKind};
///
/// // Create a constraint requiring only IRI values
/// let iri_constraint = NodeKindConstraint {
///     node_kind: NodeKind::Iri,
/// };
///
/// // Create a constraint allowing either IRIs or literals
/// let iri_or_literal_constraint = NodeKindConstraint {
///     node_kind: NodeKind::IriOrLiteral,
/// };
///
/// // This would validate node types:
/// // <http://example.org/person> (IRI - passes for Iri, IriOrLiteral)
/// // "John Doe" (Literal - fails for Iri, passes for IriOrLiteral)
/// // _:b1 (Blank Node - fails for both Iri and IriOrLiteral)
/// ```
///
/// # Validation Behavior
///
/// Each [`NodeKind`] variant defines which RDF node types are considered valid:
///
/// - [`NodeKind::Iri`]: Only named nodes (IRIs) pass validation
/// - [`NodeKind::BlankNode`]: Only blank nodes pass validation  
/// - [`NodeKind::Literal`]: Only literal values pass validation
/// - [`NodeKind::BlankNodeOrIri`]: Both blank nodes and IRIs pass validation
/// - [`NodeKind::BlankNodeOrLiteral`]: Both blank nodes and literals pass validation
/// - [`NodeKind::IriOrLiteral`]: Both IRIs and literals pass validation
///
/// # Use Cases
///
/// - **Data Quality**: Ensure properties only contain specific node types
/// - **Schema Validation**: Enforce that object properties only reference IRIs
/// - **Type Safety**: Prevent mixed node types in properties that expect consistency
/// - **API Contracts**: Validate that external data matches expected node patterns
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeKindConstraint {
    /// The kind of node that values must be
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
        store: &dyn Store,
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
