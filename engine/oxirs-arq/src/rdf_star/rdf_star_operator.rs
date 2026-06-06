//! SPARQL-star annotation patterns and query-plan operators.
//!
//! Defines [`StarPattern`] — an annotation pattern matching all annotation
//! triples attached to a quoted triple — and [`StarOperator`], the set of
//! high-level RDF-star operators (`FindAnnotations`, `AddAnnotation`,
//! `RemoveAnnotation`, `AssertQuoted`, `RetractQuoted`) used in query plans.

use crate::rdf_star::rdf_star_terms::{Annotation, QuotedTriple, StarObject, StarPredicate};
use oxirs_core::model::{NamedNode, Variable};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── StarPattern ─────────────────────────────────────────────────────────────

/// A SPARQL-star *annotation pattern*:
///
/// ```sparql
/// << <s> <p> <o> >>  <anno_pred>  <anno_obj>
/// ```
///
/// This matches all annotation triples attached to the quoted triple `<< s p o >>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarPattern {
    /// The quoted triple being annotated
    pub quoted: QuotedTriple,
    /// The annotation predicate (may be a variable)
    pub predicate: StarPredicate,
    /// The annotation object (any RDF-star object term)
    pub object: StarObject,
}

impl StarPattern {
    /// Construct a new annotation pattern
    pub fn new(quoted: QuotedTriple, predicate: StarPredicate, object: StarObject) -> Self {
        Self {
            quoted,
            predicate,
            object,
        }
    }

    /// Collect all variables in this pattern (including inside the quoted triple)
    pub fn variables(&self) -> Vec<Variable> {
        let mut vars = self.quoted.variables();
        self.predicate.collect_variables(&mut vars);
        self.object.collect_variables(&mut vars);
        vars
    }

    /// Return `true` if this pattern contains at least one variable
    pub fn is_pattern(&self) -> bool {
        !self.variables().is_empty()
    }
}

impl fmt::Display for StarPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {} .", self.quoted, self.predicate, self.object)
    }
}

// ─── StarOperator ─────────────────────────────────────────────────────────────

/// High-level SPARQL-star operators that appear in query plans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StarOperator {
    /// Find all annotation triples matching a [`StarPattern`]
    FindAnnotations {
        /// The annotation pattern to match
        pattern: StarPattern,
    },
    /// Add annotation triples to a quoted triple in the dataset
    AddAnnotation {
        /// The triple to annotate
        triple: QuotedTriple,
        /// The annotations to add
        annotations: Vec<Annotation>,
    },
    /// Remove a specific annotation predicate from a quoted triple
    RemoveAnnotation {
        /// The annotated triple
        triple: QuotedTriple,
        /// The predicate whose annotation should be removed
        predicate: NamedNode,
    },
    /// Asserta that a quoted triple exists (without annotation)
    AssertQuoted {
        /// The triple to assert
        triple: QuotedTriple,
    },
    /// Retract a quoted triple from the dataset
    RetractQuoted {
        /// The triple to retract
        triple: QuotedTriple,
    },
}

impl fmt::Display for StarOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StarOperator::FindAnnotations { pattern } => {
                write!(f, "FindAnnotations({pattern})")
            }
            StarOperator::AddAnnotation {
                triple,
                annotations,
            } => {
                write!(
                    f,
                    "AddAnnotation({triple}, [{} annotations])",
                    annotations.len()
                )
            }
            StarOperator::RemoveAnnotation { triple, predicate } => {
                write!(f, "RemoveAnnotation({triple}, <{predicate}>)")
            }
            StarOperator::AssertQuoted { triple } => {
                write!(f, "AssertQuoted({triple})")
            }
            StarOperator::RetractQuoted { triple } => {
                write!(f, "RetractQuoted({triple})")
            }
        }
    }
}
