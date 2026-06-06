//! RDF-star term types: quoted triples and their subject/predicate/object terms.
//!
//! Defines [`QuotedTriple`], [`StarSubject`], [`StarPredicate`], [`StarObject`],
//! and [`Annotation`], together with their ARQ [`Term`] / [`TriplePattern`]
//! conversions and [`fmt::Display`] implementations.

use crate::algebra::{Literal, Term, TriplePattern};
use anyhow::{anyhow, Context};
use oxirs_core::model::{NamedNode, Variable};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── Types ───────────────────────────────────────────────────────────────────

/// A *quoted triple* — an RDF-star triple usable as a subject or object.
///
/// Subjects and objects may themselves be quoted triples, enabling arbitrary nesting.
///
/// ```text
/// <<  <<  <s1>  <p1>  <o1>  >>  <certainty>  "0.9"  >>
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QuotedTriple {
    /// Subject: a named node, blank node, variable, or recursively nested quoted triple
    pub subject: StarSubject,
    /// Predicate: always a named node or a variable
    pub predicate: StarPredicate,
    /// Object: any RDF term, or a recursively nested quoted triple
    pub object: StarObject,
}

impl QuotedTriple {
    /// Construct a new quoted triple
    pub fn new(subject: StarSubject, predicate: StarPredicate, object: StarObject) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Return the nesting depth of this quoted triple (1 for a plain quoted triple,
    /// 2+ for nested quoted triples)
    pub fn nesting_depth(&self) -> usize {
        let s_depth = self.subject.nesting_depth();
        let o_depth = self.object.nesting_depth();
        1 + s_depth.max(o_depth)
    }

    /// Return `true` if the quoted triple contains any variable (it is a pattern)
    pub fn is_pattern(&self) -> bool {
        self.subject.is_variable()
            || self.predicate.is_variable()
            || self.object.is_variable()
            || self.subject.contains_variable()
            || self.object.contains_variable()
    }

    /// Collect all variables used in this quoted triple (recursive)
    pub fn variables(&self) -> Vec<Variable> {
        let mut vars = Vec::new();
        self.subject.collect_variables(&mut vars);
        self.predicate.collect_variables(&mut vars);
        self.object.collect_variables(&mut vars);
        vars
    }
}

impl fmt::Display for QuotedTriple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "<< {} {} {} >>",
            self.subject, self.predicate, self.object
        )
    }
}

// ─── StarSubject ─────────────────────────────────────────────────────────────

/// A term that can appear as the subject of a quoted triple.
/// (Subjects cannot be literals per the RDF-star spec.)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StarSubject {
    /// An IRI
    NamedNode(NamedNode),
    /// A blank node identified by a string
    BlankNode(String),
    /// A variable (for SPARQL-star patterns)
    Variable(Variable),
    /// A recursively nested quoted triple
    Quoted(Box<QuotedTriple>),
}

impl StarSubject {
    /// Return `true` if this is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, StarSubject::Variable(_))
    }

    /// Return `true` if any nested element is a variable
    pub fn contains_variable(&self) -> bool {
        match self {
            StarSubject::Quoted(qt) => qt.is_pattern(),
            _ => self.is_variable(),
        }
    }

    /// Recursively collect variable names
    pub fn collect_variables(&self, out: &mut Vec<Variable>) {
        match self {
            StarSubject::Variable(v) => out.push(v.clone()),
            StarSubject::Quoted(qt) => {
                qt.subject.collect_variables(out);
                qt.predicate.collect_variables(out);
                qt.object.collect_variables(out);
            }
            _ => {}
        }
    }

    /// Return the nesting depth contributed by this subject
    pub fn nesting_depth(&self) -> usize {
        match self {
            StarSubject::Quoted(qt) => qt.nesting_depth(),
            _ => 0,
        }
    }

    /// Convert to an ARQ [`Term`]
    pub fn to_term(&self) -> Term {
        match self {
            StarSubject::NamedNode(n) => Term::Iri(n.clone()),
            StarSubject::BlankNode(id) => Term::BlankNode(id.clone()),
            StarSubject::Variable(v) => Term::Variable(v.clone()),
            StarSubject::Quoted(qt) => Term::QuotedTriple(Box::new(qt.to_triple_pattern())),
        }
    }

    /// Try to construct from an ARQ [`Term`]
    pub fn from_term(term: &Term) -> anyhow::Result<Self> {
        match term {
            Term::Iri(n) => Ok(StarSubject::NamedNode(n.clone())),
            Term::BlankNode(id) => Ok(StarSubject::BlankNode(id.clone())),
            Term::Variable(v) => Ok(StarSubject::Variable(v.clone())),
            Term::QuotedTriple(tp) => Ok(StarSubject::Quoted(Box::new(
                QuotedTriple::from_triple_pattern(tp)?,
            ))),
            Term::Literal(_) => Err(anyhow!("literals cannot be used as RDF-star subjects")),
            Term::PropertyPath(_) => Err(anyhow!("property paths cannot be RDF-star subjects")),
        }
    }
}

impl fmt::Display for StarSubject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StarSubject::NamedNode(n) => write!(f, "{n}"),
            StarSubject::BlankNode(id) => write!(f, "_:{id}"),
            StarSubject::Variable(v) => write!(f, "?{}", v.name()),
            StarSubject::Quoted(qt) => write!(f, "{qt}"),
        }
    }
}

// ─── StarPredicate ────────────────────────────────────────────────────────────

/// A term that can appear as the predicate of a quoted triple.
/// (Only named nodes and variables are permitted.)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StarPredicate {
    /// An IRI predicate
    NamedNode(NamedNode),
    /// A variable predicate (for SPARQL-star patterns)
    Variable(Variable),
}

impl StarPredicate {
    /// Return `true` if this is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, StarPredicate::Variable(_))
    }

    /// Collect variables into `out`
    pub fn collect_variables(&self, out: &mut Vec<Variable>) {
        if let StarPredicate::Variable(v) = self {
            out.push(v.clone());
        }
    }

    /// Convert to an ARQ [`Term`]
    pub fn to_term(&self) -> Term {
        match self {
            StarPredicate::NamedNode(n) => Term::Iri(n.clone()),
            StarPredicate::Variable(v) => Term::Variable(v.clone()),
        }
    }

    /// Try to construct from an ARQ [`Term`]
    pub fn from_term(term: &Term) -> anyhow::Result<Self> {
        match term {
            Term::Iri(n) => Ok(StarPredicate::NamedNode(n.clone())),
            Term::Variable(v) => Ok(StarPredicate::Variable(v.clone())),
            other => Err(anyhow!("term {other} cannot be a predicate")),
        }
    }
}

impl fmt::Display for StarPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StarPredicate::NamedNode(n) => write!(f, "{n}"),
            StarPredicate::Variable(v) => write!(f, "?{}", v.name()),
        }
    }
}

// ─── StarObject ───────────────────────────────────────────────────────────────

/// A term that can appear as the object of a quoted triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StarObject {
    /// An IRI
    NamedNode(NamedNode),
    /// A blank node
    BlankNode(String),
    /// A literal value
    Literal(Literal),
    /// A variable (for SPARQL-star patterns)
    Variable(Variable),
    /// A recursively nested quoted triple
    Quoted(Box<QuotedTriple>),
}

impl StarObject {
    /// Return `true` if this is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, StarObject::Variable(_))
    }

    /// Return `true` if any nested element is a variable
    pub fn contains_variable(&self) -> bool {
        match self {
            StarObject::Quoted(qt) => qt.is_pattern(),
            _ => self.is_variable(),
        }
    }

    /// Collect variables into `out`
    pub fn collect_variables(&self, out: &mut Vec<Variable>) {
        match self {
            StarObject::Variable(v) => out.push(v.clone()),
            StarObject::Quoted(qt) => {
                qt.subject.collect_variables(out);
                qt.predicate.collect_variables(out);
                qt.object.collect_variables(out);
            }
            _ => {}
        }
    }

    /// Return the nesting depth contributed by this object
    pub fn nesting_depth(&self) -> usize {
        match self {
            StarObject::Quoted(qt) => qt.nesting_depth(),
            _ => 0,
        }
    }

    /// Convert to an ARQ [`Term`]
    pub fn to_term(&self) -> Term {
        match self {
            StarObject::NamedNode(n) => Term::Iri(n.clone()),
            StarObject::BlankNode(id) => Term::BlankNode(id.clone()),
            StarObject::Literal(l) => Term::Literal(l.clone()),
            StarObject::Variable(v) => Term::Variable(v.clone()),
            StarObject::Quoted(qt) => Term::QuotedTriple(Box::new(qt.to_triple_pattern())),
        }
    }

    /// Try to construct from an ARQ [`Term`]
    pub fn from_term(term: &Term) -> anyhow::Result<Self> {
        match term {
            Term::Iri(n) => Ok(StarObject::NamedNode(n.clone())),
            Term::BlankNode(id) => Ok(StarObject::BlankNode(id.clone())),
            Term::Literal(l) => Ok(StarObject::Literal(l.clone())),
            Term::Variable(v) => Ok(StarObject::Variable(v.clone())),
            Term::QuotedTriple(tp) => Ok(StarObject::Quoted(Box::new(
                QuotedTriple::from_triple_pattern(tp)?,
            ))),
            Term::PropertyPath(_) => Err(anyhow!("property paths cannot be RDF-star objects")),
        }
    }
}

impl fmt::Display for StarObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StarObject::NamedNode(n) => write!(f, "{n}"),
            StarObject::BlankNode(id) => write!(f, "_:{id}"),
            StarObject::Literal(l) => write!(f, "{l}"),
            StarObject::Variable(v) => write!(f, "?{}", v.name()),
            StarObject::Quoted(qt) => write!(f, "{qt}"),
        }
    }
}

// ─── QuotedTriple helpers ─────────────────────────────────────────────────────

impl QuotedTriple {
    /// Convert to an ARQ [`TriplePattern`]
    pub fn to_triple_pattern(&self) -> TriplePattern {
        TriplePattern::new(
            self.subject.to_term(),
            self.predicate.to_term(),
            self.object.to_term(),
        )
    }

    /// Try to construct from an ARQ [`TriplePattern`]
    pub fn from_triple_pattern(pattern: &TriplePattern) -> anyhow::Result<Self> {
        let subject =
            StarSubject::from_term(&pattern.subject).context("converting quoted triple subject")?;
        let predicate = StarPredicate::from_term(&pattern.predicate)
            .context("converting quoted triple predicate")?;
        let object =
            StarObject::from_term(&pattern.object).context("converting quoted triple object")?;
        Ok(QuotedTriple::new(subject, predicate, object))
    }

    /// Build a quoted triple from raw IRI strings (convenience for tests)
    pub fn from_iris(s: &str, p: &str, o: &str) -> anyhow::Result<Self> {
        Ok(QuotedTriple::new(
            StarSubject::NamedNode(NamedNode::new(s)?),
            StarPredicate::NamedNode(NamedNode::new(p)?),
            StarObject::NamedNode(NamedNode::new(o)?),
        ))
    }
}

// ─── Annotation ──────────────────────────────────────────────────────────────

/// A concrete annotation: a (predicate, object) pair attached to a quoted triple
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Annotation {
    /// Annotation predicate
    pub predicate: NamedNode,
    /// Annotation object
    pub object: StarObject,
}

impl Annotation {
    /// Construct a new annotation
    pub fn new(predicate: NamedNode, object: StarObject) -> Self {
        Self { predicate, object }
    }
}

impl fmt::Display for Annotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}> {}", self.predicate, self.object)
    }
}
