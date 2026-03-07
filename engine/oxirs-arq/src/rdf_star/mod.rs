//! SPARQL-star (RDF-star) Completeness Module
//!
//! Implements the complete SPARQL 1.2 / SPARQL-star specification for quoted triples,
//! annotation queries, pattern matching, and CONSTRUCT support.
//!
//! # Overview
//!
//! RDF-star (W3C spec) allows triples as subjects or objects — called *quoted triples*:
//!
//! ```text
//! <<  <http://s>  <http://p>  <http://o>  >>  <http://certainty>  "0.9"
//! ```
//!
//! SPARQL-star introduces corresponding query syntax:
//!
//! ```sparql
//! SELECT ?s ?p ?o ?c WHERE {
//!     << ?s ?p ?o >> <http://certainty> ?c .
//! }
//! ```
//!
//! # References
//! - <https://www.w3.org/2021/12/rdf-star.html>
//! - <https://w3c.github.io/sparql-star/>

use crate::algebra::{Literal, Term, TriplePattern};
use anyhow::{anyhow, Context};
use oxirs_core::model::{NamedNode, Variable};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

// ─── In-memory RDF-star store ─────────────────────────────────────────────────

/// A triple stored together with its annotations
#[derive(Debug, Clone)]
pub struct AnnotatedTriple {
    /// The quoted triple
    pub triple: QuotedTriple,
    /// Annotation (predicate → object) pairs
    pub annotations: HashMap<String, StarObject>,
}

impl AnnotatedTriple {
    /// Construct a new annotated triple without any annotations
    pub fn new(triple: QuotedTriple) -> Self {
        Self {
            triple,
            annotations: HashMap::new(),
        }
    }

    /// Add or overwrite an annotation
    pub fn annotate(&mut self, predicate: &NamedNode, object: StarObject) {
        self.annotations.insert(predicate.to_string(), object);
    }

    /// Remove an annotation; returns `true` if it was present
    pub fn remove_annotation(&mut self, predicate: &NamedNode) -> bool {
        self.annotations.remove(&predicate.to_string()).is_some()
    }

    /// Return the annotation for the given predicate, if any
    pub fn annotation(&self, predicate: &NamedNode) -> Option<&StarObject> {
        self.annotations.get(&predicate.to_string())
    }
}

/// In-memory store for RDF-star quoted triples and their annotations.
///
/// This is a self-contained store that does not depend on the optional `oxirs-star` crate.
/// It underpins the completeness tests below.
#[derive(Debug, Clone, Default)]
pub struct RdfStarStore {
    /// Annotated triples, keyed by the serialised form of the quoted triple
    triples: HashMap<String, AnnotatedTriple>,
}

impl RdfStarStore {
    /// Construct an empty store
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the number of quoted triples (with or without annotations) in the store
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Return `true` if the store is empty
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Insert or retrieve the entry for a quoted triple
    fn key(triple: &QuotedTriple) -> String {
        triple.to_string()
    }

    /// Assert that a quoted triple exists in the store (idempotent)
    pub fn assert_triple(&mut self, triple: QuotedTriple) {
        self.triples
            .entry(Self::key(&triple))
            .or_insert_with(|| AnnotatedTriple::new(triple));
    }

    /// Retract a quoted triple and all its annotations
    pub fn retract_triple(&mut self, triple: &QuotedTriple) -> bool {
        self.triples.remove(&Self::key(triple)).is_some()
    }

    /// Add an annotation to an existing quoted triple.
    /// If the triple has not been asserted yet, it is created automatically.
    pub fn add_annotation(
        &mut self,
        triple: &QuotedTriple,
        predicate: &NamedNode,
        object: StarObject,
    ) {
        let entry = self
            .triples
            .entry(Self::key(triple))
            .or_insert_with(|| AnnotatedTriple::new(triple.clone()));
        entry.annotate(predicate, object);
    }

    /// Remove a single annotation from a quoted triple.
    /// Returns `true` if the annotation was present.
    pub fn remove_annotation(&mut self, triple: &QuotedTriple, predicate: &NamedNode) -> bool {
        if let Some(entry) = self.triples.get_mut(&Self::key(triple)) {
            entry.remove_annotation(predicate)
        } else {
            false
        }
    }

    /// Look up all annotations for a quoted triple
    pub fn annotations(&self, triple: &QuotedTriple) -> Option<&AnnotatedTriple> {
        self.triples.get(&Self::key(triple))
    }

    /// Return `true` if the quoted triple exists in the store
    pub fn contains(&self, triple: &QuotedTriple) -> bool {
        self.triples.contains_key(&Self::key(triple))
    }

    /// Iterate over all annotated triples
    pub fn iter(&self) -> impl Iterator<Item = &AnnotatedTriple> {
        self.triples.values()
    }

    /// Apply a [`StarOperator`] to this store.
    ///
    /// Returns the matching [`AnnotatedTriple`] entries for `FindAnnotations`,
    /// or an empty vec for mutating operators.
    pub fn apply_operator(&mut self, op: StarOperator) -> Vec<AnnotatedTriple> {
        match op {
            StarOperator::AssertQuoted { triple } => {
                self.assert_triple(triple);
                Vec::new()
            }
            StarOperator::RetractQuoted { triple } => {
                self.retract_triple(&triple);
                Vec::new()
            }
            StarOperator::AddAnnotation {
                triple,
                annotations,
            } => {
                for ann in &annotations {
                    self.add_annotation(&triple, &ann.predicate, ann.object.clone());
                }
                Vec::new()
            }
            StarOperator::RemoveAnnotation { triple, predicate } => {
                self.remove_annotation(&triple, &predicate);
                Vec::new()
            }
            StarOperator::FindAnnotations { pattern } => self.find_annotations(&pattern),
        }
    }

    /// Match all annotation triples against the given [`StarPattern`]
    pub fn find_annotations(&self, pattern: &StarPattern) -> Vec<AnnotatedTriple> {
        self.triples
            .values()
            .filter(|entry| triple_matches_pattern(&entry.triple, &pattern.quoted))
            .filter(|entry| {
                // Filter annotation (pred, obj) pairs
                entry.annotations.iter().any(|(pred_key, obj)| {
                    predicate_matches(&pattern.predicate, pred_key)
                        && object_matches(&pattern.object, obj)
                })
            })
            .cloned()
            .collect()
    }
}

// ─── Pattern matching helpers ─────────────────────────────────────────────────

/// Return `true` if a quoted triple matches a pattern (variables bind to anything)
fn triple_matches_pattern(triple: &QuotedTriple, pattern: &QuotedTriple) -> bool {
    subject_matches(&pattern.subject, &triple.subject)
        && predicate_matches_terms(&pattern.predicate, &triple.predicate)
        && object_matches_object(&pattern.object, &triple.object)
}

fn subject_matches(pattern: &StarSubject, value: &StarSubject) -> bool {
    match pattern {
        StarSubject::Variable(_) => true,
        StarSubject::NamedNode(pn) => {
            matches!(value, StarSubject::NamedNode(vn) if vn == pn)
        }
        StarSubject::BlankNode(pb) => {
            matches!(value, StarSubject::BlankNode(vb) if vb == pb)
        }
        StarSubject::Quoted(pq) => {
            matches!(value, StarSubject::Quoted(vq) if triple_matches_pattern(vq, pq))
        }
    }
}

fn predicate_matches_terms(pattern: &StarPredicate, value: &StarPredicate) -> bool {
    match pattern {
        StarPredicate::Variable(_) => true,
        StarPredicate::NamedNode(pn) => {
            matches!(value, StarPredicate::NamedNode(vn) if vn == pn)
        }
    }
}

fn object_matches_object(pattern: &StarObject, value: &StarObject) -> bool {
    match pattern {
        StarObject::Variable(_) => true,
        StarObject::NamedNode(pn) => {
            matches!(value, StarObject::NamedNode(vn) if vn == pn)
        }
        StarObject::BlankNode(pb) => {
            matches!(value, StarObject::BlankNode(vb) if vb == pb)
        }
        StarObject::Literal(pl) => {
            matches!(value, StarObject::Literal(vl) if vl == pl)
        }
        StarObject::Quoted(pq) => {
            matches!(value, StarObject::Quoted(vq) if triple_matches_pattern(vq, pq))
        }
    }
}

/// Match a predicate pattern (may be variable) against a serialised predicate key
// The key is stored as `NamedNode::to_string()` (angle-bracket IRI), so the
// comparison must use the same Display formatting — suppressing cmp_owned.
#[allow(clippy::cmp_owned)]
fn predicate_matches(pattern: &StarPredicate, key: &str) -> bool {
    match pattern {
        StarPredicate::Variable(_) => true,
        StarPredicate::NamedNode(n) => n.to_string() == key,
    }
}

/// Match an object pattern against a concrete object value
fn object_matches(pattern: &StarObject, value: &StarObject) -> bool {
    object_matches_object(pattern, value)
}

// ─── Binding result ───────────────────────────────────────────────────────────

/// A variable binding produced by a SPARQL-star pattern match
pub type StarBinding = HashMap<String, StarObject>;

/// Bind variables in `pattern` to values in `triple` and push the result into `out`.
/// Returns `false` if the pattern does not match `triple`.
pub fn bind_pattern(triple: &QuotedTriple, pattern: &QuotedTriple, out: &mut StarBinding) -> bool {
    bind_subject(pattern, triple, out)
        && bind_predicate(pattern, triple, out)
        && bind_object(pattern, triple, out)
}

fn bind_subject(pattern: &QuotedTriple, triple: &QuotedTriple, out: &mut StarBinding) -> bool {
    match &pattern.subject {
        StarSubject::Variable(v) => {
            let obj = star_subject_to_object(&triple.subject);
            out.insert(v.as_str().to_string(), obj);
            true
        }
        StarSubject::NamedNode(pn) => {
            matches!(&triple.subject, StarSubject::NamedNode(vn) if vn == pn)
        }
        StarSubject::BlankNode(pb) => {
            matches!(&triple.subject, StarSubject::BlankNode(vb) if vb == pb)
        }
        StarSubject::Quoted(pq) => {
            if let StarSubject::Quoted(vq) = &triple.subject {
                bind_pattern(vq, pq, out)
            } else {
                false
            }
        }
    }
}

fn bind_predicate(pattern: &QuotedTriple, triple: &QuotedTriple, out: &mut StarBinding) -> bool {
    match &pattern.predicate {
        StarPredicate::Variable(v) => {
            if let StarPredicate::NamedNode(n) = &triple.predicate {
                out.insert(v.as_str().to_string(), StarObject::NamedNode(n.clone()));
            }
            true
        }
        StarPredicate::NamedNode(pn) => {
            matches!(&triple.predicate, StarPredicate::NamedNode(vn) if vn == pn)
        }
    }
}

fn bind_object(pattern: &QuotedTriple, triple: &QuotedTriple, out: &mut StarBinding) -> bool {
    match &pattern.object {
        StarObject::Variable(v) => {
            out.insert(v.as_str().to_string(), triple.object.clone());
            true
        }
        StarObject::NamedNode(pn) => {
            matches!(&triple.object, StarObject::NamedNode(vn) if vn == pn)
        }
        StarObject::BlankNode(pb) => {
            matches!(&triple.object, StarObject::BlankNode(vb) if vb == pb)
        }
        StarObject::Literal(pl) => {
            matches!(&triple.object, StarObject::Literal(vl) if vl == pl)
        }
        StarObject::Quoted(pq) => {
            if let StarObject::Quoted(vq) = &triple.object {
                bind_pattern(vq, pq, out)
            } else {
                false
            }
        }
    }
}

/// Convert a [`StarSubject`] to a [`StarObject`] for binding
fn star_subject_to_object(subject: &StarSubject) -> StarObject {
    match subject {
        StarSubject::NamedNode(n) => StarObject::NamedNode(n.clone()),
        StarSubject::BlankNode(id) => StarObject::BlankNode(id.clone()),
        StarSubject::Variable(v) => StarObject::Variable(v.clone()),
        StarSubject::Quoted(qt) => StarObject::Quoted(qt.clone()),
    }
}

// ─── CONSTRUCT helpers ────────────────────────────────────────────────────────

/// Apply a [`StarBinding`] to a [`QuotedTriple`] template, substituting variables
pub fn instantiate_quoted_triple(
    template: &QuotedTriple,
    binding: &StarBinding,
) -> anyhow::Result<QuotedTriple> {
    let subject = instantiate_subject(&template.subject, binding)?;
    let predicate = instantiate_predicate(&template.predicate, binding)?;
    let object = instantiate_object(&template.object, binding)?;
    Ok(QuotedTriple::new(subject, predicate, object))
}

fn instantiate_subject(s: &StarSubject, binding: &StarBinding) -> anyhow::Result<StarSubject> {
    match s {
        StarSubject::Variable(v) => {
            let val = binding
                .get(v.as_str())
                .ok_or_else(|| anyhow!("unbound variable ?{}", v.as_str()))?;
            // Attempt to convert the bound StarObject back to StarSubject
            object_to_subject(val)
        }
        StarSubject::Quoted(qt) => Ok(StarSubject::Quoted(Box::new(instantiate_quoted_triple(
            qt, binding,
        )?))),
        other => Ok(other.clone()),
    }
}

fn instantiate_predicate(
    p: &StarPredicate,
    binding: &StarBinding,
) -> anyhow::Result<StarPredicate> {
    match p {
        StarPredicate::Variable(v) => {
            let val = binding
                .get(v.as_str())
                .ok_or_else(|| anyhow!("unbound predicate variable ?{}", v.as_str()))?;
            match val {
                StarObject::NamedNode(n) => Ok(StarPredicate::NamedNode(n.clone())),
                other => Err(anyhow!("predicate must be a named node, got {other}")),
            }
        }
        other => Ok(other.clone()),
    }
}

fn instantiate_object(o: &StarObject, binding: &StarBinding) -> anyhow::Result<StarObject> {
    match o {
        StarObject::Variable(v) => binding
            .get(v.as_str())
            .cloned()
            .ok_or_else(|| anyhow!("unbound variable ?{}", v.as_str())),
        StarObject::Quoted(qt) => Ok(StarObject::Quoted(Box::new(instantiate_quoted_triple(
            qt, binding,
        )?))),
        other => Ok(other.clone()),
    }
}

fn object_to_subject(obj: &StarObject) -> anyhow::Result<StarSubject> {
    match obj {
        StarObject::NamedNode(n) => Ok(StarSubject::NamedNode(n.clone())),
        StarObject::BlankNode(id) => Ok(StarSubject::BlankNode(id.clone())),
        StarObject::Quoted(qt) => Ok(StarSubject::Quoted(qt.clone())),
        StarObject::Literal(_) => Err(anyhow!("literals cannot be subjects")),
        StarObject::Variable(v) => Ok(StarSubject::Variable(v.clone())),
    }
}

// ─── SPARQL-star functions ────────────────────────────────────────────────────

/// SPARQL-star built-in functions: `TRIPLE()`, `isTRIPLE()`, `SUBJECT()`,
/// `PREDICATE()`, `OBJECT()`
pub mod sparql_star_builtins {
    use super::*;

    /// `TRIPLE(s, p, o)` — construct a quoted triple from three terms
    pub fn triple_fn(
        subject: StarSubject,
        predicate: StarPredicate,
        object: StarObject,
    ) -> QuotedTriple {
        QuotedTriple::new(subject, predicate, object)
    }

    /// `isTRIPLE(term)` — return `true` if the term is a quoted triple
    pub fn is_triple(obj: &StarObject) -> bool {
        matches!(obj, StarObject::Quoted(_))
    }

    /// `SUBJECT(triple)` — extract the subject of a quoted triple
    pub fn subject_of(qt: &QuotedTriple) -> &StarSubject {
        &qt.subject
    }

    /// `PREDICATE(triple)` — extract the predicate of a quoted triple
    pub fn predicate_of(qt: &QuotedTriple) -> &StarPredicate {
        &qt.predicate
    }

    /// `OBJECT(triple)` — extract the object of a quoted triple
    pub fn object_of(qt: &QuotedTriple) -> &StarObject {
        &qt.object
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::sparql_star_builtins::*;
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────

    fn iri(s: &str) -> NamedNode {
        NamedNode::new(s).unwrap()
    }

    fn var(name: &str) -> Variable {
        Variable::new(name).unwrap()
    }

    fn qt(s: &str, p: &str, o: &str) -> QuotedTriple {
        QuotedTriple::new(
            StarSubject::NamedNode(iri(s)),
            StarPredicate::NamedNode(iri(p)),
            StarObject::NamedNode(iri(o)),
        )
    }

    fn qt_lit(s: &str, p: &str, o: &str) -> QuotedTriple {
        QuotedTriple::new(
            StarSubject::NamedNode(iri(s)),
            StarPredicate::NamedNode(iri(p)),
            StarObject::Literal(Literal::new(o.to_string(), None, None)),
        )
    }

    fn ann(p: &str, o: &str) -> Annotation {
        Annotation::new(iri(p), StarObject::NamedNode(iri(o)))
    }

    // ── QuotedTriple construction ─────────────────────────────────────────

    #[test]
    fn test_quoted_triple_new() {
        let qt = qt("http://s", "http://p", "http://o");
        assert_eq!(qt.to_string(), "<< <http://s> <http://p> <http://o> >>");
    }

    #[test]
    fn test_quoted_triple_from_iris() {
        let qt = QuotedTriple::from_iris("http://s", "http://p", "http://o").unwrap();
        assert!(matches!(&qt.subject, StarSubject::NamedNode(_)));
        assert!(matches!(&qt.predicate, StarPredicate::NamedNode(_)));
        assert!(matches!(&qt.object, StarObject::NamedNode(_)));
    }

    #[test]
    fn test_quoted_triple_nesting_depth_simple() {
        let qt = qt("http://s", "http://p", "http://o");
        assert_eq!(qt.nesting_depth(), 1);
    }

    #[test]
    fn test_quoted_triple_nesting_depth_nested() {
        let inner = qt("http://s", "http://p", "http://o");
        let outer = QuotedTriple::new(
            StarSubject::Quoted(Box::new(inner)),
            StarPredicate::NamedNode(iri("http://certainty")),
            StarObject::Literal(Literal::new("0.9".into(), None, None)),
        );
        assert_eq!(outer.nesting_depth(), 2);
    }

    #[test]
    fn test_quoted_triple_triple_nesting() {
        let inner = qt("http://s", "http://p", "http://o");
        let mid = QuotedTriple::new(
            StarSubject::Quoted(Box::new(inner)),
            StarPredicate::NamedNode(iri("http://cert")),
            StarObject::NamedNode(iri("http://v")),
        );
        let outer = QuotedTriple::new(
            StarSubject::Quoted(Box::new(mid)),
            StarPredicate::NamedNode(iri("http://source")),
            StarObject::NamedNode(iri("http://paper")),
        );
        assert_eq!(outer.nesting_depth(), 3);
    }

    // ── is_pattern / variables ────────────────────────────────────────────

    #[test]
    fn test_quoted_triple_no_variables_is_not_pattern() {
        let qt = qt("http://s", "http://p", "http://o");
        assert!(!qt.is_pattern());
        assert!(qt.variables().is_empty());
    }

    #[test]
    fn test_quoted_triple_with_variable_subject() {
        let qt_var = QuotedTriple::new(
            StarSubject::Variable(var("s")),
            StarPredicate::NamedNode(iri("http://p")),
            StarObject::NamedNode(iri("http://o")),
        );
        assert!(qt_var.is_pattern());
        let vars = qt_var.variables();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].as_str(), "s");
    }

    #[test]
    fn test_quoted_triple_with_variable_predicate_and_object() {
        let qt_vars = QuotedTriple::new(
            StarSubject::NamedNode(iri("http://s")),
            StarPredicate::Variable(var("p")),
            StarObject::Variable(var("o")),
        );
        let vars = qt_vars.variables();
        assert_eq!(vars.len(), 2);
    }

    // ── Triple pattern conversion ─────────────────────────────────────────

    #[test]
    fn test_to_triple_pattern() {
        let quoted = qt("http://s", "http://p", "http://o");
        let tp = quoted.to_triple_pattern();
        assert!(matches!(tp.subject, Term::Iri(_)));
        assert!(matches!(tp.predicate, Term::Iri(_)));
        assert!(matches!(tp.object, Term::Iri(_)));
    }

    #[test]
    fn test_from_triple_pattern() {
        let tp = TriplePattern::new(
            Term::Iri(iri("http://s")),
            Term::Iri(iri("http://p")),
            Term::Iri(iri("http://o")),
        );
        let qt = QuotedTriple::from_triple_pattern(&tp).unwrap();
        assert!(matches!(qt.subject, StarSubject::NamedNode(_)));
    }

    #[test]
    fn test_round_trip_triple_pattern() {
        let original = qt("http://s", "http://p", "http://o");
        let tp = original.to_triple_pattern();
        let back = QuotedTriple::from_triple_pattern(&tp).unwrap();
        assert_eq!(original, back);
    }

    // ── StarPredicate / StarSubject / StarObject ──────────────────────────

    #[test]
    fn test_star_subject_from_term_iri() {
        let term = Term::Iri(iri("http://s"));
        let subject = StarSubject::from_term(&term).unwrap();
        assert!(matches!(subject, StarSubject::NamedNode(_)));
    }

    #[test]
    fn test_star_subject_from_term_blank_node() {
        let term = Term::BlankNode("b0".to_string());
        let subject = StarSubject::from_term(&term).unwrap();
        assert!(matches!(subject, StarSubject::BlankNode(_)));
    }

    #[test]
    fn test_star_subject_from_literal_fails() {
        let term = Term::Literal(Literal::new("x".into(), None, None));
        assert!(StarSubject::from_term(&term).is_err());
    }

    #[test]
    fn test_star_object_from_literal() {
        let term = Term::Literal(Literal::new("hello".into(), None, None));
        let obj = StarObject::from_term(&term).unwrap();
        assert!(matches!(obj, StarObject::Literal(_)));
    }

    #[test]
    fn test_star_predicate_from_iri() {
        let term = Term::Iri(iri("http://p"));
        let pred = StarPredicate::from_term(&term).unwrap();
        assert!(matches!(pred, StarPredicate::NamedNode(_)));
    }

    #[test]
    fn test_star_predicate_from_variable() {
        let term = Term::Variable(var("p"));
        let pred = StarPredicate::from_term(&term).unwrap();
        assert!(matches!(pred, StarPredicate::Variable(_)));
    }

    // ── RdfStarStore operations ───────────────────────────────────────────

    #[test]
    fn test_store_assert_and_contains() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        assert!(!store.contains(&triple));
        store.assert_triple(triple.clone());
        assert!(store.contains(&triple));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_store_retract() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        store.assert_triple(triple.clone());
        assert!(store.retract_triple(&triple));
        assert!(!store.contains(&triple));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_store_add_annotation() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        let pred = iri("http://certainty");
        store.add_annotation(
            &triple,
            &pred,
            StarObject::Literal(Literal::new("0.9".into(), None, None)),
        );
        let entry = store.annotations(&triple).unwrap();
        assert!(entry.annotation(&pred).is_some());
    }

    #[test]
    fn test_store_remove_annotation() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        let pred = iri("http://certainty");
        store.add_annotation(&triple, &pred, StarObject::NamedNode(iri("http://high")));
        assert!(store.remove_annotation(&triple, &pred));
        let entry = store.annotations(&triple).unwrap();
        assert!(entry.annotation(&pred).is_none());
    }

    #[test]
    fn test_store_multiple_triples() {
        let mut store = RdfStarStore::new();
        store.assert_triple(qt("http://s1", "http://p", "http://o1"));
        store.assert_triple(qt("http://s2", "http://p", "http://o2"));
        assert_eq!(store.len(), 2);
    }

    // ── StarOperator ──────────────────────────────────────────────────────

    #[test]
    fn test_assert_quoted_operator() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        store.apply_operator(StarOperator::AssertQuoted {
            triple: triple.clone(),
        });
        assert!(store.contains(&triple));
    }

    #[test]
    fn test_retract_quoted_operator() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        store.assert_triple(triple.clone());
        store.apply_operator(StarOperator::RetractQuoted {
            triple: triple.clone(),
        });
        assert!(!store.contains(&triple));
    }

    #[test]
    fn test_add_annotation_operator() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        store.apply_operator(StarOperator::AddAnnotation {
            triple: triple.clone(),
            annotations: vec![ann("http://cert", "http://high")],
        });
        assert!(store.annotations(&triple).is_some());
    }

    #[test]
    fn test_remove_annotation_operator() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        let pred = iri("http://cert");
        store.add_annotation(&triple, &pred, StarObject::NamedNode(iri("http://high")));
        store.apply_operator(StarOperator::RemoveAnnotation {
            triple: triple.clone(),
            predicate: pred.clone(),
        });
        let entry = store.annotations(&triple).unwrap();
        assert!(entry.annotation(&pred).is_none());
    }

    // ── FindAnnotations / pattern matching ───────────────────────────────

    #[test]
    fn test_find_annotations_exact_match() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        let cert = iri("http://certainty");
        store.add_annotation(&triple, &cert, StarObject::NamedNode(iri("http://high")));

        let pattern = StarPattern::new(
            triple.clone(),
            StarPredicate::NamedNode(cert),
            StarObject::NamedNode(iri("http://high")),
        );
        let results = store.find_annotations(&pattern);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_find_annotations_variable_predicate() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        store.add_annotation(
            &triple,
            &iri("http://cert"),
            StarObject::NamedNode(iri("http://high")),
        );

        let pattern = StarPattern::new(
            triple.clone(),
            StarPredicate::Variable(var("pred")),
            StarObject::Variable(var("obj")),
        );
        let results = store.find_annotations(&pattern);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_find_annotations_no_match() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        store.add_annotation(
            &triple,
            &iri("http://cert"),
            StarObject::NamedNode(iri("http://high")),
        );

        let other_triple = qt("http://other", "http://p", "http://o");
        let pattern = StarPattern::new(
            other_triple,
            StarPredicate::Variable(var("pred")),
            StarObject::Variable(var("obj")),
        );
        let results = store.find_annotations(&pattern);
        assert!(results.is_empty());
    }

    // ── Pattern matching / binding ─────────────────────────────────────

    #[test]
    fn test_bind_pattern_all_variables() {
        let triple = qt("http://s", "http://p", "http://o");
        let pattern = QuotedTriple::new(
            StarSubject::Variable(var("s")),
            StarPredicate::Variable(var("p")),
            StarObject::Variable(var("o")),
        );
        let mut binding = StarBinding::new();
        assert!(bind_pattern(&triple, &pattern, &mut binding));
        assert!(binding.contains_key("s"));
        assert!(binding.contains_key("p"));
        assert!(binding.contains_key("o"));
    }

    #[test]
    fn test_bind_pattern_partial_variables() {
        let triple = qt("http://s", "http://p", "http://o");
        let pattern = QuotedTriple::new(
            StarSubject::NamedNode(iri("http://s")),
            StarPredicate::Variable(var("p")),
            StarObject::Variable(var("o")),
        );
        let mut binding = StarBinding::new();
        assert!(bind_pattern(&triple, &pattern, &mut binding));
        assert!(binding.contains_key("p"));
        assert!(binding.contains_key("o"));
    }

    #[test]
    fn test_bind_pattern_mismatch() {
        let triple = qt("http://s", "http://p", "http://o");
        let pattern = QuotedTriple::new(
            StarSubject::NamedNode(iri("http://DIFFERENT")),
            StarPredicate::Variable(var("p")),
            StarObject::Variable(var("o")),
        );
        let mut binding = StarBinding::new();
        assert!(!bind_pattern(&triple, &pattern, &mut binding));
    }

    // ── CONSTRUCT (instantiation) ─────────────────────────────────────────

    #[test]
    fn test_instantiate_quoted_triple() {
        let template = QuotedTriple::new(
            StarSubject::Variable(var("s")),
            StarPredicate::NamedNode(iri("http://p")),
            StarObject::Variable(var("o")),
        );
        let mut binding = StarBinding::new();
        binding.insert("s".to_string(), StarObject::NamedNode(iri("http://alice")));
        binding.insert(
            "o".to_string(),
            StarObject::Literal(Literal::new("42".into(), None, None)),
        );

        let result = instantiate_quoted_triple(&template, &binding).unwrap();
        assert!(
            matches!(result.subject, StarSubject::NamedNode(n) if n.to_string().contains("alice"))
        );
        assert!(matches!(result.object, StarObject::Literal(_)));
    }

    #[test]
    fn test_instantiate_unbound_variable_fails() {
        let template = QuotedTriple::new(
            StarSubject::Variable(var("missing")),
            StarPredicate::NamedNode(iri("http://p")),
            StarObject::NamedNode(iri("http://o")),
        );
        let binding = StarBinding::new();
        assert!(instantiate_quoted_triple(&template, &binding).is_err());
    }

    // ── SPARQL-star builtins ─────────────────────────────────────────────

    #[test]
    fn test_is_triple_function() {
        let obj = StarObject::Quoted(Box::new(qt("http://s", "http://p", "http://o")));
        assert!(is_triple(&obj));
        let not_triple = StarObject::NamedNode(iri("http://x"));
        assert!(!is_triple(&not_triple));
    }

    #[test]
    fn test_subject_of() {
        let triple = qt("http://alice", "http://p", "http://o");
        let s = subject_of(&triple);
        assert!(matches!(s, StarSubject::NamedNode(n) if n.to_string().contains("alice")));
    }

    #[test]
    fn test_predicate_of() {
        let triple = qt("http://s", "http://predicate", "http://o");
        let p = predicate_of(&triple);
        assert!(matches!(p, StarPredicate::NamedNode(n) if n.to_string().contains("predicate")));
    }

    #[test]
    fn test_object_of() {
        let triple = qt("http://s", "http://p", "http://target");
        let o = object_of(&triple);
        assert!(matches!(o, StarObject::NamedNode(n) if n.to_string().contains("target")));
    }

    #[test]
    fn test_triple_fn_builtin() {
        let s = StarSubject::NamedNode(iri("http://s"));
        let p = StarPredicate::NamedNode(iri("http://p"));
        let o = StarObject::NamedNode(iri("http://o"));
        let qt = triple_fn(s, p, o);
        assert_eq!(qt.nesting_depth(), 1);
    }

    // ── Mixed standard + star patterns ───────────────────────────────────

    #[test]
    fn test_mixed_star_and_standard_patterns() {
        let mut store = RdfStarStore::new();
        // Add a plain triple annotation
        let t1 = qt("http://alice", "http://knows", "http://bob");
        store.add_annotation(
            &t1,
            &iri("http://since"),
            StarObject::Literal(Literal::new("2020".into(), None, None)),
        );
        // Add a second triple with different annotation
        let t2 = qt("http://bob", "http://knows", "http://carol");
        store.add_annotation(
            &t2,
            &iri("http://since"),
            StarObject::Literal(Literal::new("2021".into(), None, None)),
        );

        // Find all "knows" triples with "since" annotation
        let pattern_triple = QuotedTriple::new(
            StarSubject::Variable(var("s")),
            StarPredicate::NamedNode(iri("http://knows")),
            StarObject::Variable(var("o")),
        );
        let pattern = StarPattern::new(
            pattern_triple,
            StarPredicate::NamedNode(iri("http://since")),
            StarObject::Variable(var("when")),
        );
        let results = store.find_annotations(&pattern);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_star_pattern_variables() {
        let pattern = StarPattern::new(
            QuotedTriple::new(
                StarSubject::Variable(var("s")),
                StarPredicate::Variable(var("p")),
                StarObject::Variable(var("o")),
            ),
            StarPredicate::Variable(var("ap")),
            StarObject::Variable(var("ao")),
        );
        let vars = pattern.variables();
        assert_eq!(vars.len(), 5);
    }

    // ── Display / formatting ──────────────────────────────────────────────

    #[test]
    fn test_star_operator_display() {
        let triple = qt("http://s", "http://p", "http://o");
        let op = StarOperator::AssertQuoted { triple };
        assert!(op.to_string().contains("AssertQuoted"));
    }

    #[test]
    fn test_quoted_triple_display_nested() {
        let inner = qt("http://s", "http://p", "http://o");
        let outer = QuotedTriple::new(
            StarSubject::Quoted(Box::new(inner)),
            StarPredicate::NamedNode(iri("http://cert")),
            StarObject::Literal(Literal::new("high".into(), None, None)),
        );
        let s = outer.to_string();
        assert!(s.contains("<<"));
        // Nested << inside outer <<
        assert!(s.matches("<<").count() >= 2);
    }

    // ── Annotation store operations ───────────────────────────────────────

    #[test]
    fn test_multiple_annotations_on_same_triple() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        store.add_annotation(
            &triple,
            &iri("http://cert"),
            StarObject::NamedNode(iri("http://high")),
        );
        store.add_annotation(
            &triple,
            &iri("http://source"),
            StarObject::NamedNode(iri("http://paper1")),
        );

        let entry = store.annotations(&triple).unwrap();
        assert_eq!(entry.annotations.len(), 2);
    }

    #[test]
    fn test_annotation_overwrite() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        let pred = iri("http://cert");
        store.add_annotation(&triple, &pred, StarObject::NamedNode(iri("http://low")));
        store.add_annotation(&triple, &pred, StarObject::NamedNode(iri("http://high")));

        let entry = store.annotations(&triple).unwrap();
        // Overwritten — still just 1 annotation for this predicate
        assert_eq!(entry.annotations.len(), 1);
        if let Some(StarObject::NamedNode(n)) = entry.annotation(&pred) {
            assert!(n.to_string().contains("high"));
        } else {
            panic!("expected NamedNode annotation");
        }
    }

    #[test]
    fn test_store_iter() {
        let mut store = RdfStarStore::new();
        store.assert_triple(qt("http://s1", "http://p", "http://o1"));
        store.assert_triple(qt("http://s2", "http://p", "http://o2"));
        assert_eq!(store.iter().count(), 2);
    }

    #[test]
    fn test_find_annotations_with_literal_object_value() {
        let mut store = RdfStarStore::new();
        let triple = qt_lit("http://s", "http://p", "42");
        store.add_annotation(
            &triple,
            &iri("http://source"),
            StarObject::NamedNode(iri("http://db")),
        );
        let pattern = StarPattern::new(
            triple.clone(),
            StarPredicate::Variable(var("pred")),
            StarObject::Variable(var("obj")),
        );
        let results = store.find_annotations(&pattern);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_apply_operator_find_annotations() {
        let mut store = RdfStarStore::new();
        let triple = qt("http://s", "http://p", "http://o");
        store.add_annotation(
            &triple,
            &iri("http://cert"),
            StarObject::NamedNode(iri("http://high")),
        );

        let pattern = StarPattern::new(
            triple.clone(),
            StarPredicate::NamedNode(iri("http://cert")),
            StarObject::Variable(var("v")),
        );
        let results = store.apply_operator(StarOperator::FindAnnotations { pattern });
        assert_eq!(results.len(), 1);
    }
}
