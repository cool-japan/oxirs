//! Minimal SPARQL Algebra stub for testing compilation

use std::collections::HashMap;

/// Variable identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable {
    pub name: String,
}

/// IRI (Internationalized Resource Identifier)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Iri {
    pub value: String,
}

/// Literal value
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal {
    pub value: String,
    pub language: Option<String>,
    pub datatype: Option<Iri>,
}

/// RDF term
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    Variable(Variable),
    Iri(Iri),
    Literal(Literal),
}

/// Triple pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TriplePattern {
    pub subject: Term,
    pub predicate: Term,
    pub object: Term,
}

/// SPARQL algebra expressions
#[derive(Debug, Clone)]
pub enum Algebra {
    Bgp(Vec<TriplePattern>),
    Join { left: Box<Algebra>, right: Box<Algebra> },
    Union { left: Box<Algebra>, right: Box<Algebra> },
}

/// SPARQL expression for filters and other operations
#[derive(Debug, Clone)]
pub enum Expression {
    Term(Term),
    And(Box<Expression>, Box<Expression>),
    Or(Box<Expression>, Box<Expression>),
    Not(Box<Expression>),
}

/// Solution - a set of variable bindings
pub type Solution = Vec<HashMap<Variable, Term>>;