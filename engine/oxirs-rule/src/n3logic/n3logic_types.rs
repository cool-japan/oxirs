//! # N3 Logic Core Types
//!
//! Core data types for Notation3 logic: rules, formulas, terms, built-ins,
//! ground triples, and variable bindings.

use std::collections::HashMap;
use std::fmt;

// ── Public types ──────────────────────────────────────────────────────────────

/// A Notation3 rule: antecedent (LHS) implies consequent (RHS).
#[derive(Debug, Clone, PartialEq)]
pub struct N3Rule {
    pub antecedent: Vec<N3Formula>,
    pub consequent: Vec<N3Formula>,
    pub universals: Vec<String>,
    pub existentials: Vec<String>,
}

impl N3Rule {
    pub fn new(antecedent: Vec<N3Formula>, consequent: Vec<N3Formula>) -> Self {
        Self {
            antecedent,
            consequent,
            universals: Vec::new(),
            existentials: Vec::new(),
        }
    }

    pub fn with_universals(mut self, vars: Vec<String>) -> Self {
        self.universals = vars;
        self
    }

    pub fn with_existentials(mut self, vars: Vec<String>) -> Self {
        self.existentials = vars;
        self
    }
}

impl fmt::Display for N3Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "N3Rule({} antecedent(s) => {} consequent(s))",
            self.antecedent.len(),
            self.consequent.len()
        )
    }
}

/// An N3 formula: triple, nested graph, or built-in.
#[derive(Debug, Clone, PartialEq)]
pub enum N3Formula {
    Triple {
        subject: N3Term,
        predicate: N3Term,
        object: N3Term,
    },
    Graph(Vec<N3Formula>),
    BuiltIn(N3BuiltIn),
}

/// An N3 term.
#[derive(Debug, Clone, PartialEq)]
pub enum N3Term {
    Iri(String),
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    BlankNode(String),
    Variable(String),
    Universal(String),
    NestedFormula(Box<Vec<N3Formula>>),
}

impl N3Term {
    pub fn value_str(&self) -> Option<&str> {
        match self {
            N3Term::Iri(s) | N3Term::BlankNode(s) | N3Term::Variable(s) | N3Term::Universal(s) => {
                Some(s.as_str())
            }
            N3Term::Literal { value, .. } => Some(value.as_str()),
            N3Term::NestedFormula(_) => None,
        }
    }

    pub fn is_variable(&self) -> bool {
        matches!(self, N3Term::Variable(_) | N3Term::Universal(_))
    }
}

impl fmt::Display for N3Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            N3Term::Iri(s) => write!(f, "<{}>", s),
            N3Term::Literal {
                value,
                datatype,
                lang,
            } => {
                write!(f, "\"{}\"", value)?;
                if let Some(dt) = datatype {
                    write!(f, "^^<{}>", dt)?;
                }
                if let Some(l) = lang {
                    write!(f, "@{}", l)?;
                }
                Ok(())
            }
            N3Term::BlankNode(s) => write!(f, "_:{}", s),
            N3Term::Variable(s) => write!(f, "?{}", s),
            N3Term::Universal(s) => write!(f, "!{}", s),
            N3Term::NestedFormula(_) => write!(f, "{{ ... }}"),
        }
    }
}

/// N3 built-in operations.
#[derive(Debug, Clone, PartialEq)]
pub enum N3BuiltIn {
    MathSum {
        args: Vec<N3Term>,
        result: N3Term,
    },
    MathDifference {
        args: Vec<N3Term>,
        result: N3Term,
    },
    MathProduct {
        args: Vec<N3Term>,
        result: N3Term,
    },
    MathQuotient {
        args: Vec<N3Term>,
        result: N3Term,
    },
    MathGreaterThan {
        left: N3Term,
        right: N3Term,
    },
    MathLessThan {
        left: N3Term,
        right: N3Term,
    },
    MathEqualTo {
        left: N3Term,
        right: N3Term,
    },
    StringConcatenation {
        args: Vec<N3Term>,
        result: N3Term,
    },
    StringLength {
        input: N3Term,
        result: N3Term,
    },
    StringContains {
        subject: N3Term,
        substring: N3Term,
    },
    LogImplies {
        antecedent: Box<N3Formula>,
        consequent: Box<N3Formula>,
    },
    LogConcludes {
        graph: N3Term,
        formula: Box<N3Formula>,
    },
    LogEqual {
        left: N3Term,
        right: N3Term,
    },
    LogNotEqual {
        left: N3Term,
        right: N3Term,
    },
}

/// A ground RDF triple.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            subject: s.into(),
            predicate: p.into(),
            object: o.into(),
        }
    }
}

/// Variable binding map.
pub type Bindings = HashMap<String, N3Term>;
