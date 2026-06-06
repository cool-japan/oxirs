//! # Datalog Semi-Naive Evaluation Engine
//!
//! This module provides a complete, production-quality Datalog evaluation engine
//! implementing the semi-naive bottom-up fixpoint evaluation algorithm.
//!
//! ## Design
//!
//! - **EDB** (Extensional Database) — base facts given as input
//! - **IDB** (Intensional Database) — facts derived by applying rules
//! - **Semi-naive evaluation** — tracks only *newly derived* (delta) facts per
//!   iteration, ensuring that each rule fires at least once using a new fact,
//!   which avoids redundant computation compared to naive evaluation
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::datalog::{DatalogProgram, SemiNaiveEvaluator};
//! use oxirs_rule::datalog::parser::parse_program;
//!
//! let src = r#"
//!     parent(alice, bob).
//!     parent(bob, carol).
//!     ancestor(X, Y) :- parent(X, Y).
//!     ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
//! "#;
//!
//! let program = parse_program(src).expect("parse");
//! let evaluator = SemiNaiveEvaluator::new();
//! let db = evaluator.evaluate(&program).expect("evaluate");
//!
//! // ancestor(alice, carol) should be derived
//! let alice_carol = vec![
//!     oxirs_rule::datalog::DatalogValue::Str("alice".to_string()),
//!     oxirs_rule::datalog::DatalogValue::Str("carol".to_string()),
//! ];
//! assert!(db.contains("ancestor", &alice_carol));
//! ```

pub mod evaluator;
pub mod parser;
pub mod tests;

use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors that can arise during Datalog processing.
#[derive(Debug, Error, Clone)]
pub enum DatalogError {
    #[error("parse error: {0}")]
    ParseError(String),
    #[error("evaluation error: {0}")]
    EvaluationError(String),
    #[error("stratification error (cyclic negation detected): {0}")]
    StratificationError(String),
}

/// A ground value that can appear as an argument in a Datalog fact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DatalogValue {
    /// String constant (quoted or unquoted lowercase identifier)
    Str(String),
    /// Integer constant
    Int(i64),
    /// Boolean constant
    Bool(bool),
}

impl std::fmt::Display for DatalogValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatalogValue::Str(s) => write!(f, "{s}"),
            DatalogValue::Int(i) => write!(f, "{i}"),
            DatalogValue::Bool(b) => write!(f, "{b}"),
        }
    }
}

/// A term in a Datalog atom — either a variable or a ground constant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DatalogTerm {
    /// Variable: identifier starting with an uppercase letter
    Variable(String),
    /// Constant: ground value
    Constant(DatalogValue),
}

impl std::fmt::Display for DatalogTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatalogTerm::Variable(v) => write!(f, "{v}"),
            DatalogTerm::Constant(c) => write!(f, "{c}"),
        }
    }
}

/// An atom in a Datalog rule: `predicate(t1, t2, ...)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DatalogAtom {
    pub predicate: String,
    pub terms: Vec<DatalogTerm>,
}

impl std::fmt::Display for DatalogAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let terms: Vec<String> = self.terms.iter().map(|t| t.to_string()).collect();
        write!(f, "{}({})", self.predicate, terms.join(", "))
    }
}

/// A ground atom (all constants): `predicate(v1, v2, ...)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DatalogFact {
    pub predicate: String,
    pub args: Vec<DatalogValue>,
}

impl std::fmt::Display for DatalogFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let args: Vec<String> = self.args.iter().map(|a| a.to_string()).collect();
        write!(f, "{}({}).", self.predicate, args.join(", "))
    }
}

/// A Datalog rule: `head :- body1, body2, ...`
#[derive(Debug, Clone)]
pub struct DatalogRule {
    pub head: DatalogAtom,
    pub body: Vec<DatalogAtom>,
}

impl std::fmt::Display for DatalogRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.body.is_empty() {
            write!(f, "{}.", self.head)
        } else {
            let body: Vec<String> = self.body.iter().map(|a| a.to_string()).collect();
            write!(f, "{} :- {}.", self.head, body.join(", "))
        }
    }
}

/// A Datalog program: set of rules + EDB (ground facts).
///
/// The EDB is immutable input; the IDB is derived by evaluation.
#[derive(Debug, Clone, Default)]
pub struct DatalogProgram {
    /// Rules (may be recursive)
    pub rules: Vec<DatalogRule>,
    /// Extensional database: ground facts given as input
    pub edb: Vec<DatalogFact>,
}

impl DatalogProgram {
    /// Create an empty program.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rule.
    pub fn add_rule(&mut self, rule: DatalogRule) {
        self.rules.push(rule);
    }

    /// Add a ground EDB fact.
    pub fn add_fact(&mut self, fact: DatalogFact) {
        self.edb.push(fact);
    }

    /// Return all IDB predicates (predicates that appear as rule heads).
    pub fn idb_predicates(&self) -> HashSet<String> {
        self.rules
            .iter()
            .map(|r| r.head.predicate.clone())
            .collect()
    }

    /// Return all EDB predicates (predicates that appear only in EDB facts, not as heads).
    pub fn edb_predicates(&self) -> HashSet<String> {
        let idb = self.idb_predicates();
        self.edb
            .iter()
            .map(|f| f.predicate.clone())
            .filter(|p| !idb.contains(p))
            .collect()
    }
}

/// A fact database mapping predicate names to sets of ground argument tuples.
///
/// The inner `HashSet<Vec<DatalogValue>>` holds all ground argument tuples
/// known to hold for that predicate.
#[derive(Debug, Clone, Default)]
pub struct FactDatabase {
    inner: HashMap<String, HashSet<Vec<DatalogValue>>>,
}

impl FactDatabase {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a ground tuple under the given predicate.
    /// Returns `true` if the tuple was newly inserted, `false` if it was already present.
    pub fn insert(&mut self, predicate: &str, args: Vec<DatalogValue>) -> bool {
        self.inner
            .entry(predicate.to_string())
            .or_default()
            .insert(args)
    }

    /// Insert from a `DatalogFact`.
    pub fn insert_fact(&mut self, fact: &DatalogFact) -> bool {
        self.insert(&fact.predicate, fact.args.clone())
    }

    /// Check whether a ground tuple is present.
    pub fn contains(&self, predicate: &str, args: &[DatalogValue]) -> bool {
        self.inner
            .get(predicate)
            .is_some_and(|set| set.contains(args))
    }

    /// Iterate all tuples for a predicate.
    pub fn tuples_for(&self, predicate: &str) -> impl Iterator<Item = &Vec<DatalogValue>> {
        static EMPTY: std::sync::OnceLock<HashSet<Vec<DatalogValue>>> = std::sync::OnceLock::new();
        let empty = EMPTY.get_or_init(HashSet::new);
        self.inner.get(predicate).unwrap_or(empty).iter()
    }

    /// Return all predicates in the database.
    pub fn predicates(&self) -> impl Iterator<Item = &String> {
        self.inner.keys()
    }

    /// Total number of ground facts across all predicates.
    pub fn len(&self) -> usize {
        self.inner.values().map(|s| s.len()).sum()
    }

    /// Returns `true` if no facts are stored.
    pub fn is_empty(&self) -> bool {
        self.inner.values().all(|s| s.is_empty())
    }

    /// Return the tuples for a predicate as a set reference (may be empty).
    pub fn get_set(&self, predicate: &str) -> Option<&HashSet<Vec<DatalogValue>>> {
        self.inner.get(predicate)
    }

    /// Merge another `FactDatabase` into this one.
    /// Returns the number of newly inserted tuples.
    pub fn merge(&mut self, other: &FactDatabase) -> usize {
        let mut count = 0;
        for (pred, tuples) in &other.inner {
            let entry = self.inner.entry(pred.clone()).or_default();
            for tuple in tuples {
                if entry.insert(tuple.clone()) {
                    count += 1;
                }
            }
        }
        count
    }
}

/// Variable binding (substitution map): variable name → ground value.
pub type Substitution = HashMap<String, DatalogValue>;

/// Re-exports for convenience.
pub use evaluator::SemiNaiveEvaluator;
pub use parser::{parse_atom, parse_program, parse_rule};
