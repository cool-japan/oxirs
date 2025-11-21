//! N3 (Notation3) type definitions
//!
//! This module defines N3-specific types that extend RDF with:
//! - Formulas (graphs as first-class values)
//! - Variables (universally or existentially quantified)
//! - Built-in predicates for logic and math

use oxirs_core::model::{BlankNode, Literal, NamedNode, Triple};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Counter for generating unique formula IDs
static FORMULA_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// An N3 variable (universal or existential)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct N3Variable {
    /// Variable name (without the ? prefix)
    pub name: String,
    /// Whether this is a universal variable (forAll) or existential (forSome)
    pub universal: bool,
}

impl N3Variable {
    /// Create a new universal variable
    pub fn universal(name: &str) -> Self {
        Self {
            name: name.to_string(),
            universal: true,
        }
    }

    /// Create a new existential variable
    pub fn existential(name: &str) -> Self {
        Self {
            name: name.to_string(),
            universal: false,
        }
    }

    /// Get the variable name with ? prefix
    pub fn prefixed_name(&self) -> String {
        format!("?{}", self.name)
    }
}

impl fmt::Display for N3Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?{}", self.name)
    }
}

/// An N3 formula (a graph that can be used as a term)
#[derive(Debug, Clone, PartialEq)]
pub struct N3Formula {
    /// Unique identifier for this formula
    pub id: usize,
    /// Triples contained in this formula
    pub triples: Vec<N3Statement>,
    /// Universal variables quantified in this formula
    pub universals: Vec<N3Variable>,
    /// Existential variables quantified in this formula
    pub existentials: Vec<N3Variable>,
}

impl N3Formula {
    /// Create a new empty formula
    pub fn new() -> Self {
        Self {
            id: FORMULA_COUNTER.fetch_add(1, Ordering::SeqCst),
            triples: Vec::new(),
            universals: Vec::new(),
            existentials: Vec::new(),
        }
    }

    /// Create a formula with the given statements
    pub fn with_statements(statements: Vec<N3Statement>) -> Self {
        Self {
            id: FORMULA_COUNTER.fetch_add(1, Ordering::SeqCst),
            triples: statements,
            universals: Vec::new(),
            existentials: Vec::new(),
        }
    }

    /// Add a statement to the formula
    pub fn add_statement(&mut self, statement: N3Statement) {
        self.triples.push(statement);
    }

    /// Add a universal variable
    pub fn add_universal(&mut self, var: N3Variable) {
        self.universals.push(var);
    }

    /// Add an existential variable
    pub fn add_existential(&mut self, var: N3Variable) {
        self.existentials.push(var);
    }

    /// Check if this formula is empty
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Get the number of statements
    pub fn len(&self) -> usize {
        self.triples.len()
    }
}

impl Default for N3Formula {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for N3Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ ")?;
        for (i, stmt) in self.triples.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", stmt)?;
        }
        write!(f, " }}")
    }
}

/// An N3 term (extends RDF terms with variables and formulas)
#[derive(Debug, Clone, PartialEq)]
pub enum N3Term {
    /// Named node (IRI)
    NamedNode(NamedNode),
    /// Blank node
    BlankNode(BlankNode),
    /// Literal value
    Literal(Literal),
    /// Variable
    Variable(N3Variable),
    /// Formula (graph as a term)
    Formula(Box<N3Formula>),
}

impl N3Term {
    /// Check if this term is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, N3Term::Variable(_))
    }

    /// Check if this term is a formula
    pub fn is_formula(&self) -> bool {
        matches!(self, N3Term::Formula(_))
    }

    /// Try to convert to an RDF subject (fails for formulas and some literals)
    pub fn as_rdf_subject(&self) -> Option<oxirs_core::model::Subject> {
        match self {
            N3Term::NamedNode(n) => Some(oxirs_core::model::Subject::NamedNode(n.clone())),
            N3Term::BlankNode(b) => Some(oxirs_core::model::Subject::BlankNode(b.clone())),
            _ => None,
        }
    }

    /// Try to convert to an RDF predicate
    pub fn as_rdf_predicate(&self) -> Option<oxirs_core::model::Predicate> {
        match self {
            N3Term::NamedNode(n) => Some(oxirs_core::model::Predicate::NamedNode(n.clone())),
            _ => None,
        }
    }

    /// Try to convert to an RDF object
    pub fn as_rdf_object(&self) -> Option<oxirs_core::model::Object> {
        match self {
            N3Term::NamedNode(n) => Some(oxirs_core::model::Object::NamedNode(n.clone())),
            N3Term::BlankNode(b) => Some(oxirs_core::model::Object::BlankNode(b.clone())),
            N3Term::Literal(l) => Some(oxirs_core::model::Object::Literal(l.clone())),
            _ => None,
        }
    }
}

impl fmt::Display for N3Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            N3Term::NamedNode(n) => write!(f, "<{}>", n.as_str()),
            N3Term::BlankNode(b) => write!(f, "_:{}", b.as_str()),
            N3Term::Literal(l) => write!(f, "{}", l),
            N3Term::Variable(v) => write!(f, "{}", v),
            N3Term::Formula(formula) => write!(f, "{}", formula),
        }
    }
}

/// An N3 statement (extends RDF triples with variables and formulas)
#[derive(Debug, Clone, PartialEq)]
pub struct N3Statement {
    /// Subject
    pub subject: N3Term,
    /// Predicate
    pub predicate: N3Term,
    /// Object
    pub object: N3Term,
}

impl N3Statement {
    /// Create a new N3 statement
    pub fn new(subject: N3Term, predicate: N3Term, object: N3Term) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Try to convert to an RDF triple (fails if contains variables or formulas)
    pub fn as_rdf_triple(&self) -> Option<Triple> {
        let subject = self.subject.as_rdf_subject()?;
        let predicate = self.predicate.as_rdf_predicate()?;
        let object = self.object.as_rdf_object()?;
        Some(Triple::new(subject, predicate, object))
    }

    /// Check if this statement contains any variables
    pub fn has_variables(&self) -> bool {
        self.subject.is_variable() || self.predicate.is_variable() || self.object.is_variable()
    }

    /// Check if this statement contains any formulas
    pub fn has_formulas(&self) -> bool {
        self.subject.is_formula() || self.predicate.is_formula() || self.object.is_formula()
    }
}

impl fmt::Display for N3Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {} .", self.subject, self.predicate, self.object)
    }
}

/// N3 implication (rule)
#[derive(Debug, Clone, PartialEq)]
pub struct N3Implication {
    /// Antecedent (left side of =>)
    pub antecedent: N3Formula,
    /// Consequent (right side of =>)
    pub consequent: N3Formula,
}

impl N3Implication {
    /// Create a new implication
    pub fn new(antecedent: N3Formula, consequent: N3Formula) -> Self {
        Self {
            antecedent,
            consequent,
        }
    }
}

impl fmt::Display for N3Implication {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} => {}", self.antecedent, self.consequent)
    }
}

/// N3 built-in predicate categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum N3BuiltinCategory {
    /// Math operations (math:sum, math:product, etc.)
    Math,
    /// String operations (string:concatenation, etc.)
    String,
    /// List operations (list:member, etc.)
    List,
    /// Comparison operations (math:greaterThan, etc.)
    Comparison,
    /// Logical operations (log:implies, etc.)
    Logic,
    /// Cryptographic operations (crypto:sha, etc.)
    Crypto,
    /// Time operations (time:inSeconds, etc.)
    Time,
}

/// N3 built-in predicate
#[derive(Debug, Clone, PartialEq)]
pub struct N3Builtin {
    /// The IRI of the built-in
    pub iri: NamedNode,
    /// Category of the built-in
    pub category: N3BuiltinCategory,
    /// Human-readable name
    pub name: String,
    /// Number of arguments (None for variadic)
    pub arity: Option<usize>,
}

impl N3Builtin {
    /// Create a new built-in definition
    pub fn new(
        iri: NamedNode,
        category: N3BuiltinCategory,
        name: &str,
        arity: Option<usize>,
    ) -> Self {
        Self {
            iri,
            category,
            name: name.to_string(),
            arity,
        }
    }
}

/// Registry of N3 built-in predicates
pub struct N3BuiltinRegistry {
    /// Map from IRI to built-in definition
    builtins: HashMap<String, N3Builtin>,
}

impl N3BuiltinRegistry {
    /// Create a new registry with standard built-ins
    pub fn new() -> Self {
        let mut builtins = HashMap::new();

        // Math built-ins
        let math_ns = "http://www.w3.org/2000/10/swap/math#";
        Self::add_builtin(
            &mut builtins,
            math_ns,
            "sum",
            N3BuiltinCategory::Math,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            math_ns,
            "difference",
            N3BuiltinCategory::Math,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            math_ns,
            "product",
            N3BuiltinCategory::Math,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            math_ns,
            "quotient",
            N3BuiltinCategory::Math,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            math_ns,
            "greaterThan",
            N3BuiltinCategory::Comparison,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            math_ns,
            "lessThan",
            N3BuiltinCategory::Comparison,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            math_ns,
            "equalTo",
            N3BuiltinCategory::Comparison,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            math_ns,
            "notEqualTo",
            N3BuiltinCategory::Comparison,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            math_ns,
            "absoluteValue",
            N3BuiltinCategory::Math,
            Some(1),
        );
        Self::add_builtin(
            &mut builtins,
            math_ns,
            "negation",
            N3BuiltinCategory::Math,
            Some(1),
        );

        // String built-ins
        let string_ns = "http://www.w3.org/2000/10/swap/string#";
        Self::add_builtin(
            &mut builtins,
            string_ns,
            "concatenation",
            N3BuiltinCategory::String,
            None,
        );
        Self::add_builtin(
            &mut builtins,
            string_ns,
            "contains",
            N3BuiltinCategory::String,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            string_ns,
            "startsWith",
            N3BuiltinCategory::String,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            string_ns,
            "endsWith",
            N3BuiltinCategory::String,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            string_ns,
            "matches",
            N3BuiltinCategory::String,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            string_ns,
            "length",
            N3BuiltinCategory::String,
            Some(1),
        );

        // List built-ins
        let list_ns = "http://www.w3.org/2000/10/swap/list#";
        Self::add_builtin(
            &mut builtins,
            list_ns,
            "member",
            N3BuiltinCategory::List,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            list_ns,
            "append",
            N3BuiltinCategory::List,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            list_ns,
            "first",
            N3BuiltinCategory::List,
            Some(1),
        );
        Self::add_builtin(
            &mut builtins,
            list_ns,
            "rest",
            N3BuiltinCategory::List,
            Some(1),
        );
        Self::add_builtin(
            &mut builtins,
            list_ns,
            "length",
            N3BuiltinCategory::List,
            Some(1),
        );

        // Logic built-ins
        let log_ns = "http://www.w3.org/2000/10/swap/log#";
        Self::add_builtin(
            &mut builtins,
            log_ns,
            "implies",
            N3BuiltinCategory::Logic,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            log_ns,
            "includes",
            N3BuiltinCategory::Logic,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            log_ns,
            "notIncludes",
            N3BuiltinCategory::Logic,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            log_ns,
            "equalTo",
            N3BuiltinCategory::Logic,
            Some(2),
        );
        Self::add_builtin(
            &mut builtins,
            log_ns,
            "notEqualTo",
            N3BuiltinCategory::Logic,
            Some(2),
        );

        // Crypto built-ins
        let crypto_ns = "http://www.w3.org/2000/10/swap/crypto#";
        Self::add_builtin(
            &mut builtins,
            crypto_ns,
            "sha",
            N3BuiltinCategory::Crypto,
            Some(1),
        );
        Self::add_builtin(
            &mut builtins,
            crypto_ns,
            "md5",
            N3BuiltinCategory::Crypto,
            Some(1),
        );

        // Time built-ins
        let time_ns = "http://www.w3.org/2000/10/swap/time#";
        Self::add_builtin(
            &mut builtins,
            time_ns,
            "inSeconds",
            N3BuiltinCategory::Time,
            Some(1),
        );
        Self::add_builtin(
            &mut builtins,
            time_ns,
            "year",
            N3BuiltinCategory::Time,
            Some(1),
        );
        Self::add_builtin(
            &mut builtins,
            time_ns,
            "month",
            N3BuiltinCategory::Time,
            Some(1),
        );
        Self::add_builtin(
            &mut builtins,
            time_ns,
            "day",
            N3BuiltinCategory::Time,
            Some(1),
        );

        Self { builtins }
    }

    fn add_builtin(
        builtins: &mut HashMap<String, N3Builtin>,
        namespace: &str,
        name: &str,
        category: N3BuiltinCategory,
        arity: Option<usize>,
    ) {
        let iri_str = format!("{}{}", namespace, name);
        if let Ok(iri) = NamedNode::new(&iri_str) {
            let builtin = N3Builtin::new(iri, category, name, arity);
            builtins.insert(iri_str, builtin);
        }
    }

    /// Check if an IRI is a built-in predicate
    pub fn is_builtin(&self, iri: &str) -> bool {
        self.builtins.contains_key(iri)
    }

    /// Get a built-in by IRI
    pub fn get(&self, iri: &str) -> Option<&N3Builtin> {
        self.builtins.get(iri)
    }

    /// Get all built-ins
    pub fn all(&self) -> impl Iterator<Item = &N3Builtin> {
        self.builtins.values()
    }
}

impl Default for N3BuiltinRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_creation() {
        let univ = N3Variable::universal("x");
        assert!(univ.universal);
        assert_eq!(univ.name, "x");
        assert_eq!(univ.prefixed_name(), "?x");

        let exist = N3Variable::existential("y");
        assert!(!exist.universal);
        assert_eq!(exist.prefixed_name(), "?y");
    }

    #[test]
    fn test_formula_creation() {
        let formula = N3Formula::new();
        assert!(formula.is_empty());
        assert_eq!(formula.len(), 0);

        let subject = N3Term::Variable(N3Variable::universal("x"));
        let predicate = N3Term::NamedNode(NamedNode::new("http://example.org/knows").unwrap());
        let object = N3Term::Variable(N3Variable::universal("y"));
        let stmt = N3Statement::new(subject, predicate, object);

        let formula = N3Formula::with_statements(vec![stmt]);
        assert!(!formula.is_empty());
        assert_eq!(formula.len(), 1);
    }

    #[test]
    fn test_n3_term_conversion() {
        let named = N3Term::NamedNode(NamedNode::new("http://example.org/resource").unwrap());
        assert!(named.as_rdf_subject().is_some());
        assert!(named.as_rdf_predicate().is_some());
        assert!(named.as_rdf_object().is_some());

        let var = N3Term::Variable(N3Variable::universal("x"));
        assert!(var.as_rdf_subject().is_none());
        assert!(var.is_variable());
    }

    #[test]
    fn test_statement_has_variables() {
        let subject = N3Term::Variable(N3Variable::universal("x"));
        let predicate = N3Term::NamedNode(NamedNode::new("http://example.org/knows").unwrap());
        let object = N3Term::NamedNode(NamedNode::new("http://example.org/person").unwrap());
        let stmt = N3Statement::new(subject, predicate, object);

        assert!(stmt.has_variables());
        assert!(!stmt.has_formulas());
    }

    #[test]
    fn test_builtin_registry() {
        let registry = N3BuiltinRegistry::new();

        assert!(registry.is_builtin("http://www.w3.org/2000/10/swap/math#sum"));
        assert!(registry.is_builtin("http://www.w3.org/2000/10/swap/string#concatenation"));
        assert!(!registry.is_builtin("http://example.org/notBuiltin"));

        let sum = registry.get("http://www.w3.org/2000/10/swap/math#sum");
        assert!(sum.is_some());
        assert_eq!(sum.unwrap().name, "sum");
        assert_eq!(sum.unwrap().category, N3BuiltinCategory::Math);
    }

    #[test]
    fn test_implication() {
        let antecedent = N3Formula::new();
        let consequent = N3Formula::new();
        let impl_rule = N3Implication::new(antecedent, consequent);

        assert!(impl_rule.antecedent.is_empty());
        assert!(impl_rule.consequent.is_empty());
    }
}
