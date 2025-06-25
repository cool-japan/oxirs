//! Core RDF term types and implementations

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashSet;
use regex::Regex;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use crate::model::{RdfTerm, SubjectTerm, ObjectTerm, NamedNode, Literal};
use crate::OxirsError;

lazy_static! {
    /// Regex for validating blank node IDs according to Turtle/N-Triples specification
    static ref BLANK_NODE_REGEX: Regex = Regex::new(
        r"^[a-zA-Z_][a-zA-Z0-9_.-]*$"
    ).expect("Blank node regex compilation failed");
    
    /// Regex for validating SPARQL variable names according to SPARQL 1.1 specification
    static ref VARIABLE_REGEX: Regex = Regex::new(
        r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    ).expect("Variable regex compilation failed");
    
    /// Global counter for unique blank node generation
    static ref BLANK_NODE_COUNTER: AtomicU64 = AtomicU64::new(0);
    
    /// Global set for collision detection (using thread-safe wrapper)
    static ref BLANK_NODE_IDS: std::sync::Mutex<HashSet<String>> = std::sync::Mutex::new(HashSet::new());
}

/// Validates a blank node identifier according to RDF specifications
fn validate_blank_node_id(id: &str) -> Result<(), OxirsError> {
    if id.is_empty() {
        return Err(OxirsError::Parse("Blank node ID cannot be empty".to_string()));
    }
    
    // Remove _: prefix if present for validation
    let clean_id = if id.starts_with("_:") {
        &id[2..]
    } else {
        id
    };
    
    if clean_id.is_empty() {
        return Err(OxirsError::Parse("Blank node ID cannot be just '_:'".to_string()));
    }
    
    if !BLANK_NODE_REGEX.is_match(clean_id) {
        return Err(OxirsError::Parse(format!(
            "Invalid blank node ID format: '{}'. Must match [a-zA-Z0-9_][a-zA-Z0-9_.-]*", 
            clean_id
        )));
    }
    
    Ok(())
}

/// Validates a SPARQL variable name according to SPARQL 1.1 specification
fn validate_variable_name(name: &str) -> Result<(), OxirsError> {
    if name.is_empty() {
        return Err(OxirsError::Parse("Variable name cannot be empty".to_string()));
    }
    
    // Remove ? or $ prefix if present for validation
    let clean_name = if name.starts_with('?') || name.starts_with('$') {
        &name[1..]
    } else {
        name
    };
    
    if clean_name.is_empty() {
        return Err(OxirsError::Parse("Variable name cannot be just '?' or '$'".to_string()));
    }
    
    if !VARIABLE_REGEX.is_match(clean_name) {
        return Err(OxirsError::Parse(format!(
            "Invalid variable name format: '{}'. Must match [a-zA-Z_][a-zA-Z0-9_]*", 
            clean_name
        )));
    }
    
    // Check for reserved keywords
    match clean_name.to_lowercase().as_str() {
        "select" | "where" | "from" | "order" | "group" | "having" | "limit" | "offset" |
        "distinct" | "reduced" | "construct" | "describe" | "ask" | "union" | "optional" |
        "filter" | "bind" | "values" | "graph" | "service" | "minus" | "exists" | "not" => {
            return Err(OxirsError::Parse(format!(
                "Variable name '{}' is a reserved SPARQL keyword", 
                clean_name
            )));
        }
        _ => {}
    }
    
    Ok(())
}

/// A blank node identifier
/// 
/// Blank nodes are local identifiers used in RDF graphs that don't have global meaning.
/// Uses Arc<str> for efficient sharing and thread-safe collision detection.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlankNode {
    id: String,
}

impl Default for BlankNode {
    fn default() -> Self {
        Self::new_unique()
    }
}

impl BlankNode {
    /// Creates a new blank node with the given identifier
    /// 
    /// # Arguments
    /// * `id` - The blank node identifier (with or without _: prefix)
    /// 
    /// # Errors
    /// Returns an error if the ID format is invalid according to RDF specifications
    pub fn new(id: impl Into<String>) -> Result<Self, OxirsError> {
        let id = id.into();
        validate_blank_node_id(&id)?;
        
        // Ensure ID has _: prefix
        let normalized_id = if id.starts_with("_:") {
            id
        } else {
            format!("_:{}", id)
        };
        
        Ok(BlankNode { id: normalized_id })
    }
    
    /// Creates a new blank node without validation
    /// 
    /// # Safety
    /// The caller must ensure the ID is valid and properly formatted
    pub fn new_unchecked(id: impl Into<String>) -> Self {
        BlankNode { id: id.into() }
    }
    
    /// Generates a new unique blank node with collision detection
    /// 
    /// This method ensures global uniqueness across all threads and sessions
    pub fn new_unique() -> Self {
        loop {
            let counter = BLANK_NODE_COUNTER.fetch_add(1, Ordering::SeqCst);
            let id = format!("_:b{}", counter);
            
            // Check for collision (though unlikely with atomic counter)
            if let Ok(mut ids) = BLANK_NODE_IDS.lock() {
                if ids.insert(id.clone()) {
                    return BlankNode::new_unchecked(id);
                }
            } else {
                // If mutex is poisoned, still return the ID
                return BlankNode::new_unchecked(id);
            }
        }
    }
    
    /// Generates a new unique blank node with a custom prefix
    pub fn new_unique_with_prefix(prefix: &str) -> Result<Self, OxirsError> {
        // Validate prefix
        if !BLANK_NODE_REGEX.is_match(prefix) {
            return Err(OxirsError::Parse(format!(
                "Invalid blank node prefix: '{}'. Must match [a-zA-Z0-9_][a-zA-Z0-9_.-]*", 
                prefix
            )));
        }
        
        loop {
            let counter = BLANK_NODE_COUNTER.fetch_add(1, Ordering::SeqCst);
            let id = format!("_:{}_{}", prefix, counter);
            
            if let Ok(mut ids) = BLANK_NODE_IDS.lock() {
                if ids.insert(id.clone()) {
                    return Ok(BlankNode::new_unchecked(id));
                }
            } else {
                return Ok(BlankNode::new_unchecked(id));
            }
        }
    }
    
    /// Returns the blank node identifier
    pub fn id(&self) -> &str {
        &self.id
    }
    
    /// Returns the blank node identifier as a string slice
    pub fn as_str(&self) -> &str {
        &self.id
    }
    
    
    /// Returns the blank node identifier without the _: prefix
    pub fn local_id(&self) -> &str {
        if self.id.starts_with("_:") {
            &self.id[2..]
        } else {
            &self.id
        }
    }
}

impl fmt::Display for BlankNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl RdfTerm for BlankNode {
    fn as_str(&self) -> &str {
        &self.id
    }
    
    fn is_blank_node(&self) -> bool {
        true
    }
}

impl SubjectTerm for BlankNode {}
impl ObjectTerm for BlankNode {}

/// A SPARQL variable
/// 
/// Variables are used in SPARQL queries and updates to represent unknown values.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Variable {
    name: String,
}

impl Variable {
    /// Creates a new variable with the given name
    /// 
    /// # Arguments  
    /// * `name` - The variable name (with or without ? or $ prefix)
    /// 
    /// # Errors
    /// Returns an error if the name format is invalid according to SPARQL 1.1 specification
    pub fn new(name: impl Into<String>) -> Result<Self, OxirsError> {
        let name = name.into();
        validate_variable_name(&name)?;
        
        // Store name without prefix for consistency
        let clean_name = if name.starts_with('?') || name.starts_with('$') {
            name[1..].to_string()
        } else {
            name
        };
        
        Ok(Variable { name: clean_name })
    }
    
    /// Creates a new variable without validation
    /// 
    /// # Safety
    /// The caller must ensure the name is valid according to SPARQL rules
    pub fn new_unchecked(name: impl Into<String>) -> Self {
        Variable { name: name.into() }
    }
    
    /// Returns the variable name (without prefix)
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Returns the variable name as a string slice (without prefix)
    pub fn as_str(&self) -> &str {
        &self.name
    }
    
    
    /// Returns the variable name with ? prefix for SPARQL syntax
    pub fn with_prefix(&self) -> String {
        format!("?{}", self.name)
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "?{}", self.name)
    }
}

impl RdfTerm for Variable {
    fn as_str(&self) -> &str {
        &self.name
    }
    
    fn is_variable(&self) -> bool {
        true
    }
}

/// Union type for all RDF terms
/// 
/// This enum can hold any type of RDF term and is used when the specific
/// type is not known at compile time.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Term {
    NamedNode(NamedNode),
    BlankNode(BlankNode),
    Literal(Literal),
    Variable(Variable),
}

impl Term {
    /// Returns true if this is a named node
    pub fn is_named_node(&self) -> bool {
        matches!(self, Term::NamedNode(_))
    }
    
    /// Returns true if this is a blank node
    pub fn is_blank_node(&self) -> bool {
        matches!(self, Term::BlankNode(_))
    }
    
    /// Returns true if this is a literal
    pub fn is_literal(&self) -> bool {
        matches!(self, Term::Literal(_))
    }
    
    /// Returns true if this is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, Term::Variable(_))
    }
    
    /// Returns the named node if this term is a named node
    pub fn as_named_node(&self) -> Option<&NamedNode> {
        match self {
            Term::NamedNode(n) => Some(n),
            _ => None,
        }
    }
    
    /// Returns the blank node if this term is a blank node
    pub fn as_blank_node(&self) -> Option<&BlankNode> {
        match self {
            Term::BlankNode(b) => Some(b),
            _ => None,
        }
    }
    
    /// Returns the literal if this term is a literal
    pub fn as_literal(&self) -> Option<&Literal> {
        match self {
            Term::Literal(l) => Some(l),
            _ => None,
        }
    }
    
    /// Returns the variable if this term is a variable
    pub fn as_variable(&self) -> Option<&Variable> {
        match self {
            Term::Variable(v) => Some(v),
            _ => None,
        }
    }
    
    /// Convert a Subject to a Term
    pub fn from_subject(subject: &crate::model::Subject) -> Term {
        match subject {
            crate::model::Subject::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Subject::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Subject::Variable(v) => Term::Variable(v.clone()),
        }
    }
    
    /// Convert a Predicate to a Term
    pub fn from_predicate(predicate: &crate::model::Predicate) -> Term {
        match predicate {
            crate::model::Predicate::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Predicate::Variable(v) => Term::Variable(v.clone()),
        }
    }
    
    /// Convert an Object to a Term
    pub fn from_object(object: &crate::model::Object) -> Term {
        match object {
            crate::model::Object::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Object::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Object::Literal(l) => Term::Literal(l.clone()),
            crate::model::Object::Variable(v) => Term::Variable(v.clone()),
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::NamedNode(n) => write!(f, "{}", n),
            Term::BlankNode(b) => write!(f, "{}", b),
            Term::Literal(l) => write!(f, "{}", l),
            Term::Variable(v) => write!(f, "{}", v),
        }
    }
}

impl RdfTerm for Term {
    fn as_str(&self) -> &str {
        match self {
            Term::NamedNode(n) => n.as_str(),
            Term::BlankNode(b) => b.as_str(),
            Term::Literal(l) => l.as_str(),
            Term::Variable(v) => v.as_str(),
        }
    }
    
    fn is_named_node(&self) -> bool {
        self.is_named_node()
    }
    
    fn is_blank_node(&self) -> bool {
        self.is_blank_node()
    }
    
    fn is_literal(&self) -> bool {
        self.is_literal()
    }
    
    fn is_variable(&self) -> bool {
        self.is_variable()
    }
}

// Conversion implementations
impl From<NamedNode> for Term {
    fn from(node: NamedNode) -> Self {
        Term::NamedNode(node)
    }
}

impl From<BlankNode> for Term {
    fn from(node: BlankNode) -> Self {
        Term::BlankNode(node)
    }
}

impl From<Literal> for Term {
    fn from(literal: Literal) -> Self {
        Term::Literal(literal)
    }
}

impl From<Variable> for Term {
    fn from(variable: Variable) -> Self {
        Term::Variable(variable)
    }
}

// Conversion implementations for union types
impl From<Subject> for Term {
    fn from(subject: Subject) -> Self {
        match subject {
            Subject::NamedNode(nn) => Term::NamedNode(nn),
            Subject::BlankNode(bn) => Term::BlankNode(bn),
            Subject::Variable(v) => Term::Variable(v),
        }
    }
}

impl From<Predicate> for Term {
    fn from(predicate: Predicate) -> Self {
        match predicate {
            Predicate::NamedNode(nn) => Term::NamedNode(nn),
            Predicate::Variable(v) => Term::Variable(v),
        }
    }
}

impl From<Object> for Term {
    fn from(object: Object) -> Self {
        match object {
            Object::NamedNode(nn) => Term::NamedNode(nn),
            Object::BlankNode(bn) => Term::BlankNode(bn),
            Object::Literal(l) => Term::Literal(l),
            Object::Variable(v) => Term::Variable(v),
        }
    }
}

/// Union type for terms that can be subjects in RDF triples
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Subject {
    NamedNode(NamedNode),
    BlankNode(BlankNode),
    Variable(Variable),
}

impl From<NamedNode> for Subject {
    fn from(node: NamedNode) -> Self {
        Subject::NamedNode(node)
    }
}

impl From<BlankNode> for Subject {
    fn from(node: BlankNode) -> Self {
        Subject::BlankNode(node)
    }
}

impl From<Variable> for Subject {
    fn from(variable: Variable) -> Self {
        Subject::Variable(variable)
    }
}

/// Union type for terms that can be predicates in RDF triples
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Predicate {
    NamedNode(NamedNode),
    Variable(Variable),
}

impl From<NamedNode> for Predicate {
    fn from(node: NamedNode) -> Self {
        Predicate::NamedNode(node)
    }
}

impl From<Variable> for Predicate {
    fn from(variable: Variable) -> Self {
        Predicate::Variable(variable)
    }
}

/// Union type for terms that can be objects in RDF triples
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Object {
    NamedNode(NamedNode),
    BlankNode(BlankNode),
    Literal(Literal),
    Variable(Variable),
}

impl From<NamedNode> for Object {
    fn from(node: NamedNode) -> Self {
        Object::NamedNode(node)
    }
}

impl From<BlankNode> for Object {
    fn from(node: BlankNode) -> Self {
        Object::BlankNode(node)
    }
}

impl From<Literal> for Object {
    fn from(literal: Literal) -> Self {
        Object::Literal(literal)
    }
}

impl From<Variable> for Object {
    fn from(variable: Variable) -> Self {
        Object::Variable(variable)
    }
}

// Term to position conversions (needed for rdfxml parser)
impl TryFrom<Term> for Subject {
    type Error = OxirsError;
    
    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::NamedNode(n) => Ok(Subject::NamedNode(n)),
            Term::BlankNode(b) => Ok(Subject::BlankNode(b)),
            Term::Variable(v) => Ok(Subject::Variable(v)),
            Term::Literal(_) => Err(OxirsError::Parse("Literals cannot be used as subjects".to_string())),
        }
    }
}

impl TryFrom<Term> for Predicate {
    type Error = OxirsError;
    
    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::NamedNode(n) => Ok(Predicate::NamedNode(n)),
            Term::Variable(v) => Ok(Predicate::Variable(v)),
            Term::BlankNode(_) => Err(OxirsError::Parse("Blank nodes cannot be used as predicates".to_string())),
            Term::Literal(_) => Err(OxirsError::Parse("Literals cannot be used as predicates".to_string())),
        }
    }
}

impl TryFrom<Term> for Object {
    type Error = OxirsError;
    
    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::NamedNode(n) => Ok(Object::NamedNode(n)),
            Term::BlankNode(b) => Ok(Object::BlankNode(b)),
            Term::Literal(l) => Ok(Object::Literal(l)),
            Term::Variable(v) => Ok(Object::Variable(v)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_blank_node() {
        let blank = BlankNode::new("b1").unwrap();
        assert_eq!(blank.id(), "_:b1");
        assert_eq!(blank.local_id(), "b1");
        assert!(blank.is_blank_node());
        assert_eq!(format!("{}", blank), "_:b1");
    }
    
    #[test]
    fn test_blank_node_with_prefix() {
        let blank = BlankNode::new("_:test").unwrap();
        assert_eq!(blank.id(), "_:test");
        assert_eq!(blank.local_id(), "test");
    }
    
    #[test]
    fn test_blank_node_unique() {
        let blank1 = BlankNode::new_unique();
        let blank2 = BlankNode::new_unique();
        assert_ne!(blank1.id(), blank2.id());
        assert!(blank1.id().starts_with("_:b"));
        assert!(blank2.id().starts_with("_:b"));
    }
    
    #[test]
    fn test_blank_node_unique_with_prefix() {
        let blank1 = BlankNode::new_unique_with_prefix("test").unwrap();
        let blank2 = BlankNode::new_unique_with_prefix("test").unwrap();
        assert_ne!(blank1.id(), blank2.id());
        assert!(blank1.id().starts_with("_:test_"));
        assert!(blank2.id().starts_with("_:test_"));
    }
    
    #[test]
    fn test_blank_node_validation() {
        // Valid IDs
        assert!(BlankNode::new("test123").is_ok());
        assert!(BlankNode::new("Test_Node").is_ok());
        assert!(BlankNode::new("node-1.2").is_ok());
        
        // Invalid IDs
        assert!(BlankNode::new("").is_err());
        assert!(BlankNode::new("_:").is_err());
        assert!(BlankNode::new("123invalid").is_err()); // Can't start with number
        assert!(BlankNode::new("invalid@char").is_err());
        assert!(BlankNode::new("invalid space").is_err());
    }
    
    #[test]
    fn test_blank_node_serde() {
        let blank = BlankNode::new("serializable").unwrap();
        let json = serde_json::to_string(&blank).unwrap();
        let deserialized: BlankNode = serde_json::from_str(&json).unwrap();
        assert_eq!(blank, deserialized);
    }
    
    #[test]
    fn test_variable() {
        let var = Variable::new("x").unwrap();
        assert_eq!(var.name(), "x");
        assert!(var.is_variable());
        assert_eq!(format!("{}", var), "?x");
        assert_eq!(var.with_prefix(), "?x");
    }
    
    #[test]
    fn test_variable_with_prefix() {
        let var1 = Variable::new("?test").unwrap();
        let var2 = Variable::new("$test").unwrap();
        assert_eq!(var1.name(), "test");
        assert_eq!(var2.name(), "test");
        assert_eq!(var1, var2); // Same after normalization
    }
    
    #[test]
    fn test_variable_validation() {
        // Valid names
        assert!(Variable::new("x").is_ok());
        assert!(Variable::new("test123").is_ok());
        assert!(Variable::new("_underscore").is_ok());
        assert!(Variable::new("?prefixed").is_ok());
        assert!(Variable::new("$prefixed").is_ok());
        
        // Invalid names
        assert!(Variable::new("").is_err());
        assert!(Variable::new("?").is_err());
        assert!(Variable::new("$").is_err());
        assert!(Variable::new("123invalid").is_err()); // Can't start with number
        assert!(Variable::new("invalid-char").is_err());
        assert!(Variable::new("invalid space").is_err());
        
        // Reserved keywords
        assert!(Variable::new("select").is_err());
        assert!(Variable::new("WHERE").is_err()); // Case insensitive
        assert!(Variable::new("?from").is_err());
    }
    
    
    #[test]
    fn test_variable_serde() {
        let var = Variable::new("serializable").unwrap();
        let json = serde_json::to_string(&var).unwrap();
        let deserialized: Variable = serde_json::from_str(&json).unwrap();
        assert_eq!(var, deserialized);
    }
    
    #[test]
    fn test_term_enum() {
        let term = Term::NamedNode(NamedNode::new("http://example.org").unwrap());
        assert!(term.is_named_node());
        assert!(term.as_named_node().is_some());
        assert!(term.as_blank_node().is_none());
    }
}