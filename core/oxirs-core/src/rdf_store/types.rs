//! SPARQL query result types

use crate::model::*;

/// SPARQL query results supporting different result types
#[derive(Debug, Clone)]
pub enum QueryResults {
    /// SELECT query results - variable bindings
    Bindings(Vec<VariableBinding>),
    /// ASK query results - boolean
    Boolean(bool),
    /// CONSTRUCT/DESCRIBE query results - RDF quads
    Graph(Vec<Quad>),
}

impl QueryResults {
    /// Create empty SELECT results
    pub fn empty_bindings() -> Self {
        QueryResults::Bindings(Vec::new())
    }

    /// Create ASK result
    pub fn boolean(value: bool) -> Self {
        QueryResults::Boolean(value)
    }

    /// Create CONSTRUCT/DESCRIBE results
    pub fn graph(quads: Vec<Quad>) -> Self {
        QueryResults::Graph(quads)
    }

    /// Check if results are empty
    pub fn is_empty(&self) -> bool {
        match self {
            QueryResults::Bindings(bindings) => bindings.is_empty(),
            QueryResults::Boolean(_) => false,
            QueryResults::Graph(quads) => quads.is_empty(),
        }
    }

    /// Get the number of results
    pub fn len(&self) -> usize {
        match self {
            QueryResults::Bindings(bindings) => bindings.len(),
            QueryResults::Boolean(_) => 1,
            QueryResults::Graph(quads) => quads.len(),
        }
    }
}

/// Variable binding for SELECT query results
#[derive(Debug, Clone, Default)]
pub struct VariableBinding {
    pub bindings: std::collections::HashMap<String, Term>,
}

impl VariableBinding {
    pub fn new() -> Self {
        Self {
            bindings: std::collections::HashMap::new(),
        }
    }

    pub fn bind(&mut self, variable: String, value: Term) {
        self.bindings.insert(variable, value);
    }

    pub fn get(&self, variable: &str) -> Option<&Term> {
        self.bindings.get(variable)
    }

    pub fn variables(&self) -> impl Iterator<Item = &String> {
        self.bindings.keys()
    }

    pub fn values(&self) -> impl Iterator<Item = &Term> {
        self.bindings.values()
    }
}
