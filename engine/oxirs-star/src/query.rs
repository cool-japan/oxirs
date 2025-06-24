//! SPARQL-star query processing and execution.
//!
//! This module provides support for SPARQL-star queries that can query
//! and manipulate quoted triples using extended SPARQL syntax.

use std::collections::{HashMap, HashSet};

use tracing::{debug, span, Level};

use crate::model::{StarGraph, StarTerm, StarTriple, Variable};
use crate::store::StarStore;
use crate::{StarError, StarResult};

/// SPARQL-star query types
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    /// SELECT query
    Select,
    /// CONSTRUCT query
    Construct,
    /// ASK query
    Ask,
    /// DESCRIBE query
    Describe,
}

/// Variable binding for query results
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    /// Variable name to term mapping
    bindings: HashMap<String, StarTerm>,
}

impl Binding {
    /// Create a new empty binding
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Add a variable binding
    pub fn bind(&mut self, variable: &str, term: StarTerm) {
        self.bindings.insert(variable.to_string(), term);
    }

    /// Get the binding for a variable
    pub fn get(&self, variable: &str) -> Option<&StarTerm> {
        self.bindings.get(variable)
    }

    /// Get all variable names
    pub fn variables(&self) -> Vec<&String> {
        self.bindings.keys().collect()
    }

    /// Check if a variable is bound
    pub fn is_bound(&self, variable: &str) -> bool {
        self.bindings.contains_key(variable)
    }

    /// Merge with another binding (fails if conflicting bindings exist)
    pub fn merge(&self, other: &Binding) -> Option<Binding> {
        let mut merged = self.clone();
        
        for (var, term) in &other.bindings {
            if let Some(existing_term) = merged.bindings.get(var) {
                if existing_term != term {
                    return None; // Conflicting bindings
                }
            } else {
                merged.bindings.insert(var.clone(), term.clone());
            }
        }
        
        Some(merged)
    }
}

impl Default for Binding {
    fn default() -> Self {
        Self::new()
    }
}

/// SPARQL-star basic graph pattern (BGP)
#[derive(Debug, Clone)]
pub struct BasicGraphPattern {
    /// Triple patterns in the BGP
    patterns: Vec<TriplePattern>,
}

/// SPARQL-star triple pattern with support for quoted triple patterns
#[derive(Debug, Clone)]
pub struct TriplePattern {
    /// Subject pattern (can be variable, term, or quoted triple pattern)
    pub subject: TermPattern,
    /// Predicate pattern (can be variable or term)
    pub predicate: TermPattern,
    /// Object pattern (can be variable, term, or quoted triple pattern)
    pub object: TermPattern,
}

/// Term pattern for SPARQL-star queries
#[derive(Debug, Clone)]
pub enum TermPattern {
    /// Concrete term
    Term(StarTerm),
    /// Variable to be bound
    Variable(String),
    /// Quoted triple pattern (SPARQL-star extension)
    QuotedTriplePattern(Box<TriplePattern>),
}

impl TermPattern {
    /// Check if this pattern matches a term with the given binding
    pub fn matches(&self, term: &StarTerm, binding: &Binding) -> bool {
        match self {
            TermPattern::Term(pattern_term) => pattern_term == term,
            TermPattern::Variable(var_name) => {
                if let Some(bound_term) = binding.get(var_name) {
                    bound_term == term
                } else {
                    true // Unbound variable matches anything
                }
            }
            TermPattern::QuotedTriplePattern(pattern) => {
                if let StarTerm::QuotedTriple(quoted_triple) = term {
                    pattern.matches(quoted_triple, binding)
                } else {
                    false
                }
            }
        }
    }

    /// Extract variables from this pattern
    pub fn extract_variables(&self, variables: &mut HashSet<String>) {
        match self {
            TermPattern::Term(_) => {},
            TermPattern::Variable(var) => {
                variables.insert(var.clone());
            }
            TermPattern::QuotedTriplePattern(pattern) => {
                pattern.extract_variables(variables);
            }
        }
    }
}

impl TriplePattern {
    /// Create a new triple pattern
    pub fn new(subject: TermPattern, predicate: TermPattern, object: TermPattern) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Check if this pattern matches a triple with the given binding
    pub fn matches(&self, triple: &StarTriple, binding: &Binding) -> bool {
        self.subject.matches(&triple.subject, binding)
            && self.predicate.matches(&triple.predicate, binding)
            && self.object.matches(&triple.object, binding)
    }

    /// Try to create a binding from this pattern and a matching triple
    pub fn try_bind(&self, triple: &StarTriple, existing_binding: &Binding) -> Option<Binding> {
        let mut new_binding = existing_binding.clone();
        
        if !self.bind_term(&self.subject, &triple.subject, &mut new_binding) {
            return None;
        }
        
        if !self.bind_term(&self.predicate, &triple.predicate, &mut new_binding) {
            return None;
        }
        
        if !self.bind_term(&self.object, &triple.object, &mut new_binding) {
            return None;
        }
        
        Some(new_binding)
    }

    /// Try to bind a term pattern to a concrete term
    fn bind_term(&self, pattern: &TermPattern, term: &StarTerm, binding: &mut Binding) -> bool {
        match pattern {
            TermPattern::Term(pattern_term) => pattern_term == term,
            TermPattern::Variable(var_name) => {
                if let Some(existing_term) = binding.get(var_name) {
                    existing_term == term
                } else {
                    binding.bind(var_name, term.clone());
                    true
                }
            }
            TermPattern::QuotedTriplePattern(quoted_pattern) => {
                if let StarTerm::QuotedTriple(quoted_triple) = term {
                    if let Some(new_binding) = quoted_pattern.try_bind(quoted_triple, binding) {
                        // Merge the bindings from the quoted triple pattern
                        for (var, value) in new_binding.bindings.iter() {
                            if !binding.is_bound(var) {
                                binding.bind(var, value.clone());
                            }
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }

    /// Extract all variables from this pattern
    pub fn extract_variables(&self, variables: &mut HashSet<String>) {
        self.subject.extract_variables(variables);
        self.predicate.extract_variables(variables);
        self.object.extract_variables(variables);
    }
}

impl BasicGraphPattern {
    /// Create a new empty BGP
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Add a triple pattern to the BGP
    pub fn add_pattern(&mut self, pattern: TriplePattern) {
        self.patterns.push(pattern);
    }

    /// Get all patterns in the BGP
    pub fn patterns(&self) -> &[TriplePattern] {
        &self.patterns
    }

    /// Extract all variables from this BGP
    pub fn extract_variables(&self) -> HashSet<String> {
        let mut variables = HashSet::new();
        for pattern in &self.patterns {
            pattern.extract_variables(&mut variables);
        }
        variables
    }
}

impl Default for BasicGraphPattern {
    fn default() -> Self {
        Self::new()
    }
}

/// SPARQL-star query executor
pub struct QueryExecutor {
    /// Reference to the RDF-star store
    store: StarStore,
}

impl QueryExecutor {
    /// Create a new query executor
    pub fn new(store: StarStore) -> Self {
        Self { store }
    }

    /// Execute a basic graph pattern against the store
    pub fn execute_bgp(&self, bgp: &BasicGraphPattern) -> StarResult<Vec<Binding>> {
        let span = span!(Level::INFO, "execute_bgp");
        let _enter = span.enter();

        if bgp.patterns.is_empty() {
            return Ok(vec![Binding::new()]);
        }

        // Start with the first pattern
        let mut current_bindings = self.execute_pattern(&bgp.patterns[0], &Binding::new())?;

        // Join with remaining patterns
        for pattern in &bgp.patterns[1..] {
            let mut new_bindings = Vec::new();
            
            for binding in &current_bindings {
                let pattern_bindings = self.execute_pattern(pattern, binding)?;
                new_bindings.extend(pattern_bindings);
            }
            
            current_bindings = new_bindings;
        }

        debug!("BGP execution produced {} bindings", current_bindings.len());
        Ok(current_bindings)
    }

    /// Execute a single triple pattern
    fn execute_pattern(&self, pattern: &TriplePattern, initial_binding: &Binding) -> StarResult<Vec<Binding>> {
        let mut bindings = Vec::new();
        
        // Get all triples from the store
        let triples = self.store.triples();
        
        for triple in triples {
            if pattern.matches(&triple, initial_binding) {
                if let Some(new_binding) = pattern.try_bind(&triple, initial_binding) {
                    bindings.push(new_binding);
                }
            }
        }
        
        Ok(bindings)
    }

    /// Execute a SELECT query (simplified)
    pub fn execute_select(&self, bgp: &BasicGraphPattern, select_vars: &[String]) -> StarResult<Vec<HashMap<String, StarTerm>>> {
        let span = span!(Level::INFO, "execute_select");
        let _enter = span.enter();

        let bindings = self.execute_bgp(bgp)?;
        let mut results = Vec::new();

        for binding in bindings {
            let mut result = HashMap::new();
            
            for var in select_vars {
                if let Some(term) = binding.get(var) {
                    result.insert(var.clone(), term.clone());
                }
            }
            
            if !result.is_empty() {
                results.push(result);
            }
        }

        debug!("SELECT query produced {} results", results.len());
        Ok(results)
    }

    /// Execute a CONSTRUCT query (simplified)
    pub fn execute_construct(&self, bgp: &BasicGraphPattern, construct_patterns: &[TriplePattern]) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "execute_construct");
        let _enter = span.enter();

        let bindings = self.execute_bgp(bgp)?;
        let mut constructed_graph = StarGraph::new();

        for binding in bindings {
            for pattern in construct_patterns {
                if let Some(triple) = self.instantiate_pattern(pattern, &binding)? {
                    constructed_graph.insert(triple)?;
                }
            }
        }

        debug!("CONSTRUCT query produced {} triples", constructed_graph.len());
        Ok(constructed_graph)
    }

    /// Execute an ASK query
    pub fn execute_ask(&self, bgp: &BasicGraphPattern) -> StarResult<bool> {
        let span = span!(Level::INFO, "execute_ask");
        let _enter = span.enter();

        let bindings = self.execute_bgp(bgp)?;
        let result = !bindings.is_empty();
        
        debug!("ASK query result: {}", result);
        Ok(result)
    }

    /// Instantiate a triple pattern with a binding
    fn instantiate_pattern(&self, pattern: &TriplePattern, binding: &Binding) -> StarResult<Option<StarTriple>> {
        let subject = self.instantiate_term_pattern(&pattern.subject, binding)?;
        let predicate = self.instantiate_term_pattern(&pattern.predicate, binding)?;
        let object = self.instantiate_term_pattern(&pattern.object, binding)?;

        if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
            Ok(Some(StarTriple::new(s, p, o)))
        } else {
            Ok(None)
        }
    }

    /// Instantiate a term pattern with a binding
    fn instantiate_term_pattern(&self, pattern: &TermPattern, binding: &Binding) -> StarResult<Option<StarTerm>> {
        match pattern {
            TermPattern::Term(term) => Ok(Some(term.clone())),
            TermPattern::Variable(var) => Ok(binding.get(var).cloned()),
            TermPattern::QuotedTriplePattern(quoted_pattern) => {
                if let Some(triple) = self.instantiate_pattern(quoted_pattern, binding)? {
                    Ok(Some(StarTerm::quoted_triple(triple)))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

/// Simple SPARQL-star query parser (very basic implementation)
pub struct QueryParser;

impl QueryParser {
    /// Parse a simple SELECT query with quoted triple patterns
    pub fn parse_simple_select(query: &str) -> StarResult<(Vec<String>, BasicGraphPattern)> {
        // This is a very simplified parser for demonstration
        // A real implementation would use a proper SPARQL grammar parser
        
        let lines: Vec<&str> = query.lines().map(|l| l.trim()).collect();
        let mut select_vars = Vec::new();
        let mut bgp = BasicGraphPattern::new();
        
        let mut in_where = false;
        
        for line in lines {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            
            if line.to_uppercase().starts_with("SELECT") {
                // Extract variables
                let parts: Vec<&str> = line.split_whitespace().collect();
                for part in &parts[1..] {
                    if part.starts_with('?') {
                        select_vars.push(part[1..].to_string());
                    }
                }
            } else if line.to_uppercase().contains("WHERE") {
                in_where = true;
            } else if in_where && line.contains('.') {
                // Parse triple pattern
                if let Ok(pattern) = Self::parse_triple_pattern(line) {
                    bgp.add_pattern(pattern);
                }
            }
        }
        
        Ok((select_vars, bgp))
    }

    /// Parse a simple triple pattern
    fn parse_triple_pattern(line: &str) -> StarResult<TriplePattern> {
        // Remove trailing dot and split
        let line = line.trim_end_matches('.').trim();
        let parts = Self::tokenize_pattern(line)?;
        
        if parts.len() != 3 {
            return Err(StarError::QueryError(
                format!("Invalid triple pattern: {}", line)
            ));
        }
        
        let subject = Self::parse_term_pattern(&parts[0])?;
        let predicate = Self::parse_term_pattern(&parts[1])?;
        let object = Self::parse_term_pattern(&parts[2])?;
        
        Ok(TriplePattern::new(subject, predicate, object))
    }

    /// Tokenize a pattern handling quoted triples
    fn tokenize_pattern(pattern: &str) -> StarResult<Vec<String>> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut chars = pattern.chars().peekable();
        let mut depth = 0;
        let mut in_string = false;

        while let Some(ch) = chars.next() {
            match ch {
                '"' => {
                    in_string = !in_string;
                    current_token.push(ch);
                }
                '<' if !in_string && chars.peek() == Some(&'<') => {
                    chars.next(); // consume second '<'
                    depth += 1;
                    current_token.push_str("<<");
                }
                '>' if !in_string && chars.peek() == Some(&'>') => {
                    chars.next(); // consume second '>'
                    depth -= 1;
                    current_token.push_str(">>");
                }
                ' ' | '\t' if !in_string && depth == 0 => {
                    if !current_token.trim().is_empty() {
                        tokens.push(current_token.trim().to_string());
                        current_token.clear();
                    }
                }
                _ => {
                    current_token.push(ch);
                }
            }
        }

        if !current_token.trim().is_empty() {
            tokens.push(current_token.trim().to_string());
        }

        Ok(tokens)
    }

    /// Parse a term pattern
    fn parse_term_pattern(term_str: &str) -> StarResult<TermPattern> {
        let term_str = term_str.trim();

        // Variable
        if term_str.starts_with('?') {
            return Ok(TermPattern::Variable(term_str[1..].to_string()));
        }

        // Quoted triple pattern
        if term_str.starts_with("<<") && term_str.ends_with(">>") {
            let inner = &term_str[2..term_str.len() - 2];
            let inner_pattern = Self::parse_triple_pattern(inner)?;
            return Ok(TermPattern::QuotedTriplePattern(Box::new(inner_pattern)));
        }

        // Regular term
        let term = Self::parse_concrete_term(term_str)?;
        Ok(TermPattern::Term(term))
    }

    /// Parse a concrete term (not a variable or pattern)
    fn parse_concrete_term(term_str: &str) -> StarResult<StarTerm> {
        // IRI
        if term_str.starts_with('<') && term_str.ends_with('>') {
            let iri = &term_str[1..term_str.len() - 1];
            return StarTerm::iri(iri);
        }

        // Blank node
        if term_str.starts_with("_:") {
            let id = &term_str[2..];
            return StarTerm::blank_node(id);
        }

        // Literal
        if term_str.starts_with('"') {
            // Simple literal parsing (not complete)
            let end_quote = term_str.rfind('"').unwrap_or(term_str.len());
            let value = &term_str[1..end_quote];
            return StarTerm::literal(value);
        }

        Err(StarError::QueryError(format!("Cannot parse term: {}", term_str)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_operations() {
        let mut binding = Binding::new();
        assert!(!binding.is_bound("x"));

        binding.bind("x", StarTerm::iri("http://example.org/alice").unwrap());
        assert!(binding.is_bound("x"));
        assert_eq!(binding.get("x"), Some(&StarTerm::iri("http://example.org/alice").unwrap()));

        let mut other = Binding::new();
        other.bind("y", StarTerm::literal("test").unwrap());

        let merged = binding.merge(&other).unwrap();
        assert!(merged.is_bound("x"));
        assert!(merged.is_bound("y"));
    }

    #[test]
    fn test_triple_pattern_matching() {
        let pattern = TriplePattern::new(
            TermPattern::Variable("x".to_string()),
            TermPattern::Term(StarTerm::iri("http://example.org/knows").unwrap()),
            TermPattern::Variable("y".to_string()),
        );

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );

        let binding = Binding::new();
        assert!(pattern.matches(&triple, &binding));

        let new_binding = pattern.try_bind(&triple, &binding).unwrap();
        assert_eq!(new_binding.get("x"), Some(&StarTerm::iri("http://example.org/alice").unwrap()));
        assert_eq!(new_binding.get("y"), Some(&StarTerm::iri("http://example.org/bob").unwrap()));
    }

    #[test]
    fn test_quoted_triple_pattern() {
        let inner_pattern = TriplePattern::new(
            TermPattern::Variable("x".to_string()),
            TermPattern::Term(StarTerm::iri("http://example.org/age").unwrap()),
            TermPattern::Variable("age".to_string()),
        );

        let outer_pattern = TriplePattern::new(
            TermPattern::QuotedTriplePattern(Box::new(inner_pattern)),
            TermPattern::Term(StarTerm::iri("http://example.org/certainty").unwrap()),
            TermPattern::Variable("cert".to_string()),
        );

        let inner_triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let outer_triple = StarTriple::new(
            StarTerm::quoted_triple(inner_triple),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        let binding = Binding::new();
        assert!(outer_pattern.matches(&outer_triple, &binding));

        let new_binding = outer_pattern.try_bind(&outer_triple, &binding).unwrap();
        assert!(new_binding.is_bound("x"));
        assert!(new_binding.is_bound("age"));
        assert!(new_binding.is_bound("cert"));
    }

    #[test]
    fn test_bgp_execution() {
        let store = StarStore::new();
        
        // Add some test data
        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );
        
        let triple2 = StarTriple::new(
            StarTerm::iri("http://example.org/bob").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/charlie").unwrap(),
        );
        
        store.insert(&triple1).unwrap();
        store.insert(&triple2).unwrap();

        let executor = QueryExecutor::new(store);
        
        // Create BGP: ?x knows ?y
        let mut bgp = BasicGraphPattern::new();
        bgp.add_pattern(TriplePattern::new(
            TermPattern::Variable("x".to_string()),
            TermPattern::Term(StarTerm::iri("http://example.org/knows").unwrap()),
            TermPattern::Variable("y".to_string()),
        ));

        let bindings = executor.execute_bgp(&bgp).unwrap();
        assert_eq!(bindings.len(), 2);
    }

    #[test]
    fn test_query_parser() {
        let query = r#"
            SELECT ?x ?y
            WHERE {
                ?x <http://example.org/knows> ?y .
            }
        "#;

        let (vars, bgp) = QueryParser::parse_simple_select(query).unwrap();
        assert_eq!(vars, vec!["x", "y"]);
        assert_eq!(bgp.patterns().len(), 1);
    }

    #[test]
    fn test_quoted_triple_query_parsing() {
        let query = r#"
            SELECT ?cert
            WHERE {
                << ?x <http://example.org/age> ?age >> <http://example.org/certainty> ?cert .
            }
        "#;

        let (vars, bgp) = QueryParser::parse_simple_select(query).unwrap();
        assert_eq!(vars, vec!["cert"]);
        assert_eq!(bgp.patterns().len(), 1);

        let pattern = &bgp.patterns()[0];
        assert!(matches!(pattern.subject, TermPattern::QuotedTriplePattern(_)));
    }
}
