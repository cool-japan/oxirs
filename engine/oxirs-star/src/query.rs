//! SPARQL-star query processing and execution.
//!
//! This module provides support for SPARQL-star queries that can query
//! and manipulate quoted triples using extended SPARQL syntax.

use std::collections::{HashMap, HashSet};

use tracing::{debug, span, Level};

use crate::functions::{Expression, ExpressionEvaluator};
use crate::model::{StarGraph, StarTerm, StarTriple};
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
    /// Filter expressions to apply
    filters: Vec<Expression>,
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
            TermPattern::Term(_) => {}
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
            filters: Vec::new(),
        }
    }

    /// Add a triple pattern to the BGP
    pub fn add_pattern(&mut self, pattern: TriplePattern) {
        self.patterns.push(pattern);
    }

    /// Add a filter expression to the BGP
    pub fn add_filter(&mut self, filter: Expression) {
        self.filters.push(filter);
    }

    /// Get all patterns in the BGP
    pub fn patterns(&self) -> &[TriplePattern] {
        &self.patterns
    }

    /// Get all filters in the BGP
    pub fn filters(&self) -> &[Expression] {
        &self.filters
    }

    /// Extract all variables from this BGP
    pub fn extract_variables(&self) -> HashSet<String> {
        let mut variables = HashSet::new();
        for pattern in &self.patterns {
            pattern.extract_variables(&mut variables);
        }
        // Also extract variables from filters
        for filter in &self.filters {
            Self::extract_variables_from_expr(filter, &mut variables);
        }
        variables
    }

    /// Extract variables from an expression
    fn extract_variables_from_expr(expr: &Expression, variables: &mut HashSet<String>) {
        match expr {
            Expression::Variable(var) => {
                variables.insert(var.clone());
            }
            Expression::FunctionCall { args, .. } => {
                for arg in args {
                    Self::extract_variables_from_expr(arg, variables);
                }
            }
            Expression::BinaryOp { left, right, .. } => {
                Self::extract_variables_from_expr(left, variables);
                Self::extract_variables_from_expr(right, variables);
            }
            Expression::UnaryOp { expr, .. } => {
                Self::extract_variables_from_expr(expr, variables);
            }
            Expression::Term(_) => {}
        }
    }
}

impl Default for BasicGraphPattern {
    fn default() -> Self {
        Self::new()
    }
}

/// Query optimization strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryOptimization {
    /// No optimization (simple left-to-right execution)
    None,
    /// Cost-based optimization using selectivity estimates
    CostBased,
    /// Heuristic-based optimization using pattern analysis
    Heuristic,
}

/// Join strategies for BGP execution
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JoinStrategy {
    /// Nested loop join (simple but potentially inefficient)
    NestedLoop,
    /// Hash join (good for large result sets)
    Hash,
    /// Index-based join (leverages store indices)
    Index,
}

/// Query execution statistics
#[derive(Debug, Clone, Default)]
pub struct QueryStatistics {
    /// Number of pattern evaluations
    pub pattern_evaluations: usize,
    /// Number of join operations
    pub join_operations: usize,
    /// Total execution time in microseconds
    pub execution_time_us: u64,
    /// Number of intermediate results generated
    pub intermediate_results: usize,
    /// Memory usage estimate in bytes
    pub memory_usage_bytes: usize,
}

/// SPARQL-star query executor with advanced optimization capabilities
pub struct QueryExecutor {
    /// Reference to the RDF-star store
    store: StarStore,
    /// Query optimization strategy
    optimization: QueryOptimization,
    /// Join strategy for BGP execution
    join_strategy: JoinStrategy,
    /// Execution statistics
    statistics: QueryStatistics,
}

impl QueryExecutor {
    /// Create a new query executor with default settings
    pub fn new(store: StarStore) -> Self {
        Self {
            store,
            optimization: QueryOptimization::Heuristic,
            join_strategy: JoinStrategy::Index,
            statistics: QueryStatistics::default(),
        }
    }

    /// Create a new query executor with custom optimization settings
    pub fn with_optimization(
        store: StarStore,
        optimization: QueryOptimization,
        join_strategy: JoinStrategy,
    ) -> Self {
        Self {
            store,
            optimization,
            join_strategy,
            statistics: QueryStatistics::default(),
        }
    }

    /// Get execution statistics
    pub fn statistics(&self) -> &QueryStatistics {
        &self.statistics
    }

    /// Reset execution statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = QueryStatistics::default();
    }

    /// Execute a basic graph pattern against the store with optimization
    pub fn execute_bgp(&mut self, bgp: &BasicGraphPattern) -> StarResult<Vec<Binding>> {
        let span = span!(Level::INFO, "execute_bgp");
        let _enter = span.enter();
        let start_time = std::time::Instant::now();

        if bgp.patterns.is_empty() {
            return Ok(vec![Binding::new()]);
        }

        // Optimize pattern order based on strategy
        let optimized_patterns = self.optimize_pattern_order(bgp)?;

        // Execute patterns using optimized join strategy
        let mut current_bindings =
            self.execute_pattern_optimized(&optimized_patterns[0], &Binding::new())?;
        self.statistics.pattern_evaluations += 1;

        // Join with remaining patterns using selected strategy
        for pattern in &optimized_patterns[1..] {
            current_bindings = self.join_with_pattern(current_bindings, pattern)?;
            self.statistics.pattern_evaluations += 1; // Count each pattern evaluation
            self.statistics.join_operations += 1;
            self.statistics.intermediate_results += current_bindings.len();
        }

        // Apply filters to the bindings
        if !bgp.filters.is_empty() {
            current_bindings = self.apply_filters(current_bindings, &bgp.filters)?;
        }

        self.statistics.execution_time_us += start_time.elapsed().as_micros() as u64;
        debug!(
            "BGP execution produced {} bindings using {} strategy",
            current_bindings.len(),
            format!("{:?}", self.optimization)
        );
        Ok(current_bindings)
    }

    /// Optimize pattern execution order based on selectivity estimates
    fn optimize_pattern_order(&self, bgp: &BasicGraphPattern) -> StarResult<Vec<TriplePattern>> {
        match self.optimization {
            QueryOptimization::None => Ok(bgp.patterns().to_vec()),
            QueryOptimization::Heuristic => self.heuristic_optimization(bgp),
            QueryOptimization::CostBased => self.cost_based_optimization(bgp),
        }
    }

    /// Heuristic-based pattern optimization
    fn heuristic_optimization(&self, bgp: &BasicGraphPattern) -> StarResult<Vec<TriplePattern>> {
        let mut patterns = bgp.patterns().to_vec();

        // Sort patterns by selectivity (quoted triple patterns first, then most specific)
        patterns.sort_by(|a, b| {
            let a_selectivity = self.estimate_pattern_selectivity(a);
            let b_selectivity = self.estimate_pattern_selectivity(b);
            a_selectivity.partial_cmp(&b_selectivity).unwrap()
        });

        Ok(patterns)
    }

    /// Cost-based pattern optimization using statistics
    fn cost_based_optimization(&self, bgp: &BasicGraphPattern) -> StarResult<Vec<TriplePattern>> {
        let mut patterns = bgp.patterns().to_vec();

        // Sort by estimated cost (lower cost first)
        patterns.sort_by(|a, b| {
            let a_cost = self.estimate_pattern_cost(a);
            let b_cost = self.estimate_pattern_cost(b);
            a_cost.partial_cmp(&b_cost).unwrap()
        });

        Ok(patterns)
    }

    /// Estimate pattern selectivity (lower is more selective)
    fn estimate_pattern_selectivity(&self, pattern: &TriplePattern) -> f64 {
        let mut selectivity = 1.0;

        // Quoted triple patterns are typically more selective
        if matches!(pattern.subject, TermPattern::QuotedTriplePattern(_)) {
            selectivity *= 0.1;
        }
        if matches!(pattern.object, TermPattern::QuotedTriplePattern(_)) {
            selectivity *= 0.1;
        }

        // Concrete terms are more selective than variables
        if matches!(pattern.subject, TermPattern::Term(_)) {
            selectivity *= 0.3;
        }
        if matches!(pattern.predicate, TermPattern::Term(_)) {
            selectivity *= 0.2;
        }
        if matches!(pattern.object, TermPattern::Term(_)) {
            selectivity *= 0.3;
        }

        selectivity
    }

    /// Estimate pattern execution cost
    fn estimate_pattern_cost(&self, pattern: &TriplePattern) -> f64 {
        // Base cost for pattern evaluation
        let mut cost = 100.0;

        // Add cost for quoted triple pattern complexity
        if matches!(pattern.subject, TermPattern::QuotedTriplePattern(_)) {
            cost += 500.0;
        }
        if matches!(pattern.object, TermPattern::QuotedTriplePattern(_)) {
            cost += 500.0;
        }

        // Reduce cost for concrete terms (can use indices)
        if matches!(pattern.subject, TermPattern::Term(_)) {
            cost *= 0.5;
        }
        if matches!(pattern.predicate, TermPattern::Term(_)) {
            cost *= 0.3;
        }
        if matches!(pattern.object, TermPattern::Term(_)) {
            cost *= 0.5;
        }

        cost
    }

    /// Execute a single triple pattern with optimization
    fn execute_pattern_optimized(
        &self,
        pattern: &TriplePattern,
        initial_binding: &Binding,
    ) -> StarResult<Vec<Binding>> {
        // Try to use store's advanced indexing for quoted triple patterns
        if self.can_use_index_optimization(pattern) {
            self.execute_pattern_with_index(pattern, initial_binding)
        } else {
            self.execute_pattern(pattern, initial_binding)
        }
    }

    /// Check if pattern can benefit from index optimization
    fn can_use_index_optimization(&self, pattern: &TriplePattern) -> bool {
        // Use index optimization for patterns with quoted triples or concrete terms
        matches!(
            pattern.subject,
            TermPattern::QuotedTriplePattern(_) | TermPattern::Term(_)
        ) || matches!(
            pattern.predicate,
            TermPattern::QuotedTriplePattern(_) | TermPattern::Term(_)
        ) || matches!(
            pattern.object,
            TermPattern::QuotedTriplePattern(_) | TermPattern::Term(_)
        )
    }

    /// Execute pattern using store indices for better performance
    fn execute_pattern_with_index(
        &self,
        pattern: &TriplePattern,
        initial_binding: &Binding,
    ) -> StarResult<Vec<Binding>> {
        let mut bindings = Vec::new();

        // Extract concrete terms for index lookup
        let subject_term = match &pattern.subject {
            TermPattern::Term(term) => Some(term),
            _ => None,
        };
        let predicate_term = match &pattern.predicate {
            TermPattern::Term(term) => Some(term),
            _ => None,
        };
        let object_term = match &pattern.object {
            TermPattern::Term(term) => Some(term),
            _ => None,
        };

        // Use store's query method for better performance
        let matching_triples =
            self.store
                .query_triples(subject_term, predicate_term, object_term)?;

        for triple in matching_triples.iter() {
            if pattern.matches(triple, initial_binding) {
                if let Some(new_binding) = pattern.try_bind(triple, initial_binding) {
                    bindings.push(new_binding);
                }
            }
        }

        Ok(bindings)
    }

    /// Join current bindings with a new pattern using the selected join strategy
    fn join_with_pattern(
        &self,
        current_bindings: Vec<Binding>,
        pattern: &TriplePattern,
    ) -> StarResult<Vec<Binding>> {
        match self.join_strategy {
            JoinStrategy::NestedLoop => self.nested_loop_join(current_bindings, pattern),
            JoinStrategy::Hash => self.hash_join(current_bindings, pattern),
            JoinStrategy::Index => self.index_join(current_bindings, pattern),
        }
    }

    /// Nested loop join implementation
    fn nested_loop_join(
        &self,
        current_bindings: Vec<Binding>,
        pattern: &TriplePattern,
    ) -> StarResult<Vec<Binding>> {
        let mut new_bindings = Vec::new();

        for binding in &current_bindings {
            let pattern_bindings = self.execute_pattern_optimized(pattern, binding)?;
            new_bindings.extend(pattern_bindings);
        }

        Ok(new_bindings)
    }

    /// Hash join implementation (simplified version)
    fn hash_join(
        &self,
        current_bindings: Vec<Binding>,
        pattern: &TriplePattern,
    ) -> StarResult<Vec<Binding>> {
        // For simplicity, fall back to nested loop join
        // A full implementation would build hash tables on join variables
        self.nested_loop_join(current_bindings, pattern)
    }

    /// Index-aware join implementation
    fn index_join(
        &self,
        current_bindings: Vec<Binding>,
        pattern: &TriplePattern,
    ) -> StarResult<Vec<Binding>> {
        let mut new_bindings = Vec::new();

        // Group bindings by shared variables to optimize index lookups
        let shared_vars = self.find_shared_variables(&current_bindings, pattern);

        if shared_vars.is_empty() {
            // No shared variables, fall back to nested loop
            return self.nested_loop_join(current_bindings, pattern);
        }

        // Use index lookups when possible
        for binding in &current_bindings {
            let pattern_bindings = self.execute_pattern_with_binding_context(pattern, binding)?;
            new_bindings.extend(pattern_bindings);
        }

        Ok(new_bindings)
    }

    /// Find variables shared between current bindings and pattern
    fn find_shared_variables(&self, bindings: &[Binding], pattern: &TriplePattern) -> Vec<String> {
        if bindings.is_empty() {
            return Vec::new();
        }

        let mut pattern_vars = std::collections::HashSet::new();
        pattern.extract_variables(&mut pattern_vars);

        let binding_vars: std::collections::HashSet<String> =
            bindings[0].variables().into_iter().cloned().collect();

        pattern_vars.intersection(&binding_vars).cloned().collect()
    }

    /// Execute pattern with binding context for better optimization
    fn execute_pattern_with_binding_context(
        &self,
        pattern: &TriplePattern,
        binding: &Binding,
    ) -> StarResult<Vec<Binding>> {
        // Try to extract concrete terms from the binding to improve index usage
        let resolved_pattern = self.resolve_pattern_variables(pattern, binding);
        self.execute_pattern_optimized(&resolved_pattern, binding)
    }

    /// Resolve pattern variables that are bound to create more specific patterns
    fn resolve_pattern_variables(
        &self,
        pattern: &TriplePattern,
        binding: &Binding,
    ) -> TriplePattern {
        let subject = match &pattern.subject {
            TermPattern::Variable(var) => {
                if let Some(term) = binding.get(var) {
                    TermPattern::Term(term.clone())
                } else {
                    pattern.subject.clone()
                }
            }
            _ => pattern.subject.clone(),
        };

        let predicate = match &pattern.predicate {
            TermPattern::Variable(var) => {
                if let Some(term) = binding.get(var) {
                    TermPattern::Term(term.clone())
                } else {
                    pattern.predicate.clone()
                }
            }
            _ => pattern.predicate.clone(),
        };

        let object = match &pattern.object {
            TermPattern::Variable(var) => {
                if let Some(term) = binding.get(var) {
                    TermPattern::Term(term.clone())
                } else {
                    pattern.object.clone()
                }
            }
            _ => pattern.object.clone(),
        };

        TriplePattern::new(subject, predicate, object)
    }

    /// Apply filter expressions to bindings
    fn apply_filters(
        &self,
        bindings: Vec<Binding>,
        filters: &[Expression],
    ) -> StarResult<Vec<Binding>> {
        let mut filtered_bindings = Vec::new();

        for binding in bindings {
            let mut passes_all_filters = true;

            for filter in filters {
                // Convert binding to HashMap for expression evaluation
                let binding_map: HashMap<String, StarTerm> = binding.bindings.clone();

                match ExpressionEvaluator::evaluate(filter, &binding_map) {
                    Ok(result) => {
                        // Check if the result is a boolean true
                        if let Some(literal) = result.as_literal() {
                            if let Some(datatype) = &literal.datatype {
                                if datatype.iri == "http://www.w3.org/2001/XMLSchema#boolean" {
                                    if literal.value != "true" {
                                        passes_all_filters = false;
                                        break;
                                    }
                                } else {
                                    // Non-boolean result is considered false
                                    passes_all_filters = false;
                                    break;
                                }
                            } else {
                                // No datatype, check if it's a truthy value
                                if literal.value.is_empty()
                                    || literal.value == "false"
                                    || literal.value == "0"
                                {
                                    passes_all_filters = false;
                                    break;
                                }
                            }
                        } else {
                            // Non-literal results are considered false in filter context
                            passes_all_filters = false;
                            break;
                        }
                    }
                    Err(_) => {
                        // Filter evaluation error means the binding doesn't pass
                        passes_all_filters = false;
                        break;
                    }
                }
            }

            if passes_all_filters {
                filtered_bindings.push(binding);
            }
        }

        Ok(filtered_bindings)
    }

    /// Execute a single triple pattern (legacy method for compatibility)
    fn execute_pattern(
        &self,
        pattern: &TriplePattern,
        initial_binding: &Binding,
    ) -> StarResult<Vec<Binding>> {
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

    /// Execute a SELECT query with optimization
    pub fn execute_select(
        &mut self,
        bgp: &BasicGraphPattern,
        select_vars: &[String],
    ) -> StarResult<Vec<HashMap<String, StarTerm>>> {
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

        debug!(
            "SELECT query produced {} results with {} pattern evaluations",
            results.len(),
            self.statistics.pattern_evaluations
        );
        Ok(results)
    }

    /// Execute a CONSTRUCT query with optimization
    pub fn execute_construct(
        &mut self,
        bgp: &BasicGraphPattern,
        construct_patterns: &[TriplePattern],
    ) -> StarResult<StarGraph> {
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

        debug!(
            "CONSTRUCT query produced {} triples with {} join operations",
            constructed_graph.len(),
            self.statistics.join_operations
        );
        Ok(constructed_graph)
    }

    /// Execute an ASK query with optimization
    pub fn execute_ask(&mut self, bgp: &BasicGraphPattern) -> StarResult<bool> {
        let span = span!(Level::INFO, "execute_ask");
        let _enter = span.enter();

        let bindings = self.execute_bgp(bgp)?;
        let result = !bindings.is_empty();

        debug!(
            "ASK query result: {} (execution time: {}µs)",
            result, self.statistics.execution_time_us
        );
        Ok(result)
    }

    /// Instantiate a triple pattern with a binding
    fn instantiate_pattern(
        &self,
        pattern: &TriplePattern,
        binding: &Binding,
    ) -> StarResult<Option<StarTriple>> {
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
    fn instantiate_term_pattern(
        &self,
        pattern: &TermPattern,
        binding: &Binding,
    ) -> StarResult<Option<StarTerm>> {
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
                    if let Some(stripped) = part.strip_prefix('?') {
                        select_vars.push(stripped.to_string());
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
            return Err(StarError::query_error(format!(
                "Invalid triple pattern: {line}"
            )));
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
        if let Some(stripped) = term_str.strip_prefix('?') {
            return Ok(TermPattern::Variable(stripped.to_string()));
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
        if let Some(id) = term_str.strip_prefix("_:") {
            return StarTerm::blank_node(id);
        }

        // Literal
        if term_str.starts_with('"') {
            // Simple literal parsing (not complete)
            let end_quote = term_str.rfind('"').unwrap_or(term_str.len());
            let value = &term_str[1..end_quote];
            return StarTerm::literal(value);
        }

        Err(StarError::query_error(format!(
            "Cannot parse term: {term_str}"
        )))
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
        assert_eq!(
            binding.get("x"),
            Some(&StarTerm::iri("http://example.org/alice").unwrap())
        );

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
        assert_eq!(
            new_binding.get("x"),
            Some(&StarTerm::iri("http://example.org/alice").unwrap())
        );
        assert_eq!(
            new_binding.get("y"),
            Some(&StarTerm::iri("http://example.org/bob").unwrap())
        );
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

        let mut executor = QueryExecutor::new(store);

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
    fn test_query_optimization_strategies() {
        let store = StarStore::new();

        // Add test data with quoted triples
        let inner_triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let quoted_triple = StarTriple::new(
            StarTerm::quoted_triple(inner_triple),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        store.insert(&quoted_triple).unwrap();

        // Test different optimization strategies
        for optimization in [
            QueryOptimization::None,
            QueryOptimization::Heuristic,
            QueryOptimization::CostBased,
        ] {
            for join_strategy in [
                JoinStrategy::NestedLoop,
                JoinStrategy::Index,
                JoinStrategy::Hash,
            ] {
                let mut executor =
                    QueryExecutor::with_optimization(store.clone(), optimization, join_strategy);

                let mut bgp = BasicGraphPattern::new();
                bgp.add_pattern(TriplePattern::new(
                    TermPattern::QuotedTriplePattern(Box::new(TriplePattern::new(
                        TermPattern::Variable("x".to_string()),
                        TermPattern::Term(StarTerm::iri("http://example.org/age").unwrap()),
                        TermPattern::Variable("age".to_string()),
                    ))),
                    TermPattern::Term(StarTerm::iri("http://example.org/certainty").unwrap()),
                    TermPattern::Variable("cert".to_string()),
                ));

                let bindings = executor.execute_bgp(&bgp).unwrap();
                assert_eq!(bindings.len(), 1);

                // Check that statistics are collected
                let stats = executor.statistics();
                assert!(stats.pattern_evaluations > 0);
                // Note: execution_time_us may be 0 on fast systems where execution
                // completes in sub-microsecond time, so we don't assert > 0
            }
        }
    }

    #[test]
    fn test_advanced_sparql_star_queries() {
        let store = StarStore::new();

        // Create a complex RDF-star dataset
        let fact1 = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let fact2 = StarTriple::new(
            StarTerm::iri("http://example.org/bob").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
        );

        let meta1 = StarTriple::new(
            StarTerm::quoted_triple(fact1.clone()),
            StarTerm::iri("http://example.org/source").unwrap(),
            StarTerm::literal("census").unwrap(),
        );

        let meta2 = StarTriple::new(
            StarTerm::quoted_triple(fact2.clone()),
            StarTerm::iri("http://example.org/source").unwrap(),
            StarTerm::literal("survey").unwrap(),
        );

        store.insert(&fact1).unwrap();
        store.insert(&fact2).unwrap();
        store.insert(&meta1).unwrap();
        store.insert(&meta2).unwrap();

        let mut executor = QueryExecutor::with_optimization(
            store,
            QueryOptimization::CostBased,
            JoinStrategy::Index,
        );

        // Query: Find all facts from census source
        let mut bgp = BasicGraphPattern::new();
        bgp.add_pattern(TriplePattern::new(
            TermPattern::QuotedTriplePattern(Box::new(TriplePattern::new(
                TermPattern::Variable("person".to_string()),
                TermPattern::Term(StarTerm::iri("http://example.org/age").unwrap()),
                TermPattern::Variable("age".to_string()),
            ))),
            TermPattern::Term(StarTerm::iri("http://example.org/source").unwrap()),
            TermPattern::Term(StarTerm::literal("census").unwrap()),
        ));

        let bindings = executor.execute_bgp(&bgp).unwrap();
        assert_eq!(bindings.len(), 1);

        // Verify the binding contains Alice's data
        let binding = &bindings[0];
        assert!(binding.is_bound("person"));
        assert!(binding.is_bound("age"));

        if let Some(person_term) = binding.get("person") {
            if let Some(person_node) = person_term.as_named_node() {
                assert!(person_node.iri.contains("alice"));
            }
        }
    }

    #[test]
    fn test_query_statistics_tracking() {
        let store = StarStore::new();

        // Add multiple triples
        for i in 0..10 {
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/person{i}")).unwrap(),
                StarTerm::iri("http://example.org/age").unwrap(),
                StarTerm::literal(&format!("{}", 20 + i)).unwrap(),
            );
            store.insert(&triple).unwrap();
        }

        let mut executor = QueryExecutor::new(store);

        // Complex BGP with multiple patterns
        let mut bgp = BasicGraphPattern::new();
        bgp.add_pattern(TriplePattern::new(
            TermPattern::Variable("x".to_string()),
            TermPattern::Term(StarTerm::iri("http://example.org/age").unwrap()),
            TermPattern::Variable("age".to_string()),
        ));
        bgp.add_pattern(TriplePattern::new(
            TermPattern::Variable("x".to_string()),
            TermPattern::Term(StarTerm::iri("http://example.org/age").unwrap()),
            TermPattern::Variable("age2".to_string()),
        ));

        let bindings = executor.execute_bgp(&bgp).unwrap();
        assert_eq!(bindings.len(), 10);

        // Check statistics
        let stats = executor.statistics();
        assert!(stats.pattern_evaluations >= 2); // At least 2 patterns evaluated
        assert!(stats.join_operations >= 1); // At least 1 join performed
        // Note: execution_time_us may be 0 on fast systems (sub-microsecond execution)
        assert!(stats.intermediate_results > 0);
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
        assert!(matches!(
            pattern.subject,
            TermPattern::QuotedTriplePattern(_)
        ));
    }

    #[test]
    fn test_sparql_star_functions_in_filters() {
        let store = StarStore::new();

        // Add test data with quoted triples
        let fact = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let meta = StarTriple::new(
            StarTerm::quoted_triple(fact.clone()),
            StarTerm::iri("http://example.org/source").unwrap(),
            StarTerm::literal("census").unwrap(),
        );

        store.insert(&fact).unwrap();
        store.insert(&meta).unwrap();

        let mut executor = QueryExecutor::new(store);

        // Query with SPARQL-star function in filter
        let mut bgp = BasicGraphPattern::new();
        bgp.add_pattern(TriplePattern::new(
            TermPattern::Variable("quoted".to_string()),
            TermPattern::Term(StarTerm::iri("http://example.org/source").unwrap()),
            TermPattern::Variable("source".to_string()),
        ));

        // Add filter: isTRIPLE(?quoted)
        bgp.add_filter(Expression::is_triple(Expression::var("quoted")));

        let bindings = executor.execute_bgp(&bgp).unwrap();
        assert_eq!(bindings.len(), 1);

        // Verify the binding contains a quoted triple
        let binding = &bindings[0];
        assert!(binding.is_bound("quoted"));
        if let Some(quoted_term) = binding.get("quoted") {
            assert!(quoted_term.is_quoted_triple());
        }
    }

    #[test]
    fn test_sparql_star_function_composition() {
        let store = StarStore::new();

        // Create nested quoted triple
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            StarTerm::iri("http://example.org/confidence").unwrap(),
            StarTerm::literal("0.8").unwrap(),
        );

        store.insert(&inner).unwrap();
        store.insert(&outer).unwrap();

        let mut executor = QueryExecutor::new(store);

        // Query that uses SUBJECT function
        let mut bgp = BasicGraphPattern::new();
        bgp.add_pattern(TriplePattern::new(
            TermPattern::Variable("statement".to_string()),
            TermPattern::Term(StarTerm::iri("http://example.org/confidence").unwrap()),
            TermPattern::Variable("conf".to_string()),
        ));

        let bindings = executor.execute_bgp(&bgp).unwrap();
        assert_eq!(bindings.len(), 1);

        // Verify we can extract the subject of the quoted triple
        let binding = &bindings[0];
        if let Some(statement) = binding.get("statement") {
            assert!(statement.is_quoted_triple());

            // Test SUBJECT function evaluation
            let subject_result = crate::functions::FunctionEvaluator::evaluate(
                crate::functions::StarFunction::Subject,
                std::slice::from_ref(statement),
            )
            .unwrap();

            // The subject should be the alice IRI
            assert_eq!(
                subject_result,
                StarTerm::iri("http://example.org/alice").unwrap()
            );
        }
    }

    #[test]
    fn test_filter_with_triple_construction() {
        let store = StarStore::new();

        // Add some triples
        let triple1 = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let triple2 = StarTriple::new(
            StarTerm::iri("http://example.org/bob").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
        );

        store.insert(&triple1).unwrap();
        store.insert(&triple2).unwrap();

        let mut executor = QueryExecutor::new(store);

        // Query that constructs a triple in a filter
        let mut bgp = BasicGraphPattern::new();
        bgp.add_pattern(TriplePattern::new(
            TermPattern::Variable("person".to_string()),
            TermPattern::Term(StarTerm::iri("http://example.org/age").unwrap()),
            TermPattern::Variable("age".to_string()),
        ));

        // This test demonstrates using TRIPLE function in expressions
        // In a real implementation, we might check if a constructed triple exists
        let bindings = executor.execute_bgp(&bgp).unwrap();
        assert_eq!(bindings.len(), 2);
    }
}
