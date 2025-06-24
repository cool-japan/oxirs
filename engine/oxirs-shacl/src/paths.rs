//! SHACL property path implementation
//! 
//! This module handles property path evaluation according to SHACL specification.

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};

use oxirs_core::{
    model::{NamedNode, Term, Triple, BlankNode, RdfTerm},
    store::Store,
    OxirsError,
};

use crate::{ShaclError, Result};

/// SHACL property path types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyPath {
    /// Simple property path (single predicate)
    Predicate(NamedNode),
    
    /// Inverse property path (^predicate)
    Inverse(Box<PropertyPath>),
    
    /// Sequence path (path1 / path2 / ...)
    Sequence(Vec<PropertyPath>),
    
    /// Alternative path (path1 | path2 | ...)
    Alternative(Vec<PropertyPath>),
    
    /// Zero-or-more path (path*)
    ZeroOrMore(Box<PropertyPath>),
    
    /// One-or-more path (path+)
    OneOrMore(Box<PropertyPath>),
    
    /// Zero-or-one path (path?)
    ZeroOrOne(Box<PropertyPath>),
}

impl PropertyPath {
    /// Create a simple predicate path
    pub fn predicate(predicate: NamedNode) -> Self {
        PropertyPath::Predicate(predicate)
    }
    
    /// Create an inverse path
    pub fn inverse(path: PropertyPath) -> Self {
        PropertyPath::Inverse(Box::new(path))
    }
    
    /// Create a sequence path
    pub fn sequence(paths: Vec<PropertyPath>) -> Self {
        PropertyPath::Sequence(paths)
    }
    
    /// Create an alternative path
    pub fn alternative(paths: Vec<PropertyPath>) -> Self {
        PropertyPath::Alternative(paths)
    }
    
    /// Create a zero-or-more path
    pub fn zero_or_more(path: PropertyPath) -> Self {
        PropertyPath::ZeroOrMore(Box::new(path))
    }
    
    /// Create a one-or-more path
    pub fn one_or_more(path: PropertyPath) -> Self {
        PropertyPath::OneOrMore(Box::new(path))
    }
    
    /// Create a zero-or-one path
    pub fn zero_or_one(path: PropertyPath) -> Self {
        PropertyPath::ZeroOrOne(Box::new(path))
    }
    
    /// Check if this is a simple predicate path
    pub fn is_predicate(&self) -> bool {
        matches!(self, PropertyPath::Predicate(_))
    }
    
    /// Get the predicate if this is a simple predicate path
    pub fn as_predicate(&self) -> Option<&NamedNode> {
        match self {
            PropertyPath::Predicate(p) => Some(p),
            _ => None,
        }
    }
    
    /// Check if this path involves complex operations (non-predicate)
    pub fn is_complex(&self) -> bool {
        !self.is_predicate()
    }
    
    /// Estimate the complexity of this path for optimization
    pub fn complexity(&self) -> usize {
        match self {
            PropertyPath::Predicate(_) => 1,
            PropertyPath::Inverse(path) => path.complexity() + 1,
            PropertyPath::Sequence(paths) => paths.iter().map(|p| p.complexity()).sum::<usize>() + 1,
            PropertyPath::Alternative(paths) => paths.iter().map(|p| p.complexity()).max().unwrap_or(0) + 1,
            PropertyPath::ZeroOrMore(path) => path.complexity() * 10, // High complexity due to recursion
            PropertyPath::OneOrMore(path) => path.complexity() * 8,
            PropertyPath::ZeroOrOne(path) => path.complexity() + 1,
        }
    }
}

/// Property path evaluator for finding values along paths
#[derive(Debug)]
pub struct PropertyPathEvaluator {
    /// Cache for path evaluation results
    cache: HashMap<String, Vec<Term>>,
    
    /// Maximum recursion depth for cyclic paths
    max_depth: usize,
    
    /// Maximum number of intermediate results to track
    max_intermediate_results: usize,
}

impl PropertyPathEvaluator {
    /// Create a new property path evaluator
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            max_depth: 50,
            max_intermediate_results: 10000,
        }
    }
    
    /// Create a new evaluator with custom limits
    pub fn with_limits(max_depth: usize, max_intermediate_results: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_depth,
            max_intermediate_results,
        }
    }
    
    /// Evaluate a property path from a starting node
    pub fn evaluate_path(&mut self, store: &Store, start_node: &Term, path: &PropertyPath, graph_name: Option<&str>) -> Result<Vec<Term>> {
        let cache_key = self.create_cache_key(start_node, path, graph_name);
        
        // Check cache first
        if let Some(cached_result) = self.cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }
        
        let result = self.evaluate_path_impl(store, start_node, path, graph_name, 0)?;
        
        // Cache the result
        self.cache.insert(cache_key, result.clone());
        
        Ok(result)
    }
    
    /// Evaluate multiple paths from the same starting node
    pub fn evaluate_multiple_paths(&mut self, store: &Store, start_node: &Term, paths: &[PropertyPath], graph_name: Option<&str>) -> Result<Vec<Term>> {
        let mut all_values = HashSet::new();
        
        for path in paths {
            let values = self.evaluate_path(store, start_node, path, graph_name)?;
            all_values.extend(values);
        }
        
        Ok(all_values.into_iter().collect())
    }
    
    /// Internal implementation of path evaluation
    fn evaluate_path_impl(&self, store: &Store, start_node: &Term, path: &PropertyPath, graph_name: Option<&str>, depth: usize) -> Result<Vec<Term>> {
        // Check recursion depth
        if depth > self.max_depth {
            return Err(ShaclError::PropertyPath(
                format!("Maximum recursion depth {} exceeded for property path evaluation", self.max_depth)
            ));
        }
        
        match path {
            PropertyPath::Predicate(predicate) => {
                self.evaluate_predicate(store, start_node, predicate, graph_name)
            }
            PropertyPath::Inverse(inner_path) => {
                self.evaluate_inverse(store, start_node, inner_path, graph_name, depth)
            }
            PropertyPath::Sequence(paths) => {
                self.evaluate_sequence(store, start_node, paths, graph_name, depth)
            }
            PropertyPath::Alternative(paths) => {
                self.evaluate_alternative(store, start_node, paths, graph_name, depth)
            }
            PropertyPath::ZeroOrMore(inner_path) => {
                self.evaluate_zero_or_more(store, start_node, inner_path, graph_name, depth)
            }
            PropertyPath::OneOrMore(inner_path) => {
                self.evaluate_one_or_more(store, start_node, inner_path, graph_name, depth)
            }
            PropertyPath::ZeroOrOne(inner_path) => {
                self.evaluate_zero_or_one(store, start_node, inner_path, graph_name, depth)
            }
        }
    }
    
    /// Evaluate a simple predicate path
    fn evaluate_predicate(&self, store: &Store, start_node: &Term, predicate: &NamedNode, graph_name: Option<&str>) -> Result<Vec<Term>> {
        let mut values = Vec::new();
        
        // Create a SPARQL query to find all values connected via the predicate
        let query = if let Some(graph) = graph_name {
            format!(r#"
                SELECT DISTINCT ?value WHERE {{
                    GRAPH <{}> {{
                        {} <{}> ?value .
                    }}
                }}
            "#, graph, format_term_for_sparql(start_node)?, predicate.as_str())
        } else {
            format!(r#"
                SELECT DISTINCT ?value WHERE {{
                    {} <{}> ?value .
                }}
            "#, format_term_for_sparql(start_node)?, predicate.as_str())
        };
        
        // TODO: Execute SPARQL query using store interface
        // For now, we'll provide a structured approach for when the interface is available
        
        // Placeholder implementation - log the query that would be executed
        tracing::debug!("Property path predicate query: {}", query);
        
        // TODO: Replace with actual store.query() call when available
        // let results = store.query(&query)?;
        // for solution in results.solutions() {
        //     if let Some(value) = solution.get("value") {
        //         values.push(value.clone());
        //     }
        // }
        
        Ok(values)
    }
    
    /// Evaluate an inverse path
    fn evaluate_inverse(&self, store: &Store, start_node: &Term, inner_path: &PropertyPath, graph_name: Option<&str>, depth: usize) -> Result<Vec<Term>> {
        match inner_path {
            PropertyPath::Predicate(predicate) => {
                // For inverse of simple predicate, find subjects where start_node is object
                self.evaluate_inverse_predicate(store, start_node, predicate, graph_name)
            }
            _ => {
                // For complex paths, we need to find all nodes that reach start_node via the path
                self.evaluate_complex_inverse(store, start_node, inner_path, graph_name, depth)
            }
        }
    }
    
    /// Evaluate inverse of a simple predicate
    fn evaluate_inverse_predicate(&self, store: &Store, start_node: &Term, predicate: &NamedNode, graph_name: Option<&str>) -> Result<Vec<Term>> {
        let mut values = Vec::new();
        
        // Create a SPARQL query to find all values that connect to start_node via the predicate
        let query = if let Some(graph) = graph_name {
            format!(r#"
                SELECT DISTINCT ?value WHERE {{
                    GRAPH <{}> {{
                        ?value <{}> {} .
                    }}
                }}
            "#, graph, predicate.as_str(), format_term_for_sparql(start_node)?)
        } else {
            format!(r#"
                SELECT DISTINCT ?value WHERE {{
                    ?value <{}> {} .
                }}
            "#, predicate.as_str(), format_term_for_sparql(start_node)?)
        };
        
        // TODO: Execute SPARQL query using store interface
        tracing::debug!("Inverse property path query: {}", query);
        
        // TODO: Replace with actual store.query() call when available
        // let results = store.query(&query)?;
        // for solution in results.solutions() {
        //     if let Some(value) = solution.get("value") {
        //         values.push(value.clone());
        //     }
        // }
        
        Ok(values)
    }
    
    /// Evaluate inverse of a complex path
    fn evaluate_complex_inverse(&self, store: &Store, start_node: &Term, inner_path: &PropertyPath, graph_name: Option<&str>, depth: usize) -> Result<Vec<Term>> {
        // This is computationally expensive - we need to find all nodes that can reach start_node
        // We'll implement this using a recursive approach with caching
        
        let mut result = Vec::new();
        let mut candidates = HashSet::new();
        
        // Strategy: Find potential candidates and test if they reach start_node via the path
        // This is expensive but necessary for correctness
        
        // First, collect all potential starting nodes from the graph
        // This is a simplified approach - in practice we'd want to be smarter about candidate selection
        
        let candidate_query = if let Some(graph) = graph_name {
            format!(r#"
                SELECT DISTINCT ?candidate WHERE {{
                    GRAPH <{}> {{
                        ?candidate ?p ?o .
                    }}
                }}
                LIMIT 10000
            "#, graph)
        } else {
            "SELECT DISTINCT ?candidate WHERE { ?candidate ?p ?o . } LIMIT 10000".to_string()
        };
        
        // TODO: Execute query to get candidates
        tracing::debug!("Complex inverse path candidate query: {}", candidate_query);
        
        // For each candidate, test if it reaches start_node via the inner path
        // TODO: Implement candidate testing
        // for candidate in candidates {
        //     let path_values = self.evaluate_path_impl(store, &candidate, inner_path, graph_name, depth + 1)?;
        //     if path_values.contains(start_node) {
        //         result.push(candidate);
        //     }
        // }
        
        // For now, return empty result with warning
        tracing::warn!("Complex inverse path evaluation not fully implemented - performance optimization needed");
        
        Ok(result)
    }
    
    /// Evaluate a sequence path (path1 / path2 / ...)
    fn evaluate_sequence(&self, store: &Store, start_node: &Term, paths: &[PropertyPath], graph_name: Option<&str>, depth: usize) -> Result<Vec<Term>> {
        if paths.is_empty() {
            return Ok(vec![start_node.clone()]);
        }
        
        let mut current_nodes = vec![start_node.clone()];
        
        for path in paths {
            let mut next_nodes = Vec::new();
            
            for node in &current_nodes {
                let path_values = self.evaluate_path_impl(store, node, path, graph_name, depth + 1)?;
                next_nodes.extend(path_values);
                
                // Check if we exceed intermediate results limit
                if next_nodes.len() > self.max_intermediate_results {
                    return Err(ShaclError::PropertyPath(
                        format!("Too many intermediate results ({}), limit is {}", 
                               next_nodes.len(), self.max_intermediate_results)
                    ));
                }
            }
            
            current_nodes = next_nodes;
            
            // If no results at any step, the sequence fails
            if current_nodes.is_empty() {
                break;
            }
        }
        
        Ok(current_nodes)
    }
    
    /// Evaluate an alternative path (path1 | path2 | ...)
    fn evaluate_alternative(&self, store: &Store, start_node: &Term, paths: &[PropertyPath], graph_name: Option<&str>, depth: usize) -> Result<Vec<Term>> {
        let mut all_values = HashSet::new();
        
        for path in paths {
            let path_values = self.evaluate_path_impl(store, start_node, path, graph_name, depth + 1)?;
            all_values.extend(path_values);
        }
        
        Ok(all_values.into_iter().collect())
    }
    
    /// Evaluate a zero-or-more path (path*)
    fn evaluate_zero_or_more(&self, store: &Store, start_node: &Term, inner_path: &PropertyPath, graph_name: Option<&str>, depth: usize) -> Result<Vec<Term>> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        // Include the starting node (zero iterations)
        result.insert(start_node.clone());
        queue.push_back(start_node.clone());
        visited.insert(start_node.clone());
        
        while let Some(current) = queue.pop_front() {
            // Check limits
            if result.len() > self.max_intermediate_results {
                return Err(ShaclError::PropertyPath(
                    format!("Too many results in zero-or-more path ({}), limit is {}", 
                           result.len(), self.max_intermediate_results)
                ));
            }
            
            let path_values = self.evaluate_path_impl(store, &current, inner_path, graph_name, depth + 1)?;
            
            for value in path_values {
                if !visited.contains(&value) {
                    visited.insert(value.clone());
                    result.insert(value.clone());
                    queue.push_back(value);
                }
            }
        }
        
        Ok(result.into_iter().collect())
    }
    
    /// Evaluate a one-or-more path (path+)
    fn evaluate_one_or_more(&self, store: &Store, start_node: &Term, inner_path: &PropertyPath, graph_name: Option<&str>, depth: usize) -> Result<Vec<Term>> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        // Start with one iteration (don't include starting node)
        let initial_values = self.evaluate_path_impl(store, start_node, inner_path, graph_name, depth + 1)?;
        
        for value in initial_values {
            result.insert(value.clone());
            queue.push_back(value.clone());
            visited.insert(value);
        }
        
        while let Some(current) = queue.pop_front() {
            // Check limits
            if result.len() > self.max_intermediate_results {
                return Err(ShaclError::PropertyPath(
                    format!("Too many results in one-or-more path ({}), limit is {}", 
                           result.len(), self.max_intermediate_results)
                ));
            }
            
            let path_values = self.evaluate_path_impl(store, &current, inner_path, graph_name, depth + 1)?;
            
            for value in path_values {
                if !visited.contains(&value) {
                    visited.insert(value.clone());
                    result.insert(value.clone());
                    queue.push_back(value);
                }
            }
        }
        
        Ok(result.into_iter().collect())
    }
    
    /// Evaluate a zero-or-one path (path?)
    fn evaluate_zero_or_one(&self, store: &Store, start_node: &Term, inner_path: &PropertyPath, graph_name: Option<&str>, depth: usize) -> Result<Vec<Term>> {
        let mut result = vec![start_node.clone()]; // Include starting node (zero iterations)
        
        let path_values = self.evaluate_path_impl(store, start_node, inner_path, graph_name, depth + 1)?;
        result.extend(path_values);
        
        // Remove duplicates
        let unique_result: HashSet<_> = result.into_iter().collect();
        Ok(unique_result.into_iter().collect())
    }
    
    /// Create a cache key for path evaluation
    fn create_cache_key(&self, start_node: &Term, path: &PropertyPath, graph_name: Option<&str>) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        start_node.as_str().hash(&mut hasher);
        path.hash(&mut hasher);
        graph_name.hash(&mut hasher);
        
        format!("path_eval_{}", hasher.finish())
    }
    
    /// Clear the evaluation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> PathCacheStats {
        PathCacheStats {
            entries: self.cache.len(),
            total_values: self.cache.values().map(|v| v.len()).sum(),
        }
    }
    
    /// Set maximum recursion depth
    pub fn set_max_depth(&mut self, max_depth: usize) {
        self.max_depth = max_depth;
    }
    
    /// Set maximum intermediate results limit
    pub fn set_max_intermediate_results(&mut self, max_results: usize) {
        self.max_intermediate_results = max_results;
    }
}

impl Default for PropertyPathEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about property path evaluation cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathCacheStats {
    pub entries: usize,
    pub total_values: usize,
}

/// Property path optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathOptimizationHints {
    /// Cache simple predicate path results
    pub cache_simple_paths: bool,
    
    /// Cache complex path results
    pub cache_complex_paths: bool,
    
    /// Maximum cache size for path results
    pub max_cache_size: usize,
    
    /// Parallel evaluation threshold
    pub parallel_threshold: usize,
    
    /// Maximum recursion depth for cyclic paths
    pub max_recursion_depth: usize,
    
    /// Maximum intermediate results
    pub max_intermediate_results: usize,
}

impl Default for PathOptimizationHints {
    fn default() -> Self {
        Self {
            cache_simple_paths: true,
            cache_complex_paths: false, // Complex paths change frequently
            max_cache_size: 5000,
            parallel_threshold: 100,
            max_recursion_depth: 50,
            max_intermediate_results: 10000,
        }
    }
}

/// Property path validation context
#[derive(Debug, Clone)]
pub struct PathValidationContext {
    /// Current recursion depth
    pub depth: usize,
    
    /// Visited nodes (for cycle detection)
    pub visited: HashSet<Term>,
    
    /// Path being evaluated
    pub current_path: PropertyPath,
    
    /// Performance statistics
    pub stats: PathEvaluationStats,
}

impl PathValidationContext {
    pub fn new(path: PropertyPath) -> Self {
        Self {
            depth: 0,
            visited: HashSet::new(),
            current_path: path,
            stats: PathEvaluationStats::default(),
        }
    }
}

/// Property path evaluation performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathEvaluationStats {
    pub total_evaluations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub total_values_found: usize,
    pub avg_values_per_evaluation: f64,
    pub max_recursion_depth_reached: usize,
}

impl PathEvaluationStats {
    pub fn record_evaluation(&mut self, values_found: usize, cache_hit: bool, depth: usize) {
        self.total_evaluations += 1;
        self.total_values_found += values_found;
        
        if cache_hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }
        
        if depth > self.max_recursion_depth_reached {
            self.max_recursion_depth_reached = depth;
        }
        
        self.avg_values_per_evaluation = self.total_values_found as f64 / self.total_evaluations as f64;
    }
    
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_evaluations as f64
        }
    }
}

/// Format a term for use in SPARQL queries
fn format_term_for_sparql(term: &Term) -> Result<String> {
    match term {
        Term::NamedNode(node) => Ok(format!("<{}>", node.as_str())),
        Term::BlankNode(node) => Ok(node.as_str().to_string()),
        Term::Literal(literal) => {
            // TODO: Proper literal formatting with datatype and language
            Ok(format!("\"{}\"", literal.as_str().replace('"', "\\\"")))
        }
        Term::Variable(var) => Ok(format!("?{}", var.name())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_property_path_creation() {
        let predicate = NamedNode::new("http://example.org/knows").unwrap();
        let path = PropertyPath::predicate(predicate.clone());
        
        assert!(path.is_predicate());
        assert_eq!(path.as_predicate(), Some(&predicate));
        assert!(!path.is_complex());
        assert_eq!(path.complexity(), 1);
    }
    
    #[test]
    fn test_inverse_path() {
        let predicate = NamedNode::new("http://example.org/knows").unwrap();
        let path = PropertyPath::inverse(PropertyPath::predicate(predicate));
        
        assert!(!path.is_predicate());
        assert!(path.is_complex());
        assert_eq!(path.complexity(), 2);
    }
    
    #[test]
    fn test_sequence_path() {
        let pred1 = NamedNode::new("http://example.org/knows").unwrap();
        let pred2 = NamedNode::new("http://example.org/friend").unwrap();
        
        let path = PropertyPath::sequence(vec![
            PropertyPath::predicate(pred1),
            PropertyPath::predicate(pred2),
        ]);
        
        assert!(!path.is_predicate());
        assert!(path.is_complex());
        assert_eq!(path.complexity(), 3); // 1 + 1 + 1 for sequence overhead
    }
    
    #[test]
    fn test_zero_or_more_path() {
        let predicate = NamedNode::new("http://example.org/knows").unwrap();
        let path = PropertyPath::zero_or_more(PropertyPath::predicate(predicate));
        
        assert!(!path.is_predicate());
        assert!(path.is_complex());
        assert_eq!(path.complexity(), 10); // High complexity due to recursion
    }
    
    #[test]
    fn test_alternative_path() {
        let pred1 = NamedNode::new("http://example.org/knows").unwrap();
        let pred2 = NamedNode::new("http://example.org/friend").unwrap();
        
        let path = PropertyPath::alternative(vec![
            PropertyPath::predicate(pred1),
            PropertyPath::predicate(pred2),
        ]);
        
        assert!(!path.is_predicate());
        assert!(path.is_complex());
        assert_eq!(path.complexity(), 2); // max(1, 1) + 1 for alternative overhead
    }
    
    #[test]
    fn test_path_evaluator_creation() {
        let evaluator = PropertyPathEvaluator::new();
        assert_eq!(evaluator.max_depth, 50);
        assert_eq!(evaluator.max_intermediate_results, 10000);
        
        let custom_evaluator = PropertyPathEvaluator::with_limits(100, 5000);
        assert_eq!(custom_evaluator.max_depth, 100);
        assert_eq!(custom_evaluator.max_intermediate_results, 5000);
    }
    
    #[test]
    fn test_path_evaluation_stats() {
        let mut stats = PathEvaluationStats::default();
        
        stats.record_evaluation(5, false, 2);  // cache miss, depth 2
        stats.record_evaluation(3, true, 1);   // cache hit, depth 1
        stats.record_evaluation(7, false, 3);  // cache miss, depth 3
        
        assert_eq!(stats.total_evaluations, 3);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 2);
        assert_eq!(stats.total_values_found, 15);
        assert_eq!(stats.avg_values_per_evaluation, 5.0);
        assert_eq!(stats.max_recursion_depth_reached, 3);
        assert_eq!(stats.cache_hit_rate(), 1.0 / 3.0);
    }
}