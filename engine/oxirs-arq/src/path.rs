//! Property Path Support for SPARQL
//!
//! This module implements SPARQL 1.1 property paths, allowing complex navigation
//! through RDF graphs using path expressions.

use crate::algebra::{Term, Variable, Iri, Binding, Solution};
use anyhow::{Result, anyhow, bail};
use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};

/// Property path expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyPath {
    /// Direct property path (predicate)
    Direct(Term),
    
    /// Inverse property path (^predicate)
    Inverse(Box<PropertyPath>),
    
    /// Sequence path (path1/path2)
    Sequence(Box<PropertyPath>, Box<PropertyPath>),
    
    /// Alternative path (path1|path2)
    Alternative(Box<PropertyPath>, Box<PropertyPath>),
    
    /// Zero-or-more path (path*)
    ZeroOrMore(Box<PropertyPath>),
    
    /// One-or-more path (path+)
    OneOrMore(Box<PropertyPath>),
    
    /// Zero-or-one path (path?)
    ZeroOrOne(Box<PropertyPath>),
    
    /// Negated property set (!(predicate1|predicate2|...))
    NegatedPropertySet(Vec<Term>),
}

/// Property path evaluation context
#[derive(Debug, Clone)]
pub struct PathContext {
    /// Maximum path length to prevent infinite loops
    pub max_length: usize,
    /// Maximum number of intermediate nodes
    pub max_nodes: usize,
    /// Enable cycle detection
    pub cycle_detection: bool,
    /// Optimization level
    pub optimization_level: PathOptimization,
}

/// Path optimization strategies
#[derive(Debug, Clone, PartialEq)]
pub enum PathOptimization {
    None,
    Basic,
    Advanced,
}

impl Default for PathContext {
    fn default() -> Self {
        Self {
            max_length: 100,
            max_nodes: 10000,
            cycle_detection: true,
            optimization_level: PathOptimization::Basic,
        }
    }
}

/// Property path evaluator
pub struct PropertyPathEvaluator {
    context: PathContext,
}

/// Path evaluation result
#[derive(Debug, Clone)]
pub struct PathResult {
    pub bindings: Solution,
    pub path_length: usize,
    pub visited_nodes: usize,
}

/// Dataset interface for property path evaluation
pub trait PathDataset {
    /// Find all triples matching subject and predicate
    fn find_outgoing(&self, subject: &Term, predicate: &Term) -> Result<Vec<Term>>;
    
    /// Find all triples matching predicate and object
    fn find_incoming(&self, predicate: &Term, object: &Term) -> Result<Vec<Term>>;
    
    /// Find all predicates between subject and object
    fn find_predicates(&self, subject: &Term, object: &Term) -> Result<Vec<Term>>;
    
    /// Get all distinct predicates in the dataset
    fn get_predicates(&self) -> Result<Vec<Term>>;
    
    /// Check if a triple exists
    fn contains_triple(&self, subject: &Term, predicate: &Term, object: &Term) -> Result<bool>;
}

impl PropertyPathEvaluator {
    pub fn new() -> Self {
        Self {
            context: PathContext::default(),
        }
    }
    
    pub fn with_context(context: PathContext) -> Self {
        Self { context }
    }
    
    /// Evaluate property path between two terms
    pub fn evaluate_path(
        &self,
        start: &Term,
        path: &PropertyPath,
        end: &Term,
        dataset: &dyn PathDataset,
    ) -> Result<bool> {
        let mut visited = HashSet::new();
        self.evaluate_path_recursive(start, path, end, dataset, &mut visited, 0)
    }
    
    /// Find all reachable nodes from start via path
    pub fn find_reachable(
        &self,
        start: &Term,
        path: &PropertyPath,
        dataset: &dyn PathDataset,
    ) -> Result<Vec<Term>> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back((start.clone(), 0));
        visited.insert(start.clone());
        
        while let Some((current, depth)) = queue.pop_front() {
            if depth >= self.context.max_length {
                continue;
            }
            
            if result.len() >= self.context.max_nodes {
                break;
            }
            
            let reachable = self.evaluate_path_step(&current, path, dataset, &mut visited, depth)?;
            
            for node in reachable {
                if !visited.contains(&node) {
                    visited.insert(node.clone());
                    result.push(node.clone());
                    queue.push_back((node, depth + 1));
                }
            }
        }
        
        Ok(result)
    }
    
    /// Evaluate property path with variable bindings
    pub fn evaluate_path_with_bindings(
        &self,
        start_var: Option<&Variable>,
        start_term: Option<&Term>,
        path: &PropertyPath,
        end_var: Option<&Variable>,
        end_term: Option<&Term>,
        dataset: &dyn PathDataset,
    ) -> Result<Solution> {
        let mut solution = Vec::new();
        
        match (start_term, end_term) {
            (Some(start), Some(end)) => {
                // Both endpoints are bound
                if self.evaluate_path(start, path, end, dataset)? {
                    let mut binding = HashMap::new();
                    if let Some(var) = start_var {
                        binding.insert(var.clone(), start.clone());
                    }
                    if let Some(var) = end_var {
                        binding.insert(var.clone(), end.clone());
                    }
                    solution.push(binding);
                }
            }
            (Some(start), None) => {
                // Start is bound, find all reachable endpoints
                let reachable = self.find_reachable(start, path, dataset)?;
                for end in reachable {
                    let mut binding = HashMap::new();
                    if let Some(var) = start_var {
                        binding.insert(var.clone(), start.clone());
                    }
                    if let Some(var) = end_var {
                        binding.insert(var.clone(), end);
                    }
                    solution.push(binding);
                }
            }
            (None, Some(end)) => {
                // End is bound, find all nodes that can reach it
                let inverse_path = self.invert_path(path);
                let reachable = self.find_reachable(end, &inverse_path, dataset)?;
                for start in reachable {
                    let mut binding = HashMap::new();
                    if let Some(var) = start_var {
                        binding.insert(var.clone(), start);
                    }
                    if let Some(var) = end_var {
                        binding.insert(var.clone(), end.clone());
                    }
                    solution.push(binding);
                }
            }
            (None, None) => {
                // Neither endpoint is bound - enumerate all path instances
                solution = self.enumerate_all_paths(path, dataset)?;
            }
        }
        
        Ok(solution)
    }
    
    /// Recursive path evaluation
    fn evaluate_path_recursive(
        &self,
        current: &Term,
        path: &PropertyPath,
        target: &Term,
        dataset: &dyn PathDataset,
        visited: &mut HashSet<Term>,
        depth: usize,
    ) -> Result<bool> {
        if depth >= self.context.max_length {
            return Ok(false);
        }
        
        if self.context.cycle_detection && visited.contains(current) {
            return Ok(false);
        }
        
        visited.insert(current.clone());
        
        let result = match path {
            PropertyPath::Direct(predicate) => {
                dataset.contains_triple(current, predicate, target)
            }
            PropertyPath::Inverse(inner_path) => {
                self.evaluate_path_recursive(target, inner_path, current, dataset, visited, depth)
            }
            PropertyPath::Sequence(first, second) => {
                // Find intermediate nodes
                let intermediate_nodes = self.evaluate_path_step(current, first, dataset, visited, depth)?;
                
                for intermediate in intermediate_nodes {
                    if self.evaluate_path_recursive(&intermediate, second, target, dataset, visited, depth + 1)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            PropertyPath::Alternative(left, right) => {
                self.evaluate_path_recursive(current, left, target, dataset, visited, depth)? ||
                self.evaluate_path_recursive(current, right, target, dataset, visited, depth)?
            }
            PropertyPath::ZeroOrMore(inner_path) => {
                // Zero case
                if current == target {
                    return Ok(true);
                }
                
                // One or more case
                self.evaluate_one_or_more(current, inner_path, target, dataset, visited, depth)
            }
            PropertyPath::OneOrMore(inner_path) => {
                self.evaluate_one_or_more(current, inner_path, target, dataset, visited, depth)
            }
            PropertyPath::ZeroOrOne(inner_path) => {
                // Zero case
                if current == target {
                    return Ok(true);
                }
                
                // One case
                self.evaluate_path_recursive(current, inner_path, target, dataset, visited, depth)
            }
            PropertyPath::NegatedPropertySet(predicates) => {
                // Find all predicates and exclude the negated ones
                let all_predicates = dataset.get_predicates()?;
                for predicate in all_predicates {
                    if !predicates.contains(&predicate) {
                        if dataset.contains_triple(current, &predicate, target)? {
                            return Ok(true);
                        }
                    }
                }
                Ok(false)
            }
        };
        
        visited.remove(current);
        result
    }
    
    /// Evaluate one step of a path
    fn evaluate_path_step(
        &self,
        current: &Term,
        path: &PropertyPath,
        dataset: &dyn PathDataset,
        visited: &mut HashSet<Term>,
        depth: usize,
    ) -> Result<Vec<Term>> {
        match path {
            PropertyPath::Direct(predicate) => {
                dataset.find_outgoing(current, predicate)
            }
            PropertyPath::Inverse(inner_path) => {
                // For inverse, we need to find incoming edges
                match inner_path.as_ref() {
                    PropertyPath::Direct(predicate) => {
                        dataset.find_incoming(predicate, current)
                    }
                    _ => {
                        // Complex inverse - not easily optimizable
                        Ok(Vec::new())
                    }
                }
            }
            PropertyPath::Sequence(first, second) => {
                let mut result = Vec::new();
                let intermediate_nodes = self.evaluate_path_step(current, first, dataset, visited, depth)?;
                
                for intermediate in intermediate_nodes {
                    let final_nodes = self.evaluate_path_step(&intermediate, second, dataset, visited, depth + 1)?;
                    result.extend(final_nodes);
                }
                
                Ok(result)
            }
            PropertyPath::Alternative(left, right) => {
                let mut result = self.evaluate_path_step(current, left, dataset, visited, depth)?;
                let right_result = self.evaluate_path_step(current, right, dataset, visited, depth)?;
                result.extend(right_result);
                Ok(result)
            }
            _ => {
                // For complex paths, use the full recursive evaluation
                Ok(Vec::new())
            }
        }
    }
    
    /// Evaluate one-or-more path
    fn evaluate_one_or_more(
        &self,
        current: &Term,
        path: &PropertyPath,
        target: &Term,
        dataset: &dyn PathDataset,
        visited: &mut HashSet<Term>,
        depth: usize,
    ) -> Result<bool> {
        let mut frontier = VecDeque::new();
        let mut local_visited = HashSet::new();
        
        frontier.push_back((current.clone(), depth));
        local_visited.insert(current.clone());
        
        while let Some((node, current_depth)) = frontier.pop_front() {
            if current_depth >= self.context.max_length {
                continue;
            }
            
            let reachable = self.evaluate_path_step(&node, path, dataset, visited, current_depth)?;
            
            for next_node in reachable {
                if next_node == *target {
                    return Ok(true);
                }
                
                if !local_visited.contains(&next_node) && current_depth < self.context.max_length - 1 {
                    local_visited.insert(next_node.clone());
                    frontier.push_back((next_node, current_depth + 1));
                }
            }
        }
        
        Ok(false)
    }
    
    /// Invert a property path
    fn invert_path(&self, path: &PropertyPath) -> PropertyPath {
        match path {
            PropertyPath::Direct(term) => PropertyPath::Inverse(Box::new(PropertyPath::Direct(term.clone()))),
            PropertyPath::Inverse(inner) => *inner.clone(),
            PropertyPath::Sequence(first, second) => {
                PropertyPath::Sequence(
                    Box::new(self.invert_path(second)),
                    Box::new(self.invert_path(first)),
                )
            }
            PropertyPath::Alternative(left, right) => {
                PropertyPath::Alternative(
                    Box::new(self.invert_path(left)),
                    Box::new(self.invert_path(right)),
                )
            }
            PropertyPath::ZeroOrMore(inner) => {
                PropertyPath::ZeroOrMore(Box::new(self.invert_path(inner)))
            }
            PropertyPath::OneOrMore(inner) => {
                PropertyPath::OneOrMore(Box::new(self.invert_path(inner)))
            }
            PropertyPath::ZeroOrOne(inner) => {
                PropertyPath::ZeroOrOne(Box::new(self.invert_path(inner)))
            }
            PropertyPath::NegatedPropertySet(predicates) => {
                PropertyPath::NegatedPropertySet(predicates.clone())
            }
        }
    }
    
    /// Enumerate all path instances (expensive operation)
    fn enumerate_all_paths(&self, _path: &PropertyPath, _dataset: &dyn PathDataset) -> Result<Solution> {
        // This is computationally expensive and should be used with caution
        // For now, return empty solution
        Ok(Vec::new())
    }
    
    /// Optimize property path for better execution
    pub fn optimize_path(&self, path: PropertyPath) -> PropertyPath {
        match self.context.optimization_level {
            PathOptimization::None => path,
            PathOptimization::Basic => self.basic_path_optimization(path),
            PathOptimization::Advanced => self.advanced_path_optimization(path),
        }
    }
    
    /// Basic path optimizations
    fn basic_path_optimization(&self, path: PropertyPath) -> PropertyPath {
        match path {
            PropertyPath::Sequence(first, second) => {
                let opt_first = self.basic_path_optimization(*first);
                let opt_second = self.basic_path_optimization(*second);
                
                // Flatten nested sequences
                match (&opt_first, &opt_second) {
                    (PropertyPath::Sequence(a, b), PropertyPath::Sequence(c, d)) => {
                        // ((a/b)/(c/d)) -> (a/b/c/d)
                        PropertyPath::Sequence(
                            Box::new(PropertyPath::Sequence(a.clone(), b.clone())),
                            Box::new(PropertyPath::Sequence(c.clone(), d.clone())),
                        )
                    }
                    _ => PropertyPath::Sequence(Box::new(opt_first), Box::new(opt_second)),
                }
            }
            PropertyPath::Alternative(left, right) => {
                let opt_left = self.basic_path_optimization(*left);
                let opt_right = self.basic_path_optimization(*right);
                PropertyPath::Alternative(Box::new(opt_left), Box::new(opt_right))
            }
            PropertyPath::ZeroOrMore(inner) => {
                let opt_inner = self.basic_path_optimization(*inner);
                PropertyPath::ZeroOrMore(Box::new(opt_inner))
            }
            PropertyPath::OneOrMore(inner) => {
                let opt_inner = self.basic_path_optimization(*inner);
                PropertyPath::OneOrMore(Box::new(opt_inner))
            }
            PropertyPath::ZeroOrOne(inner) => {
                let opt_inner = self.basic_path_optimization(*inner);
                PropertyPath::ZeroOrOne(Box::new(opt_inner))
            }
            PropertyPath::Inverse(inner) => {
                let opt_inner = self.basic_path_optimization(*inner);
                
                // Double inverse elimination: ^^p -> p
                if let PropertyPath::Inverse(inner_inner) = opt_inner {
                    *inner_inner
                } else {
                    PropertyPath::Inverse(Box::new(opt_inner))
                }
            }
            _ => path,
        }
    }
    
    /// Advanced path optimizations
    fn advanced_path_optimization(&self, path: PropertyPath) -> PropertyPath {
        let basic_opt = self.basic_path_optimization(path);
        
        match basic_opt {
            PropertyPath::Alternative(left, right) => {
                // Detect common subexpressions
                self.optimize_alternative_common_subexpressions(*left, *right)
            }
            PropertyPath::Sequence(first, second) => {
                // Try to merge consecutive direct paths
                self.optimize_sequence_direct_paths(*first, *second)
            }
            _ => basic_opt,
        }
    }
    
    /// Optimize alternatives with common subexpressions
    fn optimize_alternative_common_subexpressions(&self, left: PropertyPath, right: PropertyPath) -> PropertyPath {
        // For now, just return the original alternative
        PropertyPath::Alternative(Box::new(left), Box::new(right))
    }
    
    /// Optimize sequences of direct paths
    fn optimize_sequence_direct_paths(&self, first: PropertyPath, second: PropertyPath) -> PropertyPath {
        // For now, just return the original sequence
        PropertyPath::Sequence(Box::new(first), Box::new(second))
    }
}

impl Default for PropertyPathEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience macros for building property paths
#[macro_export]
macro_rules! path_seq {
    ($first:expr, $second:expr) => {
        PropertyPath::Sequence(Box::new($first), Box::new($second))
    };
    ($first:expr, $second:expr, $($rest:expr),+) => {
        PropertyPath::Sequence(
            Box::new($first),
            Box::new(path_seq!($second, $($rest),+))
        )
    };
}

#[macro_export]
macro_rules! path_alt {
    ($first:expr, $second:expr) => {
        PropertyPath::Alternative(Box::new($first), Box::new($second))
    };
    ($first:expr, $second:expr, $($rest:expr),+) => {
        PropertyPath::Alternative(
            Box::new($first),
            Box::new(path_alt!($second, $($rest),+))
        )
    };
}

#[macro_export]
macro_rules! path_star {
    ($path:expr) => {
        PropertyPath::ZeroOrMore(Box::new($path))
    };
}

#[macro_export]
macro_rules! path_plus {
    ($path:expr) => {
        PropertyPath::OneOrMore(Box::new($path))
    };
}

#[macro_export]
macro_rules! path_opt {
    ($path:expr) => {
        PropertyPath::ZeroOrOne(Box::new($path))
    };
}

#[macro_export]
macro_rules! path_inv {
    ($path:expr) => {
        PropertyPath::Inverse(Box::new($path))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Iri, Literal};
    use std::collections::HashMap;
    
    struct TestDataset {
        triples: Vec<(Term, Term, Term)>,
    }
    
    impl TestDataset {
        fn new() -> Self {
            Self {
                triples: vec![
                    (
                        Term::Iri(Iri("http://example.org/person1".to_string())),
                        Term::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string())),
                        Term::Iri(Iri("http://example.org/person2".to_string())),
                    ),
                    (
                        Term::Iri(Iri("http://example.org/person2".to_string())),
                        Term::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string())),
                        Term::Iri(Iri("http://example.org/person3".to_string())),
                    ),
                ],
            }
        }
    }
    
    impl PathDataset for TestDataset {
        fn find_outgoing(&self, subject: &Term, predicate: &Term) -> Result<Vec<Term>> {
            let mut result = Vec::new();
            for (s, p, o) in &self.triples {
                if s == subject && p == predicate {
                    result.push(o.clone());
                }
            }
            Ok(result)
        }
        
        fn find_incoming(&self, predicate: &Term, object: &Term) -> Result<Vec<Term>> {
            let mut result = Vec::new();
            for (s, p, o) in &self.triples {
                if p == predicate && o == object {
                    result.push(s.clone());
                }
            }
            Ok(result)
        }
        
        fn find_predicates(&self, subject: &Term, object: &Term) -> Result<Vec<Term>> {
            let mut result = Vec::new();
            for (s, p, o) in &self.triples {
                if s == subject && o == object {
                    result.push(p.clone());
                }
            }
            Ok(result)
        }
        
        fn get_predicates(&self) -> Result<Vec<Term>> {
            let mut predicates: Vec<_> = self.triples.iter().map(|(_, p, _)| p.clone()).collect();
            predicates.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
            predicates.dedup();
            Ok(predicates)
        }
        
        fn contains_triple(&self, subject: &Term, predicate: &Term, object: &Term) -> Result<bool> {
            Ok(self.triples.iter().any(|(s, p, o)| s == subject && p == predicate && o == object))
        }
    }
    
    #[test]
    fn test_direct_path() {
        let evaluator = PropertyPathEvaluator::new();
        let dataset = TestDataset::new();
        
        let path = PropertyPath::Direct(Term::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string())));
        let start = Term::Iri(Iri("http://example.org/person1".to_string()));
        let end = Term::Iri(Iri("http://example.org/person2".to_string()));
        
        assert!(evaluator.evaluate_path(&start, &path, &end, &dataset).unwrap());
    }
    
    #[test]
    fn test_sequence_path() {
        let evaluator = PropertyPathEvaluator::new();
        let dataset = TestDataset::new();
        
        let knows = PropertyPath::Direct(Term::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string())));
        let path = path_seq!(knows.clone(), knows);
        
        let start = Term::Iri(Iri("http://example.org/person1".to_string()));
        let end = Term::Iri(Iri("http://example.org/person3".to_string()));
        
        assert!(evaluator.evaluate_path(&start, &path, &end, &dataset).unwrap());
    }
    
    #[test]
    fn test_find_reachable() {
        let evaluator = PropertyPathEvaluator::new();
        let dataset = TestDataset::new();
        
        let path = PropertyPath::Direct(Term::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string())));
        let start = Term::Iri(Iri("http://example.org/person1".to_string()));
        
        let reachable = evaluator.find_reachable(&start, &path, &dataset).unwrap();
        assert_eq!(reachable.len(), 1);
        assert_eq!(reachable[0], Term::Iri(Iri("http://example.org/person2".to_string())));
    }
    
    #[test]
    fn test_path_optimization() {
        let evaluator = PropertyPathEvaluator::new();
        
        let direct = PropertyPath::Direct(Term::Iri(Iri("http://example.org/pred".to_string())));
        let double_inverse = PropertyPath::Inverse(Box::new(PropertyPath::Inverse(Box::new(direct.clone()))));
        
        let optimized = evaluator.optimize_path(double_inverse);
        assert_eq!(optimized, direct);
    }
}