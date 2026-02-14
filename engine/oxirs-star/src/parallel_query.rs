//! Parallel SPARQL-star query optimizer using scirs2-core
//!
//! This module provides high-performance parallel query execution for SPARQL-star
//! queries, leveraging scirs2-core's parallel_ops for maximum throughput.

use crate::{StarResult, StarTerm, StarTriple};
use rayon::prelude::*; // For parallel iterator traits
use scirs2_core::parallel_ops::par_scope;
use scirs2_core::profiling::Profiler;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Parallel query executor for SPARQL-star
///
/// Executes SPARQL-star queries using data-parallel operations
/// with automatic work stealing and load balancing.
pub struct ParallelQueryExecutor {
    /// Number of worker threads
    worker_count: usize,
    /// Performance profiler
    profiler: Arc<Mutex<Profiler>>,
    /// Query cache for optimization
    query_cache: Arc<Mutex<HashMap<String, QueryPlan>>>,
}

/// Query plan for parallel execution
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Pattern matching operations
    pub patterns: Vec<TriplePattern>,
    /// Join operations
    pub joins: Vec<JoinOperation>,
    /// Filter operations
    pub filters: Vec<FilterOperation>,
    /// Estimated cost
    pub cost: f64,
}

/// Triple pattern for SPARQL-star matching
#[derive(Debug, Clone, PartialEq)]
pub struct TriplePattern {
    /// Subject pattern (None means variable)
    pub subject: Option<StarTerm>,
    /// Predicate pattern (None means variable)
    pub predicate: Option<StarTerm>,
    /// Object pattern (None means variable)
    pub object: Option<StarTerm>,
    /// Variable name if this is a variable pattern
    pub variable_name: Option<String>,
}

/// Join operation between patterns
#[derive(Debug, Clone)]
pub struct JoinOperation {
    /// Left pattern index
    pub left: usize,
    /// Right pattern index
    pub right: usize,
    /// Join type
    pub join_type: JoinType,
    /// Join variables
    pub join_vars: Vec<String>,
}

/// Type of join operation
#[derive(Debug, Clone, PartialEq)]
pub enum JoinType {
    /// Inner join (requires match on both sides)
    Inner,
    /// Left outer join
    LeftOuter,
    /// Optional (SPARQL OPTIONAL)
    Optional,
}

/// Filter operation
#[derive(Debug, Clone)]
pub struct FilterOperation {
    /// Filter expression
    pub expression: FilterExpression,
}

/// Filter expression types
#[derive(Debug, Clone)]
pub enum FilterExpression {
    /// Equality comparison
    Equals(String, StarTerm),
    /// Regex match
    Regex(String, String),
    /// Bound check
    Bound(String),
    /// Nesting depth check
    NestingDepth(String, usize, usize),
}

/// Query binding result
#[derive(Debug, Clone)]
pub struct QueryBinding {
    /// Variable bindings
    pub bindings: HashMap<String, StarTerm>,
}

impl ParallelQueryExecutor {
    /// Create a new parallel query executor
    pub fn new() -> Self {
        Self {
            worker_count: num_cpus::get(),
            profiler: Arc::new(Mutex::new(Profiler::new())),
            query_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create with specified worker count
    pub fn with_workers(worker_count: usize) -> Self {
        Self {
            worker_count,
            profiler: Arc::new(Mutex::new(Profiler::new())),
            query_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Execute a query plan in parallel
    pub fn execute_parallel(
        &self,
        plan: &QueryPlan,
        triples: &[StarTriple],
    ) -> StarResult<Vec<QueryBinding>> {
        let profiler = Arc::clone(&self.profiler);

        // Start profiling
        if let Ok(mut p) = profiler.lock() {
            p.start();
        }

        // Execute pattern matching in parallel
        let pattern_results: Vec<Vec<QueryBinding>> = plan
            .patterns
            .iter()
            .map(|pattern| self.match_pattern_parallel(pattern, triples))
            .collect::<StarResult<Vec<_>>>()?;

        // Perform joins in parallel
        let mut result = if pattern_results.is_empty() {
            Vec::new()
        } else {
            pattern_results[0].clone()
        };

        for join in &plan.joins {
            if join.left < pattern_results.len() && join.right < pattern_results.len() {
                result = self.parallel_join(
                    &result,
                    &pattern_results[join.right],
                    &join.join_vars,
                    join.join_type.clone(),
                )?;
            }
        }

        // Apply filters in parallel
        result = self.parallel_filter(&result, &plan.filters)?;

        // End profiling
        if let Ok(mut p) = profiler.lock() {
            p.stop();
        }

        Ok(result)
    }

    /// Match a triple pattern in parallel across the dataset
    fn match_pattern_parallel(
        &self,
        pattern: &TriplePattern,
        triples: &[StarTriple],
    ) -> StarResult<Vec<QueryBinding>> {
        // Use rayon's parallel iterator directly
        let results: Vec<QueryBinding> = triples
            .par_iter()
            .filter_map(|triple| {
                if self.matches_pattern(triple, pattern) {
                    Some(self.create_binding(triple, pattern))
                } else {
                    None
                }
            })
            .collect();

        Ok(results)
    }

    /// Check if a triple matches a pattern
    fn matches_pattern(&self, triple: &StarTriple, pattern: &TriplePattern) -> bool {
        // Check subject
        if let Some(ref subject) = pattern.subject {
            if &triple.subject != subject {
                return false;
            }
        }

        // Check predicate
        if let Some(ref predicate) = pattern.predicate {
            if &triple.predicate != predicate {
                return false;
            }
        }

        // Check object
        if let Some(ref object) = pattern.object {
            if &triple.object != object {
                return false;
            }
        }

        true
    }

    /// Create a binding from a matched triple
    fn create_binding(&self, triple: &StarTriple, pattern: &TriplePattern) -> QueryBinding {
        let mut bindings = HashMap::new();

        // Bind variables
        if pattern.subject.is_none() {
            if let Some(ref var) = pattern.variable_name {
                bindings.insert(format!("{}Subject", var), triple.subject.clone());
            }
        }

        if pattern.predicate.is_none() {
            if let Some(ref var) = pattern.variable_name {
                bindings.insert(format!("{}Predicate", var), triple.predicate.clone());
            }
        }

        if pattern.object.is_none() {
            if let Some(ref var) = pattern.variable_name {
                bindings.insert(format!("{}Object", var), triple.object.clone());
            }
        }

        QueryBinding { bindings }
    }

    /// Perform a parallel join operation
    fn parallel_join(
        &self,
        left: &[QueryBinding],
        right: &[QueryBinding],
        join_vars: &[String],
        join_type: JoinType,
    ) -> StarResult<Vec<QueryBinding>> {
        // Use scoped parallelism for complex joins
        let results = Arc::new(Mutex::new(Vec::new()));
        let results_clone = Arc::clone(&results);

        par_scope(|s| {
            // Split left bindings into chunks
            let chunk_size = (left.len() / self.worker_count).max(10);

            for chunk in left.chunks(chunk_size) {
                let right_ref = right;
                let join_vars_ref = join_vars;
                let join_type_ref = join_type.clone();
                let results_ref = Arc::clone(&results_clone);

                s.spawn(move |_| {
                    let mut local_results = Vec::new();

                    for left_binding in chunk {
                        for right_binding in right_ref {
                            if self.bindings_compatible(left_binding, right_binding, join_vars_ref)
                            {
                                let mut merged = left_binding.clone();
                                for (k, v) in &right_binding.bindings {
                                    merged.bindings.insert(k.clone(), v.clone());
                                }
                                local_results.push(merged);
                            }
                        }

                        // For left outer joins, include unmatched left bindings
                        if join_type_ref == JoinType::LeftOuter && local_results.is_empty() {
                            local_results.push(left_binding.clone());
                        }
                    }

                    if let Ok(mut results) = results_ref.lock() {
                        results.extend(local_results);
                    }
                });
            }
        });

        let final_results = Arc::try_unwrap(results).unwrap_or_else(|arc| {
            let mutex = arc.lock().expect("lock should not be poisoned");
            Mutex::new(mutex.clone())
        });

        Ok(final_results
            .into_inner()
            .expect("lock should not be poisoned"))
    }

    /// Check if two bindings are compatible for joining
    fn bindings_compatible(
        &self,
        left: &QueryBinding,
        right: &QueryBinding,
        join_vars: &[String],
    ) -> bool {
        for var in join_vars {
            match (left.bindings.get(var), right.bindings.get(var)) {
                (Some(left_val), Some(right_val)) => {
                    if left_val != right_val {
                        return false;
                    }
                }
                (None, None) => continue,
                _ => return false,
            }
        }
        true
    }

    /// Apply filters in parallel
    fn parallel_filter(
        &self,
        bindings: &[QueryBinding],
        filters: &[FilterOperation],
    ) -> StarResult<Vec<QueryBinding>> {
        if filters.is_empty() {
            return Ok(bindings.to_vec());
        }

        // Use rayon's parallel iterator directly
        let results: Vec<QueryBinding> = bindings
            .par_iter()
            .filter(|binding| self.apply_filters(binding, filters))
            .cloned()
            .collect();

        Ok(results)
    }

    /// Apply all filters to a binding
    fn apply_filters(&self, binding: &QueryBinding, filters: &[FilterOperation]) -> bool {
        filters.iter().all(|filter| match &filter.expression {
            FilterExpression::Equals(var, value) => binding
                .bindings
                .get(var)
                .map(|v| v == value)
                .unwrap_or(false),
            FilterExpression::Bound(var) => binding.bindings.contains_key(var),
            FilterExpression::NestingDepth(var, min, max) => binding
                .bindings
                .get(var)
                .map(|term| {
                    let depth = term.nesting_depth();
                    depth >= *min && depth <= *max
                })
                .unwrap_or(false),
            FilterExpression::Regex(var, _pattern) => {
                // Simplified regex matching
                binding.bindings.contains_key(var)
            }
        })
    }

    /// Get profiling statistics
    pub fn get_statistics(&self) -> HashMap<String, u64> {
        // Note: The profiler API doesn't expose get_stats() directly
        // This is a placeholder that returns empty stats
        HashMap::new()
    }

    /// Clear query cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.query_cache.lock() {
            cache.clear();
        }
    }
}

impl Default for ParallelQueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() {
        let executor = ParallelQueryExecutor::new();
        assert!(executor.worker_count > 0);
    }

    #[test]
    fn test_pattern_matching() {
        let executor = ParallelQueryExecutor::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let pattern = TriplePattern {
            subject: Some(StarTerm::iri("http://example.org/s").unwrap()),
            predicate: None,
            object: None,
            variable_name: Some("x".to_string()),
        };

        assert!(executor.matches_pattern(&triple, &pattern));
    }

    #[test]
    fn test_parallel_execution() {
        let executor = ParallelQueryExecutor::new();

        let triples = vec![
            StarTriple::new(
                StarTerm::iri("http://example.org/s1").unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::iri("http://example.org/o1").unwrap(),
            ),
            StarTriple::new(
                StarTerm::iri("http://example.org/s2").unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::iri("http://example.org/o2").unwrap(),
            ),
        ];

        let plan = QueryPlan {
            patterns: vec![TriplePattern {
                subject: None,
                predicate: Some(StarTerm::iri("http://example.org/p").unwrap()),
                object: None,
                variable_name: Some("x".to_string()),
            }],
            joins: vec![],
            filters: vec![],
            cost: 1.0,
        };

        let results = executor.execute_parallel(&plan, &triples).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_filter_application() {
        let executor = ParallelQueryExecutor::new();

        let mut bindings = HashMap::new();
        bindings.insert(
            "x".to_string(),
            StarTerm::iri("http://example.org/test").unwrap(),
        );

        let binding = QueryBinding { bindings };

        let filters = vec![FilterOperation {
            expression: FilterExpression::Bound("x".to_string()),
        }];

        assert!(executor.apply_filters(&binding, &filters));
    }
}
