//! SPARQL-star Integration Module
//!
//! This module provides integration between oxirs-arq and oxirs-star for SPARQL 1.2 / SPARQL-star support.
//! It enables:
//! - Quoted triple patterns in SPARQL queries
//! - RDF-star specific built-in functions (TRIPLE, isTRIPLE, SUBJECT, PREDICATE, OBJECT)
//! - Conversion between ARQ algebra terms and RDF-star terms
//! - Query execution over RDF-star datasets

#[cfg(feature = "star")]
use oxirs_star::{StarConfig, StarError, StarResult, StarStore, StarTerm, StarTriple};

use crate::algebra::{Literal, Term, TriplePattern};
use anyhow::Result;
use oxirs_core::model::{NamedNode, Variable};
use std::collections::HashMap;

/// SPARQL-star executor with RDF-star support
#[cfg(feature = "star")]
pub struct SparqlStarExecutor {
    /// Underlying star store
    star_store: StarStore,
    /// Configuration for RDF-star processing
    #[allow(dead_code)]
    config: StarConfig,
}

#[cfg(feature = "star")]
impl SparqlStarExecutor {
    /// Create a new SPARQL-star executor
    pub fn new() -> Self {
        Self {
            star_store: StarStore::new(),
            config: StarConfig::default(),
        }
    }

    /// Create a SPARQL-star executor with custom configuration
    pub fn with_config(config: StarConfig) -> Self {
        Self {
            star_store: StarStore::with_config(config.clone()),
            config,
        }
    }

    /// Get the underlying star store
    pub fn store(&self) -> &StarStore {
        &self.star_store
    }

    /// Get mutable reference to the underlying star store
    pub fn store_mut(&mut self) -> &mut StarStore {
        &mut self.star_store
    }

    /// Convert ARQ Term to StarTerm
    pub fn term_to_star_term(term: &Term) -> StarResult<StarTerm> {
        match term {
            Term::Iri(iri) => StarTerm::iri(iri.as_str())
                .map_err(|e| StarError::invalid_term_type(format!("Invalid IRI: {}", e))),
            Term::Literal(lit) => {
                if let Some(ref lang) = lit.language {
                    StarTerm::literal_with_language(&lit.value, lang).map_err(|e| {
                        StarError::invalid_term_type(format!("Invalid literal: {}", e))
                    })
                } else if let Some(ref datatype) = lit.datatype {
                    StarTerm::literal_with_datatype(&lit.value, datatype.as_str()).map_err(|e| {
                        StarError::invalid_term_type(format!("Invalid literal: {}", e))
                    })
                } else {
                    StarTerm::literal(&lit.value).map_err(|e| {
                        StarError::invalid_term_type(format!("Invalid literal: {}", e))
                    })
                }
            }
            Term::BlankNode(id) => StarTerm::blank_node(id)
                .map_err(|e| StarError::invalid_term_type(format!("Invalid blank node: {}", e))),
            Term::QuotedTriple(triple) => {
                let star_triple = Self::triple_pattern_to_star_triple(triple)?;
                Ok(StarTerm::quoted_triple(star_triple))
            }
            Term::Variable(var) => {
                // Variables are not directly representable in StarTerm
                // This is a pattern matching context
                Err(StarError::invalid_term_type(format!(
                    "Cannot convert variable {} to StarTerm",
                    var
                )))
            }
            Term::PropertyPath(_) => Err(StarError::invalid_term_type(
                "Cannot convert property path to StarTerm".to_string(),
            )),
        }
    }

    /// Convert TriplePattern to StarTriple
    pub fn triple_pattern_to_star_triple(pattern: &TriplePattern) -> StarResult<StarTriple> {
        let subject = Self::term_to_star_term(&pattern.subject)?;
        let predicate = Self::term_to_star_term(&pattern.predicate)?;
        let object = Self::term_to_star_term(&pattern.object)?;

        let triple = StarTriple::new(subject, predicate, object);
        triple.validate()?;
        Ok(triple)
    }

    /// Convert StarTerm to ARQ Term
    pub fn star_term_to_term(star_term: &StarTerm) -> Result<Term> {
        match star_term {
            StarTerm::NamedNode(node) => Ok(Term::Iri(NamedNode::new(&node.iri)?)),
            StarTerm::Literal(lit) => {
                let literal = if let Some(ref lang) = lit.language {
                    Literal::with_language(lit.value.clone(), lang.clone())
                } else if let Some(ref datatype) = lit.datatype {
                    Literal::new(
                        lit.value.clone(),
                        None,
                        Some(NamedNode::new(&datatype.iri)?),
                    )
                } else {
                    Literal::new(lit.value.clone(), None, None)
                };
                Ok(Term::Literal(literal))
            }
            StarTerm::BlankNode(bn) => Ok(Term::BlankNode(bn.id.clone())),
            StarTerm::QuotedTriple(triple) => {
                let subject = Self::star_term_to_term(&triple.subject)?;
                let predicate = Self::star_term_to_term(&triple.predicate)?;
                let object = Self::star_term_to_term(&triple.object)?;
                Ok(Term::QuotedTriple(Box::new(TriplePattern::new(
                    subject, predicate, object,
                ))))
            }
            StarTerm::Variable(var) => {
                // Variables in StarTerm are not directly representable in ARQ Term
                // This shouldn't happen in practice for query results
                Err(anyhow::anyhow!(
                    "Cannot convert StarTerm::Variable {} to ARQ Term",
                    var.name
                ))
            }
        }
    }

    /// Execute a SPARQL-star query against the store
    pub fn execute_sparql_star_query(&self, query: &str) -> Result<Vec<HashMap<Variable, Term>>> {
        // Use oxirs-star's QueryExecutor
        use oxirs_star::query::{QueryExecutor, QueryParser};

        let mut executor = QueryExecutor::new(self.star_store.clone());

        // Parse the query (simplified - in production would use full SPARQL parser)
        let (select_vars, bgp) = QueryParser::parse_simple_select(query)
            .map_err(|e| anyhow::anyhow!("SPARQL-star parsing failed: {}", e))?;

        let star_results = executor
            .execute_select(&bgp, &select_vars)
            .map_err(|e| anyhow::anyhow!("SPARQL-star execution failed: {}", e))?;

        // Convert results to ARQ format
        let mut results = Vec::new();
        for star_binding in star_results {
            let mut binding = HashMap::new();
            for (var_name, star_term) in star_binding {
                let variable = Variable::new(&var_name)?;
                let term = Self::star_term_to_term(&star_term)?;
                binding.insert(variable, term);
            }
            results.push(binding);
        }

        Ok(results)
    }
}

#[cfg(feature = "star")]
impl Default for SparqlStarExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// SPARQL-star built-in functions
///
/// Note: These functions are already implemented in `crate::triple_functions` module
/// and registered via `crate::builtin::register_builtin_functions()`.
/// This module provides additional utility functions for working with quoted triples.
#[cfg(feature = "star")]
pub mod sparql_star_functions {
    use super::*;

    /// Check if a term is a quoted triple
    pub fn is_quoted_triple(term: &Term) -> bool {
        matches!(term, Term::QuotedTriple(_))
    }

    /// Extract the subject from a quoted triple term
    pub fn get_subject(term: &Term) -> Option<&Term> {
        if let Term::QuotedTriple(triple) = term {
            Some(&triple.subject)
        } else {
            None
        }
    }

    /// Extract the predicate from a quoted triple term
    pub fn get_predicate(term: &Term) -> Option<&Term> {
        if let Term::QuotedTriple(triple) = term {
            Some(&triple.predicate)
        } else {
            None
        }
    }

    /// Extract the object from a quoted triple term
    pub fn get_object(term: &Term) -> Option<&Term> {
        if let Term::QuotedTriple(triple) = term {
            Some(&triple.object)
        } else {
            None
        }
    }
}

/// Helper functions for SPARQL-star query patterns
#[cfg(feature = "star")]
pub mod pattern_matching {
    use super::*;

    /// Check if a triple pattern contains quoted triples
    pub fn has_quoted_triples(pattern: &TriplePattern) -> bool {
        matches!(pattern.subject, Term::QuotedTriple(_))
            || matches!(pattern.object, Term::QuotedTriple(_))
    }

    /// Extract all quoted triples from a term (recursively)
    pub fn extract_quoted_triples(term: &Term) -> Vec<TriplePattern> {
        let mut triples = Vec::new();
        extract_quoted_triples_recursive(term, &mut triples);
        triples
    }

    fn extract_quoted_triples_recursive(term: &Term, triples: &mut Vec<TriplePattern>) {
        if let Term::QuotedTriple(triple) = term {
            triples.push((**triple).clone());
            // Recursively extract from nested quoted triples
            extract_quoted_triples_recursive(&triple.subject, triples);
            extract_quoted_triples_recursive(&triple.predicate, triples);
            extract_quoted_triples_recursive(&triple.object, triples);
        }
    }

    /// Calculate the nesting depth of a quoted triple
    pub fn nesting_depth(term: &Term) -> usize {
        match term {
            Term::QuotedTriple(triple) => {
                let subject_depth = nesting_depth(&triple.subject);
                let predicate_depth = nesting_depth(&triple.predicate);
                let object_depth = nesting_depth(&triple.object);
                1 + subject_depth.max(predicate_depth).max(object_depth)
            }
            _ => 0,
        }
    }
}

/// Statistics and performance tracking for SPARQL-star queries
#[cfg(feature = "star")]
pub mod star_statistics {
    use serde::{Deserialize, Serialize};

    /// Statistics for SPARQL-star query execution
    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct SparqlStarStatistics {
        /// Number of quoted triple patterns matched
        pub quoted_patterns_matched: usize,
        /// Maximum nesting depth encountered
        pub max_nesting_depth: usize,
        /// Number of SPARQL-star functions evaluated
        pub star_functions_evaluated: usize,
        /// Query execution time in microseconds
        pub execution_time_us: u64,
        /// Number of results returned
        pub result_count: usize,
    }

    impl SparqlStarStatistics {
        pub fn new() -> Self {
            Self::default()
        }

        /// Record a quoted pattern match
        pub fn record_quoted_pattern(&mut self, depth: usize) {
            self.quoted_patterns_matched += 1;
            self.max_nesting_depth = self.max_nesting_depth.max(depth);
        }

        /// Record a SPARQL-star function evaluation
        pub fn record_star_function(&mut self) {
            self.star_functions_evaluated += 1;
        }

        /// Record query execution time
        pub fn record_execution_time(&mut self, duration_us: u64) {
            self.execution_time_us = duration_us;
        }

        /// Record result count
        pub fn record_results(&mut self, count: usize) {
            self.result_count = count;
        }

        /// Get average time per result (in microseconds)
        pub fn avg_time_per_result(&self) -> Option<f64> {
            if self.result_count > 0 {
                Some(self.execution_time_us as f64 / self.result_count as f64)
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "star")]
mod tests {
    use super::*;

    #[test]
    fn test_sparql_star_executor_creation() {
        let executor = SparqlStarExecutor::new();
        assert_eq!(executor.store().len(), 0);
    }

    #[test]
    fn test_term_conversion() {
        let iri_term = Term::Iri(NamedNode::new("http://example.org/test").unwrap());
        let star_term = SparqlStarExecutor::term_to_star_term(&iri_term).unwrap();
        assert!(star_term.is_named_node());

        let converted_back = SparqlStarExecutor::star_term_to_term(&star_term).unwrap();
        assert!(matches!(converted_back, Term::Iri(_)));
    }

    #[test]
    fn test_quoted_triple_conversion() {
        let subject = Term::Iri(NamedNode::new("http://example.org/s").unwrap());
        let predicate = Term::Iri(NamedNode::new("http://example.org/p").unwrap());
        let object = Term::Iri(NamedNode::new("http://example.org/o").unwrap());

        let pattern = TriplePattern::new(subject, predicate, object);
        let star_triple = SparqlStarExecutor::triple_pattern_to_star_triple(&pattern).unwrap();

        assert!(star_triple.subject.is_named_node());
        assert!(star_triple.predicate.is_named_node());
        assert!(star_triple.object.is_named_node());
    }

    #[test]
    fn test_nesting_depth_calculation() {
        use pattern_matching::nesting_depth;

        // Simple IRI has depth 0
        let simple_term = Term::Iri(NamedNode::new("http://example.org/test").unwrap());
        assert_eq!(nesting_depth(&simple_term), 0);

        // Quoted triple has depth 1
        let inner_triple = TriplePattern::new(
            Term::Iri(NamedNode::new("http://example.org/s").unwrap()),
            Term::Iri(NamedNode::new("http://example.org/p").unwrap()),
            Term::Iri(NamedNode::new("http://example.org/o").unwrap()),
        );
        let quoted_term = Term::QuotedTriple(Box::new(inner_triple));
        assert_eq!(nesting_depth(&quoted_term), 1);

        // Nested quoted triple has depth 2
        let nested_triple = TriplePattern::new(
            quoted_term.clone(),
            Term::Iri(NamedNode::new("http://example.org/certainty").unwrap()),
            Term::Literal(Literal::new("0.9".to_string(), None, None)),
        );
        let nested_term = Term::QuotedTriple(Box::new(nested_triple));
        assert_eq!(nesting_depth(&nested_term), 2);
    }

    #[test]
    fn test_statistics_tracking() {
        use star_statistics::SparqlStarStatistics;

        let mut stats = SparqlStarStatistics::new();
        assert_eq!(stats.quoted_patterns_matched, 0);

        stats.record_quoted_pattern(1);
        stats.record_quoted_pattern(2);
        stats.record_star_function();
        stats.record_execution_time(1000);
        stats.record_results(10);

        assert_eq!(stats.quoted_patterns_matched, 2);
        assert_eq!(stats.max_nesting_depth, 2);
        assert_eq!(stats.star_functions_evaluated, 1);
        assert_eq!(stats.execution_time_us, 1000);
        assert_eq!(stats.result_count, 10);
        assert_eq!(stats.avg_time_per_result(), Some(100.0));
    }

    #[test]
    fn test_sparql_star_utility_functions() {
        use sparql_star_functions::*;

        let subject = Term::Iri(NamedNode::new("http://example.org/s").unwrap());
        let predicate = Term::Iri(NamedNode::new("http://example.org/p").unwrap());
        let object = Term::Literal(Literal::new("value".to_string(), None, None));

        let triple_pattern = TriplePattern::new(subject.clone(), predicate.clone(), object.clone());
        let quoted_term = Term::QuotedTriple(Box::new(triple_pattern));

        assert!(is_quoted_triple(&quoted_term));
        assert!(!is_quoted_triple(&subject));

        assert!(get_subject(&quoted_term).is_some());
        assert!(get_predicate(&quoted_term).is_some());
        assert!(get_object(&quoted_term).is_some());

        assert!(get_subject(&subject).is_none());
    }
}
