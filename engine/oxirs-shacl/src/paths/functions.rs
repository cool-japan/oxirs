//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Result, ShaclError};
use oxirs_core::model::Term;
/// Format a term for use in SPARQL queries
pub fn format_term_for_sparql(term: &Term) -> Result<String> {
    match term {
        Term::NamedNode(node) => Ok(format!("<{}>", node.as_str())),
        Term::BlankNode(node) => Ok(format!("_:{}", node.as_str())),
        Term::Literal(literal) => {
            let value = literal.value().replace('\\', "\\\\").replace('"', "\\\"");
            let datatype = literal.datatype();
            if datatype.as_str() == "http://www.w3.org/2001/XMLSchema#string" {
                Ok(format!("\"{value}\""))
            } else {
                Ok(format!("\"{}\"^^<{}>", value, datatype.as_str()))
            }
        }
        Term::Variable(var) => Ok(format!("?{}", var.name())),
        Term::QuotedTriple(_) => Err(ShaclError::PropertyPath(
            "Quoted triples not supported in property path queries".to_string(),
        )),
    }
}
#[cfg(test)]
mod tests {
    use super::super::prelude::*;
    use super::*;
    #[test]
    fn test_property_path_creation() {
        let predicate = NamedNode::new("http://example.org/knows").expect("valid IRI");
        let path = PropertyPath::predicate(predicate.clone());
        assert!(path.is_predicate());
        assert_eq!(path.as_predicate(), Some(&predicate));
        assert!(!path.is_complex());
        assert_eq!(path.complexity(), 1);
    }
    #[test]
    fn test_inverse_path() {
        let predicate = NamedNode::new("http://example.org/knows").expect("valid IRI");
        let path = PropertyPath::inverse(PropertyPath::predicate(predicate));
        assert!(!path.is_predicate());
        assert!(path.is_complex());
        assert_eq!(path.complexity(), 2);
    }
    #[test]
    fn test_sequence_path() {
        let pred1 = NamedNode::new("http://example.org/knows").expect("valid IRI");
        let pred2 = NamedNode::new("http://example.org/friend").expect("valid IRI");
        let path = PropertyPath::sequence(vec![
            PropertyPath::predicate(pred1),
            PropertyPath::predicate(pred2),
        ]);
        assert!(!path.is_predicate());
        assert!(path.is_complex());
        assert_eq!(path.complexity(), 3);
    }
    #[test]
    fn test_zero_or_more_path() {
        let predicate = NamedNode::new("http://example.org/knows").expect("valid IRI");
        let path = PropertyPath::zero_or_more(PropertyPath::predicate(predicate));
        assert!(!path.is_predicate());
        assert!(path.is_complex());
        assert_eq!(path.complexity(), 10);
    }
    #[test]
    fn test_alternative_path() {
        let pred1 = NamedNode::new("http://example.org/knows").expect("valid IRI");
        let pred2 = NamedNode::new("http://example.org/friend").expect("valid IRI");
        let path = PropertyPath::alternative(vec![
            PropertyPath::predicate(pred1),
            PropertyPath::predicate(pred2),
        ]);
        assert!(!path.is_predicate());
        assert!(path.is_complex());
        assert_eq!(path.complexity(), 2);
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
        stats.record_evaluation(5, false, 2);
        stats.record_evaluation(3, true, 1);
        stats.record_evaluation(7, false, 3);
        assert_eq!(stats.total_evaluations, 3);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 2);
        assert_eq!(stats.total_values_found, 15);
        assert_eq!(stats.avg_values_per_evaluation, 5.0);
        assert_eq!(stats.max_recursion_depth_reached, 3);
        assert_eq!(stats.cache_hit_rate(), 1.0 / 3.0);
    }
    #[test]
    fn test_sparql_path_query_generation() {
        let evaluator = PropertyPathEvaluator::new();
        let start_node =
            Term::NamedNode(NamedNode::new("http://example.org/person1").expect("valid IRI"));
        let predicate = NamedNode::new("http://example.org/knows").expect("valid IRI");
        let path = PropertyPath::predicate(predicate);
        let hints = PathOptimizationHints::default();
        let query = evaluator
            .generate_optimized_sparql_query(&start_node, &path, None, &hints)
            .expect("generation should succeed");
        assert!(query.contains("SELECT DISTINCT ?value"));
        assert!(query.contains("<http://example.org/person1>"));
        assert!(query.contains("<http://example.org/knows>"));
        assert!(query.contains("PREFIX"));
    }
    #[test]
    fn test_sequence_path_query_generation() {
        let evaluator = PropertyPathEvaluator::new();
        let start_node =
            Term::NamedNode(NamedNode::new("http://example.org/person1").expect("valid IRI"));
        let pred1 = NamedNode::new("http://example.org/knows").expect("valid IRI");
        let pred2 = NamedNode::new("http://example.org/friend").expect("valid IRI");
        let path = PropertyPath::sequence(vec![
            PropertyPath::predicate(pred1),
            PropertyPath::predicate(pred2),
        ]);
        let hints = PathOptimizationHints::default();
        let query = evaluator
            .generate_optimized_sparql_query(&start_node, &path, None, &hints)
            .expect("generation should succeed");
        assert!(query.contains("SELECT DISTINCT ?value"));
        assert!(query.contains("?start"));
        assert!(query.contains("?inter1"));
        assert!(query.contains("BIND"));
    }
    #[test]
    fn test_alternative_path_query_generation() {
        let evaluator = PropertyPathEvaluator::new();
        let start_node =
            Term::NamedNode(NamedNode::new("http://example.org/person1").expect("valid IRI"));
        let pred1 = NamedNode::new("http://example.org/knows").expect("valid IRI");
        let pred2 = NamedNode::new("http://example.org/friend").expect("valid IRI");
        let path = PropertyPath::alternative(vec![
            PropertyPath::predicate(pred1),
            PropertyPath::predicate(pred2),
        ]);
        let hints = PathOptimizationHints::default();
        let query = evaluator
            .generate_optimized_sparql_query(&start_node, &path, None, &hints)
            .expect("generation should succeed");
        assert!(query.contains("SELECT DISTINCT ?value"));
        assert!(query.contains("UNION"));
        assert!(query.contains("<http://example.org/knows>"));
        assert!(query.contains("<http://example.org/friend>"));
    }
    #[test]
    fn test_recursive_path_query_generation() {
        let evaluator = PropertyPathEvaluator::new();
        let start_node =
            Term::NamedNode(NamedNode::new("http://example.org/person1").expect("valid IRI"));
        let predicate = NamedNode::new("http://example.org/knows").expect("valid IRI");
        let path = PropertyPath::one_or_more(PropertyPath::predicate(predicate));
        let hints = PathOptimizationHints::default();
        let query = evaluator
            .generate_optimized_sparql_query(&start_node, &path, None, &hints)
            .expect("generation should succeed");
        assert!(query.contains("SELECT DISTINCT ?value"));
        assert!(query.contains("+"));
        assert!(query.contains("MAX_RECURSION_DEPTH"));
    }
    #[test]
    fn test_path_optimization() {
        let evaluator = PropertyPathEvaluator::new();
        let simple_path =
            PropertyPath::predicate(NamedNode::new("http://example.org/knows").expect("valid IRI"));
        let optimized = evaluator.optimize_path(&simple_path);
        assert_eq!(
            optimized.optimization_strategy,
            PathOptimizationStrategy::SparqlPath
        );
        let complex_path = PropertyPath::sequence(vec![
            PropertyPath::predicate(NamedNode::new("http://example.org/knows").expect("valid IRI")),
            PropertyPath::zero_or_more(PropertyPath::predicate(
                NamedNode::new("http://example.org/friend").expect("valid IRI"),
            )),
        ]);
        let optimized = evaluator.optimize_path(&complex_path);
        assert_eq!(
            optimized.optimization_strategy,
            PathOptimizationStrategy::Hybrid
        );
    }
    #[test]
    fn test_path_validation() {
        let evaluator = PropertyPathEvaluator::new();
        let valid_path =
            PropertyPath::predicate(NamedNode::new("http://example.org/knows").expect("valid IRI"));
        let result = evaluator.validate_property_path(&valid_path);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        let invalid_path = PropertyPath::sequence(vec![]);
        let result = evaluator.validate_property_path(&invalid_path);
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
        let warning_path = PropertyPath::zero_or_more(PropertyPath::predicate(
            NamedNode::new("http://example.org/knows").expect("valid IRI"),
        ));
        let result = evaluator.validate_property_path(&warning_path);
        assert!(result.is_valid);
        assert!(!result.warnings.is_empty());
    }
    #[test]
    fn test_query_plan_generation() {
        let evaluator = PropertyPathEvaluator::new();
        let start_node =
            Term::NamedNode(NamedNode::new("http://example.org/person1").expect("valid IRI"));
        let path =
            PropertyPath::predicate(NamedNode::new("http://example.org/knows").expect("valid IRI"));
        let hints = PathOptimizationHints::default();
        let plan = evaluator
            .generate_query_plan(&start_node, &path, None, &hints)
            .expect("generation should succeed");
        assert!(!plan.query.is_empty());
        assert_eq!(plan.execution_strategy, PathExecutionStrategy::DirectSparql);
        assert!(plan.estimated_cost > 0.0);
        assert!(!plan.cache_key.is_empty());
    }
}
