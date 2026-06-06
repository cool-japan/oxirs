//! Unit tests for enhanced BIND and VALUES clause processing.
//!
//! Split out of the original `bind_values_enhanced` module (Round 32 refactor).

use crate::bind_values_enhanced_types::{
    AdvancedBindOptimizer, EnhancedBindProcessor, EnhancedValuesProcessor,
};

#[tokio::test]
async fn test_bind_expression_extraction() {
    let processor = EnhancedBindProcessor::new();

    let query = r#"
        SELECT ?name ?age ?category
        WHERE {
            ?person foaf:name ?name .
            BIND(YEAR(NOW()) - ?birthYear AS ?age)
            BIND(IF(?age < 18, "minor", "adult") AS ?category)
        }
    "#;

    let expressions = processor.extract_bind_expressions(query).unwrap();
    assert_eq!(expressions.len(), 2);
    assert_eq!(expressions[0].0, "?age");
    assert_eq!(expressions[1].0, "?category");
}

#[tokio::test]
async fn test_values_clause_extraction() {
    let processor = EnhancedValuesProcessor::new();

    let query = r#"
        SELECT ?person ?email
        WHERE {
            VALUES (?person ?email) {
                (:alice "alice@example.com")
                (:bob "bob@example.com")
                (:charlie "charlie@example.com")
            }
        }
    "#;

    let clauses = processor.extract_values_clauses(query).unwrap();
    assert!(!clauses.is_empty());
}

#[test]
fn test_expression_optimizer() {
    let optimizer = AdvancedBindOptimizer::new();

    let expr = "CONCAT(\"Hello\", \" \", \"World\")";
    let optimized = optimizer.optimize_expression(expr).unwrap();

    // Should optimize constant concatenation
    assert_ne!(optimized, expr);
}
