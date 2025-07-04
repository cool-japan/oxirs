//! Tests for pattern matching optimization

use oxirs_core::model::*;
use oxirs_core::query::algebra::{TermPattern as AlgebraTermPattern, AlgebraTriplePattern};
use oxirs_core::query::pattern_optimizer::IndexStats;
use oxirs_core::query::{IndexType, OptimizedPatternPlan, PatternExecutor, PatternOptimizer};
use oxirs_core::store::IndexedGraph;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

#[test]
fn test_pattern_optimizer_basic() {
    let stats = Arc::new(IndexStats::new());
    let optimizer = PatternOptimizer::new(stats);

    // Create a simple pattern: ?s <http://example.org/type> <http://example.org/Person>
    let pattern = AlgebraTriplePattern {
        subject: AlgebraTermPattern::Variable(Variable::new("s").unwrap()),
        predicate: AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/type").unwrap()),
        object: AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/Person").unwrap()),
    };

    let plan = optimizer.optimize_patterns(&[pattern]).unwrap();

    assert_eq!(plan.patterns.len(), 1);
    assert!(plan.total_cost > 0.0);
}

#[test]
fn test_pattern_optimizer_with_multiple_patterns() {
    let stats = Arc::new(IndexStats::new());
    let optimizer = PatternOptimizer::new(stats.clone());

    // Create multiple patterns that share variables
    let patterns = vec![
        // ?person <http://example.org/type> <http://example.org/Person>
        AlgebraTriplePattern {
            subject: AlgebraTermPattern::Variable(Variable::new("person").unwrap()),
            predicate: AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/type").unwrap()),
            object: AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/Person").unwrap()),
        },
        // ?person <http://example.org/name> ?name
        AlgebraTriplePattern {
            subject: AlgebraTermPattern::Variable(Variable::new("person").unwrap()),
            predicate: AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/name").unwrap()),
            object: AlgebraTermPattern::Variable(Variable::new("name").unwrap()),
        },
        // ?person <http://example.org/age> ?age
        AlgebraTriplePattern {
            subject: AlgebraTermPattern::Variable(Variable::new("person").unwrap()),
            predicate: AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/age").unwrap()),
            object: AlgebraTermPattern::Variable(Variable::new("age").unwrap()),
        },
    ];

    // Update statistics to make optimization decisions more realistic
    stats.update_predicate_count("http://example.org/type", 1000);
    stats.update_predicate_count("http://example.org/name", 5000);
    stats.update_predicate_count("http://example.org/age", 5000);
    stats.set_total_triples(10000);

    let plan = optimizer.optimize_patterns(&patterns).unwrap();

    // Should have all 3 patterns
    assert_eq!(plan.patterns.len(), 3);

    // First pattern should be the most selective (type)
    if let AlgebraTermPattern::NamedNode(pred) = &plan.patterns[0].0.predicate {
        assert_eq!(pred.as_str(), "http://example.org/type");
    }

    // Verify binding order
    assert_eq!(plan.binding_order.len(), 3);

    // After first pattern, ?person should be bound
    assert!(plan.binding_order[0].contains(&Variable::new("person").unwrap()));

    // After all patterns, all variables should be bound
    let final_bindings = &plan.binding_order[2];
    assert!(final_bindings.contains(&Variable::new("person").unwrap()));
    assert!(final_bindings.contains(&Variable::new("name").unwrap()));
    assert!(final_bindings.contains(&Variable::new("age").unwrap()));
}

#[test]
fn test_index_selection_strategies() {
    let stats = Arc::new(IndexStats::new());
    let optimizer = PatternOptimizer::new(stats);

    let bound_vars = HashSet::new();

    // Test 1: Pattern with bound subject - should use SPO
    let pattern1 = TriplePattern::new(
        Some(SubjectPattern::NamedNode(
            NamedNode::new("http://example.org/alice").unwrap(),
        )),
        None,
        None,
    );
    assert_eq!(
        optimizer.get_optimal_index(&pattern1, &bound_vars),
        IndexType::SPO
    );

    // Test 2: Pattern with bound predicate - should use POS
    let pattern2 = TriplePattern::new(
        None,
        Some(PredicatePattern::NamedNode(
            NamedNode::new("http://example.org/name").unwrap(),
        )),
        None,
    );
    assert_eq!(
        optimizer.get_optimal_index(&pattern2, &bound_vars),
        IndexType::POS
    );

    // Test 3: Pattern with bound object - should use OSP
    let pattern3 = TriplePattern::new(
        None,
        None,
        Some(ObjectPattern::Literal(Literal::new("Alice"))),
    );
    assert_eq!(
        optimizer.get_optimal_index(&pattern3, &bound_vars),
        IndexType::OSP
    );

    // Test 4: Pattern with bound predicate and object - should use POS
    let pattern4 = TriplePattern::new(
        None,
        Some(PredicatePattern::NamedNode(
            NamedNode::new("http://example.org/name").unwrap(),
        )),
        Some(ObjectPattern::Literal(Literal::new("Alice"))),
    );
    assert_eq!(
        optimizer.get_optimal_index(&pattern4, &bound_vars),
        IndexType::POS
    );
}

#[test]
fn test_pattern_executor_integration() {
    // Create indexed graph
    let graph = Arc::new(IndexedGraph::new());

    // Add test data
    let alice = NamedNode::new("http://example.org/alice").unwrap();
    let bob = NamedNode::new("http://example.org/bob").unwrap();
    let type_pred = NamedNode::new("http://example.org/type").unwrap();
    let name_pred = NamedNode::new("http://example.org/name").unwrap();
    let person_type = NamedNode::new("http://example.org/Person").unwrap();

    graph.insert(&Triple::new(
        alice.clone(),
        type_pred.clone(),
        person_type.clone(),
    ));
    graph.insert(&Triple::new(
        alice.clone(),
        name_pred.clone(),
        Literal::new("Alice"),
    ));
    graph.insert(&Triple::new(
        bob.clone(),
        type_pred.clone(),
        person_type.clone(),
    ));
    graph.insert(&Triple::new(
        bob.clone(),
        name_pred.clone(),
        Literal::new("Bob"),
    ));

    // Create pattern executor
    let stats = Arc::new(IndexStats::new());
    let executor = PatternExecutor::new(graph, stats.clone());

    // Create query patterns
    let patterns = vec![
        AlgebraTriplePattern {
            subject: AlgebraTermPattern::Variable(Variable::new("person").unwrap()),
            predicate: AlgebraTermPattern::NamedNode(type_pred),
            object: AlgebraTermPattern::NamedNode(person_type),
        },
        AlgebraTriplePattern {
            subject: AlgebraTermPattern::Variable(Variable::new("person").unwrap()),
            predicate: AlgebraTermPattern::NamedNode(name_pred),
            object: AlgebraTermPattern::Variable(Variable::new("name").unwrap()),
        },
    ];

    // Optimize patterns
    let optimizer = PatternOptimizer::new(stats);
    let plan = optimizer.optimize_patterns(&patterns).unwrap();

    // Execute plan
    let results = executor.execute_plan(&plan).unwrap();

    // Should have 2 results (Alice and Bob)
    assert_eq!(results.len(), 2);

    // Check that all results have both variables bound
    for result in &results {
        assert!(result.contains_key(&Variable::new("person").unwrap()));
        assert!(result.contains_key(&Variable::new("name").unwrap()));
    }
}

#[test]
fn test_pattern_optimization_performance() {
    let graph = Arc::new(IndexedGraph::new());
    let stats = Arc::new(IndexStats::new());

    // Add a larger dataset
    for i in 0..1000 {
        let subject = NamedNode::new(&format!("http://example.org/person{}", i)).unwrap();
        let type_pred = NamedNode::new("http://example.org/type").unwrap();
        let name_pred = NamedNode::new("http://example.org/name").unwrap();
        let age_pred = NamedNode::new("http://example.org/age").unwrap();
        let person_type = NamedNode::new("http://example.org/Person").unwrap();

        graph.insert(&Triple::new(subject.clone(), type_pred, person_type));
        graph.insert(&Triple::new(
            subject.clone(),
            name_pred,
            Literal::new(format!("Person {}", i)),
        ));
        graph.insert(&Triple::new(
            subject,
            age_pred,
            Literal::new(format!("{}", 20 + (i % 50))),
        ));
    }

    // Update statistics
    stats.update_predicate_count("http://example.org/type", 1000);
    stats.update_predicate_count("http://example.org/name", 1000);
    stats.update_predicate_count("http://example.org/age", 1000);
    stats.set_total_triples(3000);

    // Create complex query pattern
    let patterns = vec![
        // Find people of specific age
        AlgebraTriplePattern {
            subject: AlgebraTermPattern::Variable(Variable::new("person").unwrap()),
            predicate: AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/age").unwrap()),
            object: AlgebraTermPattern::Literal(Literal::new("25")),
        },
        // Get their type
        AlgebraTriplePattern {
            subject: AlgebraTermPattern::Variable(Variable::new("person").unwrap()),
            predicate: AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/type").unwrap()),
            object: AlgebraTermPattern::Variable(Variable::new("type").unwrap()),
        },
        // Get their name
        AlgebraTriplePattern {
            subject: AlgebraTermPattern::Variable(Variable::new("person").unwrap()),
            predicate: AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/name").unwrap()),
            object: AlgebraTermPattern::Variable(Variable::new("name").unwrap()),
        },
    ];

    let optimizer = PatternOptimizer::new(stats.clone());
    let executor = PatternExecutor::new(graph, stats);

    // Time optimization
    let start = Instant::now();
    let plan = optimizer.optimize_patterns(&patterns).unwrap();
    let optimization_time = start.elapsed();

    // Time execution
    let start = Instant::now();
    let results = executor.execute_plan(&plan).unwrap();
    let execution_time = start.elapsed();

    println!("Optimization time: {:?}", optimization_time);
    println!("Execution time: {:?}", execution_time);
    println!("Results found: {}", results.len());

    // Verify results
    assert!(results.len() > 0);
    for result in &results {
        assert_eq!(result.len(), 3); // Should have 3 variables bound
    }
}

#[test]
fn test_selective_pattern_ordering() {
    let stats = Arc::new(IndexStats::new());

    // Set up statistics to influence pattern ordering
    stats.update_predicate_count("http://example.org/rareProperty", 10);
    stats.update_predicate_count("http://example.org/commonProperty", 5000);
    stats.set_total_triples(10000);

    let optimizer = PatternOptimizer::new(stats);

    let patterns = vec![
        // Common property pattern
        AlgebraTriplePattern {
            subject: AlgebraTermPattern::Variable(Variable::new("x").unwrap()),
            predicate: AlgebraTermPattern::NamedNode(
                NamedNode::new("http://example.org/commonProperty").unwrap(),
            ),
            object: AlgebraTermPattern::Variable(Variable::new("y").unwrap()),
        },
        // Rare property pattern
        AlgebraTriplePattern {
            subject: AlgebraTermPattern::Variable(Variable::new("x").unwrap()),
            predicate: AlgebraTermPattern::NamedNode(
                NamedNode::new("http://example.org/rareProperty").unwrap(),
            ),
            object: AlgebraTermPattern::Variable(Variable::new("z").unwrap()),
        },
    ];

    let plan = optimizer.optimize_patterns(&patterns).unwrap();

    // The rare property pattern should be executed first
    if let AlgebraTermPattern::NamedNode(pred) = &plan.patterns[0].0.predicate {
        assert_eq!(pred.as_str(), "http://example.org/rareProperty");
    }
}
