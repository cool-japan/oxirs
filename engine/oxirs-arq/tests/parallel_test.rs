//! Tests for parallel query execution

use oxirs_arq::{
    algebra::{Algebra, Binding, Iri, Literal, Term, TriplePattern, Variable},
    executor::{Dataset, ExecutionContext, InMemoryDataset, ParallelConfig, QueryExecutor},
    Solution,
};
use oxirs_core::model::NamedNode;
use std::collections::HashMap;
use std::time::Instant;

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_bgp_execution() {
    use std::sync::Arc;

    // Create a dataset with many triples
    let mut dataset = InMemoryDataset::new();

    // Add test data
    for i in 0..1000 {
        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked(&format!(
                "http://example.org/person{}",
                i
            ))),
            Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            Term::Literal(Literal::string(format!("Person {}", i))),
        );

        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked(&format!(
                "http://example.org/person{}",
                i
            ))),
            Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
            Term::Literal(Literal::typed(
                (20 + (i % 60)).to_string(),
                NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer"),
            )),
        );
    }

    // Create BGP with multiple patterns
    let patterns = vec![
        TriplePattern {
            subject: Term::Variable(Variable::new("person").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            object: Term::Variable(Variable::new("name").unwrap()),
        },
        TriplePattern {
            subject: Term::Variable(Variable::new("person").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
            object: Term::Variable(Variable::new("age").unwrap()),
        },
    ];

    let algebra = Algebra::Bgp(patterns);

    // Test with parallel execution enabled
    let mut parallel_context = ExecutionContext::default();
    parallel_context.parallel = true;
    parallel_context.parallel_threshold = 100; // Lower threshold for testing

    let mut executor = QueryExecutor::with_context(parallel_context);

    let start = Instant::now();
    let (solution, stats) = executor.execute(&algebra, &dataset).unwrap();
    let parallel_time = start.elapsed();

    println!("Parallel execution time: {:?}", parallel_time);
    println!("Results found: {}", solution.len());
    println!("Stats: {:?}", stats);

    assert_eq!(solution.len(), 1000);

    // Test with parallel execution disabled
    let mut sequential_context = ExecutionContext::default();
    sequential_context.parallel = false;

    let mut executor = QueryExecutor::with_context(sequential_context);

    let start = Instant::now();
    let (solution2, _) = executor.execute(&algebra, &dataset).unwrap();
    let sequential_time = start.elapsed();

    println!("Sequential execution time: {:?}", sequential_time);

    // Results should be the same
    assert_eq!(solution.len(), solution2.len());
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_join_execution() {
    let mut dataset = InMemoryDataset::new();

    // Add parent-child relationships
    for i in 0..100 {
        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked(&format!(
                "http://example.org/person{}",
                i
            ))),
            Term::Iri(NamedNode::new_unchecked("http://example.org/parent")),
            Term::Iri(NamedNode::new_unchecked(&format!(
                "http://example.org/person{}",
                i * 2 + 100
            ))),
        );

        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked(&format!(
                "http://example.org/person{}",
                i * 2 + 100
            ))),
            Term::Iri(NamedNode::new_unchecked("http://example.org/name")),
            Term::Literal(Literal::string(format!("Child of Person {}", i))),
        );
    }

    // Create join query: find parents and their children's names
    let left_pattern = Algebra::Bgp(vec![TriplePattern {
        subject: Term::Variable(Variable::new("parent").unwrap()),
        predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/parent")),
        object: Term::Variable(Variable::new("child").unwrap()),
    }]);

    let right_pattern = Algebra::Bgp(vec![TriplePattern {
        subject: Term::Variable(Variable::new("child").unwrap()),
        predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/name")),
        object: Term::Variable(Variable::new("childName").unwrap()),
    }]);

    let join_algebra = Algebra::Join {
        left: Box::new(left_pattern),
        right: Box::new(right_pattern),
    };

    // Execute with parallel enabled
    let mut context = ExecutionContext::default();
    context.parallel = true;
    context.parallel_config.max_threads = 4;

    let mut executor = QueryExecutor::with_context(context);

    let start = Instant::now();
    let (solution, stats) = executor.execute(&join_algebra, &dataset).unwrap();
    let duration = start.elapsed();

    println!("Parallel join execution time: {:?}", duration);
    println!("Join results: {}", solution.len());
    println!("Stats: {:?}", stats);

    assert_eq!(solution.len(), 100);
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_aggregation() {
    let mut dataset = InMemoryDataset::new();

    // Add people with departments
    for i in 0..1000 {
        let dept = format!("Dept{}", i % 10);
        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked(&format!(
                "http://example.org/person{}",
                i
            ))),
            Term::Iri(NamedNode::new_unchecked("http://example.org/department")),
            Term::Literal(Literal::string(dept)),
        );

        dataset.add_triple(
            Term::Iri(NamedNode::new_unchecked(&format!(
                "http://example.org/person{}",
                i
            ))),
            Term::Iri(NamedNode::new_unchecked("http://example.org/salary")),
            Term::Literal(Literal::typed(
                (30000 + (i * 100)).to_string(),
                NamedNode::new_unchecked("http://www.w3.org/2001/XMLSchema#integer"),
            )),
        );
    }

    // Create aggregation query: COUNT and SUM by department
    let pattern = Algebra::Bgp(vec![
        TriplePattern {
            subject: Term::Variable(Variable::new("person").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/department")),
            object: Term::Variable(Variable::new("dept").unwrap()),
        },
        TriplePattern {
            subject: Term::Variable(Variable::new("person").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/salary")),
            object: Term::Variable(Variable::new("salary").unwrap()),
        },
    ]);

    use oxirs_arq::{Aggregate, GroupCondition};
    use oxirs_arq::algebra::Expression;

    let group_algebra = Algebra::Group {
        pattern: Box::new(pattern),
        variables: vec![GroupCondition {
            expr: Expression::Variable(Variable::new("dept").unwrap()),
            alias: None,
        }],
        aggregates: vec![
            (
                Variable::new("count").unwrap(),
                Aggregate::Count {
                    expr: Some(Expression::Variable(Variable::new("person").unwrap())),
                    distinct: false,
                },
            ),
            (
                Variable::new("totalSalary").unwrap(),
                Aggregate::Sum {
                    expr: Expression::Variable(Variable::new("salary").unwrap()),
                    distinct: false,
                },
            ),
        ],
    };

    // Execute with parallel enabled
    let mut context = ExecutionContext::default();
    context.parallel = true;

    let mut executor = QueryExecutor::with_context(context);

    let start = Instant::now();
    let (solution, stats) = executor.execute(&group_algebra, &dataset).unwrap();
    let duration = start.elapsed();

    println!("Parallel aggregation execution time: {:?}", duration);
    println!("Group results: {}", solution.len());
    println!("Stats: {:?}", stats);

    // Should have 10 groups (Dept0 to Dept9)
    assert_eq!(solution.len(), 10);

    // Each department should have 100 people
    for binding in &solution {
        if let Some(Term::Literal(count_lit)) = binding.get(&Variable::new("count").unwrap()) {
            assert_eq!(count_lit.value, "100");
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_configuration() {
    let mut config = ParallelConfig::default();
    config.max_threads = 8;
    config.chunk_size = 500;
    config.work_stealing = true;

    let mut context = ExecutionContext::default();
    context.parallel = true;
    context.parallel_config = config;

    let executor = QueryExecutor::with_context(context);

    // Context is private, so we can't directly verify the configuration
    // The configuration was set correctly by passing it to with_context
    // This test verifies that the executor was created successfully
}
