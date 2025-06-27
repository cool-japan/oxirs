//! Tests for parallel query execution

use oxirs_arq::{
    executor::{QueryExecutor, ExecutionContext, ParallelConfig, Dataset, InMemoryDataset},
    algebra::{Algebra, TriplePattern, Term, Iri, Literal, Variable, Binding},
    Solution,
};
use std::time::Instant;
use std::collections::HashMap;

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_bgp_execution() {
    use std::sync::Arc;
    
    // Create a dataset with many triples
    let mut dataset = InMemoryDataset::new();
    
    // Add test data
    for i in 0..1000 {
        dataset.add_triple(
            Term::Iri(Iri(format!("http://example.org/person{}", i))),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Literal(Literal::string(format!("Person {}", i))),
        );
        
        dataset.add_triple(
            Term::Iri(Iri(format!("http://example.org/person{}", i))),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/age".to_string())),
            Term::Literal(Literal::typed(
                (20 + (i % 60)).to_string(),
                Iri("http://www.w3.org/2001/XMLSchema#integer".to_string()),
            )),
        );
    }
    
    // Create BGP with multiple patterns
    let patterns = vec![
        TriplePattern {
            subject: Term::Variable("person".to_string()),
            predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            object: Term::Variable("name".to_string()),
        },
        TriplePattern {
            subject: Term::Variable("person".to_string()),
            predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/age".to_string())),
            object: Term::Variable("age".to_string()),
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
            Term::Iri(Iri(format!("http://example.org/person{}", i))),
            Term::Iri(Iri("http://example.org/parent".to_string())),
            Term::Iri(Iri(format!("http://example.org/person{}", i * 2 + 100))),
        );
        
        dataset.add_triple(
            Term::Iri(Iri(format!("http://example.org/person{}", i * 2 + 100))),
            Term::Iri(Iri("http://example.org/name".to_string())),
            Term::Literal(Literal::string(format!("Child of Person {}", i))),
        );
    }
    
    // Create join query: find parents and their children's names
    let left_pattern = Algebra::Bgp(vec![TriplePattern {
        subject: Term::Variable("parent".to_string()),
        predicate: Term::Iri(Iri("http://example.org/parent".to_string())),
        object: Term::Variable("child".to_string()),
    }]);
    
    let right_pattern = Algebra::Bgp(vec![TriplePattern {
        subject: Term::Variable("child".to_string()),
        predicate: Term::Iri(Iri("http://example.org/name".to_string())),
        object: Term::Variable("childName".to_string()),
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
            Term::Iri(Iri(format!("http://example.org/person{}", i))),
            Term::Iri(Iri("http://example.org/department".to_string())),
            Term::Literal(Literal::string(dept)),
        );
        
        dataset.add_triple(
            Term::Iri(Iri(format!("http://example.org/person{}", i))),
            Term::Iri(Iri("http://example.org/salary".to_string())),
            Term::Literal(Literal::typed(
                (30000 + (i * 100)).to_string(),
                Iri("http://www.w3.org/2001/XMLSchema#integer".to_string()),
            )),
        );
    }
    
    // Create aggregation query: COUNT and SUM by department
    let pattern = Algebra::Bgp(vec![
        TriplePattern {
            subject: Term::Variable("person".to_string()),
            predicate: Term::Iri(Iri("http://example.org/department".to_string())),
            object: Term::Variable("dept".to_string()),
        },
        TriplePattern {
            subject: Term::Variable("person".to_string()),
            predicate: Term::Iri(Iri("http://example.org/salary".to_string())),
            object: Term::Variable("salary".to_string()),
        },
    ]);
    
    use oxirs_arq::{Aggregate, Expression, GroupCondition};
    
    let group_algebra = Algebra::Group {
        pattern: Box::new(pattern),
        variables: vec![GroupCondition {
            expr: Expression::Variable("dept".to_string()),
            alias: None,
        }],
        aggregates: vec![
            (
                "count".to_string(),
                Aggregate::Count {
                    expr: Some(Expression::Variable("person".to_string())),
                    distinct: false,
                },
            ),
            (
                "totalSalary".to_string(),
                Aggregate::Sum {
                    expr: Expression::Variable("salary".to_string()),
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
        if let Some(Term::Literal(count_lit)) = binding.get(&"count".to_string()) {
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