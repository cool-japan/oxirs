//! Tests for streaming result sets

use oxirs_core::model::*;
use oxirs_core::query::{
    SolutionMetadata, StreamingQueryResults, StreamingResultBuilder, StreamingSolution,
};
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

#[test]
fn test_basic_streaming() {
    let builder = StreamingResultBuilder::new().with_buffer_size(10);

    let variables = vec![Variable::new("x").unwrap()];
    let (mut results, sender) = builder.build_select(variables.clone());

    // Send data in same thread
    for i in 0..5 {
        let mut bindings = HashMap::new();
        bindings.insert(
            variables[0].clone(),
            Some(Term::Literal(Literal::new(i.to_string()))),
        );
        sender.send(Ok(StreamingSolution::new(bindings))).unwrap();
    }
    drop(sender);

    // Receive all
    let mut count = 0;
    while let Ok(Some(_)) = results.next() {
        count += 1;
    }
    assert_eq!(count, 5);
}

#[test]
fn test_streaming_with_thread() {
    let builder = StreamingResultBuilder::new().with_buffer_size(10);

    let variables = vec![Variable::new("x").unwrap()];
    let (mut results, sender) = builder.build_select(variables.clone());

    let handle = thread::spawn(move || {
        for i in 0..10 {
            let mut bindings = HashMap::new();
            bindings.insert(
                variables[0].clone(),
                Some(Term::Literal(Literal::new(i.to_string()))),
            );
            sender.send(Ok(StreamingSolution::new(bindings))).unwrap();
        }
        drop(sender);
    });

    let mut count = 0;
    while let Ok(Some(_)) = results.next() {
        count += 1;
    }

    assert_eq!(count, 10);
    handle.join().unwrap();
}

#[test]
fn test_batch_operations() {
    let builder = StreamingResultBuilder::new();
    let variables = vec![Variable::new("x").unwrap()];
    let (mut results, sender) = builder.build_select(variables.clone());

    // Send all data first
    for i in 0..50 {
        let mut bindings = HashMap::new();
        bindings.insert(
            variables[0].clone(),
            Some(Term::Literal(Literal::new(i.to_string()))),
        );
        sender.send(Ok(StreamingSolution::new(bindings))).unwrap();
    }
    drop(sender);

    // Test batch operations
    let batch1 = results.next_batch(10).unwrap();
    assert_eq!(batch1.len(), 10);

    results.skip_results(10).unwrap();

    let batch2 = results.take_results(20).unwrap();
    assert_eq!(batch2.len(), 20);

    // Should have 10 remaining
    let remaining: Vec<_> = results.filter_map(Result::ok).collect();
    assert_eq!(remaining.len(), 10);
}

#[test]
fn test_cancellation() {
    let builder = StreamingResultBuilder::new();
    let variables = vec![Variable::new("x").unwrap()];
    let (mut results, sender) = builder.build_select(variables.clone());

    let handle = thread::spawn(move || {
        for i in 0..100 {
            let mut bindings = HashMap::new();
            bindings.insert(
                variables[0].clone(),
                Some(Term::Literal(Literal::new(i.to_string()))),
            );
            if sender.send(Ok(StreamingSolution::new(bindings))).is_err() {
                break;
            }
        }
    });

    // Process a few then cancel
    let mut count = 0;
    while let Ok(Some(_)) = results.next() {
        count += 1;
        if count == 5 {
            results.cancel();
            break;
        }
    }

    assert_eq!(count, 5);
    assert!(results.is_cancelled());

    // Clean up
    drop(results);
    handle.join().unwrap();
}

#[test]
fn test_progress_tracking() {
    let builder = StreamingResultBuilder::new().with_progress_tracking(true);

    let variables = vec![Variable::new("x").unwrap()];
    let (mut results, sender) = builder.build_select(variables.clone());

    for i in 0..10 {
        let mut bindings = HashMap::new();
        bindings.insert(
            variables[0].clone(),
            Some(Term::Literal(Literal::new(i.to_string()))),
        );
        sender.send(Ok(StreamingSolution::new(bindings))).unwrap();
    }
    drop(sender);

    let mut count = 0;
    while let Ok(Some(_)) = results.next() {
        count += 1;
        let progress = results.progress();
        assert_eq!(progress.processed, count);
    }

    let final_progress = results.progress();
    assert_eq!(final_progress.processed, 10);
    assert!(!final_progress.is_running);
}

#[test]
fn test_construct_streaming() {
    let builder = StreamingResultBuilder::new();
    let (mut results, sender) = builder.build_construct();

    for i in 0..20 {
        let triple = Triple::new(
            NamedNode::new(format!("http://example.org/s{i}")).unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new(format!("Object {i}")),
        );
        sender.send(Ok(triple)).unwrap();
    }
    drop(sender);

    let mut count = 0;
    while let Ok(Some(_)) = results.next() {
        count += 1;
    }
    assert_eq!(count, 20);
}

#[test]
fn test_solution_metadata() {
    let mut bindings = HashMap::new();
    bindings.insert(
        Variable::new("x").unwrap(),
        Some(Term::Literal(Literal::new("test"))),
    );

    let metadata = SolutionMetadata {
        source: Some("http://example.org/sparql".to_string()),
        confidence: Some(0.95),
        timestamp: Some(1234567890),
    };

    let solution = StreamingSolution::with_metadata(bindings, metadata);

    assert_eq!(
        solution.get(&Variable::new("x").unwrap()),
        Some(&Term::Literal(Literal::new("test")))
    );

    let vars: Vec<_> = solution.variables().collect();
    assert_eq!(vars.len(), 1);

    let values: Vec<_> = solution.values().collect();
    assert_eq!(values.len(), 1);
}

#[test]
fn test_streaming_query_results_enum() {
    let builder = StreamingResultBuilder::new();
    let variables = vec![Variable::new("x").unwrap()];
    let (select_results, _) = builder.build_select(variables);

    let mut query_results = StreamingQueryResults::Select(select_results);
    assert!(query_results.is_select());
    assert!(!query_results.is_ask());
    assert!(!query_results.is_construct());
    assert!(query_results.as_select().is_some());

    let ask_results = StreamingQueryResults::Ask(true);
    assert!(ask_results.is_ask());
    assert_eq!(ask_results.as_ask(), Some(true));
}

#[test]
fn test_timeout_handling() {
    let builder = StreamingResultBuilder::new().with_timeout(Duration::from_millis(100));

    let variables = vec![Variable::new("x").unwrap()];
    let (mut results, sender) = builder.build_select(variables);

    // Don't send anything, just drop
    drop(sender);

    // Should get None (channel closed)
    assert!(results.next().unwrap().is_none());
}
