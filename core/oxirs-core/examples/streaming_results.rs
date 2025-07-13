//! Example demonstrating streaming result sets for large query results

use oxirs_core::model::*;
use oxirs_core::query::{StreamingResultBuilder, StreamingSolution};
use oxirs_core::OxirsError;
use std::collections::HashMap;
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<(), OxirsError> {
    println!("=== Streaming Result Sets Example ===\n");

    // Example 1: Basic streaming SELECT results
    println!("Example 1: Basic Streaming SELECT");
    basic_streaming_example()?;

    // Example 2: Batch processing
    println!("\nExample 2: Batch Processing");
    batch_processing_example()?;

    // Example 3: Progress tracking
    println!("\nExample 3: Progress Tracking");
    progress_tracking_example()?;

    // Example 4: Cancellation
    println!("\nExample 4: Query Cancellation");
    cancellation_example()?;

    // Example 5: Streaming CONSTRUCT results
    println!("\nExample 5: Streaming CONSTRUCT");
    construct_streaming_example()?;

    // Example 6: Memory-efficient processing
    println!("\nExample 6: Memory-Efficient Processing");
    memory_efficient_example()?;

    Ok(())
}

fn basic_streaming_example() -> Result<(), OxirsError> {
    // Create streaming results for a SELECT query
    let builder = StreamingResultBuilder::new().with_buffer_size(100);

    let variables = vec![
        Variable::new("person")?,
        Variable::new("name")?,
        Variable::new("email")?,
    ];

    let (mut results, sender) = builder.build_select(variables.clone());

    // Simulate query execution in background
    thread::spawn(move || {
        for i in 0..10 {
            let mut bindings = HashMap::new();
            bindings.insert(
                variables[0].clone(),
                Some(Term::NamedNode(NamedNode::new(format!(
                    "http://example.org/person/{i}"
                ))?)),
            );
            bindings.insert(
                variables[1].clone(),
                Some(Term::Literal(Literal::new(format!("Person {i}")))),
            );
            bindings.insert(
                variables[2].clone(),
                Some(Term::Literal(Literal::new(format!(
                    "person{i}@example.com"
                )))),
            );

            sender.send(Ok(StreamingSolution::new(bindings))).unwrap();
            thread::sleep(Duration::from_millis(50)); // Simulate query delay
        }
        Ok::<(), OxirsError>(())
    });

    // Process results as they arrive
    println!("Variables: {:?}", results.variables());
    println!("Results:");

    let mut count = 0;
    while let Ok(Some(solution)) = results.next() {
        println!("  Solution {}:", count + 1);
        for var in results.variables() {
            if let Some(value) = solution.get(var) {
                println!("    ?{var} = {value}");
            }
        }
        count += 1;
    }

    println!("Total solutions: {count}");
    Ok(())
}

fn batch_processing_example() -> Result<(), OxirsError> {
    let builder = StreamingResultBuilder::new().with_buffer_size(500);

    let variables = vec![Variable::new("item")?, Variable::new("value")?];
    let (mut results, sender) = builder.build_select(variables.clone());

    // Generate many results
    thread::spawn(move || {
        for i in 0..1000 {
            let mut bindings = HashMap::new();
            bindings.insert(
                variables[0].clone(),
                Some(Term::NamedNode(NamedNode::new(format!(
                    "http://example.org/item/{i}"
                ))?)),
            );
            bindings.insert(
                variables[1].clone(),
                Some(Term::Literal(Literal::new((i * 10).to_string()))),
            );
            sender.send(Ok(StreamingSolution::new(bindings))).unwrap();
        }
        Ok::<(), OxirsError>(())
    });

    // Process in batches
    let mut batch_count = 0;
    let mut total_processed = 0;

    loop {
        let batch = results.next_batch(100)?;
        if batch.is_empty() {
            break;
        }

        batch_count += 1;
        total_processed += batch.len();

        // Process batch (e.g., bulk insert to database)
        let batch_size = batch.len();
        println!("Processing batch {batch_count} with {batch_size} items");

        // Simulate batch processing time
        thread::sleep(Duration::from_millis(10));
    }

    println!("Processed {total_processed} items in {batch_count} batches");
    Ok(())
}

fn progress_tracking_example() -> Result<(), OxirsError> {
    let builder = StreamingResultBuilder::new()
        .with_buffer_size(100)
        .with_progress_tracking(true);

    let variables = vec![Variable::new("resource")?];
    let (mut results, sender) = builder.build_select(variables.clone());

    // Simulate slow query
    thread::spawn(move || {
        for i in 0..50 {
            let mut bindings = HashMap::new();
            bindings.insert(
                variables[0].clone(),
                Some(Term::NamedNode(NamedNode::new(format!(
                    "http://example.org/resource/{i}"
                ))?)),
            );
            sender.send(Ok(StreamingSolution::new(bindings))).unwrap();
            thread::sleep(Duration::from_millis(100)); // Slow query
        }
        Ok::<(), OxirsError>(())
    });

    // Track progress while processing
    let start = Instant::now();
    let mut last_update = Instant::now();

    while let Ok(Some(_solution)) = results.next() {
        // Update progress every second
        if last_update.elapsed() >= Duration::from_secs(1) {
            let progress = results.progress();
            let elapsed = start.elapsed();
            let rate = progress.processed as f64 / elapsed.as_secs_f64();

            println!(
                "Progress: {} results processed, {:.1} results/sec, memory: {} KB",
                progress.processed,
                rate,
                progress.memory_used / 1024
            );

            last_update = Instant::now();
        }
    }

    let final_progress = results.progress();
    println!(
        "Query completed: {} total results in {:.1}s",
        final_progress.processed,
        start.elapsed().as_secs_f64()
    );

    Ok(())
}

fn cancellation_example() -> Result<(), OxirsError> {
    let builder = StreamingResultBuilder::new();
    let variables = vec![Variable::new("x")?];
    let (mut results, sender) = builder.build_select(variables.clone());

    // Start infinite query
    let producer = thread::spawn(move || {
        let mut i = 0;
        loop {
            let mut bindings = HashMap::new();
            bindings.insert(
                variables[0].clone(),
                Some(Term::Literal(Literal::new(i.to_string()))),
            );

            if sender.send(Ok(StreamingSolution::new(bindings))).is_err() {
                break; // Receiver dropped
            }

            i += 1;
            thread::sleep(Duration::from_millis(10));
        }
    });

    // Process results with limit
    let mut count = 0;
    let limit = 100;

    println!("Processing results (will cancel after {limit})...");

    while let Ok(Some(_)) = results.next() {
        count += 1;

        if count >= limit {
            println!("Cancelling query after {count} results");
            results.cancel();
            break;
        }
    }

    // Verify cancellation
    assert!(results.is_cancelled());
    println!("Query cancelled successfully");

    // Producer will stop when channel is closed
    drop(results);
    let _ = producer.join();

    Ok(())
}

fn construct_streaming_example() -> Result<(), OxirsError> {
    let builder = StreamingResultBuilder::new().with_buffer_size(200);

    let (mut results, sender) = builder.build_construct();

    // Generate triples
    thread::spawn(move || {
        let pred = NamedNode::new("http://example.org/hasValue")?;

        for i in 0..20 {
            let subj = NamedNode::new(format!("http://example.org/item/{i}"))?;
            let obj = Literal::new(format!("Value {i}"));

            let triple = Triple::new(subj, pred.clone(), obj);
            sender.send(Ok(triple)).unwrap();
            thread::sleep(Duration::from_millis(50));
        }
        Ok::<(), OxirsError>(())
    });

    // Process constructed triples
    println!("Constructed triples:");
    let mut count = 0;

    while let Ok(Some(triple)) = results.next() {
        println!("  {triple}");
        count += 1;
    }

    println!("Total triples constructed: {count}");
    Ok(())
}

fn memory_efficient_example() -> Result<(), OxirsError> {
    // Configure for minimal memory usage
    let builder = StreamingResultBuilder::new()
        .with_buffer_size(10) // Small buffer
        .with_max_memory(1024 * 1024); // 1MB limit

    let variables = vec![Variable::new("data")?];
    let (mut results, sender) = builder.build_select(variables.clone());

    // Generate large dataset
    thread::spawn(move || {
        for i in 0..10000 {
            let mut bindings = HashMap::new();
            // Each result contains some data
            let data = format!("Data item {i} with some content");
            bindings.insert(
                variables[0].clone(),
                Some(Term::Literal(Literal::new(data))),
            );

            if sender.send(Ok(StreamingSolution::new(bindings))).is_err() {
                break;
            }
        }
    });

    // Process with minimal memory footprint
    println!("Processing large dataset with minimal memory...");

    let mut count = 0;
    let mut checkpoints = vec![1000, 5000, 10000];

    while let Ok(Some(_solution)) = results.next() {
        count += 1;

        // Don't accumulate results in memory
        // Process and discard immediately

        if Some(&count) == checkpoints.first() {
            checkpoints.remove(0);
            let progress = results.progress();
            let memory_kb = progress.memory_used / 1024;
            println!("Checkpoint {count}: Memory usage = {memory_kb} KB");
        }
    }

    println!("Processed {count} results efficiently");
    Ok(())
}
