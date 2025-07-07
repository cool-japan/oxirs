//! Example demonstrating adaptive indexing that learns from query patterns

use oxirs_core::model::*;
use oxirs_core::store::{AdaptiveConfig, AdaptiveIndexManager, IndexedGraph};
use rand::Rng;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Adaptive Indexing Example ===\n");

    example_query_pattern_learning()?;
    example_index_creation()?;
    example_performance_comparison()?;

    Ok(())
}

fn example_query_pattern_learning() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Query Pattern Learning");
    println!("{}", "-".repeat(40));

    // Create adaptive index manager with low thresholds for demo
    let config = AdaptiveConfig {
        min_queries_for_index: 5,
        min_frequency_for_index: 0.01,
        maintenance_interval: Duration::from_millis(100),
        ..Default::default()
    };

    let graph = IndexedGraph::new();
    let manager = AdaptiveIndexManager::new(graph, config);

    // Insert sample data
    println!("Inserting sample RDF data...");
    for i in 0..100 {
        let person = NamedNode::new(format!("http://example.org/person{i}"))?;
        let name_pred = NamedNode::new("http://example.org/name")?;
        let age_pred = NamedNode::new("http://example.org/age")?;
        let knows_pred = NamedNode::new("http://example.org/knows")?;

        // Add name
        manager.insert(Triple::new(
            person.clone(),
            name_pred,
            Literal::new(format!("Person {i}")),
        ))?;

        // Add age
        manager.insert(Triple::new(
            person.clone(),
            age_pred,
            Literal::new(format!("{}", 20 + i % 50)),
        ))?;

        // Add relationships
        if i > 0 {
            let friend = NamedNode::new(format!("http://example.org/person{}", i - 1))?;
            manager.insert(Triple::new(person, knows_pred, friend))?;
        }
    }

    println!("Inserted {} triples\n", 100 * 3 - 1);

    // Simulate different query patterns
    println!("Simulating query patterns...");

    // Pattern 1: Frequent predicate queries (should trigger predicate index)
    let name_pred = Predicate::NamedNode(NamedNode::new("http://example.org/name")?);
    for _ in 0..10 {
        let results = manager.query(None, Some(&name_pred), None)?;
        assert_eq!(results.len(), 100);
    }
    println!("  Executed 10 queries for pattern: (? name ?)");

    // Pattern 2: Subject-predicate queries
    let person1 = Subject::NamedNode(NamedNode::new("http://example.org/person1")?);
    let knows_pred = Predicate::NamedNode(NamedNode::new("http://example.org/knows")?);
    for _ in 0..8 {
        let results = manager.query(Some(&person1), Some(&knows_pred), None)?;
        assert!(!results.is_empty());
    }
    println!("  Executed 8 queries for pattern: (person1 knows ?)");

    // Wait for maintenance to run
    std::thread::sleep(Duration::from_millis(200));

    // Check statistics
    let stats = manager.get_stats();
    println!("\nQuery Statistics:");
    for (pattern, pattern_stats) in &stats.pattern_stats {
        println!("  {:?}:", pattern);
        println!("    Query count: {}", pattern_stats.query_count);
        println!("    Avg result size: {:.2}", pattern_stats.avg_result_size);
        println!(
            "    Query frequency: {:.3} queries/sec",
            pattern_stats.query_frequency
        );
    }

    println!("\nActive adaptive indexes: {:?}", stats.active_indexes);
    println!();

    Ok(())
}

fn example_index_creation() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Automatic Index Creation");
    println!("{}", "-".repeat(40));

    let config = AdaptiveConfig {
        min_queries_for_index: 3,
        min_frequency_for_index: 0.01,
        max_adaptive_indexes: 3,
        maintenance_interval: Duration::from_millis(50),
        ..Default::default()
    };

    let graph = IndexedGraph::new();
    let manager = AdaptiveIndexManager::new(graph, config);

    // Insert diverse data
    println!("Creating diverse RDF dataset...");
    let predicates = ["type", "label", "comment", "seeAlso", "isPartOf"];
    let types = ["Person", "Organization", "Document", "Project", "Event"];

    for i in 0..200 {
        let subject = NamedNode::new(format!("http://example.org/resource{i}"))?;

        // Add type
        let type_pred = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;
        let type_obj = NamedNode::new(&format!("http://example.org/{}", types[i % types.len()]))?;
        manager.insert(Triple::new(subject.clone(), type_pred, type_obj))?;

        // Add other properties
        for (j, pred_name) in predicates.iter().enumerate() {
            if i % (j + 2) == 0 {
                let pred = NamedNode::new(&format!("http://example.org/{}", pred_name))?;
                let obj = Literal::new(&format!("{} value {}", pred_name, i));
                manager.insert(Triple::new(subject.clone(), pred, obj))?;
            }
        }
    }

    // Simulate realistic query workload
    println!("\nSimulating realistic query workload...");
    let mut rng = rand::thread_rng();

    // Heavy type queries (should create index)
    let type_pred = Predicate::NamedNode(NamedNode::new(
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    )?);
    println!("  Executing type queries...");
    for _ in 0..20 {
        manager.query(None, Some(&type_pred), None)?;
    }

    // Moderate label queries
    let label_pred = Predicate::NamedNode(NamedNode::new("http://example.org/label")?);
    println!("  Executing label queries...");
    for _ in 0..10 {
        manager.query(None, Some(&label_pred), None)?;
    }

    // Sparse random queries
    println!("  Executing random queries...");
    for _ in 0..5 {
        let random_id = rng.gen_range(0..200);
        let subject = Subject::NamedNode(NamedNode::new(&format!(
            "http://example.org/resource{}",
            random_id
        ))?);
        manager.query(Some(&subject), None, None)?;
    }

    // Wait for maintenance
    std::thread::sleep(Duration::from_millis(100));

    // Display results
    let stats = manager.get_stats();
    println!("\nAdaptive Index Status:");
    println!("  Total queries: {}", stats.total_queries);
    println!("  Active indexes created: {}", stats.active_indexes.len());
    for pattern in &stats.active_indexes {
        println!("    - {:?} index", pattern);
    }

    Ok(())
}

fn example_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nExample 3: Performance Comparison");
    println!("{}", "-".repeat(40));

    // Create two managers - one adaptive, one not
    let base_graph1 = IndexedGraph::new();
    let base_graph2 = IndexedGraph::new();

    let adaptive_config = AdaptiveConfig {
        min_queries_for_index: 5,
        min_frequency_for_index: 0.01,
        maintenance_interval: Duration::from_millis(50),
        ..Default::default()
    };

    let adaptive_manager = AdaptiveIndexManager::new(base_graph1, adaptive_config);

    // Insert same data into both
    println!("Inserting 1000 triples into both systems...");
    for i in 0..1000 {
        let triple = Triple::new(
            NamedNode::new(&format!("http://example.org/s{}", i % 100))?,
            NamedNode::new(&format!("http://example.org/p{}", i % 10))?,
            Literal::new(&format!("Object {}", i)),
        );
        adaptive_manager.insert(triple.clone())?;
        base_graph2.insert(&triple);
    }

    // Warm up adaptive indexing with repeated queries
    println!("\nWarming up adaptive indexes...");
    let frequent_pred = Predicate::NamedNode(NamedNode::new("http://example.org/p1")?);
    for _ in 0..10 {
        adaptive_manager.query(None, Some(&frequent_pred), None)?;
    }

    // Wait for index creation
    std::thread::sleep(Duration::from_millis(100));

    // Benchmark queries
    println!("\nBenchmarking query performance...");
    let num_queries = 100;

    // Benchmark adaptive system
    let start = Instant::now();
    for _ in 0..num_queries {
        adaptive_manager.query(None, Some(&frequent_pred), None)?;
    }
    let adaptive_duration = start.elapsed();

    // Benchmark base system
    let start = Instant::now();
    for _ in 0..num_queries {
        base_graph2.match_pattern(None, Some(&frequent_pred), None);
    }
    let base_duration = start.elapsed();

    println!("\nResults for {} queries:", num_queries);
    println!("  Base IndexedGraph: {:?}", base_duration);
    println!("  Adaptive Indexing: {:?}", adaptive_duration);

    let speedup = base_duration.as_secs_f64() / adaptive_duration.as_secs_f64();
    if speedup > 1.0 {
        println!("  Speedup: {:.2}x faster with adaptive indexing", speedup);
    } else {
        println!("  Note: Adaptive indexing overhead for this small dataset");
    }

    // Show final statistics
    let stats = adaptive_manager.get_stats();
    println!("\nFinal adaptive index statistics:");
    println!("  Total queries executed: {}", stats.total_queries);
    println!(
        "  Indexes automatically created: {:?}",
        stats.active_indexes
    );

    Ok(())
}
