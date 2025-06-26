//! Example demonstrating pattern matching optimization for SPARQL queries

use oxirs_core::model::*;
use oxirs_core::query::{PatternOptimizer, PatternExecutor, IndexType};
use oxirs_core::query::algebra::{TriplePattern as AlgebraTriplePattern, TermPattern};
use oxirs_core::query::pattern_optimizer::IndexStats;
use oxirs_core::store::IndexedGraph;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Pattern Matching Optimization Example ===\n");

    // Create indexed graph and statistics
    let graph = Arc::new(IndexedGraph::new());
    let stats = Arc::new(IndexStats::new());

    // Populate graph with sample data
    populate_sample_data(&graph)?;

    // Example 1: Basic pattern optimization
    println!("Example 1: Basic Pattern Optimization");
    basic_pattern_optimization(&stats)?;

    // Example 2: Multi-pattern query optimization
    println!("\nExample 2: Multi-Pattern Query Optimization");
    multi_pattern_optimization(&stats)?;

    // Example 3: Index selection demonstration
    println!("\nExample 3: Index Selection Strategies");
    index_selection_demo(&stats)?;

    // Example 4: Performance comparison
    println!("\nExample 4: Performance Comparison");
    performance_comparison(&graph, &stats)?;

    Ok(())
}

fn populate_sample_data(graph: &Arc<IndexedGraph>) -> Result<(), Box<dyn std::error::Error>> {
    // Add people
    for i in 0..100 {
        let person = NamedNode::new(&format!("http://example.org/person/{}", i))?;
        let type_pred = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;
        let name_pred = NamedNode::new("http://xmlns.com/foaf/0.1/name")?;
        let knows_pred = NamedNode::new("http://xmlns.com/foaf/0.1/knows")?;
        let age_pred = NamedNode::new("http://xmlns.com/foaf/0.1/age")?;
        let person_type = NamedNode::new("http://xmlns.com/foaf/0.1/Person")?;

        // Add type triple
        graph.insert(&Triple::new(person.clone(), type_pred, person_type));

        // Add name
        graph.insert(&Triple::new(
            person.clone(),
            name_pred,
            Literal::new(format!("Person {}", i)),
        ));

        // Add age
        graph.insert(&Triple::new(
            person.clone(),
            age_pred,
            Literal::new(format!("{}", 20 + (i % 50))),
        ));

        // Add some knows relationships
        if i > 0 {
            let friend = NamedNode::new(&format!("http://example.org/person/{}", i - 1))?;
            graph.insert(&Triple::new(person, knows_pred, friend));
        }
    }

    println!("Added {} triples to the graph", graph.len());
    Ok(())
}

fn basic_pattern_optimization(
    stats: &Arc<IndexStats>,
) -> Result<(), Box<dyn std::error::Error>> {
    let optimizer = PatternOptimizer::new(stats.clone());

    // Create a simple pattern: ?person rdf:type foaf:Person
    let pattern = AlgebraTriplePattern {
        subject: TermPattern::Variable(Variable::new("person")?),
        predicate: TermPattern::NamedNode(
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
        ),
        object: TermPattern::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/Person")?),
    };

    let plan = optimizer.optimize_patterns(&[pattern])?;

    println!("Pattern: ?person rdf:type foaf:Person");
    println!("Optimization results:");
    println!("  - Estimated cost: {:.2}", plan.total_cost);
    println!("  - Selected index: {:?}", plan.patterns[0].1.index_type);
    println!("  - Selectivity: {:.4}", plan.patterns[0].1.selectivity);

    Ok(())
}

fn multi_pattern_optimization(
    stats: &Arc<IndexStats>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Update statistics for realistic optimization
    stats.update_predicate_count("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", 100);
    stats.update_predicate_count("http://xmlns.com/foaf/0.1/name", 100);
    stats.update_predicate_count("http://xmlns.com/foaf/0.1/age", 100);
    stats.update_predicate_count("http://xmlns.com/foaf/0.1/knows", 99);
    stats.set_total_triples(399);

    let optimizer = PatternOptimizer::new(stats.clone());

    // Create a complex query:
    // Find people who are 25 years old and their friends
    let patterns = vec![
        // ?person foaf:age "25"
        AlgebraTriplePattern {
            subject: TermPattern::Variable(Variable::new("person")?),
            predicate: TermPattern::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/age")?),
            object: TermPattern::Literal(Literal::new("25")),
        },
        // ?person rdf:type foaf:Person
        AlgebraTriplePattern {
            subject: TermPattern::Variable(Variable::new("person")?),
            predicate: TermPattern::NamedNode(
                NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
            ),
            object: TermPattern::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/Person")?),
        },
        // ?person foaf:knows ?friend
        AlgebraTriplePattern {
            subject: TermPattern::Variable(Variable::new("person")?),
            predicate: TermPattern::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/knows")?),
            object: TermPattern::Variable(Variable::new("friend")?),
        },
        // ?friend foaf:name ?friendName
        AlgebraTriplePattern {
            subject: TermPattern::Variable(Variable::new("friend")?),
            predicate: TermPattern::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/name")?),
            object: TermPattern::Variable(Variable::new("friendName")?),
        },
    ];

    let plan = optimizer.optimize_patterns(&patterns)?;

    println!("Complex query optimization:");
    println!("Total estimated cost: {:.2}", plan.total_cost);
    println!("\nOptimized execution order:");

    for (i, (pattern, strategy)) in plan.patterns.iter().enumerate() {
        println!("\n{}. Pattern:", i + 1);
        print_pattern(pattern);
        println!("   Index: {:?}", strategy.index_type);
        println!("   Cost: {:.2}", strategy.estimated_cost);
        println!("   Selectivity: {:.4}", strategy.selectivity);
        println!("   Binds variables: {:?}", strategy.bound_vars);
    }

    println!("\nVariable binding progression:");
    for (i, bindings) in plan.binding_order.iter().enumerate() {
        println!("  After step {}: {:?}", i + 1, bindings);
    }

    Ok(())
}

fn index_selection_demo(stats: &Arc<IndexStats>) -> Result<(), Box<dyn std::error::Error>> {
    let optimizer = PatternOptimizer::new(stats.clone());

    // Demonstrate different index selections based on bound variables
    let scenarios = vec![
        (
            "Bound subject",
            TriplePattern::new(
                Some(SubjectPattern::NamedNode(
                    NamedNode::new("http://example.org/alice")?,
                )),
                None,
                None,
            ),
        ),
        (
            "Bound predicate",
            TriplePattern::new(
                None,
                Some(PredicatePattern::NamedNode(
                    NamedNode::new("http://xmlns.com/foaf/0.1/name")?,
                )),
                None,
            ),
        ),
        (
            "Bound object",
            TriplePattern::new(
                None,
                None,
                Some(ObjectPattern::Literal(Literal::new("Alice"))),
            ),
        ),
        (
            "Bound subject and predicate",
            TriplePattern::new(
                Some(SubjectPattern::NamedNode(
                    NamedNode::new("http://example.org/alice")?,
                )),
                Some(PredicatePattern::NamedNode(
                    NamedNode::new("http://xmlns.com/foaf/0.1/name")?,
                )),
                None,
            ),
        ),
    ];

    for (name, pattern) in scenarios {
        let index = optimizer.get_optimal_index(&pattern, &Default::default());
        println!("{}: {:?}", name, index);
    }

    Ok(())
}

fn performance_comparison(
    graph: &Arc<IndexedGraph>,
    stats: &Arc<IndexStats>,
) -> Result<(), Box<dyn std::error::Error>> {
    let optimizer = PatternOptimizer::new(stats.clone());
    let executor = PatternExecutor::new(graph.clone(), stats.clone());

    // Create patterns for finding people aged 30 and their names
    let patterns = vec![
        AlgebraTriplePattern {
            subject: TermPattern::Variable(Variable::new("person")?),
            predicate: TermPattern::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/age")?),
            object: TermPattern::Literal(Literal::new("30")),
        },
        AlgebraTriplePattern {
            subject: TermPattern::Variable(Variable::new("person")?),
            predicate: TermPattern::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/name")?),
            object: TermPattern::Variable(Variable::new("name")?),
        },
    ];

    // Time optimization
    let start = Instant::now();
    let plan = optimizer.optimize_patterns(&patterns)?;
    let optimization_time = start.elapsed();

    // Time execution
    let start = Instant::now();
    let results = executor.execute_plan(&plan)?;
    let execution_time = start.elapsed();

    println!("Performance metrics:");
    println!("  Optimization time: {:?}", optimization_time);
    println!("  Execution time: {:?}", execution_time);
    println!("  Results found: {}", results.len());

    if !results.is_empty() {
        println!("\nSample results:");
        for (i, result) in results.iter().take(3).enumerate() {
            println!("  Result {}:", i + 1);
            for (var, term) in result {
                println!("    ?{} = {}", var, term);
            }
        }
    }

    Ok(())
}

fn print_pattern(pattern: &AlgebraTriplePattern) {
    let subject = match &pattern.subject {
        TermPattern::Variable(v) => format!("?{}", v),
        TermPattern::NamedNode(n) => format!("<{}>", n.as_str()),
        TermPattern::BlankNode(b) => format!("_:{}", b.as_str()),
        TermPattern::Literal(l) => format!("\"{}\"", l.value()),
    };

    let predicate = match &pattern.predicate {
        TermPattern::Variable(v) => format!("?{}", v),
        TermPattern::NamedNode(n) => format!("<{}>", n.as_str()),
        _ => "???".to_string(),
    };

    let object = match &pattern.object {
        TermPattern::Variable(v) => format!("?{}", v),
        TermPattern::NamedNode(n) => format!("<{}>", n.as_str()),
        TermPattern::BlankNode(b) => format!("_:{}", b.as_str()),
        TermPattern::Literal(l) => format!("\"{}\"", l.value()),
    };

    println!("   {} {} {}", subject, predicate, object);
}