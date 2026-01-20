//! SPARQL-star Query Example
//!
//! This example demonstrates the usage of SPARQL 1.2 / SPARQL-star features in oxirs-arq,
//! including quoted triple patterns, RDF-star built-in functions, and nested quoted triples.
//!
//! Run with: cargo run --example sparql_star_example --features star

#[cfg(feature = "star")]
use anyhow::Result;
#[cfg(feature = "star")]
use oxirs_arq::star_integration::{
    pattern_matching, sparql_star_functions, star_statistics::SparqlStarStatistics,
    SparqlStarExecutor,
};
#[cfg(feature = "star")]
use oxirs_star::{StarTerm, StarTriple};

#[cfg(feature = "star")]
fn main() -> Result<()> {
    println!("=== SPARQL-star Example ===\n");

    // Create a SPARQL-star executor
    let mut executor = SparqlStarExecutor::new();
    let store = executor.store_mut();

    // Example 1: Basic Quoted Triples
    println!("Example 1: Basic Quoted Triples");
    println!("--------------------------------");

    // Create a basic statement: "Alice knows Bob"
    let alice_knows_bob = StarTriple::new(
        StarTerm::iri("http://example.org/Alice")?,
        StarTerm::iri("http://xmlns.com/foaf/0.1/knows")?,
        StarTerm::iri("http://example.org/Bob")?,
    );

    // Add metadata about this statement (certainty level)
    let certainty_triple = StarTriple::new(
        StarTerm::quoted_triple(alice_knows_bob.clone()),
        StarTerm::iri("http://example.org/certainty")?,
        StarTerm::literal("0.95")?,
    );

    store.insert(&certainty_triple)?;

    println!("Inserted: <<Alice knows Bob>> certainty 0.95");
    println!("Store contains {} triples\n", store.len());

    // Example 2: Provenance Tracking
    println!("Example 2: Provenance Tracking");
    println!("-------------------------------");

    // Create another statement with source information
    let bob_age_25 = StarTriple::new(
        StarTerm::iri("http://example.org/Bob")?,
        StarTerm::iri("http://xmlns.com/foaf/0.1/age")?,
        StarTerm::literal("25")?,
    );

    // Track who reported this information
    let reported_by = StarTriple::new(
        StarTerm::quoted_triple(bob_age_25.clone()),
        StarTerm::iri("http://example.org/reportedBy")?,
        StarTerm::iri("http://example.org/Alice")?,
    );

    // Track when this was reported
    let reported_at = StarTriple::new(
        StarTerm::quoted_triple(bob_age_25.clone()),
        StarTerm::iri("http://example.org/reportedAt")?,
        StarTerm::literal("2025-10-12")?,
    );

    store.insert(&reported_by)?;
    store.insert(&reported_at)?;

    println!("Inserted: <<Bob age 25>> reportedBy Alice");
    println!("Inserted: <<Bob age 25>> reportedAt 2025-10-12");
    println!("Store contains {} triples\n", store.len());

    // Example 3: Nested Quoted Triples (Meta-meta-data)
    println!("Example 3: Nested Quoted Triples");
    println!("----------------------------------");

    // Someone believes that Alice knows Bob with high certainty
    let belief_triple = StarTriple::new(
        StarTerm::iri("http://example.org/Charlie")?,
        StarTerm::iri("http://example.org/believes")?,
        StarTerm::quoted_triple(certainty_triple.clone()),
    );

    store.insert(&belief_triple)?;

    println!("Inserted: Charlie believes <<Alice knows Bob>> certainty 0.95");
    println!("This is a nested quoted triple (depth 2)");
    println!("Store contains {} triples\n", store.len());

    // Example 4: Pattern Matching Utilities
    println!("Example 4: Pattern Matching");
    println!("----------------------------");

    // Check if terms contain quoted triples
    let quoted_term =
        oxirs_arq::algebra::Term::QuotedTriple(Box::new(oxirs_arq::algebra::TriplePattern::new(
            oxirs_arq::algebra::Term::Iri(oxirs_core::model::NamedNode::new(
                "http://example.org/Alice",
            )?),
            oxirs_arq::algebra::Term::Iri(oxirs_core::model::NamedNode::new(
                "http://xmlns.com/foaf/0.1/knows",
            )?),
            oxirs_arq::algebra::Term::Iri(oxirs_core::model::NamedNode::new(
                "http://example.org/Bob",
            )?),
        )));

    let is_quoted = sparql_star_functions::is_quoted_triple(&quoted_term);
    let nesting_depth = pattern_matching::nesting_depth(&quoted_term);

    println!("Is quoted triple: {}", is_quoted);
    println!("Nesting depth: {}", nesting_depth);

    // Extract components from quoted triple
    if let Some(subject) = sparql_star_functions::get_subject(&quoted_term) {
        println!("Subject: {:?}", subject);
    }

    println!();

    // Example 5: Statistics Tracking
    println!("Example 5: Statistics Tracking");
    println!("-------------------------------");

    let mut stats = SparqlStarStatistics::new();

    // Simulate query execution statistics
    stats.record_quoted_pattern(1);
    stats.record_quoted_pattern(2); // Nested quoted triple
    stats.record_star_function();
    stats.record_execution_time(1500);
    stats.record_results(3);

    println!("Query Statistics:");
    println!(
        "  Quoted patterns matched: {}",
        stats.quoted_patterns_matched
    );
    println!("  Maximum nesting depth: {}", stats.max_nesting_depth);
    println!(
        "  SPARQL-star functions evaluated: {}",
        stats.star_functions_evaluated
    );
    println!("  Execution time: {}μs", stats.execution_time_us);
    println!("  Results returned: {}", stats.result_count);
    if let Some(avg) = stats.avg_time_per_result() {
        println!("  Average time per result: {:.2}μs", avg);
    }

    println!();

    // Example 6: Finding Triples by Quoted Pattern
    println!("Example 6: Advanced Queries");
    println!("----------------------------");

    // Find all triples that contain Alice in their quoted triple
    let alice_star = StarTerm::iri("http://example.org/Alice")?;
    let containing_alice = store.find_triples_by_quoted_pattern(Some(&alice_star), None, None);

    println!(
        "Triples containing Alice in quoted position: {}",
        containing_alice.len()
    );
    for triple in containing_alice.iter().take(3) {
        println!("  - {}", triple);
    }

    println!();

    // Example 7: Nesting Depth Analysis
    println!("Example 7: Nesting Depth Analysis");
    println!("-----------------------------------");

    // Find triples by nesting depth
    let depth_0 = store.find_triples_by_nesting_depth(0, Some(0));
    let depth_1 = store.find_triples_by_nesting_depth(1, Some(1));
    let depth_2 = store.find_triples_by_nesting_depth(2, Some(2));

    println!("Triples by nesting depth:");
    println!("  Depth 0 (no quoted triples): {}", depth_0.len());
    println!("  Depth 1 (single level): {}", depth_1.len());
    println!("  Depth 2 (nested): {}", depth_2.len());

    println!();

    // Summary
    println!("=== Summary ===");
    println!("Total triples in store: {}", store.len());
    println!("Store statistics:");
    let store_stats = store.statistics();
    println!("  Quoted triples: {}", store_stats.quoted_triples_count);
    println!(
        "  Max nesting depth: {}",
        store_stats.max_nesting_encountered
    );
    println!("  Processing time: {}μs", store_stats.processing_time_us);

    Ok(())
}

#[cfg(not(feature = "star"))]
fn main() {
    eprintln!("This example requires the 'star' feature to be enabled.");
    eprintln!("Run with: cargo run --example sparql_star_example --features star");
    std::process::exit(1);
}
