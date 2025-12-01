//! Property Path Examples
//!
//! This example demonstrates the various property path features and operations
//! available in oxirs-shacl after the SplitRS refactoring.

use oxirs_core::model::NamedNode;
use oxirs_shacl::{paths::PropertyPathEvaluator, PropertyPath};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS SHACL Property Path Examples ===\n");

    // Example 1: Simple Predicate Path
    println!("1. Simple Predicate Path");
    let knows_pred = NamedNode::new("http://xmlns.com/foaf/0.1/knows")?;
    let simple_path = PropertyPath::predicate(knows_pred.clone());
    println!("   Path: {:?}", simple_path);
    println!("   Is simple: {}", simple_path.is_predicate());
    println!("   Complexity: {}", simple_path.complexity());
    println!("   SPARQL: {}\n", simple_path.to_sparql_path()?);

    // Example 2: Inverse Path
    println!("2. Inverse Path (^knows)");
    let inverse_path = PropertyPath::inverse(simple_path.clone());
    println!("   Path: {:?}", inverse_path);
    println!("   Is complex: {}", inverse_path.is_complex());
    println!("   Complexity: {}", inverse_path.complexity());
    println!("   SPARQL: {}\n", inverse_path.to_sparql_path()?);

    // Example 3: Sequence Path
    println!("3. Sequence Path (foaf:knows / foaf:name)");
    let name_pred = NamedNode::new("http://xmlns.com/foaf/0.1/name")?;
    let sequence_path = PropertyPath::sequence(vec![
        PropertyPath::predicate(knows_pred.clone()),
        PropertyPath::predicate(name_pred),
    ]);
    println!("   Path: {:?}", sequence_path);
    println!("   Complexity: {}", sequence_path.complexity());
    println!("   SPARQL: {}\n", sequence_path.to_sparql_path()?);

    // Example 4: Alternative Path
    println!("4. Alternative Path (foaf:knows | foaf:friend)");
    let friend_pred = NamedNode::new("http://xmlns.com/foaf/0.1/friend")?;
    let alternative_path = PropertyPath::alternative(vec![
        PropertyPath::predicate(knows_pred.clone()),
        PropertyPath::predicate(friend_pred),
    ]);
    println!("   Path: {:?}", alternative_path);
    println!("   Complexity: {}", alternative_path.complexity());
    println!("   SPARQL: {}\n", alternative_path.to_sparql_path()?);

    // Example 5: Zero-or-More Path (Transitive Closure)
    println!("5. Zero-or-More Path (foaf:knows*)");
    let zero_or_more_path = PropertyPath::zero_or_more(PropertyPath::predicate(knows_pred.clone()));
    println!("   Path: {:?}", zero_or_more_path);
    println!("   Complexity: {}", zero_or_more_path.complexity());
    println!("   SPARQL: {}\n", zero_or_more_path.to_sparql_path()?);

    // Example 6: One-or-More Path
    println!("6. One-or-More Path (foaf:knows+)");
    let one_or_more_path = PropertyPath::one_or_more(PropertyPath::predicate(knows_pred.clone()));
    println!("   Path: {:?}", one_or_more_path);
    println!("   Complexity: {}", one_or_more_path.complexity());
    println!("   SPARQL: {}\n", one_or_more_path.to_sparql_path()?);

    // Example 7: Zero-or-One Path (Optional)
    println!("7. Zero-or-One Path (foaf:knows?)");
    let zero_or_one_path = PropertyPath::zero_or_one(PropertyPath::predicate(knows_pred.clone()));
    println!("   Path: {:?}", zero_or_one_path);
    println!("   Complexity: {}", zero_or_one_path.complexity());
    println!("   SPARQL: {}\n", zero_or_one_path.to_sparql_path()?);

    // Example 8: Complex Nested Path
    println!("8. Complex Nested Path: (knows / (name | email))*");
    let email_pred = NamedNode::new("http://xmlns.com/foaf/0.1/email")?;
    let complex_path = PropertyPath::zero_or_more(PropertyPath::sequence(vec![
        PropertyPath::predicate(knows_pred.clone()),
        PropertyPath::alternative(vec![
            PropertyPath::predicate(NamedNode::new("http://xmlns.com/foaf/0.1/name")?),
            PropertyPath::predicate(email_pred),
        ]),
    ]));
    println!("   Complexity: {}", complex_path.complexity());
    println!("   SPARQL: {}\n", complex_path.to_sparql_path()?);

    // Example 9: PropertyPathEvaluator Usage
    println!("9. PropertyPathEvaluator");
    let evaluator = PropertyPathEvaluator::new();
    println!("   Default max depth: {}", evaluator.max_depth());
    println!(
        "   Default max intermediate results: {}",
        evaluator.max_intermediate_results()
    );

    let custom_evaluator = PropertyPathEvaluator::with_limits(100, 50000);
    println!("   Custom max depth: {}", custom_evaluator.max_depth());
    println!(
        "   Custom max intermediate results: {}",
        custom_evaluator.max_intermediate_results()
    );

    // Example 10: Cache Statistics
    println!("\n10. Cache Statistics");
    let cache_stats = evaluator.get_cache_stats();
    println!("   Cache entries: {}", cache_stats.entries);
    println!("   Total cached values: {}", cache_stats.total_values);

    println!("\n=== All Examples Completed Successfully ===");

    Ok(())
}
