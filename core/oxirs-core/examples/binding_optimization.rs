//! Example demonstrating variable binding optimization for SPARQL queries

use oxirs_core::model::*;
use oxirs_core::query::binding_optimizer::{BindingMetadata, RelationType, ValueConstraintType};
use oxirs_core::query::{BindingIterator, BindingOptimizer, BindingSet, Constraint, TermType};
use std::collections::{HashMap, HashSet};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Variable Binding Optimization Example ===\n");

    // Example 1: Basic binding with type constraints
    println!("Example 1: Type Constraints");
    type_constraint_example()?;

    // Example 2: Value constraints and validation
    println!("\nExample 2: Value Constraints");
    value_constraint_example()?;

    // Example 3: Relationship constraints between variables
    println!("\nExample 3: Relationship Constraints");
    relationship_constraint_example()?;

    // Example 4: Binding optimization with caching
    println!("\nExample 4: Binding Optimizer with Caching");
    optimizer_caching_example()?;

    // Example 5: Complex query with multiple constraints
    println!("\nExample 5: Complex Query Optimization");
    complex_query_example()?;

    Ok(())
}

fn type_constraint_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut bindings = BindingSet::new();

    // Variables for a person query
    let var_person = Variable::new("person")?;
    let var_name = Variable::new("name")?;
    let var_age = Variable::new("age")?;

    // Add type constraints
    bindings.add_constraint(Constraint::TypeConstraint {
        variable: var_person.clone(),
        allowed_types: vec![TermType::NamedNode, TermType::BlankNode]
            .into_iter()
            .collect(),
    });

    bindings.add_constraint(Constraint::TypeConstraint {
        variable: var_name.clone(),
        allowed_types: vec![TermType::StringLiteral].into_iter().collect(),
    });

    bindings.add_constraint(Constraint::TypeConstraint {
        variable: var_age.clone(),
        allowed_types: vec![TermType::NumericLiteral].into_iter().collect(),
    });

    // Try valid bindings
    let person = Term::NamedNode(NamedNode::new("http://example.org/alice")?);
    let name = Term::Literal(Literal::new("Alice"));
    let age = Term::Literal(Literal::new("30"));

    println!("Binding ?person = {person}");
    bindings.bind(var_person.clone(), person, BindingMetadata::default())?;

    println!("Binding ?name = {name}");
    bindings.bind(var_name.clone(), name, BindingMetadata::default())?;

    println!("Binding ?age = {age}");
    bindings.bind(var_age.clone(), age, BindingMetadata::default())?;

    println!("All bindings successful with type constraints!");

    // Try invalid binding
    let invalid_age = Term::Literal(Literal::new("thirty")); // Not numeric
    println!("\nTrying invalid binding ?age = {invalid_age}");
    match bindings.bind(var_age, invalid_age, BindingMetadata::default()) {
        Ok(_) => println!("Binding succeeded (unexpected)"),
        Err(e) => println!("Binding failed as expected: {e}"),
    }

    Ok(())
}

fn value_constraint_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut bindings = BindingSet::new();

    // Variable for age with range constraint
    let var_age = Variable::new("age")?;
    bindings.add_constraint(Constraint::ValueConstraint {
        variable: var_age.clone(),
        constraint: ValueConstraintType::NumericRange {
            min: 0.0,
            max: 150.0,
        },
    });

    // Variable for email with pattern constraint
    let var_email = Variable::new("email")?;
    let email_regex = regex::Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")?;
    bindings.add_constraint(Constraint::ValueConstraint {
        variable: var_email.clone(),
        constraint: ValueConstraintType::StringPattern(email_regex),
    });

    // Variable for country with allowed values
    let var_country = Variable::new("country")?;
    let allowed_countries: HashSet<Term> = vec![
        Term::Literal(Literal::new("USA")),
        Term::Literal(Literal::new("UK")),
        Term::Literal(Literal::new("Canada")),
        Term::Literal(Literal::new("Australia")),
    ]
    .into_iter()
    .collect();

    bindings.add_constraint(Constraint::ValueConstraint {
        variable: var_country.clone(),
        constraint: ValueConstraintType::OneOf(allowed_countries),
    });

    // Test valid values
    println!("Testing valid values:");

    let valid_age = Term::Literal(Literal::new("25"));
    println!(
        "  Age 25: {}",
        if bindings
            .bind(var_age.clone(), valid_age, BindingMetadata::default())
            .is_ok()
        {
            "Valid"
        } else {
            "Invalid"
        }
    );

    let valid_email = Term::Literal(Literal::new("alice@example.com"));
    println!(
        "  Email alice@example.com: {}",
        if bindings
            .bind(var_email.clone(), valid_email, BindingMetadata::default())
            .is_ok()
        {
            "Valid"
        } else {
            "Invalid"
        }
    );

    let valid_country = Term::Literal(Literal::new("USA"));
    println!(
        "  Country USA: {}",
        if bindings
            .bind(
                var_country.clone(),
                valid_country,
                BindingMetadata::default()
            )
            .is_ok()
        {
            "Valid"
        } else {
            "Invalid"
        }
    );

    // Test invalid values
    println!("\nTesting invalid values:");

    let invalid_age = Term::Literal(Literal::new("200"));
    println!(
        "  Age 200: {}",
        if bindings
            .bind(var_age, invalid_age, BindingMetadata::default())
            .is_ok()
        {
            "Valid"
        } else {
            "Invalid"
        }
    );

    let invalid_email = Term::Literal(Literal::new("not-an-email"));
    println!(
        "  Email not-an-email: {}",
        if bindings
            .bind(var_email, invalid_email, BindingMetadata::default())
            .is_ok()
        {
            "Valid"
        } else {
            "Invalid"
        }
    );

    let invalid_country = Term::Literal(Literal::new("Mars"));
    println!(
        "  Country Mars: {}",
        if bindings
            .bind(var_country, invalid_country, BindingMetadata::default())
            .is_ok()
        {
            "Valid"
        } else {
            "Invalid"
        }
    );

    Ok(())
}

fn relationship_constraint_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create variables for a date range query
    let var_start = Variable::new("startDate")?;
    let var_end = Variable::new("endDate")?;
    let var_duration = Variable::new("duration")?;

    // Possible values
    let mut possible_values = HashMap::new();
    possible_values.insert(
        var_start.clone(),
        vec![
            Term::Literal(Literal::new("2024-01-01")),
            Term::Literal(Literal::new("2024-02-01")),
            Term::Literal(Literal::new("2024-03-01")),
        ],
    );
    possible_values.insert(
        var_end.clone(),
        vec![
            Term::Literal(Literal::new("2024-01-15")),
            Term::Literal(Literal::new("2024-02-15")),
            Term::Literal(Literal::new("2024-03-15")),
        ],
    );
    possible_values.insert(
        var_duration.clone(),
        vec![
            Term::Literal(Literal::new("10")),
            Term::Literal(Literal::new("20")),
            Term::Literal(Literal::new("30")),
        ],
    );

    // Add constraint: startDate < endDate
    let constraints = vec![Constraint::RelationshipConstraint {
        left: var_start.clone(),
        right: var_end.clone(),
        relation: RelationType::LessThan,
    }];

    // Create iterator
    let mut iterator = BindingIterator::new(
        vec![HashMap::new()],
        vec![var_start, var_end, var_duration],
        possible_values,
        constraints,
    );

    // Find valid combinations
    println!("Valid date range combinations:");
    let mut count = 0;
    while let Some(binding) = iterator.next_valid() {
        if count < 5 {
            // Show first 5
            println!(
                "  Start: {}, End: {}, Duration: {} days",
                binding.get(&Variable::new("startDate")?).unwrap(),
                binding.get(&Variable::new("endDate")?).unwrap(),
                binding.get(&Variable::new("duration")?).unwrap(),
            );
        }
        count += 1;
    }
    println!("Total valid combinations: {count}");

    Ok(())
}

fn optimizer_caching_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut optimizer = BindingOptimizer::new();

    // Create a query pattern with variables and constraints
    let vars = vec![
        Variable::new("subject")?,
        Variable::new("predicate")?,
        Variable::new("object")?,
    ];

    let constraints = vec![
        Constraint::TypeConstraint {
            variable: Variable::new("subject")?,
            allowed_types: vec![TermType::NamedNode, TermType::BlankNode]
                .into_iter()
                .collect(),
        },
        Constraint::TypeConstraint {
            variable: Variable::new("predicate")?,
            allowed_types: vec![TermType::NamedNode].into_iter().collect(),
        },
    ];

    // First optimization (cache miss)
    println!("First optimization call...");
    let start = std::time::Instant::now();
    let _bindings1 = optimizer.optimize_bindings(vars.clone(), constraints.clone());
    let duration1 = start.elapsed();
    println!("  Duration: {duration1:?}");

    // Second optimization (cache hit)
    println!("\nSecond optimization call (should hit cache)...");
    let start = std::time::Instant::now();
    let _bindings2 = optimizer.optimize_bindings(vars.clone(), constraints.clone());
    let duration2 = start.elapsed();
    println!("  Duration: {duration2:?}");

    // Third with different constraints (cache miss)
    println!("\nThird optimization call with different constraints...");
    let new_constraints = vec![Constraint::TypeConstraint {
        variable: Variable::new("object")?,
        allowed_types: vec![TermType::Literal].into_iter().collect(),
    }];
    let start = std::time::Instant::now();
    let _bindings3 = optimizer.optimize_bindings(vars, new_constraints);
    let duration3 = start.elapsed();
    println!("  Duration: {duration3:?}");

    // Show statistics
    println!("\nOptimizer statistics:");
    println!("{}", optimizer.stats());

    Ok(())
}

fn complex_query_example() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate a complex SPARQL query:
    // SELECT ?person ?name ?age ?city
    // WHERE {
    //   ?person rdf:type foaf:Person .
    //   ?person foaf:name ?name .
    //   ?person foaf:age ?age .
    //   ?person foaf:based_near ?city .
    //   FILTER(?age >= 18 && ?age <= 65)
    //   FILTER(REGEX(?name, "^[A-Z]"))
    // }

    let mut bindings = BindingSet::new();

    // Variables
    let var_person = Variable::new("person")?;
    let var_name = Variable::new("name")?;
    let var_age = Variable::new("age")?;
    let var_city = Variable::new("city")?;

    // Type constraints
    bindings.add_constraint(Constraint::TypeConstraint {
        variable: var_person.clone(),
        allowed_types: vec![TermType::NamedNode].into_iter().collect(),
    });

    bindings.add_constraint(Constraint::TypeConstraint {
        variable: var_name.clone(),
        allowed_types: vec![TermType::StringLiteral].into_iter().collect(),
    });

    bindings.add_constraint(Constraint::TypeConstraint {
        variable: var_age.clone(),
        allowed_types: vec![TermType::NumericLiteral].into_iter().collect(),
    });

    bindings.add_constraint(Constraint::TypeConstraint {
        variable: var_city.clone(),
        allowed_types: vec![TermType::NamedNode].into_iter().collect(),
    });

    // Value constraints from FILTER clauses
    bindings.add_constraint(Constraint::ValueConstraint {
        variable: var_age.clone(),
        constraint: ValueConstraintType::NumericRange {
            min: 18.0,
            max: 65.0,
        },
    });

    let name_regex = regex::Regex::new(r"^[A-Z]")?;
    bindings.add_constraint(Constraint::ValueConstraint {
        variable: var_name.clone(),
        constraint: ValueConstraintType::StringPattern(name_regex),
    });

    // Simulate query execution with bindings
    println!("Complex query constraints:");
    println!("  - ?person must be a NamedNode");
    println!("  - ?name must be a StringLiteral starting with uppercase");
    println!("  - ?age must be a NumericLiteral between 18 and 65");
    println!("  - ?city must be a NamedNode");

    // Test some bindings
    let test_data = vec![
        (
            "Valid person",
            vec![
                (
                    var_person.clone(),
                    Term::NamedNode(NamedNode::new("http://example.org/alice")?),
                ),
                (var_name.clone(), Term::Literal(Literal::new("Alice Smith"))),
                (var_age.clone(), Term::Literal(Literal::new("30"))),
                (
                    var_city.clone(),
                    Term::NamedNode(NamedNode::new("http://example.org/NYC")?),
                ),
            ],
        ),
        (
            "Invalid age",
            vec![
                (
                    var_person.clone(),
                    Term::NamedNode(NamedNode::new("http://example.org/bob")?),
                ),
                (var_name.clone(), Term::Literal(Literal::new("Bob Jones"))),
                (var_age.clone(), Term::Literal(Literal::new("70"))), // Too old
                (
                    var_city.clone(),
                    Term::NamedNode(NamedNode::new("http://example.org/LA")?),
                ),
            ],
        ),
        (
            "Invalid name",
            vec![
                (
                    var_person.clone(),
                    Term::NamedNode(NamedNode::new("http://example.org/charlie")?),
                ),
                (var_name.clone(), Term::Literal(Literal::new("charlie"))), // Lowercase
                (var_age.clone(), Term::Literal(Literal::new("25"))),
                (
                    var_city.clone(),
                    Term::NamedNode(NamedNode::new("http://example.org/SF")?),
                ),
            ],
        ),
    ];

    println!("\nTesting bindings:");
    for (label, data) in test_data {
        println!("\n{label}:");
        let mut test_bindings = bindings.clone();
        let mut all_valid = true;

        for (var, term) in data {
            match test_bindings.bind(var.clone(), term.clone(), BindingMetadata::default()) {
                Ok(_) => println!("  ✓ ?{var} = {term}"),
                Err(e) => {
                    println!("  ✗ ?{var} = {term} ({e})");
                    all_valid = false;
                }
            }
        }

        if all_valid {
            println!("  Result: All constraints satisfied");
        } else {
            println!("  Result: Constraints violated");
        }
    }

    Ok(())
}
