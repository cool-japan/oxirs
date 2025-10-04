//! Integration tests for RdfStore
//!
//! These tests verify the core functionality of the RDF store including:
//! - Basic quad/triple operations
//! - SPARQL query execution
//! - String functions, aggregates, BIND, VALUES, UNION

use oxirs_core::model::{
    GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject, Term, Triple,
};
use oxirs_core::rdf_store::{QueryResults, RdfStore};

fn create_test_quad() -> Quad {
    let subject = NamedNode::new("http://example.org/subject").unwrap();
    let predicate = NamedNode::new("http://example.org/predicate").unwrap();
    let object = Literal::new("test object");
    let graph = NamedNode::new("http://example.org/graph").unwrap();

    Quad::new(subject, predicate, object, graph)
}

fn create_test_triple() -> Triple {
    let subject = NamedNode::new("http://example.org/subject").unwrap();
    let predicate = NamedNode::new("http://example.org/predicate").unwrap();
    let object = Literal::new("test object");

    Triple::new(subject, predicate, object)
}

#[test]
fn test_store_creation() {
    let store = RdfStore::new().unwrap();
    assert!(store.is_empty().unwrap());
    assert_eq!(store.len().unwrap(), 0);
}

#[test]
fn test_store_quad_operations() {
    // Use legacy backend for faster testing
    #[allow(unused_mut)]
    let mut store = RdfStore::new_legacy().unwrap();
    let quad = create_test_quad();

    // Test insertion
    assert!(store.insert_quad(quad.clone()).unwrap());
    assert!(!store.is_empty().unwrap());
    assert_eq!(store.len().unwrap(), 1);
    assert!(store.contains_quad(&quad).unwrap());

    // Test duplicate insertion
    assert!(!store.insert_quad(quad.clone()).unwrap());
    assert_eq!(store.len().unwrap(), 1);

    // Test removal
    assert!(store.remove_quad(&quad).unwrap());
    assert!(store.is_empty().unwrap());
    assert_eq!(store.len().unwrap(), 0);
    assert!(!store.contains_quad(&quad).unwrap());

    // Test removal of non-existent quad
    assert!(!store.remove_quad(&quad).unwrap());
}

#[test]
fn test_store_triple_operations() {
    #[allow(unused_mut)]
    let mut store = RdfStore::new().unwrap();
    let triple = create_test_triple();

    // Test insertion
    assert!(store.insert_triple(triple.clone()).unwrap());
    assert!(!store.is_empty().unwrap());
    assert_eq!(store.len().unwrap(), 1);

    // Verify the triple was inserted in the default graph
    let default_graph = GraphName::DefaultGraph;
    let quads = store
        .query_quads(None, None, None, Some(&default_graph))
        .unwrap();
    assert_eq!(quads.len(), 1);
    assert_eq!(quads[0].to_triple(), triple);
}

#[test]
fn test_store_string_insertion() {
    #[allow(unused_mut)]
    let mut store = RdfStore::new().unwrap();

    let result = store
        .insert_string_triple(
            "http://example.org/subject",
            "http://example.org/predicate",
            "test object",
        )
        .unwrap();

    assert!(result);
    assert_eq!(store.len().unwrap(), 1);
}

#[test]
fn test_store_query_patterns() {
    #[allow(unused_mut)]
    let mut store = RdfStore::new().unwrap();

    // Create test data
    let subject1 = NamedNode::new("http://example.org/subject1").unwrap();
    let subject2 = NamedNode::new("http://example.org/subject2").unwrap();
    let predicate1 = NamedNode::new("http://example.org/predicate1").unwrap();
    let predicate2 = NamedNode::new("http://example.org/predicate2").unwrap();
    let object1 = Literal::new("object1");
    let object2 = Literal::new("object2");
    let graph1 = NamedNode::new("http://example.org/graph1").unwrap();
    let graph2 = NamedNode::new("http://example.org/graph2").unwrap();

    let quad1 = Quad::new(
        subject1.clone(),
        predicate1.clone(),
        object1.clone(),
        graph1,
    );
    let quad2 = Quad::new(
        subject1.clone(),
        predicate2.clone(),
        object2.clone(),
        graph2.clone(),
    );
    let quad3 = Quad::new(
        subject2,
        predicate1.clone(),
        object2.clone(),
        graph2.clone(),
    );

    // Insert test data
    store.insert_quad(quad1).unwrap();
    store.insert_quad(quad2).unwrap();
    store.insert_quad(quad3).unwrap();

    // Test query by subject
    let s = Subject::NamedNode(subject1);
    let results = store.query_quads(Some(&s), None, None, None).unwrap();
    assert_eq!(results.len(), 2);

    // Test query by predicate
    let p = Predicate::from(predicate1);
    let results = store.query_quads(None, Some(&p), None, None).unwrap();
    assert_eq!(results.len(), 2);

    // Test query by object
    let o = Object::Literal(object2);
    let results = store.query_quads(None, None, Some(&o), None).unwrap();
    assert_eq!(results.len(), 2);

    // Test query by graph
    let g = GraphName::NamedNode(graph2);
    let results = store.query_quads(None, None, None, Some(&g)).unwrap();
    assert_eq!(results.len(), 2);

    // Test combined query
    let results = store.query_quads(Some(&s), Some(&p), None, None).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_store_clear() {
    #[allow(unused_mut)]
    let mut store = RdfStore::new().unwrap();

    // Insert some data
    for i in 0..5 {
        let subject = NamedNode::new(format!("http://example.org/subject{i}")).unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new(format!("object{i}"));

        let triple = Triple::new(subject, predicate, object);
        store.insert_triple(triple).unwrap();
    }

    assert_eq!(store.len().unwrap(), 5);

    // Clear the store
    store.clear().unwrap();
    assert!(store.is_empty().unwrap());
    assert_eq!(store.len().unwrap(), 0);
}

#[test]
fn test_bulk_insert() {
    #[allow(unused_mut)]
    let mut store = RdfStore::new().unwrap();

    let mut quads = Vec::new();
    for i in 0..100 {
        let subject = NamedNode::new(format!("http://example.org/subject{i}")).unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new(format!("object{i}"));
        let graph = NamedNode::new("http://example.org/graph").unwrap();

        quads.push(Quad::new(subject, predicate, object, graph));
    }

    let ids = store.bulk_insert_quads(quads).unwrap();
    assert_eq!(ids.len(), 100);
    assert_eq!(store.len().unwrap(), 100);
}

#[test]
fn test_default_graph_operations() {
    #[allow(unused_mut)]
    let mut store = RdfStore::new().unwrap();

    // Insert into default graph
    let triple = create_test_triple();
    store.insert_triple(triple).unwrap();

    // Insert into named graph
    let quad = create_test_quad();
    store.insert_quad(quad).unwrap();

    // Query default graph
    let default_triples = store.query_triples(None, None, None).unwrap();
    assert_eq!(default_triples.len(), 1);

    // Query all quads
    let all_quads = store.iter_quads().unwrap();
    assert_eq!(all_quads.len(), 2);
}

#[test]
fn test_sparql_union() {
    #[allow(unused_mut)]
    let mut store = RdfStore::new().unwrap();

    // Insert test data: Alice (age 30, name "Alice") and Bob (age 25, name "Bob")
    let alice = NamedNode::new("http://example.org/Alice").unwrap();
    let bob = NamedNode::new("http://example.org/Bob").unwrap();
    let age_pred = NamedNode::new("http://example.org/age").unwrap();
    let name_pred = NamedNode::new("http://example.org/name").unwrap();

    store
        .insert_triple(Triple::new(
            alice.clone(),
            age_pred.clone(),
            Literal::new("30"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            alice.clone(),
            name_pred.clone(),
            Literal::new("Alice"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            bob.clone(),
            age_pred.clone(),
            Literal::new("25"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            bob.clone(),
            name_pred.clone(),
            Literal::new("Bob"),
        ))
        .unwrap();

    // Test UNION: get people who are 30 years old OR named "Bob"
    let query = r#"
        SELECT ?person WHERE {
            { ?person <http://example.org/age> "30" }
            UNION
            { ?person <http://example.org/name> "Bob" }
        }
    "#;

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 2, "Should return both Alice and Bob");

    // Verify variables
    assert_eq!(results.variables(), &["person"]);

    // Test UNION with DISTINCT (should still be 2 results)
    let query_distinct = r#"
        SELECT DISTINCT ?person WHERE {
            { ?person <http://example.org/age> "30" }
            UNION
            { ?person <http://example.org/name> "Bob" }
        }
    "#;

    let results_distinct = store.query(query_distinct).unwrap();
    assert_eq!(results_distinct.len(), 2);

    // Test UNION with multiple patterns in each branch
    let query_multi = r#"
        SELECT ?person ?value WHERE {
            { ?person <http://example.org/age> ?value }
            UNION
            { ?person <http://example.org/name> ?value }
        }
    "#;

    let results_multi = store.query(query_multi).unwrap();
    assert_eq!(
        results_multi.len(),
        4,
        "Should return 4 results (2 ages + 2 names)"
    );
}

#[test]
fn test_sparql_count_aggregate() {
    #[allow(unused_mut)]
    let mut store = RdfStore::new().unwrap();

    // Insert test data: Alice (age 30, name "Alice") and Bob (age 25, name "Bob")
    let alice = NamedNode::new("http://example.org/Alice").unwrap();
    let bob = NamedNode::new("http://example.org/Bob").unwrap();
    let charlie = NamedNode::new("http://example.org/Charlie").unwrap();
    let age_pred = NamedNode::new("http://example.org/age").unwrap();
    let name_pred = NamedNode::new("http://example.org/name").unwrap();

    store
        .insert_triple(Triple::new(
            alice.clone(),
            age_pred.clone(),
            Literal::new("30"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            alice.clone(),
            name_pred.clone(),
            Literal::new("Alice"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            bob.clone(),
            age_pred.clone(),
            Literal::new("25"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            bob.clone(),
            name_pred.clone(),
            Literal::new("Bob"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            charlie.clone(),
            name_pred.clone(),
            Literal::new("Charlie"),
        ))
        .unwrap();

    // Test COUNT(*) - count all results
    let query_count_all = r#"
        SELECT (COUNT(*) AS ?count) WHERE {
            ?person <http://example.org/name> ?name
        }
    "#;

    let results = store.query(query_count_all).unwrap();
    assert_eq!(results.len(), 1, "Should return one aggregate result");
    assert_eq!(results.variables(), &["count"]);

    // Verify count value is 3 (Alice, Bob, Charlie have names)
    if let QueryResults::Bindings(bindings) = results.results() {
        let count_val = bindings[0].get("count").unwrap();
        if let Term::Literal(lit) = count_val {
            assert_eq!(lit.value(), "3");
        } else {
            panic!("Expected literal count value");
        }
    }

    // Test COUNT(?var) - count specific variable
    let query_count_var = r#"
        SELECT (COUNT(?age) AS ?ageCount) WHERE {
            ?person <http://example.org/name> ?name .
            OPTIONAL { ?person <http://example.org/age> ?age }
        }
    "#;

    let results_var = store.query(query_count_var).unwrap();
    assert_eq!(results_var.len(), 1);

    // Verify count is 2 (only Alice and Bob have ages, Charlie doesn't)
    if let QueryResults::Bindings(bindings) = results_var.results() {
        let count_val = bindings[0].get("ageCount").unwrap();
        if let Term::Literal(lit) = count_val {
            assert_eq!(lit.value(), "2");
        } else {
            panic!("Expected literal count value");
        }
    }
}

#[test]
fn test_sparql_bind() {
    #[allow(unused_mut)]
    let mut store = RdfStore::new().unwrap();

    // Insert test data: Alice (age 30) and Bob (age 25)
    let alice = NamedNode::new("http://example.org/Alice").unwrap();
    let bob = NamedNode::new("http://example.org/Bob").unwrap();
    let age_pred = NamedNode::new("http://example.org/age").unwrap();

    store
        .insert_triple(Triple::new(
            alice.clone(),
            age_pred.clone(),
            Literal::new("30"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            bob.clone(),
            age_pred.clone(),
            Literal::new("25"),
        ))
        .unwrap();

    // Test BIND with arithmetic - convert age to months
    let query = r#"
        SELECT ?person ?age ?ageInMonths WHERE {
            ?person <http://example.org/age> ?age .
            BIND (?age * 12 AS ?ageInMonths)
        }
    "#;

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 2, "Should return 2 people");
    assert_eq!(results.variables(), &["person", "age", "ageInMonths"]);

    // Verify Alice's age in months (30 * 12 = 360)
    if let QueryResults::Bindings(bindings) = results.results() {
        let alice_binding = bindings
            .iter()
            .find(|b| {
                if let Some(Term::NamedNode(node)) = b.get("person") {
                    node.to_string().contains("Alice")
                } else {
                    false
                }
            })
            .expect("Should find Alice");

        let age_in_months = alice_binding.get("ageInMonths").unwrap();
        if let Term::Literal(lit) = age_in_months {
            assert_eq!(lit.value(), "360");
        } else {
            panic!("Expected literal value for ageInMonths");
        }
    }

    // Test BIND with addition
    let query_add = r#"
        SELECT ?person ?age ?agePlusTen WHERE {
            ?person <http://example.org/age> ?age .
            BIND (?age + 10 AS ?agePlusTen)
        }
    "#;

    let results_add = store.query(query_add).unwrap();
    assert_eq!(results_add.len(), 2);

    // Test BIND with division
    let query_div = r#"
        SELECT ?person ?age ?ageHalf WHERE {
            ?person <http://example.org/age> ?age .
            BIND (?age / 2 AS ?ageHalf)
        }
    "#;

    let results_div = store.query(query_div).unwrap();
    assert_eq!(results_div.len(), 2);

    // Verify Bob's half age (25 / 2 = 12.5)
    if let QueryResults::Bindings(bindings) = results_div.results() {
        let bob_binding = bindings
            .iter()
            .find(|b| {
                if let Some(Term::NamedNode(node)) = b.get("person") {
                    node.to_string().contains("Bob")
                } else {
                    false
                }
            })
            .expect("Should find Bob");

        let age_half = bob_binding.get("ageHalf").unwrap();
        if let Term::Literal(lit) = age_half {
            assert_eq!(lit.value(), "12.5");
        } else {
            panic!("Expected literal value for ageHalf");
        }
    }
}

#[test]
fn test_sparql_values() {
    #[allow(unused_mut)]
    let mut store = RdfStore::new().unwrap();

    // Insert test data
    let name_pred = NamedNode::new("http://example.org/name").unwrap();
    let age_pred = NamedNode::new("http://example.org/age").unwrap();

    let alice = NamedNode::new("http://example.org/Alice").unwrap();
    let bob = NamedNode::new("http://example.org/Bob").unwrap();
    let charlie = NamedNode::new("http://example.org/Charlie").unwrap();

    store
        .insert_triple(Triple::new(
            alice.clone(),
            name_pred.clone(),
            Literal::new("Alice"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            alice.clone(),
            age_pred.clone(),
            Literal::new("30"),
        ))
        .unwrap();

    store
        .insert_triple(Triple::new(
            bob.clone(),
            name_pred.clone(),
            Literal::new("Bob"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            bob.clone(),
            age_pred.clone(),
            Literal::new("25"),
        ))
        .unwrap();

    store
        .insert_triple(Triple::new(
            charlie.clone(),
            name_pred.clone(),
            Literal::new("Charlie"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            charlie.clone(),
            age_pred.clone(),
            Literal::new("35"),
        ))
        .unwrap();

    // Test VALUES with inline data
    let query = r#"
        SELECT ?name ?targetAge WHERE {
            VALUES (?name ?targetAge) {
                ("Alice" "30")
                ("Charlie" "35")
            }
            ?person <http://example.org/name> ?name .
            ?person <http://example.org/age> ?targetAge .
        }
    "#;

    let results = store.query(query).unwrap();
    assert_eq!(results.len(), 2); // Alice and Charlie match

    if let QueryResults::Bindings(bindings) = results.results() {
        // Check that we got Alice and Charlie
        let names: Vec<String> = bindings
            .iter()
            .filter_map(|b| b.get("name"))
            .filter_map(|term| {
                if let Term::Literal(lit) = term {
                    Some(lit.value().to_string())
                } else {
                    None
                }
            })
            .collect();

        assert!(names.contains(&"Alice".to_string()));
        assert!(names.contains(&"Charlie".to_string()));
        assert!(!names.contains(&"Bob".to_string())); // Bob (25) doesn't match
    } else {
        panic!("Expected bindings");
    }

    // Test VALUES without patterns (VALUES only)
    let query_values_only = r#"
        SELECT ?x ?y WHERE {
            VALUES (?x ?y) {
                ("a" "1")
                ("b" "2")
            }
        }
    "#;

    let results_only = store.query(query_values_only).unwrap();
    assert_eq!(results_only.len(), 2);
}

#[test]
fn test_sparql_string_functions() {
    #[allow(unused_mut)]
    let mut store = RdfStore::new().unwrap();

    // Insert test data with names
    let alice = NamedNode::new("http://example.org/Alice").unwrap();
    let bob = NamedNode::new("http://example.org/Bob").unwrap();
    let first_name_pred = NamedNode::new("http://example.org/firstName").unwrap();
    let last_name_pred = NamedNode::new("http://example.org/lastName").unwrap();

    store
        .insert_triple(Triple::new(
            alice.clone(),
            first_name_pred.clone(),
            Literal::new("Alice"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            alice.clone(),
            last_name_pred.clone(),
            Literal::new("Smith"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            bob.clone(),
            first_name_pred.clone(),
            Literal::new("Bob"),
        ))
        .unwrap();
    store
        .insert_triple(Triple::new(
            bob.clone(),
            last_name_pred.clone(),
            Literal::new("Jones"),
        ))
        .unwrap();

    // Test CONCAT - simple concatenation with literal
    let query_concat = r#"
        SELECT ?person ?greeting WHERE {
            ?person <http://example.org/firstName> ?name .
            BIND (CONCAT("Hello, ", ?name) AS ?greeting)
        }
    "#;

    let results_concat = store.query(query_concat).unwrap();
    assert_eq!(results_concat.len(), 2);

    if let QueryResults::Bindings(bindings) = results_concat.results() {
        let alice_binding = bindings
            .iter()
            .find(|b| {
                if let Some(Term::Literal(lit)) = b.get("name") {
                    lit.value() == "Alice"
                } else {
                    false
                }
            })
            .expect("Should find Alice");

        let greeting = alice_binding.get("greeting").unwrap();
        if let Term::Literal(lit) = greeting {
            assert_eq!(lit.value(), "Hello, Alice");
        }
    }

    // Test UCASE and LCASE
    let query_case = r#"
        SELECT ?person ?upper ?lower WHERE {
            ?person <http://example.org/firstName> ?name .
            BIND (UCASE(?name) AS ?upper)
            BIND (LCASE(?name) AS ?lower)
        }
    "#;

    let results_case = store.query(query_case).unwrap();
    assert_eq!(results_case.len(), 2);

    if let QueryResults::Bindings(bindings) = results_case.results() {
        let bob_binding = bindings
            .iter()
            .find(|b| {
                if let Some(Term::NamedNode(node)) = b.get("person") {
                    node.to_string().contains("Bob")
                } else {
                    false
                }
            })
            .expect("Should find Bob");

        let upper = bob_binding.get("upper").unwrap();
        let lower = bob_binding.get("lower").unwrap();

        if let Term::Literal(lit) = upper {
            assert_eq!(lit.value(), "BOB");
        }
        if let Term::Literal(lit) = lower {
            assert_eq!(lit.value(), "bob");
        }
    }

    // Test STRLEN
    let query_strlen = r#"
        SELECT ?person ?name ?nameLength WHERE {
            ?person <http://example.org/firstName> ?name .
            BIND (STRLEN(?name) AS ?nameLength)
        }
    "#;

    let results_strlen = store.query(query_strlen).unwrap();
    assert_eq!(results_strlen.len(), 2);

    if let QueryResults::Bindings(bindings) = results_strlen.results() {
        let alice_binding = bindings
            .iter()
            .find(|b| {
                if let Some(Term::Literal(lit)) = b.get("name") {
                    lit.value() == "Alice"
                } else {
                    false
                }
            })
            .expect("Should find Alice");

        let name_length = alice_binding.get("nameLength").unwrap();
        if let Term::Literal(lit) = name_length {
            assert_eq!(lit.value(), "5"); // "Alice" has 5 characters
        }
    }

    // Test SUBSTR
    let query_substr = r#"
        SELECT ?person ?name ?firstThree WHERE {
            ?person <http://example.org/firstName> ?name .
            BIND (SUBSTR(?name, 1, 3) AS ?firstThree)
        }
    "#;

    let results_substr = store.query(query_substr).unwrap();
    assert_eq!(results_substr.len(), 2);

    if let QueryResults::Bindings(bindings) = results_substr.results() {
        let alice_binding = bindings
            .iter()
            .find(|b| {
                if let Some(Term::Literal(lit)) = b.get("name") {
                    lit.value() == "Alice"
                } else {
                    false
                }
            })
            .expect("Should find Alice");

        let first_three = alice_binding.get("firstThree").unwrap();
        if let Term::Literal(lit) = first_three {
            assert_eq!(lit.value(), "Ali");
        }
    }

    // Test CONTAINS
    let query_contains = r#"
        SELECT ?person ?hasLowerI WHERE {
            ?person <http://example.org/firstName> ?name .
            BIND (CONTAINS(?name, "i") AS ?hasLowerI)
        }
    "#;

    let results_contains = store.query(query_contains).unwrap();
    assert_eq!(results_contains.len(), 2);

    // Test STRSTARTS
    let query_starts = r#"
        SELECT ?person ?startsWithA WHERE {
            ?person <http://example.org/firstName> ?name .
            BIND (STRSTARTS(?name, "A") AS ?startsWithA)
        }
    "#;

    let results_starts = store.query(query_starts).unwrap();
    assert_eq!(results_starts.len(), 2);

    if let QueryResults::Bindings(bindings) = results_starts.results() {
        let alice_binding = bindings
            .iter()
            .find(|b| {
                if let Some(Term::Literal(lit)) = b.get("name") {
                    lit.value() == "Alice"
                } else {
                    false
                }
            })
            .expect("Should find Alice");

        let starts_with_a = alice_binding.get("startsWithA").unwrap();
        if let Term::Literal(lit) = starts_with_a {
            assert_eq!(lit.value(), "true");
        }
    }
}
