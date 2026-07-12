// Tests for the SPARQL query module (extracted for file size compliance)
use super::*;

fn make_store_basic() -> OxiRSStore {
    let mut store = OxiRSStore::new();
    store.insert("http://a", "http://b", "http://c");
    store
}

// ---- Original tests (backward-compatible) ----

#[test]
fn test_parse_select() {
    let sparql = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
    let query = parse_select_query(sparql).expect("parse");
    assert_eq!(query.variables.len(), 3);
    assert_eq!(query.patterns.len(), 1);
}

#[test]
fn test_evaluate_select() {
    let mut store = OxiRSStore::new();
    store.insert("http://a", "http://b", "http://c");
    let sparql = "SELECT ?s ?o WHERE { ?s <http://b> ?o }";
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("s").expect("s"), "http://a");
    assert_eq!(results[0].get("o").expect("o"), "http://c");
}

#[test]
fn test_evaluate_ask() {
    let store = make_store_basic();
    assert!(execute_ask("ASK { <http://a> <http://b> <http://c> }", &store).expect("ask true"));
    assert!(
        !execute_ask("ASK { <http://x> <http://y> <http://z> }", &store).expect("ask false")
    );
}

// ---- FILTER tests ----

#[test]
fn test_filter_lang() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://name", "\"Alice\"@en");
    store.insert("http://bob", "http://name", "\"Bob\"@fr");
    let sparql = r#"SELECT ?s WHERE { ?s <http://name> ?n . FILTER(LANG(?n) = "en") }"#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("s").expect("s"), "http://alice");
}

#[test]
fn test_filter_greater_than() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://age", "\"30\"");
    store.insert("http://bob", "http://age", "\"16\"");
    let sparql = r#"SELECT ?s WHERE { ?s <http://age> ?age . FILTER(?age > 18) }"#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("s").expect("s"), "http://alice");
}

#[test]
fn test_filter_regex() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://name", "\"Alice\"");
    store.insert("http://bob", "http://name", "\"Bob\"");
    let sparql =
        r#"SELECT ?s WHERE { ?s <http://name> ?name . FILTER(regex(?name, "^Alice")) }"#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("s").expect("s"), "http://alice");
}

// ---- OPTIONAL tests ----

#[test]
fn test_optional_with_match() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://knows", "http://bob");
    store.insert("http://alice", "http://name", "\"Alice\"");
    let sparql = r#"
        SELECT ?s ?name
        WHERE {
            ?s <http://knows> <http://bob>
            OPTIONAL { ?s <http://name> ?name }
        }
    "#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("name").expect("name"), "\"Alice\"");
}

#[test]
fn test_optional_without_match() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://knows", "http://bob");
    let sparql = r#"
        SELECT ?s ?name
        WHERE {
            ?s <http://knows> <http://bob>
            OPTIONAL { ?s <http://name> ?name }
        }
    "#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert!(!results[0].contains_key("name"));
    assert_eq!(results[0].get("s").expect("s"), "http://alice");
}

#[test]
fn test_optional_multiple_subjects() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://knows", "http://bob");
    store.insert("http://carol", "http://knows", "http://bob");
    store.insert("http://alice", "http://age", "\"30\"");
    let sparql = r#"
        SELECT ?s ?age
        WHERE {
            ?s <http://knows> <http://bob>
            OPTIONAL { ?s <http://age> ?age }
        }
    "#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 2);
    let alice_row = results
        .iter()
        .find(|r| r.get("s").map(|v| v.contains("alice")).unwrap_or(false));
    assert!(alice_row.is_some());
    assert!(alice_row.expect("should succeed").contains_key("age"));
    let carol_row = results
        .iter()
        .find(|r| r.get("s").map(|v| v.contains("carol")).unwrap_or(false));
    assert!(carol_row.is_some());
    assert!(!carol_row.expect("should succeed").contains_key("age"));
}

#[test]
fn test_optional_nested() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://knows", "http://bob");
    store.insert("http://alice", "http://name", "\"Alice\"");
    store.insert("http://alice", "http://age", "\"30\"");
    let sparql = r#"
        SELECT ?s ?name ?age
        WHERE {
            ?s <http://knows> <http://bob>
            OPTIONAL { ?s <http://name> ?name }
            OPTIONAL { ?s <http://age> ?age }
        }
    "#;
    let results = execute_select(sparql, &store).expect("execute");
    assert!(!results.is_empty());
    assert!(results[0].contains_key("name"));
    assert!(results[0].contains_key("age"));
}

// ---- UNION tests ----

#[test]
fn test_union_basic() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://type", "http://Person");
    store.insert("http://company", "http://type", "http://Organization");
    let sparql = r#"
        SELECT ?s
        WHERE {
            { ?s <http://type> <http://Person> }
            UNION
            { ?s <http://type> <http://Organization> }
        }
    "#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 2);
}

#[test]
fn test_union_no_duplicate_suppression() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://type", "http://Person");
    let sparql = r#"
        SELECT ?s
        WHERE {
            { ?s <http://type> <http://Person> }
            UNION
            { ?s <http://type> <http://Person> }
        }
    "#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 2);
}

#[test]
fn test_union_disjoint_sets() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://type", "http://A");
    store.insert("http://bob", "http://type", "http://B");
    let sparql = r#"
        SELECT ?s
        WHERE {
            { ?s <http://type> <http://A> }
            UNION
            { ?s <http://type> <http://B> }
        }
    "#;
    let results = execute_select(sparql, &store).expect("execute");
    let mut subjects: Vec<String> =
        results.iter().filter_map(|r| r.get("s").cloned()).collect();
    subjects.sort();
    assert_eq!(subjects, vec!["http://alice", "http://bob"]);
}

#[test]
fn test_union_empty_branch() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://type", "http://A");
    let sparql = r#"
        SELECT ?s
        WHERE {
            { ?s <http://type> <http://A> }
            UNION
            { ?s <http://type> <http://C> }
        }
    "#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
}

// ---- FILTER NOT EXISTS ----

#[test]
fn test_filter_not_exists_basic() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://knows", "http://bob");
    store.insert("http://alice", "http://name", "\"Alice\"");
    store.insert("http://carol", "http://knows", "http://dave");
    // Alice has a name, Carol doesn't
    let sparql = r#"
        SELECT ?s
        WHERE {
            ?s <http://knows> ?o
            FILTER NOT EXISTS { ?s <http://name> ?name }
        }
    "#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert!(results[0]
        .get("s")
        .map(|v| v.contains("carol"))
        .unwrap_or(false));
}

#[test]
fn test_filter_exists_basic() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://knows", "http://bob");
    store.insert("http://alice", "http://name", "\"Alice\"");
    store.insert("http://carol", "http://knows", "http://dave");
    // Alice has a name, Carol doesn't
    let sparql = r#"
        SELECT ?s
        WHERE {
            ?s <http://knows> ?o
            FILTER EXISTS { ?s <http://name> ?name }
        }
    "#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert!(results[0]
        .get("s")
        .map(|v| v.contains("alice"))
        .unwrap_or(false));
}

// ---- FILTER built-ins ----

#[test]
fn test_filter_bound() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://knows", "http://bob");
    store.insert("http://alice", "http://name", "\"Alice\"");
    store.insert("http://carol", "http://knows", "http://dave");
    let sparql = r#"
        SELECT ?s
        WHERE {
            ?s <http://knows> ?o
            OPTIONAL { ?s <http://name> ?name }
            FILTER(BOUND(?name))
        }
    "#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert!(results[0]
        .get("s")
        .map(|v| v.contains("alice"))
        .unwrap_or(false));
}

#[test]
fn test_filter_isiri() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://knows", "http://bob");
    store.insert("http://alice", "http://name", "\"Alice\"");
    let sparql = r#"SELECT ?o WHERE { <http://alice> ?p ?o . FILTER(isIRI(?o)) }"#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert!(results[0]
        .get("o")
        .map(|v| v.contains("bob"))
        .unwrap_or(false));
}

#[test]
fn test_filter_isliteral() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://knows", "http://bob");
    store.insert("http://alice", "http://name", "\"Alice\"");
    let sparql = r#"SELECT ?o WHERE { <http://alice> ?p ?o . FILTER(isLiteral(?o)) }"#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert!(results[0]
        .get("o")
        .map(|v| v.contains("Alice"))
        .unwrap_or(false));
}

#[test]
fn test_filter_logical_and() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://age", "\"30\"");
    store.insert("http://bob", "http://age", "\"16\"");
    store.insert("http://carol", "http://age", "\"25\"");
    let sparql = r#"SELECT ?s WHERE { ?s <http://age> ?age . FILTER(?age > 18 && ?age < 28) }"#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert!(results[0]
        .get("s")
        .map(|v| v.contains("carol"))
        .unwrap_or(false));
}

#[test]
fn test_filter_strstarts() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://name", "\"Alice\"");
    store.insert("http://bob", "http://name", "\"Bob\"");
    let sparql =
        r#"SELECT ?s WHERE { ?s <http://name> ?name . FILTER(STRSTARTS(?name, "Al")) }"#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert!(results[0]
        .get("s")
        .map(|v| v.contains("alice"))
        .unwrap_or(false));
}

#[test]
fn test_filter_contains() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://name", "\"Alice Smith\"");
    store.insert("http://bob", "http://name", "\"Bob Jones\"");
    let sparql =
        r#"SELECT ?s WHERE { ?s <http://name> ?name . FILTER(CONTAINS(?name, "Smith")) }"#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
    assert!(results[0]
        .get("s")
        .map(|v| v.contains("alice"))
        .unwrap_or(false));
}

#[test]
fn test_filter_regex_case_insensitive() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://name", "\"Alice\"");
    store.insert("http://bob", "http://name", "\"Bob\"");
    let sparql =
        r#"SELECT ?s WHERE { ?s <http://name> ?name . FILTER(regex(?name, "^alice", "i")) }"#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
}

// ---- Property path tests ----

#[test]
fn test_property_path_sequence_query() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://knows", "http://bob");
    store.insert("http://bob", "http://knows", "http://carol");
    let path = property_path::PropertyPath::Sequence(
        Box::new(property_path::PropertyPath::Iri("http://knows".into())),
        Box::new(property_path::PropertyPath::Iri("http://knows".into())),
    );
    let results = path.evaluate("http://alice", &store);
    assert_eq!(results, vec!["http://carol"]);
}

#[test]
fn test_property_path_transitive_closure() {
    let mut store = OxiRSStore::new();
    store.insert("http://a", "http://p", "http://b");
    store.insert("http://b", "http://p", "http://c");
    store.insert("http://c", "http://p", "http://d");
    let path = property_path::PropertyPath::OneOrMore(Box::new(
        property_path::PropertyPath::Iri("http://p".into()),
    ));
    let mut results = path.evaluate("http://a", &store);
    results.sort();
    assert!(results.contains(&"http://b".to_string()));
    assert!(results.contains(&"http://c".to_string()));
    assert!(results.contains(&"http://d".to_string()));
}

#[test]
fn test_property_path_zero_or_more_includes_self() {
    let mut store = OxiRSStore::new();
    store.insert("http://a", "http://p", "http://b");
    let path = property_path::PropertyPath::ZeroOrMore(Box::new(
        property_path::PropertyPath::Iri("http://p".into()),
    ));
    let results = path.evaluate("http://a", &store);
    assert!(results.contains(&"http://a".to_string()));
    assert!(results.contains(&"http://b".to_string()));
}

#[test]
fn test_property_path_alternative() {
    let mut store = OxiRSStore::new();
    store.insert("http://a", "http://p1", "http://b");
    store.insert("http://a", "http://p2", "http://c");
    let path = property_path::PropertyPath::Alternative(
        Box::new(property_path::PropertyPath::Iri("http://p1".into())),
        Box::new(property_path::PropertyPath::Iri("http://p2".into())),
    );
    let mut results = path.evaluate("http://a", &store);
    results.sort();
    assert_eq!(results, vec!["http://b", "http://c"]);
}

// ---- JSON-LD tests ----

#[test]
fn test_jsonld_serialize_basic() {
    let mut store = OxiRSStore::new();
    store.insert(
        "http://example.org/alice",
        "http://example.org/name",
        "\"Alice\"",
    );
    let output = jsonld::serialize_jsonld(&store);
    assert!(output.contains("@context"));
    assert!(output.contains("@graph"));
    assert!(output.contains("alice"));
}

#[test]
fn test_jsonld_lang_tagged_literal() {
    let mut store = OxiRSStore::new();
    store.insert("http://s", "http://p", "\"hello\"@en");
    let output = jsonld::serialize_jsonld(&store);
    assert!(output.contains("@language"));
}

#[test]
fn test_jsonld_typed_literal() {
    let mut store = OxiRSStore::new();
    store.insert(
        "http://s",
        "http://p",
        "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>",
    );
    let output = jsonld::serialize_jsonld(&store);
    assert!(output.contains("xsd:integer"));
}

// ---- DISTINCT ----

#[test]
fn test_distinct() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://knows", "http://bob");
    store.insert("http://alice", "http://likes", "http://carol");
    // Without DISTINCT, would return 2 rows for alice
    let sparql = "SELECT DISTINCT ?s WHERE { ?s ?p ?o }";
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 1);
}

// ---- ORDER BY ----

#[test]
fn test_order_by_asc() {
    let mut store = OxiRSStore::new();
    store.insert("http://c", "http://p", "\"c\"");
    store.insert("http://a", "http://p", "\"a\"");
    store.insert("http://b", "http://p", "\"b\"");
    let sparql = "SELECT ?s ?v WHERE { ?s <http://p> ?v } ORDER BY ?v";
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 3);
    let vals: Vec<_> = results
        .iter()
        .map(|r| r.get("v").map(|s| s.as_str()).unwrap_or(""))
        .collect();
    assert!(vals[0] <= vals[1]);
    assert!(vals[1] <= vals[2]);
}

// ---- VALUES ----

#[test]
fn test_values_single_var() {
    let mut store = OxiRSStore::new();
    store.insert("http://alice", "http://name", "\"Alice\"");
    store.insert("http://bob", "http://name", "\"Bob\"");
    store.insert("http://carol", "http://name", "\"Carol\"");
    let sparql = r#"
        SELECT ?s ?name
        WHERE {
            VALUES ?s { <http://alice> <http://bob> }
            ?s <http://name> ?name
        }
    "#;
    let results = execute_select(sparql, &store).expect("execute");
    assert_eq!(results.len(), 2);
}

// ---- LIMIT / OFFSET ----

#[test]
fn test_limit() {
    let mut store = OxiRSStore::new();
    for i in 0..5 {
        store.insert(&format!("http://s{}", i), "http://p", "http://o");
    }
    let results = execute_select(
        "SELECT ?s WHERE { ?s <http://p> <http://o> } LIMIT 3",
        &store,
    )
    .expect("execute");
    assert_eq!(results.len(), 3);
}

#[test]
fn test_offset() {
    let mut store = OxiRSStore::new();
    for i in 0..5 {
        store.insert(&format!("http://s{}", i), "http://p", "http://o");
    }
    let results = execute_select(
        "SELECT ?s WHERE { ?s <http://p> <http://o> } OFFSET 3",
        &store,
    )
    .expect("execute");
    assert_eq!(results.len(), 2);
}

// ---- CONSTRUCT ----

#[test]
fn test_construct() {
    let mut store = OxiRSStore::new();
    store.insert("http://a", "http://b", "http://c");
    let sparql = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
    let triples = execute_construct(sparql, &store).expect("construct");
    assert_eq!(triples.len(), 1);
}

// ---- Parsed (N-Triples) data: terms carry their N-Triples serialisation ----
//
// `loadNTriples` stores terms as `<iri>` / `"lit"@lang`, while a query writes
// them as `<iri>` and `OxiRSStore::insert` accepts a bare `iri`. Matching has
// to bridge the two forms, otherwise every pattern with a concrete IRI in it
// silently returns zero rows.

fn make_store_from_ntriples() -> OxiRSStore {
    let mut store = OxiRSStore::new();
    let nt = concat!(
        "<http://ex/Gemini> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ex/Sign> .\n",
        "<http://ex/Gemini> <http://ex/element> <http://ex/air> .\n",
        "<http://ex/Gemini> <http://ex/modality> <http://ex/mutable> .\n",
        "<http://ex/Gemini> <http://ex/label> \"Gemini\"@en .\n",
        "<http://ex/Libra> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ex/Sign> .\n",
        "<http://ex/Libra> <http://ex/element> <http://ex/air> .\n",
        "<http://ex/Libra> <http://ex/modality> <http://ex/cardinal> .\n",
    );
    store.load_ntriples(nt).expect("N-Triples load");
    store
}

#[test]
fn test_ntriples_concrete_object_matches() {
    let store = make_store_from_ntriples();
    let results = execute_select(
        "SELECT ?s WHERE { ?s <http://ex/element> <http://ex/air> }",
        &store,
    )
    .expect("execute");
    assert_eq!(results.len(), 2);
}

#[test]
fn test_ntriples_join_on_two_concrete_objects() {
    let store = make_store_from_ntriples();
    let results = execute_select(
        "SELECT ?s WHERE { ?s <http://ex/element> <http://ex/air> . \
         ?s <http://ex/modality> <http://ex/mutable> }",
        &store,
    )
    .expect("execute");
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get("s").expect("s"),
        "<http://ex/Gemini>",
        "bindings keep the store's N-Triples form"
    );
}

#[test]
fn test_ntriples_type_shorthand_a() {
    let store = make_store_from_ntriples();
    let via_a = execute_select("SELECT ?s WHERE { ?s a <http://ex/Sign> }", &store).expect("a");
    let via_iri = execute_select(
        "SELECT ?s WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://ex/Sign> }",
        &store,
    )
    .expect("rdf:type");
    assert_eq!(via_a.len(), 2);
    assert_eq!(via_a.len(), via_iri.len());
}

#[test]
fn test_ntriples_fully_bound_pattern() {
    let store = make_store_from_ntriples();
    assert!(
        execute_ask(
            "ASK { <http://ex/Gemini> <http://ex/element> <http://ex/air> }",
            &store
        )
        .expect("ask"),
        "a pattern with no variables must still match"
    );
    assert!(!execute_ask(
        "ASK { <http://ex/Gemini> <http://ex/element> <http://ex/fire> }",
        &store
    )
    .expect("ask"));
}

#[test]
fn test_ntriples_literal_keeps_language_tag() {
    let store = make_store_from_ntriples();
    let results = execute_select(
        "SELECT ?l WHERE { <http://ex/Gemini> <http://ex/label> ?l }",
        &store,
    )
    .expect("execute");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("l").expect("l"), "\"Gemini\"@en");
}

#[test]
fn test_construct_instantiates_its_template() {
    let store = make_store_from_ntriples();
    let sparql = concat!(
        "PREFIX ex: <http://ex/>\n",
        "CONSTRUCT { ?s ex:hasElement ?e }\n",
        "WHERE { ?s ex:element ?e }"
    );
    let triples = execute_construct(sparql, &store).expect("construct");
    assert_eq!(triples.len(), 2, "one triple per solution");
    for triple in &triples {
        // The CONSTRUCT subsystem emits template IRIs bare; the serializers
        // wrap them. Compare on the IRI body so either form is accepted.
        assert_eq!(iri_body(&triple.predicate()), "http://ex/hasElement");
        assert_eq!(iri_body(&triple.object()), "http://ex/air");
    }
}

// ---- Solution budget ----

#[test]
fn test_solution_budget_stops_a_broad_join() {
    let mut store = OxiRSStore::new();
    // Two stars of 40 subjects each: joined without a shared variable they
    // would produce 1600 intermediate solutions.
    for i in 0..40 {
        store.insert(&format!("http://s{i}"), "http://p", &format!("http://o{i}"));
    }
    store.set_solution_budget(500);

    let cross = "SELECT ?a ?b WHERE { ?a <http://p> ?x . ?b <http://p> ?y }";
    let err = execute_select(cross, &store).expect_err("the budget must stop this join");
    assert!(err.to_string().contains("solution budget"), "{err}");

    // A join that stays inside the budget is unaffected.
    let narrow = "SELECT ?a WHERE { ?a <http://p> <http://o7> }";
    assert_eq!(execute_select(narrow, &store).expect("narrow").len(), 1);
}

#[test]
fn test_without_a_budget_a_broad_join_still_runs() {
    let mut store = OxiRSStore::new();
    for i in 0..20 {
        store.insert(&format!("http://s{i}"), "http://p", &format!("http://o{i}"));
    }
    let cross = "SELECT ?a ?b WHERE { ?a <http://p> ?x . ?b <http://p> ?y }";
    assert_eq!(execute_select(cross, &store).expect("no budget set").len(), 400);
}
