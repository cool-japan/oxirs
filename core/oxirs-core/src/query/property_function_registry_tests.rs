use super::*;
use crate::model::{Literal, NamedNode};

fn make_nn_term(iri: &str) -> Term {
    Term::NamedNode(NamedNode::new_unchecked(iri))
}

fn make_lit_term(value: &str) -> Term {
    Term::Literal(Literal::new(value))
}

fn make_int_term(value: i64) -> Term {
    make_integer_term(value)
}

// --- Registry tests ---

#[test]
fn test_registry_creation_with_builtins() {
    let registry = PropertyFunctionRegistry::new();
    assert!(!registry.is_empty());
    assert!(
        registry.len() >= 9,
        "Expected at least 9 built-in property functions, got {}",
        registry.len()
    );
}

#[test]
fn test_registry_is_property_function() {
    let registry = PropertyFunctionRegistry::new();
    assert!(registry.is_property_function("http://jena.apache.org/ARQ/list#member"));
    assert!(registry.is_property_function("http://jena.apache.org/ARQ/list#index"));
    assert!(registry.is_property_function("http://jena.apache.org/ARQ/list#length"));
    assert!(registry.is_property_function("http://jena.apache.org/ARQ/property#splitIRI"));
    assert!(registry.is_property_function("http://jena.apache.org/text#search"));
    assert!(!registry.is_property_function("http://example.org/unknown"));
}

#[test]
fn test_registry_lookup() {
    let mut registry = PropertyFunctionRegistry::new();
    let func = registry.lookup("http://jena.apache.org/ARQ/list#member");
    assert!(func.is_some());

    let func = registry.lookup("http://example.org/nonexistent");
    assert!(func.is_none());
}

#[test]
fn test_registry_register_custom() {
    let mut registry = PropertyFunctionRegistry::empty();
    assert!(registry.is_empty());

    registry.register("http://example.org/custom", Arc::new(ListMemberPF::new()));
    assert_eq!(registry.len(), 1);
    assert!(registry.is_property_function("http://example.org/custom"));
}

#[test]
fn test_registry_unregister() {
    let mut registry = PropertyFunctionRegistry::new();
    let initial_len = registry.len();

    let removed = registry.unregister("http://jena.apache.org/ARQ/list#member");
    assert!(removed);
    assert_eq!(registry.len(), initial_len - 1);

    let removed = registry.unregister("http://example.org/nonexistent");
    assert!(!removed);
}

#[test]
fn test_registry_registered_iris() {
    let registry = PropertyFunctionRegistry::new();
    let iris = registry.registered_iris();
    assert!(!iris.is_empty());
    assert!(iris.contains(&"http://jena.apache.org/ARQ/list#member".to_string()));
}

#[test]
fn test_registry_all_metadata() {
    let registry = PropertyFunctionRegistry::new();
    let metadata = registry.all_metadata();
    assert!(!metadata.is_empty());

    for m in &metadata {
        assert!(!m.iri.is_empty());
        assert!(!m.name.is_empty());
        assert!(!m.description.is_empty());
    }
}

#[test]
fn test_registry_statistics() {
    let mut registry = PropertyFunctionRegistry::new();
    let subj = PropertyFunctionArg::List(vec![
        PropertyFunctionArg::Term(make_lit_term("a")),
        PropertyFunctionArg::Term(make_lit_term("b")),
    ]);
    let obj = PropertyFunctionArg::Variable("item".to_string());

    let _ = registry.evaluate("http://jena.apache.org/ARQ/list#member", &subj, &obj);

    let stats = registry.statistics();
    assert_eq!(stats.total_evaluations, 1);
    assert!(stats.total_rows_produced >= 2);
    assert_eq!(stats.total_errors, 0);
}

#[test]
fn test_registry_reset_statistics() {
    let mut registry = PropertyFunctionRegistry::new();
    let subj = PropertyFunctionArg::Term(make_lit_term("test"));
    let obj = PropertyFunctionArg::Variable("x".to_string());

    let _ = registry.evaluate("http://jena.apache.org/ARQ/list#member", &subj, &obj);
    assert!(registry.statistics().total_evaluations > 0);

    registry.reset_statistics();
    assert_eq!(registry.statistics().total_evaluations, 0);
}

#[test]
fn test_registry_evaluate_unknown_function() {
    let mut registry = PropertyFunctionRegistry::new();
    let subj = PropertyFunctionArg::Variable("s".to_string());
    let obj = PropertyFunctionArg::Variable("o".to_string());

    let result = registry.evaluate("http://example.org/unknown", &subj, &obj);
    assert!(result.is_err());
}

#[test]
fn test_empty_registry() {
    let registry = PropertyFunctionRegistry::empty();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
}

// --- PropertyFunctionArg tests ---

#[test]
fn test_arg_display_term() {
    let arg = PropertyFunctionArg::Term(make_lit_term("hello"));
    let s = format!("{arg}");
    assert!(s.contains("hello"));
}

#[test]
fn test_arg_display_variable() {
    let arg = PropertyFunctionArg::Variable("x".to_string());
    assert_eq!(format!("{arg}"), "?x");
}

#[test]
fn test_arg_display_list() {
    let arg = PropertyFunctionArg::List(vec![
        PropertyFunctionArg::Variable("a".to_string()),
        PropertyFunctionArg::Variable("b".to_string()),
    ]);
    let s = format!("{arg}");
    assert!(s.starts_with('('));
    assert!(s.ends_with(')'));
    assert!(s.contains("?a"));
    assert!(s.contains("?b"));
}

// --- PropertyFunctionBinding tests ---

#[test]
fn test_binding_new_empty() {
    let binding = PropertyFunctionBinding::new();
    assert!(binding.is_empty());
    assert_eq!(binding.len(), 0);
}

#[test]
fn test_binding_with_values() {
    let binding = PropertyFunctionBinding::new()
        .bind("x", make_lit_term("hello"))
        .bind("y", make_int_term(42));
    assert_eq!(binding.len(), 2);
    assert!(binding.get("x").is_some());
    assert!(binding.get("y").is_some());
    assert!(binding.get("z").is_none());
}

#[test]
fn test_binding_bindings_map() {
    let binding = PropertyFunctionBinding::new().bind("a", make_lit_term("test"));
    let map = binding.bindings();
    assert_eq!(map.len(), 1);
    assert!(map.contains_key("a"));
}

// --- PropertyFunctionResult tests ---

#[test]
fn test_result_empty() {
    let result = PropertyFunctionResult::empty();
    assert!(result.is_empty());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_result_single() {
    let binding = PropertyFunctionBinding::new().bind("x", make_lit_term("val"));
    let result = PropertyFunctionResult::single(binding);
    assert_eq!(result.len(), 1);
    assert!(!result.is_empty());
}

#[test]
fn test_result_from_rows() {
    let rows = vec![
        PropertyFunctionBinding::new().bind("x", make_lit_term("a")),
        PropertyFunctionBinding::new().bind("x", make_lit_term("b")),
        PropertyFunctionBinding::new().bind("x", make_lit_term("c")),
    ];
    let result = PropertyFunctionResult::from_rows(rows);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_result_iter() {
    let rows = vec![
        PropertyFunctionBinding::new().bind("x", make_lit_term("a")),
        PropertyFunctionBinding::new().bind("x", make_lit_term("b")),
    ];
    let result = PropertyFunctionResult::from_rows(rows);
    let collected: Vec<_> = result.iter().collect();
    assert_eq!(collected.len(), 2);
}

// --- ListMemberPF tests ---

#[test]
fn test_list_member_with_list_subject() {
    let pf = ListMemberPF::new();
    let subject = PropertyFunctionArg::List(vec![
        PropertyFunctionArg::Term(make_lit_term("alpha")),
        PropertyFunctionArg::Term(make_lit_term("beta")),
        PropertyFunctionArg::Term(make_lit_term("gamma")),
    ]);
    let object = PropertyFunctionArg::Variable("item".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 3);

    // Check that indices are correct
    for (i, row) in result.iter().enumerate() {
        let idx = row.get("index").expect("index should be bound");
        assert_eq!(extract_string_value(idx), (i as i64).to_string());
    }
}

#[test]
fn test_list_member_with_single_term() {
    let pf = ListMemberPF::new();
    let subject = PropertyFunctionArg::Term(make_lit_term("single"));
    let object = PropertyFunctionArg::Variable("item".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_list_member_with_unbound_variable() {
    let pf = ListMemberPF::new();
    let subject = PropertyFunctionArg::Variable("list".to_string());
    let object = PropertyFunctionArg::Variable("item".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert!(result.is_empty());
}

#[test]
fn test_list_member_cardinality() {
    let pf = ListMemberPF::new();
    let subject_list = PropertyFunctionArg::List(vec![
        PropertyFunctionArg::Term(make_lit_term("a")),
        PropertyFunctionArg::Term(make_lit_term("b")),
    ]);
    let subject_term = PropertyFunctionArg::Term(make_lit_term("x"));
    let subject_var = PropertyFunctionArg::Variable("v".to_string());
    let obj = PropertyFunctionArg::Variable("o".to_string());

    assert_eq!(pf.estimated_cardinality(&subject_list, &obj), Some(2));
    assert_eq!(pf.estimated_cardinality(&subject_term, &obj), Some(1));
    assert_eq!(pf.estimated_cardinality(&subject_var, &obj), None);
}

#[test]
fn test_list_member_metadata() {
    let pf = ListMemberPF::new();
    let meta = pf.metadata();
    assert_eq!(meta.name, "list:member");
    assert_eq!(meta.category, "list");
    assert!(!meta.subject_must_be_bound);
}

// --- ListIndexPF tests ---

#[test]
fn test_list_index_with_list() {
    let pf = ListIndexPF::new();
    let subject = PropertyFunctionArg::List(vec![
        PropertyFunctionArg::Term(make_lit_term("first")),
        PropertyFunctionArg::Term(make_lit_term("second")),
    ]);
    let object = PropertyFunctionArg::Variable("idx_item".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 2);
}

#[test]
fn test_list_index_empty() {
    let pf = ListIndexPF::new();
    let subject = PropertyFunctionArg::Variable("list".to_string());
    let object = PropertyFunctionArg::Variable("idx".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert!(result.is_empty());
}

// --- ListLengthPF tests ---

#[test]
fn test_list_length_with_list() {
    let pf = ListLengthPF::new();
    let subject = PropertyFunctionArg::List(vec![
        PropertyFunctionArg::Term(make_lit_term("a")),
        PropertyFunctionArg::Term(make_lit_term("b")),
        PropertyFunctionArg::Term(make_lit_term("c")),
    ]);
    let object = PropertyFunctionArg::Variable("len".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);

    let row = &result.rows()[0];
    let len_term = row.get("length").expect("length should be bound");
    assert_eq!(extract_string_value(len_term), "3");
}

#[test]
fn test_list_length_with_single_term() {
    let pf = ListLengthPF::new();
    let subject = PropertyFunctionArg::Term(make_lit_term("single"));
    let object = PropertyFunctionArg::Variable("len".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);
    let row = &result.rows()[0];
    let len_term = row.get("length").expect("length should be bound");
    assert_eq!(extract_string_value(len_term), "1");
}

#[test]
fn test_list_length_with_variable_error() {
    let pf = ListLengthPF::new();
    let subject = PropertyFunctionArg::Variable("list".to_string());
    let object = PropertyFunctionArg::Variable("len".to_string());

    let result = pf.evaluate(&subject, &object);
    assert!(result.is_err());
}

#[test]
fn test_list_length_cardinality() {
    let pf = ListLengthPF::new();
    let subj = PropertyFunctionArg::Term(make_lit_term("x"));
    let obj = PropertyFunctionArg::Variable("len".to_string());
    assert_eq!(pf.estimated_cardinality(&subj, &obj), Some(1));
}

// --- SplitIriPF tests ---

#[test]
fn test_split_iri_with_hash() {
    let pf = SplitIriPF::new();
    let subject = PropertyFunctionArg::Term(make_nn_term("http://xmlns.com/foaf/0.1/#knows"));
    let object = PropertyFunctionArg::Variable("parts".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);

    let row = &result.rows()[0];
    let ns = row.get("namespace").expect("namespace should be bound");
    let ln = row.get("localname").expect("localname should be bound");
    assert_eq!(extract_string_value(ns), "http://xmlns.com/foaf/0.1/#");
    assert_eq!(extract_string_value(ln), "knows");
}

#[test]
fn test_split_iri_with_slash() {
    let pf = SplitIriPF::new();
    let subject = PropertyFunctionArg::Term(make_nn_term("http://example.org/resource/item42"));
    let object = PropertyFunctionArg::Variable("parts".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);

    let row = &result.rows()[0];
    let ns = row.get("namespace").expect("namespace should be bound");
    let ln = row.get("localname").expect("localname should be bound");
    assert_eq!(extract_string_value(ns), "http://example.org/resource/");
    assert_eq!(extract_string_value(ln), "item42");
}

#[test]
fn test_split_iri_with_variable_error() {
    let pf = SplitIriPF::new();
    let subject = PropertyFunctionArg::Variable("iri".to_string());
    let object = PropertyFunctionArg::Variable("parts".to_string());

    let result = pf.evaluate(&subject, &object);
    assert!(result.is_err());
}

#[test]
fn test_split_iri_with_list_subject() {
    let pf = SplitIriPF::new();
    let subject = PropertyFunctionArg::List(vec![PropertyFunctionArg::Term(make_nn_term(
        "http://example.org/ns#local",
    ))]);
    let object = PropertyFunctionArg::Variable("parts".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);
}

// --- LocalNamePF tests ---

#[test]
fn test_localname_extraction() {
    let pf = LocalNamePF::new();
    let subject = PropertyFunctionArg::Term(make_nn_term("http://example.org/ns#Person"));
    let object = PropertyFunctionArg::Variable("name".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);

    let row = &result.rows()[0];
    let ln = row.get("localname").expect("localname should be bound");
    assert_eq!(extract_string_value(ln), "Person");
}

#[test]
fn test_localname_metadata() {
    let pf = LocalNamePF::new();
    let meta = pf.metadata();
    assert_eq!(meta.name, "apf:localname");
    assert_eq!(meta.category, "string");
}

// --- NamespacePF tests ---

#[test]
fn test_namespace_extraction() {
    let pf = NamespacePF::new();
    let subject = PropertyFunctionArg::Term(make_nn_term("http://example.org/ns#Person"));
    let object = PropertyFunctionArg::Variable("ns".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);

    let row = &result.rows()[0];
    let ns = row.get("namespace").expect("namespace should be bound");
    assert_eq!(extract_string_value(ns), "http://example.org/ns#");
}

// --- TextSearchPF tests ---

#[test]
fn test_text_search_with_query_term() {
    let pf = TextSearchPF::new();
    let subject = PropertyFunctionArg::Variable("s".to_string());
    let object = PropertyFunctionArg::Term(make_lit_term("rust programming"));

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert!(!result.is_empty());

    let row = &result.rows()[0];
    let query = row.get("query").expect("query should be bound");
    assert_eq!(extract_string_value(query), "rust programming");
}

#[test]
fn test_text_search_with_list_args() {
    let pf = TextSearchPF::new();
    let subject = PropertyFunctionArg::Variable("s".to_string());
    let object = PropertyFunctionArg::List(vec![PropertyFunctionArg::Term(make_lit_term(
        "search query",
    ))]);

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert!(!result.is_empty());
}

#[test]
fn test_text_search_with_variable_error() {
    let pf = TextSearchPF::new();
    let subject = PropertyFunctionArg::Variable("s".to_string());
    let object = PropertyFunctionArg::Variable("q".to_string());

    let result = pf.evaluate(&subject, &object);
    assert!(result.is_err());
}

#[test]
fn test_text_search_metadata() {
    let pf = TextSearchPF::new();
    let meta = pf.metadata();
    assert_eq!(meta.category, "text");
    assert!(meta.object_must_be_bound);
}

// --- ConcatPF tests ---

#[test]
fn test_concat_with_list() {
    let pf = ConcatPF::new();
    let subject = PropertyFunctionArg::List(vec![
        PropertyFunctionArg::Term(make_lit_term("Hello")),
        PropertyFunctionArg::Term(make_lit_term(" ")),
        PropertyFunctionArg::Term(make_lit_term("World")),
    ]);
    let object = PropertyFunctionArg::Variable("result".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);

    let row = &result.rows()[0];
    let val = row.get("result").expect("result should be bound");
    assert_eq!(extract_string_value(val), "Hello World");
}

#[test]
fn test_concat_with_single_term() {
    let pf = ConcatPF::new();
    let subject = PropertyFunctionArg::Term(make_lit_term("solo"));
    let object = PropertyFunctionArg::Variable("result".to_string());

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);

    let row = &result.rows()[0];
    let val = row.get("result").expect("result should be bound");
    assert_eq!(extract_string_value(val), "solo");
}

#[test]
fn test_concat_with_variable_error() {
    let pf = ConcatPF::new();
    let subject = PropertyFunctionArg::Variable("parts".to_string());
    let object = PropertyFunctionArg::Variable("result".to_string());

    let result = pf.evaluate(&subject, &object);
    assert!(result.is_err());
}

#[test]
fn test_concat_cardinality() {
    let pf = ConcatPF::new();
    let subj = PropertyFunctionArg::Term(make_lit_term("x"));
    let obj = PropertyFunctionArg::Variable("r".to_string());
    assert_eq!(pf.estimated_cardinality(&subj, &obj), Some(1));
}

// --- StrSplitPF tests ---

#[test]
fn test_str_split_basic() {
    let pf = StrSplitPF::new();
    let subject = PropertyFunctionArg::Term(make_lit_term("a,b,c"));
    let object = PropertyFunctionArg::Term(make_lit_term(","));

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 3);

    let parts: Vec<String> = result
        .iter()
        .map(|r| extract_string_value(r.get("part").expect("part should be bound")))
        .collect();
    assert_eq!(parts, vec!["a", "b", "c"]);
}

#[test]
fn test_str_split_with_list_delimiter() {
    let pf = StrSplitPF::new();
    let subject = PropertyFunctionArg::Term(make_lit_term("x|y|z"));
    let object = PropertyFunctionArg::List(vec![PropertyFunctionArg::Term(make_lit_term("|"))]);

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_str_split_single_part() {
    let pf = StrSplitPF::new();
    let subject = PropertyFunctionArg::Term(make_lit_term("nosep"));
    let object = PropertyFunctionArg::Term(make_lit_term(","));

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_str_split_with_indices() {
    let pf = StrSplitPF::new();
    let subject = PropertyFunctionArg::Term(make_lit_term("one:two:three"));
    let object = PropertyFunctionArg::Term(make_lit_term(":"));

    let result = pf
        .evaluate(&subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 3);

    for (i, row) in result.iter().enumerate() {
        let idx = row.get("index").expect("index should be bound");
        assert_eq!(extract_string_value(idx), (i as i64).to_string());
    }
}

// --- Validation tests ---

#[test]
fn test_default_validation_passes() {
    let pf = ListMemberPF::new();
    let subject = PropertyFunctionArg::Term(make_lit_term("x"));
    let object = PropertyFunctionArg::Variable("y".to_string());

    let result = pf.validate(&subject, &object);
    assert!(result.is_ok());
}

#[test]
fn test_validation_max_object_args_exceeded() {
    let pf = ListIndexPF::new();
    let subject = PropertyFunctionArg::Term(make_lit_term("x"));
    let object = PropertyFunctionArg::List(vec![
        PropertyFunctionArg::Variable("a".to_string()),
        PropertyFunctionArg::Variable("b".to_string()),
        PropertyFunctionArg::Variable("c".to_string()),
    ]);

    let result = pf.validate(&subject, &object);
    assert!(result.is_err());
}

// --- Helper function tests ---

#[test]
fn test_split_iri_helper_hash() {
    let (ns, ln) = split_iri("http://example.org/ns#Thing");
    assert_eq!(ns, "http://example.org/ns#");
    assert_eq!(ln, "Thing");
}

#[test]
fn test_split_iri_helper_slash() {
    let (ns, ln) = split_iri("http://example.org/resource/item");
    assert_eq!(ns, "http://example.org/resource/");
    assert_eq!(ln, "item");
}

#[test]
fn test_split_iri_helper_no_separator() {
    let (ns, ln) = split_iri("urn:isbn:123456");
    assert_eq!(ns, "urn:isbn:");
    assert_eq!(ln, "123456");
}

#[test]
fn test_make_integer_term() {
    let term = make_integer_term(42);
    assert_eq!(extract_string_value(&term), "42");
}

#[test]
fn test_make_double_term() {
    let term = make_double_term(3.125);
    let s = extract_string_value(&term);
    assert!(s.starts_with("3.125"));
}

#[test]
fn test_make_string_term() {
    let term = make_string_term("hello");
    assert_eq!(extract_string_value(&term), "hello");
}

#[test]
fn test_extract_iri_string_success() {
    let term = make_nn_term("http://example.org/test");
    let result = extract_iri_string(&term);
    assert!(result.is_ok());
    assert_eq!(result.expect("should succeed"), "http://example.org/test");
}

#[test]
fn test_extract_iri_string_failure() {
    let term = make_lit_term("not an IRI");
    let result = extract_iri_string(&term);
    assert!(result.is_err());
}

// --- Factory tests ---

#[derive(Debug)]
struct TestFactory;

impl PropertyFunctionFactory for TestFactory {
    fn create(&self, _iri: &str) -> Result<Box<dyn PropertyFunction>, OxirsError> {
        Ok(Box::new(ListMemberPF::new()))
    }
}

#[test]
fn test_factory_registration() {
    let mut registry = PropertyFunctionRegistry::empty();
    registry.register_factory("http://example.org/factory-func", Arc::new(TestFactory));
    assert!(registry.is_property_function("http://example.org/factory-func"));

    // Lookup should trigger factory creation
    let func = registry.lookup("http://example.org/factory-func");
    assert!(func.is_some());
}

#[test]
fn test_factory_cached_after_creation() {
    let mut registry = PropertyFunctionRegistry::empty();
    registry.register_factory("http://example.org/cached-func", Arc::new(TestFactory));

    // First lookup creates via factory
    let func1 = registry.lookup("http://example.org/cached-func");
    assert!(func1.is_some());

    // Second lookup should return cached instance
    let func2 = registry.lookup("http://example.org/cached-func");
    assert!(func2.is_some());
}

// --- Integration test: registry evaluate ---

#[test]
fn test_registry_evaluate_list_member() {
    let mut registry = PropertyFunctionRegistry::new();
    let subject = PropertyFunctionArg::List(vec![
        PropertyFunctionArg::Term(make_lit_term("x")),
        PropertyFunctionArg::Term(make_lit_term("y")),
    ]);
    let object = PropertyFunctionArg::Variable("member".to_string());

    let result = registry
        .evaluate("http://jena.apache.org/ARQ/list#member", &subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 2);

    // Verify stats were updated
    assert_eq!(registry.statistics().total_evaluations, 1);
    assert_eq!(registry.statistics().total_rows_produced, 2);
}

#[test]
fn test_registry_evaluate_list_length() {
    let mut registry = PropertyFunctionRegistry::new();
    let subject = PropertyFunctionArg::List(vec![
        PropertyFunctionArg::Term(make_lit_term("a")),
        PropertyFunctionArg::Term(make_lit_term("b")),
        PropertyFunctionArg::Term(make_lit_term("c")),
        PropertyFunctionArg::Term(make_lit_term("d")),
    ]);
    let object = PropertyFunctionArg::Variable("len".to_string());

    let result = registry
        .evaluate("http://jena.apache.org/ARQ/list#length", &subject, &object)
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);

    let row = &result.rows()[0];
    let len = row.get("length").expect("length should be bound");
    assert_eq!(extract_string_value(len), "4");
}

#[test]
fn test_registry_evaluate_split_iri() {
    let mut registry = PropertyFunctionRegistry::new();
    let subject = PropertyFunctionArg::Term(make_nn_term("http://xmlns.com/foaf/0.1/Person"));
    let object = PropertyFunctionArg::Variable("parts".to_string());

    let result = registry
        .evaluate(
            "http://jena.apache.org/ARQ/property#splitIRI",
            &subject,
            &object,
        )
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_registry_evaluate_concat() {
    let mut registry = PropertyFunctionRegistry::new();
    let subject = PropertyFunctionArg::List(vec![
        PropertyFunctionArg::Term(make_lit_term("foo")),
        PropertyFunctionArg::Term(make_lit_term("bar")),
    ]);
    let object = PropertyFunctionArg::Variable("result".to_string());

    let result = registry
        .evaluate(
            "http://jena.apache.org/ARQ/property#concat",
            &subject,
            &object,
        )
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 1);

    let row = &result.rows()[0];
    let val = row.get("result").expect("result should be bound");
    assert_eq!(extract_string_value(val), "foobar");
}

#[test]
fn test_registry_evaluate_str_split() {
    let mut registry = PropertyFunctionRegistry::new();
    let subject = PropertyFunctionArg::Term(make_lit_term("one-two-three"));
    let object = PropertyFunctionArg::Term(make_lit_term("-"));

    let result = registry
        .evaluate(
            "http://jena.apache.org/ARQ/property#strSplit",
            &subject,
            &object,
        )
        .expect("evaluation should succeed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_per_function_stats_tracking() {
    let mut registry = PropertyFunctionRegistry::new();

    let subj = PropertyFunctionArg::Term(make_lit_term("test"));
    let obj = PropertyFunctionArg::Variable("x".to_string());

    // Evaluate different functions
    let _ = registry.evaluate("http://jena.apache.org/ARQ/list#member", &subj, &obj);
    let _ = registry.evaluate("http://jena.apache.org/ARQ/list#member", &subj, &obj);
    let _ = registry.evaluate("http://jena.apache.org/ARQ/list#length", &subj, &obj);

    let stats = registry.statistics();
    assert_eq!(stats.total_evaluations, 3);
    assert_eq!(
        stats
            .per_function_counts
            .get("http://jena.apache.org/ARQ/list#member"),
        Some(&2)
    );
    assert_eq!(
        stats
            .per_function_counts
            .get("http://jena.apache.org/ARQ/list#length"),
        Some(&1)
    );
}
