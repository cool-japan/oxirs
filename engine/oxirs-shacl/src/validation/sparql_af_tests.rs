//! Integration tests for SHACL-AF SPARQL target features
//!
//! Covers:
//! - `sh:SPARQLTarget` selection (focus nodes identified via SELECT)
//! - `sh:SPARQLTargetType` parameterised target types
//! - `sh:SPARQLAskValidator` (ASK-based constraints)
//! - Combined SPARQL + regular constraints
//! - Deactivated SPARQL targets / validators
//! - Error handling / executor failures
//! - PrefixMap and SubstitutionContext utilities

use std::collections::HashMap;

use crate::sparql_af::{
    ask_validator::{
        FailingAskExecutor, MockAskExecutor, SparqlAskValidator, SparqlAskValidatorBuilder,
    },
    sparql_target::{this_row, SparqlAfTarget, SparqlTargetMock},
    target_type::{SparqlTargetType, SparqlTargetTypeRegistry},
    PrefixMap, SubstitutionContext,
};

// ============================================================
// 1.  sh:SPARQLTarget focus node selection
// ============================================================

#[test]
fn test_sparql_target_selects_nodes_by_type() {
    let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a <http://schema.org/Person> }");

    let evaluator = SparqlTargetMock::new().with_response(
        "schema.org/Person",
        vec![
            this_row("http://example.org/Alice"),
            this_row("http://example.org/Bob"),
        ],
    );

    let result = target
        .evaluate(&evaluator)
        .expect("evaluate should succeed");
    assert_eq!(result.count(), 2, "should find two focus nodes");
    assert!(result.has_nodes());
    assert!(!result.was_deactivated);
}

#[test]
fn test_sparql_target_empty_result_no_focus_nodes() {
    let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a <http://example.org/Unknown> }");
    let evaluator = SparqlTargetMock::new(); // always returns empty

    let result = target
        .evaluate(&evaluator)
        .expect("evaluate should succeed");
    assert_eq!(result.count(), 0);
    assert!(!result.has_nodes());
}

#[test]
fn test_sparql_target_with_prefix_declarations() {
    let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a ex:Employee }")
        .with_prefix("ex", "http://example.org/");

    let query = target.build_query();
    assert!(
        query.contains("PREFIX ex: <http://example.org/>"),
        "prefix should be in query"
    );
    assert!(
        query.contains("ex:Employee"),
        "query body should use prefix"
    );
}

#[test]
fn test_sparql_target_parameter_substitution_in_query() {
    let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a $targetClass }")
        .with_parameter("targetClass", "<http://example.org/Manager>");

    let query = target.build_query();
    assert!(
        query.contains("<http://example.org/Manager>"),
        "parameter should be substituted"
    );
    assert!(
        !query.contains("$targetClass"),
        "placeholder should be removed"
    );
}

#[test]
fn test_sparql_target_deactivated_returns_empty() {
    let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a <http://example.org/Active> }")
        .deactivated();

    let evaluator =
        SparqlTargetMock::new().with_default(vec![this_row("http://example.org/Node1")]);

    let result = target
        .evaluate(&evaluator)
        .expect("should not fail even when deactivated");
    assert!(result.was_deactivated);
    assert_eq!(result.count(), 0, "deactivated target must return no nodes");
}

#[test]
fn test_sparql_target_filters_rows_without_this_binding() {
    let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a ex:Item }");

    // Only one row has the correct `?this` key
    let mut bad_row = HashMap::new();
    bad_row.insert("node".to_string(), "http://example.org/X".to_string());

    let evaluator = SparqlTargetMock::new().with_response(
        "ex:Item",
        vec![this_row("http://example.org/Good"), bad_row],
    );

    let result = target.evaluate(&evaluator).expect("should succeed");
    assert_eq!(result.result_rows, 2); // 2 SPARQL rows
    assert_eq!(result.count(), 1); // only 1 had ?this
}

#[test]
fn test_sparql_target_label_field() {
    let target =
        SparqlAfTarget::new("SELECT ?this WHERE { ?this ?p ?o }").with_label("AllNodesTarget");
    assert_eq!(target.label, Some("AllNodesTarget".to_string()));
}

// ============================================================
// 2.  sh:SPARQLTargetType parameterised target types
// ============================================================

#[test]
fn test_sparql_target_type_basic_instantiation() {
    let tt = SparqlTargetType::new(
        "http://example.org/types/ClassTarget",
        "SELECT ?this WHERE { ?this a $cls }",
    )
    .require_param("cls");

    let mut bindings = HashMap::new();
    bindings.insert(
        "cls".to_string(),
        "<http://example.org/Vehicle>".to_string(),
    );

    let instance = tt
        .instantiate(bindings)
        .expect("instantiation should succeed");
    let query = instance.build_query();
    assert!(query.contains("<http://example.org/Vehicle>"));
    assert!(!query.contains("$cls"));
}

#[test]
fn test_sparql_target_type_missing_required_param_is_error() {
    let tt = SparqlTargetType::new(
        "http://example.org/types/PropTarget",
        "SELECT ?this WHERE { ?this $prop ?o }",
    )
    .require_param("prop");

    let result = tt.instantiate(HashMap::new());
    assert!(
        result.is_err(),
        "missing required param should produce an error"
    );
}

#[test]
fn test_sparql_target_type_optional_param_uses_default() {
    let tt = SparqlTargetType::new(
        "http://example.org/types/LimitedTarget",
        "SELECT ?this WHERE { ?this a ex:Node } LIMIT $limit",
    )
    .optional_param("limit", "100");

    let instance = tt
        .instantiate(HashMap::new())
        .expect("optional params should use defaults");
    let query = instance.build_query();
    assert!(query.contains("100"), "default value should be substituted");
}

#[test]
fn test_sparql_target_type_registry_register_and_instantiate() {
    let mut registry = SparqlTargetTypeRegistry::new();
    registry.register(
        SparqlTargetType::new(
            "http://example.org/types/PersonTarget",
            "SELECT ?this WHERE { ?this a ex:Person }",
        )
        .with_prefix("ex", "http://example.org/"),
    );

    assert_eq!(registry.len(), 1);

    let instance = registry
        .instantiate("http://example.org/types/PersonTarget", HashMap::new())
        .expect("should find and instantiate");

    let evaluator = SparqlTargetMock::new()
        .with_response("ex:Person", vec![this_row("http://example.org/Charlie")]);

    let nodes = instance
        .evaluate(&evaluator)
        .expect("evaluate should succeed");
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0], "http://example.org/Charlie");
}

#[test]
fn test_sparql_target_type_registry_unknown_iri_is_error() {
    let registry = SparqlTargetTypeRegistry::new();
    let result = registry.instantiate("http://example.org/types/Nonexistent", HashMap::new());
    assert!(result.is_err(), "unknown type IRI should return an error");
}

#[test]
fn test_sparql_target_type_multiple_params() {
    let tt = SparqlTargetType::new(
        "http://example.org/types/PropertyValueTarget",
        "SELECT ?this WHERE { ?this $prop $value }",
    )
    .require_param("prop")
    .require_param("value");

    let mut bindings = HashMap::new();
    bindings.insert(
        "prop".to_string(),
        "<http://example.org/status>".to_string(),
    );
    bindings.insert("value".to_string(), "\"active\"".to_string());

    let instance = tt
        .instantiate(bindings)
        .expect("should succeed with both params");
    let query = instance.build_query();
    assert!(query.contains("<http://example.org/status>"));
    assert!(query.contains("\"active\""));
}

// ============================================================
// 3.  sh:SPARQLAskValidator  ASK-based constraints
// ============================================================

#[test]
fn test_ask_validator_conforms_when_ask_true() {
    let v = SparqlAskValidator::new("ASK { $this a ex:ValidEntity }");
    let executor = MockAskExecutor::conforming();

    let result = v
        .validate_node("<http://example.org/Node>", &executor)
        .expect("validation should not fail");

    assert!(result.conforms);
    assert!(result.violation.is_none());
    assert!(!result.was_deactivated);
}

#[test]
fn test_ask_validator_violates_when_ask_false() {
    let v = SparqlAskValidator::new("ASK { $this a ex:ValidEntity }")
        .with_message("Node must be an ex:ValidEntity");
    let executor = MockAskExecutor::violating();

    let result = v
        .validate_node("<http://example.org/InvalidNode>", &executor)
        .expect("validation should not fail");

    assert!(!result.conforms);
    assert!(result.is_violated());
    let viol = result.violation.expect("violation should exist");
    assert!(viol.message.contains("ValidEntity"));
}

#[test]
fn test_ask_validator_deactivated_always_conforms() {
    let v = SparqlAskValidator::new("ASK { $this a ex:RequiredType }").deactivated();
    let executor = MockAskExecutor::violating();

    let result = v
        .validate_node("<http://example.org/X>", &executor)
        .expect("deactivated validator should not fail");

    assert!(result.conforms);
    assert!(result.was_deactivated);
    assert!(result.violation.is_none());
}

#[test]
fn test_ask_validator_executor_error_propagates() {
    let v = SparqlAskValidator::new("ASK { $this ?p ?o }");
    let executor = FailingAskExecutor::new("SPARQL endpoint timed out");

    let result = v.validate_node("<http://example.org/Node>", &executor);
    assert!(
        result.is_err(),
        "executor failure should propagate as error"
    );
}

#[test]
fn test_ask_validator_with_param_substitution() {
    let v = SparqlAskValidator::new("ASK { $this a $requiredClass }")
        .with_parameter("requiredClass", "<http://example.org/Manager>");

    let query = v.build_query("<http://example.org/Alice>");
    assert!(query.contains("<http://example.org/Manager>"));
    assert!(!query.contains("$requiredClass"));
    assert!(query.contains("<http://example.org/Alice>"));
}

#[test]
fn test_ask_validator_batch_validate_nodes() {
    let v = SparqlAskValidator::new("ASK { $this a ex:ActiveUser }");
    // Nodes containing "InactiveUser" in the query will violate; others conform
    let executor = MockAskExecutor::conforming().with_response("InactiveUser", false);

    let nodes: Vec<&str> = vec!["<http://example.org/Alice>", "<http://example.org/Bob>"];

    let results = v.validate_nodes(&nodes, &executor).expect("should succeed");
    assert_eq!(results.len(), 2);
    // Both use the same executor that returns conforming for all (no InactiveUser in query)
    for r in &results {
        assert!(r.conforms);
    }
}

#[test]
fn test_ask_validator_builder_constructs_correctly() {
    let v = SparqlAskValidatorBuilder::new()
        .with_query("ASK { $this ex:isValid true }")
        .with_message("Resource must be marked valid")
        .with_prefix("ex", "http://example.org/")
        .with_iri("http://example.org/constraint/IsValid")
        .build()
        .expect("builder should succeed");

    assert!(v.message.is_some());
    assert_eq!(
        v.constraint_iri,
        Some("http://example.org/constraint/IsValid".to_string())
    );
    assert!(v.prefixes.0.contains_key("ex"));
}

#[test]
fn test_ask_validator_builder_fails_on_empty_query() {
    let result = SparqlAskValidatorBuilder::new().with_message("msg").build();
    assert!(result.is_err(), "empty query should produce an error");
}

// ============================================================
// 4.  PrefixMap and SubstitutionContext utility tests
// ============================================================

#[test]
fn test_prefix_map_with_shacl_defaults() {
    let map = PrefixMap::new().with_shacl_defaults();
    let decls = map.render_declarations();
    assert!(decls.contains("PREFIX sh: <http://www.w3.org/ns/shacl#>"));
    assert!(decls.contains("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"));
    assert!(decls.contains("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"));
    assert!(decls.contains("PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"));
}

#[test]
fn test_prefix_map_merge_does_not_lose_entries() {
    let mut base = PrefixMap::new()
        .with_prefix("ex", "http://example.org/")
        .with_prefix("owl", "http://www.w3.org/2002/07/owl#");
    let extra = PrefixMap::new().with_prefix("sh", "http://www.w3.org/ns/shacl#");
    base.merge(&extra);
    assert_eq!(base.0.len(), 3);
}

#[test]
fn test_substitution_context_all_placeholders_replaced() {
    let ctx = SubstitutionContext::new()
        .with_this("<http://example.org/Item1>")
        .bind("class", "<http://example.org/Product>")
        .bind("status", "\"published\"");

    let query = "SELECT ?this WHERE { $this a $class ; ex:status $status }";
    let result = ctx.apply(query);

    assert!(!result.contains("$this"));
    assert!(!result.contains("$class"));
    assert!(!result.contains("$status"));
    assert!(result.contains("<http://example.org/Item1>"));
    assert!(result.contains("<http://example.org/Product>"));
    assert!(result.contains("\"published\""));
}

#[test]
fn test_substitution_context_unbound_placeholder_stays() {
    let ctx = SubstitutionContext::new().with_this("<http://example.org/X>");
    let query = "ASK { $this a $undeclaredParam }";
    let result = ctx.apply(query);
    // $this should be substituted, but unbound param stays
    assert!(!result.contains("$this"));
    assert!(result.contains("$undeclaredParam"));
}

// ============================================================
// 5.  Combined SPARQL-AF + regular constraint scenario
// ============================================================

#[test]
fn test_combined_sparql_target_and_ask_validator() {
    // Step 1: Use a SPARQL target to identify focus nodes
    let target = SparqlAfTarget::new("SELECT ?this WHERE { ?this a <http://schema.org/Employee> }");

    let target_evaluator = SparqlTargetMock::new().with_response(
        "schema.org/Employee",
        vec![
            this_row("http://example.org/Alice"),
            this_row("http://example.org/Bob"),
            this_row("http://example.org/Charlie"),
        ],
    );

    let target_result = target
        .evaluate(&target_evaluator)
        .expect("target evaluation should succeed");

    assert_eq!(target_result.count(), 3);

    // Step 2: Validate each focus node with an ASK validator
    let ask_validator = SparqlAskValidator::new("ASK { $this <http://schema.org/isActive> true }")
        .with_message("Employee {?this} must be active");

    // Alice and Bob are active, Charlie is not
    let ask_executor = MockAskExecutor::conforming().with_response("Charlie", false);

    let mut violations = 0;
    for node in &target_result.focus_nodes {
        let result = ask_validator
            .validate_node(node, &ask_executor)
            .expect("ask validation should not fail");
        if result.is_violated() {
            violations += 1;
        }
    }

    assert_eq!(
        violations, 1,
        "only Charlie should violate the ASK constraint"
    );
}

#[test]
fn test_sparql_target_type_registry_with_multiple_types() {
    let mut registry = SparqlTargetTypeRegistry::new();

    // Register a "class instances" target type
    registry.register(
        SparqlTargetType::new(
            "http://example.org/types/ClassInstances",
            "SELECT ?this WHERE { ?this a $cls }",
        )
        .require_param("cls"),
    );

    // Register a "subjects-of property" target type
    registry.register(
        SparqlTargetType::new(
            "http://example.org/types/SubjectsOf",
            "SELECT ?this WHERE { ?this $property ?obj }",
        )
        .require_param("property"),
    );

    assert_eq!(registry.len(), 2);
    assert!(!registry.is_empty());

    // Instantiate the first type
    let mut bindings1 = HashMap::new();
    bindings1.insert("cls".to_string(), "<http://example.org/Sensor>".to_string());
    let inst1 = registry
        .instantiate("http://example.org/types/ClassInstances", bindings1)
        .expect("should instantiate");

    let evaluator = SparqlTargetMock::new()
        .with_response("Sensor", vec![this_row("http://example.org/Sensor1")]);
    let nodes1 = inst1.evaluate(&evaluator).expect("should succeed");
    assert_eq!(nodes1.len(), 1);
}
