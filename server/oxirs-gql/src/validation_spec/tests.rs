//! Tests for the GraphQL specification validation rules.

use crate::ast::{Definition, Document, Selection, SelectionSet};
use crate::types::{BuiltinScalars, FieldType, GraphQLType, ObjectType, Schema};
use crate::validation::{SpecRule, ValidationRule};

use super::SpecValidator;

fn make_schema() -> Schema {
    let mut schema = Schema::new();

    let query_type = ObjectType::new("Query".to_string())
        .with_field(
            "hello".to_string(),
            FieldType::new(
                "hello".to_string(),
                GraphQLType::Scalar(BuiltinScalars::string()),
            ),
        )
        .with_field(
            "world".to_string(),
            FieldType::new(
                "world".to_string(),
                GraphQLType::Scalar(BuiltinScalars::string()),
            ),
        );

    schema.add_type(GraphQLType::Object(query_type));
    schema.set_query_type("Query".to_string());

    schema
}

fn validate(schema: &Schema, query: &str) -> Vec<crate::validation::ValidationError> {
    let doc = crate::parser::parse_document(query).expect("parse failed");
    SpecValidator::new(schema).validate(&doc)
}

// ── ExecutableDefinitions ────────────────────────────────────────────────

#[test]
fn test_executable_definitions_valid() {
    let schema = make_schema();
    let errors = validate(&schema, "{ hello }");
    assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
}

// ── UniqueOperationNames ─────────────────────────────────────────────────

#[test]
fn test_unique_operation_names_valid() {
    let schema = make_schema();
    let errors = validate(&schema, "query Foo { hello } query Bar { hello }");
    let unique_errors: Vec<_> = errors
        .iter()
        .filter(|e| e.rule == ValidationRule::Spec(SpecRule::UniqueOperationNames))
        .collect();
    assert!(
        unique_errors.is_empty(),
        "unexpected errors: {:?}",
        unique_errors
    );
}

#[test]
fn test_unique_operation_names_invalid() {
    let schema = make_schema();
    let errors = validate(&schema, "query Foo { hello } query Foo { world }");
    assert!(
        errors
            .iter()
            .any(|e| e.rule == ValidationRule::Spec(SpecRule::UniqueOperationNames)),
        "expected UniqueOperationNames error, got: {:?}",
        errors
    );
}

// ── LoneAnonymousOperation ────────────────────────────────────────────────

#[test]
fn test_lone_anonymous_operation_valid() {
    let schema = make_schema();
    let errors = validate(&schema, "{ hello }");
    let lone_errors: Vec<_> = errors
        .iter()
        .filter(|e| e.rule == ValidationRule::Spec(SpecRule::LoneAnonymousOperation))
        .collect();
    assert!(
        lone_errors.is_empty(),
        "unexpected errors: {:?}",
        lone_errors
    );
}

#[test]
fn test_lone_anonymous_operation_invalid() {
    let schema = make_schema();
    let errors = validate(&schema, "{ hello } query Foo { world }");
    assert!(
        errors
            .iter()
            .any(|e| e.rule == ValidationRule::Spec(SpecRule::LoneAnonymousOperation)),
        "expected LoneAnonymousOperation error, got: {:?}",
        errors
    );
}

// ── UniqueFragmentNames ───────────────────────────────────────────────────

#[test]
fn test_unique_fragment_names_valid() {
    let schema = make_schema();
    let errors = validate(&schema, "fragment A on Query { hello } query Q { ...A }");
    let frag_errors: Vec<_> = errors
        .iter()
        .filter(|e| e.rule == ValidationRule::Spec(SpecRule::UniqueFragmentNames))
        .collect();
    assert!(
        frag_errors.is_empty(),
        "unexpected errors: {:?}",
        frag_errors
    );
}

// ── NoFragmentCycles ──────────────────────────────────────────────────────

#[test]
fn test_no_fragment_cycles_valid() {
    let schema = make_schema();
    let errors = validate(&schema, "fragment A on Query { hello } query Q { ...A }");
    let cycle_errors: Vec<_> = errors
        .iter()
        .filter(|e| e.rule == ValidationRule::Spec(SpecRule::NoFragmentCycles))
        .collect();
    assert!(
        cycle_errors.is_empty(),
        "unexpected cycle errors: {:?}",
        cycle_errors
    );
}

#[test]
fn test_no_fragment_cycles_invalid() {
    let schema = make_schema();
    // Fragment A spreads B which spreads A — cycle
    let errors = validate(
        &schema,
        "fragment A on Query { ...B } fragment B on Query { ...A } query Q { ...A }",
    );
    assert!(
        errors
            .iter()
            .any(|e| e.rule == ValidationRule::Spec(SpecRule::NoFragmentCycles)),
        "expected NoFragmentCycles error, got: {:?}",
        errors
    );
}

// ── NoUnusedFragments ─────────────────────────────────────────────────────

#[test]
fn test_no_unused_fragments_valid() {
    let schema = make_schema();
    let errors = validate(&schema, "fragment A on Query { hello } query Q { ...A }");
    let unused_errors: Vec<_> = errors
        .iter()
        .filter(|e| e.rule == ValidationRule::Spec(SpecRule::NoUnusedFragments))
        .collect();
    assert!(
        unused_errors.is_empty(),
        "unexpected errors: {:?}",
        unused_errors
    );
}

#[test]
fn test_no_unused_fragments_invalid() {
    let schema = make_schema();
    // Fragment A is defined but never used
    let errors = validate(&schema, "fragment A on Query { hello } query Q { world }");
    assert!(
        errors
            .iter()
            .any(|e| e.rule == ValidationRule::Spec(SpecRule::NoUnusedFragments)),
        "expected NoUnusedFragments error, got: {:?}",
        errors
    );
}

// ── UniqueVariableNames ───────────────────────────────────────────────────

#[test]
fn test_unique_variable_names_invalid() {
    let schema = make_schema();
    // Parser won't surface this for us — build doc manually
    let doc = Document {
        definitions: vec![Definition::Operation(crate::ast::OperationDefinition {
            operation_type: crate::ast::OperationType::Query,
            name: Some("Q".to_string()),
            variable_definitions: vec![
                crate::ast::VariableDefinition {
                    variable: crate::ast::Variable {
                        name: "x".to_string(),
                    },
                    type_: crate::ast::Type::NamedType("String".to_string()),
                    default_value: None,
                    directives: vec![],
                },
                crate::ast::VariableDefinition {
                    variable: crate::ast::Variable {
                        name: "x".to_string(),
                    },
                    type_: crate::ast::Type::NamedType("String".to_string()),
                    default_value: None,
                    directives: vec![],
                },
            ],
            directives: vec![],
            selection_set: SelectionSet {
                selections: vec![Selection::Field(crate::ast::Field {
                    alias: None,
                    name: "hello".to_string(),
                    arguments: vec![],
                    directives: vec![],
                    selection_set: None,
                })],
            },
        })],
    };
    let errors = SpecValidator::new(&schema).validate(&doc);
    assert!(
        errors
            .iter()
            .any(|e| e.rule == ValidationRule::Spec(SpecRule::UniqueVariableNames)),
        "expected UniqueVariableNames error, got: {:?}",
        errors
    );
}

// ── KnownFragmentNames ────────────────────────────────────────────────────

#[test]
fn test_known_fragment_names_invalid() {
    let schema = make_schema();
    // Spread refers to fragment "Undefined" which doesn't exist
    let doc = Document {
        definitions: vec![Definition::Operation(crate::ast::OperationDefinition {
            operation_type: crate::ast::OperationType::Query,
            name: None,
            variable_definitions: vec![],
            directives: vec![],
            selection_set: SelectionSet {
                selections: vec![Selection::FragmentSpread(crate::ast::FragmentSpread {
                    fragment_name: "Undefined".to_string(),
                    directives: vec![],
                })],
            },
        })],
    };
    let errors = SpecValidator::new(&schema).validate(&doc);
    assert!(
        errors
            .iter()
            .any(|e| e.rule == ValidationRule::Spec(SpecRule::KnownFragmentNames)),
        "expected KnownFragmentNames error, got: {:?}",
        errors
    );
}

// ── OverlappingFieldsCanBeMerged ──────────────────────────────────────────

#[test]
fn test_overlapping_fields_conflict() {
    let schema = make_schema();
    // Same response name "x" maps to different fields
    let doc = Document {
        definitions: vec![Definition::Operation(crate::ast::OperationDefinition {
            operation_type: crate::ast::OperationType::Query,
            name: None,
            variable_definitions: vec![],
            directives: vec![],
            selection_set: SelectionSet {
                selections: vec![
                    Selection::Field(crate::ast::Field {
                        alias: Some("x".to_string()),
                        name: "hello".to_string(),
                        arguments: vec![],
                        directives: vec![],
                        selection_set: None,
                    }),
                    Selection::Field(crate::ast::Field {
                        alias: Some("x".to_string()),
                        name: "world".to_string(),
                        arguments: vec![],
                        directives: vec![],
                        selection_set: None,
                    }),
                ],
            },
        })],
    };
    let errors = SpecValidator::new(&schema).validate(&doc);
    assert!(
        errors
            .iter()
            .any(|e| e.rule == ValidationRule::Spec(SpecRule::OverlappingFieldsCanBeMerged)),
        "expected OverlappingFieldsCanBeMerged error, got: {:?}",
        errors
    );
}

// ── ValuesOfCorrectType: Int 32-bit boundary ──────────────────────────────

#[test]
fn test_spec_validator_new() {
    let schema = make_schema();
    let _validator = SpecValidator::new(&schema);
}
