//! Comprehensive criterion benchmarks for the oxirs-gql GraphQL server crate.
//!
//! Covers:
//! - Query parsing (simple, complex fragments, deeply nested, introspection)
//! - GraphQL spec validation (all 25 rules — valid path, violation path, batch)
//! - Argument / type-level validation (ValuesOfCorrectType, ProvidedRequiredArguments)
//! - Throughput across several document sizes

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxirs_gql::{
    parser::Parser,
    types::{ArgumentType, BuiltinScalars, FieldType, GraphQLType, ObjectType, Schema},
    validation::ValidationConfig,
    validation_spec::SpecValidator,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a minimal Schema that mirrors GraphQLServer::start()
// ─────────────────────────────────────────────────────────────────────────────

fn build_bench_schema() -> Schema {
    let mut schema = Schema::new();

    // Field: hello -> String
    let hello_field = FieldType::new(
        "hello".to_string(),
        GraphQLType::Scalar(BuiltinScalars::string()),
    )
    .with_description("A simple greeting message".to_string());

    // Field: version -> String
    let version_field = FieldType::new(
        "version".to_string(),
        GraphQLType::Scalar(BuiltinScalars::string()),
    )
    .with_description("OxiRS GraphQL version".to_string());

    // Field: triples -> Int
    let triples_field = FieldType::new(
        "triples".to_string(),
        GraphQLType::Scalar(BuiltinScalars::int()),
    )
    .with_description("Count of triples in the store".to_string());

    // Field: subjects(limit: Int) -> [String]
    let subjects_field = FieldType::new(
        "subjects".to_string(),
        GraphQLType::List(Box::new(GraphQLType::Scalar(BuiltinScalars::string()))),
    )
    .with_description("List of subject IRIs".to_string())
    .with_argument(
        "limit".to_string(),
        ArgumentType::new(
            "limit".to_string(),
            GraphQLType::Scalar(BuiltinScalars::int()),
        )
        .with_description("Maximum number of subjects to return".to_string()),
    );

    // Field: predicates(limit: Int) -> [String]
    let predicates_field = FieldType::new(
        "predicates".to_string(),
        GraphQLType::List(Box::new(GraphQLType::Scalar(BuiltinScalars::string()))),
    )
    .with_description("List of predicate IRIs".to_string())
    .with_argument(
        "limit".to_string(),
        ArgumentType::new(
            "limit".to_string(),
            GraphQLType::Scalar(BuiltinScalars::int()),
        )
        .with_description("Maximum number of predicates to return".to_string()),
    );

    // Field: sparql(query: String!) -> String
    let sparql_field = FieldType::new(
        "sparql".to_string(),
        GraphQLType::Scalar(BuiltinScalars::string()),
    )
    .with_description("Execute a raw SPARQL query".to_string())
    .with_argument(
        "query".to_string(),
        ArgumentType::new(
            "query".to_string(),
            GraphQLType::NonNull(Box::new(GraphQLType::Scalar(BuiltinScalars::string()))),
        )
        .with_description("The SPARQL query to execute".to_string()),
    );

    let query_type = ObjectType::new("Query".to_string())
        .with_description("Root query type for RDF data access".to_string())
        .with_field("hello".to_string(), hello_field)
        .with_field("version".to_string(), version_field)
        .with_field("triples".to_string(), triples_field)
        .with_field("subjects".to_string(), subjects_field)
        .with_field("predicates".to_string(), predicates_field)
        .with_field("sparql".to_string(), sparql_field);

    schema.add_type(GraphQLType::Object(query_type));
    schema.set_query_type("Query".to_string());
    schema
}

// ─────────────────────────────────────────────────────────────────────────────
// Canonical document strings (all `'static` — no transmute UB risk)
// ─────────────────────────────────────────────────────────────────────────────

const SIMPLE_QUERY: &str = r#"
    {
        hello
    }
"#;

const COMPLEX_QUERY_WITH_FRAGMENTS: &str = r#"
    fragment MetaFields on Query {
        version
        triples
    }

    query FetchResources {
        hello
        subjects(limit: 10)
        predicates(limit: 5)
        ...MetaFields
    }
"#;

// 6 levels of nesting via inline-fragments (Query has no object sub-fields,
// so we use alias + inline fragments to force parser work at depth ≥ 5)
const DEEPLY_NESTED_QUERY: &str = r#"
    query DeepNesting {
        ... on Query {
            ... on Query {
                ... on Query {
                    ... on Query {
                        ... on Query {
                            hello
                            version
                        }
                    }
                }
            }
        }
    }
"#;

const INTROSPECTION_QUERY: &str = r#"
    query IntrospectionQuery {
        __schema {
            queryType { name }
            mutationType { name }
            subscriptionType { name }
        }
        __type(name: "Query") {
            kind
            name
            fields { name }
        }
    }
"#;

// Valid document: uses real fields defined in the bench schema
const VALID_SPEC_DOCUMENT: &str = r#"
    query FetchAll {
        hello
        version
        triples
        subjects(limit: 50)
        predicates(limit: 50)
    }
"#;

// Document with multiple spec violations (no schema dependence required):
// - Duplicate operation name → UniqueOperationNames
// - Subscription with two top-level fields → SingleFieldSubscriptions
const VIOLATIONS_DOCUMENT: &str = r#"
    query Dup { hello }
    query Dup { version }
"#;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: parse doc, expect a specific number of definitions
// ─────────────────────────────────────────────────────────────────────────────

fn parse(src: &str) -> oxirs_gql::ast::Document {
    Parser::new(src)
        .parse_document()
        .expect("bench document must parse successfully")
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 1 – Query Parsing
// ─────────────────────────────────────────────────────────────────────────────

fn bench_query_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_query");

    group.bench_function("simple", |b| {
        b.iter(|| {
            Parser::new(SIMPLE_QUERY)
                .parse_document()
                .expect("simple parse")
        });
    });

    group.bench_function("complex_with_fragments", |b| {
        b.iter(|| {
            Parser::new(COMPLEX_QUERY_WITH_FRAGMENTS)
                .parse_document()
                .expect("fragment parse")
        });
    });

    group.bench_function("deeply_nested_5_levels", |b| {
        b.iter(|| {
            Parser::new(DEEPLY_NESTED_QUERY)
                .parse_document()
                .expect("deep nested parse")
        });
    });

    group.bench_function("introspection_query", |b| {
        b.iter(|| {
            Parser::new(INTROSPECTION_QUERY)
                .parse_document()
                .expect("introspection parse")
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 2 – Spec validation (SpecValidator — all 25 rules)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_spec_validation(c: &mut Criterion) {
    let schema = build_bench_schema();
    let validator = SpecValidator::new(&schema);

    // Parse documents once outside the timing loop.
    let valid_doc = parse(VALID_SPEC_DOCUMENT);
    let violations_doc = parse(VIOLATIONS_DOCUMENT);

    // Build a batch of 100 varied valid queries
    let batch_queries: Vec<String> = (0..100)
        .map(|i| {
            if i % 3 == 0 {
                format!("query Batch{i} {{ hello }}")
            } else if i % 3 == 1 {
                format!("query Batch{i} {{ version triples }}")
            } else {
                format!("query Batch{i} {{ subjects(limit: {i}) predicates(limit: {i}) }}")
            }
        })
        .collect();
    let batch_docs: Vec<oxirs_gql::ast::Document> =
        batch_queries.iter().map(|q| parse(q)).collect();

    let mut group = c.benchmark_group("spec_validation");

    group.bench_function("valid_document_all_25_rules_pass", |b| {
        b.iter(|| {
            let errors = validator.validate(&valid_doc);
            // Callers typically check is_empty()
            black_box(errors)
        });
    });

    group.bench_function("document_with_violations_rejection_speed", |b| {
        b.iter(|| {
            let errors = validator.validate(&violations_doc);
            black_box(errors)
        });
    });

    group.bench_function("batch_100_queries", |b| {
        b.iter(|| {
            let all_errors: Vec<_> = batch_docs
                .iter()
                .map(|doc| validator.validate(doc))
                .collect();
            black_box(all_errors)
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 3 – Argument / type validation (via SpecValidator rule coverage)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_argument_validation(c: &mut Criterion) {
    let schema = build_bench_schema();
    let validator = SpecValidator::new(&schema);

    // ValuesOfCorrectType: correct Int argument value
    let correct_type_doc = parse(r#"query CorrectType { subjects(limit: 25) }"#);

    // ProvidedRequiredArguments: sparql(query:) requires a non-null argument;
    // omitting it exercises the rule.
    let missing_required_doc = parse(r#"query MissingArg { sparql }"#);

    // Deeply named variables path (VariablesAreInputTypes)
    let variables_doc = parse(
        r#"query WithVar($limit: Int) { subjects(limit: $limit) predicates(limit: $limit) }"#,
    );

    let mut group = c.benchmark_group("argument_validation");

    group.bench_function("values_of_correct_type_pass", |b| {
        b.iter(|| black_box(validator.validate(&correct_type_doc)));
    });

    group.bench_function("provided_required_args_violation", |b| {
        b.iter(|| black_box(validator.validate(&missing_required_doc)));
    });

    group.bench_function("variables_are_input_types", |b| {
        b.iter(|| black_box(validator.validate(&variables_doc)));
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 4 – Throughput at several document sizes
// ─────────────────────────────────────────────────────────────────────────────

fn bench_throughput(c: &mut Criterion) {
    // Build documents of increasing byte sizes by stacking field selections.
    let targets_bytes: &[usize] = &[100, 500, 1_000, 5_000, 10_000];

    let mut group = c.benchmark_group("parse_throughput");

    for &target in targets_bytes {
        // Construct a query large enough to hit target bytes.
        // We repeat field aliases (a0: hello, a1: version, …) until we reach size.
        let fields_needed = target / "        a0: hello\n".len() + 1;
        let mut body = "query LargeQuery {\n".to_string();
        for i in 0..fields_needed {
            if i % 2 == 0 {
                body.push_str(&format!("        a{i}: hello\n"));
            } else {
                body.push_str(&format!("        b{i}: version\n"));
            }
        }
        body.push('}');

        // Ensure we have at least `target` bytes
        while body.len() < target {
            let idx = body.len();
            body.insert_str(body.len() - 1, &format!("\n        z{idx}: triples\n"));
        }

        let byte_len = body.len() as u64;
        group.throughput(Throughput::Bytes(byte_len));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{target}_bytes")),
            &body,
            |b, query_str| {
                b.iter(|| {
                    Parser::new(query_str)
                        .parse_document()
                        .expect("throughput bench parse")
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 5 – QueryValidator (security/depth/complexity checks)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_query_validator(c: &mut Criterion) {
    let schema = build_bench_schema();
    let config = ValidationConfig::default();
    let validator = oxirs_gql::validation::QueryValidator::new(config, schema.clone());

    let simple_doc = parse(SIMPLE_QUERY);
    let complex_doc = parse(COMPLEX_QUERY_WITH_FRAGMENTS);

    let mut group = c.benchmark_group("query_validator");

    group.bench_function("simple_query", |b| {
        b.iter(|| black_box(validator.validate(&simple_doc)));
    });

    group.bench_function("complex_with_fragments", |b| {
        b.iter(|| black_box(validator.validate(&complex_doc)));
    });

    // Validator with tight limits — exercises early-exit on depth violation
    let strict_config = ValidationConfig {
        max_depth: 1,
        max_complexity: 5,
        ..ValidationConfig::default()
    };
    let strict_validator =
        oxirs_gql::validation::QueryValidator::new(strict_config, schema.clone());

    let deep_doc = parse(VALID_SPEC_DOCUMENT);

    group.bench_function("strict_limits_violation_path", |b| {
        b.iter(|| black_box(strict_validator.validate(&deep_doc)));
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 6 – Spec validation: individual rules in isolation
// ─────────────────────────────────────────────────────────────────────────────

fn bench_spec_rules_isolation(c: &mut Criterion) {
    let schema = build_bench_schema();
    let validator = SpecValidator::new(&schema);

    // Unique operation names — 10 uniquely-named operations
    let multi_ops: Vec<String> = (0..10)
        .map(|i| format!("query Op{i} {{ hello }}"))
        .collect();
    let multi_ops_str = multi_ops.join("\n");
    let multi_ops_doc = parse(&multi_ops_str);

    // Fragment cycle: A spreads B, B spreads A (NoFragmentCycles)
    let cycle_doc = parse(
        r#"
        fragment A on Query { ...B }
        fragment B on Query { ...A }
        query Q { ...A }
        "#,
    );

    // Known directives violation: @unknownDir
    let unknown_directive_doc = parse(r#"query WithDir { hello @unknownDir }"#);

    // Unique argument names violation: duplicate arg in call
    let dup_arg_doc = parse(r#"query DupArg { subjects(limit: 5, limit: 10) }"#);

    let mut group = c.benchmark_group("spec_rules_isolation");

    group.bench_function("unique_operation_names_10_ops", |b| {
        b.iter(|| black_box(validator.validate(&multi_ops_doc)));
    });

    group.bench_function("no_fragment_cycles_cycle_detected", |b| {
        b.iter(|| black_box(validator.validate(&cycle_doc)));
    });

    group.bench_function("known_directives_unknown_dir", |b| {
        b.iter(|| black_box(validator.validate(&unknown_directive_doc)));
    });

    group.bench_function("unique_argument_names_duplicate", |b| {
        b.iter(|| black_box(validator.validate(&dup_arg_doc)));
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark 7 – Schema construction cost
// ─────────────────────────────────────────────────────────────────────────────

fn bench_schema_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("schema_construction");

    group.bench_function("build_bench_schema", |b| {
        b.iter(|| black_box(build_bench_schema()));
    });

    group.bench_function("schema_get_type_hit", |b| {
        let schema = build_bench_schema();
        b.iter(|| black_box(schema.get_type("Query")));
    });

    group.bench_function("schema_get_type_miss", |b| {
        let schema = build_bench_schema();
        b.iter(|| black_box(schema.get_type("NonExistentType")));
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Criterion group + main
// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_query_parsing,
    bench_spec_validation,
    bench_argument_validation,
    bench_throughput,
    bench_query_validator,
    bench_spec_rules_isolation,
    bench_schema_construction,
);
criterion_main!(benches);
