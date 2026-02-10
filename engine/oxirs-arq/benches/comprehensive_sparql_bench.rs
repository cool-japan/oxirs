//! Comprehensive SPARQL Query Engine Benchmarking Suite
//!
//! Beta.1 Feature: Production-Ready SPARQL Performance Benchmarking
//!
//! This benchmark suite provides comprehensive performance testing across all
//! SPARQL operations:
//! - Query parsing and validation
//! - Pattern matching and evaluation
//! - Join operations (hash, merge, nested loop)
//! - Filter and aggregation performance
//! - Memory usage and scalability

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_arq::{
    executor::{ConcreteStoreDataset, QueryExecutor},
    query::QueryParser,
};
use oxirs_core::{
    model::{Literal, NamedNode, Quad},
    rdf_store::ConcreteStore,
};
use std::time::Duration;

/// Benchmark SPARQL query parsing
fn bench_query_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparql_parsing");
    group.measurement_time(Duration::from_secs(10));

    let queries = vec![
        ("simple_select", "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"),
        ("filtered_select", "SELECT ?s ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name . FILTER(?name = \"Alice\") }"),
        ("join_query", "SELECT ?s ?name ?email WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name . ?s <http://xmlns.com/foaf/0.1/mbox> ?email }"),
        ("union_query", "SELECT ?person WHERE { { ?person a <http://xmlns.com/foaf/0.1/Person> } UNION { ?person a <http://schema.org/Person> } }"),
        ("optional_query", "SELECT ?s ?name ?email WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name . OPTIONAL { ?s <http://xmlns.com/foaf/0.1/mbox> ?email } }"),
    ];

    for (name, query) in queries {
        group.bench_with_input(BenchmarkId::from_parameter(name), query, |b, query| {
            b.iter(|| {
                let mut parser = QueryParser::new();
                let _ = black_box(parser.parse(query));
            });
        });
    }

    group.finish();
}

/// Benchmark basic pattern matching
fn bench_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_matching");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![100, 1_000, 10_000];

    for size in sizes {
        let store = setup_test_store(size);
        let dataset = ConcreteStoreDataset::new(store);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("single_pattern", size),
            &dataset,
            |b, dataset| {
                let query = "SELECT ?s ?o WHERE { ?s <http://example.org/p> ?o }";
                let mut parser = QueryParser::new();
                let query_ast = parser.parse(query).unwrap();
                let algebra = query_ast.where_clause;

                b.iter(|| {
                    let mut executor = QueryExecutor::new();
                    black_box(executor.execute(&algebra, dataset).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark join operations
fn bench_join_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("join_operations");
    group.measurement_time(Duration::from_secs(15));

    let sizes = vec![100, 500, 1_000];

    for size in sizes {
        let store = setup_test_store(size);
        let dataset = ConcreteStoreDataset::new(store);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("two_way_join", size),
            &dataset,
            |b, dataset| {
                let query = r#"
                    SELECT ?s ?name ?email WHERE {
                        ?s <http://xmlns.com/foaf/0.1/name> ?name .
                        ?s <http://xmlns.com/foaf/0.1/mbox> ?email
                    }
                "#;
                let mut parser = QueryParser::new();
                let query_ast = parser.parse(query).unwrap();
                let algebra = query_ast.where_clause;

                b.iter(|| {
                    let mut executor = QueryExecutor::new();
                    black_box(executor.execute(&algebra, dataset).unwrap());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("three_way_join", size),
            &dataset,
            |b, dataset| {
                let query = r#"
                    SELECT ?s ?name ?email ?age WHERE {
                        ?s <http://xmlns.com/foaf/0.1/name> ?name .
                        ?s <http://xmlns.com/foaf/0.1/mbox> ?email .
                        ?s <http://xmlns.com/foaf/0.1/age> ?age
                    }
                "#;
                let mut parser = QueryParser::new();
                let query_ast = parser.parse(query).unwrap();
                let algebra = query_ast.where_clause;

                b.iter(|| {
                    let mut executor = QueryExecutor::new();
                    black_box(executor.execute(&algebra, dataset).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark filter operations
fn bench_filter_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_operations");
    group.measurement_time(Duration::from_secs(10));

    let store = setup_test_store(1_000);
    let dataset = ConcreteStoreDataset::new(store);

    let filters = vec![
        (
            "equality",
            r#"SELECT ?s ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name . FILTER(?name = "Alice") }"#,
        ),
        (
            "regex",
            r#"SELECT ?s ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name . FILTER(REGEX(?name, "^A")) }"#,
        ),
        (
            "numeric",
            r#"SELECT ?s ?age WHERE { ?s <http://xmlns.com/foaf/0.1/age> ?age . FILTER(?age > 25) }"#,
        ),
        (
            "compound",
            r#"SELECT ?s ?name ?age WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name . ?s <http://xmlns.com/foaf/0.1/age> ?age . FILTER(?age > 20 && ?age < 40) }"#,
        ),
    ];

    for (name, query) in filters {
        group.bench_with_input(BenchmarkId::from_parameter(name), query, |b, query| {
            let mut parser = QueryParser::new();
            let query_ast = parser.parse(query).unwrap();
            let algebra = query_ast.where_clause;

            b.iter(|| {
                let mut executor = QueryExecutor::new();
                black_box(executor.execute(&algebra, &dataset).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark OPTIONAL pattern matching
fn bench_optional_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("optional_patterns");
    group.measurement_time(Duration::from_secs(10));

    let store = setup_test_store(1_000);
    let dataset = ConcreteStoreDataset::new(store);

    group.bench_function("simple_optional", |b| {
        let query = r#"
            SELECT ?s ?name ?email WHERE {
                ?s <http://xmlns.com/foaf/0.1/name> ?name .
                OPTIONAL { ?s <http://xmlns.com/foaf/0.1/mbox> ?email }
            }
        "#;
        let mut parser = QueryParser::new();
        let query_ast = parser.parse(query).unwrap();
        let algebra = query_ast.where_clause;

        b.iter(|| {
            let mut executor = QueryExecutor::new();
            black_box(executor.execute(&algebra, &dataset).unwrap());
        });
    });

    group.bench_function("nested_optional", |b| {
        let query = r#"
            SELECT ?s ?name ?email ?phone WHERE {
                ?s <http://xmlns.com/foaf/0.1/name> ?name .
                OPTIONAL {
                    ?s <http://xmlns.com/foaf/0.1/mbox> ?email .
                    OPTIONAL { ?s <http://xmlns.com/foaf/0.1/phone> ?phone }
                }
            }
        "#;
        let mut parser = QueryParser::new();
        let query_ast = parser.parse(query).unwrap();
        let algebra = query_ast.where_clause;

        b.iter(|| {
            let mut executor = QueryExecutor::new();
            black_box(executor.execute(&algebra, &dataset).unwrap());
        });
    });

    group.finish();
}

/// Benchmark UNION operations
fn bench_union_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("union_operations");
    group.measurement_time(Duration::from_secs(10));

    let store = setup_test_store(1_000);
    let dataset = ConcreteStoreDataset::new(store);

    group.bench_function("simple_union", |b| {
        let query = r#"
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?person WHERE {
                { ?person rdf:type <http://xmlns.com/foaf/0.1/Person> }
                UNION
                { ?person rdf:type <http://schema.org/Person> }
            }
        "#;
        let mut parser = QueryParser::new();
        let query_ast = parser.parse(query).unwrap();
        let algebra = query_ast.where_clause;

        b.iter(|| {
            let mut executor = QueryExecutor::new();
            black_box(executor.execute(&algebra, &dataset).unwrap());
        });
    });

    group.bench_function("multiple_union", |b| {
        let query = r#"
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?entity WHERE {
                { ?entity rdf:type <http://xmlns.com/foaf/0.1/Person> }
                UNION
                { ?entity rdf:type <http://xmlns.com/foaf/0.1/Organization> }
                UNION
                { ?entity rdf:type <http://schema.org/Thing> }
            }
        "#;
        let mut parser = QueryParser::new();
        let query_ast = parser.parse(query).unwrap();
        let algebra = query_ast.where_clause;

        b.iter(|| {
            let mut executor = QueryExecutor::new();
            black_box(executor.execute(&algebra, &dataset).unwrap());
        });
    });

    group.finish();
}

/// Benchmark aggregation operations
/// TODO: Support SPARQL 1.1 aggregation syntax with expressions in SELECT clause
/// Currently, the parser does not support syntax like (COUNT(?s) AS ?count)
/// or GROUP BY and HAVING clauses. These queries are commented out pending
/// full SPARQL 1.1 aggregation parser implementation.
fn bench_aggregation_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregation");
    group.measurement_time(Duration::from_secs(10));

    let store = setup_test_store(10_000);
    let dataset = ConcreteStoreDataset::new(store);

    // NOTE: The following SPARQL 1.1 aggregation queries are not yet supported:
    // - SELECT (COUNT(?s) AS ?count) WHERE { ?s ?p ?o }
    // - SELECT (COUNT(DISTINCT ?p) AS ?count) WHERE { ?s ?p ?o }
    // - SELECT ?p (COUNT(?s) AS ?count) WHERE { ?s ?p ?o } GROUP BY ?p
    // - SELECT ?p (COUNT(?s) AS ?count) WHERE { ?s ?p ?o } GROUP BY ?p HAVING(COUNT(?s) > 10)
    //
    // These require parser support for:
    // 1. Aggregate expressions in SELECT clause with AS bindings
    // 2. GROUP BY clause
    // 3. HAVING clause
    // 4. DISTINCT modifier in aggregates

    // Placeholder benchmark using basic queries until aggregation support is added
    group.bench_function("basic_count_query", |b| {
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100";
        let mut parser = QueryParser::new();
        let query_ast = parser.parse(query).unwrap();
        let algebra = query_ast.where_clause;

        b.iter(|| {
            let mut executor = QueryExecutor::new();
            black_box(executor.execute(&algebra, &dataset).unwrap());
        });
    });

    group.finish();
}

/// Benchmark different query forms
fn bench_query_forms(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_forms");
    group.measurement_time(Duration::from_secs(10));

    let store = setup_test_store(1_000);
    let dataset = ConcreteStoreDataset::new(store);

    group.bench_function("select", |b| {
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100";
        let mut parser = QueryParser::new();
        let query_ast = parser.parse(query).unwrap();
        let algebra = query_ast.where_clause;

        b.iter(|| {
            let mut executor = QueryExecutor::new();
            black_box(executor.execute(&algebra, &dataset).unwrap());
        });
    });

    group.bench_function("ask", |b| {
        let query = "ASK WHERE { ?s <http://xmlns.com/foaf/0.1/name> \"Alice\" }";
        let mut parser = QueryParser::new();
        let query_ast = parser.parse(query).unwrap();
        let algebra = query_ast.where_clause;

        b.iter(|| {
            let mut executor = QueryExecutor::new();
            black_box(executor.execute(&algebra, &dataset).unwrap());
        });
    });

    // NOTE: CONSTRUCT queries require full triple pattern support in the parser
    // Currently skipping CONSTRUCT benchmarks pending parser enhancements

    // NOTE: DESCRIBE queries are not yet fully supported by the parser
    // Currently skipping DESCRIBE benchmarks pending parser enhancements

    group.finish();
}

/// Benchmark query scalability
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_scalability");
    group.measurement_time(Duration::from_secs(15));

    let sizes = vec![1_000, 10_000, 100_000];

    for size in sizes {
        let store = setup_test_store(size);
        let dataset = ConcreteStoreDataset::new(store);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("full_scan", size),
            &dataset,
            |b, dataset| {
                let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
                let mut parser = QueryParser::new();
                let query_ast = parser.parse(query).unwrap();
                let algebra = query_ast.where_clause;

                b.iter(|| {
                    let mut executor = QueryExecutor::new();
                    black_box(executor.execute(&algebra, dataset).unwrap());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("selective_query", size),
            &dataset,
            |b, dataset| {
                let query = "SELECT ?s ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name . FILTER(?name = \"Alice0\") }";
                let mut parser = QueryParser::new();
                let query_ast = parser.parse(query).unwrap();
                let algebra = query_ast.where_clause;

                b.iter(|| {
                    let mut executor = QueryExecutor::new();
                    black_box(executor.execute(&algebra, dataset).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Helper: Setup test store with data
fn setup_test_store(size: usize) -> ConcreteStore {
    let store = ConcreteStore::new().unwrap();

    for i in 0..size {
        // Add person with name
        let subject = NamedNode::new(format!("http://example.org/person{i}")).unwrap();
        let name_pred = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
        let name_obj = Literal::new(format!("Alice{i}"));
        store
            .insert_quad(Quad::new_default_graph(
                subject.clone(),
                name_pred,
                name_obj,
            ))
            .unwrap();

        // Add email (50% of entities)
        if i % 2 == 0 {
            let email_pred = NamedNode::new("http://xmlns.com/foaf/0.1/mbox").unwrap();
            let email_obj = Literal::new(format!("alice{i}@example.org"));
            store
                .insert_quad(Quad::new_default_graph(
                    subject.clone(),
                    email_pred,
                    email_obj,
                ))
                .unwrap();
        }

        // Add age (70% of entities)
        if i % 10 < 7 {
            let age_pred = NamedNode::new("http://xmlns.com/foaf/0.1/age").unwrap();
            let age_obj = Literal::new(format!("{}", 20 + (i % 50)));
            store
                .insert_quad(Quad::new_default_graph(subject.clone(), age_pred, age_obj))
                .unwrap();
        }

        // Add type assertions
        let type_pred = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let type_obj = if i % 10 < 8 {
            NamedNode::new("http://xmlns.com/foaf/0.1/Person").unwrap()
        } else {
            NamedNode::new("http://schema.org/Person").unwrap()
        };
        store
            .insert_quad(Quad::new_default_graph(subject, type_pred, type_obj))
            .unwrap();
    }

    store
}

criterion_group!(
    benches,
    bench_query_parsing,
    bench_pattern_matching,
    bench_join_operations,
    bench_filter_operations,
    bench_optional_patterns,
    bench_union_operations,
    bench_aggregation_operations,
    bench_query_forms,
    bench_scalability
);
criterion_main!(benches);
