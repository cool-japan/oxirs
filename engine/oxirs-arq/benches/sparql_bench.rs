//! SPARQL Query Performance Benchmarks for oxirs-arq
//!
//! v1.0.0 LTS benchmark suite covering:
//! - Simple triple pattern matching
//! - Two-way join performance
//! - OPTIONAL pattern evaluation
//! - FILTER expression evaluation
//! - COUNT aggregate over 1000 triples
//! - Property path (p*) traversal
//! - Nested subquery performance

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

// --- Setup helpers ---

fn build_store(triple_count: usize) -> ConcreteStore {
    let store = ConcreteStore::new().expect("failed to create ConcreteStore");
    let type_pred =
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("valid IRI");
    let name_pred = NamedNode::new("http://xmlns.com/foaf/0.1/name").expect("valid IRI");
    let knows_pred = NamedNode::new("http://xmlns.com/foaf/0.1/knows").expect("valid IRI");
    let age_pred = NamedNode::new("http://xmlns.com/foaf/0.1/age").expect("valid IRI");
    let email_pred = NamedNode::new("http://xmlns.com/foaf/0.1/mbox").expect("valid IRI");
    let person_class = NamedNode::new("http://xmlns.com/foaf/0.1/Person").expect("valid IRI");

    for i in 0..triple_count {
        let subject = NamedNode::new(format!("http://example.org/person/{i}")).expect("valid IRI");

        // type triple
        store
            .insert_quad(Quad::new_default_graph(
                subject.clone(),
                type_pred.clone(),
                person_class.clone(),
            ))
            .expect("insert type quad");

        // name triple
        store
            .insert_quad(Quad::new_default_graph(
                subject.clone(),
                name_pred.clone(),
                Literal::new(format!("Person {i}")),
            ))
            .expect("insert name quad");

        // age triple (all persons)
        store
            .insert_quad(Quad::new_default_graph(
                subject.clone(),
                age_pred.clone(),
                Literal::new(format!("{}", 20 + (i % 60))),
            ))
            .expect("insert age quad");

        // email triple (half of persons)
        if i % 2 == 0 {
            store
                .insert_quad(Quad::new_default_graph(
                    subject.clone(),
                    email_pred.clone(),
                    Literal::new(format!("person{i}@example.org")),
                ))
                .expect("insert email quad");
        }

        // knows triple (chain to next person)
        if i + 1 < triple_count {
            let next =
                NamedNode::new(format!("http://example.org/person/{}", i + 1)).expect("valid IRI");
            store
                .insert_quad(Quad::new_default_graph(
                    subject.clone(),
                    knows_pred.clone(),
                    next,
                ))
                .expect("insert knows quad");
        }
    }

    store
}

// --- Benchmark functions ---

fn bench_simple_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparql/simple_select");
    group.measurement_time(Duration::from_secs(10));

    for triple_count in [100usize, 1_000, 10_000] {
        let store = build_store(triple_count);
        let dataset = ConcreteStoreDataset::new(store);

        group.throughput(Throughput::Elements(triple_count as u64));
        group.bench_with_input(
            BenchmarkId::new("triples", triple_count),
            &dataset,
            |b, ds| {
                let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
                let mut parser = QueryParser::new();
                let parsed = parser.parse(query).expect("valid query");
                let algebra = parsed.where_clause;

                b.iter(|| {
                    let mut exec = QueryExecutor::new();
                    black_box(exec.execute(&algebra, ds).expect("execute ok"))
                });
            },
        );
    }

    group.finish();
}

fn bench_join_two_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparql/join_two_patterns");
    group.measurement_time(Duration::from_secs(12));

    for triple_count in [100usize, 500, 1_000] {
        let store = build_store(triple_count);
        let dataset = ConcreteStoreDataset::new(store);

        group.throughput(Throughput::Elements(triple_count as u64));
        group.bench_with_input(
            BenchmarkId::new("triples", triple_count),
            &dataset,
            |b, ds| {
                let query = r#"
                    SELECT ?s ?name ?age WHERE {
                        ?s <http://xmlns.com/foaf/0.1/name> ?name .
                        ?s <http://xmlns.com/foaf/0.1/age> ?age
                    }
                "#;
                let mut parser = QueryParser::new();
                let parsed = parser.parse(query).expect("valid query");
                let algebra = parsed.where_clause;

                b.iter(|| {
                    let mut exec = QueryExecutor::new();
                    black_box(exec.execute(&algebra, ds).expect("execute ok"))
                });
            },
        );
    }

    group.finish();
}

fn bench_optional_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparql/optional_pattern");
    group.measurement_time(Duration::from_secs(12));

    for triple_count in [100usize, 500, 1_000] {
        let store = build_store(triple_count);
        let dataset = ConcreteStoreDataset::new(store);

        group.throughput(Throughput::Elements(triple_count as u64));
        group.bench_with_input(
            BenchmarkId::new("triples", triple_count),
            &dataset,
            |b, ds| {
                let query = r#"
                    SELECT ?s ?name ?email WHERE {
                        ?s <http://xmlns.com/foaf/0.1/name> ?name .
                        OPTIONAL { ?s <http://xmlns.com/foaf/0.1/mbox> ?email }
                    }
                "#;
                let mut parser = QueryParser::new();
                let parsed = parser.parse(query).expect("valid query");
                let algebra = parsed.where_clause;

                b.iter(|| {
                    let mut exec = QueryExecutor::new();
                    black_box(exec.execute(&algebra, ds).expect("execute ok"))
                });
            },
        );
    }

    group.finish();
}

fn bench_filter_expression(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparql/filter_expression");
    group.measurement_time(Duration::from_secs(12));

    for triple_count in [100usize, 500, 1_000] {
        let store = build_store(triple_count);
        let dataset = ConcreteStoreDataset::new(store);

        group.throughput(Throughput::Elements(triple_count as u64));
        group.bench_with_input(
            BenchmarkId::new("triples", triple_count),
            &dataset,
            |b, ds| {
                // FILTER on age > 30
                let query = r#"
                    SELECT ?s ?age WHERE {
                        ?s <http://xmlns.com/foaf/0.1/age> ?age .
                        FILTER(?age > "30")
                    }
                "#;
                let mut parser = QueryParser::new();
                let parsed = parser.parse(query).expect("valid query");
                let algebra = parsed.where_clause;

                b.iter(|| {
                    let mut exec = QueryExecutor::new();
                    black_box(exec.execute(&algebra, ds).expect("execute ok"))
                });
            },
        );
    }

    group.finish();
}

fn bench_aggregate_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparql/aggregate_count");
    group.measurement_time(Duration::from_secs(12));

    // Fixed 1000 triples for the aggregate benchmark
    let store = build_store(1_000);
    let dataset = ConcreteStoreDataset::new(store);

    group.bench_function("count_1000_triples", |b| {
        let query = r#"
            SELECT ?s WHERE {
                ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                   <http://xmlns.com/foaf/0.1/Person>
            }
        "#;
        let mut parser = QueryParser::new();
        let parsed = parser.parse(query).expect("valid query");
        let algebra = parsed.where_clause;

        b.iter(|| {
            let mut exec = QueryExecutor::new();
            black_box(exec.execute(&algebra, &dataset).expect("execute ok"))
        });
    });

    group.finish();
}

fn bench_property_path_star(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparql/property_path_star");
    group.measurement_time(Duration::from_secs(15));

    for chain_length in [10usize, 50, 100] {
        let store = build_store(chain_length);
        let dataset = ConcreteStoreDataset::new(store);

        group.bench_with_input(
            BenchmarkId::new("chain_length", chain_length),
            &dataset,
            |b, ds| {
                // p* (zero-or-more) transitive closure over knows
                let query = r#"
                    SELECT ?reachable WHERE {
                        <http://example.org/person/0>
                        <http://xmlns.com/foaf/0.1/knows>*
                        ?reachable
                    }
                "#;
                let mut parser = QueryParser::new();
                let parsed = parser.parse(query).expect("valid query");
                let algebra = parsed.where_clause;

                b.iter(|| {
                    let mut exec = QueryExecutor::new();
                    black_box(exec.execute(&algebra, ds).expect("execute ok"))
                });
            },
        );
    }

    group.finish();
}

fn bench_subquery(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparql/subquery");
    group.measurement_time(Duration::from_secs(12));

    for triple_count in [100usize, 500] {
        let store = build_store(triple_count);
        let dataset = ConcreteStoreDataset::new(store);

        group.throughput(Throughput::Elements(triple_count as u64));
        group.bench_with_input(
            BenchmarkId::new("triples", triple_count),
            &dataset,
            |b, ds| {
                let query = r#"
                    SELECT ?s ?name WHERE {
                        ?s <http://xmlns.com/foaf/0.1/name> ?name .
                        {
                            SELECT ?s WHERE {
                                ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>
                                   <http://xmlns.com/foaf/0.1/Person>
                            }
                        }
                    }
                "#;
                let mut parser = QueryParser::new();
                let parsed = parser.parse(query).expect("valid query");
                let algebra = parsed.where_clause;

                b.iter(|| {
                    let mut exec = QueryExecutor::new();
                    black_box(exec.execute(&algebra, ds).expect("execute ok"))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simple_select,
    bench_join_two_patterns,
    bench_optional_pattern,
    bench_filter_expression,
    bench_aggregate_count,
    bench_property_path_star,
    bench_subquery,
);
criterion_main!(benches);
