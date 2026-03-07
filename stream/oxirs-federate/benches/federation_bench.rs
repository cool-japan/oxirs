//! Federation Performance Benchmarks for oxirs-federate
//!
//! v1.0.0 LTS benchmark suite covering:
//! - SPARQL query decomposition into per-endpoint subqueries
//! - Cost model evaluation
//! - Join order optimization for 5-way join
//! - Result merging from multiple endpoints

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_federate::query_rewrite::{
    CostEstimator, EndpointInfo, EndpointSubquery, FederationOptimizer, QueryDecomposer,
};
use std::time::Duration;

// --- Helpers ---

fn make_endpoints(count: usize) -> Vec<EndpointInfo> {
    let namespaces = [
        "foaf", "schema", "owl", "skos", "dc", "dcterms", "prov", "geo",
    ];
    (0..count)
        .map(|i| {
            EndpointInfo::new(format!("http://endpoint{i}.example.org/sparql"))
                .with_affinity(namespaces[i % namespaces.len()])
        })
        .collect()
}

fn make_subquery(
    endpoint_index: usize,
    estimated_results: usize,
    priority: f64,
) -> EndpointSubquery {
    EndpointSubquery {
        endpoint_url: format!("http://endpoint{endpoint_index}.example.org/sparql"),
        sparql: format!(
            "SELECT ?s ?p ?o WHERE {{ ?s <http://example.org/p/{endpoint_index}> ?o }}",
        ),
        estimated_results,
        priority,
    }
}

// --- Benchmarks ---

fn bench_query_decompose(c: &mut Criterion) {
    let mut group = c.benchmark_group("federation/query_decompose");
    group.measurement_time(Duration::from_secs(10));

    // Varying endpoint counts
    for endpoint_count in [2usize, 4, 8] {
        let endpoints = make_endpoints(endpoint_count);
        let decomposer = QueryDecomposer::new(endpoints);

        group.bench_with_input(
            BenchmarkId::new("endpoints", endpoint_count),
            &endpoint_count,
            |b, _| {
                let query = r#"
                    SELECT ?person ?name ?org ?topic WHERE {
                        ?person a foaf:Person .
                        ?person foaf:name ?name .
                        ?person schema:memberOf ?org .
                        ?org dc:subject ?topic
                    }
                "#;
                b.iter(|| {
                    black_box(
                        decomposer
                            .decompose(black_box(query))
                            .expect("decompose ok"),
                    )
                });
            },
        );
    }

    // Simple 2-pattern query
    group.bench_function("simple_2_pattern", |b| {
        let endpoints = make_endpoints(2);
        let decomposer = QueryDecomposer::new(endpoints);
        let query = "SELECT ?s ?name WHERE { ?s a foaf:Person . ?s foaf:name ?name }";
        b.iter(|| {
            black_box(
                decomposer
                    .decompose(black_box(query))
                    .expect("decompose ok"),
            )
        });
    });

    // Complex 6-pattern query
    group.bench_function("complex_6_pattern", |b| {
        let endpoints = make_endpoints(4);
        let decomposer = QueryDecomposer::new(endpoints);
        let query = r#"
            SELECT ?p ?name ?email ?org ?city ?country WHERE {
                ?p a foaf:Person .
                ?p foaf:name ?name .
                ?p foaf:mbox ?email .
                ?p schema:worksFor ?org .
                ?org schema:addressLocality ?city .
                ?org schema:addressCountry ?country
            }
        "#;
        b.iter(|| {
            black_box(
                decomposer
                    .decompose(black_box(query))
                    .expect("decompose ok"),
            )
        });
    });

    group.finish();
}

fn bench_cost_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("federation/cost_estimation");
    group.measurement_time(Duration::from_secs(10));

    let estimator = CostEstimator::new();

    let subquery = make_subquery(0, 1000, 1.0);

    group.bench_function("single_subquery", |b| {
        b.iter(|| black_box(estimator.estimate_cost(black_box(&subquery))));
    });

    group.bench_function("join_cost_estimation", |b| {
        b.iter(|| black_box(estimator.estimate_join_cost(black_box(1000), black_box(500))));
    });

    group.bench_function("join_cost_large", |b| {
        b.iter(|| black_box(estimator.estimate_join_cost(black_box(100_000), black_box(50_000))));
    });

    // Estimate costs across multiple subqueries
    let subqueries: Vec<EndpointSubquery> = (0..10)
        .map(|i| make_subquery(i % 4, (i + 1) * 100, (i % 3) as f64))
        .collect();

    group.throughput(Throughput::Elements(subqueries.len() as u64));
    group.bench_function("batch_10_subqueries", |b| {
        b.iter(|| {
            let total: f64 = subqueries
                .iter()
                .map(|sq| estimator.estimate_cost(black_box(sq)))
                .sum();
            black_box(total)
        });
    });

    group.finish();
}

fn bench_join_order_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("federation/join_order_optimization");
    group.measurement_time(Duration::from_secs(12));

    let optimizer = FederationOptimizer::new();
    let endpoints = make_endpoints(3);
    let decomposer = QueryDecomposer::new(endpoints);

    // 5-way join query
    let five_way_query = r#"
        SELECT ?person ?name ?email ?org ?city WHERE {
            ?person a foaf:Person .
            ?person foaf:name ?name .
            ?person foaf:mbox ?email .
            ?person schema:worksFor ?org .
            ?org schema:addressLocality ?city
        }
    "#;

    group.bench_function("5_way_join_optimization", |b| {
        b.iter(|| {
            let federated = decomposer
                .decompose(black_box(five_way_query))
                .expect("decompose ok");
            black_box(
                optimizer
                    .optimize(black_box(federated))
                    .expect("optimize ok"),
            )
        });
    });

    // 3-way join for comparison
    let three_way_query = r#"
        SELECT ?person ?name ?email WHERE {
            ?person a foaf:Person .
            ?person foaf:name ?name .
            ?person foaf:mbox ?email
        }
    "#;

    group.bench_function("3_way_join_optimization", |b| {
        b.iter(|| {
            let federated = decomposer
                .decompose(black_box(three_way_query))
                .expect("decompose ok");
            black_box(
                optimizer
                    .optimize(black_box(federated))
                    .expect("optimize ok"),
            )
        });
    });

    // Vary endpoint counts
    for endpoint_count in [2usize, 4, 8] {
        let eps = make_endpoints(endpoint_count);
        let d = QueryDecomposer::new(eps);

        group.bench_with_input(
            BenchmarkId::new("endpoint_count", endpoint_count),
            &endpoint_count,
            |b, _| {
                b.iter(|| {
                    let federated = d
                        .decompose(black_box(five_way_query))
                        .expect("decompose ok");
                    black_box(
                        optimizer
                            .optimize(black_box(federated))
                            .expect("optimize ok"),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_result_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("federation/result_merge");
    group.measurement_time(Duration::from_secs(10));

    let optimizer = FederationOptimizer::new();

    let query = r#"
        SELECT ?s ?p ?o WHERE {
            ?s ?p ?o .
            ?s foaf:name ?name .
            ?s schema:knows ?friend
        }
    "#;

    for endpoint_count in [2usize, 3, 5] {
        let eps = make_endpoints(endpoint_count);
        let d = QueryDecomposer::new(eps);

        group.bench_with_input(
            BenchmarkId::new("merge_from_endpoints", endpoint_count),
            &endpoint_count,
            |b, _| {
                b.iter(|| {
                    let federated = d.decompose(black_box(query)).expect("decompose ok");
                    let plan = optimizer
                        .optimize(black_box(federated))
                        .expect("optimize ok");
                    // Count total subqueries as result merge proxy
                    black_box(plan.endpoint_subqueries().len())
                });
            },
        );
    }

    // Benchmark plan summary generation
    let endpoints = make_endpoints(3);
    let d = QueryDecomposer::new(endpoints);
    let federated = d.decompose(query).expect("decompose ok");
    let plan = optimizer.optimize(federated).expect("optimize ok");

    group.bench_function("plan_summary_generation", |b| {
        b.iter(|| black_box(plan.summary()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_query_decompose,
    bench_cost_estimation,
    bench_join_order_optimization,
    bench_result_merge,
);
criterion_main!(benches);
