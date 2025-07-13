//! Comprehensive benchmarks for OxiRS Federation Engine
//!
//! This benchmark suite tests various aspects of the federation engine performance
//! including query planning, service discovery, result integration, and caching.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_federate::cache::{FederationCache, QueryResultCache};
use oxirs_federate::distributed_tracing::{
    DistributedTracingManager, QueryContext, QueryPriority, SpanStatus,
}; // Fix QueryContext import
use oxirs_federate::executor::types::{
    ExecutionStatus, QueryResultData, SparqlHead, SparqlResults, SparqlResultsData, SparqlValue,
    StepResult,
};
use oxirs_federate::integration::ResultIntegrator;
use oxirs_federate::network_optimizer::{EncodingFormat, NetworkOptimizer};
use oxirs_federate::planner::planning::types::StepType;
use oxirs_federate::planner::{PlannerConfig, QueryPlanner};
use oxirs_federate::request_batcher::{
    BatchableRequest, BatchingStrategy, RequestBatch, RequestBatcher, RequestPriority,
};
use oxirs_federate::service_registry::ServiceRegistry;
use oxirs_federate::source_selection::{
    AdvancedSourceSelector, SourceSelectionConfig, TriplePattern,
};
use oxirs_federate::*;
use std::collections::HashMap;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark service registration performance
fn bench_service_registration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("service_registration", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut registry = ServiceRegistry::new();

                // Register 100 services
                for i in 0..100 {
                    let service = FederatedService::new_sparql(
                        format!("service-{i}"),
                        format!("Test Service {i}"),
                        format!("http://example.com/sparql/{i}"),
                    );
                    registry.register(service).await.unwrap();
                }
            })
        });
    });
}

/// Benchmark query planning for different query complexities
fn bench_query_planning(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let queries = vec![
        ("simple", "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"),
        (
            "join",
            "SELECT ?s ?name ?friend WHERE { ?s foaf:name ?name . ?s foaf:knows ?friend }",
        ),
        (
            "complex",
            r#"
            SELECT ?person ?name ?age ?company ?title WHERE {
                ?person foaf:name ?name .
                ?person foaf:age ?age .
                ?person work:worksFor ?company .
                ?person work:hasTitle ?title .
                FILTER(?age > 25)
            }
        "#,
        ),
        (
            "union",
            r#"
            SELECT ?entity ?type WHERE {
                { ?entity a foaf:Person } UNION
                { ?entity a org:Organization } UNION 
                { ?entity a foaf:Document }
            }
        "#,
        ),
    ];

    let mut group = c.benchmark_group("query_planning");

    for (name, query) in queries {
        group.bench_with_input(BenchmarkId::new("analyze", name), query, |b, query| {
            b.iter(|| {
                rt.block_on(async {
                    let planner = QueryPlanner::new();
                    planner.analyze_sparql(query).await.unwrap()
                })
            });
        });
    }

    group.finish();
}

/// Benchmark source selection algorithms
fn bench_source_selection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("source_selection_large_registry", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut registry = ServiceRegistry::new();

                // Create 1000 services with different capabilities
                for i in 0..1000 {
                    let mut service = FederatedService::new_sparql(
                        format!("service-{i}"),
                        format!("Test Service {i}"),
                        format!("http://example.com/sparql/{i}"),
                    );

                    // Add random capabilities
                    if i % 2 == 0 {
                        service
                            .capabilities
                            .insert(ServiceCapability::FullTextSearch);
                    }
                    if i % 3 == 0 {
                        service.capabilities.insert(ServiceCapability::Geospatial);
                    }
                    if i % 5 == 0 {
                        service.capabilities.insert(ServiceCapability::RdfStar);
                    }

                    registry.register(service).await.unwrap();
                }

                // Benchmark source selection
                let planner = QueryPlanner::new();
                let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
                let query_info = planner.analyze_sparql(query).await.unwrap();

                // Use AdvancedSourceSelector for source selection
                let selector = AdvancedSourceSelector::new(SourceSelectionConfig::default());
                let patterns = vec![TriplePattern {
                    subject: "?s".to_string(),
                    predicate: "?p".to_string(),
                    object: "?o".to_string(),
                    graph: None,
                }];

                selector
                    .select_sources(&patterns, &[], &registry)
                    .await
                    .unwrap()
            })
        });
    });
}

/// Benchmark caching performance
fn bench_caching_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let cache_sizes = vec![100, 1000, 10000];

    let mut group = c.benchmark_group("caching");

    for size in cache_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("cache_operations", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let cache = FederationCache::new();

                        // Perform cache operations
                        let mut hits = 0;

                        for i in 0..size {
                            let query_hash = format!("query-{i}");

                            // Try to get from cache (should miss)
                            if cache.get_query_result(&query_hash).await.is_some() {
                                hits += 1;
                            }

                            // Put a result in cache
                            let sparql_result = create_sample_sparql_result(i, 1);
                            let cached_result = QueryResultCache::Sparql(sparql_result);
                            cache
                                .put_query_result(&query_hash, cached_result, None)
                                .await;

                            // Try to get from cache again (should hit)
                            if cache.get_query_result(&query_hash).await.is_some() {
                                hits += 1;
                            }
                        }

                        hits
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark result integration performance
fn bench_result_integration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let result_counts = vec![10, 100, 1000];

    let mut group = c.benchmark_group("result_integration");

    for count in result_counts {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("merge_sparql_results", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let integrator = ResultIntegrator::new();
                        let mut results = Vec::new();

                        // Create sample results
                        for i in 0..count {
                            let result = create_sample_sparql_result(i, 10); // 10 bindings per result
                            let step_result = StepResult {
                                step_id: format!("step-{i}"),
                                step_type: StepType::ServiceQuery,
                                status: ExecutionStatus::Success,
                                data: Some(QueryResultData::Sparql(result)),
                                error: None,
                                execution_time: Duration::from_millis(10),
                                service_id: Some(format!("service-{i}")),
                                memory_used: 1024,
                                result_size: 10,
                                success: true,
                                error_message: None,
                                service_response_time: Duration::from_millis(5),
                                cache_hit: false,
                            };
                            results.push(step_result);
                        }

                        // Fix method name and type
                        integrator.integrate_sparql_results(results).await.unwrap()
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark network optimization
fn bench_network_optimization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let data_sizes = vec![1024, 10240, 102400]; // 1KB, 10KB, 100KB

    let mut group = c.benchmark_group("network_optimization");

    for size in data_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("compression", size), &size, |b, &size| {
            b.iter(|| {
                rt.block_on(async {
                    let optimizer = NetworkOptimizer::new();
                    let test_data = vec![65u8; size]; // Create test data

                    let compressed = optimizer
                        .compress_data(&test_data, EncodingFormat::Json)
                        .await
                        .unwrap();

                    optimizer.decompress_data(&compressed).await.unwrap()
                })
            });
        });
    }

    group.finish();
}

/// Benchmark request batching
fn bench_request_batching(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let batch_sizes = vec![1, 10, 50, 100];

    let mut group = c.benchmark_group("request_batching");

    for size in batch_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_processing", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let batcher = RequestBatcher::new();
                        let registry = ServiceRegistry::new();

                        // Create batch requests
                        let mut batch = RequestBatch {
                            id: "bench-batch".to_string(),
                            service_id: "test-service".to_string(),
                            requests: Vec::new(),
                            created_at: std::time::Instant::now(),
                            batch_strategy: BatchingStrategy::SmallBatch { size },
                            estimated_processing_time: Duration::from_millis(50),
                        };

                        for i in 0..size {
                            let request = BatchableRequest {
                                id: format!("req-{i}"),
                                service_id: "test-service".to_string(),
                                query: format!("SELECT ?s ?p ?o WHERE {{ ?s ?p ?o{i} }}"),
                                variables: None,
                                priority: RequestPriority::Normal,
                                timestamp: std::time::Instant::now(),
                                timeout: None,
                                response_sender: None,
                            };
                            batch.requests.push(request);
                        }

                        batcher.execute_batch(batch, &registry).await.unwrap()
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark distributed tracing overhead
fn bench_distributed_tracing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("tracing_overhead", |b| {
        b.iter(|| {
            rt.block_on(async {
                let tracer = DistributedTracingManager::new();

                let query_context = QueryContext {
                    query: "SELECT ?s ?p ?o WHERE { ?s ?p ?o }".to_string(),
                    query_type: "SPARQL".to_string(),
                    user_id: Some("bench-user".to_string()),
                    session_id: Some("bench-session".to_string()),
                    priority: QueryPriority::Normal,
                };

                // Start trace
                let trace = tracer.start_trace(query_context).await.unwrap();

                // Create 10 spans
                let mut span_ids = Vec::new();
                for i in 0..10 {
                    let span = tracer
                        .create_span(&trace, &format!("operation-{i}"), "bench-service", None)
                        .await
                        .unwrap();
                    span_ids.push(span.span_id);
                }

                // Finish all spans
                for span_id in span_ids {
                    tracer.finish_span(&span_id, SpanStatus::Ok).await.unwrap();
                }

                // Complete trace
                tracer.complete_trace(&trace.trace_id).await.unwrap()
            })
        });
    });
}

// Helper functions

fn create_sample_sparql_result(offset: usize, binding_count: usize) -> SparqlResults {
    let mut bindings = Vec::new();

    for i in 0..binding_count {
        let mut binding = HashMap::new();
        binding.insert(
            "s".to_string(),
            SparqlValue {
                value_type: "uri".to_string(),
                value: format!("http://example.org/resource{}", offset * binding_count + i),
                datatype: None,
                lang: None,
                quoted_triple: None,
            },
        );
        binding.insert(
            "p".to_string(),
            SparqlValue {
                value_type: "uri".to_string(),
                value: "http://example.org/predicate".to_string(),
                datatype: None,
                lang: None,
                quoted_triple: None,
            },
        );
        binding.insert(
            "o".to_string(),
            SparqlValue {
                value_type: "literal".to_string(),
                value: format!("Object {i}"),
                datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                lang: None,
                quoted_triple: None,
            },
        );
        bindings.push(binding);
    }

    SparqlResults {
        head: SparqlHead {
            vars: vec!["s".to_string(), "p".to_string(), "o".to_string()],
        },
        results: SparqlResultsData { bindings },
    }
}

criterion_group!(
    benches,
    bench_service_registration,
    bench_query_planning,
    bench_source_selection,
    bench_caching_performance,
    bench_result_integration,
    bench_network_optimization,
    bench_request_batching,
    bench_distributed_tracing,
);

criterion_main!(benches);
