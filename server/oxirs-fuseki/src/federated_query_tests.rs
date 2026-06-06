//! Tests for the federated SPARQL query optimizer.
//!
//! Exercises `FederatedQueryOptimizer::extract_service_patterns`,
//! `QueryPlanner::create_execution_plan`, and `ResultMerger::merge_results`
//! using the public APIs of the sibling `federated_query_*` modules.

#![cfg(test)]

use crate::{
    federated_query_executor::FederatedQueryOptimizer,
    federated_query_types::{
        QueryPlanner, QueryResults, ResultMerger, ResultMetadata, ServicePattern,
    },
    metrics::MetricsService,
};
use std::{collections::HashMap, sync::Arc};

fn monitoring_config() -> crate::config::MonitoringConfig {
    crate::config::MonitoringConfig {
        metrics: crate::config::MetricsConfig {
            enabled: false,
            endpoint: "/metrics".to_string(),
            port: Some(9000),
            namespace: "oxirs_fuseki".to_string(),
            collect_system_metrics: true,
            histogram_buckets: vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ],
        },
        health_checks: crate::config::HealthCheckConfig {
            enabled: false,
            interval_secs: 30,
            timeout_secs: 5,
            checks: vec!["store".to_string(), "memory".to_string()],
        },
        tracing: crate::config::TracingConfig {
            enabled: false,
            endpoint: None,
            service_name: "oxirs-fuseki".to_string(),
            sample_rate: 0.1,
            output: crate::config::TracingOutput::Stdout,
        },
        prometheus: Some(crate::config::PrometheusConfig {
            enabled: false,
            endpoint: "/metrics".to_string(),
            port: Some(9090),
            namespace: "oxirs_fuseki".to_string(),
            job_name: "oxirs-fuseki".to_string(),
            instance: "localhost:3030".to_string(),
            scrape_interval_secs: 15,
            timeout_secs: 10,
        }),
    }
}

#[tokio::test]
async fn test_extract_service_patterns() {
    let config = monitoring_config();
    let optimizer = FederatedQueryOptimizer::new(Arc::new(MetricsService::new(config).unwrap()));

    let query = r#"
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?person ?name ?friend
        WHERE {
            ?person foaf:name ?name .
            SERVICE <http://example.org/sparql> {
                ?person foaf:knows ?friend .
            }
        }
    "#;

    let patterns = optimizer.extract_service_patterns(query).unwrap();
    assert_eq!(patterns.len(), 1);
    assert_eq!(patterns[0].service_url, "http://example.org/sparql");
    assert!(!patterns[0].is_silent);
}

#[tokio::test]
async fn test_multiple_service_patterns() {
    let config = monitoring_config();
    let optimizer = FederatedQueryOptimizer::new(Arc::new(MetricsService::new(config).unwrap()));

    let query = r#"
        SELECT ?s ?p ?o
        WHERE {
            SERVICE <http://endpoint1.org/sparql> {
                ?s ?p ?o
            }
            SERVICE SILENT <http://endpoint2.org/sparql> {
                ?s ?p2 ?o2
            }
        }
    "#;

    let patterns = optimizer.extract_service_patterns(query).unwrap();
    assert_eq!(patterns.len(), 2);
    assert!(!patterns[0].is_silent);
    assert!(patterns[1].is_silent);
}

#[tokio::test]
async fn test_query_planner() {
    let planner = QueryPlanner::new();
    let service_patterns = vec![ServicePattern {
        service_url: "http://test.org/sparql".to_string(),
        pattern: "?s ?p ?o".to_string(),
        is_silent: false,
        is_optional: false,
    }];

    let plan = planner
        .create_execution_plan("SELECT * WHERE { ?s ?p ?o }", &service_patterns)
        .await
        .unwrap();
    assert!(!plan.fragments.is_empty());
}

#[tokio::test]
async fn test_result_merger_union() {
    let merger = ResultMerger::new();

    let results = vec![
        QueryResults {
            bindings: vec![HashMap::from([(
                "x".to_string(),
                serde_json::json!("value1"),
            )])],
            metadata: ResultMetadata {
                total_execution_time_ms: 100,
                endpoint_times: HashMap::new(),
                result_count: 1,
                partial_results: false,
            },
        },
        QueryResults {
            bindings: vec![HashMap::from([(
                "x".to_string(),
                serde_json::json!("value2"),
            )])],
            metadata: ResultMetadata {
                total_execution_time_ms: 150,
                endpoint_times: HashMap::new(),
                result_count: 1,
                partial_results: false,
            },
        },
    ];

    let merged = merger.merge_results(results).await.unwrap();
    assert_eq!(merged.bindings.len(), 2);
    assert_eq!(merged.metadata.total_execution_time_ms, 250);
}
