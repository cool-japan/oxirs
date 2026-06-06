use super::service_delegation_executor::{ParallelServiceExecutor, ServiceResultMerger};
use super::service_delegation_manager::ServiceDelegationManager;
use super::service_delegation_types::{
    MergeStrategy, ResponseStatus, ServiceEndpoint, ServiceEndpointInfo, ServiceHealth,
    ServiceQueryRequest, ServiceQueryResponse,
};
use std::collections::HashMap;
use std::time::Duration;

#[tokio::test]
async fn test_service_clause_extraction() {
    let manager = ServiceDelegationManager::new();
    let query = "SELECT ?s WHERE { SERVICE <http://example.org/sparql> { ?s ?p ?o } }";
    let clauses = manager.extract_service_clauses(query).unwrap();
    assert_eq!(clauses.len(), 1);
    assert!(clauses[0].contains("SERVICE <http://example.org/sparql>"));
}

#[tokio::test]
async fn test_endpoint_registration() {
    let manager = ServiceDelegationManager::new();
    let endpoint = ServiceEndpoint {
        url: "http://example.org/sparql".to_string(),
        name: "test-endpoint".to_string(),
        supported_features: std::collections::HashSet::new(),
        authentication: None,
        timeout: Duration::from_secs(30),
        retry_count: 3,
        health_status: ServiceHealth::Healthy,
        response_time_avg: None,
        last_checked: None,
    };
    manager.register_endpoint(endpoint).await.unwrap();
    let health = manager
        .get_endpoint_health("http://example.org/sparql")
        .await;
    assert_eq!(health, Some(ServiceHealth::Healthy));
}

#[tokio::test]
async fn test_parallel_execution() {
    let executor = ParallelServiceExecutor::new();
    let requests = vec![
        ServiceQueryRequest {
            service_url: "http://example1.org/sparql".to_string(),
            query: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
            parameters: HashMap::new(),
            timeout: None,
            headers: HashMap::new(),
        },
        ServiceQueryRequest {
            service_url: "http://example2.org/sparql".to_string(),
            query: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
            parameters: HashMap::new(),
            timeout: None,
            headers: HashMap::new(),
        },
    ];
    let responses = executor.execute_parallel(requests).await.unwrap();
    assert_eq!(responses.len(), 2);
}

#[tokio::test]
async fn test_result_merging() {
    let merger = ServiceResultMerger::new();
    let responses = vec![
        ServiceQueryResponse {
            status: ResponseStatus::Success,
            results: Some(serde_json::json!({
                "head": { "vars": ["s"] },
                "results": { "bindings": [{"s": {"type": "uri", "value": "http://example.org/1"}}] }
            })),
            error_message: None,
            execution_time: Duration::from_millis(100),
            endpoint_info: ServiceEndpointInfo {
                url: "http://example1.org/sparql".to_string(),
                response_time: Duration::from_millis(100),
                attempt_count: 1,
            },
        },
        ServiceQueryResponse {
            status: ResponseStatus::Success,
            results: Some(serde_json::json!({
                "head": { "vars": ["s"] },
                "results": { "bindings": [{"s": {"type": "uri", "value": "http://example.org/2"}}] }
            })),
            error_message: None,
            execution_time: Duration::from_millis(150),
            endpoint_info: ServiceEndpointInfo {
                url: "http://example2.org/sparql".to_string(),
                response_time: Duration::from_millis(150),
                attempt_count: 1,
            },
        },
    ];
    let merged = merger
        .merge_results(responses, Some(MergeStrategy::Union))
        .await
        .unwrap();
    let bindings = merged["results"]["bindings"].as_array().unwrap();
    assert_eq!(bindings.len(), 2);
}
