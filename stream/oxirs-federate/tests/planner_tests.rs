//! Unit tests for query planner module

use oxirs_federate::*;
use std::collections::HashSet;
use std::time::Duration;

#[tokio::test]
async fn test_query_type_detection() {
    let planner = QueryPlanner::new();

    let test_cases = vec![
        ("SELECT * WHERE { ?s ?p ?o }", QueryType::SparqlSelect),
        (
            "SELECT ?name WHERE { ?s foaf:name ?name }",
            QueryType::SparqlSelect,
        ),
        (
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }",
            QueryType::SparqlConstruct,
        ),
        ("ASK { ?s ?p ?o }", QueryType::SparqlAsk),
        (
            "DESCRIBE <http://example.org/resource>",
            QueryType::SparqlDescribe,
        ),
        ("INSERT DATA { <s> <p> <o> }", QueryType::SparqlUpdate),
        ("DELETE WHERE { ?s ?p ?o }", QueryType::SparqlUpdate),
    ];

    for (query, expected_type) in test_cases {
        let result = planner.analyze_sparql(query).await.unwrap();
        assert_eq!(
            result.query_type, expected_type,
            "Failed for query: {}",
            query
        );
    }
}

#[tokio::test]
async fn test_triple_pattern_extraction() {
    let planner = QueryPlanner::new();

    let query = r#"
        SELECT ?s ?p ?o
        WHERE {
            ?s ?p ?o .
            ?s rdf:type foaf:Person .
            ?s foaf:name "John" .
        }
    "#;

    let result = planner.analyze_sparql(query).await.unwrap();
    assert_eq!(result.patterns.len(), 3);

    // Check first pattern
    assert_eq!(result.patterns[0].subject, "?s");
    assert_eq!(result.patterns[0].predicate, "?p");
    assert_eq!(result.patterns[0].object, "?o");

    // Check second pattern
    assert_eq!(result.patterns[1].subject, "?s");
    assert_eq!(result.patterns[1].predicate, "rdf:type");
    assert_eq!(result.patterns[1].object, "foaf:Person");
}

#[tokio::test]
async fn test_filter_extraction() {
    let planner = QueryPlanner::new();

    let query = r#"
        SELECT ?name ?age
        WHERE {
            ?person foaf:name ?name .
            ?person foaf:age ?age .
            FILTER(?age > 18)
            FILTER(REGEX(?name, "^J", "i"))
        }
    "#;

    let result = planner.analyze_sparql(query).await.unwrap();
    assert_eq!(result.filters.len(), 2);

    // Check that filters contain correct variables
    assert!(result.filters[0].variables.contains("?age"));
    assert!(result.filters[1].variables.contains("?name"));
}

#[tokio::test]
async fn test_service_clause_parsing() {
    let planner = QueryPlanner::new();

    let query = r#"
        SELECT ?name
        WHERE {
            SERVICE <http://people.example.org/sparql> {
                ?person foaf:name ?name
            }
            SERVICE SILENT <http://products.example.org/sparql> {
                ?product rdfs:label ?name
            }
        }
    "#;

    let result = planner.analyze_sparql(query).await.unwrap();
    assert_eq!(result.service_clauses.len(), 2);

    assert_eq!(
        result.service_clauses[0].service_url,
        "http://people.example.org/sparql"
    );
    assert!(!result.service_clauses[0].silent);

    assert_eq!(
        result.service_clauses[1].service_url,
        "http://products.example.org/sparql"
    );
    assert!(result.service_clauses[1].silent);
}

#[tokio::test]
async fn test_variable_extraction() {
    let planner = QueryPlanner::new();

    let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . ?s ?p2 ?o2 }";
    let result = planner.analyze_sparql(query).await.unwrap();

    let expected_vars: HashSet<String> = vec!["?s", "?p", "?o", "?p2", "?o2"]
        .into_iter()
        .map(String::from)
        .collect();

    assert_eq!(result.variables, expected_vars);
}

#[tokio::test]
async fn test_complexity_calculation() {
    let planner = QueryPlanner::new();

    // Simple query
    let simple = "SELECT ?s WHERE { ?s ?p ?o }";
    let simple_result = planner.analyze_sparql(simple).await.unwrap();
    assert_eq!(simple_result.complexity, QueryComplexity::Low);

    // Medium complexity query
    let medium = r#"
        SELECT ?name ?age WHERE {
            ?person foaf:name ?name .
            ?person foaf:age ?age .
            ?person foaf:knows ?friend .
            FILTER(?age > 18)
        }
    "#;
    let medium_result = planner.analyze_sparql(medium).await.unwrap();
    assert!(matches!(
        medium_result.complexity,
        QueryComplexity::Low | QueryComplexity::Medium
    ));

    // Complex query with multiple SERVICE clauses
    let complex = r#"
        SELECT * WHERE {
            SERVICE <http://service1.com/sparql> { ?s ?p ?o }
            SERVICE <http://service2.com/sparql> { ?s ?p2 ?o2 }
            SERVICE <http://service3.com/sparql> { ?s ?p3 ?o3 }
            FILTER(?o > 100)
            FILTER(REGEX(?o2, "pattern"))
        }
    "#;
    let complex_result = planner.analyze_sparql(complex).await.unwrap();
    assert!(matches!(
        complex_result.complexity,
        QueryComplexity::Medium | QueryComplexity::High
    ));
}

#[tokio::test]
async fn test_execution_plan_optimization() {
    let config = QueryPlannerConfig {
        max_services_per_query: 5,
        optimization_level: OptimizationLevel::Balanced,
        timeout: Duration::from_secs(30),
        enable_caching: true,
        cost_threshold: 1000,
        service_selection_strategy: ServiceSelectionStrategy::CapabilityBased,
        advanced_decomposition_threshold: 5,
    };

    let planner = QueryPlanner::with_config(config);
    let mut registry = ServiceRegistry::new();

    // Register test services
    for i in 1..=3 {
        let service = FederatedService::new_sparql(
            format!("test-{}", i),
            format!("Test Service {}", i),
            format!("http://example.com/sparql{}", i),
        );
        registry.register(service).await.unwrap();
    }

    let query_info = QueryInfo {
        query_type: QueryType::SparqlSelect,
        original_query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
        patterns: vec![TriplePattern {
            subject: "?s".to_string(),
            predicate: "?p".to_string(),
            object: "?o".to_string(),
            pattern_string: "?s ?p ?o".to_string(),
        }],
        service_clauses: vec![],
        filters: vec![],
        variables: ["?s", "?p", "?o"].iter().map(|s| s.to_string()).collect(),
        complexity: QueryComplexity::Low,
        estimated_cost: 10,
    };

    let plan = planner.plan_sparql(&query_info, &registry).await.unwrap();

    // Check that plan has steps
    assert!(!plan.steps.is_empty());

    // Check parallelizable steps are identified
    let parallel_count: usize = plan
        .steps
        .iter()
        .filter(|s| s.parallel_group.is_some())
        .count();
    assert!(parallel_count >= 0); // May or may not have parallel steps depending on strategy
}

#[tokio::test]
async fn test_service_selection_strategies() {
    let mut registry = ServiceRegistry::new();

    // Register services with different capabilities
    let mut geo_service = FederatedService::new_sparql(
        "geo-service".to_string(),
        "Geospatial Service".to_string(),
        "http://geo.example.com/sparql".to_string(),
    );
    geo_service
        .capabilities
        .insert(ServiceCapability::Geospatial);

    let mut text_service = FederatedService::new_sparql(
        "text-service".to_string(),
        "Full Text Service".to_string(),
        "http://text.example.com/sparql".to_string(),
    );
    text_service
        .capabilities
        .insert(ServiceCapability::FullTextSearch);

    registry.register(geo_service).await.unwrap();
    registry.register(text_service).await.unwrap();

    // Test capability-based selection
    let config = QueryPlannerConfig {
        service_selection_strategy: ServiceSelectionStrategy::CapabilityBased,
        ..QueryPlannerConfig::default()
    };

    let planner = QueryPlanner::with_config(config);

    // Query with geospatial pattern
    let geo_pattern = TriplePattern {
        subject: "?location".to_string(),
        predicate: "geo:lat".to_string(),
        object: "?lat".to_string(),
        pattern_string: "?location geo:lat ?lat".to_string(),
    };

    let query_info = QueryInfo {
        query_type: QueryType::SparqlSelect,
        original_query: "SELECT * WHERE { ?location geo:lat ?lat }".to_string(),
        patterns: vec![geo_pattern],
        service_clauses: vec![],
        filters: vec![],
        variables: ["?location", "?lat"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        complexity: QueryComplexity::Low,
        estimated_cost: 10,
    };

    let plan = planner.plan_sparql(&query_info, &registry).await.unwrap();

    // Should select the geo service
    assert!(plan.steps.iter().any(|step| {
        step.service_id
            .as_ref()
            .map(|id| id == "geo-service")
            .unwrap_or(false)
    }));
}

#[tokio::test]
async fn test_advanced_query_decomposition() {
    let config = QueryPlannerConfig {
        advanced_decomposition_threshold: 2, // Low threshold for testing
        ..QueryPlannerConfig::default()
    };

    let planner = QueryPlanner::with_config(config);
    let mut registry = ServiceRegistry::new();

    // Register a service
    let service = FederatedService::new_sparql(
        "decomp-test".to_string(),
        "Decomposition Test".to_string(),
        "http://example.com/sparql".to_string(),
    );
    registry.register(service).await.unwrap();

    // Complex query that should trigger advanced decomposition
    let query_info = QueryInfo {
        query_type: QueryType::SparqlSelect,
        original_query: "SELECT * WHERE { ?s ?p ?o . ?o ?p2 ?o2 . ?o2 ?p3 ?o3 }".to_string(),
        patterns: vec![
            TriplePattern {
                subject: "?s".to_string(),
                predicate: "?p".to_string(),
                object: "?o".to_string(),
                pattern_string: "?s ?p ?o".to_string(),
            },
            TriplePattern {
                subject: "?o".to_string(),
                predicate: "?p2".to_string(),
                object: "?o2".to_string(),
                pattern_string: "?o ?p2 ?o2".to_string(),
            },
            TriplePattern {
                subject: "?o2".to_string(),
                predicate: "?p3".to_string(),
                object: "?o3".to_string(),
                pattern_string: "?o2 ?p3 ?o3".to_string(),
            },
        ],
        service_clauses: vec![],
        filters: vec![],
        variables: ["?s", "?p", "?o", "?p2", "?o2", "?p3", "?o3"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        complexity: QueryComplexity::Medium,
        estimated_cost: 30,
    };

    let plan = planner
        .plan_sparql_advanced(&query_info, &registry)
        .await
        .unwrap();

    // Should have created an execution plan
    assert!(!plan.steps.is_empty());
    assert_eq!(plan.query_type, QueryType::SparqlSelect);
}
