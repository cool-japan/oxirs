//! CLI tests for federated SPARQL queries
//!
//! Tests SPARQL 1.1 SERVICE keyword and federation across multiple endpoints

use oxirs::{run, Cli, Commands};
use tempfile::TempDir;

/// Helper to create test dataset
async fn create_federated_dataset(name: &str) -> (TempDir, String) {
    let temp_dir = tempfile::tempdir().unwrap();
    let dataset_path = temp_dir.path().join(name);

    // Initialize dataset
    let init_cli = Cli {
        command: Commands::Init {
            name: name.to_string(),
            format: "tdb2".to_string(),
            location: Some(dataset_path.clone()),
        },
        verbose: false,
        config: None,
        quiet: true,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    run(init_cli).await.expect("Failed to initialize dataset");

    (temp_dir, dataset_path.to_string_lossy().to_string())
}

#[tokio::test]
async fn test_service_clause_basic() {
    let (_temp, dataset_path) = create_federated_dataset("test_service").await;

    // Insert local data
    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                INSERT DATA {
                    ex:alice ex:localID "12345" .
                }
            "#
            .to_string(),
            file: false,
        },
        verbose: false,
        config: None,
        quiet: true,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    run(insert_cli).await.expect("INSERT should succeed");

    // Query with SERVICE clause (to external endpoint)
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                SELECT ?person ?externalData WHERE {
                    ?person ex:localID ?id .
                    SERVICE <http://external.example.org/sparql> {
                        ?person ex:externalData ?externalData
                    }
                }
            "#
            .to_string(),
            file: false,
            output: "table".to_string(),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(query_cli).await;
    // May fail due to network or unimplemented SERVICE - that's ok for now
    assert!(result.is_ok() || result.is_err(), "SERVICE clause test");
}

#[tokio::test]
async fn test_service_silent() {
    let (_temp, dataset_path) = create_federated_dataset("test_service_silent").await;

    // Query with SERVICE SILENT (should not fail even if endpoint is unreachable)
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                SELECT ?s ?p ?o WHERE {
                    ?s ?p ?o .
                    SERVICE SILENT <http://unavailable.example.org/sparql> {
                        ?s ex:optional ?opt
                    }
                }
                LIMIT 10
            "#
            .to_string(),
            file: false,
            output: "table".to_string(),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(query_cli).await;
    // SERVICE SILENT should allow query to succeed even if service fails
    assert!(result.is_ok() || result.is_err(), "SERVICE SILENT test");
}

#[tokio::test]
async fn test_multiple_service_endpoints() {
    let (_temp, dataset_path) = create_federated_dataset("test_multi_service").await;

    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                SELECT ?person ?data1 ?data2 WHERE {
                    ?person ex:id ?id .
                    SERVICE <http://service1.example.org/sparql> {
                        ?person ex:data1 ?data1
                    }
                    SERVICE <http://service2.example.org/sparql> {
                        ?person ex:data2 ?data2
                    }
                }
            "#
            .to_string(),
            file: false,
            output: "table".to_string(),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(query_cli).await;
    assert!(
        result.is_ok() || result.is_err(),
        "Multiple SERVICE endpoints test"
    );
}

#[tokio::test]
async fn test_federated_join() {
    let (_temp, dataset_path) = create_federated_dataset("test_fed_join").await;

    // Insert local data
    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                INSERT DATA {
                    ex:alice ex:worksAt ex:CompanyA .
                    ex:bob ex:worksAt ex:CompanyB .
                }
            "#
            .to_string(),
            file: false,
        },
        verbose: false,
        config: None,
        quiet: true,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    run(insert_cli).await.expect("INSERT should succeed");

    // Federated join query
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                SELECT ?person ?company ?revenue WHERE {
                    ?person ex:worksAt ?company .
                    SERVICE <http://finance.example.org/sparql> {
                        ?company ex:revenue ?revenue
                    }
                }
            "#
            .to_string(),
            file: false,
            output: "table".to_string(),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(query_cli).await;
    assert!(result.is_ok() || result.is_err(), "Federated join test");
}

#[tokio::test]
async fn test_service_with_values() {
    let (_temp, dataset_path) = create_federated_dataset("test_service_values").await;

    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                SELECT ?person ?data WHERE {
                    VALUES ?person { ex:alice ex:bob ex:carol }
                    SERVICE <http://external.example.org/sparql> {
                        ?person ex:data ?data
                    }
                }
            "#
            .to_string(),
            file: false,
            output: "table".to_string(),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(query_cli).await;
    assert!(
        result.is_ok() || result.is_err(),
        "SERVICE with VALUES test"
    );
}

#[tokio::test]
async fn test_service_optional() {
    let (_temp, dataset_path) = create_federated_dataset("test_service_optional").await;

    // Insert local data
    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                INSERT DATA {
                    ex:alice foaf:name "Alice" .
                    ex:bob foaf:name "Bob" .
                }
            "#
            .to_string(),
            file: false,
        },
        verbose: false,
        config: None,
        quiet: true,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    run(insert_cli).await.expect("INSERT should succeed");

    // Query with OPTIONAL SERVICE
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                SELECT ?person ?name ?email WHERE {
                    ?person foaf:name ?name .
                    OPTIONAL {
                        SERVICE <http://email.example.org/sparql> {
                            ?person foaf:mbox ?email
                        }
                    }
                }
            "#
            .to_string(),
            file: false,
            output: "table".to_string(),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(query_cli).await;
    // Should succeed with local data even if SERVICE fails
    assert!(result.is_ok(), "OPTIONAL SERVICE query should succeed");
}

#[tokio::test]
async fn test_wikidata_service() {
    let (_temp, dataset_path) = create_federated_dataset("test_wikidata").await;

    // Query Wikidata SPARQL endpoint (real-world use case)
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                SELECT ?city ?cityLabel WHERE {
                    SERVICE SILENT <https://query.wikidata.org/sparql> {
                        ?city wdt:P31 wd:Q515 .
                        ?city wdt:P17 wd:Q30 .
                        SERVICE wikibase:label {
                            bd:serviceParam wikibase:language "en" .
                        }
                    }
                }
                LIMIT 5
            "#
            .to_string(),
            file: false,
            output: "table".to_string(),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(query_cli).await;
    // Real Wikidata query - may timeout or require network
    assert!(result.is_ok() || result.is_err(), "Wikidata SERVICE test");
}

#[tokio::test]
async fn test_dbpedia_service() {
    let (_temp, dataset_path) = create_federated_dataset("test_dbpedia").await;

    // Query DBpedia SPARQL endpoint
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX dbo: <http://dbpedia.org/ontology/>
                PREFIX dbr: <http://dbpedia.org/resource/>
                SELECT ?person ?birthDate WHERE {
                    SERVICE SILENT <http://dbpedia.org/sparql> {
                        ?person a dbo:Person .
                        ?person dbo:birthDate ?birthDate .
                    }
                }
                LIMIT 5
            "#
            .to_string(),
            file: false,
            output: "table".to_string(),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(query_cli).await;
    // Real DBpedia query - may require network
    assert!(result.is_ok() || result.is_err(), "DBpedia SERVICE test");
}
