//! Comprehensive CLI tests for SPARQL query functionality
//!
//! Tests various SPARQL query types, output formats, and error handling

use oxirs::{run, Cli, Commands};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// Test fixture with sample RDF data
struct TestDataset {
    _temp_dir: TempDir,
    dataset_path: PathBuf,
    _dataset_name: String,
}

impl TestDataset {
    /// Create a new test dataset with sample RDF data
    async fn new(name: &str) -> Self {
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

        // Create sample RDF data
        let sample_data = r#"
            @prefix ex: <http://example.org/> .
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .
            @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

            ex:alice a foaf:Person ;
                     foaf:name "Alice" ;
                     foaf:age "30"^^xsd:integer ;
                     foaf:knows ex:bob .

            ex:bob a foaf:Person ;
                   foaf:name "Bob" ;
                   foaf:age "25"^^xsd:integer ;
                   foaf:knows ex:alice .

            ex:carol a foaf:Person ;
                     foaf:name "Carol" ;
                     foaf:age "28"^^xsd:integer .
        "#;

        let data_file = temp_dir.path().join("sample.ttl");
        fs::write(&data_file, sample_data).unwrap();

        // Import the data
        let import_cli = Cli {
            command: Commands::Import {
                dataset: dataset_path.to_string_lossy().to_string(),
                file: data_file,
                format: Some("turtle".to_string()),
                graph: None,
                resume: false,
            },
            verbose: false,
            config: None,
            quiet: true,
            no_color: true,
            interactive: false,
            profile: None,
            completion: None,
        };

        run(import_cli).await.expect("Failed to import data");

        Self {
            _temp_dir: temp_dir,
            dataset_path,
            _dataset_name: name.to_string(),
        }
    }
}

#[tokio::test]
async fn test_select_query_basic() {
    let dataset = TestDataset::new("test_select").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: "SELECT ?s ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }"
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

    let result = run(cli).await;
    assert!(result.is_ok(), "SELECT query should succeed");
}

#[tokio::test]
async fn test_select_query_with_filter() {
    let dataset = TestDataset::new("test_filter").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: r#"
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                SELECT ?name ?age WHERE {
                    ?person foaf:name ?name ;
                            foaf:age ?age .
                    FILTER (?age > "26"^^xsd:integer)
                }
                ORDER BY DESC(?age)
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

    let result = run(cli).await;
    assert!(result.is_ok(), "SELECT query with FILTER should succeed");
}

#[tokio::test]
async fn test_ask_query_true() {
    let dataset = TestDataset::new("test_ask_true").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: r#"
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                ASK { ?person foaf:name "Alice" }
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

    let result = run(cli).await;
    assert!(result.is_ok(), "ASK query should succeed");
}

#[tokio::test]
async fn test_ask_query_false() {
    let dataset = TestDataset::new("test_ask_false").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: r#"
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                ASK { ?person foaf:name "NonExistent" }
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

    let result = run(cli).await;
    assert!(result.is_ok(), "ASK query should succeed even when false");
}

#[tokio::test]
async fn test_construct_query() {
    let dataset = TestDataset::new("test_construct").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: r#"
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                PREFIX ex: <http://example.org/>
                CONSTRUCT {
                    ?person ex:hasName ?name
                } WHERE {
                    ?person foaf:name ?name
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

    let result = run(cli).await;
    assert!(result.is_ok(), "CONSTRUCT query should succeed");
}

#[tokio::test]
async fn test_describe_query() {
    let dataset = TestDataset::new("test_describe").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: "DESCRIBE <http://example.org/alice>".to_string(),
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

    let result = run(cli).await;
    assert!(result.is_ok(), "DESCRIBE query should succeed");
}

#[tokio::test]
async fn test_query_json_output() {
    let dataset = TestDataset::new("test_json_output").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: "SELECT ?s ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name } LIMIT 5"
                .to_string(),
            file: false,
            output: "json".to_string(),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    assert!(result.is_ok(), "Query with JSON output should succeed");
}

#[tokio::test]
async fn test_query_csv_output() {
    let dataset = TestDataset::new("test_csv_output").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: "SELECT ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }".to_string(),
            file: false,
            output: "csv".to_string(),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    assert!(result.is_ok(), "Query with CSV output should succeed");
}

#[tokio::test]
async fn test_query_xml_output() {
    let dataset = TestDataset::new("test_xml_output").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: "SELECT ?name WHERE { ?s <http://xmlns.com/foaf/0.1/name> ?name }".to_string(),
            file: false,
            output: "xml".to_string(),
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    assert!(result.is_ok(), "Query with XML output should succeed");
}

#[tokio::test]
async fn test_query_from_file() {
    let dataset = TestDataset::new("test_query_file").await;
    let temp_dir = tempfile::tempdir().unwrap();
    let query_file = temp_dir.path().join("query.rq");

    fs::write(&query_file, "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10").unwrap();

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: query_file.to_string_lossy().to_string(),
            file: true,
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

    let result = run(cli).await;
    assert!(result.is_ok(), "Query from file should succeed");
}

#[tokio::test]
async fn test_query_with_optional() {
    let dataset = TestDataset::new("test_optional").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: r#"
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                SELECT ?person ?name ?knows WHERE {
                    ?person foaf:name ?name .
                    OPTIONAL { ?person foaf:knows ?knows }
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

    let result = run(cli).await;
    assert!(result.is_ok(), "Query with OPTIONAL should succeed");
}

#[tokio::test]
async fn test_query_with_union() {
    let dataset = TestDataset::new("test_union").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: r#"
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                SELECT ?person ?value WHERE {
                    {
                        ?person foaf:name ?value
                    } UNION {
                        ?person foaf:age ?value
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

    let result = run(cli).await;
    assert!(result.is_ok(), "Query with UNION should succeed");
}

// Error handling tests

#[tokio::test]
async fn test_query_invalid_syntax() {
    let dataset = TestDataset::new("test_invalid_syntax").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: "INVALID QUERY SYNTAX".to_string(),
            file: false,
            output: "table".to_string(),
        },
        verbose: false,
        config: None,
        quiet: true,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    assert!(result.is_err(), "Query with invalid syntax should fail");
}

#[tokio::test]
async fn test_query_nonexistent_dataset() {
    let cli = Cli {
        command: Commands::Query {
            dataset: "/nonexistent/dataset/path".to_string(),
            query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
            file: false,
            output: "table".to_string(),
        },
        verbose: false,
        config: None,
        quiet: true,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    assert!(result.is_err(), "Query on nonexistent dataset should fail");
}

#[tokio::test]
async fn test_query_invalid_output_format() {
    let dataset = TestDataset::new("test_invalid_format").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
            file: false,
            output: "invalid_format".to_string(),
        },
        verbose: false,
        config: None,
        quiet: true,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(cli).await;
    assert!(
        result.is_err(),
        "Query with invalid output format should fail"
    );
}

#[tokio::test]
async fn test_query_count_aggregation() {
    let dataset = TestDataset::new("test_count").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: r#"
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                SELECT (COUNT(?person) as ?count) WHERE {
                    ?person a foaf:Person
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

    let result = run(cli).await;
    assert!(result.is_ok(), "COUNT aggregation query should succeed");
}

#[tokio::test]
async fn test_query_group_by() {
    let dataset = TestDataset::new("test_group_by").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset.dataset_path.to_string_lossy().to_string(),
            query: r#"
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                SELECT ?name (COUNT(?knows) as ?friendCount) WHERE {
                    ?person foaf:name ?name .
                    OPTIONAL { ?person foaf:knows ?knows }
                }
                GROUP BY ?name
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

    let result = run(cli).await;
    assert!(result.is_ok(), "GROUP BY query should succeed");
}
