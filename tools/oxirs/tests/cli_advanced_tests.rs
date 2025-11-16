//! Advanced CLI tests for batch operations, updates, and performance
//!
//! Tests advanced SPARQL operations, batch processing, and edge cases

use oxirs::{run, Cli, Commands};
use std::fs;
use tempfile::TempDir;

/// Helper to create test dataset
async fn create_test_dataset(name: &str) -> (TempDir, String) {
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
async fn test_update_insert_data() {
    let (_temp, dataset_path) = create_test_dataset("test_update_insert").await;

    let cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                INSERT DATA {
                    ex:alice foaf:name "Alice" .
                    ex:alice foaf:age 30 .
                }
            "#
            .to_string(),
            file: false,
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
    assert!(result.is_ok(), "INSERT DATA should succeed");

    // Verify data was inserted by querying
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: "SELECT ?name WHERE { <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> ?name }".to_string(),
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

    let query_result = run(query_cli).await;
    assert!(query_result.is_ok(), "Query after INSERT should succeed");
}

#[tokio::test]
async fn test_update_delete_data() {
    let (_temp, dataset_path) = create_test_dataset("test_update_delete").await;

    // First insert data
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

    // Then delete data
    let delete_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                DELETE DATA {
                    ex:bob foaf:name "Bob" .
                }
            "#
            .to_string(),
            file: false,
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(delete_cli).await;
    assert!(result.is_ok(), "DELETE DATA should succeed");
}

#[tokio::test]
async fn test_update_delete_where() {
    let (_temp, dataset_path) = create_test_dataset("test_delete_where").await;

    // Insert test data
    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                INSERT DATA {
                    ex:alice foaf:name "Alice" ; foaf:age 30 .
                    ex:bob foaf:name "Bob" ; foaf:age 25 .
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

    // Delete with WHERE clause
    let delete_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path,
            update: r#"
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                DELETE { ?person foaf:age ?age }
                WHERE { ?person foaf:age ?age . FILTER(?age < 26) }
            "#
            .to_string(),
            file: false,
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(delete_cli).await;
    assert!(result.is_ok(), "DELETE WHERE should succeed");
}

#[tokio::test]
async fn test_update_insert_where() {
    let (_temp, dataset_path) = create_test_dataset("test_insert_where").await;

    // Insert initial data
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

    // Insert new triples based on existing data
    let update_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path,
            update: r#"
                PREFIX ex: <http://example.org/>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                INSERT { ?person ex:hasName ?name }
                WHERE { ?person foaf:name ?name }
            "#
            .to_string(),
            file: false,
        },
        verbose: false,
        config: None,
        quiet: false,
        no_color: true,
        interactive: false,
        profile: None,
        completion: None,
    };

    let result = run(update_cli).await;
    assert!(result.is_ok(), "INSERT WHERE should succeed");
}

#[tokio::test]
async fn test_update_from_file() {
    let (_temp, dataset_path) = create_test_dataset("test_update_file").await;
    let temp_dir = tempfile::tempdir().unwrap();
    let update_file = temp_dir.path().join("update.ru");

    fs::write(
        &update_file,
        r#"
        PREFIX ex: <http://example.org/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        INSERT DATA {
            ex:alice foaf:name "Alice from file" .
        }
    "#,
    )
    .unwrap();

    let cli = Cli {
        command: Commands::Update {
            dataset: dataset_path,
            update: update_file.to_string_lossy().to_string(),
            file: true,
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
    assert!(result.is_ok(), "Update from file should succeed");
}

#[tokio::test]
async fn test_large_dataset_query() {
    let (_temp, dataset_path) = create_test_dataset("test_large_dataset").await;

    // Insert many triples
    let mut insert_query = String::from("PREFIX ex: <http://example.org/>\nINSERT DATA {\n");
    for i in 0..1000 {
        insert_query.push_str(&format!("  ex:entity{} ex:value \"Value {}\" .\n", i, i));
    }
    insert_query.push('}');

    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: insert_query,
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

    run(insert_cli).await.expect("Large INSERT should succeed");

    // Query the large dataset
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }".to_string(),
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
    assert!(result.is_ok(), "Query on large dataset should succeed");
}

#[tokio::test]
async fn test_query_with_limit_and_offset() {
    let (_temp, dataset_path) = create_test_dataset("test_limit_offset").await;

    // Insert test data
    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                INSERT DATA {
                    ex:entity1 ex:value "Value 1" .
                    ex:entity2 ex:value "Value 2" .
                    ex:entity3 ex:value "Value 3" .
                    ex:entity4 ex:value "Value 4" .
                    ex:entity5 ex:value "Value 5" .
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

    // Query with LIMIT and OFFSET
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: "SELECT ?s ?v WHERE { ?s <http://example.org/value> ?v } ORDER BY ?s LIMIT 2 OFFSET 1".to_string(),
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
    assert!(result.is_ok(), "Query with LIMIT and OFFSET should succeed");
}

#[tokio::test]
async fn test_complex_path_query() {
    let (_temp, dataset_path) = create_test_dataset("test_path_query").await;

    // Insert test data with relationships
    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                INSERT DATA {
                    ex:alice ex:knows ex:bob .
                    ex:bob ex:knows ex:carol .
                    ex:carol ex:knows ex:dave .
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

    // Query with property path
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: "PREFIX ex: <http://example.org/>\nSELECT ?person WHERE { ex:alice ex:knows+ ?person }".to_string(),
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
    assert!(result.is_ok(), "Complex path query should succeed");
}

#[tokio::test]
async fn test_query_with_bind() {
    let (_temp, dataset_path) = create_test_dataset("test_bind").await;

    // Insert test data
    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                INSERT DATA {
                    ex:alice ex:firstName "Alice" ; ex:lastName "Smith" .
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

    // Query with BIND
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                SELECT ?fullName WHERE {
                    ?person ex:firstName ?first ; ex:lastName ?last .
                    BIND(CONCAT(?first, " ", ?last) AS ?fullName)
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
    assert!(result.is_ok(), "Query with BIND should succeed");
}

// Edge case tests

#[tokio::test]
async fn test_empty_dataset_query() {
    let (_temp, dataset_path) = create_test_dataset("test_empty").await;

    let cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
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
    assert!(
        result.is_ok(),
        "Query on empty dataset should succeed with no results"
    );
}

#[tokio::test]
async fn test_update_invalid_syntax() {
    let (_temp, dataset_path) = create_test_dataset("test_invalid_update").await;

    let cli = Cli {
        command: Commands::Update {
            dataset: dataset_path,
            update: "INVALID UPDATE SYNTAX".to_string(),
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

    let result = run(cli).await;
    assert!(result.is_err(), "Update with invalid syntax should fail");
}

#[tokio::test]
async fn test_unicode_data() {
    let (_temp, dataset_path) = create_test_dataset("test_unicode").await;

    // Insert Unicode data
    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                INSERT DATA {
                    ex:person1 foaf:name "Alice" .
                    ex:person2 foaf:name "鈴木" .
                    ex:person3 foaf:name "François" .
                    ex:person4 foaf:name "Владимир" .
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

    run(insert_cli)
        .await
        .expect("Unicode INSERT should succeed");

    // Query Unicode data
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }"
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
    assert!(result.is_ok(), "Query with Unicode data should succeed");
}
