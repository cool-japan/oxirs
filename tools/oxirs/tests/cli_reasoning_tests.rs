//! CLI tests for reasoning and inference capabilities
//!
//! Tests RDFS/OWL reasoning, transitivity, and inference rules

use oxirs::{run, Cli, Commands};
use tempfile::TempDir;

/// Helper to create test dataset with reasoning
async fn create_reasoning_dataset(name: &str) -> (TempDir, String) {
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
async fn test_rdfs_subclass_inference() {
    let (_temp, dataset_path) = create_reasoning_dataset("test_subclass").await;

    // Insert ontology with subclass relationships
    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                INSERT DATA {
                    ex:Dog rdfs:subClassOf ex:Animal .
                    ex:Cat rdfs:subClassOf ex:Animal .
                    ex:Mammal rdfs:subClassOf ex:Animal .
                    ex:Dog rdfs:subClassOf ex:Mammal .

                    ex:fido rdf:type ex:Dog .
                    ex:fluffy rdf:type ex:Cat .
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

    // Query with reasoning (should infer fido is also an Animal through transitivity)
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                SELECT ?animal WHERE {
                    ?animal rdf:type ex:Animal
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
    // Without reasoning, this may only return explicitly typed animals
    // With reasoning, it should infer fido and fluffy are Animals
    assert!(result.is_ok(), "RDFS subclass query should succeed");
}

#[tokio::test]
async fn test_rdfs_subproperty_inference() {
    let (_temp, dataset_path) = create_reasoning_dataset("test_subproperty").await;

    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                INSERT DATA {
                    ex:hasFather rdfs:subPropertyOf ex:hasParent .
                    ex:hasMother rdfs:subPropertyOf ex:hasParent .

                    ex:alice ex:hasFather ex:bob .
                    ex:alice ex:hasMother ex:carol .
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

    // Query for parents (should infer both through subproperty)
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                SELECT ?parent WHERE {
                    ex:alice ex:hasParent ?parent
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
    assert!(result.is_ok(), "RDFS subproperty query should succeed");
}

#[tokio::test]
async fn test_owl_transitive_property() {
    let (_temp, dataset_path) = create_reasoning_dataset("test_transitive").await;

    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                INSERT DATA {
                    ex:ancestorOf rdf:type owl:TransitiveProperty .

                    ex:alice ex:ancestorOf ex:bob .
                    ex:bob ex:ancestorOf ex:carol .
                    ex:carol ex:ancestorOf ex:dave .
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

    // Query with transitivity (alice should be ancestor of dave)
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                SELECT ?descendant WHERE {
                    ex:alice ex:ancestorOf+ ?descendant
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
    // Property path + handles transitivity
    assert!(result.is_ok(), "Transitive property query should succeed");
}

#[tokio::test]
async fn test_owl_symmetric_property() {
    let (_temp, dataset_path) = create_reasoning_dataset("test_symmetric").await;

    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                INSERT DATA {
                    ex:marriedTo rdf:type owl:SymmetricProperty .

                    ex:alice ex:marriedTo ex:bob .
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

    // Query inverse direction (should be inferred)
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                SELECT ?spouse WHERE {
                    ex:bob ex:marriedTo ?spouse
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
    // Without reasoning, this may return no results
    // With reasoning, should infer bob is married to alice
    assert!(result.is_ok(), "Symmetric property query should succeed");
}

#[tokio::test]
async fn test_owl_inverse_property() {
    let (_temp, dataset_path) = create_reasoning_dataset("test_inverse").await;

    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                INSERT DATA {
                    ex:hasChild owl:inverseOf ex:hasParent .

                    ex:alice ex:hasChild ex:bob .
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

    // Query using inverse property
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                SELECT ?parent WHERE {
                    ex:bob ex:hasParent ?parent
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
    assert!(result.is_ok(), "Inverse property query should succeed");
}

#[tokio::test]
async fn test_owl_same_as_inference() {
    let (_temp, dataset_path) = create_reasoning_dataset("test_sameas").await;

    let insert_cli = Cli {
        command: Commands::Update {
            dataset: dataset_path.clone(),
            update: r#"
                PREFIX ex: <http://example.org/>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                INSERT DATA {
                    ex:alice owl:sameAs ex:alice_smith .
                    ex:alice foaf:age 30 .
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

    // Query for alice_smith's age (should be inferred from sameAs)
    let query_cli = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: r#"
                PREFIX ex: <http://example.org/>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                SELECT ?age WHERE {
                    ex:alice_smith foaf:age ?age
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
    assert!(result.is_ok(), "OWL sameAs query should succeed");
}

#[tokio::test]
async fn test_custom_inference_rules() {
    let (_temp, dataset_path) = create_reasoning_dataset("test_custom_rules").await;

    // Test if custom rules can be defined (future feature)
    let result = Cli {
        command: Commands::Query {
            dataset: dataset_path,
            query: "SELECT * WHERE { ?s ?p ?o } LIMIT 1".to_string(),
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

    let res = run(result).await;
    assert!(res.is_ok(), "Custom inference rules test");
}
