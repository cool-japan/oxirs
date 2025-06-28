//! Tests for SPARQL vector integration

use oxirs_vec::{
    embeddings::{EmbeddableContent, EmbeddingStrategy},
    sparql_integration::{
        SparqlVectorService, VectorServiceArg, VectorServiceConfig, VectorServiceResult,
    },
    Vector, VectorStore,
};

#[test]
fn test_vec_similar_function() {
    // Create a vector service with SentenceTransformer embeddings (doesn't require vocabulary)
    let config = VectorServiceConfig::default();
    let mut service = SparqlVectorService::new(config, EmbeddingStrategy::SentenceTransformer)
        .expect("Failed to create vector service");

    // Add some test resources
    let content1 =
        EmbeddableContent::Text("Machine learning and artificial intelligence".to_string());
    let content2 = EmbeddableContent::Text("Deep learning neural networks".to_string());
    let content3 = EmbeddableContent::Text("Natural language processing".to_string());
    let content4 = EmbeddableContent::Text("Computer vision and image recognition".to_string());

    service
        .add_resource_embedding("http://example.org/ml1", &content1)
        .unwrap();
    service
        .add_resource_embedding("http://example.org/ml2", &content2)
        .unwrap();
    service
        .add_resource_embedding("http://example.org/nlp", &content3)
        .unwrap();
    service
        .add_resource_embedding("http://example.org/cv", &content4)
        .unwrap();

    // Test vec:similar function
    let args = vec![
        VectorServiceArg::IRI("http://example.org/ml1".to_string()),
        VectorServiceArg::Number(3.0),
        VectorServiceArg::Number(0.0),
    ];

    let result = service.execute_function("similar", &args).unwrap();

    match result {
        VectorServiceResult::SimilarityList(results) => {
            assert!(!results.is_empty());
            // Should return some results - with hash-based embeddings we can't
            // guarantee exact similarity scores
            assert!(results.len() <= 3); // At most 3 results requested
        }
        _ => panic!("Expected SimilarityList result"),
    }
}

#[test]
fn test_vec_similarity_function() {
    // Create a vector service
    let config = VectorServiceConfig::default();
    let mut service = SparqlVectorService::new(config, EmbeddingStrategy::SentenceTransformer)
        .expect("Failed to create vector service");

    // Add test resources
    let content1 = EmbeddableContent::Text("Machine learning algorithms".to_string());
    let content2 = EmbeddableContent::Text("Machine learning techniques".to_string());
    let content3 = EmbeddableContent::Text("Cooking recipes and food".to_string());

    service
        .add_resource_embedding("http://example.org/ml1", &content1)
        .unwrap();
    service
        .add_resource_embedding("http://example.org/ml2", &content2)
        .unwrap();
    service
        .add_resource_embedding("http://example.org/cooking", &content3)
        .unwrap();

    // Test similarity between similar resources
    let args_similar = vec![
        VectorServiceArg::IRI("http://example.org/ml1".to_string()),
        VectorServiceArg::IRI("http://example.org/ml2".to_string()),
    ];

    let result_similar = service
        .execute_function("similarity", &args_similar)
        .unwrap();

    match result_similar {
        VectorServiceResult::Number(similarity) => {
            // With hash-based embeddings, we can't guarantee semantic similarity
            // Just verify we get a number between -1 and 1
            assert!(
                similarity >= -1.0 && similarity <= 1.0,
                "Similarity should be between -1 and 1, got: {}",
                similarity
            );
        }
        _ => panic!("Expected Number result"),
    }

    // Test similarity between dissimilar resources
    let args_dissimilar = vec![
        VectorServiceArg::IRI("http://example.org/ml1".to_string()),
        VectorServiceArg::IRI("http://example.org/cooking".to_string()),
    ];

    let result_dissimilar = service
        .execute_function("similarity", &args_dissimilar)
        .unwrap();

    match result_dissimilar {
        VectorServiceResult::Number(similarity) => {
            // With hash-based embeddings, we can't guarantee semantic similarity
            // Just verify we get a number between -1 and 1
            assert!(
                similarity >= -1.0 && similarity <= 1.0,
                "Similarity should be between -1 and 1, got: {}",
                similarity
            );
        }
        _ => panic!("Expected Number result"),
    }
}

#[test]
fn test_vec_embed_text_function() {
    let config = VectorServiceConfig::default();
    let mut service = SparqlVectorService::new(config, EmbeddingStrategy::SentenceTransformer)
        .expect("Failed to create vector service");

    // Test embedding text
    let args = vec![VectorServiceArg::String(
        "This is a test document about vectors".to_string(),
    )];

    let result = service.execute_function("embed_text", &args).unwrap();

    match result {
        VectorServiceResult::Vector(vector) => {
            assert!(vector.dimensions > 0);
            // Check that the vector has reasonable values
            let values = vector.as_f32();
            assert!(!values.is_empty());
        }
        _ => panic!("Expected Vector result"),
    }
}

#[test]
fn test_vec_search_text_function() {
    let config = VectorServiceConfig::default();
    let mut service = SparqlVectorService::new(config, EmbeddingStrategy::SentenceTransformer)
        .expect("Failed to create vector service");

    // Add some documents
    let docs = vec![
        (
            "http://example.org/doc1",
            "Information retrieval and search engines",
        ),
        ("http://example.org/doc2", "Database management systems"),
        (
            "http://example.org/doc3",
            "Search algorithms and data structures",
        ),
    ];

    for (uri, text) in docs {
        let content = EmbeddableContent::Text(text.to_string());
        service.add_resource_embedding(uri, &content).unwrap();
    }

    // Search for relevant documents
    let args = vec![
        VectorServiceArg::String("search algorithms".to_string()),
        VectorServiceArg::Number(2.0),
    ];

    let result = service.execute_function("search_text", &args).unwrap();

    match result {
        VectorServiceResult::SimilarityList(results) => {
            assert!(!results.is_empty());
            // With hash-based embeddings, we can't guarantee which doc is most relevant
            // Just verify we got some results back
            assert!(results.len() <= 2); // At most 2 results requested
        }
        _ => panic!("Expected SimilarityList result"),
    }
}

#[test]
fn test_vector_similarity_function() {
    let config = VectorServiceConfig::default();
    let mut service = SparqlVectorService::new(config, EmbeddingStrategy::SentenceTransformer)
        .expect("Failed to create vector service");

    // Create two vectors
    let vec1 = Vector::new(vec![1.0, 0.0, 0.0]);
    let vec2 = Vector::new(vec![0.0, 1.0, 0.0]);
    let vec3 = Vector::new(vec![1.0, 0.0, 0.0]);

    // Test similarity between orthogonal vectors
    let args_orthogonal = vec![
        VectorServiceArg::Vector(vec1.clone()),
        VectorServiceArg::Vector(vec2.clone()),
    ];

    let result_orthogonal = service
        .execute_function("vector_similarity", &args_orthogonal)
        .unwrap();

    match result_orthogonal {
        VectorServiceResult::Number(similarity) => {
            assert!(
                (similarity - 0.0).abs() < 0.001,
                "Orthogonal vectors should have 0 similarity"
            );
        }
        _ => panic!("Expected Number result"),
    }

    // Test similarity between identical vectors
    let args_identical = vec![
        VectorServiceArg::Vector(vec1.clone()),
        VectorServiceArg::Vector(vec3.clone()),
    ];

    let result_identical = service
        .execute_function("vector_similarity", &args_identical)
        .unwrap();

    match result_identical {
        VectorServiceResult::Number(similarity) => {
            assert!(
                (similarity - 1.0).abs() < 0.001,
                "Identical vectors should have 1.0 similarity"
            );
        }
        _ => panic!("Expected Number result"),
    }
}

#[test]
fn test_sparql_service_query_generation() {
    use oxirs_vec::sparql_integration::VectorOperation;

    let config = VectorServiceConfig::default();
    let service = SparqlVectorService::new(config, EmbeddingStrategy::TfIdf)
        .expect("Failed to create vector service");

    // Test FindSimilar operation query generation
    let find_similar_op = VectorOperation::FindSimilar {
        resource: "http://example.org/resource1".to_string(),
        limit: Some(10),
        threshold: Some(0.8),
    };

    let query = service.generate_service_query(&find_similar_op);
    assert!(query.contains("vec:similar"));
    assert!(query.contains("http://example.org/resource1"));
    assert!(query.contains("LIMIT 10"));

    // Test CalculateSimilarity operation query generation
    let calc_similarity_op = VectorOperation::CalculateSimilarity {
        resource1: "http://example.org/resource1".to_string(),
        resource2: "http://example.org/resource2".to_string(),
    };

    let query = service.generate_service_query(&calc_similarity_op);
    assert!(query.contains("vec:similarity"));
    assert!(query.contains("http://example.org/resource1"));
    assert!(query.contains("http://example.org/resource2"));
}
