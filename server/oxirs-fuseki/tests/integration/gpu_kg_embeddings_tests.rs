//! Integration tests for GPU-accelerated knowledge graph embeddings
//!
//! These tests verify the correctness and functionality of the GPU embedding generator
//! with various knowledge graph structures and embedding models.

use oxirs_fuseki::gpu_kg_embeddings::{
    EmbeddingConfig, EmbeddingModel, GpuBackendType, GpuEmbeddingGenerator,
};

#[test]
fn test_generator_creation() {
    let config = EmbeddingConfig::default();
    let generator = GpuEmbeddingGenerator::new(config);

    assert!(generator.is_ok());

    let generator = generator.unwrap();
    let stats = generator.get_statistics();

    assert_eq!(stats.num_entities, 0);
    assert_eq!(stats.num_relations, 0);
    assert_eq!(stats.embedding_dim, 128);
}

#[test]
fn test_initialize_from_simple_triples() {
    let config = EmbeddingConfig {
        embedding_dim: 64,
        backend: GpuBackendType::Cpu,
        ..Default::default()
    };

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples = vec![
        ("Alice".to_string(), "knows".to_string(), "Bob".to_string()),
        ("Bob".to_string(), "knows".to_string(), "Charlie".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    let stats = generator.get_statistics();
    assert_eq!(stats.num_entities, 3); // Alice, Bob, Charlie
    assert_eq!(stats.num_relations, 1); // knows
    assert_eq!(stats.embedding_dim, 64);
}

#[test]
fn test_initialize_from_complex_graph() {
    let mut config = EmbeddingConfig::default();
    config.backend = GpuBackendType::Cpu;

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples = vec![
        ("Alice".to_string(), "knows".to_string(), "Bob".to_string()),
        ("Bob".to_string(), "knows".to_string(), "Charlie".to_string()),
        ("Charlie".to_string(), "works_at".to_string(), "Company_A".to_string()),
        ("Alice".to_string(), "lives_in".to_string(), "City_X".to_string()),
        ("Bob".to_string(), "lives_in".to_string(), "City_Y".to_string()),
        ("Company_A".to_string(), "located_in".to_string(), "City_Z".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    let stats = generator.get_statistics();
    assert_eq!(stats.num_entities, 7); // Alice, Bob, Charlie, Company_A, City_X, City_Y, City_Z
    assert_eq!(stats.num_relations, 3); // knows, works_at, lives_in, located_in
}

#[test]
fn test_get_entity_embedding() {
    let mut config = EmbeddingConfig::default();
    config.backend = GpuBackendType::Cpu;
    config.embedding_dim = 128;

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples = vec![
        ("Alice".to_string(), "knows".to_string(), "Bob".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    let embedding = generator.get_entity_embedding("Alice");
    assert!(embedding.is_some());

    let embedding = embedding.unwrap();
    assert_eq!(embedding.len(), 128);

    // Check normalization (embedding should be normalized)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.1); // Approximately normalized
}

#[test]
fn test_get_relation_embedding() {
    let mut config = EmbeddingConfig::default();
    config.backend = GpuBackendType::Cpu;
    config.embedding_dim = 64;

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples = vec![
        ("Alice".to_string(), "knows".to_string(), "Bob".to_string()),
        ("Bob".to_string(), "likes".to_string(), "Charlie".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    let embedding = generator.get_relation_embedding("knows");
    assert!(embedding.is_some());
    assert_eq!(embedding.unwrap().len(), 64);

    let embedding = generator.get_relation_embedding("likes");
    assert!(embedding.is_some());
    assert_eq!(embedding.unwrap().len(), 64);
}

#[test]
fn test_get_nonexistent_entity() {
    let mut config = EmbeddingConfig::default();
    config.backend = GpuBackendType::Cpu;

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples = vec![
        ("Alice".to_string(), "knows".to_string(), "Bob".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    let embedding = generator.get_entity_embedding("NonExistent");
    assert!(embedding.is_none());
}

#[test]
fn test_training_transe() {
    let mut config = EmbeddingConfig {
        embedding_dim: 32,
        learning_rate: 0.01,
        batch_size: 4,
        num_negatives: 5,
        model: EmbeddingModel::TransE,
        backend: GpuBackendType::Cpu,
        use_mixed_precision: false,
        use_tensor_cores: false,
    };

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples = vec![
        ("Alice".to_string(), "knows".to_string(), "Bob".to_string()),
        ("Bob".to_string(), "knows".to_string(), "Charlie".to_string()),
        ("Charlie".to_string(), "knows".to_string(), "David".to_string()),
        ("David".to_string(), "knows".to_string(), "Eve".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    let metrics = generator.train(&triples, 5).unwrap();

    assert_eq!(metrics.epochs, 5);
    assert!(metrics.average_loss >= 0.0);
    assert!(!metrics.gpu_accelerated); // CPU backend
    assert!(!metrics.tensor_core_used); // Disabled
}

#[test]
fn test_training_distmult() {
    let mut config = EmbeddingConfig {
        embedding_dim: 32,
        learning_rate: 0.01,
        batch_size: 2,
        num_negatives: 3,
        model: EmbeddingModel::DistMult,
        backend: GpuBackendType::Cpu,
        use_mixed_precision: false,
        use_tensor_cores: false,
    };

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples = vec![
        ("A".to_string(), "r1".to_string(), "B".to_string()),
        ("B".to_string(), "r2".to_string(), "C".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    let metrics = generator.train(&triples, 3).unwrap();

    assert_eq!(metrics.epochs, 3);
    assert!(metrics.average_loss >= 0.0);
}

#[test]
fn test_find_similar_entities() {
    let mut config = EmbeddingConfig {
        embedding_dim: 64,
        backend: GpuBackendType::Cpu,
        batch_size: 4,
        ..Default::default()
    };

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples = vec![
        ("Alice".to_string(), "knows".to_string(), "Bob".to_string()),
        ("Alice".to_string(), "knows".to_string(), "Charlie".to_string()),
        ("Bob".to_string(), "knows".to_string(), "Charlie".to_string()),
        ("David".to_string(), "knows".to_string(), "Eve".to_string()),
        ("Frank".to_string(), "works_at".to_string(), "Company".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    // Train to get meaningful embeddings
    generator.train(&triples, 10).unwrap();

    // Find similar entities to Alice
    let similar = generator.find_similar_entities("Alice", 3);

    // Should return at most 3 entities
    assert!(similar.len() <= 3);

    // All similarities should be between -1 and 1
    for (_, similarity) in &similar {
        assert!(*similarity >= -1.0 && *similarity <= 1.0);
    }
}

#[test]
fn test_find_similar_nonexistent_entity() {
    let mut config = EmbeddingConfig::default();
    config.backend = GpuBackendType::Cpu;

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples = vec![
        ("Alice".to_string(), "knows".to_string(), "Bob".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    let similar = generator.find_similar_entities("NonExistent", 5);
    assert_eq!(similar.len(), 0);
}

#[test]
fn test_empty_knowledge_graph() {
    let mut config = EmbeddingConfig::default();
    config.backend = GpuBackendType::Cpu;

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples: Vec<(String, String, String)> = vec![];

    generator.initialize_from_triples(&triples).unwrap();

    let stats = generator.get_statistics();
    assert_eq!(stats.num_entities, 0);
    assert_eq!(stats.num_relations, 0);
    assert_eq!(stats.total_parameters, 0);
}

#[test]
fn test_self_loop_triples() {
    let mut config = EmbeddingConfig::default();
    config.backend = GpuBackendType::Cpu;

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples = vec![
        ("Alice".to_string(), "knows".to_string(), "Alice".to_string()),
        ("Bob".to_string(), "likes".to_string(), "Bob".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    let stats = generator.get_statistics();
    assert_eq!(stats.num_entities, 2); // Alice and Bob
    assert_eq!(stats.num_relations, 2); // knows and likes
}

#[test]
fn test_duplicate_triples() {
    let mut config = EmbeddingConfig::default();
    config.backend = GpuBackendType::Cpu;

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples = vec![
        ("Alice".to_string(), "knows".to_string(), "Bob".to_string()),
        ("Alice".to_string(), "knows".to_string(), "Bob".to_string()), // Duplicate
        ("Bob".to_string(), "knows".to_string(), "Charlie".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    // Training should handle duplicates gracefully
    let metrics = generator.train(&triples, 2).unwrap();
    assert_eq!(metrics.epochs, 2);
}

#[test]
fn test_large_knowledge_graph() {
    let mut config = EmbeddingConfig {
        embedding_dim: 128,
        batch_size: 64,
        backend: GpuBackendType::Cpu,
        ..Default::default()
    };

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    // Create a larger knowledge graph
    let mut triples = Vec::new();
    for i in 0..100 {
        for j in 0..5 {
            triples.push((
                format!("entity_{}", i),
                format!("relation_{}", j),
                format!("entity_{}", (i + j + 1) % 100),
            ));
        }
    }

    generator.initialize_from_triples(&triples).unwrap();

    let stats = generator.get_statistics();
    assert_eq!(stats.num_entities, 100);
    assert_eq!(stats.num_relations, 5);

    // Train on large graph
    let metrics = generator.train(&triples, 2).unwrap();
    assert_eq!(metrics.epochs, 2);
}

#[test]
fn test_statistics_total_parameters() {
    let mut config = EmbeddingConfig {
        embedding_dim: 64,
        backend: GpuBackendType::Cpu,
        ..Default::default()
    };

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    let triples = vec![
        ("A".to_string(), "r".to_string(), "B".to_string()),
        ("B".to_string(), "r".to_string(), "C".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    let stats = generator.get_statistics();
    // 3 entities + 1 relation, each with 64-dimensional embedding
    assert_eq!(stats.total_parameters, (3 + 1) * 64);
}

#[test]
fn test_different_embedding_models() {
    let models = vec![
        EmbeddingModel::TransE,
        EmbeddingModel::DistMult,
        EmbeddingModel::ComplEx,
        EmbeddingModel::RotatE,
    ];

    for model in models {
        let config = EmbeddingConfig {
            embedding_dim: 32,
            batch_size: 2,
            model,
            backend: GpuBackendType::Cpu,
            use_mixed_precision: false,
            use_tensor_cores: false,
        };

        let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

        let triples = vec![
            ("A".to_string(), "r".to_string(), "B".to_string()),
            ("B".to_string(), "r".to_string(), "C".to_string()),
        ];

        generator.initialize_from_triples(&triples).unwrap();

        // Should be able to train with any model
        let metrics = generator.train(&triples, 2).unwrap();
        assert_eq!(metrics.epochs, 2);
    }
}

#[test]
fn test_realistic_social_network() {
    let mut config = EmbeddingConfig {
        embedding_dim: 64,
        batch_size: 8,
        backend: GpuBackendType::Cpu,
        ..Default::default()
    };

    let mut generator = GpuEmbeddingGenerator::new(config).unwrap();

    // Create a realistic social network
    let triples = vec![
        // Friendship relations
        ("Alice".to_string(), "friend_of".to_string(), "Bob".to_string()),
        ("Bob".to_string(), "friend_of".to_string(), "Charlie".to_string()),
        ("Alice".to_string(), "friend_of".to_string(), "David".to_string()),

        // Work relations
        ("Alice".to_string(), "colleague_of".to_string(), "Eve".to_string()),
        ("Bob".to_string(), "colleague_of".to_string(), "Frank".to_string()),

        // Family relations
        ("Alice".to_string(), "sibling_of".to_string(), "Grace".to_string()),
        ("Bob".to_string(), "parent_of".to_string(), "Henry".to_string()),

        // Location
        ("Alice".to_string(), "lives_in".to_string(), "NYC".to_string()),
        ("Bob".to_string(), "lives_in".to_string(), "SF".to_string()),
        ("Charlie".to_string(), "lives_in".to_string(), "LA".to_string()),
    ];

    generator.initialize_from_triples(&triples).unwrap();

    let stats = generator.get_statistics();
    assert!(stats.num_entities >= 10);
    assert!(stats.num_relations >= 5);

    // Train embeddings
    let metrics = generator.train(&triples, 20).unwrap();
    assert_eq!(metrics.epochs, 20);

    // Find similar people to Alice
    let similar = generator.find_similar_entities("Alice", 5);
    assert!(similar.len() > 0);
    assert!(similar.len() <= 5);
}

#[test]
fn test_embedding_config_defaults() {
    let config = EmbeddingConfig::default();

    assert_eq!(config.embedding_dim, 128);
    assert_eq!(config.learning_rate, 0.01);
    assert_eq!(config.batch_size, 1024);
    assert_eq!(config.num_negatives, 10);
    assert_eq!(config.model, EmbeddingModel::TransE);
    assert_eq!(config.backend, GpuBackendType::Cuda);
    assert!(config.use_mixed_precision);
    assert!(config.use_tensor_cores);
}
