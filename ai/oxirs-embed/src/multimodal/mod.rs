//! Multi-modal embeddings and cross-modal alignment for unified representation learning
//!
//! This module provides advanced multi-modal integration capabilities for combining
//! text, knowledge graph, and other modalities into unified embedding spaces.

// Import and re-export the modular implementation
mod r#impl;

pub use r#impl::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EmbeddingModel, ModelConfig};
    use ndarray::Array1;

    #[test]
    fn test_cross_modal_config_default() {
        let config = CrossModalConfig::default();
        assert_eq!(config.text_dim, 768);
        assert_eq!(config.kg_dim, 128);
        assert_eq!(config.unified_dim, 512);
        assert_eq!(config.contrastive_config.temperature, 0.07);
    }

    #[test]
    fn test_multimodal_embedding_creation() {
        let config = CrossModalConfig::default();
        let model = MultiModalEmbedding::new(config);

        assert_eq!(model.model_type(), "MultiModalEmbedding");
        assert!(!model.is_trained());
        assert_eq!(model.text_embeddings.len(), 0);
        assert_eq!(model.kg_embeddings.len(), 0);
    }

    #[test]
    fn test_text_encoder() {
        let encoder = TextEncoder::new("BERT".to_string(), 768, 512);
        let embedding = encoder.encode("This is a test sentence").unwrap();
        assert_eq!(embedding.len(), 512);
    }

    #[test]
    fn test_kg_encoder() {
        let encoder = KGEncoder::new("ComplEx".to_string(), 128, 128, 512);
        let entity_emb = Array1::from_vec(vec![0.1; 128]);
        let encoded = encoder.encode_entity(&entity_emb).unwrap();
        assert_eq!(encoded.len(), 512);
    }

    #[test]
    fn test_alignment_network() {
        let network = AlignmentNetwork::new("CrossModalAttention".to_string(), 512, 512, 256, 512);
        let text_emb = Array1::from_vec(vec![0.1; 512]);
        let kg_emb = Array1::from_vec(vec![0.2; 512]);

        let (unified, score) = network.align(&text_emb, &kg_emb).unwrap();
        assert_eq!(unified.len(), 512);
        assert!((-1.0..=1.0).contains(&score));
    }

    #[tokio::test]
    async fn test_multimodal_training() {
        let config = CrossModalConfig::default();
        let mut model = MultiModalEmbedding::new(config);

        // Add some training data
        model.add_text_kg_alignment("This is a person", "http://example.org/Person");
        model.add_entity_description("http://example.org/Person", "A human being");
        model.add_property_text("http://example.org/knows", "knows relationship");

        let stats = model.train(Some(10)).await.unwrap();

        assert!(model.is_trained());
        assert_eq!(stats.epochs_completed, 10);
        assert!(stats.training_time_seconds > 0.0);
    }

    #[tokio::test]
    async fn test_unified_embedding_generation() {
        let config = CrossModalConfig::default();
        let mut model = MultiModalEmbedding::new(config);

        let unified = model
            .generate_unified_embedding("A scientist working on AI", "http://example.org/Scientist")
            .await
            .unwrap();

        assert_eq!(unified.len(), 512); // unified_dim
        assert!(model
            .text_embeddings
            .contains_key("A scientist working on AI"));
        assert!(model
            .kg_embeddings
            .contains_key("http://example.org/Scientist"));
    }

    #[tokio::test]
    async fn test_zero_shot_prediction() {
        let config = CrossModalConfig::default();
        let mut model = MultiModalEmbedding::new(config);

        // Add some KG embeddings
        let scientist_embedding = model.get_or_create_kg_embedding("scientist").unwrap();
        let doctor_embedding = model.get_or_create_kg_embedding("doctor").unwrap();
        let teacher_embedding = model.get_or_create_kg_embedding("teacher").unwrap();

        model
            .kg_embeddings
            .insert("scientist".to_string(), scientist_embedding);
        model
            .kg_embeddings
            .insert("doctor".to_string(), doctor_embedding);
        model
            .kg_embeddings
            .insert("teacher".to_string(), teacher_embedding);

        let candidates = vec![
            "scientist".to_string(),
            "doctor".to_string(),
            "teacher".to_string(),
        ];
        let predictions = model
            .zero_shot_prediction("A person who does research", &candidates)
            .await
            .unwrap();

        assert_eq!(predictions.len(), 3);
        assert!(predictions[0].1 >= predictions[1].1); // Scores should be sorted
    }

    #[test]
    fn test_contrastive_loss() {
        let config = CrossModalConfig::default();
        let mut model = MultiModalEmbedding::new(config);

        // Add some embeddings - text embeddings are 512-dim, kg embeddings are raw 128-dim
        model.text_embeddings.insert(
            "positive text".to_string(),
            Array1::from_vec(vec![1.0; 512]),
        );
        model.kg_embeddings.insert(
            "positive_entity".to_string(),
            Array1::from_vec(vec![1.0; 128]),
        );
        model.text_embeddings.insert(
            "negative text".to_string(),
            Array1::from_vec(vec![-1.0; 512]),
        );
        model.kg_embeddings.insert(
            "negative_entity".to_string(),
            Array1::from_vec(vec![-1.0; 128]),
        );

        let positive_pairs = vec![("positive text".to_string(), "positive_entity".to_string())];
        let negative_pairs = vec![("positive text".to_string(), "negative_entity".to_string())];

        let loss = model
            .contrastive_loss(&positive_pairs, &negative_pairs)
            .unwrap();
        assert!(loss >= 0.0);
    }

    #[tokio::test]
    async fn test_few_shot_learning() {
        let config = CrossModalConfig {
            base_config: ModelConfig {
                dimensions: 128, // Match kg_dim for consistency
                ..Default::default()
            },
            text_dim: 128,    // Use consistent dimensions
            kg_dim: 128,      // Keep original
            unified_dim: 128, // Use consistent dimensions
            ..Default::default()
        };
        let model = MultiModalEmbedding::new(config);

        // Create support examples (training data for few-shot learning)
        let support_examples = vec![
            (
                "Scientists study biology".to_string(),
                "scientist".to_string(),
                "profession".to_string(),
            ),
            (
                "Doctors treat patients".to_string(),
                "doctor".to_string(),
                "profession".to_string(),
            ),
            (
                "Dogs are pets".to_string(),
                "dog".to_string(),
                "animal".to_string(),
            ),
            (
                "Cats meow loudly".to_string(),
                "cat".to_string(),
                "animal".to_string(),
            ),
        ];

        // Create query examples (test data)
        let query_examples = vec![
            (
                "Teachers educate students".to_string(),
                "teacher".to_string(),
            ),
            ("Birds fly in the sky".to_string(), "bird".to_string()),
        ];

        let predictions = model
            .few_shot_learn(&support_examples, &query_examples)
            .await
            .unwrap();

        assert_eq!(predictions.len(), 2);
        assert!(predictions[0].1 >= 0.0 && predictions[0].1 <= 1.0); // Valid confidence score
        assert!(predictions[1].1 >= 0.0 && predictions[1].1 <= 1.0);
    }

    #[test]
    fn test_few_shot_learning_components() {
        let few_shot = FewShotLearning::default();
        assert_eq!(few_shot.support_size, 5);
        assert_eq!(few_shot.query_size, 15);
        assert_eq!(few_shot.num_ways, 3);
        assert!(matches!(
            few_shot.meta_algorithm,
            MetaAlgorithm::PrototypicalNetworks
        ));
    }

    #[test]
    fn test_prototype_computation() {
        let few_shot = FewShotLearning::default();
        let embeddings = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![2.0, 3.0, 4.0]),
            Array1::from_vec(vec![3.0, 4.0, 5.0]),
        ];

        let prototype = few_shot.compute_prototype(&embeddings).unwrap();
        assert_eq!(prototype.len(), 3);
        assert!((prototype[0] - 2.0).abs() < 1e-6); // Mean should be 2.0
        assert!((prototype[1] - 3.0).abs() < 1e-6); // Mean should be 3.0
        assert!((prototype[2] - 4.0).abs() < 1e-6); // Mean should be 4.0
    }

    #[test]
    fn test_distance_metrics() {
        let few_shot = FewShotLearning::default();
        let emb1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let emb2 = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        let euclidean_dist = few_shot.compute_distance(&emb1, &emb2);
        assert!((euclidean_dist - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_real_time_finetuning() {
        let config = CrossModalConfig::default();
        let mut model = MultiModalEmbedding::new(config);

        let loss = model
            .real_time_update("New scientific discovery", "researcher", "profession")
            .await
            .unwrap();

        assert!(loss >= 0.0);
    }

    #[test]
    fn test_real_time_finetuning_components() {
        let mut rt_finetuning = RealTimeFinetuning::default();

        rt_finetuning.add_example(
            "Example text".to_string(),
            "example_entity".to_string(),
            "example_label".to_string(),
        );

        assert_eq!(rt_finetuning.online_buffer.len(), 1);
        assert_eq!(rt_finetuning.update_count, 1);
        assert!(!rt_finetuning.should_update()); // Shouldn't update after just 1 example
    }

    #[test]
    fn test_ewc_config() {
        let ewc_config = EWCConfig::default();
        assert_eq!(ewc_config.lambda, 0.1);
        assert!(ewc_config.fisher_information.is_empty());
        assert!(ewc_config.optimal_params.is_empty());
    }
}
