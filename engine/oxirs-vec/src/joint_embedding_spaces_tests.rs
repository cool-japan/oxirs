//! Tests for Joint Embedding Spaces

#[cfg(test)]
mod tests {
    use crate::cross_modal_embeddings::Modality;
    use crate::joint_embedding_spaces::{
        CLIPAligner, CrossModalAttention, JointEmbeddingConfig, JointEmbeddingSpace,
    };
    use crate::Vector;
    use anyhow::Result;

    #[test]
    fn test_joint_embedding_space() -> Result<()> {
        let config = JointEmbeddingConfig::default();
        let joint_space = JointEmbeddingSpace::new(config);

        let text_embedding = Vector::new(vec![0.1; 768]);
        let image_embedding = Vector::new(vec![0.2; 2048]);

        let joint_text = joint_space.project_to_joint_space(Modality::Text, &text_embedding)?;
        let joint_image = joint_space.project_to_joint_space(Modality::Image, &image_embedding)?;

        assert_eq!(joint_text.dimensions, 512);
        assert_eq!(joint_image.dimensions, 512);

        let similarity = joint_space.cross_modal_similarity(
            Modality::Text,
            &text_embedding,
            Modality::Image,
            &image_embedding,
        )?;

        assert!((-1.0..=1.0).contains(&similarity));
        Ok(())
    }

    #[test]
    fn test_cross_modal_attention() -> Result<()> {
        let attention = CrossModalAttention::new(128, 4, 0.1, true);

        let query = Vector::new(vec![0.1; 128]);
        let key = Vector::new(vec![0.2; 128]);
        let value = Vector::new(vec![0.3; 128]);

        let result = attention.cross_attention(&query, &key, &value)?;
        assert_eq!(result.dimensions, 128);
        Ok(())
    }

    #[test]
    fn test_contrastive_learning() -> Result<()> {
        let config = JointEmbeddingConfig::default();
        let mut joint_space = JointEmbeddingSpace::new(config);

        let positive_pairs = vec![(
            Modality::Text,
            Vector::new(vec![0.1; 768]),
            Modality::Image,
            Vector::new(vec![0.1; 2048]),
        )];

        let negative_pairs = vec![(
            Modality::Text,
            Vector::new(vec![0.1; 768]),
            Modality::Image,
            Vector::new(vec![-0.1; 2048]),
        )];

        let loss = joint_space.contrastive_align(&positive_pairs, &negative_pairs)?;

        assert!(loss >= 0.0);
        Ok(())
    }

    #[test]
    fn test_clip_aligner() {
        let config = JointEmbeddingConfig::default();
        let aligner = CLIPAligner::new(config);

        let text_words = vec!["hello", "world"];
        let text_embedding = aligner.create_text_embedding(&text_words);
        assert_eq!(text_embedding.dimensions, 768);

        let (cache_size, _) = aligner.joint_space.get_cache_stats();
        assert_eq!(cache_size, 0);
    }
}
