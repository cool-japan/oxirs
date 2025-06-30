//! Builders for specialized transformer embedding models

use super::types::{PoolingStrategy, TransformerConfig, TransformerType};
use crate::ModelConfig;

/// Builder for creating specialized transformer embedding models
pub struct TransformerBuilder;

impl TransformerBuilder {
    /// Create BERT configuration for general text embedding
    pub fn bert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::BERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::CLS,
            fine_tune: false,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 1000,
            gradient_accumulation_steps: 1,
            normalize_embeddings: false,
        }
    }

    /// Create RoBERTa configuration with improved training
    pub fn roberta_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::RoBERTa,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true,
            learning_rate_schedule: "cosine".to_string(),
            warmup_steps: 500,
            gradient_accumulation_steps: 2,
            normalize_embeddings: true,
        }
    }

    /// Create Sentence-BERT configuration for sentence embeddings
    pub fn sentence_bert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::SentenceBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: false,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 0,
            gradient_accumulation_steps: 1,
            normalize_embeddings: true,
        }
    }

    /// Create SciBERT configuration for scientific text
    pub fn scibert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::SciBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 1000,
            gradient_accumulation_steps: 2,
            normalize_embeddings: true,
        }
    }

    /// Create BioBERT configuration for biomedical text
    pub fn biobert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::BioBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true,
            learning_rate_schedule: "cosine".to_string(),
            warmup_steps: 800,
            gradient_accumulation_steps: 4,
            normalize_embeddings: true,
        }
    }

    /// Create CodeBERT configuration for code understanding
    pub fn codebert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::CodeBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 600,
            gradient_accumulation_steps: 2,
            normalize_embeddings: true,
        }
    }

    /// Create LegalBERT configuration for legal text
    pub fn legalbert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::LegalBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true,
            learning_rate_schedule: "polynomial".to_string(),
            warmup_steps: 500,
            gradient_accumulation_steps: 2,
            normalize_embeddings: true,
        }
    }

    /// Create NewsBERT configuration for news and journalism
    pub fn newsbert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::NewsBERT,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 500,
            gradient_accumulation_steps: 2,
            normalize_embeddings: true,
        }
    }

    /// Create SocialMediaBERT configuration for social media text
    pub fn social_media_bert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::SocialMediaBERT,
            max_sequence_length: 280, // Twitter-like length
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Max,
            fine_tune: true,
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 400,
            gradient_accumulation_steps: 1,
            normalize_embeddings: true,
        }
    }

    /// Create multilingual BERT configuration
    pub fn multilingual_bert_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::MBert,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true,
            learning_rate_schedule: "cosine".to_string(),
            warmup_steps: 1000,
            gradient_accumulation_steps: 2,
            normalize_embeddings: true,
        }
    }

    /// Create XLM-RoBERTa configuration for cross-lingual tasks
    pub fn xlm_roberta_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::XLMR,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true,
            learning_rate_schedule: "cosine".to_string(),
            warmup_steps: 1200,
            gradient_accumulation_steps: 4,
            normalize_embeddings: true,
        }
    }

    /// Create custom configuration with specified parameters
    pub fn custom_config(
        transformer_type: TransformerType,
        dimensions: usize,
        max_sequence_length: usize,
        pooling_strategy: PoolingStrategy,
        fine_tune: bool,
        learning_rate_schedule: String,
    ) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type,
            max_sequence_length,
            use_pooling: true,
            pooling_strategy,
            fine_tune,
            learning_rate_schedule,
            warmup_steps: if fine_tune { 500 } else { 0 },
            gradient_accumulation_steps: if fine_tune { 2 } else { 1 },
            normalize_embeddings: true,
        }
    }

    /// Create configuration optimized for knowledge graphs
    pub fn knowledge_graph_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::SentenceBERT,
            max_sequence_length: 256, // Shorter sequences for entities/relations
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: true,
            learning_rate_schedule: "cosine".to_string(),
            warmup_steps: 300,
            gradient_accumulation_steps: 1,
            normalize_embeddings: true,
        }
    }

    /// Create configuration for large documents
    pub fn long_document_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::RoBERTa,
            max_sequence_length: 2048, // Longer sequences
            use_pooling: true,
            pooling_strategy: PoolingStrategy::AttentionWeighted,
            fine_tune: true,
            learning_rate_schedule: "polynomial".to_string(),
            warmup_steps: 1500,
            gradient_accumulation_steps: 8, // Larger batches for long docs
            normalize_embeddings: true,
        }
    }

    /// Create configuration for real-time applications
    pub fn realtime_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::SentenceBERT,
            max_sequence_length: 128, // Shorter for speed
            use_pooling: true,
            pooling_strategy: PoolingStrategy::Mean,
            fine_tune: false, // Pre-trained for speed
            learning_rate_schedule: "linear".to_string(),
            warmup_steps: 0,
            gradient_accumulation_steps: 1,
            normalize_embeddings: true,
        }
    }

    /// Create configuration for high-accuracy applications
    pub fn high_accuracy_config(dimensions: usize) -> TransformerConfig {
        TransformerConfig {
            base_config: ModelConfig::default().with_dimensions(dimensions),
            transformer_type: TransformerType::RoBERTa,
            max_sequence_length: 512,
            use_pooling: true,
            pooling_strategy: PoolingStrategy::AttentionWeighted,
            fine_tune: true,
            learning_rate_schedule: "cosine".to_string(),
            warmup_steps: 2000,
            gradient_accumulation_steps: 4,
            normalize_embeddings: true,
        }
    }

    /// Get recommended configuration based on use case
    pub fn get_recommended_config(use_case: &str, dimensions: usize) -> TransformerConfig {
        match use_case.to_lowercase().as_str() {
            "scientific" | "science" => Self::scibert_config(dimensions),
            "biomedical" | "medical" | "biology" => Self::biobert_config(dimensions),
            "code" | "programming" | "software" => Self::codebert_config(dimensions),
            "legal" | "law" | "court" => Self::legalbert_config(dimensions),
            "news" | "journalism" | "media" => Self::newsbert_config(dimensions),
            "social" | "social_media" | "twitter" => Self::social_media_bert_config(dimensions),
            "multilingual" | "cross_language" => Self::multilingual_bert_config(dimensions),
            "knowledge_graph" | "kg" | "ontology" => Self::knowledge_graph_config(dimensions),
            "long_document" | "documents" => Self::long_document_config(dimensions),
            "realtime" | "fast" | "speed" => Self::realtime_config(dimensions),
            "accuracy" | "precise" | "high_quality" => Self::high_accuracy_config(dimensions),
            _ => Self::sentence_bert_config(dimensions), // Default
        }
    }

    /// Get model recommendations for domain
    pub fn get_domain_recommendations(domain: &str) -> Vec<(&'static str, &'static str)> {
        match domain.to_lowercase().as_str() {
            "scientific" => vec![
                ("SciBERT", "Best for scientific literature and terminology"),
                ("RoBERTa", "Good general performance with fine-tuning"),
                ("BERT", "Basic scientific text understanding"),
            ],
            "biomedical" => vec![
                ("BioBERT", "Specialized for biomedical text and terminology"),
                ("SciBERT", "Good for scientific aspects of biomedical text"),
                ("SentenceBERT", "Good for general biomedical sentence similarity"),
            ],
            "code" => vec![
                ("CodeBERT", "Specialized for code understanding and generation"),
                ("RoBERTa", "Good for code comments and documentation"),
                ("BERT", "Basic code-related text understanding"),
            ],
            "legal" => vec![
                ("LegalBERT", "Specialized for legal documents and terminology"),
                ("RoBERTa", "Good for general legal text with fine-tuning"),
                ("BERT", "Basic legal text understanding"),
            ],
            "social_media" => vec![
                ("SocialMediaBERT", "Specialized for social media text and slang"),
                ("RoBERTa", "Good for informal text"),
                ("SentenceBERT", "Good for short text similarity"),
            ],
            _ => vec![
                ("SentenceBERT", "Good general-purpose sentence embeddings"),
                ("RoBERTa", "Robust performance across domains"),
                ("BERT", "Baseline transformer performance"),
            ],
        }
    }

    /// Validate configuration parameters
    pub fn validate_config(config: &TransformerConfig) -> Result<(), String> {
        if config.base_config.dimensions == 0 {
            return Err("Dimensions must be greater than 0".to_string());
        }

        if config.max_sequence_length == 0 {
            return Err("Max sequence length must be greater than 0".to_string());
        }

        if config.gradient_accumulation_steps == 0 {
            return Err("Gradient accumulation steps must be greater than 0".to_string());
        }

        // Check if dimensions match transformer type
        let expected_dim = config.transformer_type.default_dimensions();
        if config.base_config.dimensions != expected_dim && 
           config.base_config.dimensions != 768 && 
           config.base_config.dimensions != 512 && 
           config.base_config.dimensions != 384 {
            return Err(format!(
                "Unusual dimensions {} for {:?}. Expected {} or common sizes (768, 512, 384)",
                config.base_config.dimensions, config.transformer_type, expected_dim
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_config_creation() {
        let config = TransformerBuilder::bert_config(768);
        assert_eq!(config.transformer_type, TransformerType::BERT);
        assert_eq!(config.base_config.dimensions, 768);
        assert_eq!(config.pooling_strategy, PoolingStrategy::CLS);
        assert!(!config.fine_tune);
    }

    #[test]
    fn test_sentence_bert_config() {
        let config = TransformerBuilder::sentence_bert_config(384);
        assert_eq!(config.transformer_type, TransformerType::SentenceBERT);
        assert_eq!(config.base_config.dimensions, 384);
        assert_eq!(config.pooling_strategy, PoolingStrategy::Mean);
        assert!(config.normalize_embeddings);
    }

    #[test]
    fn test_domain_specific_configs() {
        let sci_config = TransformerBuilder::scibert_config(768);
        assert_eq!(sci_config.transformer_type, TransformerType::SciBERT);
        assert!(sci_config.fine_tune);

        let bio_config = TransformerBuilder::biobert_config(768);
        assert_eq!(bio_config.transformer_type, TransformerType::BioBERT);
        assert_eq!(bio_config.learning_rate_schedule, "cosine");

        let code_config = TransformerBuilder::codebert_config(768);
        assert_eq!(code_config.transformer_type, TransformerType::CodeBERT);
        assert_eq!(code_config.warmup_steps, 600);
    }

    #[test]
    fn test_custom_config() {
        let config = TransformerBuilder::custom_config(
            TransformerType::RoBERTa,
            512,
            256,
            PoolingStrategy::Max,
            true,
            "cosine".to_string(),
        );

        assert_eq!(config.transformer_type, TransformerType::RoBERTa);
        assert_eq!(config.base_config.dimensions, 512);
        assert_eq!(config.max_sequence_length, 256);
        assert_eq!(config.pooling_strategy, PoolingStrategy::Max);
        assert!(config.fine_tune);
    }

    #[test]
    fn test_recommended_configs() {
        let scientific = TransformerBuilder::get_recommended_config("scientific", 768);
        assert_eq!(scientific.transformer_type, TransformerType::SciBERT);

        let code = TransformerBuilder::get_recommended_config("code", 768);
        assert_eq!(code.transformer_type, TransformerType::CodeBERT);

        let default = TransformerBuilder::get_recommended_config("unknown", 384);
        assert_eq!(default.transformer_type, TransformerType::SentenceBERT);
    }

    #[test]
    fn test_specialized_configs() {
        let kg_config = TransformerBuilder::knowledge_graph_config(256);
        assert_eq!(kg_config.max_sequence_length, 256);
        assert!(kg_config.fine_tune);

        let realtime_config = TransformerBuilder::realtime_config(128);
        assert_eq!(realtime_config.max_sequence_length, 128);
        assert!(!realtime_config.fine_tune);

        let long_doc_config = TransformerBuilder::long_document_config(768);
        assert_eq!(long_doc_config.max_sequence_length, 2048);
        assert_eq!(long_doc_config.gradient_accumulation_steps, 8);
    }

    #[test]
    fn test_config_validation() {
        let valid_config = TransformerBuilder::bert_config(768);
        assert!(TransformerBuilder::validate_config(&valid_config).is_ok());

        let mut invalid_config = valid_config.clone();
        invalid_config.base_config.dimensions = 0;
        assert!(TransformerBuilder::validate_config(&invalid_config).is_err());

        invalid_config = valid_config.clone();
        invalid_config.max_sequence_length = 0;
        assert!(TransformerBuilder::validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_domain_recommendations() {
        let sci_recs = TransformerBuilder::get_domain_recommendations("scientific");
        assert!(!sci_recs.is_empty());
        assert!(sci_recs[0].0.contains("SciBERT"));

        let code_recs = TransformerBuilder::get_domain_recommendations("code");
        assert!(!code_recs.is_empty());
        assert!(code_recs[0].0.contains("CodeBERT"));

        let default_recs = TransformerBuilder::get_domain_recommendations("unknown");
        assert!(!default_recs.is_empty());
    }
}