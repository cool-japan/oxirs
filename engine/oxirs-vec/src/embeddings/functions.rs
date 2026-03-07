//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Vector;
use anyhow::Result;

use super::types::{EmbeddableContent, EmbeddingConfig};

/// Embedding generator trait
pub trait EmbeddingGenerator: Send + Sync + AsAny {
    /// Generate embedding for content
    fn generate(&self, content: &EmbeddableContent) -> Result<Vector>;
    /// Generate embeddings for multiple contents in batch
    fn generate_batch(&self, contents: &[EmbeddableContent]) -> Result<Vec<Vector>> {
        contents.iter().map(|c| self.generate(c)).collect()
    }
    /// Get the embedding dimensions
    fn dimensions(&self) -> usize;
    /// Get the model configuration
    fn config(&self) -> &EmbeddingConfig;
}
/// Extension trait to add downcast functionality
pub trait AsAny {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SentenceTransformerGenerator, TransformerModelType};
    #[test]
    fn test_transformer_model_types() {
        let config = EmbeddingConfig::default();
        let bert = SentenceTransformerGenerator::new(config.clone());
        assert!(matches!(bert.model_type(), TransformerModelType::BERT));
        assert_eq!(bert.dimensions(), 384);
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        assert!(matches!(
            roberta.model_type(),
            TransformerModelType::RoBERTa
        ));
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        assert!(matches!(
            distilbert.model_type(),
            TransformerModelType::DistilBERT
        ));
        assert_eq!(distilbert.dimensions(), 384);
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());
        assert!(matches!(
            multibert.model_type(),
            TransformerModelType::MultiBERT
        ));
    }
    #[test]
    fn test_model_details() {
        let config = EmbeddingConfig::default();
        let bert = SentenceTransformerGenerator::new(config.clone());
        let bert_details = bert.model_details();
        assert_eq!(bert_details.vocab_size, 30522);
        assert_eq!(bert_details.num_layers, 12);
        assert_eq!(bert_details.hidden_size, 768);
        assert!(bert_details.supports_languages.contains(&"en".to_string()));
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        let roberta_details = roberta.model_details();
        assert_eq!(roberta_details.vocab_size, 50265);
        assert_eq!(roberta_details.max_position_embeddings, 514);
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        let distilbert_details = distilbert.model_details();
        assert_eq!(distilbert_details.num_layers, 6);
        assert_eq!(distilbert_details.hidden_size, 384);
        assert!(distilbert_details.model_size_mb < bert_details.model_size_mb);
        assert!(
            distilbert_details.typical_inference_time_ms < bert_details.typical_inference_time_ms
        );
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());
        let multibert_details = multibert.model_details();
        assert_eq!(multibert_details.vocab_size, 120000);
        assert!(multibert_details.supports_languages.len() > 10);
        assert!(multibert_details
            .supports_languages
            .contains(&"zh".to_string()));
        assert!(multibert_details
            .supports_languages
            .contains(&"de".to_string()));
    }
    #[test]
    fn test_language_support() {
        let config = EmbeddingConfig::default();
        let bert = SentenceTransformerGenerator::new(config.clone());
        assert!(bert.supports_language("en"));
        assert!(!bert.supports_language("zh"));
        assert!(!bert.supports_language("de"));
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        assert!(distilbert.supports_language("en"));
        assert!(!distilbert.supports_language("zh"));
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());
        assert!(multibert.supports_language("en"));
        assert!(multibert.supports_language("zh"));
        assert!(multibert.supports_language("de"));
        assert!(multibert.supports_language("fr"));
        assert!(multibert.supports_language("es"));
        assert!(!multibert.supports_language("unknown_lang"));
    }
    #[test]
    fn test_efficiency_ratings() {
        let config = EmbeddingConfig::default();
        let bert = SentenceTransformerGenerator::new(config.clone());
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());
        assert!(distilbert.efficiency_rating() > bert.efficiency_rating());
        assert!(distilbert.efficiency_rating() > roberta.efficiency_rating());
        assert!(distilbert.efficiency_rating() > multibert.efficiency_rating());
        assert!(bert.efficiency_rating() > roberta.efficiency_rating());
        assert!(bert.efficiency_rating() > multibert.efficiency_rating());
        assert!(roberta.efficiency_rating() > multibert.efficiency_rating());
    }
    #[test]
    fn test_inference_time_estimation() {
        let config = EmbeddingConfig::default();
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        let bert = SentenceTransformerGenerator::new(config.clone());
        let short_time_distilbert = distilbert.estimate_inference_time(50);
        let short_time_bert = bert.estimate_inference_time(50);
        let long_time_distilbert = distilbert.estimate_inference_time(500);
        let long_time_bert = bert.estimate_inference_time(500);
        assert!(short_time_distilbert < short_time_bert);
        assert!(long_time_distilbert < long_time_bert);
        assert!(long_time_distilbert > short_time_distilbert);
        assert!(long_time_bert > short_time_bert);
    }
    #[test]
    fn test_model_specific_text_preprocessing() {
        let config = EmbeddingConfig::default();
        let bert = SentenceTransformerGenerator::new(config.clone());
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());
        let text = "Hello World";
        let bert_processed = bert.preprocess_text_for_model(text, 512).unwrap();
        assert!(bert_processed.contains("[CLS]"));
        assert!(bert_processed.contains("[SEP]"));
        assert!(bert_processed.contains("hello world"));
        let roberta_processed = roberta.preprocess_text_for_model(text, 512).unwrap();
        assert!(roberta_processed.contains("<s>"));
        assert!(roberta_processed.contains("</s>"));
        assert!(roberta_processed.contains("Hello World"));
        let latin_text = "Hello World";
        let chinese_text = "你好世界";
        let latin_processed = multibert
            .preprocess_text_for_model(latin_text, 512)
            .unwrap();
        let chinese_processed = multibert
            .preprocess_text_for_model(chinese_text, 512)
            .unwrap();
        assert!(latin_processed.contains("hello world"));
        assert!(chinese_processed.contains("你好世界"));
    }
    #[test]
    fn test_embedding_generation_differences() {
        let config = EmbeddingConfig::default();
        let bert = SentenceTransformerGenerator::new(config.clone());
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        let content = EmbeddableContent::Text("This is a test sentence".to_string());
        let bert_embedding = bert.generate(&content).unwrap();
        let roberta_embedding = roberta.generate(&content).unwrap();
        let distilbert_embedding = distilbert.generate(&content).unwrap();
        assert_ne!(bert_embedding.as_f32(), roberta_embedding.as_f32());
        assert_ne!(bert_embedding.as_f32(), distilbert_embedding.as_f32());
        assert_ne!(roberta_embedding.as_f32(), distilbert_embedding.as_f32());
        assert_eq!(distilbert_embedding.dimensions, 384);
        assert_eq!(bert_embedding.dimensions, 384);
        assert_eq!(roberta_embedding.dimensions, 384);
        if config.normalize {
            let bert_magnitude: f32 = bert_embedding
                .as_f32()
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            let roberta_magnitude: f32 = roberta_embedding
                .as_f32()
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            let distilbert_magnitude: f32 = distilbert_embedding
                .as_f32()
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            assert!((bert_magnitude - 1.0).abs() < 0.1);
            assert!((roberta_magnitude - 1.0).abs() < 0.1);
            assert!((distilbert_magnitude - 1.0).abs() < 0.1);
        }
    }
    #[test]
    fn test_tokenization_differences() {
        let config = EmbeddingConfig::default();
        let bert = SentenceTransformerGenerator::new(config.clone());
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());
        let model_details_bert = bert.get_model_details();
        let model_details_roberta = roberta.get_model_details();
        let model_details_multibert = multibert.get_model_details();
        let complex_word = "preprocessing";
        let bert_tokens =
            bert.simulate_wordpiece_tokenization(complex_word, model_details_bert.vocab_size);
        let roberta_tokens =
            roberta.simulate_bpe_tokenization(complex_word, model_details_roberta.vocab_size);
        let multibert_tokens = multibert
            .simulate_multilingual_tokenization(complex_word, model_details_multibert.vocab_size);
        assert!(roberta_tokens.len() >= bert_tokens.len());
        assert!(multibert_tokens.len() <= bert_tokens.len());
        for token in &bert_tokens {
            assert!(*token < model_details_bert.vocab_size as u32);
        }
        for token in &roberta_tokens {
            assert!(*token < model_details_roberta.vocab_size as u32);
        }
        for token in &multibert_tokens {
            assert!(*token < model_details_multibert.vocab_size as u32);
        }
    }
    #[test]
    fn test_model_size_comparisons() {
        let config = EmbeddingConfig::default();
        let bert = SentenceTransformerGenerator::new(config.clone());
        let roberta = SentenceTransformerGenerator::roberta(config.clone());
        let distilbert = SentenceTransformerGenerator::distilbert(config.clone());
        let multibert = SentenceTransformerGenerator::multilingual_bert(config.clone());
        let bert_size = bert.model_size_mb();
        let roberta_size = roberta.model_size_mb();
        let distilbert_size = distilbert.model_size_mb();
        let multibert_size = multibert.model_size_mb();
        assert!(distilbert_size < bert_size);
        assert!(distilbert_size < roberta_size);
        assert!(distilbert_size < multibert_size);
        assert!(multibert_size > bert_size);
        assert!(multibert_size > roberta_size);
        assert!(multibert_size > distilbert_size);
        assert!(roberta_size > bert_size);
    }
}
