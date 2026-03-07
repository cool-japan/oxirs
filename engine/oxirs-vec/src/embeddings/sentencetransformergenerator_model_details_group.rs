//! # SentenceTransformerGenerator - model_details_group Methods
//!
//! This module contains method implementations for `SentenceTransformerGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ModelDetails, TransformerModelType};

use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

impl SentenceTransformerGenerator {
    /// Get detailed information about the current model
    pub fn model_details(&self) -> ModelDetails {
        self.get_model_details()
    }
    /// Get model-specific vocabulary size and training details
    pub(super) fn get_model_details(&self) -> ModelDetails {
        match &self.model_type {
            TransformerModelType::BERT => ModelDetails {
                vocab_size: 30522,
                num_layers: 12,
                num_attention_heads: 12,
                hidden_size: 768,
                intermediate_size: 3072,
                max_position_embeddings: 512,
                supports_languages: vec!["en".to_string()],
                model_size_mb: 440,
                typical_inference_time_ms: 50,
            },
            TransformerModelType::RoBERTa => ModelDetails {
                vocab_size: 50265,
                num_layers: 12,
                num_attention_heads: 12,
                hidden_size: 768,
                intermediate_size: 3072,
                max_position_embeddings: 514,
                supports_languages: vec!["en".to_string()],
                model_size_mb: 470,
                typical_inference_time_ms: 55,
            },
            TransformerModelType::DistilBERT => ModelDetails {
                vocab_size: 30522,
                num_layers: 6,
                num_attention_heads: 12,
                hidden_size: 384,
                intermediate_size: 1536,
                max_position_embeddings: 512,
                supports_languages: vec!["en".to_string()],
                model_size_mb: 250,
                typical_inference_time_ms: 25,
            },
            TransformerModelType::MultiBERT => ModelDetails {
                vocab_size: 120000,
                num_layers: 12,
                num_attention_heads: 12,
                hidden_size: 768,
                intermediate_size: 3072,
                max_position_embeddings: 512,
                supports_languages: vec![
                    "en".to_string(),
                    "de".to_string(),
                    "fr".to_string(),
                    "es".to_string(),
                    "it".to_string(),
                    "pt".to_string(),
                    "ru".to_string(),
                    "zh".to_string(),
                    "ja".to_string(),
                    "ko".to_string(),
                    "ar".to_string(),
                    "hi".to_string(),
                    "th".to_string(),
                    "tr".to_string(),
                    "pl".to_string(),
                    "nl".to_string(),
                    "sv".to_string(),
                    "da".to_string(),
                    "no".to_string(),
                    "fi".to_string(),
                ],
                model_size_mb: 670,
                typical_inference_time_ms: 70,
            },
            TransformerModelType::Custom(_path) => ModelDetails {
                vocab_size: 50000,
                num_layers: 12,
                num_attention_heads: 12,
                hidden_size: self.config.dimensions,
                intermediate_size: self.config.dimensions * 4,
                max_position_embeddings: self.config.max_sequence_length,
                supports_languages: vec!["unknown".to_string()],
                model_size_mb: 500,
                typical_inference_time_ms: 60,
            },
        }
    }
}
