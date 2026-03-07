//! # SentenceTransformerGenerator - builders Methods
//!
//! This module contains method implementations for `SentenceTransformerGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{EmbeddingConfig, TransformerModelType};

use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

impl SentenceTransformerGenerator {
    pub fn with_model_type(config: EmbeddingConfig, model_type: TransformerModelType) -> Self {
        Self { config, model_type }
    }
    /// Create a new RoBERTa model generator
    pub fn roberta(config: EmbeddingConfig) -> Self {
        Self::with_model_type(config, TransformerModelType::RoBERTa)
    }
    /// Create a new DistilBERT model generator
    pub fn distilbert(config: EmbeddingConfig) -> Self {
        let adjusted_config = EmbeddingConfig {
            dimensions: 384,
            ..config
        };
        Self::with_model_type(adjusted_config, TransformerModelType::DistilBERT)
    }
    /// Create a new multilingual BERT model generator
    pub fn multilingual_bert(config: EmbeddingConfig) -> Self {
        Self::with_model_type(config, TransformerModelType::MultiBERT)
    }
}
