//! # SentenceTransformerGenerator - accessors Methods
//!
//! This module contains method implementations for `SentenceTransformerGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TransformerModelType;

use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

impl SentenceTransformerGenerator {
    /// Get model-specific configuration adjustments
    pub(super) fn get_model_config(&self) -> (usize, usize, f32) {
        match &self.model_type {
            TransformerModelType::BERT => (self.config.dimensions, 512, 1.0),
            TransformerModelType::RoBERTa => (self.config.dimensions, 514, 0.95),
            TransformerModelType::DistilBERT => (self.config.dimensions, 512, 1.5),
            TransformerModelType::MultiBERT => (self.config.dimensions, 512, 0.8),
            TransformerModelType::Custom(_) => {
                (self.config.dimensions, self.config.max_sequence_length, 1.0)
            }
        }
    }
}
