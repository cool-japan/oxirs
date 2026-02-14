//! # SentenceTransformerGenerator - efficiency_rating_group Methods
//!
//! This module contains method implementations for `SentenceTransformerGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TransformerModelType;

use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

impl SentenceTransformerGenerator {
    /// Get efficiency rating (higher is better/faster)
    pub fn efficiency_rating(&self) -> f32 {
        match &self.model_type {
            TransformerModelType::DistilBERT => 1.5,
            TransformerModelType::BERT => 1.0,
            TransformerModelType::RoBERTa => 0.95,
            TransformerModelType::MultiBERT => 0.8,
            TransformerModelType::Custom(_) => 1.0,
        }
    }
}
