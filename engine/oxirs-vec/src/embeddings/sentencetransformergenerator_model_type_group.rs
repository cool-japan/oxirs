//! # SentenceTransformerGenerator - model_type_group Methods
//!
//! This module contains method implementations for `SentenceTransformerGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TransformerModelType;

use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

impl SentenceTransformerGenerator {
    /// Get the current model type
    pub fn model_type(&self) -> &TransformerModelType {
        &self.model_type
    }
}
