//! # SentenceTransformerGenerator - model_size_mb_group Methods
//!
//! This module contains method implementations for `SentenceTransformerGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

impl SentenceTransformerGenerator {
    /// Get the memory footprint of the model in MB
    pub fn model_size_mb(&self) -> usize {
        self.get_model_details().model_size_mb
    }
}
