//! # SentenceTransformerGenerator - supports_language_group Methods
//!
//! This module contains method implementations for `SentenceTransformerGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

impl SentenceTransformerGenerator {
    /// Check if the model supports a specific language
    pub fn supports_language(&self, language_code: &str) -> bool {
        let details = self.get_model_details();
        details
            .supports_languages
            .contains(&language_code.to_string())
    }
}
