//! # SentenceTransformerGenerator - estimate_inference_time_group Methods
//!
//! This module contains method implementations for `SentenceTransformerGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

impl SentenceTransformerGenerator {
    /// Get the estimated inference time for a given text length
    pub fn estimate_inference_time(&self, text_length: usize) -> u64 {
        let details = self.get_model_details();
        let base_time = details.typical_inference_time_ms;
        let length_factor = (text_length as f64 / 100.0).sqrt().max(1.0);
        (base_time as f64 * length_factor) as u64
    }
}
