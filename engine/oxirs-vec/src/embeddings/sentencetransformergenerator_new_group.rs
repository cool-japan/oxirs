//! # SentenceTransformerGenerator - new_group Methods
//!
//! This module contains method implementations for `SentenceTransformerGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Vector;
use anyhow::Result;

use super::types::{EmbeddingConfig, ModelDetails, TransformerModelType};

use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

impl SentenceTransformerGenerator {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            model_type: TransformerModelType::default(),
        }
    }
    /// Generate embedding with model-specific processing
    pub(super) fn generate_with_model(&self, text: &str) -> Result<Vector> {
        let _text_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut hasher);
            hasher.finish()
        };
        let (dimensions, max_len, _efficiency) = self.get_model_config();
        let model_details = self.get_model_details();
        let processed_text = self.preprocess_text_for_model(text, max_len)?;
        let token_ids = self.simulate_tokenization(&processed_text, &model_details);
        let values =
            self.generate_embeddings_from_tokens(&token_ids, dimensions, &model_details)?;
        if self.config.normalize {
            let magnitude: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                let mut normalized_values = values;
                for value in &mut normalized_values {
                    *value /= magnitude;
                }
                return Ok(Vector::new(normalized_values));
            }
        }
        Ok(Vector::new(values))
    }
    /// Simulate tokenization process for different models
    pub(super) fn simulate_tokenization(
        &self,
        text: &str,
        model_details: &ModelDetails,
    ) -> Vec<u32> {
        let mut token_ids = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            let subwords = match &self.model_type {
                TransformerModelType::RoBERTa => {
                    self.simulate_bpe_tokenization(word, model_details.vocab_size)
                }
                TransformerModelType::DistilBERT | TransformerModelType::BERT => {
                    self.simulate_wordpiece_tokenization(word, model_details.vocab_size)
                }
                TransformerModelType::MultiBERT => {
                    self.simulate_multilingual_tokenization(word, model_details.vocab_size)
                }
                TransformerModelType::Custom(_) => {
                    vec![self.word_to_token_id(word, model_details.vocab_size)]
                }
            };
            token_ids.extend(subwords);
        }
        token_ids.truncate(model_details.max_position_embeddings - 2);
        token_ids
    }
    /// Simulate BPE tokenization (used by RoBERTa)
    pub(super) fn simulate_bpe_tokenization(&self, word: &str, vocab_size: usize) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut remaining = word;
        while !remaining.is_empty() {
            let chunk_size = if remaining.len() > 4 {
                4
            } else {
                remaining.len()
            };
            let chunk = &remaining[..chunk_size];
            tokens.push(self.word_to_token_id(chunk, vocab_size));
            remaining = &remaining[chunk_size..];
        }
        tokens
    }
    /// Convert word to token ID
    pub(super) fn word_to_token_id(&self, word: &str, vocab_size: usize) -> u32 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        word.hash(&mut hasher);
        (hasher.finish() % vocab_size as u64) as u32
    }
}
