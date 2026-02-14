//! # SentenceTransformerGenerator - simulate_wordpiece_tokenization_group Methods
//!
//! This module contains method implementations for `SentenceTransformerGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

impl SentenceTransformerGenerator {
    /// Simulate WordPiece tokenization (used by BERT)
    pub(super) fn simulate_wordpiece_tokenization(
        &self,
        word: &str,
        vocab_size: usize,
    ) -> Vec<u32> {
        if word.len() <= 6 {
            vec![self.word_to_token_id(word, vocab_size)]
        } else {
            let mid = word.len() / 2;
            vec![
                self.word_to_token_id(&word[..mid], vocab_size),
                self.word_to_token_id(&format!("##{}", &word[mid..]), vocab_size),
            ]
        }
    }
}
