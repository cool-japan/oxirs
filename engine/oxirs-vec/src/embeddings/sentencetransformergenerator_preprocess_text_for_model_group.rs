//! # SentenceTransformerGenerator - preprocess_text_for_model_group Methods
//!
//! This module contains method implementations for `SentenceTransformerGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::TransformerModelType;

use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

impl SentenceTransformerGenerator {
    /// Preprocess text according to model-specific requirements
    pub(super) fn preprocess_text_for_model(&self, text: &str, max_len: usize) -> Result<String> {
        let processed = match &self.model_type {
            TransformerModelType::BERT => {
                let truncated = if text.len() > max_len - 20 {
                    &text[..max_len - 20]
                } else {
                    text
                };
                format!("[CLS] {} [SEP]", truncated.to_lowercase())
            }
            TransformerModelType::RoBERTa => {
                let truncated = if text.len() > max_len - 10 {
                    &text[..max_len - 10]
                } else {
                    text
                };
                format!("<s>{truncated}</s>")
            }
            TransformerModelType::DistilBERT => {
                let truncated = if text.len() > max_len - 20 {
                    &text[..max_len - 20]
                } else {
                    text
                };
                format!("[CLS] {} [SEP]", truncated.to_lowercase())
            }
            TransformerModelType::MultiBERT => {
                let truncated = if text.len() > max_len - 20 {
                    &text[..max_len - 20]
                } else {
                    text
                };
                let has_non_latin = !text.is_ascii();
                if has_non_latin {
                    format!("[CLS] {truncated} [SEP]")
                } else {
                    format!("[CLS] {} [SEP]", truncated.to_lowercase())
                }
            }
            TransformerModelType::Custom(_) => {
                let truncated = if text.len() > max_len {
                    &text[..max_len]
                } else {
                    text
                };
                truncated.to_string()
            }
        };
        Ok(processed)
    }
}
