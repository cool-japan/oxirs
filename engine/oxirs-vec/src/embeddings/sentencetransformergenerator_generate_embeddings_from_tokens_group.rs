//! # SentenceTransformerGenerator - generate_embeddings_from_tokens_group Methods
//!
//! This module contains method implementations for `SentenceTransformerGenerator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::{ModelDetails, TransformerModelType};

use super::sentencetransformergenerator_type::SentenceTransformerGenerator;

impl SentenceTransformerGenerator {
    /// Generate embeddings from token IDs using model-specific patterns
    pub(super) fn generate_embeddings_from_tokens(
        &self,
        token_ids: &[u32],
        dimensions: usize,
        model_details: &ModelDetails,
    ) -> Result<Vec<f32>> {
        let mut values = vec![0.0; dimensions];
        match &self.model_type {
            TransformerModelType::BERT => {
                self.generate_bert_style_embeddings(token_ids, &mut values, model_details)
            }
            TransformerModelType::RoBERTa => {
                self.generate_roberta_style_embeddings(token_ids, &mut values, model_details)
            }
            TransformerModelType::DistilBERT => {
                self.generate_distilbert_style_embeddings(token_ids, &mut values, model_details)
            }
            TransformerModelType::MultiBERT => {
                self.generate_multibert_style_embeddings(token_ids, &mut values, model_details)
            }
            TransformerModelType::Custom(_) => {
                self.generate_custom_style_embeddings(token_ids, &mut values, model_details)
            }
        }
        Ok(values)
    }
    /// Generate BERT-style embeddings
    fn generate_bert_style_embeddings(
        &self,
        token_ids: &[u32],
        values: &mut [f32],
        _model_details: &ModelDetails,
    ) {
        for (i, &token_id) in token_ids.iter().enumerate() {
            let mut seed = token_id as u64;
            for value in values.iter_mut() {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = (seed as f32) / (u64::MAX as f32);
                let position_encoding =
                    ((i as f32 / 512.0) * 2.0 * std::f32::consts::PI).sin() * 0.1;
                *value += ((normalized - 0.5) * 2.0) + position_encoding;
            }
        }
        if !token_ids.is_empty() {
            for value in values.iter_mut() {
                *value /= token_ids.len() as f32;
            }
        }
    }
    /// Generate RoBERTa-style embeddings (no segment embeddings, different position encoding)
    fn generate_roberta_style_embeddings(
        &self,
        token_ids: &[u32],
        values: &mut [f32],
        _model_details: &ModelDetails,
    ) {
        for (i, &token_id) in token_ids.iter().enumerate() {
            let mut seed = token_id.wrapping_mul(31415927);
            for value in values.iter_mut() {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = (seed as f32) / (u64::MAX as f32);
                let position_encoding =
                    ((i as f32 + 2.0) / 514.0 * 2.0 * std::f32::consts::PI).cos() * 0.1;
                *value += ((normalized - 0.5) * 2.0) + position_encoding;
            }
        }
        if !token_ids.is_empty() {
            for value in values.iter_mut() {
                *value /= token_ids.len() as f32;
            }
        }
    }
    /// Generate DistilBERT-style embeddings (simpler, faster)
    fn generate_distilbert_style_embeddings(
        &self,
        token_ids: &[u32],
        values: &mut [f32],
        _model_details: &ModelDetails,
    ) {
        for (i, &token_id) in token_ids.iter().enumerate() {
            let mut seed = token_id as u64;
            for value in values.iter_mut() {
                seed = seed.wrapping_mul(982451653).wrapping_add(12345);
                let normalized = (seed as f32) / (u64::MAX as f32);
                let position_encoding = (i as f32 / 512.0).sin() * 0.05;
                *value += ((normalized - 0.5) * 1.5) + position_encoding;
            }
        }
        if !token_ids.is_empty() {
            for value in values.iter_mut() {
                *value /= token_ids.len() as f32;
            }
        }
    }
    /// Generate multilingual BERT-style embeddings
    fn generate_multibert_style_embeddings(
        &self,
        token_ids: &[u32],
        values: &mut [f32],
        _model_details: &ModelDetails,
    ) {
        for (i, &token_id) in token_ids.iter().enumerate() {
            let mut seed = token_id.wrapping_mul(2654435761);
            for j in 0..values.len() {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = (seed as f32) / (u64::MAX as f32);
                let position_encoding =
                    ((i as f32 / 512.0) * 2.0 * std::f32::consts::PI).sin() * 0.08;
                let cross_lingual_bias =
                    (j as f32 / values.len() as f32 * std::f32::consts::PI).cos() * 0.05;
                values[j] += ((normalized - 0.5) * 1.8) + position_encoding + cross_lingual_bias;
            }
        }
        if !token_ids.is_empty() {
            for value in values.iter_mut() {
                *value /= token_ids.len() as f32;
            }
        }
    }
    /// Generate custom model embeddings
    fn generate_custom_style_embeddings(
        &self,
        token_ids: &[u32],
        values: &mut [f32],
        _model_details: &ModelDetails,
    ) {
        for &token_id in token_ids {
            let mut seed = token_id as u64;
            for value in values.iter_mut() {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let normalized = (seed as f32) / (u64::MAX as f32);
                *value += (normalized - 0.5) * 2.0;
            }
        }
        if !token_ids.is_empty() {
            for value in values.iter_mut() {
                *value /= token_ids.len() as f32;
            }
        }
    }
}
