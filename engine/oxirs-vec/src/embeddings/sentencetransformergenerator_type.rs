//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{EmbeddingConfig, TransformerModelType};

/// Transformer-based embedding generator supporting multiple models
pub struct SentenceTransformerGenerator {
    pub(super) config: EmbeddingConfig,
    pub(super) model_type: TransformerModelType,
}
