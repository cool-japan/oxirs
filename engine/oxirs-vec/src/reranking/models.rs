//! Cross-encoder model backends

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelBackend {
    Local,
    Api,
    Remote,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub backend: ModelBackend,
    pub model_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEncoderModel {
    config: ModelConfig,
}

impl CrossEncoderModel {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}
