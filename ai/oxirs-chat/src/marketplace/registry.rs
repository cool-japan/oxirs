//! Model registry trait and core types for the model marketplace.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::MarketplaceError;

/// A single model entry in the marketplace registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Unique model identifier, e.g. `"meta-llama/Llama-2-7b-chat-hf"`.
    pub id: String,
    /// Human-friendly display name.
    pub name: String,
    /// Origin of the model.
    pub source: ModelSource,
    /// Classification of the model's primary capability.
    pub model_type: ModelType,
    /// Download / disk size in bytes, `None` if unknown.
    pub size_bytes: Option<u64>,
    /// Searchable tags (e.g. `["chat", "7b", "llama"]`).
    pub tags: Vec<String>,
    /// Short description of the model.
    pub description: String,
    /// Direct download URL if available.
    pub download_url: Option<String>,
}

/// Where the model originates from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelSource {
    /// Hosted on the HuggingFace Hub.
    HuggingFaceHub {
        /// Hub repository ID, e.g. `"meta-llama/Llama-2-7b-chat-hf"`.
        repo_id: String,
    },
    /// Running in a local Ollama server.
    Ollama {
        /// Base URL of the Ollama server, e.g. `"http://localhost:11434"`.
        server_url: String,
        /// Model name as reported by Ollama, e.g. `"llama2"`.
        name: String,
    },
    /// A GGUF file on the local filesystem.
    LocalFile {
        /// Absolute path to the `.gguf` file.
        path: PathBuf,
    },
}

/// The primary capability of a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    /// Dialogue / instruction-following models.
    Chat,
    /// Text completion / base models.
    Completion,
    /// Embedding / representation models.
    Embedding,
    /// Type could not be determined.
    Unknown,
}

/// A source of model entries that can be queried.
///
/// All methods take `&self`, so implementors must use interior mutability if
/// they need shared mutable state.  The trait is `Send + Sync` so marketplace
/// registries can be shared across async tasks.
pub trait ModelRegistry: Send + Sync {
    /// List all models available from this registry.
    fn list_models(&self) -> Result<Vec<ModelEntry>, MarketplaceError>;

    /// Search for models whose name, id, or tags contain `query` (case-insensitive).
    fn search(&self, query: &str) -> Result<Vec<ModelEntry>, MarketplaceError>;

    /// Retrieve a single model by its unique `id`, or `None` if not found.
    fn get_model(&self, id: &str) -> Result<Option<ModelEntry>, MarketplaceError>;

    /// Short human-readable name for this registry, used in log messages and UI.
    fn source_name(&self) -> &'static str;
}
