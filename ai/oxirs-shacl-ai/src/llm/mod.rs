//! LLM integration for SHACL constraint generation and completion APIs.
//!
//! Provides provider-agnostic abstractions for invoking large language models
//! to produce SHACL shapes from natural-language descriptions, ontology
//! fragments, or example RDF triples.
//!
//! ## Modules
//!
//! - [`constraint_generator`]: Original SHACL-specific LLM abstraction (flat prompt/response).
//! - [`provider`]: Newer chat-completion abstraction (multi-turn messages).
//! - [`local`]: Deterministic offline provider for testing.
//! - [`prompt`]: SHACL-specific prompt builders.
//! - [`openai`]: OpenAI backend (feature `llm-network`).
//! - [`anthropic`]: Anthropic backend (feature `llm-network`).

pub mod constraint_generator;

// Newer completion-API abstractions
#[cfg(feature = "llm-network")]
pub mod anthropic;
pub mod local;
#[cfg(feature = "llm-network")]
pub mod openai;
pub mod prompt;
pub mod provider;

// Re-export the original constraint-generator types (unchanged).
pub use constraint_generator::{
    BatchGenerationRequest, BatchItemResult, GeneratedShaclShape, GeneratorStats,
    LlmConstraintGenerator, LlmConstraintGeneratorConfig, LlmProvider, LlmRequest, LlmResponse,
    PromptTemplate, StubLlmProvider, TokenUsage,
};

// Re-export the newer completion-API types.
// `CompletionProvider` is the new trait (aliased to avoid collision with the
// legacy `LlmProvider` name at crate root).
// `TokenUsage` already exported above from constraint_generator; the new one
// is re-exported under an alias.
pub use local::LocalProvider;
pub use prompt::ShaclPrompts;
pub use provider::{
    Capabilities, CompletionProvider, CompletionRequest, CompletionResponse, LlmError, Message,
    Role, TokenUsage as CompletionTokenUsage,
};

// Feature-gated network providers
#[cfg(feature = "llm-network")]
pub use anthropic::AnthropicProvider;
#[cfg(feature = "llm-network")]
pub use openai::OpenAiProvider;
