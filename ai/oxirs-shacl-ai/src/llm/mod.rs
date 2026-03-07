//! LLM integration for SHACL constraint generation
//!
//! Provides provider-agnostic abstractions for invoking large language models
//! to produce SHACL shapes from natural-language descriptions, ontology
//! fragments, or example RDF triples.

pub mod constraint_generator;

pub use constraint_generator::{
    BatchGenerationRequest, BatchItemResult, GeneratedShaclShape, GeneratorStats,
    LlmConstraintGenerator, LlmConstraintGeneratorConfig, LlmProvider, LlmRequest, LlmResponse,
    PromptTemplate, StubLlmProvider, TokenUsage,
};
