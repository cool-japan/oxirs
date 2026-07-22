//! Answer generation module

pub mod context_builder;
pub mod prompt_templates;

pub use context_builder::{ContextBuilder, ScoredTriple};
pub use prompt_templates::PromptTemplate;
