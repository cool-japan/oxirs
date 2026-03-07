//! Additional LLM provider integrations
//!
//! This module contains mock implementations of multiple LLM providers
//! for testing and development purposes. Production deployments would
//! replace the mock HTTP layer with real API calls.

pub mod claude;
pub mod gemini;

pub use claude::{ClaudeClient, ClaudeConfig, ClaudeModel, ClaudeResponse};
pub use gemini::{
    GeminiClient, GeminiConfig, GeminiModel, GeminiResponse, HarmBlockThreshold, HarmCategory,
    SafetySetting,
};
