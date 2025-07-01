//! Type definitions for LLM requests and responses
//!
//! Contains all data structures used for communicating with LLM providers
//! and managing request/response lifecycle.

use anyhow::Result;
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, pin::Pin, time::{Duration, Instant}};

/// LLM request context
#[derive(Debug, Clone)]
pub struct LLMRequest {
    pub messages: Vec<ChatMessage>,
    pub system_prompt: Option<String>,
    pub temperature: f32,
    pub max_tokens: Option<usize>,
    pub use_case: UseCase,
    pub priority: Priority,
    pub timeout: Option<Duration>,
}

/// Chat message for LLM interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// Use case classification for intelligent routing
#[derive(Debug, Clone, PartialEq)]
pub enum UseCase {
    SimpleQuery,
    ComplexReasoning,
    SparqlGeneration,
    KnowledgeExtraction,
    Conversation,
    Analysis,
    CodeGeneration,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

/// LLM response
#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub content: String,
    pub model_used: String,
    pub provider_used: String,
    pub usage: Usage,
    pub latency: Duration,
    pub quality_score: Option<f32>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Streaming response chunk
#[derive(Debug, Clone)]
pub struct LLMResponseChunk {
    pub content: String,
    pub is_final: bool,
    pub chunk_index: usize,
    pub model_used: String,
    pub provider_used: String,
    pub latency: Duration,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Streaming response wrapper
pub struct LLMResponseStream {
    pub stream: Pin<Box<dyn Stream<Item = Result<LLMResponseChunk>> + Send>>,
    pub model_used: String,
    pub provider_used: String,
    pub started_at: Instant,
}

/// Token usage information
#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub cost: f64,
}

/// Routing candidate for model selection
#[derive(Debug, Clone)]
pub(crate) struct RoutingCandidate {
    pub provider: String,
    pub model: String,
    pub score: f32,
}