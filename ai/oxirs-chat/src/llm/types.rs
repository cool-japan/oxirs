//! Type definitions for LLM requests and responses
//!
//! Contains all data structures used for communicating with LLM providers
//! and managing request/response lifecycle.

use anyhow::Result;
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    pin::Pin,
    time::{Duration, Instant},
};

/// Serde helper for Duration serialization
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_millis().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

/// Serde helper for Option<Duration> serialization
mod option_duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match duration {
            Some(d) => Some(d.as_millis()).serialize(serializer),
            None => None::<u128>.serialize(serializer),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis_opt = Option::<u64>::deserialize(deserializer)?;
        Ok(millis_opt.map(Duration::from_millis))
    }
}

/// LLM request context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRequest {
    pub messages: Vec<ChatMessage>,
    pub system_prompt: Option<String>,
    pub temperature: f32,
    pub max_tokens: Option<usize>,
    pub use_case: UseCase,
    pub priority: Priority,
    #[serde(with = "option_duration_serde")]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UseCase {
    SimpleQuery,
    ComplexReasoning,
    SparqlGeneration,
    KnowledgeExtraction,
    Conversation,
    Analysis,
    CodeGeneration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

/// LLM response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: String,
    pub model_used: String,
    pub provider_used: String,
    pub usage: Usage,
    #[serde(with = "duration_serde")]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub cost: f64,
}

/// Routing candidate for model selection
#[derive(Debug, Clone)]
pub struct RoutingCandidate {
    pub provider: String,
    pub model: String,
    pub score: f32,
}
