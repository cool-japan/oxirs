//! LLM Integration Module for OxiRS Chat
//!
//! Provides unified interface for multiple LLM providers including OpenAI, Anthropic Claude,
//! and local models with intelligent routing and fallback strategies.

use anyhow::{anyhow, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
        CreateChatCompletionRequestArgs, Role,
    },
    Client as OpenAIClient,
};
use futures_util::{Stream, StreamExt};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{Mutex as TokioMutex, RwLock};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

// Module declarations
pub mod anthropic_provider;
pub mod circuit_breaker;
pub mod config;
pub mod local_provider;
pub mod manager;
pub mod openai_provider;
pub mod providers;
pub mod reasoning;
pub mod types;

// Re-export commonly used types
pub use anthropic_provider::AnthropicProvider;
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerStats};
pub use config::{
    BackoffStrategy, CircuitBreakerConfig, CircuitBreakerState, FallbackConfig, LLMConfig,
    ModelConfig, ProviderConfig, RateLimitConfig, RoutingConfig, RoutingStrategy,
};
pub use local_provider::LocalModelProvider;
pub use manager::{EnhancedLLMManager, LLMManager};
pub use openai_provider::OpenAIProvider;
pub use providers::LLMProvider;
pub use types::{
    ChatMessage, ChatRole, LLMRequest, LLMResponse, LLMResponseChunk, LLMResponseStream, Priority,
    RoutingCandidate, Usage, UseCase,
};
