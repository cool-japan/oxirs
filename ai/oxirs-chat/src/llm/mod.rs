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
pub mod config;
pub mod circuit_breaker;
pub mod types;
pub mod providers;
pub mod manager;
pub mod reasoning;
pub mod openai_provider;
pub mod anthropic_provider;
pub mod local_provider;

// Re-export commonly used types
pub use config::{
    LLMConfig, ProviderConfig, ModelConfig, RoutingConfig, RoutingStrategy,
    FallbackConfig, BackoffStrategy, RateLimitConfig, CircuitBreakerConfig, CircuitBreakerState
};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerStats};
pub use types::{
    LLMRequest, LLMResponse, LLMResponseChunk, LLMResponseStream, ChatMessage, ChatRole,
    UseCase, Priority, Usage, RoutingCandidate
};
pub use providers::LLMProvider;
pub use manager::{LLMManager, EnhancedLLMManager};
pub use openai_provider::OpenAIProvider;
pub use anthropic_provider::AnthropicProvider;
pub use local_provider::LocalModelProvider;