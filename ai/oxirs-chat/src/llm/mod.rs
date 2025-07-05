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
pub mod cross_modal_reasoning;
pub mod federated_learning;
pub mod fine_tuning;
pub mod local_provider;
pub mod manager;
pub mod neural_architecture_search;
pub mod openai_provider;
pub mod performance_optimization;
pub mod providers;
pub mod real_time_adaptation;
pub mod reasoning;
pub mod types;

// Re-export commonly used types
pub use anthropic_provider::AnthropicProvider;
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerStats};
pub use config::{
    BackoffStrategy, CircuitBreakerConfig, CircuitBreakerState, FallbackConfig, LLMConfig,
    ModelConfig, ProviderConfig, RateLimitConfig, RoutingConfig, RoutingStrategy,
};
pub use cross_modal_reasoning::{
    CrossModalConfig, CrossModalInput, CrossModalReasoning, CrossModalResponse, CrossModalStats,
    DataFormat, FusionStrategy, ImageFormat, ImageInput, ReasoningModality, StructuredData,
};
pub use federated_learning::{
    AggregationStrategy, FederatedCoordinator, FederatedLearningConfig, FederatedNode,
    FederationStatistics, PrivacyConfig,
};
pub use fine_tuning::{
    FineTuningConfig, FineTuningEngine, FineTuningJob, FineTuningStatistics, JobStatus,
    TrainingExample, TrainingParameters,
};
pub use local_provider::LocalModelProvider;
pub use manager::{EnhancedLLMManager, LLMManager};
pub use neural_architecture_search::{
    ArchitectureOptimizer, ArchitectureSearch, ArchitectureSearchConfig, ModelArchitecture,
    SearchResult,
};
pub use openai_provider::OpenAIProvider;
pub use performance_optimization::{
    BenchmarkConfig, BenchmarkResult, LoadBalanceStrategy, OptimizationRecommendation,
    PerformanceConfig, PerformanceMetrics, PerformanceOptimizer, PerformanceReport,
};
pub use providers::LLMProvider;
pub use real_time_adaptation::{
    AdaptationConfig, AdaptationMetrics, AdaptationStrategy, RealTimeAdaptation,
};
pub use types::{
    ChatMessage, ChatRole, LLMRequest, LLMResponse, LLMResponseChunk, LLMResponseStream, Priority,
    RoutingCandidate, Usage, UseCase,
};
