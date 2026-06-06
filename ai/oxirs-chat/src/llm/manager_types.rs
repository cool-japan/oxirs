//! LLM Manager Types
//!
//! Manager configuration, provider types, model types, request/response types,
//! routing types, session management, usage tracking, and rate limiting.

use anyhow::{anyhow, Result};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::Mutex as TokioMutex;

use super::types::LLMRequest;

// ─── Usage / Session Stats ────────────────────────────────────────────────────

/// Usage statistics for external queries
#[derive(Debug, Clone)]
pub struct UsageStats {
    pub total_requests: usize,
    pub total_tokens: usize,
    pub total_cost: f64,
}

/// Session statistics
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    pub active_sessions: usize,
    pub total_sessions: usize,
    pub average_session_duration: f64,
}

/// Detailed metrics
#[derive(Debug, Clone, Default)]
pub struct DetailedMetrics {
    pub performance_metrics: HashMap<String, f64>,
    pub error_rates: HashMap<String, f64>,
    pub response_times: Vec<f64>,
    pub total_sessions: usize,
    pub active_sessions: usize,
    pub total_messages: usize,
}

/// Backup report for session backup operations
#[derive(Debug, Clone, Default)]
pub struct BackupReport {
    pub sessions_backed_up: usize,
    pub backup_size: u64,
    pub backup_path: String,
    pub successful_backups: usize,
    pub failed_backups: usize,
}

/// Restore report for session restore operations
#[derive(Debug, Clone, Default)]
pub struct RestoreReport {
    pub sessions_restored: usize,
    pub restore_size: u64,
    pub restore_path: String,
    pub failed_restorations: usize,
}

// ─── Session ──────────────────────────────────────────────────────────────────

/// Serde helper for SystemTime serialization
pub mod systemtime_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).map_err(|e| {
            serde::ser::Error::custom(format!("SystemTime before UNIX_EPOCH: {}", e))
        })?;
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + std::time::Duration::from_secs(secs))
    }
}

/// A chat session with locking capabilities
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Session {
    pub id: String,
    pub messages: Vec<crate::messages::Message>,
    #[serde(with = "systemtime_serde")]
    pub created_at: std::time::SystemTime,
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

/// Thread-safe session wrapper
pub type LockedSession = Arc<TokioMutex<Session>>;

impl Session {
    pub fn new(id: String) -> Self {
        Self {
            id,
            messages: Vec::new(),
            created_at: std::time::SystemTime::now(),
            last_activity: chrono::Utc::now(),
        }
    }

    pub async fn process_message(
        &mut self,
        content: String,
        llm_manager: &mut super::manager::LLMManager,
    ) -> Result<crate::messages::Message> {
        use super::types::{ChatMessage, ChatRole, Priority, UseCase};

        // Update last activity
        self.last_activity = chrono::Utc::now();

        // Create user message
        let user_msg = crate::messages::Message {
            id: format!("msg_{}", uuid::Uuid::new_v4()),
            role: crate::messages::MessageRole::User,
            content: crate::messages::MessageContent::Text(content.clone()),
            timestamp: chrono::Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: Some(content.len() / 4),
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };

        // Add user message to session
        self.messages.push(user_msg.clone());

        // Prepare LLM request
        let llm_request = LLMRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: content.clone(),
                metadata: None,
            }],
            system_prompt: Some("You are a helpful AI assistant integrated with OxiRS Chat. Provide informative and helpful responses.".to_string()),
            max_tokens: Some(1000),
            temperature: 0.7,
            use_case: UseCase::Conversation,
            priority: Priority::Normal,
            timeout: Some(Duration::from_secs(30)),
        };

        // Generate response using LLM
        let llm_response = llm_manager.generate_response(llm_request).await?;

        // Create assistant message
        let response = crate::messages::Message {
            id: format!("msg_{}", uuid::Uuid::new_v4()),
            role: crate::messages::MessageRole::Assistant,
            content: crate::messages::MessageContent::Text(llm_response.content),
            timestamp: chrono::Utc::now(),
            metadata: Some(crate::messages::MessageMetadata {
                source: Some("llm-manager".to_string()),
                confidence: Some(0.85),
                processing_time_ms: Some(llm_response.latency.as_millis() as u64),
                model_used: Some(llm_response.model_used.clone()),
                temperature: Some(0.7),
                max_tokens: Some(1000),
                custom_fields: std::collections::HashMap::new(),
            }),
            thread_id: None,
            parent_message_id: Some(user_msg.id),
            token_count: Some(llm_response.usage.completion_tokens),
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };

        // Add response to session
        self.messages.push(response.clone());

        Ok(response)
    }
}

// ─── Usage Tracker ────────────────────────────────────────────────────────────

/// Usage tracking for monitoring and billing
pub struct UsageTracker {
    pub total_requests: usize,
    pub total_tokens: usize,
    pub total_cost: f64,
    pub provider_usage: HashMap<String, ProviderUsage>,
}

impl Default for UsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl UsageTracker {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            total_tokens: 0,
            total_cost: 0.0,
            provider_usage: HashMap::new(),
        }
    }

    pub fn track_usage(&mut self, provider: &str, tokens: usize, cost: f64) {
        self.total_requests += 1;
        self.total_tokens += tokens;
        self.total_cost += cost;

        let provider_stats = self.provider_usage.entry(provider.to_string()).or_default();
        provider_stats.requests += 1;
        provider_stats.tokens += tokens;
        provider_stats.cost += cost;
    }
}

#[derive(Debug, Clone)]
pub struct ProviderUsage {
    pub requests: usize,
    pub tokens: usize,
    pub cost: f64,
}

impl Default for ProviderUsage {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderUsage {
    pub fn new() -> Self {
        Self {
            requests: 0,
            tokens: 0,
            cost: 0.0,
        }
    }
}

// ─── Rate Limiter ─────────────────────────────────────────────────────────────

/// Token bucket for rate limiting
#[derive(Debug, Clone)]
pub(crate) struct TokenBucket {
    pub tokens: f64,
    pub last_refill: std::time::Instant,
    pub capacity: f64,
    pub refill_rate: f64,
    pub requests_per_minute: usize,
    pub window_start: std::time::Instant,
    pub request_count: usize,
}

impl TokenBucket {
    pub fn new(capacity: f64, refill_rate: f64, requests_per_minute: usize) -> Self {
        let now = std::time::Instant::now();
        Self {
            tokens: capacity,
            last_refill: now,
            capacity,
            refill_rate,
            requests_per_minute,
            window_start: now,
            request_count: 0,
        }
    }

    pub fn refill(&mut self) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let tokens_to_add = elapsed * self.refill_rate;
        self.tokens = (self.tokens + tokens_to_add).min(self.capacity);
        self.last_refill = now;

        if now.duration_since(self.window_start) >= Duration::from_secs(60) {
            self.request_count = 0;
            self.window_start = now;
        }
    }

    pub fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();

        if self.request_count >= self.requests_per_minute {
            return false;
        }

        if self.tokens >= tokens {
            self.tokens -= tokens;
            self.request_count += 1;
            true
        } else {
            false
        }
    }

    pub fn get_wait_time(&self) -> Duration {
        let tokens_needed = 1.0 - self.tokens;
        if tokens_needed <= 0.0 {
            Duration::from_secs(0)
        } else {
            Duration::from_secs_f64(tokens_needed / self.refill_rate)
        }
    }
}

/// Rate limiter implementation using token bucket algorithm
pub struct RateLimiter {
    pub(crate) enabled: bool,
    pub(crate) buckets: Arc<TokioMutex<HashMap<String, TokenBucket>>>,
}

impl RateLimiter {
    pub fn new(_config: &super::config::RateLimitConfig) -> Self {
        Self {
            enabled: true,
            buckets: Arc::new(TokioMutex::new(HashMap::new())),
        }
    }

    pub async fn check_rate_limit(&self, provider: &str) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let mut buckets = self.buckets.lock().await;
        let bucket = buckets
            .entry(provider.to_string())
            .or_insert_with(|| TokenBucket::new(10.0, 1.0, 60));

        if bucket.try_consume(1.0) {
            Ok(())
        } else {
            let wait_time = bucket.get_wait_time();
            Err(anyhow!(
                "Rate limit exceeded for provider: {}. Please wait {:?}",
                provider,
                wait_time
            ))
        }
    }

    pub async fn get_rate_limit_status(&self, provider: &str) -> Result<RateLimitStatus> {
        if !self.enabled {
            return Ok(RateLimitStatus {
                provider: provider.to_string(),
                tokens_available: f64::INFINITY,
                requests_remaining: usize::MAX,
                reset_time: None,
            });
        }

        let mut buckets = self.buckets.lock().await;
        let bucket = buckets
            .entry(provider.to_string())
            .or_insert_with(|| TokenBucket::new(10.0, 1.0, 60));

        bucket.refill();
        let next_reset = bucket.window_start + Duration::from_secs(60);

        Ok(RateLimitStatus {
            provider: provider.to_string(),
            tokens_available: bucket.tokens,
            requests_remaining: bucket
                .requests_per_minute
                .saturating_sub(bucket.request_count),
            reset_time: Some(next_reset),
        })
    }
}

/// Rate limit status information
#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    pub provider: String,
    pub tokens_available: f64,
    pub requests_remaining: usize,
    pub reset_time: Option<std::time::Instant>,
}

// ─── Comprehensive Stats ──────────────────────────────────────────────────────

/// Comprehensive statistics including Version 1.3 capabilities
#[derive(Debug, Clone)]
pub struct ComprehensiveStats {
    pub session_stats: SessionStats,
    pub fine_tuning_stats: Option<super::fine_tuning::FineTuningStatistics>,
    pub federation_stats: Option<super::federated_learning::FederationStatistics>,
    pub adaptation_metrics: Option<super::real_time_adaptation::AdaptationMetrics>,
    pub cross_modal_stats: Option<super::cross_modal_reasoning::CrossModalStats>,
    pub performance_report: Option<super::performance_optimization::PerformanceReport>,
    pub version: String,
    pub capabilities_enabled: CapabilityStatus,
}

/// Status of Version 1.3 capabilities
#[derive(Debug, Clone)]
pub struct CapabilityStatus {
    pub fine_tuning: bool,
    pub architecture_search: bool,
    pub federated_learning: bool,
    pub real_time_adaptation: bool,
    pub cross_modal_reasoning: bool,
    pub performance_optimization: bool,
}

// Re-export LLMRequest and LLMResponse for convenience within the module
pub use super::types::{LLMRequest as ManagerLLMRequest, LLMResponse as ManagerLLMResponse};
