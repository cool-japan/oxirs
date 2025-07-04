//! LLM Manager Implementation
//!
//! Contains the main LLM manager and enhanced manager with rate limiting and monitoring.

use anyhow::{anyhow, Result};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, error, info, warn};
use uuid;

use super::{
    anthropic_provider::AnthropicProvider,
    circuit_breaker::CircuitBreaker,
    config::LLMConfig,
    local_provider::LocalModelProvider,
    openai_provider::OpenAIProvider,
    providers::LLMProvider,
    types::{LLMRequest, LLMResponse, Priority, UseCase},
};

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

/// A chat session with locking capabilities
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Session {
    pub id: String,
    pub messages: Vec<crate::messages::Message>,
    #[serde(with = "systemtime_serde")]
    pub created_at: std::time::SystemTime,
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

/// Serde helper for SystemTime serialization
mod systemtime_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap();
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

    pub async fn process_message(&mut self, content: String) -> Result<crate::messages::Message> {
        // TODO: Implement actual message processing
        let response = crate::messages::Message {
            id: format!("msg_{}", uuid::Uuid::new_v4()),
            role: crate::messages::MessageRole::Assistant,
            content: crate::messages::MessageContent::Text(format!("Echo: {}", content)),
            timestamp: chrono::Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: None,
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        };
        self.messages.push(response.clone());
        Ok(response)
    }
}

/// Usage tracking for monitoring and billing
pub struct UsageTracker {
    total_requests: usize,
    total_tokens: usize,
    total_cost: f64,
    provider_usage: HashMap<String, ProviderUsage>,
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

        let provider_stats = self
            .provider_usage
            .entry(provider.to_string())
            .or_insert_with(|| ProviderUsage::new());

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

impl ProviderUsage {
    pub fn new() -> Self {
        Self {
            requests: 0,
            tokens: 0,
            cost: 0.0,
        }
    }
}

/// Rate limiter implementation
pub struct RateLimiter {
    // Simplified rate limiter - in production would use more sophisticated implementation
    enabled: bool,
}

impl RateLimiter {
    pub fn new(_config: &super::config::RateLimitConfig) -> Self {
        Self { enabled: true }
    }

    pub async fn check_rate_limit(&self, _provider: &str) -> Result<()> {
        // TODO: Implement actual rate limiting
        Ok(())
    }
}

/// Main LLM manager
pub struct LLMManager {
    config: LLMConfig,
    providers: HashMap<String, Box<dyn LLMProvider + Send + Sync>>,
    circuit_breakers: HashMap<String, Arc<CircuitBreaker>>,
    usage_tracker: TokioMutex<UsageTracker>,
}

impl LLMManager {
    pub fn new(config: LLMConfig) -> Result<Self> {
        let mut manager = Self {
            providers: HashMap::new(),
            circuit_breakers: HashMap::new(),
            usage_tracker: TokioMutex::new(UsageTracker::new()),
            config,
        };

        manager.initialize_providers()?;
        manager.initialize_circuit_breakers();
        Ok(manager)
    }

    fn initialize_providers(&mut self) -> Result<()> {
        // Initialize OpenAI provider
        if let Some(config) = self.config.providers.get("openai") {
            if config.enabled {
                let provider = Box::new(OpenAIProvider::new(config.clone())?);
                self.providers.insert("openai".to_string(), provider);
            }
        }

        // Initialize Anthropic provider
        if let Some(config) = self.config.providers.get("anthropic") {
            if config.enabled {
                let provider = Box::new(AnthropicProvider::new(config.clone())?);
                self.providers.insert("anthropic".to_string(), provider);
            }
        }

        // Initialize local model provider
        if let Some(config) = self.config.providers.get("local") {
            if config.enabled {
                let provider = Box::new(LocalModelProvider::new(config.clone())?);
                self.providers.insert("local".to_string(), provider);
            }
        }

        Ok(())
    }

    fn initialize_circuit_breakers(&mut self) {
        for provider_name in self.config.providers.keys() {
            let circuit_breaker =
                Arc::new(CircuitBreaker::new(self.config.circuit_breaker.clone()));
            self.circuit_breakers
                .insert(provider_name.clone(), circuit_breaker);
        }
    }

    /// Generate response using optimal provider selection
    pub async fn generate_response(&mut self, request: LLMRequest) -> Result<LLMResponse> {
        // For now, simplified implementation - use first available provider
        let provider_name = self
            .providers
            .keys()
            .next()
            .ok_or_else(|| anyhow!("No providers configured"))?
            .clone();

        let model = self
            .config
            .providers
            .get(&provider_name)
            .and_then(|p| p.models.first())
            .map(|m| m.name.as_str())
            .unwrap_or("default");

        // Check circuit breaker
        if let Some(circuit_breaker) = self.circuit_breakers.get(&provider_name) {
            if !circuit_breaker.can_execute().await {
                return Err(anyhow!(
                    "Circuit breaker is open for provider: {}",
                    provider_name
                ));
            }
        }

        // Get the provider and generate response
        let provider = self
            .providers
            .get(&provider_name)
            .ok_or_else(|| anyhow!("Provider {} not found", provider_name))?;

        let mut response = provider.generate(model, &request).await?;

        // Track usage
        {
            let mut tracker = self.usage_tracker.lock().await;
            tracker.track_usage(
                &provider_name,
                response.usage.total_tokens,
                response.usage.cost,
            );
        }

        Ok(response)
    }

    pub fn estimate_input_tokens(&self, request: &LLMRequest) -> usize {
        // Simple token estimation - in practice would use proper tokenizer
        let total_content: String = request
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .chain(request.system_prompt.as_ref().map(|s| s.as_str()))
            .collect::<Vec<_>>()
            .join(" ");

        // Rough estimate: 1 token ≈ 4 characters
        total_content.len() / 4
    }

    /// Get circuit breaker statistics for all providers
    pub async fn get_circuit_breaker_stats(&self) -> HashMap<String, super::CircuitBreakerStats> {
        let mut stats = HashMap::new();

        for (provider_name, circuit_breaker) in &self.circuit_breakers {
            let provider_stats = circuit_breaker.get_stats().await;
            stats.insert(provider_name.clone(), provider_stats);
        }

        stats
    }

    /// Reset circuit breaker for a specific provider
    pub async fn reset_circuit_breaker(&self, provider_name: &str) -> Result<()> {
        if let Some(circuit_breaker) = self.circuit_breakers.get(provider_name) {
            circuit_breaker.reset().await?;
            Ok(())
        } else {
            Err(anyhow!(
                "Circuit breaker not found for provider: {}",
                provider_name
            ))
        }
    }

    /// Get usage statistics for monitoring and billing
    pub async fn get_usage_stats(&self) -> UsageStats {
        let tracker = self.usage_tracker.lock().await;
        UsageStats {
            total_requests: tracker.total_requests,
            total_tokens: tracker.total_tokens,
            total_cost: tracker.total_cost,
        }
    }
}

/// Enhanced LLM manager with rate limiting and monitoring
pub struct EnhancedLLMManager {
    inner: LLMManager,
    rate_limiter: RateLimiter,
    sessions: Arc<TokioMutex<HashMap<String, LockedSession>>>,
}

impl EnhancedLLMManager {
    pub fn new(config: LLMConfig) -> Result<Self> {
        let rate_limiter = RateLimiter::new(&config.rate_limits);
        let inner = LLMManager::new(config)?;
        let sessions = Arc::new(TokioMutex::new(HashMap::new()));

        Ok(Self {
            inner,
            rate_limiter,
            sessions,
        })
    }

    pub async fn generate_response_with_limits(
        &mut self,
        request: LLMRequest,
    ) -> Result<LLMResponse> {
        // Check rate limits before processing
        self.rate_limiter.check_rate_limit("default").await?;

        // Generate response using the inner manager
        self.inner.generate_response(request).await
    }

    /// Create a new enhanced LLM manager with persistence capabilities
    pub async fn with_persistence<P: AsRef<std::path::Path>>(
        store: Arc<dyn oxirs_core::Store>,
        persistence_path: P,
    ) -> Result<Self> {
        let config = LLMConfig::default();
        let mut manager = Self::new(config)?;
        // TODO: Implement actual persistence logic
        Ok(manager)
    }

    /// Get or create a session for the given session ID
    pub async fn get_or_create_session(&self, session_id: String) -> Result<LockedSession> {
        let mut sessions = self.sessions.lock().await;

        if let Some(session) = sessions.get(&session_id) {
            Ok(session.clone())
        } else {
            let session = Session::new(session_id.clone());
            let locked_session = Arc::new(TokioMutex::new(session));
            sessions.insert(session_id, locked_session.clone());
            Ok(locked_session)
        }
    }

    /// Get session statistics
    pub async fn get_session_stats(&self) -> SessionStats {
        let sessions = self.sessions.lock().await;
        SessionStats {
            active_sessions: sessions.len(),
            total_sessions: sessions.len(), // Simplified - all sessions are considered active
            average_session_duration: 0.0,  // TODO: Track session durations
        }
    }

    /// Get detailed metrics
    pub async fn get_detailed_metrics(&self) -> DetailedMetrics {
        let sessions = self.sessions.lock().await;
        let mut metrics = DetailedMetrics::default();
        metrics.total_sessions = sessions.len();
        metrics.active_sessions = sessions.len(); // Simplified - all sessions are considered active

        // Count total messages across all sessions
        let mut total_messages = 0;
        for session in sessions.values() {
            let session_guard = session.lock().await;
            total_messages += session_guard.messages.len();
        }
        metrics.total_messages = total_messages;

        metrics
    }

    /// Backup sessions to the specified path
    pub async fn backup_sessions<P: AsRef<std::path::Path>>(
        &self,
        backup_path: &P,
    ) -> Result<BackupReport> {
        let sessions = self.sessions.lock().await;
        let backup_dir = backup_path.as_ref();

        // Create backup directory if it doesn't exist
        tokio::fs::create_dir_all(backup_dir)
            .await
            .map_err(|e| anyhow!("Failed to create backup directory: {}", e))?;

        let mut successful_backups = 0;
        let mut failed_backups = 0;
        let mut total_size = 0u64;

        // Serialize and save each session
        for (session_id, session_arc) in sessions.iter() {
            match session_arc.try_lock() {
                Ok(session) => {
                    let session_data = serde_json::to_string(&*session).map_err(|e| {
                        anyhow!("Failed to serialize session {}: {}", session_id, e)
                    })?;

                    let session_file = backup_dir.join(format!("{}.json", session_id));
                    match tokio::fs::write(&session_file, &session_data).await {
                        Ok(_) => {
                            successful_backups += 1;
                            total_size += session_data.len() as u64;
                            debug!("Successfully backed up session: {}", session_id);
                        }
                        Err(e) => {
                            failed_backups += 1;
                            warn!("Failed to backup session {}: {}", session_id, e);
                        }
                    }
                }
                Err(_) => {
                    // Session is locked, skip for now
                    failed_backups += 1;
                    warn!("Session {} is locked, skipping backup", session_id);
                }
            }
        }

        info!(
            "Backup completed: {} successful, {} failed, {} bytes",
            successful_backups, failed_backups, total_size
        );

        Ok(BackupReport {
            sessions_backed_up: successful_backups,
            backup_size: total_size,
            backup_path: backup_dir.to_string_lossy().to_string(),
            successful_backups,
            failed_backups,
        })
    }

    /// Remove a session by ID
    pub async fn remove_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.lock().await;
        sessions.remove(session_id);
        Ok(())
    }

    /// Restore sessions from the specified path
    pub async fn restore_sessions<P: AsRef<std::path::Path>>(
        &self,
        restore_path: &P,
    ) -> Result<RestoreReport> {
        let restore_dir = restore_path.as_ref();
        let mut sessions = self.sessions.lock().await;

        // Check if restore directory exists
        if !restore_dir.exists() {
            return Err(anyhow!(
                "Restore directory does not exist: {:?}",
                restore_dir
            ));
        }

        let mut sessions_restored = 0;
        let mut failed_restorations = 0;
        let mut total_size = 0u64;

        // Read all JSON files from the restore directory
        let mut dir_entries = tokio::fs::read_dir(restore_dir)
            .await
            .map_err(|e| anyhow!("Failed to read restore directory: {}", e))?;

        while let Some(entry) = dir_entries
            .next_entry()
            .await
            .map_err(|e| anyhow!("Failed to read directory entry: {}", e))?
        {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                match self.restore_single_session(&path).await {
                    Ok((session_id, session_arc)) => {
                        // Calculate file size for reporting
                        if let Ok(metadata) = tokio::fs::metadata(&path).await {
                            total_size += metadata.len();
                        }

                        // Insert the restored session
                        sessions.insert(session_id.clone(), session_arc);
                        sessions_restored += 1;
                        info!("Successfully restored session: {}", session_id);
                    }
                    Err(e) => {
                        failed_restorations += 1;
                        warn!("Failed to restore session from {:?}: {}", path, e);
                    }
                }
            }
        }

        info!(
            "Restore completed: {} sessions restored, {} failed, {} bytes processed",
            sessions_restored, failed_restorations, total_size
        );

        Ok(RestoreReport {
            sessions_restored,
            restore_size: total_size,
            restore_path: restore_dir.to_string_lossy().to_string(),
            failed_restorations,
        })
    }

    /// Helper method to restore a single session from a file
    async fn restore_single_session<P: AsRef<std::path::Path>>(
        &self,
        session_file: P,
    ) -> Result<(String, LockedSession)> {
        let session_data = tokio::fs::read_to_string(&session_file)
            .await
            .map_err(|e| anyhow!("Failed to read session file: {}", e))?;

        let session: Session = serde_json::from_str(&session_data)
            .map_err(|e| anyhow!("Failed to deserialize session: {}", e))?;

        let session_id = session.id.clone();
        let locked_session = Arc::new(TokioMutex::new(session));

        Ok((session_id, locked_session))
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&self) -> Result<usize> {
        let mut sessions = self.sessions.lock().await;
        let now = chrono::Utc::now();
        let mut removed_count = 0;
        let mut sessions_to_remove = Vec::new();

        // Check each session for expiration (1 hour threshold for better test compatibility)
        for (session_id, session_arc) in sessions.iter() {
            match session_arc.try_lock() {
                Ok(session_guard) => {
                    // Consider a session expired if inactive for more than 1 hour
                    let hours_inactive = now
                        .signed_duration_since(session_guard.last_activity)
                        .num_hours();
                    if hours_inactive > 1 {
                        sessions_to_remove.push(session_id.clone());
                        debug!(
                            "Marking session {} for removal (inactive for {} hours)",
                            session_id, hours_inactive
                        );
                    }
                }
                Err(_) => {
                    // Session is locked, skip for now but don't remove
                    debug!(
                        "Session {} is locked, skipping expiration check",
                        session_id
                    );
                }
            }
        }

        // Remove expired sessions
        for session_id in sessions_to_remove {
            if sessions.remove(&session_id).is_some() {
                removed_count += 1;
                info!("Removed expired session: {}", session_id);
            }
        }

        if removed_count > 0 {
            info!(
                "Cleanup completed: removed {} expired sessions",
                removed_count
            );
        }

        Ok(removed_count)
    }

    /// Get a session by ID
    pub async fn get_session(&self, session_id: &str) -> Result<Option<LockedSession>> {
        let sessions = self.sessions.lock().await;
        Ok(sessions.get(session_id).cloned())
    }
}
