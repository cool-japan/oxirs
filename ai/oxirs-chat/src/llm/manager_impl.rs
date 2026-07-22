//! LLM Manager — Core implementation
//!
//! `LLMManager` (fallback chain execution) and `EnhancedLLMManager`
//! (rate limiting, session management, Version 1.3 capabilities).

use anyhow::{anyhow, Result};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, error, info, warn};

use super::{
    anthropic_provider::AnthropicProvider,
    cache::{CacheConfig, ResponseCache},
    circuit_breaker::CircuitBreaker,
    config::LLMConfig,
    cross_modal_reasoning::{
        CrossModalConfig, CrossModalInput, CrossModalReasoning, CrossModalResponse,
    },
    federated_learning::{FederatedCoordinator, FederatedLearningConfig},
    fine_tuning::{FineTuningConfig, FineTuningEngine},
    health_checker::{HealthCheckConfig, HealthChecker},
    local_provider::LocalModelProvider,
    neural_architecture_search::{ArchitectureSearch, ArchitectureSearchConfig},
    performance_optimization::{
        BenchmarkConfig, BenchmarkResult, OptimizationRecommendation, PerformanceConfig,
        PerformanceOptimizer, PerformanceReport,
    },
    providers::LLMProvider,
    real_time_adaptation::{AdaptationConfig, InteractionData, RealTimeAdaptation},
    token_budget::{BudgetConfig, TokenBudget},
    types::{LLMRequest, LLMResponse, LLMResponseStream},
};

use super::manager_types::{
    BackupReport, CapabilityStatus, ComprehensiveStats, DetailedMetrics, LockedSession,
    RateLimiter, RestoreReport, Session, SessionStats, UsageStats, UsageTracker,
};

// The OpenAI provider is only available when the `openai` feature is enabled
// (async-openai is opt-in to keep aws-lc-rs out of the default build).
#[cfg(feature = "openai")]
use super::openai_provider::OpenAIProvider;

/// Main LLM manager with fallback chain support
pub struct LLMManager {
    pub(crate) config: LLMConfig,
    pub(crate) providers: HashMap<String, Box<dyn LLMProvider + Send + Sync>>,
    pub(crate) circuit_breakers: HashMap<String, Arc<CircuitBreaker>>,
    pub(crate) usage_tracker: TokioMutex<UsageTracker>,
    pub(crate) response_cache: Arc<ResponseCache>,
    pub(crate) health_checker: Arc<HealthChecker>,
    pub(crate) token_budget: Arc<TokenBudget>,
}

impl LLMManager {
    pub fn new(config: LLMConfig) -> Result<Self> {
        let response_cache = Arc::new(ResponseCache::new(CacheConfig::default()));
        let health_checker = Arc::new(HealthChecker::new(HealthCheckConfig::default()));
        let token_budget = Arc::new(TokenBudget::new(BudgetConfig::default()));

        let mut manager = Self {
            providers: HashMap::new(),
            circuit_breakers: HashMap::new(),
            usage_tracker: TokioMutex::new(UsageTracker::new()),
            response_cache,
            health_checker,
            token_budget,
            config,
        };

        manager.initialize_providers()?;
        manager.initialize_circuit_breakers();
        manager.initialize_health_monitoring();
        Ok(manager)
    }

    fn initialize_health_monitoring(&self) {
        for provider_name in self.providers.keys() {
            let health_checker = Arc::clone(&self.health_checker);
            let provider_id = provider_name.clone();
            tokio::spawn(async move {
                health_checker.register_provider(provider_id).await;
            });
        }
    }

    fn initialize_providers(&mut self) -> Result<()> {
        if let Some(config) = self.config.providers.get("openai") {
            if config.enabled {
                #[cfg(feature = "openai")]
                {
                    let provider = Box::new(OpenAIProvider::new(config.clone())?);
                    self.providers.insert("openai".to_string(), provider);
                }
                #[cfg(not(feature = "openai"))]
                {
                    warn!(
                        "OpenAI provider is configured and enabled but the `openai` \
                         feature is not compiled in; skipping registration. Rebuild \
                         oxirs-chat with `--features openai` to enable it."
                    );
                }
            }
        }

        if let Some(config) = self.config.providers.get("anthropic") {
            if config.enabled {
                let provider = Box::new(AnthropicProvider::new(config.clone())?);
                self.providers.insert("anthropic".to_string(), provider);
            }
        }

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

    /// Generate response with fallback chain: OpenAI → Anthropic → Ollama (local)
    pub async fn generate_response(&mut self, request: LLMRequest) -> Result<LLMResponse> {
        let start_time = std::time::Instant::now();

        // Step 1: Check cache
        if let Some(cached_response) = self.response_cache.get(&request).await {
            debug!("Cache hit for request");
            return Ok(cached_response);
        }

        // Step 2: Check token budget
        let user_id = "default_user".to_string();
        let estimated_tokens =
            self.estimate_input_tokens(&request) + request.max_tokens.unwrap_or(1000);

        if let Err(e) = self
            .token_budget
            .check_budget(&user_id, estimated_tokens as u64)
            .await
        {
            warn!("Token budget exceeded: {}", e);
            return Err(e);
        }

        // Step 3: Get provider fallback chain
        let provider_chain = self.get_provider_fallback_chain().await;

        if provider_chain.is_empty() {
            return Err(anyhow!("No providers available"));
        }

        // Step 4: Try each provider in the chain
        let mut last_error: Option<anyhow::Error> = None;

        for (provider_name, model_name) in provider_chain {
            info!(
                "Attempting provider: {} with model: {}",
                provider_name, model_name
            );

            if let Some(circuit_breaker) = self.circuit_breakers.get(&provider_name) {
                if !circuit_breaker.can_execute().await {
                    warn!("Circuit breaker is open for provider: {}", provider_name);
                    continue;
                }
            }

            if !self
                .health_checker
                .is_provider_healthy(&provider_name)
                .await
            {
                warn!("Provider {} is unhealthy, skipping", provider_name);
                continue;
            }

            match self
                .try_provider(&provider_name, &model_name, &request)
                .await
            {
                Ok(response) => {
                    let elapsed = start_time.elapsed();

                    self.record_success(&provider_name, elapsed).await;

                    self.response_cache
                        .put(&request, response.clone(), provider_name.clone())
                        .await;

                    self.token_budget
                        .record_usage(&user_id, response.usage.total_tokens as u64)
                        .await?;

                    {
                        let mut tracker = self.usage_tracker.lock().await;
                        tracker.track_usage(
                            &provider_name,
                            response.usage.total_tokens,
                            response.usage.cost,
                        );
                    }

                    info!(
                        "Successfully generated response using provider {} in {:?}",
                        provider_name, elapsed
                    );

                    return Ok(response);
                }
                Err(e) => {
                    let elapsed = start_time.elapsed();

                    self.record_failure(&provider_name, elapsed).await;

                    warn!("Provider {} failed: {}", provider_name, e);
                    last_error = Some(e);

                    if elapsed.as_millis() < 500 {
                        debug!("Fast failover to next provider ({}ms)", elapsed.as_millis());
                    }

                    continue;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("All providers failed")))
    }

    /// Generate a response as a real, incremental stream.
    ///
    /// Walks the same provider fallback chain as [`Self::generate_response`] but
    /// invokes each provider's `generate_stream` implementation, returning the
    /// first provider's live token stream. For providers with genuine
    /// server-sent-event streaming (OpenAI, Anthropic) tokens are delivered as
    /// the model produces them; providers without an SSE endpoint frame their
    /// completed response. This is the single real streaming entry point used by
    /// the chat streaming API (replacing the previous artificial re-chunking).
    pub async fn generate_response_stream(&self, request: LLMRequest) -> Result<LLMResponseStream> {
        let start_time = std::time::Instant::now();
        let provider_chain = self.get_provider_fallback_chain().await;
        if provider_chain.is_empty() {
            return Err(anyhow!("No providers available"));
        }

        let mut last_error: Option<anyhow::Error> = None;
        for (provider_name, model_name) in provider_chain {
            if let Some(circuit_breaker) = self.circuit_breakers.get(&provider_name) {
                if !circuit_breaker.can_execute().await {
                    warn!("Circuit breaker is open for provider: {}", provider_name);
                    continue;
                }
            }
            if !self
                .health_checker
                .is_provider_healthy(&provider_name)
                .await
            {
                warn!("Provider {} is unhealthy, skipping", provider_name);
                continue;
            }

            let provider = match self.providers.get(&provider_name) {
                Some(p) => p,
                None => continue,
            };
            if !provider.supports_streaming() {
                continue;
            }

            match provider.generate_stream(&model_name, &request).await {
                Ok(stream) => {
                    self.record_success(&provider_name, start_time.elapsed())
                        .await;
                    info!("Streaming response via provider {}", provider_name);
                    return Ok(stream);
                }
                Err(e) => {
                    self.record_failure(&provider_name, start_time.elapsed())
                        .await;
                    warn!("Provider {} streaming failed: {}", provider_name, e);
                    last_error = Some(e);
                    continue;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("All providers failed to stream")))
    }

    async fn try_provider(
        &self,
        provider_name: &str,
        model_name: &str,
        request: &LLMRequest,
    ) -> Result<LLMResponse> {
        let provider = self
            .providers
            .get(provider_name)
            .ok_or_else(|| anyhow!("Provider {} not found", provider_name))?;

        let response = provider.generate(model_name, request).await?;
        Ok(response)
    }

    async fn get_provider_fallback_chain(&self) -> Vec<(String, String)> {
        let mut chain = Vec::new();
        let priority_order = ["openai", "anthropic", "local"];

        for provider_name in priority_order {
            if let Some(provider_config) = self.config.providers.get(provider_name) {
                if provider_config.enabled && self.providers.contains_key(provider_name) {
                    if let Some(model) = provider_config.models.first() {
                        chain.push((provider_name.to_string(), model.name.clone()));
                    }
                }
            }
        }

        chain
    }

    async fn record_success(&self, provider_name: &str, latency: Duration) {
        if let Some(circuit_breaker) = self.circuit_breakers.get(provider_name) {
            circuit_breaker.record_result(true, latency).await;
        }

        let provider_id = provider_name.to_string();
        if let Err(e) = self
            .health_checker
            .record_call(&provider_id, true, latency)
            .await
        {
            error!("Failed to record success for {}: {}", provider_name, e);
        }
    }

    async fn record_failure(&self, provider_name: &str, latency: Duration) {
        if let Some(circuit_breaker) = self.circuit_breakers.get(provider_name) {
            circuit_breaker.record_result(false, latency).await;
        }

        let provider_id = provider_name.to_string();
        if let Err(e) = self
            .health_checker
            .record_call(&provider_id, false, latency)
            .await
        {
            error!("Failed to record failure for {}: {}", provider_name, e);
        }
    }

    pub fn estimate_input_tokens(&self, request: &LLMRequest) -> usize {
        let total_content: String = request
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .chain(request.system_prompt.as_deref())
            .collect::<Vec<_>>()
            .join(" ");

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

/// Enhanced LLM manager with rate limiting and monitoring, plus Version 1.3 capabilities
pub struct EnhancedLLMManager {
    pub(crate) inner: LLMManager,
    pub(crate) rate_limiter: RateLimiter,
    pub(crate) sessions: Arc<TokioMutex<HashMap<String, LockedSession>>>,
    pub(crate) fine_tuning_engine: Option<Arc<FineTuningEngine>>,
    pub(crate) architecture_search: Option<Arc<ArchitectureSearch>>,
    pub(crate) federated_coordinator: Option<Arc<FederatedCoordinator>>,
    pub(crate) real_time_adaptation: Option<Arc<RealTimeAdaptation>>,
    pub(crate) cross_modal_reasoning: Option<Arc<CrossModalReasoning>>,
    pub(crate) performance_optimizer: Option<Arc<PerformanceOptimizer>>,
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
            fine_tuning_engine: None,
            architecture_search: None,
            federated_coordinator: None,
            real_time_adaptation: None,
            cross_modal_reasoning: None,
            performance_optimizer: None,
        })
    }

    /// Enable fine-tuning capabilities
    pub fn with_fine_tuning(mut self, config: super::fine_tuning::EngineConfig) -> Self {
        self.fine_tuning_engine = Some(Arc::new(FineTuningEngine::new(config)));
        self
    }

    /// Enable neural architecture search
    pub fn with_architecture_search(mut self) -> Self {
        self.architecture_search = Some(Arc::new(ArchitectureSearch::new()));
        self
    }

    /// Enable federated learning
    pub fn with_federated_learning(mut self, config: FederatedLearningConfig) -> Self {
        self.federated_coordinator = Some(Arc::new(FederatedCoordinator::new(config)));
        self
    }

    /// Enable real-time adaptation
    pub fn with_real_time_adaptation(mut self, config: AdaptationConfig) -> Self {
        self.real_time_adaptation = Some(Arc::new(RealTimeAdaptation::new(config)));
        self
    }

    /// Enable cross-modal reasoning
    pub fn with_cross_modal_reasoning(mut self, config: CrossModalConfig) -> Self {
        if let Ok(placeholder_manager) = LLMManager::new(LLMConfig::default()) {
            let inner_manager = Arc::new(tokio::sync::RwLock::new(placeholder_manager));
            self.cross_modal_reasoning =
                Some(Arc::new(CrossModalReasoning::new(config, inner_manager)));
        }
        self
    }

    /// Enable performance optimization
    pub fn with_performance_optimization(mut self, config: PerformanceConfig) -> Self {
        self.performance_optimizer = Some(Arc::new(PerformanceOptimizer::new(config)));
        self
    }

    pub async fn generate_response_with_limits(
        &mut self,
        request: LLMRequest,
    ) -> Result<LLMResponse> {
        self.rate_limiter.check_rate_limit("default").await?;
        self.inner.generate_response(request).await
    }

    /// Create a new enhanced LLM manager with persistence capabilities
    pub async fn with_persistence<P: AsRef<std::path::Path>>(
        _store: Arc<dyn oxirs_core::Store>,
        persistence_path: P,
    ) -> Result<Self> {
        let path = persistence_path.as_ref();
        if !path.exists() {
            std::fs::create_dir_all(path)
                .map_err(|e| anyhow!("Failed to create persistence directory: {}", e))?;
        }

        let mut config = LLMConfig::default();
        config.rate_limits.burst_allowed = true;

        let mut manager = Self::new(config)?;
        manager.load_persisted_sessions(path).await?;
        manager
            .setup_session_persistence(path.to_path_buf())
            .await?;

        info!(
            "Enhanced LLM manager initialized with persistence at: {:?}",
            path
        );
        Ok(manager)
    }

    async fn load_persisted_sessions<P: AsRef<std::path::Path>>(
        &mut self,
        persistence_path: P,
    ) -> Result<()> {
        let path = persistence_path.as_ref();
        let session_files = std::fs::read_dir(path)
            .map_err(|e| anyhow!("Failed to read persistence directory: {}", e))?;

        let mut loaded_count = 0;

        for entry in session_files.flatten() {
            let file_path = entry.path();
            if file_path.extension().and_then(|s| s.to_str()) == Some("session") {
                if let Some(session_id) = file_path.file_stem().and_then(|s| s.to_str()) {
                    match self
                        .load_session_from_file(&file_path, session_id.to_string())
                        .await
                    {
                        Ok(_) => {
                            loaded_count += 1;
                            debug!("Loaded session: {}", session_id);
                        }
                        Err(e) => {
                            warn!(
                                "Failed to load session {} from {:?}: {}",
                                session_id, file_path, e
                            );
                        }
                    }
                }
            }
        }

        info!("Loaded {} persisted sessions", loaded_count);
        Ok(())
    }

    async fn load_session_from_file<P: AsRef<std::path::Path>>(
        &mut self,
        file_path: P,
        session_id: String,
    ) -> Result<()> {
        use tokio::fs::File;
        use tokio::io::AsyncReadExt;

        let mut file = File::open(file_path).await?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents).await?;

        let session: Session =
            oxicode::serde::decode_from_slice(&contents, oxicode::config::standard())
                .map_err(|e| anyhow!("Failed to deserialize session: {}", e))?
                .0;

        let locked_session = Arc::new(TokioMutex::new(session));
        let mut sessions = self.sessions.lock().await;
        sessions.insert(session_id, locked_session);

        Ok(())
    }

    async fn setup_session_persistence(&self, persistence_path: std::path::PathBuf) -> Result<()> {
        let sessions = Arc::clone(&self.sessions);
        let path = persistence_path;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300));

            loop {
                interval.tick().await;

                let sessions_guard = sessions.lock().await;
                for (session_id, session_arc) in sessions_guard.iter() {
                    let session_guard = session_arc.lock().await;
                    let session_file = path.join(format!("{session_id}.session"));

                    match Self::save_session_to_file(&session_guard, &session_file).await {
                        Ok(_) => debug!("Saved session: {}", session_id),
                        Err(e) => error!("Failed to save session {}: {}", session_id, e),
                    }
                }
            }
        });

        Ok(())
    }

    async fn save_session_to_file<P: AsRef<std::path::Path>>(
        session: &Session,
        file_path: P,
    ) -> Result<()> {
        use tokio::fs::File;
        use tokio::io::AsyncWriteExt;

        let serialized = oxicode::serde::encode_to_vec(session, oxicode::config::standard())
            .map_err(|e| anyhow!("Failed to serialize session: {}", e))?;

        let temp_path = file_path.as_ref().with_extension("session.tmp");

        {
            let mut file = File::create(&temp_path).await?;
            file.write_all(&serialized).await?;
            file.sync_all().await?;
        }

        std::fs::rename(&temp_path, file_path.as_ref())?;

        Ok(())
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

        let mut total_duration_secs = 0.0;
        let mut valid_sessions = 0;

        for session_arc in sessions.values() {
            let session = session_arc.lock().await;

            let duration = session
                .last_activity
                .signed_duration_since(chrono::DateTime::<chrono::Utc>::from(session.created_at))
                .to_std()
                .unwrap_or(Duration::from_secs(0));

            total_duration_secs += duration.as_secs_f64();
            valid_sessions += 1;
        }

        let average_session_duration = if valid_sessions > 0 {
            total_duration_secs / valid_sessions as f64
        } else {
            0.0
        };

        SessionStats {
            active_sessions: sessions.len(),
            total_sessions: sessions.len(),
            average_session_duration,
        }
    }

    /// Get detailed metrics
    pub async fn get_detailed_metrics(&self) -> DetailedMetrics {
        let sessions = self.sessions.lock().await;
        let mut metrics = DetailedMetrics {
            total_sessions: sessions.len(),
            active_sessions: sessions.len(),
            ..Default::default()
        };

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

        tokio::fs::create_dir_all(backup_dir)
            .await
            .map_err(|e| anyhow!("Failed to create backup directory: {}", e))?;

        let mut successful_backups = 0;
        let mut failed_backups = 0;
        let mut total_size = 0u64;

        for (session_id, session_arc) in sessions.iter() {
            match session_arc.try_lock() {
                Ok(session) => {
                    let session_data = serde_json::to_string(&*session).map_err(|e| {
                        anyhow!("Failed to serialize session {}: {}", session_id, e)
                    })?;

                    let session_file = backup_dir.join(format!("{session_id}.json"));
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

        if !restore_dir.exists() {
            return Err(anyhow!(
                "Restore directory does not exist: {:?}",
                restore_dir
            ));
        }

        let mut sessions_restored = 0;
        let mut failed_restorations = 0;
        let mut total_size = 0u64;

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
                        if let Ok(metadata) = tokio::fs::metadata(&path).await {
                            total_size += metadata.len();
                        }

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

        for (session_id, session_arc) in sessions.iter() {
            match session_arc.try_lock() {
                Ok(session_guard) => {
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
                    debug!(
                        "Session {} is locked, skipping expiration check",
                        session_id
                    );
                }
            }
        }

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

    // ── Version 1.3 Capability Methods ───────────────────────────────────────────

    /// Submit a fine-tuning job
    pub async fn submit_fine_tuning_job(&self, config: FineTuningConfig) -> Result<String> {
        if let Some(engine) = &self.fine_tuning_engine {
            engine.submit_job(config).await
        } else {
            Err(anyhow!("Fine-tuning engine not enabled"))
        }
    }

    /// Get fine-tuning job status
    pub async fn get_fine_tuning_status(
        &self,
        job_id: &str,
    ) -> Result<super::fine_tuning::FineTuningJob> {
        if let Some(engine) = &self.fine_tuning_engine {
            engine.get_job_status(job_id).await
        } else {
            Err(anyhow!("Fine-tuning engine not enabled"))
        }
    }

    /// Start neural architecture search
    pub async fn start_architecture_search(
        &self,
        config: ArchitectureSearchConfig,
    ) -> Result<String> {
        if let Some(search) = &self.architecture_search {
            search.start_search(config).await
        } else {
            Err(anyhow!("Architecture search not enabled"))
        }
    }

    /// Get architecture search status
    pub async fn get_architecture_search_status(
        &self,
        search_id: &str,
    ) -> Result<super::neural_architecture_search::SearchState> {
        if let Some(search) = &self.architecture_search {
            search.get_search_status(search_id).await
        } else {
            Err(anyhow!("Architecture search not enabled"))
        }
    }

    /// Start federated learning round
    pub async fn start_federation_round(&self) -> Result<usize> {
        if let Some(coordinator) = &self.federated_coordinator {
            coordinator.start_federation_round().await
        } else {
            Err(anyhow!("Federated learning not enabled"))
        }
    }

    /// Register federated learning node
    pub async fn register_federated_node(
        &self,
        node: super::federated_learning::FederatedNode,
    ) -> Result<()> {
        if let Some(coordinator) = &self.federated_coordinator {
            coordinator.register_node(node).await
        } else {
            Err(anyhow!("Federated learning not enabled"))
        }
    }

    /// Process interaction for real-time adaptation
    pub async fn process_adaptation_interaction(
        &self,
        request: &LLMRequest,
        response: &LLMResponse,
    ) -> Result<()> {
        if let Some(adaptation) = &self.real_time_adaptation {
            let interaction = InteractionData {
                interaction_id: format!("interaction_{}", uuid::Uuid::new_v4()),
                request: request.clone(),
                response: response.clone(),
                user_feedback: None,
                context_information: super::real_time_adaptation::ContextInformation {
                    user_profile: super::real_time_adaptation::UserProfile {
                        user_id: "default_user".to_string(),
                        expertise_level: super::real_time_adaptation::ExpertiseLevel::Intermediate,
                        preferences: super::real_time_adaptation::UserPreferences {
                            response_style:
                                super::real_time_adaptation::ResponseStyle::Conversational,
                            detail_level: super::real_time_adaptation::DetailLevel::Medium,
                            preferred_formats: vec!["text".to_string()],
                            language_preferences: vec!["en".to_string()],
                        },
                        interaction_history: super::real_time_adaptation::InteractionHistory {
                            total_interactions: 0,
                            average_satisfaction: 0.8,
                            common_topics: vec![],
                            feedback_patterns: HashMap::new(),
                        },
                    },
                    session_context: super::real_time_adaptation::SessionContext {
                        session_id: "default_session".to_string(),
                        session_duration: Duration::from_secs(300),
                        conversation_flow: super::real_time_adaptation::ConversationFlow {
                            topic_transitions: vec![],
                            question_types: vec![],
                            complexity_progression: vec![],
                        },
                        current_objectives: vec![],
                    },
                    domain_context: super::real_time_adaptation::DomainContext {
                        primary_domain: "general".to_string(),
                        secondary_domains: vec![],
                        domain_expertise_required: 0.5,
                        domain_specific_patterns: HashMap::new(),
                    },
                    temporal_context: super::real_time_adaptation::TemporalContext {
                        time_of_day: "day".to_string(),
                        day_of_week: "weekday".to_string(),
                        seasonal_patterns: vec![],
                        trending_topics: vec![],
                    },
                },
                timestamp: std::time::SystemTime::now(),
            };
            adaptation.process_interaction(interaction).await
        } else {
            Ok(())
        }
    }

    /// Enhanced response generation with adaptation tracking
    pub async fn generate_enhanced_response(&mut self, request: LLMRequest) -> Result<LLMResponse> {
        let response = self.generate_response_with_limits(request.clone()).await?;
        self.process_adaptation_interaction(&request, &response)
            .await?;
        Ok(response)
    }

    /// Perform cross-modal reasoning
    pub async fn perform_cross_modal_reasoning(
        &self,
        input: CrossModalInput,
        query: &str,
    ) -> Result<CrossModalResponse> {
        if let Some(reasoning) = &self.cross_modal_reasoning {
            reasoning.reason(input, query).await.map_err(|e| anyhow!(e))
        } else {
            Err(anyhow!("Cross-modal reasoning not enabled"))
        }
    }

    /// Get cross-modal reasoning history
    pub async fn get_cross_modal_history(&self) -> Result<Vec<CrossModalResponse>> {
        if let Some(reasoning) = &self.cross_modal_reasoning {
            Ok(reasoning.get_reasoning_history().await)
        } else {
            Err(anyhow!("Cross-modal reasoning not enabled"))
        }
    }

    /// Clear cross-modal reasoning history
    pub async fn clear_cross_modal_history(&self) -> Result<()> {
        if let Some(reasoning) = &self.cross_modal_reasoning {
            reasoning.clear_history().await;
            Ok(())
        } else {
            Err(anyhow!("Cross-modal reasoning not enabled"))
        }
    }

    /// Get cross-modal reasoning statistics
    pub async fn get_cross_modal_stats(
        &self,
    ) -> Result<super::cross_modal_reasoning::CrossModalStats> {
        if let Some(reasoning) = &self.cross_modal_reasoning {
            Ok(reasoning.get_stats().await)
        } else {
            Err(anyhow!("Cross-modal reasoning not enabled"))
        }
    }

    /// Optimize request for better performance
    pub async fn optimize_request(
        &self,
        request: &LLMRequest,
    ) -> Result<super::performance_optimization::OptimizedRequest> {
        if let Some(optimizer) = &self.performance_optimizer {
            optimizer
                .optimize_request(request)
                .await
                .map_err(|e| anyhow!(e))
        } else {
            Err(anyhow!("Performance optimization not enabled"))
        }
    }

    /// Run system benchmark
    pub async fn run_benchmark(&self, config: BenchmarkConfig) -> Result<BenchmarkResult> {
        if let Some(optimizer) = &self.performance_optimizer {
            optimizer
                .benchmark_system(config)
                .await
                .map_err(|e| anyhow!(e))
        } else {
            Err(anyhow!("Performance optimization not enabled"))
        }
    }

    /// Get optimization recommendations
    pub async fn get_optimization_recommendations(
        &self,
    ) -> Result<Vec<OptimizationRecommendation>> {
        if let Some(optimizer) = &self.performance_optimizer {
            optimizer
                .generate_optimization_recommendations()
                .await
                .map_err(|e| anyhow!(e))
        } else {
            Err(anyhow!("Performance optimization not enabled"))
        }
    }

    /// Get comprehensive performance report
    pub async fn get_performance_report(&self) -> Result<PerformanceReport> {
        if let Some(optimizer) = &self.performance_optimizer {
            optimizer
                .get_performance_report()
                .await
                .map_err(|e| anyhow!(e))
        } else {
            Err(anyhow!("Performance optimization not enabled"))
        }
    }

    /// Get comprehensive system statistics including Version 1.3 capabilities
    pub async fn get_comprehensive_stats(&self) -> Result<ComprehensiveStats> {
        let basic_stats = self.get_session_stats().await;

        let fine_tuning_stats = if let Some(engine) = &self.fine_tuning_engine {
            Some(engine.get_training_statistics().await?)
        } else {
            None
        };

        let federation_stats = if let Some(coordinator) = &self.federated_coordinator {
            Some(coordinator.get_federation_statistics().await?)
        } else {
            None
        };

        let adaptation_metrics = if let Some(adaptation) = &self.real_time_adaptation {
            Some(adaptation.get_adaptation_metrics().await?)
        } else {
            None
        };

        let cross_modal_stats = if let Some(reasoning) = &self.cross_modal_reasoning {
            Some(reasoning.get_stats().await)
        } else {
            None
        };

        let performance_report = if let Some(optimizer) = &self.performance_optimizer {
            Some(
                optimizer
                    .get_performance_report()
                    .await
                    .unwrap_or_else(|_| PerformanceReport {
                        current_metrics:
                            super::performance_optimization::PerformanceMetrics::default(),
                        benchmark_results: Vec::new(),
                        recommendations: Vec::new(),
                        cache_statistics: super::performance_optimization::CacheStatistics {
                            total_entries: 0,
                            total_size_bytes: 0,
                            hit_rate: 0.0,
                            miss_rate: 1.0,
                            eviction_count: 0,
                            average_access_count: 0.0,
                            average_compression_ratio: 0.0,
                        },
                        compression_statistics:
                            super::performance_optimization::CompressionStatistics {
                                total_compressed_requests: 0,
                                average_compression_ratio: 0.0,
                                total_bytes_saved: 0,
                                compression_time_average: Duration::from_millis(0),
                            },
                        optimization_summary:
                            super::performance_optimization::OptimizationSummary {
                                overall_performance_score: 0.0,
                                target_achievement_rate: 0.0,
                                bottleneck_analysis: Vec::new(),
                                improvement_potential: 0.0,
                                optimization_status:
                                    super::performance_optimization::OptimizationStatus::Critical,
                            },
                        generated_at: std::time::SystemTime::now(),
                    }),
            )
        } else {
            None
        };

        Ok(ComprehensiveStats {
            session_stats: basic_stats,
            fine_tuning_stats,
            federation_stats,
            adaptation_metrics,
            cross_modal_stats,
            performance_report,
            version: "1.3+".to_string(),
            capabilities_enabled: CapabilityStatus {
                fine_tuning: self.fine_tuning_engine.is_some(),
                architecture_search: self.architecture_search.is_some(),
                federated_learning: self.federated_coordinator.is_some(),
                real_time_adaptation: self.real_time_adaptation.is_some(),
                cross_modal_reasoning: self.cross_modal_reasoning.is_some(),
                performance_optimization: self.performance_optimizer.is_some(),
            },
        })
    }
}
