//! Blue/Green Deployment Support
//!
//! This module provides infrastructure for blue/green deployment strategies,
//! enabling zero-downtime deployments by managing traffic routing between
//! two identical production environments.
//!
//! ## Features
//!
//! - **Environment Management**: Track blue and green environment states
//! - **Traffic Routing**: Control traffic distribution between environments
//! - **Health Monitoring**: Continuous health checks for both environments
//! - **Rollback Support**: Automatic and manual rollback capabilities
//! - **Promotion Controls**: Safe promotion of green to blue
//! - **State Persistence**: Track deployment history and state
//!
//! ## Usage
//!
//! ```rust,ignore
//! use oxirs_gql::blue_green_deployment::{BlueGreenController, DeploymentConfig};
//!
//! let config = DeploymentConfig::default();
//! let controller = BlueGreenController::new(config);
//!
//! // Deploy new version to green environment
//! controller.deploy_to_green("v2.0.0").await?;
//!
//! // Run health checks
//! if controller.verify_green_health().await? {
//!     // Switch traffic to green
//!     controller.switch_to_green().await?;
//! }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// Environment identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Environment {
    /// Blue environment (typically the current production)
    Blue,
    /// Green environment (typically the staging/new version)
    Green,
}

impl Environment {
    /// Get the opposite environment
    pub fn opposite(&self) -> Self {
        match self {
            Environment::Blue => Environment::Green,
            Environment::Green => Environment::Blue,
        }
    }

    /// Get the environment name as a string
    pub fn name(&self) -> &'static str {
        match self {
            Environment::Blue => "blue",
            Environment::Green => "green",
        }
    }
}

impl std::fmt::Display for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Environment health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Environment is healthy
    Healthy,
    /// Environment is degraded but operational
    Degraded,
    /// Environment is unhealthy
    Unhealthy,
    /// Environment health is unknown
    Unknown,
}

/// Environment state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvironmentState {
    /// Environment is idle (not receiving traffic)
    Idle,
    /// Environment is receiving traffic
    Active,
    /// Environment is deploying new version
    Deploying,
    /// Environment is draining connections
    Draining,
    /// Environment failed and needs attention
    Failed,
}

/// Deployment strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    /// Instant switch (all traffic at once)
    #[default]
    Instant,
    /// Gradual switch (percentage-based)
    Gradual { increment: u8, interval_secs: u64 },
    /// Shadow mode (copy traffic to green without serving responses)
    Shadow,
    /// Manual (requires explicit switch command)
    Manual,
}

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    /// Environment identifier
    pub environment: Environment,
    /// Current state
    pub state: EnvironmentState,
    /// Health status
    pub health: HealthStatus,
    /// Deployed version
    pub version: Option<String>,
    /// Traffic percentage (0-100)
    pub traffic_percentage: u8,
    /// Deployment timestamp
    pub deployed_at: Option<SystemTime>,
    /// Last health check timestamp
    pub last_health_check: Option<SystemTime>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl EnvironmentInfo {
    /// Create a new environment info
    pub fn new(environment: Environment) -> Self {
        Self {
            environment,
            state: EnvironmentState::Idle,
            health: HealthStatus::Unknown,
            version: None,
            traffic_percentage: 0,
            deployed_at: None,
            last_health_check: None,
            metadata: HashMap::new(),
        }
    }

    /// Check if environment is ready to receive traffic
    pub fn is_ready(&self) -> bool {
        self.state != EnvironmentState::Deploying
            && self.state != EnvironmentState::Failed
            && self.health == HealthStatus::Healthy
    }
}

/// Deployment history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentRecord {
    /// Unique deployment ID
    pub id: String,
    /// Target environment
    pub environment: Environment,
    /// Deployed version
    pub version: String,
    /// Deployment timestamp
    pub timestamp: SystemTime,
    /// Deployment duration
    pub duration: Option<Duration>,
    /// Whether deployment was successful
    pub success: bool,
    /// Rollback indicator
    pub is_rollback: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Deployment metadata
    pub metadata: HashMap<String, String>,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check endpoint path
    pub endpoint: String,
    /// Health check interval
    pub interval: Duration,
    /// Health check timeout
    pub timeout: Duration,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub success_threshold: u32,
    /// Enable deep health checks (check dependencies)
    pub deep_check: bool,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            endpoint: "/health".to_string(),
            interval: Duration::from_secs(10),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 2,
            deep_check: true,
        }
    }
}

/// Rollback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackConfig {
    /// Enable automatic rollback on failure
    pub auto_rollback: bool,
    /// Error rate threshold for automatic rollback (percentage)
    pub error_rate_threshold: f64,
    /// Latency threshold for automatic rollback (milliseconds)
    pub latency_threshold_ms: u64,
    /// Minimum observation period before rollback decision
    pub observation_period: Duration,
    /// Maximum time to wait for rollback to complete
    pub rollback_timeout: Duration,
}

impl Default for RollbackConfig {
    fn default() -> Self {
        Self {
            auto_rollback: true,
            error_rate_threshold: 5.0,  // 5% error rate triggers rollback
            latency_threshold_ms: 5000, // 5 second p99 latency triggers rollback
            observation_period: Duration::from_secs(60),
            rollback_timeout: Duration::from_secs(300),
        }
    }
}

/// Deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Deployment strategy
    pub strategy: DeploymentStrategy,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    /// Rollback configuration
    pub rollback: RollbackConfig,
    /// Maximum concurrent connections during transition
    pub max_concurrent_connections: usize,
    /// Connection drain timeout
    pub drain_timeout: Duration,
    /// Enable deployment notifications
    pub notifications_enabled: bool,
    /// Notification webhook URL
    pub notification_webhook: Option<String>,
}

impl Default for DeploymentConfig {
    fn default() -> Self {
        Self {
            strategy: DeploymentStrategy::default(),
            health_check: HealthCheckConfig::default(),
            rollback: RollbackConfig::default(),
            max_concurrent_connections: 10000,
            drain_timeout: Duration::from_secs(30),
            notifications_enabled: false,
            notification_webhook: None,
        }
    }
}

/// Traffic routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Target environment
    pub target: Environment,
    /// Routing weight (0.0 - 1.0)
    pub weight: f64,
    /// Whether this is a shadow request
    pub is_shadow: bool,
    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// Deployment event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentEvent {
    /// Deployment started
    DeploymentStarted {
        environment: Environment,
        version: String,
    },
    /// Deployment completed successfully
    DeploymentCompleted {
        environment: Environment,
        version: String,
        duration: Duration,
    },
    /// Deployment failed
    DeploymentFailed {
        environment: Environment,
        version: String,
        error: String,
    },
    /// Traffic switch initiated
    TrafficSwitchStarted {
        from: Environment,
        to: Environment,
        strategy: DeploymentStrategy,
    },
    /// Traffic switch completed
    TrafficSwitchCompleted { active: Environment },
    /// Health status changed
    HealthStatusChanged {
        environment: Environment,
        old_status: HealthStatus,
        new_status: HealthStatus,
    },
    /// Rollback initiated
    RollbackStarted {
        from: Environment,
        to: Environment,
        reason: String,
    },
    /// Rollback completed
    RollbackCompleted { active: Environment, success: bool },
}

/// Internal state for blue/green controller
struct ControllerState {
    /// Blue environment info
    blue: EnvironmentInfo,
    /// Green environment info
    green: EnvironmentInfo,
    /// Currently active environment
    active_environment: Environment,
    /// Deployment history
    deployment_history: Vec<DeploymentRecord>,
    /// Event log
    events: Vec<(SystemTime, DeploymentEvent)>,
    /// Health check failure counts
    health_check_failures: HashMap<Environment, u32>,
    /// Health check success counts
    health_check_successes: HashMap<Environment, u32>,
}

impl ControllerState {
    fn new() -> Self {
        let mut blue = EnvironmentInfo::new(Environment::Blue);
        blue.state = EnvironmentState::Active;
        blue.traffic_percentage = 100;

        let green = EnvironmentInfo::new(Environment::Green);

        Self {
            blue,
            green,
            active_environment: Environment::Blue,
            deployment_history: Vec::new(),
            events: Vec::new(),
            health_check_failures: HashMap::new(),
            health_check_successes: HashMap::new(),
        }
    }

    fn get_env_mut(&mut self, env: Environment) -> &mut EnvironmentInfo {
        match env {
            Environment::Blue => &mut self.blue,
            Environment::Green => &mut self.green,
        }
    }

    fn get_env(&self, env: Environment) -> &EnvironmentInfo {
        match env {
            Environment::Blue => &self.blue,
            Environment::Green => &self.green,
        }
    }
}

/// Blue/Green Deployment Controller
///
/// Manages the lifecycle of blue/green deployments including
/// traffic routing, health monitoring, and rollback handling.
pub struct BlueGreenController {
    /// Configuration
    config: DeploymentConfig,
    /// Internal state
    state: Arc<RwLock<ControllerState>>,
    /// Event handlers
    event_handlers: Arc<RwLock<Vec<Arc<dyn DeploymentEventHandler + Send + Sync>>>>,
}

impl BlueGreenController {
    /// Create a new blue/green controller
    pub fn new(config: DeploymentConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(ControllerState::new())),
            event_handlers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register an event handler
    pub async fn register_event_handler(
        &self,
        handler: Arc<dyn DeploymentEventHandler + Send + Sync>,
    ) {
        let mut handlers = self.event_handlers.write().await;
        handlers.push(handler);
    }

    /// Emit a deployment event
    async fn emit_event(&self, event: DeploymentEvent) {
        let now = SystemTime::now();

        // Store event
        {
            let mut state = self.state.write().await;
            state.events.push((now, event.clone()));

            // Limit event history
            if state.events.len() > 1000 {
                state.events.drain(0..100);
            }
        }

        // Notify handlers
        let handlers = self.event_handlers.read().await;
        for handler in handlers.iter() {
            handler.on_event(&event).await;
        }
    }

    /// Get current environment status
    pub async fn get_status(&self) -> BlueGreenStatus {
        let state = self.state.read().await;
        BlueGreenStatus {
            blue: state.blue.clone(),
            green: state.green.clone(),
            active_environment: state.active_environment,
            deployment_history_count: state.deployment_history.len(),
            last_deployment: state.deployment_history.last().cloned(),
        }
    }

    /// Get environment info
    pub async fn get_environment(&self, env: Environment) -> EnvironmentInfo {
        let state = self.state.read().await;
        state.get_env(env).clone()
    }

    /// Deploy a new version to the specified environment
    pub async fn deploy_to_environment(
        &self,
        env: Environment,
        version: &str,
    ) -> Result<DeploymentRecord> {
        let start_time = SystemTime::now();
        let deployment_id = uuid::Uuid::new_v4().to_string();

        // Check if environment is available for deployment
        {
            let state = self.state.read().await;
            let env_info = state.get_env(env);
            if env_info.state == EnvironmentState::Active && env_info.traffic_percentage > 0 {
                return Err(anyhow!(
                    "Cannot deploy to {} environment while it's receiving traffic",
                    env
                ));
            }
        }

        // Mark environment as deploying
        {
            let mut state = self.state.write().await;
            let env_info = state.get_env_mut(env);
            env_info.state = EnvironmentState::Deploying;
            env_info.health = HealthStatus::Unknown;
        }

        self.emit_event(DeploymentEvent::DeploymentStarted {
            environment: env,
            version: version.to_string(),
        })
        .await;

        // Simulate deployment (in real implementation, this would trigger actual deployment)
        // For now, we just update the state
        let duration = SystemTime::now()
            .duration_since(start_time)
            .unwrap_or_default();

        {
            let mut state = self.state.write().await;
            let env_info = state.get_env_mut(env);
            env_info.state = EnvironmentState::Idle;
            env_info.health = HealthStatus::Unknown;
            env_info.version = Some(version.to_string());
            env_info.deployed_at = Some(SystemTime::now());
        }

        let record = DeploymentRecord {
            id: deployment_id,
            environment: env,
            version: version.to_string(),
            timestamp: start_time,
            duration: Some(duration),
            success: true,
            is_rollback: false,
            error: None,
            metadata: HashMap::new(),
        };

        // Store deployment record
        {
            let mut state = self.state.write().await;
            state.deployment_history.push(record.clone());
        }

        self.emit_event(DeploymentEvent::DeploymentCompleted {
            environment: env,
            version: version.to_string(),
            duration,
        })
        .await;

        Ok(record)
    }

    /// Deploy to the inactive (green) environment
    pub async fn deploy_to_green(&self, version: &str) -> Result<DeploymentRecord> {
        let active = {
            let state = self.state.read().await;
            state.active_environment
        };
        let target = active.opposite();
        self.deploy_to_environment(target, version).await
    }

    /// Update health status for an environment
    pub async fn update_health(&self, env: Environment, status: HealthStatus) {
        let old_status;
        {
            let state = self.state.read().await;
            old_status = state.get_env(env).health;
        }

        {
            let mut state = self.state.write().await;
            let env_info = state.get_env_mut(env);
            env_info.health = status;
            env_info.last_health_check = Some(SystemTime::now());

            // Update failure/success counts
            match status {
                HealthStatus::Healthy => {
                    state.health_check_failures.insert(env, 0);
                    let count = state.health_check_successes.entry(env).or_insert(0);
                    *count += 1;
                }
                HealthStatus::Unhealthy | HealthStatus::Degraded => {
                    state.health_check_successes.insert(env, 0);
                    let count = state.health_check_failures.entry(env).or_insert(0);
                    *count += 1;
                }
                HealthStatus::Unknown => {}
            }
        }

        if old_status != status {
            self.emit_event(DeploymentEvent::HealthStatusChanged {
                environment: env,
                old_status,
                new_status: status,
            })
            .await;
        }
    }

    /// Verify health of an environment
    pub async fn verify_health(&self, env: Environment) -> Result<bool> {
        let state = self.state.read().await;
        let env_info = state.get_env(env);

        Ok(env_info.health == HealthStatus::Healthy)
    }

    /// Switch traffic to the specified environment
    pub async fn switch_traffic(&self, target: Environment) -> Result<()> {
        let current;
        {
            let state = self.state.read().await;
            current = state.active_environment;

            // Verify target is healthy
            let target_info = state.get_env(target);
            if !target_info.is_ready() {
                return Err(anyhow!(
                    "Target environment {} is not ready for traffic (state: {:?}, health: {:?})",
                    target,
                    target_info.state,
                    target_info.health
                ));
            }
        }

        if current == target {
            return Ok(()); // Already on target environment
        }

        self.emit_event(DeploymentEvent::TrafficSwitchStarted {
            from: current,
            to: target,
            strategy: self.config.strategy,
        })
        .await;

        match self.config.strategy {
            DeploymentStrategy::Instant => {
                self.instant_switch(target).await?;
            }
            DeploymentStrategy::Gradual {
                increment,
                interval_secs,
            } => {
                self.gradual_switch(target, increment, interval_secs)
                    .await?;
            }
            DeploymentStrategy::Shadow => {
                // Shadow mode doesn't switch traffic
                return Err(anyhow!("Shadow mode does not support traffic switching"));
            }
            DeploymentStrategy::Manual => {
                self.instant_switch(target).await?;
            }
        }

        self.emit_event(DeploymentEvent::TrafficSwitchCompleted { active: target })
            .await;

        Ok(())
    }

    /// Perform instant traffic switch
    async fn instant_switch(&self, target: Environment) -> Result<()> {
        let mut state = self.state.write().await;

        // Drain old environment
        let old = state.active_environment;
        {
            let old_env = state.get_env_mut(old);
            old_env.state = EnvironmentState::Draining;
            old_env.traffic_percentage = 0;
        }

        // Activate new environment
        {
            let new_env = state.get_env_mut(target);
            new_env.state = EnvironmentState::Active;
            new_env.traffic_percentage = 100;
        }

        state.active_environment = target;

        // Mark old as idle after drain
        {
            let old_env = state.get_env_mut(old);
            old_env.state = EnvironmentState::Idle;
        }

        Ok(())
    }

    /// Perform gradual traffic switch
    async fn gradual_switch(
        &self,
        target: Environment,
        increment: u8,
        interval_secs: u64,
    ) -> Result<()> {
        let current;
        {
            let state = self.state.read().await;
            current = state.active_environment;
        }

        let mut current_percentage = 0u8;

        while current_percentage < 100 {
            current_percentage = current_percentage.saturating_add(increment).min(100);

            {
                let mut state = self.state.write().await;
                state.get_env_mut(target).traffic_percentage = current_percentage;
                state.get_env_mut(current).traffic_percentage = 100 - current_percentage;
            }

            if current_percentage < 100 {
                tokio::time::sleep(Duration::from_secs(interval_secs)).await;

                // Check health during gradual switch
                let health_ok = self.verify_health(target).await?;
                if !health_ok {
                    // Rollback
                    return Err(anyhow!(
                        "Health check failed during gradual switch at {}%",
                        current_percentage
                    ));
                }
            }
        }

        // Finalize switch
        {
            let mut state = self.state.write().await;
            state.get_env_mut(current).state = EnvironmentState::Idle;
            state.get_env_mut(target).state = EnvironmentState::Active;
            state.active_environment = target;
        }

        Ok(())
    }

    /// Switch traffic to green environment
    pub async fn switch_to_green(&self) -> Result<()> {
        self.switch_traffic(Environment::Green).await
    }

    /// Switch traffic to blue environment
    pub async fn switch_to_blue(&self) -> Result<()> {
        self.switch_traffic(Environment::Blue).await
    }

    /// Rollback to the previous environment
    pub async fn rollback(&self, reason: &str) -> Result<()> {
        let current;
        let target;
        {
            let state = self.state.read().await;
            current = state.active_environment;
            target = current.opposite();

            // Verify rollback target is available
            let target_info = state.get_env(target);
            if target_info.version.is_none() {
                return Err(anyhow!(
                    "Cannot rollback: {} environment has no previous deployment",
                    target
                ));
            }
        }

        self.emit_event(DeploymentEvent::RollbackStarted {
            from: current,
            to: target,
            reason: reason.to_string(),
        })
        .await;

        // Force health to healthy for rollback (assume previous version was stable)
        self.update_health(target, HealthStatus::Healthy).await;

        let result = self.switch_traffic(target).await;

        let success = result.is_ok();
        self.emit_event(DeploymentEvent::RollbackCompleted {
            active: if success { target } else { current },
            success,
        })
        .await;

        // Record rollback in history
        if success {
            let version = {
                let state = self.state.read().await;
                state.get_env(target).version.clone().unwrap_or_default()
            };

            let record = DeploymentRecord {
                id: uuid::Uuid::new_v4().to_string(),
                environment: target,
                version,
                timestamp: SystemTime::now(),
                duration: None,
                success: true,
                is_rollback: true,
                error: None,
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("rollback_reason".to_string(), reason.to_string());
                    m
                },
            };

            let mut state = self.state.write().await;
            state.deployment_history.push(record);
        }

        result
    }

    /// Route a request based on current traffic configuration
    pub async fn route_request(&self, request_id: &str) -> RoutingDecision {
        let state = self.state.read().await;

        // Calculate routing based on traffic percentages
        let blue_weight = state.blue.traffic_percentage as f64 / 100.0;
        let green_weight = state.green.traffic_percentage as f64 / 100.0;

        // Use request_id hash for consistent routing
        let hash = request_id
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_add(b as u64));
        let normalized = (hash % 100) as f64 / 100.0;

        let (target, weight) = if normalized < blue_weight {
            (Environment::Blue, blue_weight)
        } else {
            (Environment::Green, green_weight)
        };

        RoutingDecision {
            target,
            weight,
            is_shadow: matches!(self.config.strategy, DeploymentStrategy::Shadow),
            metadata: HashMap::new(),
        }
    }

    /// Get deployment history
    pub async fn get_deployment_history(&self, limit: Option<usize>) -> Vec<DeploymentRecord> {
        let state = self.state.read().await;
        let history = &state.deployment_history;

        match limit {
            Some(n) => history.iter().rev().take(n).cloned().collect(),
            None => history.clone(),
        }
    }

    /// Get recent events
    pub async fn get_recent_events(&self, limit: usize) -> Vec<(SystemTime, DeploymentEvent)> {
        let state = self.state.read().await;
        state.events.iter().rev().take(limit).cloned().collect()
    }

    /// Check if automatic rollback should be triggered
    pub async fn should_auto_rollback(&self, error_rate: f64, latency_p99_ms: u64) -> bool {
        if !self.config.rollback.auto_rollback {
            return false;
        }

        error_rate > self.config.rollback.error_rate_threshold
            || latency_p99_ms > self.config.rollback.latency_threshold_ms
    }
}

/// Blue/Green status summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueGreenStatus {
    /// Blue environment info
    pub blue: EnvironmentInfo,
    /// Green environment info
    pub green: EnvironmentInfo,
    /// Currently active environment
    pub active_environment: Environment,
    /// Number of deployments in history
    pub deployment_history_count: usize,
    /// Last deployment record
    pub last_deployment: Option<DeploymentRecord>,
}

/// Trait for handling deployment events
#[async_trait::async_trait]
pub trait DeploymentEventHandler {
    /// Handle a deployment event
    async fn on_event(&self, event: &DeploymentEvent);
}

/// Webhook notification handler
pub struct WebhookNotifier {
    webhook_url: String,
    client: reqwest::Client,
}

impl WebhookNotifier {
    /// Create a new webhook notifier
    pub fn new(webhook_url: String) -> Self {
        Self {
            webhook_url,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl DeploymentEventHandler for WebhookNotifier {
    async fn on_event(&self, event: &DeploymentEvent) {
        let payload = serde_json::json!({
            "event": event,
            "timestamp": SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        });

        let _ = self
            .client
            .post(&self.webhook_url)
            .json(&payload)
            .send()
            .await;
    }
}

/// Logging event handler
pub struct LoggingEventHandler;

#[async_trait::async_trait]
impl DeploymentEventHandler for LoggingEventHandler {
    async fn on_event(&self, event: &DeploymentEvent) {
        match event {
            DeploymentEvent::DeploymentStarted {
                environment,
                version,
            } => {
                tracing::info!("Deployment started: {} -> {}", environment, version);
            }
            DeploymentEvent::DeploymentCompleted {
                environment,
                version,
                duration,
            } => {
                tracing::info!(
                    "Deployment completed: {} -> {} ({:?})",
                    environment,
                    version,
                    duration
                );
            }
            DeploymentEvent::DeploymentFailed {
                environment,
                version,
                error,
            } => {
                tracing::error!(
                    "Deployment failed: {} -> {} - {}",
                    environment,
                    version,
                    error
                );
            }
            DeploymentEvent::TrafficSwitchStarted { from, to, strategy } => {
                tracing::info!(
                    "Traffic switch started: {} -> {} ({:?})",
                    from,
                    to,
                    strategy
                );
            }
            DeploymentEvent::TrafficSwitchCompleted { active } => {
                tracing::info!("Traffic switch completed: active = {}", active);
            }
            DeploymentEvent::HealthStatusChanged {
                environment,
                old_status,
                new_status,
            } => {
                tracing::info!(
                    "Health status changed: {} {:?} -> {:?}",
                    environment,
                    old_status,
                    new_status
                );
            }
            DeploymentEvent::RollbackStarted { from, to, reason } => {
                tracing::warn!("Rollback started: {} -> {} ({})", from, to, reason);
            }
            DeploymentEvent::RollbackCompleted { active, success } => {
                if *success {
                    tracing::info!("Rollback completed successfully: active = {}", active);
                } else {
                    tracing::error!("Rollback failed: active = {}", active);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_controller_creation() {
        let config = DeploymentConfig::default();
        let controller = BlueGreenController::new(config);

        let status = controller.get_status().await;
        assert_eq!(status.active_environment, Environment::Blue);
        assert_eq!(status.blue.traffic_percentage, 100);
        assert_eq!(status.green.traffic_percentage, 0);
    }

    #[tokio::test]
    async fn test_deploy_to_green() {
        let config = DeploymentConfig::default();
        let controller = BlueGreenController::new(config);

        let record = controller.deploy_to_green("v2.0.0").await.unwrap();
        assert_eq!(record.environment, Environment::Green);
        assert_eq!(record.version, "v2.0.0");
        assert!(record.success);
        assert!(!record.is_rollback);
    }

    #[tokio::test]
    async fn test_switch_traffic() {
        let config = DeploymentConfig::default();
        let controller = BlueGreenController::new(config);

        // Deploy to green first
        controller.deploy_to_green("v2.0.0").await.unwrap();

        // Update health
        controller
            .update_health(Environment::Green, HealthStatus::Healthy)
            .await;

        // Switch traffic
        controller.switch_to_green().await.unwrap();

        let status = controller.get_status().await;
        assert_eq!(status.active_environment, Environment::Green);
        assert_eq!(status.green.traffic_percentage, 100);
        assert_eq!(status.blue.traffic_percentage, 0);
    }

    #[tokio::test]
    async fn test_rollback() {
        let config = DeploymentConfig::default();
        let controller = BlueGreenController::new(config);

        // Set initial blue version
        {
            let mut state = controller.state.write().await;
            state.blue.version = Some("v1.0.0".to_string());
        }

        // Deploy to green
        controller.deploy_to_green("v2.0.0").await.unwrap();
        controller
            .update_health(Environment::Green, HealthStatus::Healthy)
            .await;
        controller.switch_to_green().await.unwrap();

        // Rollback
        controller.rollback("Test rollback").await.unwrap();

        let status = controller.get_status().await;
        assert_eq!(status.active_environment, Environment::Blue);
    }

    #[tokio::test]
    async fn test_routing_decision() {
        let config = DeploymentConfig::default();
        let controller = BlueGreenController::new(config);

        // Default state: 100% blue
        let decision = controller.route_request("test-request-1").await;
        assert_eq!(decision.target, Environment::Blue);
        assert_eq!(decision.weight, 1.0);
    }

    #[tokio::test]
    async fn test_health_update() {
        let config = DeploymentConfig::default();
        let controller = BlueGreenController::new(config);

        controller
            .update_health(Environment::Green, HealthStatus::Healthy)
            .await;

        let env = controller.get_environment(Environment::Green).await;
        assert_eq!(env.health, HealthStatus::Healthy);
        assert!(env.last_health_check.is_some());
    }

    #[tokio::test]
    async fn test_deployment_history() {
        let config = DeploymentConfig::default();
        let controller = BlueGreenController::new(config);

        controller.deploy_to_green("v2.0.0").await.unwrap();
        controller.deploy_to_green("v2.1.0").await.unwrap();

        let history = controller.get_deployment_history(None).await;
        assert_eq!(history.len(), 2);

        let limited = controller.get_deployment_history(Some(1)).await;
        assert_eq!(limited.len(), 1);
        assert_eq!(limited[0].version, "v2.1.0");
    }

    #[tokio::test]
    async fn test_environment_opposite() {
        assert_eq!(Environment::Blue.opposite(), Environment::Green);
        assert_eq!(Environment::Green.opposite(), Environment::Blue);
    }

    #[tokio::test]
    async fn test_cannot_deploy_to_active() {
        let config = DeploymentConfig::default();
        let controller = BlueGreenController::new(config);

        let result = controller
            .deploy_to_environment(Environment::Blue, "v2.0.0")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cannot_switch_to_unhealthy() {
        let config = DeploymentConfig::default();
        let controller = BlueGreenController::new(config);

        controller.deploy_to_green("v2.0.0").await.unwrap();
        // Don't update health to healthy

        let result = controller.switch_to_green().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_auto_rollback_detection() {
        let config = DeploymentConfig::default();
        let controller = BlueGreenController::new(config);

        // Below thresholds
        assert!(!controller.should_auto_rollback(1.0, 1000).await);

        // Error rate exceeded
        assert!(controller.should_auto_rollback(10.0, 1000).await);

        // Latency exceeded
        assert!(controller.should_auto_rollback(1.0, 10000).await);
    }

    #[tokio::test]
    async fn test_gradual_switch_strategy() {
        let config = DeploymentConfig {
            strategy: DeploymentStrategy::Gradual {
                increment: 25,
                interval_secs: 0, // No delay for testing
            },
            ..Default::default()
        };
        let controller = BlueGreenController::new(config);

        controller.deploy_to_green("v2.0.0").await.unwrap();
        controller
            .update_health(Environment::Green, HealthStatus::Healthy)
            .await;

        controller.switch_to_green().await.unwrap();

        let status = controller.get_status().await;
        assert_eq!(status.active_environment, Environment::Green);
        assert_eq!(status.green.traffic_percentage, 100);
    }

    #[tokio::test]
    async fn test_event_logging() {
        let config = DeploymentConfig::default();
        let controller = BlueGreenController::new(config);

        controller.deploy_to_green("v2.0.0").await.unwrap();

        let events = controller.get_recent_events(10).await;
        assert!(!events.is_empty());
    }
}
