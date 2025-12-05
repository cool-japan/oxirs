//! Configuration Hot-Reload Support
//!
//! Provides automatic configuration file watching and hot-reload capabilities
//! for runtime-changeable server settings.

use crate::config::ServerConfig;
use crate::error::{FusekiError, FusekiResult};
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use validator::Validate;

/// Configuration reload manager
pub struct ConfigReloadManager {
    /// Path to configuration file
    config_path: PathBuf,
    /// Current configuration (shared with server)
    current_config: Arc<RwLock<ServerConfig>>,
    /// File watcher
    _watcher: Option<RecommendedWatcher>,
}

impl ConfigReloadManager {
    /// Create a new configuration reload manager
    pub fn new(
        config_path: PathBuf,
        current_config: Arc<RwLock<ServerConfig>>,
    ) -> FusekiResult<Self> {
        info!(
            "Initializing configuration hot-reload manager for {:?}",
            config_path
        );

        Ok(Self {
            config_path,
            current_config,
            _watcher: None,
        })
    }

    /// Start watching the configuration file for changes
    pub fn start_watching(&mut self) -> FusekiResult<()> {
        let config_path = self.config_path.clone();
        let current_config = self.current_config.clone();

        // Create file watcher
        let (tx, rx) = std::sync::mpsc::channel();

        let mut watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                if let Err(e) = tx.send(res) {
                    error!("Failed to send file watch event: {}", e);
                }
            },
            Config::default(),
        )
        .map_err(|e| FusekiError::internal(format!("Failed to create file watcher: {}", e)))?;

        // Watch the config file
        watcher
            .watch(&config_path, RecursiveMode::NonRecursive)
            .map_err(|e| FusekiError::internal(format!("Failed to watch config file: {}", e)))?;

        info!("Started watching configuration file: {:?}", config_path);

        // Spawn background thread to handle file changes
        let config_path_clone = config_path.clone();
        std::thread::spawn(move || {
            for res in rx {
                match res {
                    Ok(event) => {
                        if matches!(event.kind, notify::EventKind::Modify(_)) {
                            info!("Configuration file changed, reloading: {:?}", event.paths);

                            // Attempt to reload configuration
                            match Self::reload_config_file(&config_path_clone, &current_config) {
                                Ok(changes) => {
                                    if changes.is_empty() {
                                        info!("Configuration reloaded successfully (no changes detected)");
                                    } else {
                                        info!("Configuration reloaded successfully. {} change(s) applied:", changes.len());
                                        for change in changes {
                                            info!("  - {}", change);
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("Failed to reload configuration: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("File watch error: {}", e);
                    }
                }
            }
        });

        self._watcher = Some(watcher);
        Ok(())
    }

    /// Reload configuration from file
    fn reload_config_file(
        config_path: &PathBuf,
        current_config: &Arc<RwLock<ServerConfig>>,
    ) -> FusekiResult<Vec<String>> {
        debug!("Reading configuration file: {:?}", config_path);

        // Read and parse new configuration
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| FusekiError::internal(format!("Failed to read config file: {}", e)))?;

        let new_config: ServerConfig = toml::from_str(&config_str)
            .map_err(|e| FusekiError::internal(format!("Failed to parse config: {}", e)))?;

        // Validate new configuration
        if let Err(e) = new_config.validate() {
            return Err(FusekiError::internal(format!(
                "Invalid configuration: {}",
                e
            )));
        }

        // Detect changes and apply runtime-changeable settings
        let changes = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { Self::apply_runtime_changes(current_config, new_config).await })
        });

        changes
    }

    /// Apply runtime-changeable configuration settings
    async fn apply_runtime_changes(
        current_config: &Arc<RwLock<ServerConfig>>,
        new_config: ServerConfig,
    ) -> FusekiResult<Vec<String>> {
        let mut changes = Vec::new();
        let mut config = current_config.write().await;

        // Check and apply CORS changes
        if config.security.cors.enabled != new_config.security.cors.enabled {
            config.security.cors = new_config.security.cors.clone();
            changes.push("CORS settings updated".to_string());
        }

        // Check and apply rate limiting changes
        if config
            .performance
            .rate_limiting
            .as_ref()
            .map(|r| r.requests_per_minute)
            != new_config
                .performance
                .rate_limiting
                .as_ref()
                .map(|r| r.requests_per_minute)
        {
            config.performance.rate_limiting = new_config.performance.rate_limiting.clone();
            changes.push("Rate limiting settings updated".to_string());
        }

        // Check and apply monitoring changes
        if config.monitoring.metrics.enabled != new_config.monitoring.metrics.enabled {
            config.monitoring = new_config.monitoring.clone();
            changes.push("Monitoring settings updated".to_string());
        }

        // Check and apply cache settings
        if config.performance.caching.enabled != new_config.performance.caching.enabled
            || config.performance.caching.max_size != new_config.performance.caching.max_size
        {
            config.performance.caching = new_config.performance.caching.clone();
            changes.push("Cache settings updated".to_string());
        }

        // Note: Some settings require server restart
        let restart_required = Self::check_restart_required(&config, &new_config);
        if !restart_required.is_empty() {
            warn!("The following changes require a server restart:");
            for item in &restart_required {
                warn!("  - {}", item);
                changes.push(format!("{} (restart required)", item));
            }
        }

        // Update the full configuration (for reference, even if not all changes are applied)
        *config = new_config;

        Ok(changes)
    }

    /// Check which settings require a server restart
    fn check_restart_required(old_config: &ServerConfig, new_config: &ServerConfig) -> Vec<String> {
        let mut restart_items = Vec::new();

        if old_config.server.port != new_config.server.port {
            restart_items.push(format!(
                "Server port changed: {} -> {}",
                old_config.server.port, new_config.server.port
            ));
        }

        if old_config.server.host != new_config.server.host {
            restart_items.push(format!(
                "Server host changed: {} -> {}",
                old_config.server.host, new_config.server.host
            ));
        }

        if old_config.server.tls.is_some() != new_config.server.tls.is_some() {
            restart_items.push("TLS configuration changed".to_string());
        }

        restart_items
    }

    /// Manual reload trigger (for API endpoint)
    pub async fn trigger_reload(&self) -> FusekiResult<Vec<String>> {
        info!("Manual configuration reload triggered");
        Self::reload_config_file(&self.config_path, &self.current_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_reload_manager_creation() {
        let config_path = PathBuf::from("/tmp/test-config.toml");
        let config = ServerConfig::default();
        let config_arc = Arc::new(RwLock::new(config));

        let manager = ConfigReloadManager::new(config_path.clone(), config_arc);
        assert!(manager.is_ok());

        let mgr = manager.unwrap();
        assert_eq!(mgr.config_path, config_path);
    }
}
