//! Configuration parsing and loading for OxiRS Core.
//!
//! Handles TOML/JSON deserialization, config file loading, and
//! environment variable overrides.

use crate::config_types::{
    ConfigError, ConfigSource, ConfigWatcher, ConfigurationManager, Environment, OxirsConfig,
    PerformanceProfile,
};
use std::path::Path;
use std::sync::{Arc, RwLock};

impl ConfigurationManager {
    /// Create a new configuration manager with default settings.
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(OxirsConfig::default())),
            environment: Environment::Development,
            config_sources: Vec::new(),
            watchers: Vec::new(),
        }
    }

    /// Load configuration from a file (TOML or JSON, detected by extension).
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), ConfigError> {
        let content = std::fs::read_to_string(&path)
            .map_err(|_| ConfigError::FileNotFound(path.as_ref().to_path_buf()))?;

        let config: OxirsConfig = if path.as_ref().extension() == Some(std::ffi::OsStr::new("toml"))
        {
            toml::from_str(&content).map_err(|e| ConfigError::InvalidFormat(e.to_string()))?
        } else {
            serde_json::from_str(&content)?
        };

        self.update_config(config)?;
        self.config_sources.push(ConfigSource::File {
            path: path.as_ref().to_path_buf(),
        });

        Ok(())
    }

    /// Load configuration overrides from environment variables.
    ///
    /// Supported variables:
    /// - `OXIRS_PERFORMANCE_PROFILE` — name of a [`PerformanceProfile`] variant
    /// - `OXIRS_THREAD_COUNT`       — worker thread count (usize)
    pub fn load_from_environment(&mut self) -> Result<(), ConfigError> {
        let mut config = self.get_config();

        if let Ok(profile_str) = std::env::var("OXIRS_PERFORMANCE_PROFILE") {
            if let Ok(profile) =
                serde_json::from_str::<PerformanceProfile>(&format!("\"{}\"", profile_str))
            {
                config.performance.profile = profile;
            }
        }

        if let Ok(threads_str) = std::env::var("OXIRS_THREAD_COUNT") {
            if let Ok(threads) = threads_str.parse::<usize>() {
                config.concurrency.thread_pool.worker_threads = threads;
            }
        }

        self.update_config(config)?;
        self.config_sources.push(ConfigSource::Environment);

        Ok(())
    }

    /// Return a clone of the current configuration.
    pub fn get_config(&self) -> OxirsConfig {
        self.config
            .read()
            .expect("config RwLock should not be poisoned")
            .clone()
    }

    /// Replace the current configuration after validation.
    pub fn update_config(&mut self, new_config: OxirsConfig) -> Result<(), ConfigError> {
        crate::config_validation::validate_config(&new_config)?;
        *self
            .config
            .write()
            .expect("config RwLock should not be poisoned") = new_config;
        Ok(())
    }

    /// Set the active performance profile and apply its default settings.
    pub fn set_performance_profile(
        &mut self,
        profile: PerformanceProfile,
    ) -> Result<(), ConfigError> {
        let mut config = self.get_config();
        config.performance.profile = profile;
        config.performance.custom_settings = profile.get_config();
        self.update_config(config)
    }

    /// Return the current performance profile.
    pub fn get_performance_profile(&self) -> PerformanceProfile {
        self.get_config().performance.profile
    }

    /// Register a callback to be invoked when the configuration changes.
    pub fn add_watcher<F>(&mut self, source: ConfigSource, callback: F)
    where
        F: Fn(&OxirsConfig) + Send + Sync + 'static,
    {
        self.watchers.push(ConfigWatcher {
            source,
            callback: Box::new(callback),
        });
    }

    /// Start asynchronous configuration monitoring (file watchers, env monitors, …).
    pub async fn start_monitoring(&self) -> Result<(), ConfigError> {
        Ok(())
    }
}

impl Default for ConfigurationManager {
    fn default() -> Self {
        Self::new()
    }
}
