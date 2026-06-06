//! Configuration validation for OxiRS Core.
//!
//! Provides field constraint checking, cross-field validation, and
//! semantic validation of configuration values.

use crate::config_types::{ConfigError, OxirsConfig, PerformanceProfile};

/// Validate a complete [`OxirsConfig`].
///
/// Returns `Ok(())` if the configuration is valid, or a descriptive
/// [`ConfigError::ValidationError`] otherwise.
pub fn validate_config(config: &OxirsConfig) -> Result<(), ConfigError> {
    validate_concurrency(config)?;
    validate_memory(config)?;
    validate_performance_profile(config)?;
    Ok(())
}

// --- Section validators ---

fn validate_concurrency(config: &OxirsConfig) -> Result<(), ConfigError> {
    if config.concurrency.thread_pool.worker_threads == 0 {
        return Err(ConfigError::ValidationError(
            "Worker thread count cannot be zero".to_string(),
        ));
    }
    Ok(())
}

fn validate_memory(config: &OxirsConfig) -> Result<(), ConfigError> {
    if config.memory.arena.initial_size > config.memory.arena.max_size {
        return Err(ConfigError::ValidationError(
            "Initial arena size cannot exceed maximum".to_string(),
        ));
    }
    Ok(())
}

fn validate_performance_profile(config: &OxirsConfig) -> Result<(), ConfigError> {
    match config.performance.profile {
        PerformanceProfile::RealTime if config.concurrency.thread_pool.worker_threads > 4 => {
            return Err(ConfigError::ValidationError(
                "Real-time profile should use fewer threads".to_string(),
            ));
        }
        PerformanceProfile::EdgeComputing if config.memory.arena.max_size > 128 * 1024 * 1024 => {
            return Err(ConfigError::ValidationError(
                "Edge computing profile should use less memory".to_string(),
            ));
        }
        _ => {}
    }
    Ok(())
}
