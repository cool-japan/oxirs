//! # RegistryConfig - Trait Implementations
//!
//! This module contains trait implementations for `RegistryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RegistryConfig;
use std::time::Duration;

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            health_check_interval: Duration::from_secs(30),
            service_timeout: Duration::from_secs(10),
            max_retries: 3,
            connection_pool_size: 10,
            auto_discovery: true,
            capability_refresh_interval: Duration::from_secs(300),
        }
    }
}
