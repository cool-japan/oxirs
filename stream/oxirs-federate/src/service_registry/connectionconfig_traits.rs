//! # ConnectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `ConnectionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ConnectionConfig;
use std::time::Duration;

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_connections: 10,
            keep_alive: true,
            compression: true,
        }
    }
}
