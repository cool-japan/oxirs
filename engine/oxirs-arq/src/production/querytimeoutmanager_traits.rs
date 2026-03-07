//! # QueryTimeoutManager - Trait Implementations
//!
//! This module contains trait implementations for `QueryTimeoutManager`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::RwLock;
use std::time::Duration;

use super::types::{QueryTimeoutManager, TimeoutAction};

impl Default for QueryTimeoutManager {
    fn default() -> Self {
        Self {
            soft_timeout: RwLock::new(Duration::from_secs(30)),
            hard_timeout: RwLock::new(Duration::from_secs(300)),
            warning_thresholds: RwLock::new(vec![0.5, 0.75, 0.9]),
            active_queries: RwLock::new(HashMap::new()),
            next_query_id: AtomicU64::new(1),
            timeout_action: RwLock::new(TimeoutAction::Cancel),
        }
    }
}
