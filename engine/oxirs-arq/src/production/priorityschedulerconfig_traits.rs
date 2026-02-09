//! # PrioritySchedulerConfig - Trait Implementations
//!
//! This module contains trait implementations for `PrioritySchedulerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

use super::types::{PrioritySchedulerConfig, QueryPriority};

impl Default for PrioritySchedulerConfig {
    fn default() -> Self {
        let mut max_concurrent = HashMap::new();
        max_concurrent.insert(QueryPriority::Critical, 10);
        max_concurrent.insert(QueryPriority::High, 8);
        max_concurrent.insert(QueryPriority::Normal, 5);
        max_concurrent.insert(QueryPriority::Low, 3);
        max_concurrent.insert(QueryPriority::Batch, 2);
        Self {
            max_per_priority: 100,
            max_total_queued: 500,
            max_concurrent_per_priority: max_concurrent,
            enable_aging: true,
            aging_threshold: Duration::from_secs(30),
        }
    }
}
