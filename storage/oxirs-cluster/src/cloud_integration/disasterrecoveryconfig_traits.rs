//! # DisasterRecoveryConfig - Trait Implementations
//!
//! This module contains trait implementations for `DisasterRecoveryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CloudProvider, DisasterRecoveryConfig};

impl Default for DisasterRecoveryConfig {
    fn default() -> Self {
        Self {
            primary_provider: CloudProvider::AWS,
            secondary_providers: vec![CloudProvider::GCP, CloudProvider::Azure],
            rto_seconds: 300,
            rpo_seconds: 60,
            auto_failover_enabled: true,
            health_check_interval_secs: 30,
            failover_threshold: 3,
            continuous_replication: true,
            replication_batch_size: 100,
        }
    }
}
