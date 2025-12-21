//! # GraphQLCapabilities - Trait Implementations
//!
//! This module contains trait implementations for `GraphQLCapabilities`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::GraphQLCapabilities;

impl Default for GraphQLCapabilities {
    fn default() -> Self {
        Self {
            graphql_version: "June 2018".to_string(),
            supports_subscriptions: false,
            max_query_depth: Some(10),
            max_query_complexity: Some(1000),
            introspection_enabled: true,
            federation_version: Some("v1.0".to_string()),
        }
    }
}
