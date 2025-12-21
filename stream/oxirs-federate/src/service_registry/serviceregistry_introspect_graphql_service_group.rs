//! # ServiceRegistry - introspect_graphql_service_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::{GraphQLCapabilities, GraphQLService};
use anyhow::Result;
use tracing::{info, warn};

impl ServiceRegistry {
    /// Introspect GraphQL service
    pub(super) async fn introspect_graphql_service(
        &self,
        service: &GraphQLService,
    ) -> Result<(GraphQLCapabilities, Option<String>)> {
        info!("Introspecting GraphQL service: {}", service.url);
        let mut capabilities = GraphQLCapabilities::default();
        let endpoint_url = service.url.to_string();
        let schema = match self.fetch_graphql_schema(&endpoint_url).await {
            Ok(schema) => {
                capabilities.introspection_enabled = true;
                Some(schema)
            }
            Err(_) => {
                capabilities.introspection_enabled = false;
                warn!("GraphQL introspection disabled for {}", service.url);
                None
            }
        };
        if let Some(ref schema_content) = schema {
            capabilities.graphql_version = self.detect_graphql_version(schema_content).await;
            capabilities.supports_subscriptions =
                self.detect_subscription_support(schema_content).await;
            capabilities.federation_version = self.detect_federation_version(schema_content).await;
            if let Ok(depth) = self.estimate_max_query_depth(&endpoint_url).await {
                capabilities.max_query_depth = Some(depth);
            }
            if let Ok(complexity) = self.estimate_max_query_complexity(&endpoint_url).await {
                capabilities.max_query_complexity = Some(complexity);
            }
        }
        info!(
            "GraphQL introspection completed for {}: {:?}",
            service.url, capabilities
        );
        Ok((capabilities, schema))
    }
}
