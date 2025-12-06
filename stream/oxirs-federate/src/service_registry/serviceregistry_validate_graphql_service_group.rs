//! # ServiceRegistry - validate_graphql_service_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::GraphQLService;
use anyhow::{anyhow, Result};

impl ServiceRegistry {
    /// Validate GraphQL service
    pub(super) async fn validate_graphql_service(&self, service: &GraphQLService) -> Result<()> {
        if service.url.scheme() != "http" && service.url.scheme() != "https" {
            return Err(anyhow!("Invalid URL scheme: {}", service.url.scheme()));
        }
        let introspection_query = r#"
            query {
                __schema {
                    types {
                        name
                    }
                }
            }
        "#;
        let response = self
            .http_client
            .post(service.url.clone())
            .json(&serde_json::json!({ "query" : introspection_query }))
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(anyhow!("Service not accessible: {}", response.status()));
        }
        Ok(())
    }
}
