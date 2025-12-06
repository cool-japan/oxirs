//! # ServiceRegistry - test_methods Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use anyhow::{anyhow, Result};
use std::time::Duration;

impl ServiceRegistry {
    /// Helper method to test a SPARQL query
    pub(super) async fn test_sparql_query(
        &self,
        endpoint_url: &str,
        query: &str,
    ) -> Result<String> {
        let response = self
            .http_client
            .post(endpoint_url)
            .header("Content-Type", "application/sparql-query")
            .header("Accept", "application/sparql-results+json")
            .body(query.to_string())
            .timeout(Duration::from_secs(5))
            .send()
            .await?;
        if response.status().is_success() {
            Ok(response.text().await?)
        } else {
            Err(anyhow!("Query failed: {}", response.status()))
        }
    }
}
