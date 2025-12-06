//! # ServiceRegistry - test_methods Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use anyhow::{anyhow, Result};
use std::time::Duration;

impl ServiceRegistry {
    /// Helper method to test a specific result format
    pub(super) async fn test_result_format(
        &self,
        endpoint_url: &str,
        query: &str,
        format: &str,
    ) -> Result<String> {
        let response = self
            .http_client
            .post(endpoint_url)
            .header("Content-Type", "application/sparql-query")
            .header("Accept", format)
            .body(query.to_string())
            .timeout(Duration::from_secs(5))
            .send()
            .await?;
        if response.status().is_success() {
            Ok(response.text().await?)
        } else {
            Err(anyhow!("Format {} not supported", format))
        }
    }
}
