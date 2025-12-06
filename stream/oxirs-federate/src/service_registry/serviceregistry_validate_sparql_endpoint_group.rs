//! # ServiceRegistry - validate_sparql_endpoint_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::SparqlEndpoint;
use anyhow::{anyhow, Result};

impl ServiceRegistry {
    /// Validate SPARQL endpoint
    pub(super) async fn validate_sparql_endpoint(&self, endpoint: &SparqlEndpoint) -> Result<()> {
        if endpoint.url.scheme() != "http" && endpoint.url.scheme() != "https" {
            return Err(anyhow!("Invalid URL scheme: {}", endpoint.url.scheme()));
        }
        let response = self.http_client.get(endpoint.url.clone()).send().await?;
        if !response.status().is_success() {
            return Err(anyhow!("Endpoint not accessible: {}", response.status()));
        }
        Ok(())
    }
}
