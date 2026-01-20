//! # Store - fetch_rdf_from_url_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Fetch RDF data from URL using HTTP
    pub(super) async fn fetch_rdf_from_url(
        &self,
        url: &str,
    ) -> FusekiResult<(String, Option<String>)> {
        let response = reqwest::get(url).await.map_err(|e| {
            FusekiError::update_execution(format!("Failed to fetch '{}': {e}", url))
        })?;
        if !response.status().is_success() {
            return Err(FusekiError::update_execution(format!(
                "HTTP error fetching '{}': {}",
                url,
                response.status()
            )));
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let body = response.text().await.map_err(|e| {
            FusekiError::update_execution(format!("Failed to read response body: {e}"))
        })?;
        Ok((body, content_type))
    }
}
