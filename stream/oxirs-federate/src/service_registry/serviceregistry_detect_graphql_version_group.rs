//! # ServiceRegistry - detect_graphql_version_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use anyhow::Result;
use std::time::Duration;

impl ServiceRegistry {
    /// Detect GraphQL specification version from schema features
    pub(super) async fn detect_graphql_version(&self, schema: &str) -> String {
        if schema.contains("@oneOf") || schema.contains("__DirectiveLocation.ARGUMENT_DEFINITION") {
            return "October 2021".to_string();
        }
        if schema.contains("@specifiedBy") || schema.contains("__DirectiveLocation.SCALAR") {
            return "June 2020".to_string();
        }
        if schema.contains("interfaces") && schema.contains("__Type") {
            return "June 2018".to_string();
        }
        "October 2015".to_string()
    }
    /// Detect subscription support from schema
    pub(super) async fn detect_subscription_support(&self, schema: &str) -> bool {
        schema.contains("subscriptionType") && !schema.contains("\"subscriptionType\": null")
    }
    /// Detect federation version from schema directives
    pub(super) async fn detect_federation_version(&self, schema: &str) -> Option<String> {
        if schema.contains("@federation__") || schema.contains("_service") {
            if schema.contains("@shareable") || schema.contains("@inaccessible") {
                return Some("v2.0".to_string());
            } else if schema.contains("@key") || schema.contains("@external") {
                return Some("v1.0".to_string());
            }
        }
        None
    }
    /// Estimate maximum query depth by testing increasingly deep queries
    pub(super) async fn estimate_max_query_depth(&self, endpoint_url: &str) -> Result<u32> {
        for depth in 1u32..=20u32 {
            let mut query = String::from("query { __schema { ");
            for _ in 0..depth {
                query.push_str("types { ");
            }
            query.push_str("name");
            for _ in 0..depth {
                query.push_str(" }");
            }
            query.push_str(" } }");
            let request_body = serde_json::json!({ "query" : query });
            let response = self
                .http_client
                .post(endpoint_url)
                .header("Content-Type", "application/json")
                .json(&request_body)
                .timeout(Duration::from_secs(5))
                .send()
                .await;
            match response {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        return Ok(depth.saturating_sub(1));
                    }
                    let text = resp.text().await.unwrap_or_default();
                    if text.contains("error") && text.contains("depth") {
                        return Ok(depth.saturating_sub(1));
                    }
                }
                Err(_) => return Ok(depth.saturating_sub(1)),
            }
        }
        Ok(20)
    }
    /// Estimate maximum query complexity by testing increasingly complex queries
    pub(super) async fn estimate_max_query_complexity(&self, endpoint_url: &str) -> Result<u32> {
        for complexity in &[10, 50, 100, 500, 1000, 5000] {
            let mut query = String::from("query { __schema { types { name kind description ");
            for i in 0..*complexity / 10 {
                query.push_str(&format!("field{i}: name "));
            }
            query.push_str("} } }");
            let request_body = serde_json::json!({ "query" : query });
            let response = self
                .http_client
                .post(endpoint_url)
                .header("Content-Type", "application/json")
                .json(&request_body)
                .timeout(Duration::from_secs(5))
                .send()
                .await;
            match response {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        return Ok(*complexity);
                    }
                    let text = resp.text().await.unwrap_or_default();
                    if text.contains("error")
                        && (text.contains("complexity") || text.contains("too complex"))
                    {
                        return Ok(*complexity);
                    }
                }
                Err(_) => return Ok(*complexity),
            }
        }
        Ok(5000)
    }
}
