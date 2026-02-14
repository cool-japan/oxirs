//! Schema stitching engine for merging multiple GraphQL schemas

use anyhow::{Context, Result};
use chrono;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::config::{RemoteEndpoint, RetryStrategy};
use crate::ast::Value;
use crate::introspection::IntrospectionQuery;
use crate::types::{GraphQLType, ObjectType, Schema};

/// Schema stitching engine for merging multiple GraphQL schemas
pub struct SchemaStitcher {
    /// Local schema
    local_schema: Arc<Schema>,
    /// Remote schemas cache
    remote_schemas: Arc<RwLock<HashMap<String, CachedSchema>>>,
    /// HTTP client for remote introspection
    http_client: reqwest::Client,
}

#[derive(Debug, Clone)]
struct CachedSchema {
    /// The actual schema
    schema: Schema,
    /// Schema version from the service
    #[allow(dead_code)]
    version: Option<String>,
    /// Timestamp when schema was cached
    cached_at: chrono::DateTime<chrono::Utc>,
    /// TTL for the cached schema
    ttl_seconds: u64,
}

impl CachedSchema {
    fn is_expired(&self) -> bool {
        let now = chrono::Utc::now();
        let age_seconds = (now - self.cached_at).num_seconds() as u64;
        age_seconds > self.ttl_seconds
    }

    fn new(schema: Schema, version: Option<String>, ttl_seconds: u64) -> Self {
        Self {
            schema,
            version,
            cached_at: chrono::Utc::now(),
            ttl_seconds,
        }
    }
}

impl SchemaStitcher {
    pub fn new(local_schema: Arc<Schema>) -> Self {
        Self {
            local_schema,
            remote_schemas: Arc::new(RwLock::new(HashMap::new())),
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// Introspect a remote GraphQL endpoint with retry logic
    pub async fn introspect_remote(&self, endpoint: &RemoteEndpoint) -> Result<Schema> {
        // Check cache first
        {
            let cache = self.remote_schemas.read().await;
            if let Some(cached) = cache.get(&endpoint.id) {
                if !cached.is_expired() {
                    tracing::debug!("Using cached schema for endpoint: {}", endpoint.id);
                    return Ok(cached.schema.clone());
                }
            }
        }

        // Check endpoint health before attempting introspection
        if let Some(health_url) = &endpoint.health_check_url {
            self.check_endpoint_health(health_url, endpoint).await?;
        }

        // Perform introspection with retry logic
        let (schema, introspection_result) = self.introspect_with_retry(endpoint).await?;

        // Extract version from introspection result
        let schema_version = self.extract_schema_version(&introspection_result);

        // Cache the schema with version information
        {
            let mut cache = self.remote_schemas.write().await;
            cache.insert(
                endpoint.id.clone(),
                CachedSchema::new(schema.clone(), schema_version, 3600), // 1 hour cache
            );
        }

        tracing::info!(
            "Successfully introspected and cached schema for endpoint: {}",
            endpoint.id
        );
        Ok(schema)
    }

    /// Check endpoint health
    async fn check_endpoint_health(
        &self,
        health_url: &str,
        endpoint: &RemoteEndpoint,
    ) -> Result<()> {
        let response = self
            .http_client
            .get(health_url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
            .context("Health check request failed")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Endpoint {} health check failed with status: {}",
                endpoint.id,
                response.status()
            ));
        }

        tracing::debug!("Health check passed for endpoint: {}", endpoint.id);
        Ok(())
    }

    /// Perform introspection with retry logic
    async fn introspect_with_retry(
        &self,
        endpoint: &RemoteEndpoint,
    ) -> Result<(Schema, serde_json::Value)> {
        let mut last_error = None;

        for attempt in 0..=endpoint.max_retries {
            if attempt > 0 {
                // Apply retry strategy
                let delay = self.calculate_retry_delay(&endpoint.retry_strategy, attempt);
                tracing::warn!(
                    "Retrying introspection for endpoint {} (attempt {}/{})",
                    endpoint.id,
                    attempt + 1,
                    endpoint.max_retries + 1
                );
                tokio::time::sleep(delay).await;
            }

            match self.perform_introspection(endpoint).await {
                Ok((schema, introspection_result)) => {
                    if attempt > 0 {
                        tracing::info!(
                            "Introspection succeeded for endpoint {} after {} retries",
                            endpoint.id,
                            attempt
                        );
                    }
                    return Ok((schema, introspection_result));
                }
                Err(e) => {
                    last_error = Some(e);
                    tracing::warn!(
                        "Introspection attempt {} failed for endpoint {}: {}",
                        attempt + 1,
                        endpoint.id,
                        last_error
                            .as_ref()
                            .expect("last_error should be set after failed attempt")
                    );
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!(
                "All introspection attempts failed for endpoint: {}",
                endpoint.id
            )
        }))
    }

    /// Calculate retry delay based on strategy
    fn calculate_retry_delay(&self, strategy: &RetryStrategy, attempt: u32) -> std::time::Duration {
        match strategy {
            RetryStrategy::None => std::time::Duration::from_millis(0),
            RetryStrategy::FixedDelay { delay_ms } => std::time::Duration::from_millis(*delay_ms),
            RetryStrategy::ExponentialBackoff {
                initial_delay_ms,
                max_delay_ms,
                multiplier,
            } => {
                let delay = (*initial_delay_ms as f64) * multiplier.powi(attempt as i32);
                let delay = delay.min(*max_delay_ms as f64);

                // Add jitter (±25%)
                let jitter = fastrand::f64() * 0.5 - 0.25; // -25% to +25%
                let final_delay = delay * (1.0 + jitter);

                std::time::Duration::from_millis(final_delay.max(0.0) as u64)
            }
        }
    }

    /// Perform the actual introspection request
    async fn perform_introspection(
        &self,
        endpoint: &RemoteEndpoint,
    ) -> Result<(Schema, serde_json::Value)> {
        // Build introspection query
        let introspection_query = IntrospectionQuery::full_query();

        let mut request = self
            .http_client
            .post(&endpoint.url)
            .json(&serde_json::json!({
                "query": introspection_query,
                "variables": {}
            }));

        // Add authentication if provided
        if let Some(auth) = &endpoint.auth_header {
            request = request.header("Authorization", auth);
        }

        // Execute request
        let response = request
            .timeout(std::time::Duration::from_secs(endpoint.timeout_secs))
            .send()
            .await
            .context("Failed to send introspection request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!(
                "Introspection request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse introspection response as JSON")?;

        // Check for GraphQL errors
        if let Some(errors) = response_json.get("errors") {
            return Err(anyhow::anyhow!(
                "GraphQL introspection errors: {}",
                serde_json::to_string_pretty(errors)?
            ));
        }

        let data = response_json
            .get("data")
            .ok_or_else(|| anyhow::anyhow!("No data field in introspection response"))?;

        // Convert introspection result to Schema
        let schema = self.introspection_to_schema(data)?;

        Ok((schema, response_json))
    }

    /// Extract schema version from introspection result
    fn extract_schema_version(&self, introspection_result: &serde_json::Value) -> Option<String> {
        // Try to extract version from schema description
        if let Some(schema_obj) = introspection_result
            .get("data")?
            .get("__schema")?
            .as_object()
        {
            if let Some(description) = schema_obj.get("description").and_then(|d| d.as_str()) {
                return self.extract_version_from_description(description);
            }
        }

        None
    }

    /// Extract version string from description text
    fn extract_version_from_description(&self, description: &str) -> Option<String> {
        // Common version patterns
        let version_pattern_strs = vec![
            r"version\s*:?\s*([0-9]+\.[0-9]+\.[0-9]+)",
            r"v([0-9]+\.[0-9]+\.[0-9]+)",
            r"([0-9]+\.[0-9]+\.[0-9]+)",
        ];

        for pattern_str in &version_pattern_strs {
            if let Ok(pattern) = regex::Regex::new(pattern_str) {
                if let Some(captures) = pattern.captures(description) {
                    if let Some(version_match) = captures.get(1) {
                        return Some(version_match.as_str().to_string());
                    }
                }
            }
        }

        None
    }

    /// Convert introspection result to Schema
    fn introspection_to_schema(&self, data: &serde_json::Value) -> Result<Schema> {
        // This is a simplified implementation
        // In a real scenario, you'd parse the full introspection result
        let mut schema = Schema::new();

        if let Some(schema_data) = data.get("__schema") {
            if let Some(types) = schema_data.get("types").and_then(|t| t.as_array()) {
                for type_def in types {
                    if let Some(type_name) = type_def.get("name").and_then(|n| n.as_str()) {
                        // Skip built-in types
                        if type_name.starts_with("__") {
                            continue;
                        }

                        // Create a basic type (this should be expanded)
                        let gql_type = GraphQLType::Object(ObjectType {
                            name: type_name.to_string(),
                            description: type_def
                                .get("description")
                                .and_then(|d| d.as_str())
                                .map(|s| s.to_string()),
                            fields: HashMap::new(),
                            interfaces: Vec::new(),
                        });

                        schema.add_type(gql_type);
                    }
                }
            }
        }

        Ok(schema)
    }

    /// Merge multiple schemas into one
    pub async fn merge_schemas(&self, endpoints: &[RemoteEndpoint]) -> Result<Schema> {
        let mut merged_schema = (*self.local_schema).clone();

        for endpoint in endpoints {
            let remote_schema = self.introspect_remote(endpoint).await?;
            self.merge_schema_into(&mut merged_schema, &remote_schema, endpoint)?;
        }

        Ok(merged_schema)
    }

    /// Merge a remote schema into the local schema
    pub fn merge_schema_into(
        &self,
        target: &mut Schema,
        source: &Schema,
        endpoint: &RemoteEndpoint,
    ) -> Result<()> {
        let namespace = endpoint.namespace.as_deref().unwrap_or(&endpoint.id);

        for (type_name, type_def) in &source.types {
            let prefixed_name = if type_name.starts_with("__") {
                // Don't prefix introspection types
                type_name.clone()
            } else {
                format!("{namespace}_{type_name}")
            };

            // Check for conflicts
            if target.get_type(&prefixed_name).is_some() {
                tracing::warn!(
                    "Type conflict detected: {} from endpoint {} conflicts with existing type",
                    prefixed_name,
                    endpoint.id
                );
                // Apply conflict resolution strategy here
                continue;
            }

            target.add_type(type_def.clone());
        }

        Ok(())
    }

    /// Parse a default value string into a GraphQL Value
    pub fn parse_default_value(&self, default_str: &str) -> Result<Value> {
        // Simple parser for common default values
        let trimmed = default_str.trim();

        if trimmed == "null" {
            return Ok(Value::NullValue);
        }

        if trimmed == "true" {
            return Ok(Value::BooleanValue(true));
        }

        if trimmed == "false" {
            return Ok(Value::BooleanValue(false));
        }

        // Try parsing as string (quoted)
        if trimmed.starts_with('"') && trimmed.ends_with('"') {
            let inner = &trimmed[1..trimmed.len() - 1];
            return Ok(Value::StringValue(inner.to_string()));
        }

        // Try parsing as integer
        if let Ok(int_val) = trimmed.parse::<i64>() {
            return Ok(Value::IntValue(int_val));
        }

        // Try parsing as float
        if let Ok(float_val) = trimmed.parse::<f64>() {
            return Ok(Value::FloatValue(float_val));
        }

        // Default to string if can't parse
        Ok(Value::StringValue(trimmed.to_string()))
    }
}
