//! Schema stitching engine for merging multiple GraphQL schemas

use anyhow::{anyhow, Context, Result};
use chrono;
use std::collections::{HashMap, HashSet};
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

// ---------------------------------------------------------------------------
// Merge-directive-aware schema stitching
// ---------------------------------------------------------------------------

/// Conflict resolution strategy for duplicate type names.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ConflictResolution {
    /// Keep the type from the first schema that defines it and discard others.
    KeepFirst,
    /// Keep the type from the last schema that defines it.
    KeepLast,
    /// Attempt to merge fields from all schemas that define the type.
    #[default]
    MergeFields,
    /// Raise an error on any conflict.
    Error,
}

/// A fragment of a GraphQL schema supplied to the stitcher.
///
/// Each `SchemaFragment` represents one source schema.  Directives and
/// conflict resolution hints can be attached per-type using the `@merge`
/// directive syntax.
#[derive(Debug, Clone)]
pub struct SchemaFragment {
    /// Human-readable name for this fragment (e.g. the service name).
    pub name: String,
    /// Type definitions in this fragment.
    pub types: HashMap<String, StitchTypeDefinition>,
    /// Per-type conflict resolution hint.
    pub merge_directives: HashMap<String, MergeDirective>,
}

impl SchemaFragment {
    /// Create an empty fragment with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            types: HashMap::new(),
            merge_directives: HashMap::new(),
        }
    }

    /// Add a type definition to the fragment.
    pub fn with_type(mut self, def: StitchTypeDefinition) -> Self {
        self.types.insert(def.name.clone(), def);
        self
    }

    /// Attach a `@merge` directive to a type in this fragment.
    pub fn with_merge_directive(
        mut self,
        type_name: impl Into<String>,
        dir: MergeDirective,
    ) -> Self {
        self.merge_directives.insert(type_name.into(), dir);
        self
    }
}

/// A field definition used within schema stitching.
#[derive(Debug, Clone)]
pub struct StitchFieldDef {
    /// Field name.
    pub name: String,
    /// Declared type string (e.g. `"String!"`, `"[Int]"`).
    pub field_type: String,
    /// Optional description.
    pub description: Option<String>,
    /// Source fragment name.
    pub source: String,
}

impl StitchFieldDef {
    /// Create a new field definition.
    pub fn new(
        name: impl Into<String>,
        field_type: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            field_type: field_type.into(),
            description: None,
            source: source.into(),
        }
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// A type definition used within schema stitching.
#[derive(Debug, Clone)]
pub struct StitchTypeDefinition {
    /// Type name.
    pub name: String,
    /// Whether this is a root type (`Query`, `Mutation`, `Subscription`).
    pub is_root: bool,
    /// Field definitions for this type.
    pub fields: Vec<StitchFieldDef>,
    /// Optional description.
    pub description: Option<String>,
}

impl StitchTypeDefinition {
    /// Create a new object type definition.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            is_root: false,
            fields: Vec::new(),
            description: None,
        }
    }

    /// Mark this type as a root type.
    pub fn as_root(mut self) -> Self {
        self.is_root = true;
        self
    }

    /// Add a field to this type.
    pub fn with_field(mut self, field: StitchFieldDef) -> Self {
        self.fields.push(field);
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// `@merge` directive attached to a type, instructing the stitcher how to
/// combine it with identically-named types from other fragments.
#[derive(Debug, Clone)]
pub struct MergeDirective {
    /// Key field used for entity stitching (mirrors `@key(fields: "...")`).
    pub key_field: Option<String>,
    /// Conflict resolution override for this specific type.
    pub resolution: ConflictResolution,
    /// Whether fields from duplicate types should be deduplicated by name.
    pub deduplicate_fields: bool,
}

impl Default for MergeDirective {
    fn default() -> Self {
        Self {
            key_field: None,
            resolution: ConflictResolution::MergeFields,
            deduplicate_fields: true,
        }
    }
}

/// The merged result of stitching multiple `SchemaFragment` objects.
#[derive(Debug, Clone)]
pub struct StitchedSchema {
    /// Merged type definitions, keyed by type name.
    pub types: HashMap<String, StitchTypeDefinition>,
    /// Conflicts detected during stitching.
    pub conflicts: Vec<StitchConflict>,
    /// Source fragments that contributed to this schema.
    pub sources: Vec<String>,
}

impl StitchedSchema {
    /// Returns `true` if any conflicts were detected.
    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
    }

    /// Returns total number of types in the merged schema.
    pub fn type_count(&self) -> usize {
        self.types.len()
    }

    /// Returns total number of fields across all types.
    pub fn total_field_count(&self) -> usize {
        self.types.values().map(|t| t.fields.len()).sum()
    }

    /// Get a type by name.
    pub fn get_type(&self, name: &str) -> Option<&StitchTypeDefinition> {
        self.types.get(name)
    }
}

/// A conflict detected when two fragments define the same type differently.
#[derive(Debug, Clone)]
pub struct StitchConflict {
    /// The conflicting type name.
    pub type_name: String,
    /// The fragment that introduced the conflict.
    pub conflicting_source: String,
    /// Human-readable description of the conflict.
    pub description: String,
}

/// Schema stitcher that merges multiple `SchemaFragment` objects using
/// `@merge` directive hints for conflict resolution.
///
/// # Conflict resolution
///
/// When two fragments define the same type name:
/// - `ConflictResolution::KeepFirst`   — the first fragment wins.
/// - `ConflictResolution::KeepLast`    — the last fragment wins.
/// - `ConflictResolution::MergeFields` — fields from all fragments are merged;
///   if `deduplicate_fields` is set in the `@merge` directive, duplicate field
///   names are collapsed (keeping the first definition).
/// - `ConflictResolution::Error`       — stitching fails with an error.
pub struct MergeDirectiveSchemaStitcher {
    /// Default conflict resolution when no `@merge` directive is present.
    pub default_resolution: ConflictResolution,
}

impl MergeDirectiveSchemaStitcher {
    /// Create a new stitcher with default (`MergeFields`) resolution.
    pub fn new() -> Self {
        Self {
            default_resolution: ConflictResolution::default(),
        }
    }

    /// Create a stitcher with a custom default resolution strategy.
    pub fn with_resolution(resolution: ConflictResolution) -> Self {
        Self {
            default_resolution: resolution,
        }
    }

    /// Stitch a list of `SchemaFragment` objects into a single `StitchedSchema`.
    pub fn stitch(&self, fragments: &[SchemaFragment]) -> Result<StitchedSchema> {
        let mut merged_types: HashMap<String, StitchTypeDefinition> = HashMap::new();
        let mut conflicts: Vec<StitchConflict> = Vec::new();
        let sources: Vec<String> = fragments.iter().map(|f| f.name.clone()).collect();

        for fragment in fragments {
            for (type_name, type_def) in &fragment.types {
                if let Some(existing) = merged_types.get_mut(type_name) {
                    // Conflict — determine resolution.
                    // Use per-type directive if it exists, otherwise fall back to global default.
                    let directive_opt = fragment.merge_directives.get(type_name).cloned();
                    let (resolution, directive) = if let Some(dir) = directive_opt {
                        let res = dir.resolution.clone();
                        (res, dir)
                    } else {
                        (self.default_resolution.clone(), MergeDirective::default())
                    };

                    match resolution {
                        ConflictResolution::KeepFirst => {
                            // Do nothing — keep the existing definition.
                            conflicts.push(StitchConflict {
                                type_name: type_name.clone(),
                                conflicting_source: fragment.name.clone(),
                                description: format!(
                                    "Type '{type_name}' from '{}' ignored (KeepFirst)",
                                    fragment.name
                                ),
                            });
                        }
                        ConflictResolution::KeepLast => {
                            *existing = type_def.clone();
                            conflicts.push(StitchConflict {
                                type_name: type_name.clone(),
                                conflicting_source: fragment.name.clone(),
                                description: format!(
                                    "Type '{type_name}' replaced by '{}' (KeepLast)",
                                    fragment.name
                                ),
                            });
                        }
                        ConflictResolution::MergeFields => {
                            let dedup = directive.deduplicate_fields;
                            let mut seen: HashSet<String> =
                                existing.fields.iter().map(|f| f.name.clone()).collect();
                            for field in &type_def.fields {
                                if dedup && seen.contains(&field.name) {
                                    // Skip duplicate field.
                                    continue;
                                }
                                seen.insert(field.name.clone());
                                existing.fields.push(field.clone());
                            }
                            // Record the merge event as a conflict entry.
                            conflicts.push(StitchConflict {
                                type_name: type_name.clone(),
                                conflicting_source: fragment.name.clone(),
                                description: format!(
                                    "Type '{type_name}' from '{}' merged (MergeFields)",
                                    fragment.name
                                ),
                            });
                        }
                        ConflictResolution::Error => {
                            return Err(anyhow!(
                                "Conflict: type '{type_name}' defined in both '{}' and '{}'",
                                existing
                                    .fields
                                    .first()
                                    .map(|f| f.source.as_str())
                                    .unwrap_or("unknown"),
                                fragment.name
                            ));
                        }
                    }
                } else {
                    merged_types.insert(type_name.clone(), type_def.clone());
                }
            }
        }

        Ok(StitchedSchema {
            types: merged_types,
            conflicts,
            sources,
        })
    }
}

impl Default for MergeDirectiveSchemaStitcher {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests for MergeDirectiveSchemaStitcher
// ---------------------------------------------------------------------------

#[cfg(test)]
mod merge_tests {
    use super::*;

    fn user_fragment() -> SchemaFragment {
        SchemaFragment::new("users")
            .with_type(
                StitchTypeDefinition::new("User")
                    .with_field(StitchFieldDef::new("id", "ID!", "users"))
                    .with_field(StitchFieldDef::new("name", "String!", "users")),
            )
            .with_type(
                StitchTypeDefinition::new("Query")
                    .as_root()
                    .with_field(StitchFieldDef::new("user", "User", "users")),
            )
    }

    fn product_fragment() -> SchemaFragment {
        SchemaFragment::new("products")
            .with_type(
                StitchTypeDefinition::new("Product")
                    .with_field(StitchFieldDef::new("sku", "String!", "products"))
                    .with_field(StitchFieldDef::new("price", "Float!", "products")),
            )
            .with_type(
                StitchTypeDefinition::new("Query")
                    .as_root()
                    .with_field(StitchFieldDef::new("products", "[Product!]!", "products")),
            )
    }

    #[test]
    fn test_stitch_single_fragment() {
        let stitcher = MergeDirectiveSchemaStitcher::new();
        let result = stitcher.stitch(&[user_fragment()]).expect("should succeed");
        assert_eq!(result.type_count(), 2); // User + Query
        assert!(!result.has_conflicts());
        assert_eq!(result.sources, vec!["users"]);
    }

    #[test]
    fn test_stitch_merges_types_from_two_fragments() {
        let stitcher = MergeDirectiveSchemaStitcher::new();
        let result = stitcher
            .stitch(&[user_fragment(), product_fragment()])
            .expect("should succeed");
        // User, Product, Query (merged)
        assert_eq!(result.type_count(), 3);
        let query = result.get_type("Query").expect("should succeed");
        // user + products fields
        assert_eq!(query.fields.len(), 2);
    }

    #[test]
    fn test_stitch_no_conflict_for_unique_types() {
        let stitcher = MergeDirectiveSchemaStitcher::new();
        let result = stitcher
            .stitch(&[user_fragment(), product_fragment()])
            .expect("should succeed");
        // Only Query has a conflict (merge).
        // conflicts list records the merge event.
        // User and Product are unique — no conflicts.
        let user_conflict = result.conflicts.iter().any(|c| c.type_name == "User");
        assert!(!user_conflict);
    }

    #[test]
    fn test_stitch_merge_fields_deduplicates_same_field() {
        let stitcher = MergeDirectiveSchemaStitcher::new();
        // Both fragments define the same type with an overlapping field.
        let frag_a = SchemaFragment::new("a").with_type(
            StitchTypeDefinition::new("Shared")
                .with_field(StitchFieldDef::new("id", "ID!", "a"))
                .with_field(StitchFieldDef::new("name", "String", "a")),
        );
        let frag_b = SchemaFragment::new("b").with_type(
            StitchTypeDefinition::new("Shared")
                .with_field(StitchFieldDef::new("id", "ID!", "b")) // duplicate
                .with_field(StitchFieldDef::new("extra", "Int", "b")),
        );
        let result = stitcher.stitch(&[frag_a, frag_b]).expect("should succeed");
        let shared = result.get_type("Shared").expect("should succeed");
        // id, name, extra — id deduplicated
        assert_eq!(shared.fields.len(), 3);
    }

    #[test]
    fn test_stitch_keep_first_resolution() {
        let stitcher = MergeDirectiveSchemaStitcher::with_resolution(ConflictResolution::KeepFirst);
        let frag_a = SchemaFragment::new("a").with_type(
            StitchTypeDefinition::new("Config")
                .with_field(StitchFieldDef::new("version", "Int", "a")),
        );
        let frag_b = SchemaFragment::new("b").with_type(
            StitchTypeDefinition::new("Config")
                .with_field(StitchFieldDef::new("debug", "Boolean", "b")),
        );
        let result = stitcher.stitch(&[frag_a, frag_b]).expect("should succeed");
        let cfg = result.get_type("Config").expect("should succeed");
        // Only frag_a's field kept.
        assert_eq!(cfg.fields.len(), 1);
        assert_eq!(cfg.fields[0].name, "version");
    }

    #[test]
    fn test_stitch_keep_last_resolution() {
        let stitcher = MergeDirectiveSchemaStitcher::with_resolution(ConflictResolution::KeepLast);
        let frag_a = SchemaFragment::new("a").with_type(
            StitchTypeDefinition::new("Config")
                .with_field(StitchFieldDef::new("version", "Int", "a")),
        );
        let frag_b = SchemaFragment::new("b").with_type(
            StitchTypeDefinition::new("Config")
                .with_field(StitchFieldDef::new("debug", "Boolean", "b")),
        );
        let result = stitcher.stitch(&[frag_a, frag_b]).expect("should succeed");
        let cfg = result.get_type("Config").expect("should succeed");
        // Only frag_b's field kept.
        assert_eq!(cfg.fields.len(), 1);
        assert_eq!(cfg.fields[0].name, "debug");
    }

    #[test]
    fn test_stitch_error_resolution_fails() {
        let stitcher = MergeDirectiveSchemaStitcher::with_resolution(ConflictResolution::Error);
        let frag_a = SchemaFragment::new("a").with_type(
            StitchTypeDefinition::new("Conflict").with_field(StitchFieldDef::new("x", "Int", "a")),
        );
        let frag_b = SchemaFragment::new("b").with_type(
            StitchTypeDefinition::new("Conflict").with_field(StitchFieldDef::new("y", "Int", "b")),
        );
        let result = stitcher.stitch(&[frag_a, frag_b]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Conflict"));
    }

    #[test]
    fn test_stitch_merge_directive_overrides_resolution() {
        // Global = KeepFirst, but per-type @merge uses MergeFields.
        let stitcher = MergeDirectiveSchemaStitcher::with_resolution(ConflictResolution::KeepFirst);
        let frag_a = SchemaFragment::new("a").with_type(
            StitchTypeDefinition::new("Widget").with_field(StitchFieldDef::new("id", "ID!", "a")),
        );
        let frag_b = SchemaFragment::new("b")
            .with_type(
                StitchTypeDefinition::new("Widget")
                    .with_field(StitchFieldDef::new("label", "String", "b")),
            )
            .with_merge_directive(
                "Widget",
                MergeDirective {
                    key_field: Some("id".to_string()),
                    resolution: ConflictResolution::MergeFields,
                    deduplicate_fields: true,
                },
            );
        let result = stitcher.stitch(&[frag_a, frag_b]).expect("should succeed");
        let widget = result.get_type("Widget").expect("should succeed");
        // Both fields merged thanks to per-type directive.
        assert_eq!(widget.fields.len(), 2);
    }

    #[test]
    fn test_stitch_sources_recorded() {
        let stitcher = MergeDirectiveSchemaStitcher::new();
        let result = stitcher
            .stitch(&[user_fragment(), product_fragment()])
            .expect("should succeed");
        assert!(result.sources.contains(&"users".to_string()));
        assert!(result.sources.contains(&"products".to_string()));
    }

    #[test]
    fn test_stitch_empty_input() {
        let stitcher = MergeDirectiveSchemaStitcher::new();
        let result = stitcher.stitch(&[]).expect("should succeed");
        assert_eq!(result.type_count(), 0);
        assert!(!result.has_conflicts());
    }

    #[test]
    fn test_stitch_total_field_count() {
        let stitcher = MergeDirectiveSchemaStitcher::new();
        let result = stitcher
            .stitch(&[user_fragment(), product_fragment()])
            .expect("should succeed");
        // User(2) + Product(2) + Query(2) = 6
        assert_eq!(result.total_field_count(), 6);
    }

    #[test]
    fn test_stitch_get_type_unknown_returns_none() {
        let stitcher = MergeDirectiveSchemaStitcher::new();
        let result = stitcher.stitch(&[user_fragment()]).expect("should succeed");
        assert!(result.get_type("NonExistent").is_none());
    }

    #[test]
    fn test_stitch_conflict_list_has_merge_events() {
        let stitcher = MergeDirectiveSchemaStitcher::new();
        let result = stitcher
            .stitch(&[user_fragment(), product_fragment()])
            .expect("should succeed");
        // Query type was merged — should appear in conflicts.
        let query_conflict = result.conflicts.iter().any(|c| c.type_name == "Query");
        assert!(query_conflict);
    }

    #[test]
    fn test_stitch_root_type_marked() {
        let stitcher = MergeDirectiveSchemaStitcher::new();
        let result = stitcher.stitch(&[user_fragment()]).expect("should succeed");
        let query = result.get_type("Query").expect("should succeed");
        assert!(query.is_root);
    }

    #[test]
    fn test_stitch_three_fragments() {
        let stitcher = MergeDirectiveSchemaStitcher::new();
        let frag_c = SchemaFragment::new("reviews").with_type(
            StitchTypeDefinition::new("Review")
                .with_field(StitchFieldDef::new("rating", "Int!", "reviews")),
        );
        let result = stitcher
            .stitch(&[user_fragment(), product_fragment(), frag_c])
            .expect("should succeed");
        assert!(result.get_type("Review").is_some());
        assert_eq!(result.sources.len(), 3);
    }

    #[test]
    fn test_stitch_field_source_preserved() {
        let stitcher = MergeDirectiveSchemaStitcher::new();
        let result = stitcher.stitch(&[user_fragment()]).expect("should succeed");
        let user = result.get_type("User").expect("should succeed");
        assert!(user.fields.iter().all(|f| f.source == "users"));
    }
}
