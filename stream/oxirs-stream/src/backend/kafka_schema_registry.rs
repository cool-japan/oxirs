//! # Kafka Schema Registry Integration
//!
//! Provides comprehensive schema registry support for Kafka streaming,
//! enabling type-safe RDF event serialization and evolution.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

#[cfg(feature = "kafka")]
use reqwest::{Client as HttpClient, StatusCode};

use crate::backend::kafka::SchemaRegistryConfig;

/// Schema types supported by the registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaType {
    Avro,
    Json,
    Protobuf,
}

impl ToString for SchemaType {
    fn to_string(&self) -> String {
        match self {
            SchemaType::Avro => "AVRO".to_string(),
            SchemaType::Json => "JSON".to_string(),
            SchemaType::Protobuf => "PROTOBUF".to_string(),
        }
    }
}

/// Schema compatibility levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityLevel {
    Backward,
    BackwardTransitive,
    Forward,
    ForwardTransitive,
    Full,
    FullTransitive,
    None,
}

impl ToString for CompatibilityLevel {
    fn to_string(&self) -> String {
        match self {
            CompatibilityLevel::Backward => "BACKWARD".to_string(),
            CompatibilityLevel::BackwardTransitive => "BACKWARD_TRANSITIVE".to_string(),
            CompatibilityLevel::Forward => "FORWARD".to_string(),
            CompatibilityLevel::ForwardTransitive => "FORWARD_TRANSITIVE".to_string(),
            CompatibilityLevel::Full => "FULL".to_string(),
            CompatibilityLevel::FullTransitive => "FULL_TRANSITIVE".to_string(),
            CompatibilityLevel::None => "NONE".to_string(),
        }
    }
}

/// Schema metadata from registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMetadata {
    pub id: u32,
    pub version: u32,
    pub schema: String,
    pub schema_type: SchemaType,
    pub subject: String,
    pub references: Vec<SchemaReference>,
}

/// Schema reference for complex schemas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaReference {
    pub name: String,
    pub subject: String,
    pub version: u32,
}

/// Schema registry client for Kafka
pub struct SchemaRegistryClient {
    config: SchemaRegistryConfig,
    #[cfg(feature = "kafka")]
    http_client: HttpClient,
    schema_cache: Arc<RwLock<HashMap<String, SchemaMetadata>>>,
    id_cache: Arc<RwLock<HashMap<u32, SchemaMetadata>>>,
}

impl SchemaRegistryClient {
    /// Helper to create request builder with auth
    #[cfg(feature = "kafka")]
    fn request_builder(&self, method: reqwest::Method, url: &str) -> reqwest::RequestBuilder {
        let mut builder = self.http_client.request(method, url);

        // Add basic auth if configured
        if let (Some(ref username), Some(ref password)) =
            (&self.config.username, &self.config.password)
        {
            builder = builder.basic_auth(username, Some(password));
        }

        builder
    }

    /// Create a new schema registry client
    pub fn new(config: SchemaRegistryConfig) -> Result<Self> {
        #[cfg(feature = "kafka")]
        {
            let client_builder = HttpClient::builder()
                .timeout(std::time::Duration::from_millis(config.timeout_ms as u64));

            let http_client = client_builder
                .build()
                .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

            Ok(Self {
                config,
                http_client,
                schema_cache: Arc::new(RwLock::new(HashMap::with_capacity(100))),
                id_cache: Arc::new(RwLock::new(HashMap::with_capacity(100))),
            })
        }

        #[cfg(not(feature = "kafka"))]
        {
            Ok(Self {
                config,
                schema_cache: Arc::new(RwLock::new(HashMap::with_capacity(100))),
                id_cache: Arc::new(RwLock::new(HashMap::with_capacity(100))),
            })
        }
    }

    /// Register a new schema version
    #[cfg(feature = "kafka")]
    pub async fn register_schema(
        &self,
        subject: &str,
        schema: &str,
        schema_type: SchemaType,
        references: Option<Vec<SchemaReference>>,
    ) -> Result<SchemaMetadata> {
        let url = format!("{}/subjects/{}/versions", self.config.url, subject);

        let request_body = serde_json::json!({
            "schema": schema,
            "schemaType": schema_type.to_string(),
            "references": references.unwrap_or_default()
        });

        let response = self
            .request_builder(reqwest::Method::POST, &url)
            .header("Content-Type", "application/vnd.schemaregistry.v1+json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to register schema: {}", e))?;

        if response.status() == StatusCode::OK || response.status() == StatusCode::CREATED {
            let result: serde_json::Value = response
                .json()
                .await
                .map_err(|e| anyhow!("Failed to parse response: {}", e))?;

            let id = result["id"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing schema ID in response"))?
                as u32;

            // Get full schema metadata
            self.get_schema_by_id(id).await
        } else {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(anyhow!(
                "Failed to register schema: {} - {}",
                status,
                error_text
            ))
        }
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn register_schema(
        &self,
        subject: &str,
        schema: &str,
        schema_type: SchemaType,
        references: Option<Vec<SchemaReference>>,
    ) -> Result<SchemaMetadata> {
        // Mock implementation
        let metadata = SchemaMetadata {
            id: 1,
            version: 1,
            schema: schema.to_string(),
            schema_type,
            subject: subject.to_string(),
            references: references.unwrap_or_default(),
        };

        self.schema_cache
            .write()
            .await
            .insert(subject.to_string(), metadata.clone());
        self.id_cache.write().await.insert(1, metadata.clone());

        info!("Mock: registered schema for subject {}", subject);
        Ok(metadata)
    }

    /// Get schema by ID
    #[cfg(feature = "kafka")]
    pub async fn get_schema_by_id(&self, id: u32) -> Result<SchemaMetadata> {
        // Check cache first
        if let Some(metadata) = self.id_cache.read().await.get(&id) {
            return Ok(metadata.clone());
        }

        let url = format!("{}/schemas/ids/{}", self.config.url, id);

        let response = self
            .request_builder(reqwest::Method::GET, &url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to get schema: {}", e))?;

        if response.status() == StatusCode::OK {
            let result: serde_json::Value = response
                .json()
                .await
                .map_err(|e| anyhow!("Failed to parse response: {}", e))?;

            let metadata = SchemaMetadata {
                id,
                version: 1, // Version needs to be fetched separately
                schema: result["schema"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing schema in response"))?
                    .to_string(),
                schema_type: SchemaType::Json, // Default, should be parsed from response
                subject: String::new(),        // Subject needs to be fetched separately
                references: vec![],
            };

            // Cache the result
            self.id_cache.write().await.insert(id, metadata.clone());

            Ok(metadata)
        } else {
            Err(anyhow!("Failed to get schema: {}", response.status()))
        }
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn get_schema_by_id(&self, id: u32) -> Result<SchemaMetadata> {
        if let Some(metadata) = self.id_cache.read().await.get(&id) {
            Ok(metadata.clone())
        } else {
            Err(anyhow!("Schema not found for ID: {}", id))
        }
    }

    /// Get latest schema version for a subject
    #[cfg(feature = "kafka")]
    pub async fn get_latest_schema(&self, subject: &str) -> Result<SchemaMetadata> {
        // Check cache first
        if let Some(metadata) = self.schema_cache.read().await.get(subject) {
            return Ok(metadata.clone());
        }

        let url = format!("{}/subjects/{}/versions/latest", self.config.url, subject);

        let response = self
            .request_builder(reqwest::Method::GET, &url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to get latest schema: {}", e))?;

        if response.status() == StatusCode::OK {
            let result: serde_json::Value = response
                .json()
                .await
                .map_err(|e| anyhow!("Failed to parse response: {}", e))?;

            let metadata = SchemaMetadata {
                id: result["id"]
                    .as_u64()
                    .ok_or_else(|| anyhow!("Missing schema ID"))? as u32,
                version: result["version"]
                    .as_u64()
                    .ok_or_else(|| anyhow!("Missing schema version"))?
                    as u32,
                schema: result["schema"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing schema"))?
                    .to_string(),
                schema_type: SchemaType::Json, // Should be parsed from response
                subject: subject.to_string(),
                references: vec![], // Should be parsed from response
            };

            // Cache the result
            self.schema_cache
                .write()
                .await
                .insert(subject.to_string(), metadata.clone());
            self.id_cache
                .write()
                .await
                .insert(metadata.id, metadata.clone());

            Ok(metadata)
        } else if response.status() == StatusCode::NOT_FOUND {
            Err(anyhow!("Schema subject not found: {}", subject))
        } else {
            Err(anyhow!(
                "Failed to get latest schema: {}",
                response.status()
            ))
        }
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn get_latest_schema(&self, subject: &str) -> Result<SchemaMetadata> {
        if let Some(metadata) = self.schema_cache.read().await.get(subject) {
            Ok(metadata.clone())
        } else {
            Err(anyhow!("Schema not found for subject: {}", subject))
        }
    }

    /// Check schema compatibility
    #[cfg(feature = "kafka")]
    pub async fn check_compatibility(
        &self,
        subject: &str,
        schema: &str,
        version: Option<String>,
    ) -> Result<bool> {
        let version_str = version.unwrap_or_else(|| "latest".to_string());
        let url = format!(
            "{}/compatibility/subjects/{}/versions/{}",
            self.config.url, subject, version_str
        );

        let request_body = serde_json::json!({
            "schema": schema
        });

        let response = self
            .request_builder(reqwest::Method::POST, &url)
            .header("Content-Type", "application/vnd.schemaregistry.v1+json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to check compatibility: {}", e))?;

        if response.status() == StatusCode::OK {
            let result: serde_json::Value = response
                .json()
                .await
                .map_err(|e| anyhow!("Failed to parse response: {}", e))?;

            Ok(result["is_compatible"].as_bool().unwrap_or(false))
        } else {
            Err(anyhow!(
                "Failed to check compatibility: {}",
                response.status()
            ))
        }
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn check_compatibility(
        &self,
        _subject: &str,
        _schema: &str,
        _version: Option<String>,
    ) -> Result<bool> {
        // Mock always returns compatible
        Ok(true)
    }

    /// Set compatibility level for a subject
    #[cfg(feature = "kafka")]
    pub async fn set_compatibility_level(
        &self,
        subject: &str,
        level: CompatibilityLevel,
    ) -> Result<()> {
        let url = format!("{}/config/{}", self.config.url, subject);

        let request_body = serde_json::json!({
            "compatibility": level.to_string()
        });

        let response = self
            .request_builder(reqwest::Method::PUT, &url)
            .header("Content-Type", "application/vnd.schemaregistry.v1+json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to set compatibility level: {}", e))?;

        if response.status() == StatusCode::OK {
            info!(
                "Set compatibility level for subject {} to {}",
                subject,
                level.to_string()
            );
            Ok(())
        } else {
            Err(anyhow!(
                "Failed to set compatibility level: {}",
                response.status()
            ))
        }
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn set_compatibility_level(
        &self,
        subject: &str,
        level: CompatibilityLevel,
    ) -> Result<()> {
        info!(
            "Mock: set compatibility level for subject {} to {}",
            subject,
            level.to_string()
        );
        Ok(())
    }

    /// Delete a subject and all its versions
    #[cfg(feature = "kafka")]
    pub async fn delete_subject(&self, subject: &str) -> Result<Vec<u32>> {
        let url = format!("{}/subjects/{}", self.config.url, subject);

        let response = self
            .request_builder(reqwest::Method::DELETE, &url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to delete subject: {}", e))?;

        if response.status() == StatusCode::OK {
            let versions: Vec<u32> = response
                .json()
                .await
                .map_err(|e| anyhow!("Failed to parse response: {}", e))?;

            // Remove from cache
            self.schema_cache.write().await.remove(subject);

            info!("Deleted subject {} with versions: {:?}", subject, versions);
            Ok(versions)
        } else {
            Err(anyhow!("Failed to delete subject: {}", response.status()))
        }
    }

    #[cfg(not(feature = "kafka"))]
    pub async fn delete_subject(&self, subject: &str) -> Result<Vec<u32>> {
        self.schema_cache.write().await.remove(subject);
        info!("Mock: deleted subject {}", subject);
        Ok(vec![1])
    }

    /// Clear schema cache
    pub async fn clear_cache(&self) {
        self.schema_cache.write().await.clear();
        self.id_cache.write().await.clear();
        debug!("Cleared schema registry cache");
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let schema_cache_size = self.schema_cache.read().await.len();
        let id_cache_size = self.id_cache.read().await.len();
        (schema_cache_size, id_cache_size)
    }
}

/// RDF Event schema definitions
pub struct RdfEventSchemas;

impl RdfEventSchemas {
    /// Get JSON schema for RDF triple events
    pub fn triple_event_schema() -> &'static str {
        r#"{
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "event_id": { "type": "string" },
                "event_type": { 
                    "type": "string",
                    "enum": ["triple_added", "triple_removed"]
                },
                "timestamp": { "type": "string", "format": "date-time" },
                "data": {
                    "type": "object",
                    "properties": {
                        "subject": { "type": "string", "format": "uri" },
                        "predicate": { "type": "string", "format": "uri" },
                        "object": { "type": "string" },
                        "graph": { "type": ["string", "null"], "format": "uri" }
                    },
                    "required": ["subject", "predicate", "object"]
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source": { "type": "string" },
                        "user": { "type": ["string", "null"] },
                        "context": { "type": ["string", "null"] }
                    }
                }
            },
            "required": ["event_id", "event_type", "timestamp", "data"]
        }"#
    }

    /// Get JSON schema for graph operations
    pub fn graph_event_schema() -> &'static str {
        r#"{
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "event_id": { "type": "string" },
                "event_type": { 
                    "type": "string",
                    "enum": ["graph_created", "graph_cleared", "graph_deleted"]
                },
                "timestamp": { "type": "string", "format": "date-time" },
                "data": {
                    "type": "object",
                    "properties": {
                        "graph": { "type": ["string", "null"], "format": "uri" }
                    }
                },
                "metadata": {
                    "type": "object"
                }
            },
            "required": ["event_id", "event_type", "timestamp", "data"]
        }"#
    }

    /// Get JSON schema for SPARQL update events
    pub fn sparql_update_schema() -> &'static str {
        r#"{
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "event_id": { "type": "string" },
                "event_type": { 
                    "type": "string",
                    "const": "sparql_update"
                },
                "timestamp": { "type": "string", "format": "date-time" },
                "data": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "operation_type": { 
                            "type": "string",
                            "enum": ["INSERT", "DELETE", "UPDATE", "LOAD", "CLEAR", "CREATE", "DROP"]
                        }
                    },
                    "required": ["query", "operation_type"]
                },
                "metadata": {
                    "type": "object"
                }
            },
            "required": ["event_id", "event_type", "timestamp", "data"]
        }"#
    }

    /// Register all RDF event schemas
    pub async fn register_all_schemas(
        client: &SchemaRegistryClient,
        subject_prefix: &str,
    ) -> Result<()> {
        // Register triple event schema
        client
            .register_schema(
                &format!("{}-triple-event", subject_prefix),
                Self::triple_event_schema(),
                SchemaType::Json,
                None,
            )
            .await?;

        // Register graph event schema
        client
            .register_schema(
                &format!("{}-graph-event", subject_prefix),
                Self::graph_event_schema(),
                SchemaType::Json,
                None,
            )
            .await?;

        // Register SPARQL update schema
        client
            .register_schema(
                &format!("{}-sparql-update", subject_prefix),
                Self::sparql_update_schema(),
                SchemaType::Json,
                None,
            )
            .await?;

        info!(
            "Registered all RDF event schemas with prefix: {}",
            subject_prefix
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_schema_registry_client() {
        let config = SchemaRegistryConfig {
            url: "http://localhost:8081".to_string(),
            username: None,
            password: None,
            timeout_ms: 5000,
            cache_size: 100,
        };

        let client = SchemaRegistryClient::new(config).unwrap();

        // Test schema registration (mock mode)
        let metadata = client
            .register_schema(
                "test-subject",
                RdfEventSchemas::triple_event_schema(),
                SchemaType::Json,
                None,
            )
            .await
            .unwrap();

        assert_eq!(metadata.subject, "test-subject");
        assert_eq!(metadata.id, 1);

        // Test cache
        let (schema_cache_size, id_cache_size) = client.get_cache_stats().await;
        assert_eq!(schema_cache_size, 1);
        assert_eq!(id_cache_size, 1);
    }

    #[test]
    fn test_rdf_schemas() {
        // Verify schemas are valid JSON
        let triple_schema = RdfEventSchemas::triple_event_schema();
        let parsed: serde_json::Value = serde_json::from_str(triple_schema).unwrap();
        assert_eq!(parsed["$schema"], "http://json-schema.org/draft-07/schema#");

        let graph_schema = RdfEventSchemas::graph_event_schema();
        let parsed: serde_json::Value = serde_json::from_str(graph_schema).unwrap();
        assert_eq!(parsed["type"], "object");

        let sparql_schema = RdfEventSchemas::sparql_update_schema();
        let parsed: serde_json::Value = serde_json::from_str(sparql_schema).unwrap();
        assert!(
            parsed["properties"]["data"]["properties"]["operation_type"]["enum"]
                .as_array()
                .unwrap()
                .contains(&serde_json::Value::String("INSERT".to_string()))
        );
    }
}
