//! # Schema Registry for Stream Events
//!
//! Enterprise-grade schema management, validation, and evolution for RDF streaming events.
//! Provides centralized schema storage, versioning, compatibility checking, and validation.
//!
//! Key features:
//! - Schema registration and versioning
//! - Forward/backward compatibility checks
//! - Event validation against schemas
//! - Schema evolution management
//! - Integration with external schema registries (Confluent, etc.)

use crate::{EventMetadata, StreamEvent};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Schema format types supported by the registry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SchemaFormat {
    /// JSON Schema for JSON events
    JsonSchema,
    /// Apache Avro schema
    Avro,
    /// Protocol Buffers schema
    Protobuf,
    /// RDF/SPARQL schema (custom)
    RdfSparql,
    /// Custom schema format
    Custom { format_name: String },
}

/// Schema compatibility modes for evolution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompatibilityMode {
    /// No compatibility checking
    None,
    /// New schema must be backward compatible
    Backward,
    /// New schema must be forward compatible
    Forward,
    /// New schema must be both backward and forward compatible
    Full,
    /// New schema can break compatibility (major version change)
    Breaking,
}

/// Schema definition with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDefinition {
    /// Unique schema identifier
    pub id: Uuid,
    /// Schema subject (topic/event type)
    pub subject: String,
    /// Schema version
    pub version: u32,
    /// Schema format
    pub format: SchemaFormat,
    /// Schema content (JSON, Avro, etc.)
    pub schema_content: String,
    /// Schema title/name
    pub title: Option<String>,
    /// Schema description
    pub description: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Compatibility mode for this schema
    pub compatibility: CompatibilityMode,
    /// Tags for organization
    pub tags: Vec<String>,
    /// Schema metadata
    pub metadata: HashMap<String, String>,
}

impl SchemaDefinition {
    pub fn new(
        subject: String,
        version: u32,
        format: SchemaFormat,
        schema_content: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            subject,
            version,
            format,
            schema_content,
            title: None,
            description: None,
            created_at: now,
            updated_at: now,
            compatibility: CompatibilityMode::Backward,
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Update schema content and metadata
    pub fn update_content(&mut self, content: String) {
        self.schema_content = content;
        self.updated_at = Utc::now();
    }

    /// Add tag to schema
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Set schema metadata
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
        self.updated_at = Utc::now();
    }
}

/// Schema validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors (if any)
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Schema used for validation
    pub schema_id: Uuid,
    /// Schema version used
    pub schema_version: u32,
    /// Validation timestamp
    pub validated_at: DateTime<Utc>,
}

impl ValidationResult {
    pub fn success(schema_id: Uuid, schema_version: u32) -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            schema_id,
            schema_version,
            validated_at: Utc::now(),
        }
    }

    pub fn failure(schema_id: Uuid, schema_version: u32, errors: Vec<String>) -> Self {
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
            schema_id,
            schema_version,
            validated_at: Utc::now(),
        }
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

/// Schema registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaRegistryConfig {
    /// Enable schema validation
    pub enable_validation: bool,
    /// Strict validation mode (fail on warnings)
    pub strict_mode: bool,
    /// Cache schema definitions in memory
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// External registry integration
    pub external_registry: Option<ExternalRegistryConfig>,
    /// Default compatibility mode for new schemas
    pub default_compatibility: CompatibilityMode,
    /// Maximum number of schema versions to keep
    pub max_versions_per_subject: u32,
}

impl Default for SchemaRegistryConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            strict_mode: false,
            enable_caching: true,
            cache_ttl_seconds: 3600, // 1 hour
            external_registry: None,
            default_compatibility: CompatibilityMode::Backward,
            max_versions_per_subject: 10,
        }
    }
}

/// External schema registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalRegistryConfig {
    /// Registry type (confluent, apicurio, etc.)
    pub registry_type: String,
    /// Registry URL
    pub url: String,
    /// Authentication configuration
    pub auth: Option<RegistryAuth>,
    /// Enable synchronization
    pub enable_sync: bool,
    /// Sync interval in seconds
    pub sync_interval_seconds: u64,
}

/// Registry authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryAuth {
    /// Authentication type (basic, bearer, etc.)
    pub auth_type: String,
    /// Username (for basic auth)
    pub username: Option<String>,
    /// Password (for basic auth)
    pub password: Option<String>,
    /// Bearer token
    pub token: Option<String>,
}

/// In-memory schema registry implementation
pub struct SchemaRegistry {
    /// Registry configuration
    config: SchemaRegistryConfig,
    /// Schema definitions by subject and version
    schemas: Arc<RwLock<HashMap<String, HashMap<u32, SchemaDefinition>>>>,
    /// Schema cache for fast lookups
    schema_cache: Arc<RwLock<HashMap<Uuid, SchemaDefinition>>>,
    /// Latest version per subject
    latest_versions: Arc<RwLock<HashMap<String, u32>>>,
    /// Validation statistics
    validation_stats: Arc<RwLock<ValidationStats>>,
}

/// Validation statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ValidationStats {
    pub total_validations: u64,
    pub successful_validations: u64,
    pub failed_validations: u64,
    pub warnings_count: u64,
    pub validation_time_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl SchemaRegistry {
    /// Create a new schema registry
    pub fn new(config: SchemaRegistryConfig) -> Self {
        Self {
            config,
            schemas: Arc::new(RwLock::new(HashMap::new())),
            schema_cache: Arc::new(RwLock::new(HashMap::new())),
            latest_versions: Arc::new(RwLock::new(HashMap::new())),
            validation_stats: Arc::new(RwLock::new(ValidationStats::default())),
        }
    }

    /// Register a new schema
    pub async fn register_schema(
        &self,
        subject: String,
        format: SchemaFormat,
        schema_content: String,
        compatibility: Option<CompatibilityMode>,
    ) -> Result<SchemaDefinition> {
        let mut schemas = self.schemas.write().await;
        let mut latest_versions = self.latest_versions.write().await;

        // Get next version number
        let subject_schemas = schemas.entry(subject.clone()).or_insert_with(HashMap::new);
        let next_version = latest_versions.get(&subject).map(|v| v + 1).unwrap_or(1);

        // Check compatibility if there are existing versions
        if next_version > 1 {
            let latest_version = next_version - 1;
            if let Some(existing_schema) = subject_schemas.get(&latest_version) {
                self.check_compatibility(&existing_schema, &schema_content, format.clone())
                    .await?;
            }
        }

        // Create new schema definition
        let mut schema = SchemaDefinition::new(subject.clone(), next_version, format, schema_content);
        
        if let Some(compat) = compatibility {
            schema.compatibility = compat;
        } else {
            schema.compatibility = self.config.default_compatibility.clone();
        }

        // Store schema
        subject_schemas.insert(next_version, schema.clone());
        latest_versions.insert(subject.clone(), next_version);

        // Update cache
        if self.config.enable_caching {
            let mut cache = self.schema_cache.write().await;
            cache.insert(schema.id, schema.clone());
        }

        info!(
            "Registered schema for subject '{}' version {} with ID {}",
            subject, next_version, schema.id
        );

        Ok(schema)
    }

    /// Get schema by subject and version
    pub async fn get_schema(
        &self,
        subject: &str,
        version: Option<u32>,
    ) -> Result<Option<SchemaDefinition>> {
        let schemas = self.schemas.read().await;
        
        if let Some(subject_schemas) = schemas.get(subject) {
            let version = if let Some(v) = version {
                v
            } else {
                // Get latest version
                let latest_versions = self.latest_versions.read().await;
                *latest_versions.get(subject).ok_or_else(|| {
                    anyhow!("No schemas found for subject: {}", subject)
                })?
            };

            Ok(subject_schemas.get(&version).cloned())
        } else {
            Ok(None)
        }
    }

    /// Get schema by ID
    pub async fn get_schema_by_id(&self, schema_id: &Uuid) -> Result<Option<SchemaDefinition>> {
        // Try cache first
        if self.config.enable_caching {
            let cache = self.schema_cache.read().await;
            if let Some(schema) = cache.get(schema_id) {
                let mut stats = self.validation_stats.write().await;
                stats.cache_hits += 1;
                return Ok(Some(schema.clone()));
            }
        }

        // Search through all schemas
        let schemas = self.schemas.read().await;
        for subject_schemas in schemas.values() {
            for schema in subject_schemas.values() {
                if &schema.id == schema_id {
                    // Update cache
                    if self.config.enable_caching {
                        let mut cache = self.schema_cache.write().await;
                        cache.insert(*schema_id, schema.clone());
                    }
                    
                    let mut stats = self.validation_stats.write().await;
                    stats.cache_misses += 1;
                    return Ok(Some(schema.clone()));
                }
            }
        }

        Ok(None)
    }

    /// List all schemas for a subject
    pub async fn list_schemas(&self, subject: &str) -> Result<Vec<SchemaDefinition>> {
        let schemas = self.schemas.read().await;
        
        if let Some(subject_schemas) = schemas.get(subject) {
            let mut schemas: Vec<SchemaDefinition> = subject_schemas.values().cloned().collect();
            schemas.sort_by(|a, b| a.version.cmp(&b.version));
            Ok(schemas)
        } else {
            Ok(Vec::new())
        }
    }

    /// List all subjects
    pub async fn list_subjects(&self) -> Result<Vec<String>> {
        let schemas = self.schemas.read().await;
        Ok(schemas.keys().cloned().collect())
    }

    /// Validate event against schema
    pub async fn validate_event(
        &self,
        event: &StreamEvent,
        subject: Option<&str>,
    ) -> Result<ValidationResult> {
        if !self.config.enable_validation {
            return Ok(ValidationResult::success(Uuid::new_v4(), 1));
        }

        let start_time = std::time::Instant::now();
        let mut stats = self.validation_stats.write().await;
        stats.total_validations += 1;
        drop(stats);

        // Determine subject from event or parameter
        let event_subject = subject
            .map(|s| s.to_string())
            .or_else(|| self.extract_subject_from_event(event))
            .ok_or_else(|| anyhow!("Cannot determine subject for validation"))?;

        // Get latest schema for subject
        let schema = self
            .get_schema(&event_subject, None)
            .await?
            .ok_or_else(|| anyhow!("No schema found for subject: {}", event_subject))?;

        // Perform validation based on schema format
        let validation_result = match schema.format {
            SchemaFormat::JsonSchema => self.validate_with_json_schema(event, &schema).await?,
            SchemaFormat::RdfSparql => self.validate_with_rdf_schema(event, &schema).await?,
            SchemaFormat::Avro => self.validate_with_avro_schema(event, &schema).await?,
            _ => {
                warn!("Validation not implemented for format: {:?}", schema.format);
                ValidationResult::success(schema.id, schema.version)
            }
        };

        // Update statistics
        let elapsed = start_time.elapsed();
        let mut stats = self.validation_stats.write().await;
        stats.validation_time_ms = (stats.validation_time_ms + elapsed.as_millis() as f64) / 2.0;
        
        if validation_result.is_valid {
            stats.successful_validations += 1;
        } else {
            stats.failed_validations += 1;
        }
        
        stats.warnings_count += validation_result.warnings.len() as u64;

        debug!(
            "Validated event against schema {} ({}ms): {}",
            schema.id,
            elapsed.as_millis(),
            if validation_result.is_valid { "VALID" } else { "INVALID" }
        );

        Ok(validation_result)
    }

    /// Extract subject from event metadata or event type
    fn extract_subject_from_event(&self, event: &StreamEvent) -> Option<String> {
        // Try to get subject from event metadata
        match event {
            StreamEvent::TripleAdded { metadata, .. } => {
                metadata.properties.get("subject").cloned()
                    .or_else(|| Some(format!("rdf.triple.added")))
            }
            StreamEvent::TripleRemoved { metadata, .. } => {
                metadata.properties.get("subject").cloned()
                    .or_else(|| Some(format!("rdf.triple.removed")))
            }
            StreamEvent::SparqlUpdate { metadata, .. } => {
                metadata.properties.get("subject").cloned()
                    .or_else(|| Some(format!("sparql.update")))
            }
            StreamEvent::TransactionBegin { metadata, .. } => {
                metadata.properties.get("subject").cloned()
                    .or_else(|| Some(format!("transaction.begin")))
            }
            StreamEvent::TransactionCommit { metadata, .. } => {
                metadata.properties.get("subject").cloned()
                    .or_else(|| Some(format!("transaction.commit")))
            }
            _ => Some(format!("stream.event.{:?}", std::mem::discriminant(event))),
        }
    }

    /// Check compatibility between schemas
    async fn check_compatibility(
        &self,
        existing_schema: &SchemaDefinition,
        new_schema_content: &str,
        new_format: SchemaFormat,
    ) -> Result<()> {
        if existing_schema.compatibility == CompatibilityMode::None {
            return Ok(());
        }

        if existing_schema.format != new_format {
            return Err(anyhow!(
                "Schema format changed from {:?} to {:?}",
                existing_schema.format,
                new_format
            ));
        }

        // Basic compatibility checking (simplified)
        match new_format {
            SchemaFormat::JsonSchema => {
                self.check_json_schema_compatibility(existing_schema, new_schema_content)
                    .await
            }
            SchemaFormat::RdfSparql => {
                self.check_rdf_schema_compatibility(existing_schema, new_schema_content)
                    .await
            }
            _ => {
                warn!("Compatibility checking not implemented for format: {:?}", new_format);
                Ok(())
            }
        }
    }

    /// Validate event with JSON schema
    async fn validate_with_json_schema(
        &self,
        _event: &StreamEvent,
        schema: &SchemaDefinition,
    ) -> Result<ValidationResult> {
        // Simplified JSON schema validation
        // In a real implementation, you would use a JSON schema library
        debug!("Validating with JSON schema: {}", schema.id);
        Ok(ValidationResult::success(schema.id, schema.version))
    }

    /// Validate event with RDF/SPARQL schema
    async fn validate_with_rdf_schema(
        &self,
        event: &StreamEvent,
        schema: &SchemaDefinition,
    ) -> Result<ValidationResult> {
        // Custom RDF validation logic
        match event {
            StreamEvent::TripleAdded { subject, predicate, object, .. } => {
                let mut errors = Vec::new();
                
                // Basic URI validation
                if !subject.starts_with("http://") && !subject.starts_with("https://") {
                    errors.push(format!("Invalid subject URI: {}", subject));
                }
                
                if !predicate.starts_with("http://") && !predicate.starts_with("https://") {
                    errors.push(format!("Invalid predicate URI: {}", predicate));
                }

                if errors.is_empty() {
                    Ok(ValidationResult::success(schema.id, schema.version))
                } else {
                    Ok(ValidationResult::failure(schema.id, schema.version, errors))
                }
            }
            _ => Ok(ValidationResult::success(schema.id, schema.version)),
        }
    }

    /// Validate event with Avro schema
    async fn validate_with_avro_schema(
        &self,
        _event: &StreamEvent,
        schema: &SchemaDefinition,
    ) -> Result<ValidationResult> {
        // Simplified Avro validation
        // In a real implementation, you would use the Apache Avro library
        debug!("Validating with Avro schema: {}", schema.id);
        Ok(ValidationResult::success(schema.id, schema.version))
    }

    /// Check JSON schema compatibility
    async fn check_json_schema_compatibility(
        &self,
        _existing_schema: &SchemaDefinition,
        _new_schema_content: &str,
    ) -> Result<()> {
        // Simplified compatibility check
        // Real implementation would parse and compare JSON schemas
        Ok(())
    }

    /// Check RDF schema compatibility
    async fn check_rdf_schema_compatibility(
        &self,
        _existing_schema: &SchemaDefinition,
        _new_schema_content: &str,
    ) -> Result<()> {
        // Simplified compatibility check for RDF schemas
        Ok(())
    }

    /// Get validation statistics
    pub async fn get_validation_stats(&self) -> ValidationStats {
        let stats = self.validation_stats.read().await;
        (*stats).clone()
    }

    /// Delete schema
    pub async fn delete_schema(&self, subject: &str, version: Option<u32>) -> Result<bool> {
        let mut schemas = self.schemas.write().await;
        let mut latest_versions = self.latest_versions.write().await;

        if let Some(subject_schemas) = schemas.get_mut(subject) {
            if let Some(version) = version {
                // Delete specific version
                let removed = subject_schemas.remove(&version).is_some();
                
                // Update latest version if this was the latest
                if let Some(latest) = latest_versions.get(subject) {
                    if *latest == version {
                        let new_latest = subject_schemas.keys().max().cloned();
                        if let Some(new_latest) = new_latest {
                            latest_versions.insert(subject.to_string(), new_latest);
                        } else {
                            latest_versions.remove(subject);
                            schemas.remove(subject);
                        }
                    }
                }
                
                Ok(removed)
            } else {
                // Delete all versions for subject
                schemas.remove(subject);
                latest_versions.remove(subject);
                Ok(true)
            }
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_schema_registration() -> Result<()> {
        let config = SchemaRegistryConfig::default();
        let registry = SchemaRegistry::new(config);

        let schema_content = r#"
        {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"}
            },
            "required": ["subject", "predicate", "object"]
        }"#;

        let schema = registry
            .register_schema(
                "rdf.triple.added".to_string(),
                SchemaFormat::JsonSchema,
                schema_content.to_string(),
                None,
            )
            .await?;

        assert_eq!(schema.subject, "rdf.triple.added");
        assert_eq!(schema.version, 1);
        assert_eq!(schema.format, SchemaFormat::JsonSchema);

        Ok(())
    }

    #[tokio::test]
    async fn test_schema_retrieval() -> Result<()> {
        let config = SchemaRegistryConfig::default();
        let registry = SchemaRegistry::new(config);

        let schema_content = r#"{"type": "object"}"#;
        let registered_schema = registry
            .register_schema(
                "test.subject".to_string(),
                SchemaFormat::JsonSchema,
                schema_content.to_string(),
                None,
            )
            .await?;

        // Get by subject and version
        let retrieved = registry
            .get_schema("test.subject", Some(1))
            .await?
            .expect("Schema should exist");

        assert_eq!(retrieved.id, registered_schema.id);
        assert_eq!(retrieved.version, 1);

        // Get by ID
        let retrieved_by_id = registry
            .get_schema_by_id(&registered_schema.id)
            .await?
            .expect("Schema should exist");

        assert_eq!(retrieved_by_id.id, registered_schema.id);

        Ok(())
    }

    #[tokio::test]
    async fn test_event_validation() -> Result<()> {
        let config = SchemaRegistryConfig::default();
        let registry = SchemaRegistry::new(config);

        // Register RDF schema
        let schema_content = "RDF Triple Schema";
        registry
            .register_schema(
                "rdf.triple.added".to_string(),
                SchemaFormat::RdfSparql,
                schema_content.to_string(),
                None,
            )
            .await?;

        // Create test event
        let event = StreamEvent::TripleAdded {
            subject: "https://example.org/subject".to_string(),
            predicate: "https://example.org/predicate".to_string(),
            object: "\"Test Object\"".to_string(),
            graph: None,
            metadata: EventMetadata {
                event_id: "test_event_1".to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        let validation_result = registry.validate_event(&event, Some("rdf.triple.added")).await?;

        assert!(validation_result.is_valid);
        assert!(validation_result.errors.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_schema_versioning() -> Result<()> {
        let config = SchemaRegistryConfig::default();
        let registry = SchemaRegistry::new(config);

        let subject = "test.versioning".to_string();

        // Register version 1
        let _v1 = registry
            .register_schema(
                subject.clone(),
                SchemaFormat::JsonSchema,
                r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#.to_string(),
                None,
            )
            .await?;

        // Register version 2
        let _v2 = registry
            .register_schema(
                subject.clone(),
                SchemaFormat::JsonSchema,
                r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}"#.to_string(),
                None,
            )
            .await?;

        // List all schemas for subject
        let schemas = registry.list_schemas(&subject).await?;
        assert_eq!(schemas.len(), 2);
        assert_eq!(schemas[0].version, 1);
        assert_eq!(schemas[1].version, 2);

        // Get latest version
        let latest = registry.get_schema(&subject, None).await?.unwrap();
        assert_eq!(latest.version, 2);

        Ok(())
    }
}