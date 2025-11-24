//! GraphQL Mesh Integration
//!
//! This module provides integration with GraphQL Mesh concepts for combining
//! multiple data sources into a unified GraphQL API.
//!
//! ## Features
//!
//! - **Source Management**: Define and manage multiple data sources
//! - **Transform Pipelines**: Apply transforms to source schemas
//! - **Type Merging**: Merge types across sources
//! - **Cross-Source Relationships**: Define relationships between sources
//! - **Caching Layer**: Unified caching across sources

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// Data source type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceType {
    /// GraphQL endpoint
    GraphQL,
    /// REST API
    REST,
    /// OpenAPI specification
    OpenAPI,
    /// SOAP service
    SOAP,
    /// gRPC service
    GRPC,
    /// OData endpoint
    OData,
    /// Database (SQL)
    Database,
    /// JSON Schema
    JsonSchema,
    /// Custom source
    Custom,
}

/// Data source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSource {
    /// Source identifier
    pub id: String,
    /// Source name
    pub name: String,
    /// Source type
    pub source_type: SourceType,
    /// Endpoint URL
    pub endpoint: String,
    /// Schema (SDL for GraphQL, spec for OpenAPI, etc.)
    pub schema: Option<String>,
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    /// Headers to include
    pub headers: HashMap<String, String>,
    /// Transforms to apply
    pub transforms: Vec<Transform>,
    /// Cache configuration
    pub cache: Option<CacheConfig>,
    /// Health check endpoint
    pub health_check: Option<String>,
    /// Is enabled
    pub enabled: bool,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl DataSource {
    /// Create a new data source
    pub fn new(id: &str, name: &str, source_type: SourceType, endpoint: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            source_type,
            endpoint: endpoint.to_string(),
            schema: None,
            auth: None,
            headers: HashMap::new(),
            transforms: Vec::new(),
            cache: None,
            health_check: None,
            enabled: true,
            metadata: HashMap::new(),
        }
    }

    /// Add authentication
    pub fn with_auth(mut self, auth: AuthConfig) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Add header
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    /// Add transform
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transforms.push(transform);
        self
    }

    /// Set cache configuration
    pub fn with_cache(mut self, cache: CacheConfig) -> Self {
        self.cache = Some(cache);
        self
    }
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthConfig {
    /// No authentication
    None,
    /// Basic authentication
    Basic { username: String, password: String },
    /// Bearer token
    Bearer { token: String },
    /// API key
    ApiKey { key: String, header: String },
    /// OAuth2
    OAuth2 {
        client_id: String,
        client_secret: String,
        token_url: String,
        scopes: Vec<String>,
    },
    /// Custom
    Custom { config: HashMap<String, String> },
}

/// Transform type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Transform {
    /// Prefix type names
    Prefix { value: String },
    /// Rename type
    Rename { from: String, to: String },
    /// Filter types
    FilterTypes { include: Vec<String> },
    /// Filter fields
    FilterFields {
        type_name: String,
        include: Vec<String>,
    },
    /// Add field
    AddField {
        type_name: String,
        field_name: String,
        field_type: String,
    },
    /// Encapsulate with namespace
    Encapsulate { namespace: String },
    /// Snapshot (freeze schema)
    Snapshot { name: String },
    /// Custom transform
    Custom {
        name: String,
        config: HashMap<String, String>,
    },
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Cache TTL
    pub ttl: Duration,
    /// Max cache entries
    pub max_entries: usize,
    /// Cache key strategy
    pub key_strategy: CacheKeyStrategy,
    /// Invalidation events
    pub invalidation_events: Vec<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(60),
            max_entries: 1000,
            key_strategy: CacheKeyStrategy::default(),
            invalidation_events: Vec::new(),
        }
    }
}

/// Cache key strategy
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum CacheKeyStrategy {
    /// Full query hash
    #[default]
    QueryHash,
    /// Field-level caching
    FieldLevel,
    /// Custom key
    Custom { template: String },
}

/// Type merge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeMergeConfig {
    /// Type name
    pub type_name: String,
    /// Field to use as key
    pub key_field: String,
    /// Sources that provide this type
    pub sources: Vec<String>,
    /// Resolution strategy
    pub resolution: MergeResolution,
}

/// Merge resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeResolution {
    /// First source wins
    First,
    /// Last source wins
    Last,
    /// Merge all fields
    MergeAll,
    /// Fail on conflict
    FailOnConflict,
}

/// Cross-source relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSourceRelation {
    /// Source type
    pub source_type: String,
    /// Source field
    pub source_field: String,
    /// Source ID
    pub source_id: String,
    /// Target type
    pub target_type: String,
    /// Target field for lookup
    pub target_key_field: String,
    /// Target source ID
    pub target_source_id: String,
    /// Relation type
    pub relation_type: RelationType,
}

/// Relation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationType {
    /// One to one
    OneToOne,
    /// One to many
    OneToMany,
    /// Many to one
    ManyToOne,
    /// Many to many
    ManyToMany,
}

/// Mesh configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshConfig {
    /// Name
    pub name: String,
    /// Data sources
    pub sources: Vec<DataSource>,
    /// Type merges
    pub type_merges: Vec<TypeMergeConfig>,
    /// Cross-source relations
    pub relations: Vec<CrossSourceRelation>,
    /// Global transforms
    pub global_transforms: Vec<Transform>,
    /// Enable introspection
    pub enable_introspection: bool,
    /// Serve config
    pub serve: ServeConfig,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            name: "mesh".to_string(),
            sources: Vec::new(),
            type_merges: Vec::new(),
            relations: Vec::new(),
            global_transforms: Vec::new(),
            enable_introspection: true,
            serve: ServeConfig::default(),
        }
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServeConfig {
    /// Host
    pub host: String,
    /// Port
    pub port: u16,
    /// Enable playground
    pub playground: bool,
    /// CORS origins
    pub cors: Vec<String>,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 4000,
            playground: true,
            cors: vec!["*".to_string()],
        }
    }
}

/// Source health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceHealth {
    /// Source ID
    pub source_id: String,
    /// Is healthy
    pub healthy: bool,
    /// Last check time
    pub last_check: SystemTime,
    /// Last error
    pub last_error: Option<String>,
    /// Latency (ms)
    pub latency_ms: u64,
}

/// Internal state
struct MeshState {
    /// Config
    config: MeshConfig,
    /// Source health
    source_health: HashMap<String, SourceHealth>,
    /// Merged schema SDL
    merged_schema: Option<String>,
    /// Last merge time
    last_merge: Option<SystemTime>,
}

impl MeshState {
    fn new(config: MeshConfig) -> Self {
        Self {
            config,
            source_health: HashMap::new(),
            merged_schema: None,
            last_merge: None,
        }
    }
}

/// GraphQL Mesh Manager
///
/// Manages multiple data sources and combines them into a unified GraphQL API.
pub struct GraphQLMesh {
    /// Internal state
    state: Arc<RwLock<MeshState>>,
}

impl GraphQLMesh {
    /// Create a new mesh from config
    pub fn new(config: MeshConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(MeshState::new(config))),
        }
    }

    /// Add a data source
    pub async fn add_source(&self, source: DataSource) -> Result<()> {
        let mut state = self.state.write().await;

        if state.config.sources.iter().any(|s| s.id == source.id) {
            return Err(anyhow!("Source '{}' already exists", source.id));
        }

        state.config.sources.push(source);
        state.merged_schema = None; // Invalidate merged schema

        Ok(())
    }

    /// Remove a data source
    pub async fn remove_source(&self, source_id: &str) -> Result<()> {
        let mut state = self.state.write().await;

        let initial_len = state.config.sources.len();
        state.config.sources.retain(|s| s.id != source_id);

        if state.config.sources.len() == initial_len {
            return Err(anyhow!("Source '{}' not found", source_id));
        }

        state.source_health.remove(source_id);
        state.merged_schema = None;

        Ok(())
    }

    /// Get all sources
    pub async fn get_sources(&self) -> Vec<DataSource> {
        let state = self.state.read().await;
        state.config.sources.clone()
    }

    /// Get source by ID
    pub async fn get_source(&self, source_id: &str) -> Option<DataSource> {
        let state = self.state.read().await;
        state
            .config
            .sources
            .iter()
            .find(|s| s.id == source_id)
            .cloned()
    }

    /// Update source health
    pub async fn update_source_health(
        &self,
        source_id: &str,
        healthy: bool,
        error: Option<String>,
        latency_ms: u64,
    ) {
        let mut state = self.state.write().await;

        state.source_health.insert(
            source_id.to_string(),
            SourceHealth {
                source_id: source_id.to_string(),
                healthy,
                last_check: SystemTime::now(),
                last_error: error,
                latency_ms,
            },
        );
    }

    /// Get source health
    pub async fn get_source_health(&self, source_id: &str) -> Option<SourceHealth> {
        let state = self.state.read().await;
        state.source_health.get(source_id).cloned()
    }

    /// Get all source health
    pub async fn get_all_health(&self) -> Vec<SourceHealth> {
        let state = self.state.read().await;
        state.source_health.values().cloned().collect()
    }

    /// Add type merge configuration
    pub async fn add_type_merge(&self, merge: TypeMergeConfig) {
        let mut state = self.state.write().await;
        state.config.type_merges.push(merge);
        state.merged_schema = None;
    }

    /// Add cross-source relation
    pub async fn add_relation(&self, relation: CrossSourceRelation) {
        let mut state = self.state.write().await;
        state.config.relations.push(relation);
        state.merged_schema = None;
    }

    /// Build merged schema
    pub async fn build_schema(&self) -> Result<String> {
        let mut state = self.state.write().await;

        let mut sdl = String::new();

        // Build Query type from all sources
        sdl.push_str("type Query {\n");

        for source in &state.config.sources {
            if !source.enabled {
                continue;
            }

            // Apply prefix transform if present
            let prefix = source
                .transforms
                .iter()
                .find_map(|t| match t {
                    Transform::Prefix { value } => Some(value.clone()),
                    _ => None,
                })
                .unwrap_or_default();

            sdl.push_str(&format!(
                "  # Source: {} ({})\n",
                source.name,
                source.source_type.name()
            ));
            sdl.push_str(&format!(
                "  {}health: Boolean @source(name: \"{}\")\n",
                prefix.to_lowercase(),
                source.id
            ));
        }

        sdl.push_str("}\n\n");

        // Add directive definition
        sdl.push_str("directive @source(name: String!) on FIELD_DEFINITION\n");

        state.merged_schema = Some(sdl.clone());
        state.last_merge = Some(SystemTime::now());

        Ok(sdl)
    }

    /// Get merged schema
    pub async fn get_merged_schema(&self) -> Option<String> {
        let state = self.state.read().await;
        state.merged_schema.clone()
    }

    /// Get configuration
    pub async fn get_config(&self) -> MeshConfig {
        let state = self.state.read().await;
        state.config.clone()
    }

    /// Validate configuration
    pub async fn validate(&self) -> Vec<ValidationError> {
        let state = self.state.read().await;
        let mut errors = Vec::new();

        // Check for duplicate source IDs
        let mut seen_ids = std::collections::HashSet::new();
        for source in &state.config.sources {
            if !seen_ids.insert(&source.id) {
                errors.push(ValidationError {
                    path: format!("sources.{}", source.id),
                    message: "Duplicate source ID".to_string(),
                });
            }
        }

        // Check type merges reference existing sources
        for merge in &state.config.type_merges {
            for source_id in &merge.sources {
                if !state.config.sources.iter().any(|s| &s.id == source_id) {
                    errors.push(ValidationError {
                        path: format!("type_merges.{}.sources", merge.type_name),
                        message: format!("Unknown source: {}", source_id),
                    });
                }
            }
        }

        // Check relations reference existing sources
        for relation in &state.config.relations {
            if !state
                .config
                .sources
                .iter()
                .any(|s| s.id == relation.source_id)
            {
                errors.push(ValidationError {
                    path: format!("relations.{}.source_id", relation.source_field),
                    message: format!("Unknown source: {}", relation.source_id),
                });
            }
            if !state
                .config
                .sources
                .iter()
                .any(|s| s.id == relation.target_source_id)
            {
                errors.push(ValidationError {
                    path: format!("relations.{}.target_source_id", relation.source_field),
                    message: format!("Unknown source: {}", relation.target_source_id),
                });
            }
        }

        errors
    }

    /// Export configuration as YAML
    pub async fn export_yaml(&self) -> Result<String> {
        let state = self.state.read().await;
        let yaml = serde_json::to_string_pretty(&state.config)?;
        Ok(yaml)
    }
}

impl SourceType {
    /// Get source type name
    pub fn name(&self) -> &'static str {
        match self {
            SourceType::GraphQL => "GraphQL",
            SourceType::REST => "REST",
            SourceType::OpenAPI => "OpenAPI",
            SourceType::SOAP => "SOAP",
            SourceType::GRPC => "gRPC",
            SourceType::OData => "OData",
            SourceType::Database => "Database",
            SourceType::JsonSchema => "JSON Schema",
            SourceType::Custom => "Custom",
        }
    }
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error path
    pub path: String,
    /// Error message
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mesh_creation() {
        let config = MeshConfig::default();
        let mesh = GraphQLMesh::new(config);

        let sources = mesh.get_sources().await;
        assert!(sources.is_empty());
    }

    #[tokio::test]
    async fn test_add_source() {
        let mesh = GraphQLMesh::new(MeshConfig::default());

        let source = DataSource::new(
            "users",
            "Users API",
            SourceType::GraphQL,
            "https://api.example.com/graphql",
        );

        mesh.add_source(source).await.unwrap();

        let sources = mesh.get_sources().await;
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].id, "users");
    }

    #[tokio::test]
    async fn test_duplicate_source() {
        let mesh = GraphQLMesh::new(MeshConfig::default());

        let source1 = DataSource::new(
            "api",
            "API 1",
            SourceType::GraphQL,
            "https://api1.example.com",
        );
        let source2 = DataSource::new(
            "api",
            "API 2",
            SourceType::GraphQL,
            "https://api2.example.com",
        );

        mesh.add_source(source1).await.unwrap();
        let result = mesh.add_source(source2).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_remove_source() {
        let mesh = GraphQLMesh::new(MeshConfig::default());

        let source = DataSource::new(
            "users",
            "Users",
            SourceType::GraphQL,
            "https://api.example.com",
        );
        mesh.add_source(source).await.unwrap();

        mesh.remove_source("users").await.unwrap();

        let sources = mesh.get_sources().await;
        assert!(sources.is_empty());
    }

    #[tokio::test]
    async fn test_source_health() {
        let mesh = GraphQLMesh::new(MeshConfig::default());

        let source = DataSource::new("api", "API", SourceType::GraphQL, "https://api.example.com");
        mesh.add_source(source).await.unwrap();

        mesh.update_source_health("api", true, None, 50).await;

        let health = mesh.get_source_health("api").await.unwrap();
        assert!(health.healthy);
        assert_eq!(health.latency_ms, 50);
    }

    #[tokio::test]
    async fn test_build_schema() {
        let mesh = GraphQLMesh::new(MeshConfig::default());

        let source = DataSource::new(
            "users",
            "Users",
            SourceType::GraphQL,
            "https://api.example.com",
        )
        .with_transform(Transform::Prefix {
            value: "Users_".to_string(),
        });

        mesh.add_source(source).await.unwrap();

        let schema = mesh.build_schema().await.unwrap();
        assert!(schema.contains("type Query"));
        assert!(schema.contains("users_health"));
    }

    #[tokio::test]
    async fn test_validation() {
        let mesh = GraphQLMesh::new(MeshConfig::default());

        mesh.add_type_merge(TypeMergeConfig {
            type_name: "User".to_string(),
            key_field: "id".to_string(),
            sources: vec!["nonexistent".to_string()],
            resolution: MergeResolution::MergeAll,
        })
        .await;

        let errors = mesh.validate().await;
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_data_source_builder() {
        let source = DataSource::new("api", "API", SourceType::REST, "https://api.example.com")
            .with_auth(AuthConfig::Bearer {
                token: "secret".to_string(),
            })
            .with_header("X-Custom", "value")
            .with_cache(CacheConfig::default());

        assert!(source.auth.is_some());
        assert!(source.headers.contains_key("X-Custom"));
        assert!(source.cache.is_some());
    }

    #[test]
    fn test_source_type_name() {
        assert_eq!(SourceType::GraphQL.name(), "GraphQL");
        assert_eq!(SourceType::REST.name(), "REST");
        assert_eq!(SourceType::GRPC.name(), "gRPC");
    }
}
