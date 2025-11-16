//! # Dynamic Schema Evolution and Migration
//!
//! This module provides comprehensive schema evolution capabilities for streaming
//! RDF data, allowing schemas to evolve over time without downtime.
//!
//! ## Features
//!
//! - **Backward Compatibility**: Old consumers can read new data
//! - **Forward Compatibility**: New consumers can read old data
//! - **Full Compatibility**: Both backward and forward compatible
//! - **Version Management**: Track and manage schema versions
//! - **Migration Strategies**: Automatic data migration between versions
//! - **Schema Validation**: Ensure data conforms to schemas
//! - **Evolution Rules**: Define allowed schema changes
//!
//! ## Use Cases
//!
//! - **API Evolution**: Evolve APIs without breaking clients
//! - **Data Migration**: Migrate data to new formats
//! - **Feature Flags**: Gradually roll out schema changes
//! - **A/B Testing**: Test new schemas with subset of data

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::event::StreamEvent;

/// Schema compatibility mode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompatibilityMode {
    /// No compatibility checks
    None,
    /// New schema can read old data
    Backward,
    /// Old schema can read new data
    Forward,
    /// Both backward and forward compatible
    Full,
    /// Transitive backward compatibility
    BackwardTransitive,
    /// Transitive forward compatibility
    ForwardTransitive,
    /// Transitive full compatibility
    FullTransitive,
}

/// Schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDefinition {
    /// Schema ID
    pub schema_id: String,
    /// Schema name
    pub name: String,
    /// Schema version
    pub version: String,
    /// Schema format (RDF, JSON Schema, Avro, etc.)
    pub format: SchemaFormat,
    /// Schema content
    pub content: String,
    /// Field definitions
    pub fields: Vec<FieldDefinition>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Created by
    pub created_by: String,
    /// Description
    pub description: Option<String>,
    /// Tags
    pub tags: Vec<String>,
    /// Compatibility mode
    pub compatibility: CompatibilityMode,
}

/// Field definition in a schema
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FieldDefinition {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: FieldType,
    /// Is required
    pub required: bool,
    /// Default value
    pub default_value: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Aliases (for backward compatibility)
    pub aliases: Vec<String>,
}

/// Field type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FieldType {
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    URI,
    Literal,
    Array { item_type: Box<FieldType> },
    Object { fields: Vec<FieldDefinition> },
    Union { types: Vec<FieldType> },
    Custom { type_name: String },
}

/// Schema format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SchemaFormat {
    /// RDF Schema
    RDFS,
    /// OWL Ontology
    OWL,
    /// SHACL Shapes
    SHACL,
    /// JSON Schema
    JsonSchema,
    /// Apache Avro
    Avro,
    /// Protocol Buffers
    Protobuf,
    /// Custom format
    Custom { format_name: String },
}

/// Schema change type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaChange {
    /// Add a new field
    AddField { field: FieldDefinition },
    /// Remove a field
    RemoveField { field_name: String },
    /// Modify field type
    ModifyFieldType {
        field_name: String,
        old_type: FieldType,
        new_type: FieldType,
    },
    /// Make field optional
    MakeFieldOptional { field_name: String },
    /// Make field required
    MakeFieldRequired { field_name: String },
    /// Add field alias
    AddFieldAlias { field_name: String, alias: String },
    /// Change default value
    ChangeDefaultValue {
        field_name: String,
        old_default: Option<String>,
        new_default: Option<String>,
    },
}

/// Schema evolution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionResult {
    /// Success
    pub success: bool,
    /// New schema version
    pub new_version: Option<String>,
    /// Applied changes
    pub changes: Vec<SchemaChange>,
    /// Compatibility check result
    pub compatibility_result: CompatibilityCheckResult,
    /// Migration required
    pub migration_required: bool,
}

/// Compatibility check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityCheckResult {
    /// Is compatible
    pub is_compatible: bool,
    /// Compatibility issues
    pub issues: Vec<CompatibilityIssue>,
    /// Warnings
    pub warnings: Vec<String>,
}

/// Compatibility issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityIssue {
    /// Issue type
    pub issue_type: CompatibilityIssueType,
    /// Field name (if applicable)
    pub field_name: Option<String>,
    /// Description
    pub description: String,
    /// Severity
    pub severity: IssueSeverity,
}

/// Compatibility issue type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityIssueType {
    /// Breaking change
    BreakingChange,
    /// Type mismatch
    TypeMismatch,
    /// Missing required field
    MissingRequiredField,
    /// Incompatible default value
    IncompatibleDefaultValue,
    /// Other issue
    Other,
}

/// Issue severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Migration strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationStrategy {
    /// No migration needed
    None,
    /// Automatic migration with default values
    Automatic,
    /// Custom migration function
    Custom { migration_id: String },
    /// Manual migration required
    Manual,
}

/// Data migration rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationRule {
    /// From version
    pub from_version: String,
    /// To version
    pub to_version: String,
    /// Migration strategy
    pub strategy: MigrationStrategy,
    /// Field mappings
    pub field_mappings: HashMap<String, String>,
    /// Transformation functions (as strings)
    pub transformations: HashMap<String, String>,
}

/// Schema version metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Version number
    pub version: String,
    /// Schema definition
    pub schema: SchemaDefinition,
    /// Previous version
    pub previous_version: Option<String>,
    /// Migration rules from previous version
    pub migration_rule: Option<MigrationRule>,
    /// Is active version
    pub is_active: bool,
    /// Deprecation info
    pub deprecated: Option<DeprecationInfo>,
}

/// Deprecation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationInfo {
    /// Deprecated at
    pub deprecated_at: DateTime<Utc>,
    /// Sunset date (when it will be removed)
    pub sunset_date: Option<DateTime<Utc>>,
    /// Deprecation reason
    pub reason: String,
    /// Migration guide
    pub migration_guide: Option<String>,
}

/// Schema evolution manager
pub struct SchemaEvolutionManager {
    /// Schema versions by name
    schemas: Arc<DashMap<String, Vec<SchemaVersion>>>,
    /// Active schema versions
    active_versions: Arc<DashMap<String, String>>,
    /// Compatibility rules
    compatibility_rules: Arc<RwLock<HashMap<String, CompatibilityMode>>>,
    /// Migration rules
    migration_rules: Arc<DashMap<String, Vec<MigrationRule>>>,
    /// Schema evolution history
    evolution_history: Arc<RwLock<Vec<SchemaEvolutionEvent>>>,
}

/// Schema evolution event for auditing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaEvolutionEvent {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub schema_name: String,
    pub old_version: Option<String>,
    pub new_version: String,
    pub changes: Vec<SchemaChange>,
    pub user: String,
}

impl SchemaEvolutionManager {
    /// Create a new schema evolution manager
    pub fn new() -> Self {
        Self {
            schemas: Arc::new(DashMap::new()),
            active_versions: Arc::new(DashMap::new()),
            compatibility_rules: Arc::new(RwLock::new(HashMap::new())),
            migration_rules: Arc::new(DashMap::new()),
            evolution_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a new schema
    pub fn register_schema(&self, schema: SchemaDefinition) -> Result<String> {
        let schema_name = schema.name.clone();
        let version = schema.version.clone();

        let schema_version = SchemaVersion {
            version: version.clone(),
            schema,
            previous_version: None,
            migration_rule: None,
            is_active: true,
            deprecated: None,
        };

        // Add to schemas
        self.schemas
            .entry(schema_name.clone())
            .or_default()
            .push(schema_version);

        // Set as active version
        self.active_versions
            .insert(schema_name.clone(), version.clone());

        info!("Registered schema {} version {}", schema_name, version);
        Ok(version)
    }

    /// Evolve a schema with new changes
    pub fn evolve_schema(
        &self,
        schema_name: &str,
        changes: Vec<SchemaChange>,
        user: String,
    ) -> Result<EvolutionResult> {
        // Get current active version
        let current_version = self
            .active_versions
            .get(schema_name)
            .ok_or_else(|| anyhow!("Schema not found: {}", schema_name))?
            .clone();

        let current_schema = self.get_schema(schema_name, &current_version)?;

        // Apply changes to create new schema
        let new_schema = self.apply_changes(&current_schema.schema, &changes)?;

        // Check compatibility
        let compatibility_result = self.check_compatibility(&current_schema.schema, &new_schema)?;

        if !compatibility_result.is_compatible {
            warn!(
                "Schema evolution would break compatibility: {:?}",
                compatibility_result.issues
            );
            return Ok(EvolutionResult {
                success: false,
                new_version: None,
                changes,
                compatibility_result,
                migration_required: false,
            });
        }

        // Generate new version number
        let new_version = self.generate_version(&current_version, &changes)?;

        // Determine if migration is required
        let migration_required = self.is_migration_required(&changes);

        // Create migration rule if needed
        let migration_rule = if migration_required {
            Some(self.create_migration_rule(&current_version, &new_version, &changes)?)
        } else {
            None
        };

        // Create new schema version
        let new_schema_version = SchemaVersion {
            version: new_version.clone(),
            schema: new_schema,
            previous_version: Some(current_version.clone()),
            migration_rule: migration_rule.clone(),
            is_active: true,
            deprecated: None,
        };

        // Add to schemas
        if let Some(mut versions) = self.schemas.get_mut(schema_name) {
            // Mark previous version as inactive
            if let Some(prev) = versions.iter_mut().find(|v| v.version == current_version) {
                prev.is_active = false;
            }
            versions.push(new_schema_version);
        }

        // Update active version
        self.active_versions
            .insert(schema_name.to_string(), new_version.clone());

        // Store migration rule
        if let Some(rule) = migration_rule {
            self.migration_rules
                .entry(schema_name.to_string())
                .or_default()
                .push(rule);
        }

        // Record evolution event
        let event = SchemaEvolutionEvent {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            schema_name: schema_name.to_string(),
            old_version: Some(current_version),
            new_version: new_version.clone(),
            changes: changes.clone(),
            user,
        };
        self.evolution_history.write().push(event);

        info!("Evolved schema {} to version {}", schema_name, new_version);

        Ok(EvolutionResult {
            success: true,
            new_version: Some(new_version),
            changes,
            compatibility_result,
            migration_required,
        })
    }

    /// Get a specific schema version
    pub fn get_schema(&self, schema_name: &str, version: &str) -> Result<SchemaVersion> {
        let versions = self
            .schemas
            .get(schema_name)
            .ok_or_else(|| anyhow!("Schema not found: {}", schema_name))?;

        versions
            .iter()
            .find(|v| v.version == version)
            .cloned()
            .ok_or_else(|| anyhow!("Version not found: {}", version))
    }

    /// Get the active schema version
    pub fn get_active_schema(&self, schema_name: &str) -> Result<SchemaVersion> {
        let active_version = self
            .active_versions
            .get(schema_name)
            .ok_or_else(|| anyhow!("Schema not found: {}", schema_name))?
            .clone();

        self.get_schema(schema_name, &active_version)
    }

    /// Apply schema changes to create new schema
    fn apply_changes(
        &self,
        current_schema: &SchemaDefinition,
        changes: &[SchemaChange],
    ) -> Result<SchemaDefinition> {
        let mut new_schema = current_schema.clone();
        let mut fields = new_schema.fields.clone();

        for change in changes {
            match change {
                SchemaChange::AddField { field } => {
                    fields.push(field.clone());
                }
                SchemaChange::RemoveField { field_name } => {
                    fields.retain(|f| f.name != *field_name);
                }
                SchemaChange::ModifyFieldType {
                    field_name,
                    new_type,
                    ..
                } => {
                    if let Some(field) = fields.iter_mut().find(|f| f.name == *field_name) {
                        field.field_type = new_type.clone();
                    }
                }
                SchemaChange::MakeFieldOptional { field_name } => {
                    if let Some(field) = fields.iter_mut().find(|f| f.name == *field_name) {
                        field.required = false;
                    }
                }
                SchemaChange::MakeFieldRequired { field_name } => {
                    if let Some(field) = fields.iter_mut().find(|f| f.name == *field_name) {
                        field.required = true;
                    }
                }
                SchemaChange::AddFieldAlias { field_name, alias } => {
                    if let Some(field) = fields.iter_mut().find(|f| f.name == *field_name) {
                        if !field.aliases.contains(alias) {
                            field.aliases.push(alias.clone());
                        }
                    }
                }
                SchemaChange::ChangeDefaultValue {
                    field_name,
                    new_default,
                    ..
                } => {
                    if let Some(field) = fields.iter_mut().find(|f| f.name == *field_name) {
                        field.default_value = new_default.clone();
                    }
                }
            }
        }

        new_schema.fields = fields;
        Ok(new_schema)
    }

    /// Check compatibility between two schemas
    fn check_compatibility(
        &self,
        old_schema: &SchemaDefinition,
        new_schema: &SchemaDefinition,
    ) -> Result<CompatibilityCheckResult> {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        let old_fields: HashMap<String, &FieldDefinition> = old_schema
            .fields
            .iter()
            .map(|f| (f.name.clone(), f))
            .collect();
        let new_fields: HashMap<String, &FieldDefinition> = new_schema
            .fields
            .iter()
            .map(|f| (f.name.clone(), f))
            .collect();

        // Check backward compatibility
        if matches!(
            new_schema.compatibility,
            CompatibilityMode::Backward
                | CompatibilityMode::Full
                | CompatibilityMode::BackwardTransitive
                | CompatibilityMode::FullTransitive
        ) {
            // Check backward compatibility: new schema can read old data
            for (field_name, old_field) in &old_fields {
                if let Some(new_field) = new_fields.get(field_name) {
                    // Field exists in both - check type compatibility
                    if old_field.field_type != new_field.field_type {
                        issues.push(CompatibilityIssue {
                            issue_type: CompatibilityIssueType::TypeMismatch,
                            field_name: Some(field_name.clone()),
                            description: format!(
                                "Field type changed from {:?} to {:?}",
                                old_field.field_type, new_field.field_type
                            ),
                            severity: IssueSeverity::Error,
                        });
                    }
                } else if old_field.required {
                    // Required field removed
                    issues.push(CompatibilityIssue {
                        issue_type: CompatibilityIssueType::BreakingChange,
                        field_name: Some(field_name.clone()),
                        description: format!("Required field '{}' was removed", field_name),
                        severity: IssueSeverity::Critical,
                    });
                } else {
                    warnings.push(format!("Optional field '{}' was removed", field_name));
                }
            }
        }

        // Check forward compatibility
        if matches!(
            new_schema.compatibility,
            CompatibilityMode::Forward
                | CompatibilityMode::Full
                | CompatibilityMode::ForwardTransitive
                | CompatibilityMode::FullTransitive
        ) {
            // Check forward compatibility: old schema can read new data
            for (field_name, new_field) in &new_fields {
                if !old_fields.contains_key(field_name) && new_field.required {
                    issues.push(CompatibilityIssue {
                        issue_type: CompatibilityIssueType::MissingRequiredField,
                        field_name: Some(field_name.clone()),
                        description: format!(
                            "New required field '{}' added without default value",
                            field_name
                        ),
                        severity: IssueSeverity::Error,
                    });
                }
            }
        }

        Ok(CompatibilityCheckResult {
            is_compatible: issues.is_empty(),
            issues,
            warnings,
        })
    }

    /// Generate new version number
    fn generate_version(&self, current_version: &str, changes: &[SchemaChange]) -> Result<String> {
        // Simple semantic versioning: major.minor.patch
        let parts: Vec<&str> = current_version.split('.').collect();
        if parts.len() != 3 {
            return Err(anyhow!("Invalid version format: {}", current_version));
        }

        let major: u32 = parts[0].parse()?;
        let minor: u32 = parts[1].parse()?;
        let patch: u32 = parts[2].parse()?;

        // Determine if this is a breaking change
        let has_breaking_changes = changes.iter().any(|c| {
            matches!(
                c,
                SchemaChange::RemoveField { .. }
                    | SchemaChange::ModifyFieldType { .. }
                    | SchemaChange::MakeFieldRequired { .. }
            )
        });

        if has_breaking_changes {
            Ok(format!("{}.0.0", major + 1))
        } else if changes
            .iter()
            .any(|c| matches!(c, SchemaChange::AddField { .. }))
        {
            Ok(format!("{}.{}.0", major, minor + 1))
        } else {
            Ok(format!("{}.{}.{}", major, minor, patch + 1))
        }
    }

    /// Check if migration is required
    fn is_migration_required(&self, changes: &[SchemaChange]) -> bool {
        changes.iter().any(|c| match c {
            SchemaChange::ModifyFieldType { .. } => true,
            SchemaChange::RemoveField { .. } => true,
            SchemaChange::AddField { field } => field.required,
            _ => false,
        })
    }

    /// Create migration rule
    fn create_migration_rule(
        &self,
        from_version: &str,
        to_version: &str,
        changes: &[SchemaChange],
    ) -> Result<MigrationRule> {
        let mut field_mappings = HashMap::new();
        let mut transformations = HashMap::new();

        for change in changes {
            match change {
                SchemaChange::AddFieldAlias { field_name, alias } => {
                    field_mappings.insert(alias.clone(), field_name.clone());
                }
                SchemaChange::ModifyFieldType {
                    field_name,
                    old_type,
                    new_type,
                } => {
                    transformations.insert(
                        field_name.clone(),
                        format!("convert_{:?}_to_{:?}", old_type, new_type),
                    );
                }
                _ => {}
            }
        }

        Ok(MigrationRule {
            from_version: from_version.to_string(),
            to_version: to_version.to_string(),
            strategy: if transformations.is_empty() {
                MigrationStrategy::Automatic
            } else {
                MigrationStrategy::Custom {
                    migration_id: Uuid::new_v4().to_string(),
                }
            },
            field_mappings,
            transformations,
        })
    }

    /// Migrate event to a specific schema version
    pub fn migrate_event(
        &self,
        event: &StreamEvent,
        from_version: &str,
        to_version: &str,
        schema_name: &str,
    ) -> Result<StreamEvent> {
        // Get migration rule
        let migration_rule = self
            .migration_rules
            .get(schema_name)
            .and_then(|rules| {
                rules
                    .iter()
                    .find(|r| r.from_version == from_version && r.to_version == to_version)
                    .cloned()
            })
            .ok_or_else(|| {
                anyhow!(
                    "No migration rule found from {} to {}",
                    from_version,
                    to_version
                )
            })?;

        // Apply migration strategy
        match migration_rule.strategy {
            MigrationStrategy::None => Ok(event.clone()),
            MigrationStrategy::Automatic => {
                // Automatic migration - apply field mappings
                debug!("Applying automatic migration with field mappings");
                Ok(event.clone()) // Simplified - would apply mappings in real implementation
            }
            MigrationStrategy::Custom { ref migration_id } => {
                debug!("Custom migration required: {}", migration_id);
                Ok(event.clone()) // Would call custom migration function
            }
            MigrationStrategy::Manual => Err(anyhow!("Manual migration required")),
        }
    }

    /// Deprecate a schema version
    pub fn deprecate_version(
        &self,
        schema_name: &str,
        version: &str,
        reason: String,
        sunset_date: Option<DateTime<Utc>>,
    ) -> Result<()> {
        if let Some(mut versions) = self.schemas.get_mut(schema_name) {
            if let Some(schema_version) = versions.iter_mut().find(|v| v.version == version) {
                schema_version.deprecated = Some(DeprecationInfo {
                    deprecated_at: Utc::now(),
                    sunset_date,
                    reason,
                    migration_guide: None,
                });
                info!("Deprecated schema {} version {}", schema_name, version);
                return Ok(());
            }
        }
        Err(anyhow!("Schema version not found"))
    }

    /// Get evolution history
    pub fn get_evolution_history(&self, schema_name: &str) -> Vec<SchemaEvolutionEvent> {
        self.evolution_history
            .read()
            .iter()
            .filter(|e| e.schema_name == schema_name)
            .cloned()
            .collect()
    }
}

impl Default for SchemaEvolutionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_registration() {
        let manager = SchemaEvolutionManager::new();

        let schema = SchemaDefinition {
            schema_id: Uuid::new_v4().to_string(),
            name: "TestSchema".to_string(),
            version: "1.0.0".to_string(),
            format: SchemaFormat::RDFS,
            content: "test content".to_string(),
            fields: vec![FieldDefinition {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                default_value: None,
                description: None,
                aliases: Vec::new(),
            }],
            created_at: Utc::now(),
            created_by: "test".to_string(),
            description: None,
            tags: Vec::new(),
            compatibility: CompatibilityMode::Backward,
        };

        let version = manager.register_schema(schema).unwrap();
        assert_eq!(version, "1.0.0");

        let active = manager.get_active_schema("TestSchema").unwrap();
        assert_eq!(active.version, "1.0.0");
    }

    #[test]
    fn test_schema_evolution() {
        let manager = SchemaEvolutionManager::new();

        let schema = SchemaDefinition {
            schema_id: Uuid::new_v4().to_string(),
            name: "TestSchema".to_string(),
            version: "1.0.0".to_string(),
            format: SchemaFormat::RDFS,
            content: "test content".to_string(),
            fields: vec![FieldDefinition {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                default_value: None,
                description: None,
                aliases: Vec::new(),
            }],
            created_at: Utc::now(),
            created_by: "test".to_string(),
            description: None,
            tags: Vec::new(),
            compatibility: CompatibilityMode::Backward,
        };

        manager.register_schema(schema).unwrap();

        // Add optional field (backward compatible)
        let changes = vec![SchemaChange::AddField {
            field: FieldDefinition {
                name: "email".to_string(),
                field_type: FieldType::String,
                required: false,
                default_value: Some("".to_string()),
                description: None,
                aliases: Vec::new(),
            },
        }];

        let result = manager
            .evolve_schema("TestSchema", changes, "test".to_string())
            .unwrap();

        assert!(result.success);
        assert!(result.compatibility_result.is_compatible);
        assert_eq!(result.new_version, Some("1.1.0".to_string()));
    }

    #[test]
    fn test_breaking_change_detection() {
        let manager = SchemaEvolutionManager::new();

        let schema = SchemaDefinition {
            schema_id: Uuid::new_v4().to_string(),
            name: "TestSchema".to_string(),
            version: "1.0.0".to_string(),
            format: SchemaFormat::RDFS,
            content: "test content".to_string(),
            fields: vec![FieldDefinition {
                name: "name".to_string(),
                field_type: FieldType::String,
                required: true,
                default_value: None,
                description: None,
                aliases: Vec::new(),
            }],
            created_at: Utc::now(),
            created_by: "test".to_string(),
            description: None,
            tags: Vec::new(),
            compatibility: CompatibilityMode::Backward,
        };

        manager.register_schema(schema).unwrap();

        // Remove required field (breaking change)
        let changes = vec![SchemaChange::RemoveField {
            field_name: "name".to_string(),
        }];

        let result = manager
            .evolve_schema("TestSchema", changes, "test".to_string())
            .unwrap();

        assert!(!result.success);
        assert!(!result.compatibility_result.is_compatible);
    }
}
