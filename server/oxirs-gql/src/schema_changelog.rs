//! Schema Changelog Generation
//!
//! This module provides automatic changelog generation for GraphQL schema changes,
//! tracking additions, modifications, and deprecations over time.
//!
//! ## Features
//!
//! - **Change Detection**: Detect additions, removals, and modifications
//! - **Breaking Change Analysis**: Identify breaking vs non-breaking changes
//! - **Version Tracking**: Track schema versions over time
//! - **Changelog Export**: Generate changelogs in various formats
//! - **Diff Generation**: Visual diffs between schema versions

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

/// Change type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeType {
    /// Type added
    TypeAdded,
    /// Type removed
    TypeRemoved,
    /// Type modified
    TypeModified,
    /// Field added
    FieldAdded,
    /// Field removed
    FieldRemoved,
    /// Field modified
    FieldModified,
    /// Argument added
    ArgumentAdded,
    /// Argument removed
    ArgumentRemoved,
    /// Directive added
    DirectiveAdded,
    /// Directive removed
    DirectiveRemoved,
    /// Enum value added
    EnumValueAdded,
    /// Enum value removed
    EnumValueRemoved,
    /// Input field added
    InputFieldAdded,
    /// Input field removed
    InputFieldRemoved,
    /// Description changed
    DescriptionChanged,
    /// Deprecation added
    DeprecationAdded,
    /// Deprecation removed
    DeprecationRemoved,
}

impl ChangeType {
    /// Check if this change type is breaking
    pub fn is_breaking(&self) -> bool {
        matches!(
            self,
            ChangeType::TypeRemoved
                | ChangeType::FieldRemoved
                | ChangeType::ArgumentRemoved
                | ChangeType::EnumValueRemoved
                | ChangeType::InputFieldRemoved
        )
    }

    /// Get severity level
    pub fn severity(&self) -> ChangeSeverity {
        if self.is_breaking() {
            ChangeSeverity::Breaking
        } else {
            match self {
                ChangeType::DeprecationAdded => ChangeSeverity::Dangerous,
                ChangeType::FieldModified | ChangeType::TypeModified => ChangeSeverity::Dangerous,
                _ => ChangeSeverity::NonBreaking,
            }
        }
    }
}

/// Change severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeSeverity {
    /// Breaking change - will break existing clients
    Breaking,
    /// Dangerous change - may cause issues
    Dangerous,
    /// Non-breaking change - safe to deploy
    NonBreaking,
}

/// Schema change entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaChange {
    /// Change ID
    pub id: String,
    /// Change type
    pub change_type: ChangeType,
    /// Affected path (e.g., "User.email")
    pub path: String,
    /// Previous value (if applicable)
    pub previous_value: Option<String>,
    /// New value (if applicable)
    pub new_value: Option<String>,
    /// Description of the change
    pub description: String,
    /// Whether this is a breaking change
    pub is_breaking: bool,
    /// Migration guide (if applicable)
    pub migration_guide: Option<String>,
}

impl SchemaChange {
    /// Create a new schema change
    pub fn new(change_type: ChangeType, path: &str, description: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            change_type,
            path: path.to_string(),
            previous_value: None,
            new_value: None,
            description: description.to_string(),
            is_breaking: change_type.is_breaking(),
            migration_guide: None,
        }
    }

    /// Set previous value
    pub fn with_previous(mut self, value: &str) -> Self {
        self.previous_value = Some(value.to_string());
        self
    }

    /// Set new value
    pub fn with_new(mut self, value: &str) -> Self {
        self.new_value = Some(value.to_string());
        self
    }

    /// Set migration guide
    pub fn with_migration(mut self, guide: &str) -> Self {
        self.migration_guide = Some(guide.to_string());
        self
    }
}

/// Schema version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Version identifier
    pub version: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Schema hash
    pub schema_hash: String,
    /// Changes from previous version
    pub changes: Vec<SchemaChange>,
    /// Number of breaking changes
    pub breaking_changes_count: usize,
    /// Author (if available)
    pub author: Option<String>,
    /// Commit hash (if available)
    pub commit_hash: Option<String>,
    /// Release notes
    pub release_notes: Option<String>,
}

impl SchemaVersion {
    /// Create a new schema version
    pub fn new(version: &str, schema_hash: &str) -> Self {
        Self {
            version: version.to_string(),
            timestamp: SystemTime::now(),
            schema_hash: schema_hash.to_string(),
            changes: Vec::new(),
            breaking_changes_count: 0,
            author: None,
            commit_hash: None,
            release_notes: None,
        }
    }

    /// Add a change
    pub fn add_change(&mut self, change: SchemaChange) {
        if change.is_breaking {
            self.breaking_changes_count += 1;
        }
        self.changes.push(change);
    }
}

/// Simple type definition for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDef {
    /// Type name
    pub name: String,
    /// Type kind
    pub kind: String,
    /// Description
    pub description: Option<String>,
    /// Fields
    pub fields: HashMap<String, FieldDef>,
    /// Enum values
    pub enum_values: Vec<String>,
    /// Implements interfaces
    pub implements: Vec<String>,
    /// Is deprecated
    pub deprecated: bool,
}

/// Simple field definition for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: String,
    /// Description
    pub description: Option<String>,
    /// Arguments
    pub arguments: HashMap<String, String>,
    /// Is deprecated
    pub deprecated: bool,
    /// Deprecation reason
    pub deprecation_reason: Option<String>,
}

/// Schema snapshot for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaSnapshot {
    /// Types
    pub types: HashMap<String, TypeDef>,
    /// Hash of the schema
    pub hash: String,
    /// Timestamp
    pub timestamp: SystemTime,
}

impl SchemaSnapshot {
    /// Create a new snapshot
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            hash: String::new(),
            timestamp: SystemTime::now(),
        }
    }

    /// Add a type
    pub fn add_type(&mut self, type_def: TypeDef) {
        self.types.insert(type_def.name.clone(), type_def);
    }

    /// Compute hash
    pub fn compute_hash(&mut self) {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();

        let mut type_names: Vec<_> = self.types.keys().collect();
        type_names.sort();

        for name in type_names {
            hasher.update(name.as_bytes());
            if let Some(t) = self.types.get(name) {
                hasher.update(t.kind.as_bytes());
                let mut field_names: Vec<_> = t.fields.keys().collect();
                field_names.sort();
                for f in field_names {
                    hasher.update(f.as_bytes());
                    if let Some(field) = t.fields.get(f) {
                        hasher.update(field.field_type.as_bytes());
                    }
                }
            }
        }

        self.hash = hex::encode(hasher.finalize());
    }
}

impl Default for SchemaSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

/// Changelog entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangelogEntry {
    /// Version
    pub version: String,
    /// Date
    pub date: String,
    /// Added items
    pub added: Vec<String>,
    /// Changed items
    pub changed: Vec<String>,
    /// Deprecated items
    pub deprecated: Vec<String>,
    /// Removed items
    pub removed: Vec<String>,
    /// Breaking changes
    pub breaking: Vec<String>,
}

/// Internal state
struct ChangelogState {
    /// Version history
    versions: Vec<SchemaVersion>,
    /// Current snapshot
    current_snapshot: Option<SchemaSnapshot>,
    /// Previous snapshot
    previous_snapshot: Option<SchemaSnapshot>,
}

impl ChangelogState {
    fn new() -> Self {
        Self {
            versions: Vec::new(),
            current_snapshot: None,
            previous_snapshot: None,
        }
    }
}

/// Schema Changelog Generator
///
/// Tracks schema changes and generates changelogs.
pub struct SchemaChangelogGenerator {
    /// Internal state
    state: Arc<RwLock<ChangelogState>>,
}

impl SchemaChangelogGenerator {
    /// Create a new changelog generator
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(ChangelogState::new())),
        }
    }

    /// Set current schema snapshot
    pub async fn set_snapshot(&self, snapshot: SchemaSnapshot) {
        let mut state = self.state.write().await;

        // Move current to previous
        if state.current_snapshot.is_some() {
            state.previous_snapshot = state.current_snapshot.take();
        }

        state.current_snapshot = Some(snapshot);
    }

    /// Compare two snapshots and generate changes
    pub fn compare_snapshots(old: &SchemaSnapshot, new: &SchemaSnapshot) -> Vec<SchemaChange> {
        let mut changes = Vec::new();

        let old_types: HashSet<_> = old.types.keys().collect();
        let new_types: HashSet<_> = new.types.keys().collect();

        // Removed types
        for name in old_types.difference(&new_types) {
            changes.push(SchemaChange::new(
                ChangeType::TypeRemoved,
                name,
                &format!("Type '{}' was removed", name),
            ));
        }

        // Added types
        for name in new_types.difference(&old_types) {
            changes.push(SchemaChange::new(
                ChangeType::TypeAdded,
                name,
                &format!("Type '{}' was added", name),
            ));
        }

        // Modified types
        for name in old_types.intersection(&new_types) {
            let old_type = &old.types[*name];
            let new_type = &new.types[*name];

            // Check fields
            let old_fields: HashSet<_> = old_type.fields.keys().collect();
            let new_fields: HashSet<_> = new_type.fields.keys().collect();

            for field_name in old_fields.difference(&new_fields) {
                changes.push(SchemaChange::new(
                    ChangeType::FieldRemoved,
                    &format!("{}.{}", name, field_name),
                    &format!("Field '{}.{}' was removed", name, field_name),
                ));
            }

            for field_name in new_fields.difference(&old_fields) {
                changes.push(SchemaChange::new(
                    ChangeType::FieldAdded,
                    &format!("{}.{}", name, field_name),
                    &format!("Field '{}.{}' was added", name, field_name),
                ));
            }

            for field_name in old_fields.intersection(&new_fields) {
                let old_field = &old_type.fields[*field_name];
                let new_field = &new_type.fields[*field_name];

                if old_field.field_type != new_field.field_type {
                    changes.push(
                        SchemaChange::new(
                            ChangeType::FieldModified,
                            &format!("{}.{}", name, field_name),
                            &format!(
                                "Field '{}.{}' type changed from '{}' to '{}'",
                                name, field_name, old_field.field_type, new_field.field_type
                            ),
                        )
                        .with_previous(&old_field.field_type)
                        .with_new(&new_field.field_type),
                    );
                }

                if !old_field.deprecated && new_field.deprecated {
                    changes.push(SchemaChange::new(
                        ChangeType::DeprecationAdded,
                        &format!("{}.{}", name, field_name),
                        &format!(
                            "Field '{}.{}' was deprecated: {}",
                            name,
                            field_name,
                            new_field
                                .deprecation_reason
                                .as_deref()
                                .unwrap_or("No reason provided")
                        ),
                    ));
                }

                // Check arguments
                let old_args: HashSet<_> = old_field.arguments.keys().collect();
                let new_args: HashSet<_> = new_field.arguments.keys().collect();

                for arg_name in old_args.difference(&new_args) {
                    changes.push(SchemaChange::new(
                        ChangeType::ArgumentRemoved,
                        &format!("{}.{}({})", name, field_name, arg_name),
                        &format!(
                            "Argument '{}' was removed from '{}.{}'",
                            arg_name, name, field_name
                        ),
                    ));
                }

                for arg_name in new_args.difference(&old_args) {
                    changes.push(SchemaChange::new(
                        ChangeType::ArgumentAdded,
                        &format!("{}.{}({})", name, field_name, arg_name),
                        &format!(
                            "Argument '{}' was added to '{}.{}'",
                            arg_name, name, field_name
                        ),
                    ));
                }
            }

            // Check enum values
            if old_type.kind == "ENUM" {
                let old_values: HashSet<_> = old_type.enum_values.iter().collect();
                let new_values: HashSet<_> = new_type.enum_values.iter().collect();

                for value in old_values.difference(&new_values) {
                    changes.push(SchemaChange::new(
                        ChangeType::EnumValueRemoved,
                        &format!("{}.{}", name, value),
                        &format!("Enum value '{}.{}' was removed", name, value),
                    ));
                }

                for value in new_values.difference(&old_values) {
                    changes.push(SchemaChange::new(
                        ChangeType::EnumValueAdded,
                        &format!("{}.{}", name, value),
                        &format!("Enum value '{}.{}' was added", name, value),
                    ));
                }
            }
        }

        changes
    }

    /// Generate a new version
    pub async fn generate_version(&self, version: &str) -> Result<SchemaVersion> {
        let mut state = self.state.write().await;

        let current = state
            .current_snapshot
            .as_ref()
            .ok_or_else(|| anyhow!("No current schema snapshot"))?;

        let mut schema_version = SchemaVersion::new(version, &current.hash);

        // Compare with previous if available
        if let Some(previous) = &state.previous_snapshot {
            let changes = Self::compare_snapshots(previous, current);
            for change in changes {
                schema_version.add_change(change);
            }
        }

        state.versions.push(schema_version.clone());

        Ok(schema_version)
    }

    /// Get all versions
    pub async fn get_versions(&self) -> Vec<SchemaVersion> {
        let state = self.state.read().await;
        state.versions.clone()
    }

    /// Get version by identifier
    pub async fn get_version(&self, version: &str) -> Option<SchemaVersion> {
        let state = self.state.read().await;
        state
            .versions
            .iter()
            .find(|v| v.version == version)
            .cloned()
    }

    /// Generate markdown changelog
    pub async fn generate_markdown(&self) -> String {
        let state = self.state.read().await;
        let mut md = String::new();

        md.push_str("# Changelog\n\n");
        md.push_str("All notable changes to the GraphQL schema are documented here.\n\n");

        for version in state.versions.iter().rev() {
            md.push_str(&format!("## [{}]\n\n", version.version));

            let date = version
                .timestamp
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| {
                    let secs = d.as_secs();
                    let days = secs / 86400;
                    format!("{} days since epoch", days)
                })
                .unwrap_or_else(|_| "Unknown".to_string());

            md.push_str(&format!("Released: {}\n\n", date));

            if version.breaking_changes_count > 0 {
                md.push_str(&format!(
                    "⚠️ **{} Breaking Change(s)**\n\n",
                    version.breaking_changes_count
                ));
            }

            // Group changes by type
            let mut added = Vec::new();
            let mut changed = Vec::new();
            let mut deprecated = Vec::new();
            let mut removed = Vec::new();

            for change in &version.changes {
                let entry = format!("- `{}`: {}", change.path, change.description);
                match change.change_type {
                    ChangeType::TypeAdded
                    | ChangeType::FieldAdded
                    | ChangeType::EnumValueAdded
                    | ChangeType::ArgumentAdded => added.push(entry),
                    ChangeType::TypeRemoved
                    | ChangeType::FieldRemoved
                    | ChangeType::EnumValueRemoved
                    | ChangeType::ArgumentRemoved => removed.push(entry),
                    ChangeType::DeprecationAdded => deprecated.push(entry),
                    _ => changed.push(entry),
                }
            }

            if !added.is_empty() {
                md.push_str("### Added\n\n");
                for item in &added {
                    md.push_str(&format!("{}\n", item));
                }
                md.push('\n');
            }

            if !changed.is_empty() {
                md.push_str("### Changed\n\n");
                for item in &changed {
                    md.push_str(&format!("{}\n", item));
                }
                md.push('\n');
            }

            if !deprecated.is_empty() {
                md.push_str("### Deprecated\n\n");
                for item in &deprecated {
                    md.push_str(&format!("{}\n", item));
                }
                md.push('\n');
            }

            if !removed.is_empty() {
                md.push_str("### Removed\n\n");
                for item in &removed {
                    md.push_str(&format!("{}\n", item));
                }
                md.push('\n');
            }

            md.push_str("---\n\n");
        }

        md
    }

    /// Generate JSON changelog
    pub async fn generate_json(&self) -> String {
        let state = self.state.read().await;
        serde_json::to_string_pretty(&state.versions).unwrap_or_default()
    }

    /// Check for breaking changes between current and previous
    pub async fn has_breaking_changes(&self) -> bool {
        let state = self.state.read().await;

        let (current, previous) = match (&state.current_snapshot, &state.previous_snapshot) {
            (Some(c), Some(p)) => (c, p),
            _ => return false,
        };

        let changes = Self::compare_snapshots(previous, current);
        changes.iter().any(|c| c.is_breaking)
    }

    /// Get breaking changes summary
    pub async fn get_breaking_changes(&self) -> Vec<SchemaChange> {
        let state = self.state.read().await;

        let (current, previous) = match (&state.current_snapshot, &state.previous_snapshot) {
            (Some(c), Some(p)) => (c, p),
            _ => return Vec::new(),
        };

        Self::compare_snapshots(previous, current)
            .into_iter()
            .filter(|c| c.is_breaking)
            .collect()
    }
}

impl Default for SchemaChangelogGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_snapshot() -> SchemaSnapshot {
        let mut snapshot = SchemaSnapshot::new();

        let mut user_type = TypeDef {
            name: "User".to_string(),
            kind: "OBJECT".to_string(),
            description: None,
            fields: HashMap::new(),
            enum_values: Vec::new(),
            implements: Vec::new(),
            deprecated: false,
        };

        user_type.fields.insert(
            "id".to_string(),
            FieldDef {
                name: "id".to_string(),
                field_type: "ID!".to_string(),
                description: None,
                arguments: HashMap::new(),
                deprecated: false,
                deprecation_reason: None,
            },
        );

        user_type.fields.insert(
            "name".to_string(),
            FieldDef {
                name: "name".to_string(),
                field_type: "String".to_string(),
                description: None,
                arguments: HashMap::new(),
                deprecated: false,
                deprecation_reason: None,
            },
        );

        snapshot.add_type(user_type);
        snapshot.compute_hash();
        snapshot
    }

    #[tokio::test]
    async fn test_changelog_creation() {
        let generator = SchemaChangelogGenerator::new();
        let versions = generator.get_versions().await;
        assert!(versions.is_empty());
    }

    #[tokio::test]
    async fn test_set_snapshot() {
        let generator = SchemaChangelogGenerator::new();
        let snapshot = create_test_snapshot();

        generator.set_snapshot(snapshot).await;

        let state = generator.state.read().await;
        assert!(state.current_snapshot.is_some());
    }

    #[tokio::test]
    async fn test_compare_snapshots_field_added() {
        let old = create_test_snapshot();

        let mut new = create_test_snapshot();
        if let Some(user) = new.types.get_mut("User") {
            user.fields.insert(
                "email".to_string(),
                FieldDef {
                    name: "email".to_string(),
                    field_type: "String".to_string(),
                    description: None,
                    arguments: HashMap::new(),
                    deprecated: false,
                    deprecation_reason: None,
                },
            );
        }

        let changes = SchemaChangelogGenerator::compare_snapshots(&old, &new);

        assert!(!changes.is_empty());
        assert!(changes
            .iter()
            .any(|c| c.change_type == ChangeType::FieldAdded));
    }

    #[tokio::test]
    async fn test_compare_snapshots_field_removed() {
        let old = create_test_snapshot();

        let mut new = create_test_snapshot();
        if let Some(user) = new.types.get_mut("User") {
            user.fields.remove("name");
        }

        let changes = SchemaChangelogGenerator::compare_snapshots(&old, &new);

        assert!(!changes.is_empty());
        assert!(changes
            .iter()
            .any(|c| c.change_type == ChangeType::FieldRemoved));
        assert!(changes.iter().any(|c| c.is_breaking));
    }

    #[tokio::test]
    async fn test_generate_version() {
        let generator = SchemaChangelogGenerator::new();

        let snapshot1 = create_test_snapshot();
        generator.set_snapshot(snapshot1).await;

        let mut snapshot2 = create_test_snapshot();
        if let Some(user) = snapshot2.types.get_mut("User") {
            user.fields.insert(
                "email".to_string(),
                FieldDef {
                    name: "email".to_string(),
                    field_type: "String".to_string(),
                    description: None,
                    arguments: HashMap::new(),
                    deprecated: false,
                    deprecation_reason: None,
                },
            );
        }
        generator.set_snapshot(snapshot2).await;

        let version = generator.generate_version("1.1.0").await.unwrap();

        assert_eq!(version.version, "1.1.0");
        assert!(!version.changes.is_empty());
    }

    #[tokio::test]
    async fn test_generate_markdown() {
        let generator = SchemaChangelogGenerator::new();

        let snapshot1 = create_test_snapshot();
        generator.set_snapshot(snapshot1).await;

        let mut snapshot2 = create_test_snapshot();
        if let Some(user) = snapshot2.types.get_mut("User") {
            user.fields.insert(
                "email".to_string(),
                FieldDef {
                    name: "email".to_string(),
                    field_type: "String".to_string(),
                    description: None,
                    arguments: HashMap::new(),
                    deprecated: false,
                    deprecation_reason: None,
                },
            );
        }
        generator.set_snapshot(snapshot2).await;
        generator.generate_version("1.1.0").await.unwrap();

        let markdown = generator.generate_markdown().await;

        assert!(markdown.contains("# Changelog"));
        assert!(markdown.contains("[1.1.0]"));
        assert!(markdown.contains("Added"));
    }

    #[tokio::test]
    async fn test_breaking_changes_detection() {
        let generator = SchemaChangelogGenerator::new();

        let snapshot1 = create_test_snapshot();
        generator.set_snapshot(snapshot1).await;

        let mut snapshot2 = create_test_snapshot();
        if let Some(user) = snapshot2.types.get_mut("User") {
            user.fields.remove("name");
        }
        generator.set_snapshot(snapshot2).await;

        assert!(generator.has_breaking_changes().await);

        let breaking = generator.get_breaking_changes().await;
        assert!(!breaking.is_empty());
    }

    #[tokio::test]
    async fn test_type_added_removed() {
        let old = create_test_snapshot();

        let mut new = SchemaSnapshot::new();
        new.add_type(TypeDef {
            name: "Post".to_string(),
            kind: "OBJECT".to_string(),
            description: None,
            fields: HashMap::new(),
            enum_values: Vec::new(),
            implements: Vec::new(),
            deprecated: false,
        });

        let changes = SchemaChangelogGenerator::compare_snapshots(&old, &new);

        assert!(changes
            .iter()
            .any(|c| c.change_type == ChangeType::TypeRemoved));
        assert!(changes
            .iter()
            .any(|c| c.change_type == ChangeType::TypeAdded));
    }

    #[test]
    fn test_change_severity() {
        assert!(ChangeType::FieldRemoved.is_breaking());
        assert!(ChangeType::TypeRemoved.is_breaking());
        assert!(!ChangeType::FieldAdded.is_breaking());
        assert!(!ChangeType::TypeAdded.is_breaking());

        assert_eq!(
            ChangeType::FieldRemoved.severity(),
            ChangeSeverity::Breaking
        );
        assert_eq!(
            ChangeType::FieldAdded.severity(),
            ChangeSeverity::NonBreaking
        );
        assert_eq!(
            ChangeType::DeprecationAdded.severity(),
            ChangeSeverity::Dangerous
        );
    }
}
