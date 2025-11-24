//! Automated API Documentation Versioning
//!
//! This module provides automatic versioning for API documentation,
//! tracking schema changes and maintaining versioned documentation.
//!
//! ## Features
//!
//! - **Version Management**: Semantic versioning for API
//! - **Documentation Snapshots**: Store documentation at each version
//! - **Version Comparison**: Compare documentation across versions
//! - **Deprecation Tracking**: Track deprecated features across versions
//! - **Migration Guides**: Generate guides for version upgrades

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

/// Semantic version
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticVersion {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
    /// Patch version
    pub patch: u32,
    /// Pre-release label
    pub prerelease: Option<String>,
}

impl SemanticVersion {
    /// Create a new version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            prerelease: None,
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('-').collect();
        let version_parts = parts[0];
        let prerelease = parts.get(1).map(|s| s.to_string());

        let nums: Vec<u32> = version_parts
            .split('.')
            .map(|p| p.parse())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| anyhow!("Invalid version format: {}", s))?;

        if nums.len() != 3 {
            return Err(anyhow!("Version must have 3 parts: {}", s));
        }

        Ok(Self {
            major: nums[0],
            minor: nums[1],
            patch: nums[2],
            prerelease,
        })
    }

    /// Bump major version
    pub fn bump_major(&self) -> Self {
        Self::new(self.major + 1, 0, 0)
    }

    /// Bump minor version
    pub fn bump_minor(&self) -> Self {
        Self::new(self.major, self.minor + 1, 0)
    }

    /// Bump patch version
    pub fn bump_patch(&self) -> Self {
        Self::new(self.major, self.minor, self.patch + 1)
    }

    /// Compare versions
    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        match self.major.cmp(&other.major) {
            std::cmp::Ordering::Equal => match self.minor.cmp(&other.minor) {
                std::cmp::Ordering::Equal => self.patch.cmp(&other.patch),
                ord => ord,
            },
            ord => ord,
        }
    }

    /// Check if this version is compatible with another
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major
    }
}

impl std::fmt::Display for SemanticVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(pre) = &self.prerelease {
            write!(f, "{}.{}.{}-{}", self.major, self.minor, self.patch, pre)
        } else {
            write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
        }
    }
}

impl PartialOrd for SemanticVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(std::cmp::Ord::cmp(self, other))
    }
}

impl Ord for SemanticVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.compare(other)
    }
}

/// Documentation entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationEntry {
    /// Entry path (e.g., "Query.users")
    pub path: String,
    /// Entry type
    pub entry_type: EntryType,
    /// Description
    pub description: String,
    /// Example
    pub example: Option<String>,
    /// Deprecated
    pub deprecated: bool,
    /// Deprecation message
    pub deprecation_message: Option<String>,
    /// Since version
    pub since_version: Option<String>,
    /// Until version (if removed)
    pub until_version: Option<String>,
    /// Tags
    pub tags: Vec<String>,
    /// Related entries
    pub related: Vec<String>,
}

/// Entry type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryType {
    Type,
    Field,
    Argument,
    Enum,
    EnumValue,
    Directive,
    Scalar,
}

/// Documentation snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationSnapshot {
    /// Version
    pub version: SemanticVersion,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Entries
    pub entries: HashMap<String, DocumentationEntry>,
    /// Metadata
    pub metadata: DocumentationMetadata,
}

impl DocumentationSnapshot {
    /// Create a new snapshot
    pub fn new(version: SemanticVersion) -> Self {
        Self {
            version,
            timestamp: SystemTime::now(),
            entries: HashMap::new(),
            metadata: DocumentationMetadata::default(),
        }
    }

    /// Add an entry
    pub fn add_entry(&mut self, entry: DocumentationEntry) {
        self.entries.insert(entry.path.clone(), entry);
    }

    /// Get entry by path
    pub fn get_entry(&self, path: &str) -> Option<&DocumentationEntry> {
        self.entries.get(path)
    }

    /// Get deprecated entries
    pub fn get_deprecated_entries(&self) -> Vec<&DocumentationEntry> {
        self.entries.values().filter(|e| e.deprecated).collect()
    }
}

/// Documentation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationMetadata {
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Contact email
    pub contact_email: Option<String>,
    /// Terms of service URL
    pub terms_of_service: Option<String>,
    /// License
    pub license: Option<String>,
    /// Server URL
    pub server_url: Option<String>,
    /// Custom tags
    pub tags: Vec<DocumentationTag>,
}

impl Default for DocumentationMetadata {
    fn default() -> Self {
        Self {
            title: "GraphQL API".to_string(),
            description: "GraphQL API documentation".to_string(),
            contact_email: None,
            terms_of_service: None,
            license: None,
            server_url: None,
            tags: Vec::new(),
        }
    }
}

/// Documentation tag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationTag {
    /// Tag name
    pub name: String,
    /// Description
    pub description: String,
}

/// Version diff
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    /// From version
    pub from_version: SemanticVersion,
    /// To version
    pub to_version: SemanticVersion,
    /// Added entries
    pub added: Vec<String>,
    /// Removed entries
    pub removed: Vec<String>,
    /// Modified entries
    pub modified: Vec<ModifiedEntry>,
    /// Newly deprecated
    pub newly_deprecated: Vec<String>,
}

/// Modified entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModifiedEntry {
    /// Path
    pub path: String,
    /// Changes
    pub changes: Vec<String>,
}

/// Migration guide
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationGuide {
    /// From version
    pub from_version: SemanticVersion,
    /// To version
    pub to_version: SemanticVersion,
    /// Steps
    pub steps: Vec<MigrationStep>,
    /// Breaking changes
    pub breaking_changes: Vec<BreakingChange>,
}

/// Migration step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStep {
    /// Step number
    pub step: u32,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Code example
    pub code_example: Option<CodeExample>,
}

/// Code example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    /// Before code
    pub before: String,
    /// After code
    pub after: String,
    /// Language
    pub language: String,
}

/// Breaking change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakingChange {
    /// Path affected
    pub path: String,
    /// Change type
    pub change_type: String,
    /// Description
    pub description: String,
    /// Workaround
    pub workaround: Option<String>,
}

/// Internal state
struct VersioningState {
    /// All snapshots by version
    snapshots: HashMap<SemanticVersion, DocumentationSnapshot>,
    /// Version history (ordered)
    version_history: Vec<SemanticVersion>,
    /// Current version
    current_version: Option<SemanticVersion>,
    /// Migration guides
    migration_guides: HashMap<(SemanticVersion, SemanticVersion), MigrationGuide>,
}

impl VersioningState {
    fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
            version_history: Vec::new(),
            current_version: None,
            migration_guides: HashMap::new(),
        }
    }
}

/// API Documentation Versioning Manager
///
/// Manages versioned API documentation with snapshot storage
/// and migration guide generation.
pub struct ApiVersioningManager {
    /// Internal state
    state: Arc<RwLock<VersioningState>>,
}

impl ApiVersioningManager {
    /// Create a new versioning manager
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(VersioningState::new())),
        }
    }

    /// Add a documentation snapshot
    pub async fn add_snapshot(&self, snapshot: DocumentationSnapshot) -> Result<()> {
        let mut state = self.state.write().await;

        let version = snapshot.version.clone();

        if state.snapshots.contains_key(&version) {
            return Err(anyhow!("Version {} already exists", version));
        }

        state.snapshots.insert(version.clone(), snapshot);
        state.version_history.push(version.clone());
        state.version_history.sort();
        state.current_version = Some(version);

        Ok(())
    }

    /// Get current version
    pub async fn get_current_version(&self) -> Option<SemanticVersion> {
        let state = self.state.read().await;
        state.current_version.clone()
    }

    /// Get all versions
    pub async fn get_all_versions(&self) -> Vec<SemanticVersion> {
        let state = self.state.read().await;
        state.version_history.clone()
    }

    /// Get snapshot by version
    pub async fn get_snapshot(&self, version: &SemanticVersion) -> Option<DocumentationSnapshot> {
        let state = self.state.read().await;
        state.snapshots.get(version).cloned()
    }

    /// Get latest snapshot
    pub async fn get_latest_snapshot(&self) -> Option<DocumentationSnapshot> {
        let state = self.state.read().await;
        state
            .current_version
            .as_ref()
            .and_then(|v| state.snapshots.get(v))
            .cloned()
    }

    /// Compare two versions
    pub async fn compare_versions(
        &self,
        from: &SemanticVersion,
        to: &SemanticVersion,
    ) -> Result<VersionDiff> {
        let state = self.state.read().await;

        let from_snapshot = state
            .snapshots
            .get(from)
            .ok_or_else(|| anyhow!("Version {} not found", from))?;

        let to_snapshot = state
            .snapshots
            .get(to)
            .ok_or_else(|| anyhow!("Version {} not found", to))?;

        let from_paths: std::collections::HashSet<_> = from_snapshot.entries.keys().collect();
        let to_paths: std::collections::HashSet<_> = to_snapshot.entries.keys().collect();

        let added: Vec<String> = to_paths
            .difference(&from_paths)
            .map(|s| (*s).clone())
            .collect();
        let removed: Vec<String> = from_paths
            .difference(&to_paths)
            .map(|s| (*s).clone())
            .collect();

        let mut modified = Vec::new();
        let mut newly_deprecated = Vec::new();

        for path in from_paths.intersection(&to_paths) {
            let from_entry = &from_snapshot.entries[*path];
            let to_entry = &to_snapshot.entries[*path];

            let mut changes = Vec::new();

            if from_entry.description != to_entry.description {
                changes.push("Description changed".to_string());
            }

            if from_entry.deprecated != to_entry.deprecated {
                if to_entry.deprecated {
                    newly_deprecated.push((*path).clone());
                }
                changes.push(format!(
                    "Deprecated: {} -> {}",
                    from_entry.deprecated, to_entry.deprecated
                ));
            }

            if !changes.is_empty() {
                modified.push(ModifiedEntry {
                    path: (*path).clone(),
                    changes,
                });
            }
        }

        Ok(VersionDiff {
            from_version: from.clone(),
            to_version: to.clone(),
            added,
            removed,
            modified,
            newly_deprecated,
        })
    }

    /// Generate migration guide
    pub async fn generate_migration_guide(
        &self,
        from: &SemanticVersion,
        to: &SemanticVersion,
    ) -> Result<MigrationGuide> {
        let diff = self.compare_versions(from, to).await?;

        let mut steps = Vec::new();
        let mut step_num = 1;

        // Steps for removed entries
        for path in &diff.removed {
            steps.push(MigrationStep {
                step: step_num,
                title: format!("Handle removed: {}", path),
                description: format!(
                    "The entry '{}' has been removed. Update your code to no longer use this.",
                    path
                ),
                code_example: None,
            });
            step_num += 1;
        }

        // Steps for deprecated entries
        for path in &diff.newly_deprecated {
            steps.push(MigrationStep {
                step: step_num,
                title: format!("Update deprecated: {}", path),
                description: format!(
                    "The entry '{}' is now deprecated. Consider migrating to the recommended alternative.",
                    path
                ),
                code_example: None,
            });
            step_num += 1;
        }

        // Breaking changes
        let breaking_changes: Vec<BreakingChange> = diff
            .removed
            .iter()
            .map(|path| BreakingChange {
                path: path.clone(),
                change_type: "Removed".to_string(),
                description: format!("Entry '{}' was removed", path),
                workaround: None,
            })
            .collect();

        Ok(MigrationGuide {
            from_version: from.clone(),
            to_version: to.clone(),
            steps,
            breaking_changes,
        })
    }

    /// Store migration guide
    pub async fn store_migration_guide(&self, guide: MigrationGuide) {
        let mut state = self.state.write().await;
        let key = (guide.from_version.clone(), guide.to_version.clone());
        state.migration_guides.insert(key, guide);
    }

    /// Get migration guide
    pub async fn get_migration_guide(
        &self,
        from: &SemanticVersion,
        to: &SemanticVersion,
    ) -> Option<MigrationGuide> {
        let state = self.state.read().await;
        state
            .migration_guides
            .get(&(from.clone(), to.clone()))
            .cloned()
    }

    /// Generate documentation index
    pub async fn generate_index(&self) -> String {
        let state = self.state.read().await;

        let mut index = String::new();
        index.push_str("# API Documentation Versions\n\n");

        for version in state.version_history.iter().rev() {
            if let Some(snapshot) = state.snapshots.get(version) {
                index.push_str(&format!("## Version {}\n\n", version));
                index.push_str(&format!("- **Title**: {}\n", snapshot.metadata.title));
                index.push_str(&format!("- **Entries**: {}\n", snapshot.entries.len()));
                index.push_str(&format!(
                    "- **Deprecated**: {}\n",
                    snapshot.get_deprecated_entries().len()
                ));
                index.push('\n');
            }
        }

        index
    }

    /// Export version as OpenAPI-compatible JSON
    pub async fn export_openapi(&self, version: &SemanticVersion) -> Result<String> {
        let state = self.state.read().await;

        let snapshot = state
            .snapshots
            .get(version)
            .ok_or_else(|| anyhow!("Version {} not found", version))?;

        let openapi = serde_json::json!({
            "openapi": "3.0.0",
            "info": {
                "title": snapshot.metadata.title,
                "description": snapshot.metadata.description,
                "version": version.to_string(),
                "contact": {
                    "email": snapshot.metadata.contact_email
                },
                "license": {
                    "name": snapshot.metadata.license
                }
            },
            "servers": [{
                "url": snapshot.metadata.server_url
            }],
            "paths": {},
            "components": {
                "schemas": {}
            }
        });

        Ok(serde_json::to_string_pretty(&openapi)?)
    }

    /// Get entries by tag
    pub async fn get_entries_by_tag(&self, version: &SemanticVersion, tag: &str) -> Vec<String> {
        let state = self.state.read().await;

        state
            .snapshots
            .get(version)
            .map(|s| {
                s.entries
                    .iter()
                    .filter(|(_, e)| e.tags.contains(&tag.to_string()))
                    .map(|(k, _)| k.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get next suggested version
    pub async fn suggest_next_version(&self, has_breaking_changes: bool) -> SemanticVersion {
        let state = self.state.read().await;

        let current = state
            .current_version
            .as_ref()
            .cloned()
            .unwrap_or_else(|| SemanticVersion::new(0, 1, 0));

        if has_breaking_changes {
            current.bump_major()
        } else {
            current.bump_minor()
        }
    }
}

impl Default for ApiVersioningManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_version_parse() {
        let v = SemanticVersion::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert_eq!(v.to_string(), "1.2.3");

        let v = SemanticVersion::parse("1.0.0-beta").unwrap();
        assert_eq!(v.prerelease, Some("beta".to_string()));
    }

    #[test]
    fn test_version_bumping() {
        let v = SemanticVersion::new(1, 2, 3);

        let major = v.bump_major();
        assert_eq!(major.to_string(), "2.0.0");

        let minor = v.bump_minor();
        assert_eq!(minor.to_string(), "1.3.0");

        let patch = v.bump_patch();
        assert_eq!(patch.to_string(), "1.2.4");
    }

    #[test]
    fn test_version_comparison() {
        let v1 = SemanticVersion::new(1, 0, 0);
        let v2 = SemanticVersion::new(2, 0, 0);
        let v3 = SemanticVersion::new(1, 1, 0);

        assert!(v1 < v2);
        assert!(v1 < v3);
        assert!(v3 < v2);
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = SemanticVersion::new(1, 0, 0);
        let v2 = SemanticVersion::new(1, 5, 0);
        let v3 = SemanticVersion::new(2, 0, 0);

        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));
    }

    #[tokio::test]
    async fn test_add_snapshot() {
        let manager = ApiVersioningManager::new();

        let version = SemanticVersion::new(1, 0, 0);
        let snapshot = DocumentationSnapshot::new(version.clone());

        manager.add_snapshot(snapshot).await.unwrap();

        let current = manager.get_current_version().await;
        assert_eq!(current, Some(version));
    }

    #[tokio::test]
    async fn test_compare_versions() {
        let manager = ApiVersioningManager::new();

        let v1 = SemanticVersion::new(1, 0, 0);
        let mut snap1 = DocumentationSnapshot::new(v1.clone());
        snap1.add_entry(DocumentationEntry {
            path: "Query.users".to_string(),
            entry_type: EntryType::Field,
            description: "Get users".to_string(),
            example: None,
            deprecated: false,
            deprecation_message: None,
            since_version: None,
            until_version: None,
            tags: Vec::new(),
            related: Vec::new(),
        });
        manager.add_snapshot(snap1).await.unwrap();

        let v2 = SemanticVersion::new(1, 1, 0);
        let mut snap2 = DocumentationSnapshot::new(v2.clone());
        snap2.add_entry(DocumentationEntry {
            path: "Query.users".to_string(),
            entry_type: EntryType::Field,
            description: "Get all users".to_string(),
            example: None,
            deprecated: false,
            deprecation_message: None,
            since_version: None,
            until_version: None,
            tags: Vec::new(),
            related: Vec::new(),
        });
        snap2.add_entry(DocumentationEntry {
            path: "Query.posts".to_string(),
            entry_type: EntryType::Field,
            description: "Get posts".to_string(),
            example: None,
            deprecated: false,
            deprecation_message: None,
            since_version: None,
            until_version: None,
            tags: Vec::new(),
            related: Vec::new(),
        });
        manager.add_snapshot(snap2).await.unwrap();

        let diff = manager.compare_versions(&v1, &v2).await.unwrap();

        assert_eq!(diff.added.len(), 1);
        assert!(diff.added.contains(&"Query.posts".to_string()));
        assert_eq!(diff.modified.len(), 1);
    }

    #[tokio::test]
    async fn test_migration_guide_generation() {
        let manager = ApiVersioningManager::new();

        let v1 = SemanticVersion::new(1, 0, 0);
        let mut snap1 = DocumentationSnapshot::new(v1.clone());
        snap1.add_entry(DocumentationEntry {
            path: "Query.oldField".to_string(),
            entry_type: EntryType::Field,
            description: "Old field".to_string(),
            example: None,
            deprecated: false,
            deprecation_message: None,
            since_version: None,
            until_version: None,
            tags: Vec::new(),
            related: Vec::new(),
        });
        manager.add_snapshot(snap1).await.unwrap();

        let v2 = SemanticVersion::new(2, 0, 0);
        let snap2 = DocumentationSnapshot::new(v2.clone());
        manager.add_snapshot(snap2).await.unwrap();

        let guide = manager.generate_migration_guide(&v1, &v2).await.unwrap();

        assert!(!guide.steps.is_empty());
        assert!(!guide.breaking_changes.is_empty());
    }

    #[tokio::test]
    async fn test_suggest_next_version() {
        let manager = ApiVersioningManager::new();

        let v1 = SemanticVersion::new(1, 0, 0);
        let snap1 = DocumentationSnapshot::new(v1.clone());
        manager.add_snapshot(snap1).await.unwrap();

        let next_minor = manager.suggest_next_version(false).await;
        assert_eq!(next_minor.to_string(), "1.1.0");

        let next_major = manager.suggest_next_version(true).await;
        assert_eq!(next_major.to_string(), "2.0.0");
    }

    #[tokio::test]
    async fn test_get_deprecated_entries() {
        let v1 = SemanticVersion::new(1, 0, 0);
        let mut snapshot = DocumentationSnapshot::new(v1);

        snapshot.add_entry(DocumentationEntry {
            path: "Query.oldField".to_string(),
            entry_type: EntryType::Field,
            description: "Old field".to_string(),
            example: None,
            deprecated: true,
            deprecation_message: Some("Use newField instead".to_string()),
            since_version: None,
            until_version: None,
            tags: Vec::new(),
            related: Vec::new(),
        });

        snapshot.add_entry(DocumentationEntry {
            path: "Query.newField".to_string(),
            entry_type: EntryType::Field,
            description: "New field".to_string(),
            example: None,
            deprecated: false,
            deprecation_message: None,
            since_version: None,
            until_version: None,
            tags: Vec::new(),
            related: Vec::new(),
        });

        let deprecated = snapshot.get_deprecated_entries();
        assert_eq!(deprecated.len(), 1);
        assert_eq!(deprecated[0].path, "Query.oldField");
    }

    #[tokio::test]
    async fn test_generate_index() {
        let manager = ApiVersioningManager::new();

        let v1 = SemanticVersion::new(1, 0, 0);
        let snap1 = DocumentationSnapshot::new(v1);
        manager.add_snapshot(snap1).await.unwrap();

        let index = manager.generate_index().await;
        assert!(index.contains("Version 1.0.0"));
    }
}
