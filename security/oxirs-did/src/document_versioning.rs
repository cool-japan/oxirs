//! # DID Document Versioning
//!
//! Provides append-only version history for DID documents with diff computation,
//! time-travel queries, and version metadata.
//!
//! ## Features
//!
//! - **Append-only history**: Immutable version log with monotonic version numbers
//! - **Diff computation**: Compute field-level diffs between any two versions
//! - **Time-travel queries**: Retrieve document state at any point in time
//! - **Version metadata**: Author, reason, timestamp for each version
//! - **Rollback**: Create new versions that revert to a previous state

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────
// Document types
// ─────────────────────────────────────────────

/// A simplified DID Document for versioning purposes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DidDocumentSnapshot {
    /// The DID subject.
    pub id: String,
    /// Controller(s).
    pub controller: Vec<String>,
    /// Verification method IDs.
    pub verification_methods: Vec<String>,
    /// Authentication method IDs.
    pub authentication: Vec<String>,
    /// Assertion method IDs.
    pub assertion_method: Vec<String>,
    /// Service endpoint entries (id -> endpoint).
    pub services: HashMap<String, String>,
    /// Additional custom properties.
    pub properties: HashMap<String, String>,
}

impl DidDocumentSnapshot {
    /// Create a new document snapshot.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            controller: Vec::new(),
            verification_methods: Vec::new(),
            authentication: Vec::new(),
            assertion_method: Vec::new(),
            services: HashMap::new(),
            properties: HashMap::new(),
        }
    }

    /// Add a controller.
    pub fn with_controller(mut self, controller: impl Into<String>) -> Self {
        self.controller.push(controller.into());
        self
    }

    /// Add a verification method.
    pub fn with_verification_method(mut self, vm: impl Into<String>) -> Self {
        self.verification_methods.push(vm.into());
        self
    }

    /// Add a service.
    pub fn with_service(mut self, id: impl Into<String>, endpoint: impl Into<String>) -> Self {
        self.services.insert(id.into(), endpoint.into());
        self
    }

    /// Add a property.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
}

/// Metadata for a version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadata {
    /// Version number (monotonically increasing).
    pub version: u64,
    /// When this version was created.
    pub created_at: DateTime<Utc>,
    /// Who created this version (optional DID).
    pub author: Option<String>,
    /// Reason for the change.
    pub reason: Option<String>,
    /// Whether this is a rollback version.
    pub is_rollback: bool,
    /// If rollback, which version was reverted to.
    pub rollback_to: Option<u64>,
}

/// A versioned document entry in the history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedDocument {
    /// The document snapshot.
    pub document: DidDocumentSnapshot,
    /// Version metadata.
    pub metadata: VersionMetadata,
}

// ─────────────────────────────────────────────
// Diff types
// ─────────────────────────────────────────────

/// A single change between two versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffEntry {
    /// A field was added.
    Added { field: String, value: String },
    /// A field was removed.
    Removed { field: String, value: String },
    /// A field was modified.
    Modified {
        field: String,
        old_value: String,
        new_value: String,
    },
}

/// Diff between two document versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    /// Source version.
    pub from_version: u64,
    /// Target version.
    pub to_version: u64,
    /// List of changes.
    pub changes: Vec<DiffEntry>,
}

impl VersionDiff {
    /// Number of changes.
    pub fn change_count(&self) -> usize {
        self.changes.len()
    }

    /// Whether there are no changes.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Count additions.
    pub fn additions(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, DiffEntry::Added { .. }))
            .count()
    }

    /// Count removals.
    pub fn removals(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, DiffEntry::Removed { .. }))
            .count()
    }

    /// Count modifications.
    pub fn modifications(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, DiffEntry::Modified { .. }))
            .count()
    }
}

// ─────────────────────────────────────────────
// DidDocumentVersionStore
// ─────────────────────────────────────────────

/// Append-only version store for DID documents.
pub struct DidDocumentVersionStore {
    /// DID that this store tracks.
    did: String,
    /// Version history (ordered by version number).
    versions: Vec<VersionedDocument>,
    /// Current version counter.
    current_version: u64,
}

impl DidDocumentVersionStore {
    /// Create a new version store for a DID with an initial document.
    pub fn new(initial_document: DidDocumentSnapshot) -> Self {
        let did = initial_document.id.clone();
        let versioned = VersionedDocument {
            document: initial_document,
            metadata: VersionMetadata {
                version: 1,
                created_at: Utc::now(),
                author: None,
                reason: Some("Initial version".to_string()),
                is_rollback: false,
                rollback_to: None,
            },
        };
        Self {
            did,
            versions: vec![versioned],
            current_version: 1,
        }
    }

    /// Get the DID this store tracks.
    pub fn did(&self) -> &str {
        &self.did
    }

    /// Get the current version number.
    pub fn current_version(&self) -> u64 {
        self.current_version
    }

    /// Get the total number of versions.
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }

    /// Get the current (latest) document.
    pub fn current_document(&self) -> Option<&DidDocumentSnapshot> {
        self.versions.last().map(|v| &v.document)
    }

    /// Append a new version.
    pub fn append(
        &mut self,
        document: DidDocumentSnapshot,
        author: Option<String>,
        reason: Option<String>,
    ) -> u64 {
        self.current_version += 1;
        let versioned = VersionedDocument {
            document,
            metadata: VersionMetadata {
                version: self.current_version,
                created_at: Utc::now(),
                author,
                reason,
                is_rollback: false,
                rollback_to: None,
            },
        };
        self.versions.push(versioned);
        self.current_version
    }

    /// Get a document at a specific version.
    pub fn get_version(&self, version: u64) -> Option<&VersionedDocument> {
        self.versions.iter().find(|v| v.metadata.version == version)
    }

    /// Get a document at a specific point in time (latest version <= timestamp).
    pub fn get_at_time(&self, timestamp: DateTime<Utc>) -> Option<&VersionedDocument> {
        self.versions
            .iter()
            .rev()
            .find(|v| v.metadata.created_at <= timestamp)
    }

    /// Compute the diff between two versions.
    pub fn diff(&self, from_version: u64, to_version: u64) -> Option<VersionDiff> {
        let from = self.get_version(from_version)?;
        let to = self.get_version(to_version)?;

        let changes = compute_diff(&from.document, &to.document);

        Some(VersionDiff {
            from_version,
            to_version,
            changes,
        })
    }

    /// Rollback to a previous version (creates a new version with the old state).
    pub fn rollback(&mut self, target_version: u64, author: Option<String>) -> Option<u64> {
        let target_doc = self
            .get_version(target_version)
            .map(|v| v.document.clone())?;

        self.current_version += 1;
        let versioned = VersionedDocument {
            document: target_doc,
            metadata: VersionMetadata {
                version: self.current_version,
                created_at: Utc::now(),
                author,
                reason: Some(format!("Rollback to version {target_version}")),
                is_rollback: true,
                rollback_to: Some(target_version),
            },
        };
        self.versions.push(versioned);
        Some(self.current_version)
    }

    /// Get version metadata for all versions.
    pub fn version_history(&self) -> Vec<&VersionMetadata> {
        self.versions.iter().map(|v| &v.metadata).collect()
    }

    /// Get versions that were authored by a specific DID.
    pub fn versions_by_author(&self, author: &str) -> Vec<&VersionedDocument> {
        self.versions
            .iter()
            .filter(|v| v.metadata.author.as_deref() == Some(author))
            .collect()
    }

    /// Check if a specific version exists.
    pub fn has_version(&self, version: u64) -> bool {
        self.versions.iter().any(|v| v.metadata.version == version)
    }
}

// ─── Diff computation ────────────────────────────────────

fn compute_diff(from: &DidDocumentSnapshot, to: &DidDocumentSnapshot) -> Vec<DiffEntry> {
    let mut changes = Vec::new();

    // Controllers
    diff_vec("controller", &from.controller, &to.controller, &mut changes);

    // Verification methods
    diff_vec(
        "verificationMethod",
        &from.verification_methods,
        &to.verification_methods,
        &mut changes,
    );

    // Authentication
    diff_vec(
        "authentication",
        &from.authentication,
        &to.authentication,
        &mut changes,
    );

    // Assertion methods
    diff_vec(
        "assertionMethod",
        &from.assertion_method,
        &to.assertion_method,
        &mut changes,
    );

    // Services
    diff_map("service", &from.services, &to.services, &mut changes);

    // Properties
    diff_map("property", &from.properties, &to.properties, &mut changes);

    changes
}

fn diff_vec(field: &str, from: &[String], to: &[String], changes: &mut Vec<DiffEntry>) {
    let from_set: std::collections::HashSet<&str> = from.iter().map(|s| s.as_str()).collect();
    let to_set: std::collections::HashSet<&str> = to.iter().map(|s| s.as_str()).collect();

    for &item in to_set.difference(&from_set) {
        changes.push(DiffEntry::Added {
            field: field.to_string(),
            value: item.to_string(),
        });
    }
    for &item in from_set.difference(&to_set) {
        changes.push(DiffEntry::Removed {
            field: field.to_string(),
            value: item.to_string(),
        });
    }
}

fn diff_map(
    field: &str,
    from: &HashMap<String, String>,
    to: &HashMap<String, String>,
    changes: &mut Vec<DiffEntry>,
) {
    for (key, to_val) in to {
        match from.get(key) {
            Some(from_val) if from_val != to_val => {
                changes.push(DiffEntry::Modified {
                    field: format!("{field}.{key}"),
                    old_value: from_val.clone(),
                    new_value: to_val.clone(),
                });
            }
            None => {
                changes.push(DiffEntry::Added {
                    field: format!("{field}.{key}"),
                    value: to_val.clone(),
                });
            }
            _ => {} // unchanged
        }
    }
    for (key, from_val) in from {
        if !to.contains_key(key) {
            changes.push(DiffEntry::Removed {
                field: format!("{field}.{key}"),
                value: from_val.clone(),
            });
        }
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn initial_doc() -> DidDocumentSnapshot {
        DidDocumentSnapshot::new("did:example:123")
            .with_controller("did:example:123")
            .with_verification_method("did:example:123#key-1")
            .with_service("website", "https://example.com")
    }

    fn updated_doc() -> DidDocumentSnapshot {
        DidDocumentSnapshot::new("did:example:123")
            .with_controller("did:example:123")
            .with_controller("did:example:456")
            .with_verification_method("did:example:123#key-1")
            .with_verification_method("did:example:123#key-2")
            .with_service("website", "https://new.example.com")
            .with_service("api", "https://api.example.com")
    }

    fn make_store() -> DidDocumentVersionStore {
        DidDocumentVersionStore::new(initial_doc())
    }

    // ═══ Construction tests ══════════════════════════════

    #[test]
    fn test_new_store() {
        let store = make_store();
        assert_eq!(store.did(), "did:example:123");
        assert_eq!(store.current_version(), 1);
        assert_eq!(store.version_count(), 1);
    }

    #[test]
    fn test_current_document() {
        let store = make_store();
        let doc = store.current_document();
        assert!(doc.is_some());
        assert_eq!(doc.expect("doc").id, "did:example:123");
    }

    // ═══ Append tests ════════════════════════════════════

    #[test]
    fn test_append_version() {
        let mut store = make_store();
        let v = store.append(
            updated_doc(),
            Some("admin".to_string()),
            Some("Add key-2".to_string()),
        );
        assert_eq!(v, 2);
        assert_eq!(store.current_version(), 2);
        assert_eq!(store.version_count(), 2);
    }

    #[test]
    fn test_append_multiple() {
        let mut store = make_store();
        store.append(updated_doc(), None, None);
        store.append(initial_doc(), None, None);
        assert_eq!(store.version_count(), 3);
        assert_eq!(store.current_version(), 3);
    }

    // ═══ Get version tests ═══════════════════════════════

    #[test]
    fn test_get_version_1() {
        let store = make_store();
        let v1 = store.get_version(1);
        assert!(v1.is_some());
        assert_eq!(v1.expect("v1").metadata.version, 1);
    }

    #[test]
    fn test_get_version_nonexistent() {
        let store = make_store();
        assert!(store.get_version(99).is_none());
    }

    #[test]
    fn test_has_version() {
        let store = make_store();
        assert!(store.has_version(1));
        assert!(!store.has_version(2));
    }

    // ═══ Time travel tests ═══════════════════════════════

    #[test]
    fn test_get_at_time_before_creation() {
        let store = make_store();
        let past = Utc::now() - chrono::Duration::hours(1);
        assert!(store.get_at_time(past).is_none());
    }

    #[test]
    fn test_get_at_time_after_creation() {
        let store = make_store();
        let future = Utc::now() + chrono::Duration::hours(1);
        let doc = store.get_at_time(future);
        assert!(doc.is_some());
        assert_eq!(doc.expect("doc").metadata.version, 1);
    }

    // ═══ Diff tests ══════════════════════════════════════

    #[test]
    fn test_diff_same_version() {
        let store = make_store();
        let diff = store.diff(1, 1);
        assert!(diff.is_some());
        assert!(diff.expect("diff").is_empty());
    }

    #[test]
    fn test_diff_added_controller() {
        let mut store = make_store();
        store.append(updated_doc(), None, None);
        let diff = store.diff(1, 2).expect("diff should exist");
        let added_controllers: Vec<_> = diff
            .changes
            .iter()
            .filter(|c| matches!(c, DiffEntry::Added { field, .. } if field == "controller"))
            .collect();
        assert!(!added_controllers.is_empty());
    }

    #[test]
    fn test_diff_added_service() {
        let mut store = make_store();
        store.append(updated_doc(), None, None);
        let diff = store.diff(1, 2).expect("diff");
        let has_api_added = diff
            .changes
            .iter()
            .any(|c| matches!(c, DiffEntry::Added { field, .. } if field == "service.api"));
        assert!(has_api_added);
    }

    #[test]
    fn test_diff_modified_service() {
        let mut store = make_store();
        store.append(updated_doc(), None, None);
        let diff = store.diff(1, 2).expect("diff");
        let has_website_modified = diff
            .changes
            .iter()
            .any(|c| matches!(c, DiffEntry::Modified { field, .. } if field == "service.website"));
        assert!(has_website_modified);
    }

    #[test]
    fn test_diff_nonexistent_version() {
        let store = make_store();
        assert!(store.diff(1, 99).is_none());
    }

    #[test]
    fn test_diff_change_count() {
        let mut store = make_store();
        store.append(updated_doc(), None, None);
        let diff = store.diff(1, 2).expect("diff");
        assert!(diff.change_count() > 0);
    }

    #[test]
    fn test_diff_additions_removals() {
        let mut store = make_store();
        store.append(updated_doc(), None, None);
        let diff = store.diff(1, 2).expect("diff");
        assert!(diff.additions() > 0);
    }

    // ═══ Rollback tests ══════════════════════════════════

    #[test]
    fn test_rollback() {
        let mut store = make_store();
        store.append(updated_doc(), None, None);
        let v3 = store.rollback(1, Some("admin".to_string()));
        assert!(v3.is_some());
        assert_eq!(v3.expect("v3"), 3);
        assert_eq!(store.version_count(), 3);

        // Current document should match v1
        let current = store.current_document().expect("doc");
        let v1_doc = &store.get_version(1).expect("v1").document;
        assert_eq!(current, v1_doc);
    }

    #[test]
    fn test_rollback_metadata() {
        let mut store = make_store();
        store.append(updated_doc(), None, None);
        store.rollback(1, None);
        let v3 = store.get_version(3).expect("v3");
        assert!(v3.metadata.is_rollback);
        assert_eq!(v3.metadata.rollback_to, Some(1));
    }

    #[test]
    fn test_rollback_nonexistent() {
        let mut store = make_store();
        assert!(store.rollback(99, None).is_none());
    }

    // ═══ Version history tests ═══════════════════════════

    #[test]
    fn test_version_history() {
        let mut store = make_store();
        store.append(updated_doc(), None, None);
        let history = store.version_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].version, 1);
        assert_eq!(history[1].version, 2);
    }

    // ═══ Author filter tests ═════════════════════════════

    #[test]
    fn test_versions_by_author() {
        let mut store = make_store();
        store.append(updated_doc(), Some("admin".to_string()), None);
        store.append(initial_doc(), Some("user".to_string()), None);
        let admin_versions = store.versions_by_author("admin");
        assert_eq!(admin_versions.len(), 1);
    }

    #[test]
    fn test_versions_by_author_none() {
        let store = make_store();
        let results = store.versions_by_author("nobody");
        assert!(results.is_empty());
    }

    // ═══ Document snapshot builder tests ═════════════════

    #[test]
    fn test_snapshot_builder() {
        let doc = DidDocumentSnapshot::new("did:example:test")
            .with_controller("did:example:ctrl")
            .with_verification_method("did:example:test#key-1")
            .with_service("svc1", "https://svc.example.com")
            .with_property("created", "2024-01-01");
        assert_eq!(doc.controller.len(), 1);
        assert_eq!(doc.verification_methods.len(), 1);
        assert_eq!(doc.services.len(), 1);
        assert_eq!(doc.properties.len(), 1);
    }

    // ═══ VersionDiff method tests ════════════════════════

    #[test]
    fn test_version_diff_methods() {
        let diff = VersionDiff {
            from_version: 1,
            to_version: 2,
            changes: vec![
                DiffEntry::Added {
                    field: "controller".to_string(),
                    value: "did:example:new".to_string(),
                },
                DiffEntry::Removed {
                    field: "service.old".to_string(),
                    value: "https://old.example.com".to_string(),
                },
                DiffEntry::Modified {
                    field: "service.web".to_string(),
                    old_value: "https://old.example.com".to_string(),
                    new_value: "https://new.example.com".to_string(),
                },
            ],
        };
        assert_eq!(diff.change_count(), 3);
        assert!(!diff.is_empty());
        assert_eq!(diff.additions(), 1);
        assert_eq!(diff.removals(), 1);
        assert_eq!(diff.modifications(), 1);
    }

    #[test]
    fn test_empty_diff() {
        let diff = VersionDiff {
            from_version: 1,
            to_version: 1,
            changes: vec![],
        };
        assert!(diff.is_empty());
        assert_eq!(diff.change_count(), 0);
    }

    // ═══ Initial version metadata test ═══════════════════

    #[test]
    fn test_initial_version_metadata() {
        let store = make_store();
        let v1 = store.get_version(1).expect("v1");
        assert!(!v1.metadata.is_rollback);
        assert!(v1.metadata.rollback_to.is_none());
        assert_eq!(v1.metadata.reason, Some("Initial version".to_string()));
    }
}
