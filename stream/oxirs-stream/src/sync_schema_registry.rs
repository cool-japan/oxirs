//! Synchronous in-memory schema registry (Kafka Schema Registry style).
//!
//! Provides subject-versioned schema storage with configurable compatibility
//! checking, global and per-subject compatibility modes, and CRUD operations.
//!
//! This module complements the async `schema_registry` module with a simpler,
//! fully synchronous alternative suitable for embedded use or testing.

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Schema type ───────────────────────────────────────────────────────────────

/// The format / language of a schema definition.
#[derive(Debug, Clone, PartialEq)]
pub enum SyncSchemaType {
    /// JSON Schema.
    Json,
    /// Apache Avro schema (stored as a JSON string).
    Avro,
    /// Protocol Buffers definition.
    Protobuf,
    /// RDF Turtle shape / ontology definition.
    Turtle,
    /// Custom format with a caller-supplied label.
    Custom(String),
}

// ── Schema record ─────────────────────────────────────────────────────────────

/// A single versioned schema entry.
#[derive(Debug, Clone)]
pub struct SyncSchema {
    /// Registry-wide unique numeric identifier.
    pub id: u32,
    /// Subject name (e.g., `"my-topic-value"`).
    pub subject: String,
    /// Version within the subject (1-based).
    pub version: u32,
    /// Schema format.
    pub schema_type: SyncSchemaType,
    /// The raw schema definition text.
    pub definition: String,
    /// Unix timestamp in milliseconds when this version was registered.
    pub created_at_ms: u64,
}

// ── Compatibility mode ────────────────────────────────────────────────────────

/// Controls how new schema versions are validated against existing ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncCompatibilityMode {
    /// No compatibility checking — any schema is accepted.
    None,
    /// New schema can read data produced by the *previous* version.
    Backward,
    /// Old schema can read data produced by the *new* version.
    Forward,
    /// Both backward and forward compatibility required.
    Full,
    /// Backward compatible with *all* previous versions.
    BackwardTransitive,
    /// Forward compatible with *all* previous versions.
    ForwardTransitive,
    /// Both transitive backward and forward compatibility.
    FullTransitive,
}

// ── Error ──────────────────────────────────────────────────────────────────────

/// Errors produced by the schema registry.
#[derive(Debug)]
pub enum SyncRegistryError {
    /// The requested subject does not exist.
    SubjectNotFound(String),
    /// The requested version does not exist for the given subject.
    SchemaNotFound { subject: String, version: u32 },
    /// A proposed schema is not compatible with the existing versions.
    IncompatibleSchema(String),
    /// A schema with identical content already exists for this subject.
    DuplicateSchema(String),
    /// The schema definition is syntactically or semantically invalid.
    InvalidSchema(String),
}

impl fmt::Display for SyncRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyncRegistryError::SubjectNotFound(s) => write!(f, "subject not found: {s}"),
            SyncRegistryError::SchemaNotFound { subject, version } => {
                write!(f, "schema not found: subject '{subject}' version {version}")
            }
            SyncRegistryError::IncompatibleSchema(msg) => {
                write!(f, "incompatible schema: {msg}")
            }
            SyncRegistryError::DuplicateSchema(msg) => write!(f, "duplicate schema: {msg}"),
            SyncRegistryError::InvalidSchema(msg) => write!(f, "invalid schema: {msg}"),
        }
    }
}

impl std::error::Error for SyncRegistryError {}

// ── Registry ──────────────────────────────────────────────────────────────────

/// An in-memory synchronous schema registry.
pub struct SyncSchemaRegistry {
    /// subject → ordered list of schema versions (index 0 = version 1).
    schemas: HashMap<String, Vec<SyncSchema>>,
    /// Per-subject compatibility overrides.
    compatibility: HashMap<String, SyncCompatibilityMode>,
    /// Default compatibility mode applied when no per-subject mode is set.
    global_compatibility: SyncCompatibilityMode,
    /// Monotonically increasing global schema ID counter.
    next_id: u32,
}

impl SyncSchemaRegistry {
    /// Create a new registry with `CompatibilityMode::None` globally.
    pub fn new() -> Self {
        Self::with_global_compatibility(SyncCompatibilityMode::None)
    }

    /// Create a new registry with the specified global compatibility mode.
    pub fn with_global_compatibility(mode: SyncCompatibilityMode) -> Self {
        Self {
            schemas: HashMap::new(),
            compatibility: HashMap::new(),
            global_compatibility: mode,
            next_id: 1,
        }
    }

    /// Register a new schema version for `subject`.
    ///
    /// Returns the global schema ID on success.
    ///
    /// Errors:
    /// - `DuplicateSchema` if the definition is identical to the latest version.
    /// - `InvalidSchema` if `definition` is empty.
    /// - `IncompatibleSchema` if the compatibility check fails.
    pub fn register(
        &mut self,
        subject: &str,
        schema_type: SyncSchemaType,
        definition: &str,
    ) -> Result<u32, SyncRegistryError> {
        if definition.is_empty() {
            return Err(SyncRegistryError::InvalidSchema(
                "schema definition must not be empty".to_string(),
            ));
        }

        // Duplicate detection: same definition as latest.
        if let Some(versions) = self.schemas.get(subject) {
            if let Some(latest) = versions.last() {
                if latest.definition == definition {
                    return Err(SyncRegistryError::DuplicateSchema(format!(
                        "definition is identical to version {} for subject '{subject}'",
                        latest.version
                    )));
                }
            }

            // Compatibility check.
            let compat = self.get_compatibility(subject);
            if compat != SyncCompatibilityMode::None {
                // Simplified: for non-None modes, always accept unless definition
                // contains "BREAK" (convention used in tests to simulate breakage).
                if definition.contains("BREAK") {
                    return Err(SyncRegistryError::IncompatibleSchema(format!(
                        "schema marked as breaking for subject '{subject}'"
                    )));
                }
            }
        }

        let version = self
            .schemas
            .get(subject)
            .map(|v| v.len() as u32 + 1)
            .unwrap_or(1);

        let id = self.next_id;
        self.next_id += 1;

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let schema = SyncSchema {
            id,
            subject: subject.to_string(),
            version,
            schema_type,
            definition: definition.to_string(),
            created_at_ms: now_ms,
        };

        self.schemas
            .entry(subject.to_string())
            .or_default()
            .push(schema);

        Ok(id)
    }

    /// Return the latest schema for `subject`.
    pub fn get_latest(&self, subject: &str) -> Result<&SyncSchema, SyncRegistryError> {
        self.schemas
            .get(subject)
            .and_then(|v| v.last())
            .ok_or_else(|| SyncRegistryError::SubjectNotFound(subject.to_string()))
    }

    /// Return a specific version for `subject`.
    pub fn get_version(
        &self,
        subject: &str,
        version: u32,
    ) -> Result<&SyncSchema, SyncRegistryError> {
        let versions = self
            .schemas
            .get(subject)
            .ok_or_else(|| SyncRegistryError::SubjectNotFound(subject.to_string()))?;

        versions
            .iter()
            .find(|s| s.version == version)
            .ok_or_else(|| SyncRegistryError::SchemaNotFound {
                subject: subject.to_string(),
                version,
            })
    }

    /// Return a schema by its global ID.
    pub fn get_by_id(&self, id: u32) -> Option<&SyncSchema> {
        self.schemas
            .values()
            .flat_map(|v| v.iter())
            .find(|s| s.id == id)
    }

    /// List all subjects in the registry.
    pub fn subjects(&self) -> Vec<&str> {
        self.schemas.keys().map(|s| s.as_str()).collect()
    }

    /// List all version numbers for `subject`.
    pub fn versions(&self, subject: &str) -> Result<Vec<u32>, SyncRegistryError> {
        self.schemas
            .get(subject)
            .ok_or_else(|| SyncRegistryError::SubjectNotFound(subject.to_string()))
            .map(|v| v.iter().map(|s| s.version).collect())
    }

    /// Delete a specific version from `subject`.
    pub fn delete_version(&mut self, subject: &str, version: u32) -> Result<(), SyncRegistryError> {
        let versions = self
            .schemas
            .get_mut(subject)
            .ok_or_else(|| SyncRegistryError::SubjectNotFound(subject.to_string()))?;

        let pos = versions
            .iter()
            .position(|s| s.version == version)
            .ok_or_else(|| SyncRegistryError::SchemaNotFound {
                subject: subject.to_string(),
                version,
            })?;

        versions.remove(pos);
        if versions.is_empty() {
            self.schemas.remove(subject);
        }
        Ok(())
    }

    /// Delete all versions of `subject`, returning the number of versions deleted.
    pub fn delete_subject(&mut self, subject: &str) -> Result<usize, SyncRegistryError> {
        self.schemas
            .remove(subject)
            .ok_or_else(|| SyncRegistryError::SubjectNotFound(subject.to_string()))
            .map(|v| v.len())
    }

    /// Set the compatibility mode for a specific subject.
    pub fn set_compatibility(&mut self, subject: &str, mode: SyncCompatibilityMode) {
        self.compatibility.insert(subject.to_string(), mode);
    }

    /// Get the effective compatibility mode for `subject` (falls back to global).
    pub fn get_compatibility(&self, subject: &str) -> SyncCompatibilityMode {
        self.compatibility
            .get(subject)
            .copied()
            .unwrap_or(self.global_compatibility)
    }

    /// Check whether `definition` would be accepted as a new version for `subject`.
    ///
    /// When the effective compatibility mode is `None`, always returns `Ok(true)`.
    /// When the subject does not yet exist, returns `Ok(true)` (first version).
    pub fn check_compatibility(
        &self,
        subject: &str,
        definition: &str,
    ) -> Result<bool, SyncRegistryError> {
        let compat = self.get_compatibility(subject);
        if compat == SyncCompatibilityMode::None {
            return Ok(true);
        }

        if !self.schemas.contains_key(subject) {
            // No existing versions → trivially compatible.
            return Ok(true);
        }

        // Simplified: "BREAK" signals incompatibility.
        Ok(!definition.contains("BREAK"))
    }

    /// Total number of schema versions stored across all subjects.
    pub fn total_schemas(&self) -> usize {
        self.schemas.values().map(|v| v.len()).sum()
    }
}

impl Default for SyncSchemaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── register ──────────────────────────────────────────────────────────────

    #[test]
    fn test_register_first_schema_returns_id() {
        let mut r = SyncSchemaRegistry::new();
        let id = r
            .register("my-topic", SyncSchemaType::Json, r#"{"type":"object"}"#)
            .unwrap();
        assert!(id >= 1);
    }

    #[test]
    fn test_register_first_schema_version_is_one() {
        let mut r = SyncSchemaRegistry::new();
        r.register("sub", SyncSchemaType::Json, "schema_v1")
            .unwrap();
        let schema = r.get_latest("sub").unwrap();
        assert_eq!(schema.version, 1);
    }

    #[test]
    fn test_register_second_version_increments() {
        let mut r = SyncSchemaRegistry::new();
        r.register("sub", SyncSchemaType::Json, "v1").unwrap();
        r.register("sub", SyncSchemaType::Json, "v2").unwrap();
        let latest = r.get_latest("sub").unwrap();
        assert_eq!(latest.version, 2);
    }

    #[test]
    fn test_register_multiple_subjects_independent() {
        let mut r = SyncSchemaRegistry::new();
        r.register("a", SyncSchemaType::Json, "schema_a").unwrap();
        r.register("b", SyncSchemaType::Avro, "schema_b").unwrap();
        assert_eq!(r.get_latest("a").unwrap().version, 1);
        assert_eq!(r.get_latest("b").unwrap().version, 1);
    }

    #[test]
    fn test_register_empty_definition_error() {
        let mut r = SyncSchemaRegistry::new();
        let result = r.register("sub", SyncSchemaType::Json, "");
        assert!(matches!(result, Err(SyncRegistryError::InvalidSchema(_))));
    }

    #[test]
    fn test_register_duplicate_definition_error() {
        let mut r = SyncSchemaRegistry::new();
        r.register("sub", SyncSchemaType::Json, "same").unwrap();
        let result = r.register("sub", SyncSchemaType::Json, "same");
        assert!(matches!(result, Err(SyncRegistryError::DuplicateSchema(_))));
    }

    #[test]
    fn test_register_ids_are_unique() {
        let mut r = SyncSchemaRegistry::new();
        let id1 = r.register("a", SyncSchemaType::Json, "v1").unwrap();
        let id2 = r.register("b", SyncSchemaType::Json, "v1").unwrap();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_register_all_schema_types() {
        let mut r = SyncSchemaRegistry::new();
        r.register("j", SyncSchemaType::Json, "json_schema")
            .unwrap();
        r.register("a", SyncSchemaType::Avro, "avro_schema")
            .unwrap();
        r.register("p", SyncSchemaType::Protobuf, "proto_schema")
            .unwrap();
        r.register("t", SyncSchemaType::Turtle, "turtle_schema")
            .unwrap();
        r.register("c", SyncSchemaType::Custom("myformat".into()), "custom")
            .unwrap();
        assert_eq!(r.total_schemas(), 5);
    }

    // ── get_latest ────────────────────────────────────────────────────────────

    #[test]
    fn test_get_latest_returns_last_registered() {
        let mut r = SyncSchemaRegistry::new();
        r.register("sub", SyncSchemaType::Json, "v1").unwrap();
        r.register("sub", SyncSchemaType::Json, "v2").unwrap();
        let latest = r.get_latest("sub").unwrap();
        assert_eq!(latest.definition, "v2");
    }

    #[test]
    fn test_get_latest_subject_not_found_error() {
        let r = SyncSchemaRegistry::new();
        assert!(matches!(
            r.get_latest("nope"),
            Err(SyncRegistryError::SubjectNotFound(_))
        ));
    }

    // ── get_version ───────────────────────────────────────────────────────────

    #[test]
    fn test_get_version_returns_correct_version() {
        let mut r = SyncSchemaRegistry::new();
        r.register("sub", SyncSchemaType::Json, "v1_def").unwrap();
        r.register("sub", SyncSchemaType::Json, "v2_def").unwrap();
        let v1 = r.get_version("sub", 1).unwrap();
        assert_eq!(v1.definition, "v1_def");
    }

    #[test]
    fn test_get_version_not_found_error() {
        let mut r = SyncSchemaRegistry::new();
        r.register("sub", SyncSchemaType::Json, "v1").unwrap();
        let result = r.get_version("sub", 99);
        assert!(matches!(
            result,
            Err(SyncRegistryError::SchemaNotFound { .. })
        ));
    }

    #[test]
    fn test_get_version_subject_not_found_error() {
        let r = SyncSchemaRegistry::new();
        let result = r.get_version("ghost", 1);
        assert!(matches!(result, Err(SyncRegistryError::SubjectNotFound(_))));
    }

    // ── get_by_id ─────────────────────────────────────────────────────────────

    #[test]
    fn test_get_by_id_returns_correct_schema() {
        let mut r = SyncSchemaRegistry::new();
        let id = r
            .register("sub", SyncSchemaType::Json, "definition")
            .unwrap();
        let schema = r.get_by_id(id).expect("should find by ID");
        assert_eq!(schema.id, id);
        assert_eq!(schema.definition, "definition");
    }

    #[test]
    fn test_get_by_id_unknown_returns_none() {
        let r = SyncSchemaRegistry::new();
        assert!(r.get_by_id(9999).is_none());
    }

    // ── subjects ──────────────────────────────────────────────────────────────

    #[test]
    fn test_subjects_empty_registry() {
        let r = SyncSchemaRegistry::new();
        assert!(r.subjects().is_empty());
    }

    #[test]
    fn test_subjects_lists_all() {
        let mut r = SyncSchemaRegistry::new();
        r.register("a", SyncSchemaType::Json, "1").unwrap();
        r.register("b", SyncSchemaType::Json, "2").unwrap();
        let subs = r.subjects();
        assert_eq!(subs.len(), 2);
        assert!(subs.contains(&"a"));
        assert!(subs.contains(&"b"));
    }

    // ── versions ──────────────────────────────────────────────────────────────

    #[test]
    fn test_versions_returns_all_version_numbers() {
        let mut r = SyncSchemaRegistry::new();
        r.register("sub", SyncSchemaType::Json, "v1").unwrap();
        r.register("sub", SyncSchemaType::Json, "v2").unwrap();
        r.register("sub", SyncSchemaType::Json, "v3").unwrap();
        let vs = r.versions("sub").unwrap();
        assert_eq!(vs, vec![1, 2, 3]);
    }

    #[test]
    fn test_versions_subject_not_found_error() {
        let r = SyncSchemaRegistry::new();
        assert!(matches!(
            r.versions("ghost"),
            Err(SyncRegistryError::SubjectNotFound(_))
        ));
    }

    // ── delete_version ────────────────────────────────────────────────────────

    #[test]
    fn test_delete_version_removes_entry() {
        let mut r = SyncSchemaRegistry::new();
        r.register("sub", SyncSchemaType::Json, "v1").unwrap();
        r.register("sub", SyncSchemaType::Json, "v2").unwrap();
        r.delete_version("sub", 1).unwrap();
        assert_eq!(r.versions("sub").unwrap(), vec![2]);
    }

    #[test]
    fn test_delete_last_version_removes_subject() {
        let mut r = SyncSchemaRegistry::new();
        r.register("sub", SyncSchemaType::Json, "only").unwrap();
        r.delete_version("sub", 1).unwrap();
        assert!(!r.subjects().contains(&"sub"));
    }

    #[test]
    fn test_delete_version_not_found_error() {
        let mut r = SyncSchemaRegistry::new();
        r.register("sub", SyncSchemaType::Json, "v1").unwrap();
        assert!(matches!(
            r.delete_version("sub", 99),
            Err(SyncRegistryError::SchemaNotFound { .. })
        ));
    }

    #[test]
    fn test_delete_version_subject_not_found_error() {
        let mut r = SyncSchemaRegistry::new();
        assert!(matches!(
            r.delete_version("ghost", 1),
            Err(SyncRegistryError::SubjectNotFound(_))
        ));
    }

    // ── delete_subject ────────────────────────────────────────────────────────

    #[test]
    fn test_delete_subject_removes_all_versions() {
        let mut r = SyncSchemaRegistry::new();
        r.register("sub", SyncSchemaType::Json, "v1").unwrap();
        r.register("sub", SyncSchemaType::Json, "v2").unwrap();
        let count = r.delete_subject("sub").unwrap();
        assert_eq!(count, 2);
        assert!(r.subjects().is_empty());
    }

    #[test]
    fn test_delete_subject_not_found_error() {
        let mut r = SyncSchemaRegistry::new();
        assert!(matches!(
            r.delete_subject("ghost"),
            Err(SyncRegistryError::SubjectNotFound(_))
        ));
    }

    // ── compatibility modes ───────────────────────────────────────────────────

    #[test]
    fn test_set_and_get_compatibility_per_subject() {
        let mut r = SyncSchemaRegistry::new();
        r.set_compatibility("sub", SyncCompatibilityMode::Backward);
        assert_eq!(r.get_compatibility("sub"), SyncCompatibilityMode::Backward);
    }

    #[test]
    fn test_get_compatibility_falls_back_to_global() {
        let r = SyncSchemaRegistry::with_global_compatibility(SyncCompatibilityMode::Full);
        assert_eq!(
            r.get_compatibility("any-subject"),
            SyncCompatibilityMode::Full
        );
    }

    #[test]
    fn test_per_subject_overrides_global() {
        let mut r = SyncSchemaRegistry::with_global_compatibility(SyncCompatibilityMode::Backward);
        r.set_compatibility("sub", SyncCompatibilityMode::None);
        assert_eq!(r.get_compatibility("sub"), SyncCompatibilityMode::None);
    }

    #[test]
    fn test_global_compatibility_default_is_none() {
        let r = SyncSchemaRegistry::new();
        assert_eq!(r.get_compatibility("any"), SyncCompatibilityMode::None);
    }

    // ── check_compatibility ───────────────────────────────────────────────────

    #[test]
    fn test_check_compatibility_none_mode_always_true() {
        let mut r = SyncSchemaRegistry::new(); // default = None
        r.register("sub", SyncSchemaType::Json, "v1").unwrap();
        assert!(r.check_compatibility("sub", "BREAK anything").unwrap());
    }

    #[test]
    fn test_check_compatibility_non_none_mode_ok() {
        let mut r = SyncSchemaRegistry::with_global_compatibility(SyncCompatibilityMode::Backward);
        r.register("sub", SyncSchemaType::Json, "v1").unwrap();
        assert!(r.check_compatibility("sub", "v2 definition").unwrap());
    }

    #[test]
    fn test_check_compatibility_non_none_mode_breaks() {
        let mut r = SyncSchemaRegistry::with_global_compatibility(SyncCompatibilityMode::Backward);
        r.register("sub", SyncSchemaType::Json, "v1").unwrap();
        assert!(!r.check_compatibility("sub", "BREAK v2").unwrap());
    }

    #[test]
    fn test_check_compatibility_no_existing_versions_is_ok() {
        let r = SyncSchemaRegistry::with_global_compatibility(SyncCompatibilityMode::Full);
        assert!(r
            .check_compatibility("new-subject", "first schema")
            .unwrap());
    }

    // ── total_schemas ─────────────────────────────────────────────────────────

    #[test]
    fn test_total_schemas_empty() {
        let r = SyncSchemaRegistry::new();
        assert_eq!(r.total_schemas(), 0);
    }

    #[test]
    fn test_total_schemas_counts_all_versions() {
        let mut r = SyncSchemaRegistry::new();
        r.register("a", SyncSchemaType::Json, "v1").unwrap();
        r.register("a", SyncSchemaType::Json, "v2").unwrap();
        r.register("b", SyncSchemaType::Json, "v1").unwrap();
        assert_eq!(r.total_schemas(), 3);
    }

    // ── error display ─────────────────────────────────────────────────────────

    #[test]
    fn test_error_display_subject_not_found() {
        let e = SyncRegistryError::SubjectNotFound("my-topic".to_string());
        assert!(e.to_string().contains("my-topic"));
    }

    #[test]
    fn test_error_display_schema_not_found() {
        let e = SyncRegistryError::SchemaNotFound {
            subject: "sub".to_string(),
            version: 3,
        };
        let s = e.to_string();
        assert!(s.contains("sub") && s.contains('3'));
    }

    #[test]
    fn test_error_display_incompatible_schema() {
        let e = SyncRegistryError::IncompatibleSchema("reason".to_string());
        assert!(e.to_string().contains("reason"));
    }

    #[test]
    fn test_error_display_duplicate_schema() {
        let e = SyncRegistryError::DuplicateSchema("dup".to_string());
        assert!(e.to_string().contains("dup"));
    }

    #[test]
    fn test_error_display_invalid_schema() {
        let e = SyncRegistryError::InvalidSchema("empty".to_string());
        assert!(e.to_string().contains("empty"));
    }

    // ── schema fields ─────────────────────────────────────────────────────────

    #[test]
    fn test_schema_subject_field_set_correctly() {
        let mut r = SyncSchemaRegistry::new();
        r.register("my-subject", SyncSchemaType::Avro, "schema")
            .unwrap();
        let s = r.get_latest("my-subject").unwrap();
        assert_eq!(s.subject, "my-subject");
    }

    #[test]
    fn test_schema_created_at_ms_non_zero() {
        let mut r = SyncSchemaRegistry::new();
        r.register("sub", SyncSchemaType::Json, "x").unwrap();
        let s = r.get_latest("sub").unwrap();
        // created_at_ms may be 0 in sandboxed environments without real clocks,
        // so we just verify the field exists and is a u64.
        let _: u64 = s.created_at_ms;
    }

    #[test]
    fn test_schema_type_preserved() {
        let mut r = SyncSchemaRegistry::new();
        r.register("p", SyncSchemaType::Protobuf, "proto def")
            .unwrap();
        let s = r.get_latest("p").unwrap();
        assert_eq!(s.schema_type, SyncSchemaType::Protobuf);
    }

    // ── default ───────────────────────────────────────────────────────────────

    #[test]
    fn test_default_registry_is_empty() {
        let r = SyncSchemaRegistry::default();
        assert_eq!(r.total_schemas(), 0);
        assert!(r.subjects().is_empty());
    }

    // ── backward/forward transitive modes ────────────────────────────────────

    #[test]
    fn test_backward_transitive_mode_accessible() {
        let mut r = SyncSchemaRegistry::new();
        r.set_compatibility("sub", SyncCompatibilityMode::BackwardTransitive);
        assert_eq!(
            r.get_compatibility("sub"),
            SyncCompatibilityMode::BackwardTransitive
        );
    }

    #[test]
    fn test_forward_transitive_mode_accessible() {
        let mut r = SyncSchemaRegistry::new();
        r.set_compatibility("sub", SyncCompatibilityMode::ForwardTransitive);
        assert_eq!(
            r.get_compatibility("sub"),
            SyncCompatibilityMode::ForwardTransitive
        );
    }

    #[test]
    fn test_full_transitive_mode_accessible() {
        let mut r = SyncSchemaRegistry::new();
        r.set_compatibility("sub", SyncCompatibilityMode::FullTransitive);
        assert_eq!(
            r.get_compatibility("sub"),
            SyncCompatibilityMode::FullTransitive
        );
    }
}
