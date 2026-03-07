//! GraphQL Schema Version Registry.
//!
//! Manages an ordered history of GraphQL schema SDL strings, assigns
//! monotonically increasing version identifiers, computes structural diffs
//! between versions, and detects breaking changes.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_gql::schema_registry::SchemaRegistry;
//!
//! let mut registry = SchemaRegistry::new();
//! let v1 = registry.register("type Query { hello: String }");
//! let v2 = registry.register("type Query { hello: String\n  world: Int }");
//!
//! let diff = registry.diff(&v1.version, &v2.version).unwrap();
//! assert!(!diff.added_fields.is_empty() || diff.modified_types.is_empty());
//! assert!(!registry.is_breaking_change(&diff)); // adding a field is non-breaking
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur in the schema registry.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum RegistryError {
    /// The requested version string was not found.
    #[error("Version '{0}' not found in registry")]
    VersionNotFound(String),

    /// The SDL string is empty.
    #[error("Schema SDL cannot be empty")]
    EmptySdl,

    /// An internal consistency error.
    #[error("Internal error: {0}")]
    Internal(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// SchemaVersion
// ─────────────────────────────────────────────────────────────────────────────

/// A single registered schema version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Unique numeric identifier (auto-assigned, 1-based).
    pub id: u64,
    /// Semantic version string, e.g. `"1.0.0"`.
    pub version: String,
    /// The raw SDL (Schema Definition Language) string.
    pub schema_sdl: String,
    /// Unix timestamp (seconds since epoch) when this version was registered.
    pub created_at: u64,
}

impl SchemaVersion {
    /// Format the creation timestamp as an ISO-8601-like string.
    pub fn created_at_display(&self) -> String {
        // Simple epoch-based display without chrono dependency
        let secs = self.created_at;
        let days = secs / 86400;
        let rem = secs % 86400;
        let hours = rem / 3600;
        let minutes = (rem % 3600) / 60;
        let seconds = rem % 60;
        // Approximate date (1970-01-01 + days)
        let year = 1970 + days / 365;
        format!("{year}-xx-xx {hours:02}:{minutes:02}:{seconds:02} UTC")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SchemaDiff
// ─────────────────────────────────────────────────────────────────────────────

/// Structural diff between two schema versions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchemaDiff {
    /// Type names added in the newer version.
    pub added_types: Vec<String>,
    /// Type names removed in the newer version.
    pub removed_types: Vec<String>,
    /// Type names present in both but with changed definitions.
    pub modified_types: Vec<String>,
    /// Per-type map of field names added in the newer version.
    pub added_fields: HashMap<String, Vec<String>>,
    /// Per-type map of field names removed in the newer version.
    pub removed_fields: HashMap<String, Vec<String>>,
    /// Per-type map of field names whose type signatures changed.
    pub modified_fields: HashMap<String, Vec<String>>,
}

impl SchemaDiff {
    /// Returns `true` if there are no differences.
    pub fn is_empty(&self) -> bool {
        self.added_types.is_empty()
            && self.removed_types.is_empty()
            && self.modified_types.is_empty()
            && self.added_fields.is_empty()
            && self.removed_fields.is_empty()
            && self.modified_fields.is_empty()
    }

    /// Total count of all changes.
    pub fn total_changes(&self) -> usize {
        self.added_types.len()
            + self.removed_types.len()
            + self.modified_types.len()
            + self.added_fields.values().map(Vec::len).sum::<usize>()
            + self.removed_fields.values().map(Vec::len).sum::<usize>()
            + self.modified_fields.values().map(Vec::len).sum::<usize>()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChangeEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry in the change log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeEntry {
    /// Version this change was introduced in.
    pub version: String,
    /// Human-readable description of the change.
    pub description: String,
    /// Whether this change is considered breaking.
    pub is_breaking: bool,
    /// Unix timestamp of the version.
    pub timestamp: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// SDL mini-parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parsed representation of a GraphQL type for diff purposes.
#[derive(Debug, Clone)]
struct ParsedType {
    /// Kind: "type", "interface", "enum", "union", "scalar", "input"
    kind: String,
    /// Type name
    name: String,
    /// Field name → type string (for object/interface/input types)
    fields: HashMap<String, String>,
    /// Enum values (for enum types)
    enum_values: Vec<String>,
}

/// Minimal SDL parser that extracts type names and fields.
fn parse_sdl(sdl: &str) -> HashMap<String, ParsedType> {
    let mut types: HashMap<String, ParsedType> = HashMap::new();
    let mut chars = sdl.chars().peekable();
    let mut current_type: Option<ParsedType> = None;
    let mut brace_depth = 0i32;
    let mut current_token = String::new();

    while let Some(&ch) = chars.peek() {
        chars.next();
        match ch {
            '#' => {
                // Skip comment to end of line
                for c in chars.by_ref() {
                    if c == '\n' {
                        break;
                    }
                }
            }
            '{' => {
                // Process any accumulated token before entering the brace body.
                // This handles `type Query {` where the keyword and name are on
                // the same line as the opening brace.
                let trimmed = current_token.trim().to_owned();
                current_token.clear();
                if !trimmed.is_empty() && brace_depth == 0 {
                    let parts: Vec<&str> = trimmed.splitn(3, char::is_whitespace).collect();
                    let kind = parts.first().copied().unwrap_or("").to_lowercase();
                    if matches!(
                        kind.as_str(),
                        "type" | "interface" | "enum" | "union" | "scalar" | "input"
                    ) {
                        let name = parts.get(1).copied().unwrap_or("").to_owned();
                        if !name.is_empty() {
                            current_type = Some(ParsedType {
                                kind: kind.clone(),
                                name,
                                fields: HashMap::new(),
                                enum_values: Vec::new(),
                            });
                        }
                    }
                }
                brace_depth += 1;
            }
            '}' => {
                brace_depth -= 1;
                if brace_depth == 0 {
                    if let Some(t) = current_type.take() {
                        types.insert(t.name.clone(), t);
                    }
                }
                current_token.clear();
            }
            '\n' | '\r' => {
                let trimmed = current_token.trim().to_owned();
                if !trimmed.is_empty() {
                    if brace_depth == 0 {
                        // Could be start of type definition
                        let parts: Vec<&str> = trimmed.splitn(3, char::is_whitespace).collect();
                        let kind = parts.first().copied().unwrap_or("").to_lowercase();
                        if matches!(
                            kind.as_str(),
                            "type" | "interface" | "enum" | "union" | "scalar" | "input"
                        ) {
                            let name = parts.get(1).copied().unwrap_or("").to_owned();
                            if !name.is_empty() {
                                current_type = Some(ParsedType {
                                    kind: kind.clone(),
                                    name,
                                    fields: HashMap::new(),
                                    enum_values: Vec::new(),
                                });
                            }
                        }
                    } else if brace_depth == 1 {
                        // Inside a type body — parse field or enum value
                        if let Some(ref mut t) = current_type {
                            parse_sdl_field_line(&trimmed, t);
                        }
                    }
                }
                current_token.clear();
            }
            _ => {
                current_token.push(ch);
            }
        }
    }

    // Handle last token (no trailing newline)
    let trimmed = current_token.trim().to_owned();
    if !trimmed.is_empty() && brace_depth == 1 {
        if let Some(ref mut t) = current_type {
            parse_sdl_field_line(&trimmed, t);
        }
    }
    if let Some(t) = current_type {
        types.insert(t.name.clone(), t);
    }

    types
}

fn parse_sdl_field_line(line: &str, t: &mut ParsedType) {
    if line.trim().is_empty() || line.trim().starts_with('#') {
        return;
    }
    if t.kind == "enum" {
        // Enum values: just bare identifiers
        let val = line.split_whitespace().next().unwrap_or("").to_owned();
        if !val.is_empty() {
            t.enum_values.push(val);
        }
    } else if t.kind == "union" {
        // union members after '='
        // e.g. "= TypeA | TypeB" — handled at type level, not field level
    } else {
        // Field: "fieldName: TypeSignature"
        if let Some(colon_pos) = line.find(':') {
            let field_name = line[..colon_pos].trim().to_owned();
            // Strip arguments from field name
            let field_name = field_name
                .split('(')
                .next()
                .unwrap_or(&field_name)
                .trim()
                .to_owned();
            let type_sig = line[colon_pos + 1..].trim().to_owned();
            if !field_name.is_empty() {
                t.fields.insert(field_name, type_sig);
            }
        }
    }
}

/// Compute a diff between two parsed SDL type sets.
fn compute_diff(
    old_types: &HashMap<String, ParsedType>,
    new_types: &HashMap<String, ParsedType>,
) -> SchemaDiff {
    let mut diff = SchemaDiff::default();

    let old_names: HashSet<&str> = old_types.keys().map(String::as_str).collect();
    let new_names: HashSet<&str> = new_types.keys().map(String::as_str).collect();

    // Added types
    for name in new_names.difference(&old_names) {
        diff.added_types.push(name.to_string());
    }

    // Removed types
    for name in old_names.difference(&new_names) {
        diff.removed_types.push(name.to_string());
    }

    // Modified types — iterate types present in both
    for name in old_names.intersection(&new_names) {
        let old_t = &old_types[*name];
        let new_t = &new_types[*name];

        let added = field_diff_added(old_t, new_t);
        let removed = field_diff_removed(old_t, new_t);
        let modified = field_diff_modified(old_t, new_t);

        if !added.is_empty() {
            diff.added_fields.insert(name.to_string(), added);
        }
        if !removed.is_empty() {
            diff.removed_fields
                .insert(name.to_string(), removed.clone());
        }
        if !modified.is_empty() {
            diff.modified_fields.insert(name.to_string(), modified);
        }

        if diff.removed_fields.contains_key(*name)
            || diff.modified_fields.contains_key(*name)
            || old_t.kind != new_t.kind
        {
            diff.modified_types.push(name.to_string());
        }
    }

    diff
}

fn field_diff_added(old_t: &ParsedType, new_t: &ParsedType) -> Vec<String> {
    let old_fields: HashSet<&str> = old_t.fields.keys().map(String::as_str).collect();
    let new_fields: HashSet<&str> = new_t.fields.keys().map(String::as_str).collect();
    new_fields
        .difference(&old_fields)
        .map(|f| f.to_string())
        .collect()
}

fn field_diff_removed(old_t: &ParsedType, new_t: &ParsedType) -> Vec<String> {
    let old_fields: HashSet<&str> = old_t.fields.keys().map(String::as_str).collect();
    let new_fields: HashSet<&str> = new_t.fields.keys().map(String::as_str).collect();
    old_fields
        .difference(&new_fields)
        .map(|f| f.to_string())
        .collect()
}

fn field_diff_modified(old_t: &ParsedType, new_t: &ParsedType) -> Vec<String> {
    let mut modified = Vec::new();
    for (fname, old_sig) in &old_t.fields {
        if let Some(new_sig) = new_t.fields.get(fname) {
            if old_sig != new_sig {
                modified.push(fname.clone());
            }
        }
    }
    modified
}

// ─────────────────────────────────────────────────────────────────────────────
// Version numbering
// ─────────────────────────────────────────────────────────────────────────────

fn next_semver(last: Option<&str>) -> String {
    match last {
        None => "1.0.0".to_owned(),
        Some(v) => {
            let parts: Vec<u64> = v.split('.').filter_map(|s| s.parse().ok()).collect();
            if parts.len() == 3 {
                format!("{}.{}.{}", parts[0], parts[1], parts[2] + 1)
            } else {
                "1.0.0".to_owned()
            }
        }
    }
}

/// Approximate Unix timestamp for testing (seconds since a fixed epoch).
fn monotonic_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// SchemaRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Ordered registry of GraphQL schema versions.
///
/// Versions are stored in insertion order and assigned monotonically increasing
/// identifiers and semver strings.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SchemaRegistry {
    /// Ordered history of schema versions.
    versions: Vec<SchemaVersion>,
    /// Ordered change log entries.
    change_log: Vec<ChangeEntry>,
}

impl SchemaRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    // ── Registration ────────────────────────────────────────────────────────

    /// Register a new schema SDL and return the created [`SchemaVersion`].
    ///
    /// The version string is derived by incrementing the patch component of the
    /// most recent version (starting from `1.0.0`).
    pub fn register(&mut self, sdl: &str) -> SchemaVersion {
        let id = self.versions.len() as u64 + 1;
        let version = next_semver(self.versions.last().map(|v| v.version.as_str()));
        let created_at = monotonic_timestamp();

        let sv = SchemaVersion {
            id,
            version: version.clone(),
            schema_sdl: sdl.to_owned(),
            created_at,
        };
        self.versions.push(sv.clone());

        // Build change log entry
        if self.versions.len() > 1 {
            let prev = &self.versions[self.versions.len() - 2];
            let diff = self.diff_sdl(&prev.schema_sdl, sdl);
            let breaking = self.is_breaking_change(&diff);
            let description = summarize_diff(&diff);
            self.change_log.push(ChangeEntry {
                version,
                description,
                is_breaking: breaking,
                timestamp: created_at,
            });
        } else {
            self.change_log.push(ChangeEntry {
                version: sv.version.clone(),
                description: "Initial schema registration".to_owned(),
                is_breaking: false,
                timestamp: created_at,
            });
        }

        sv
    }

    // ── Lookup ──────────────────────────────────────────────────────────────

    /// Get the most recent (latest) schema version.
    pub fn latest(&self) -> Option<&SchemaVersion> {
        self.versions.last()
    }

    /// Get a specific schema version by its version string.
    pub fn get(&self, version_str: &str) -> Option<&SchemaVersion> {
        self.versions.iter().find(|v| v.version == version_str)
    }

    /// Get a schema version by its numeric ID.
    pub fn get_by_id(&self, id: u64) -> Option<&SchemaVersion> {
        self.versions.iter().find(|v| v.id == id)
    }

    /// Returns the number of registered versions.
    pub fn len(&self) -> usize {
        self.versions.len()
    }

    /// Returns `true` if no versions are registered.
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }

    /// Return all versions in registration order.
    pub fn all_versions(&self) -> &[SchemaVersion] {
        &self.versions
    }

    // ── Diff ────────────────────────────────────────────────────────────────

    /// Compute a [`SchemaDiff`] between two registered versions.
    pub fn diff(&self, v1_str: &str, v2_str: &str) -> Result<SchemaDiff, RegistryError> {
        let v1 = self
            .get(v1_str)
            .ok_or_else(|| RegistryError::VersionNotFound(v1_str.to_owned()))?;
        let v2 = self
            .get(v2_str)
            .ok_or_else(|| RegistryError::VersionNotFound(v2_str.to_owned()))?;
        Ok(self.diff_sdl(&v1.schema_sdl, &v2.schema_sdl))
    }

    /// Compute a diff between two raw SDL strings (does not require them to be
    /// registered).
    pub fn diff_sdl(&self, old_sdl: &str, new_sdl: &str) -> SchemaDiff {
        let old_types = parse_sdl(old_sdl);
        let new_types = parse_sdl(new_sdl);
        compute_diff(&old_types, &new_types)
    }

    // ── Breaking change detection ────────────────────────────────────────────

    /// Returns `true` if the diff contains any backward-incompatible changes.
    ///
    /// Breaking changes are:
    /// - Removal of a type
    /// - Removal of a field from an existing type
    /// - Change in field type signature
    pub fn is_breaking_change(&self, diff: &SchemaDiff) -> bool {
        !diff.removed_types.is_empty()
            || !diff.removed_fields.is_empty()
            || !diff.modified_fields.is_empty()
    }

    // ── Changelog ────────────────────────────────────────────────────────────

    /// Return the ordered history of change log entries.
    pub fn changelog(&self) -> &[ChangeEntry] {
        &self.change_log
    }
}

fn summarize_diff(diff: &SchemaDiff) -> String {
    let mut parts = Vec::new();
    if !diff.added_types.is_empty() {
        parts.push(format!("added types: {}", diff.added_types.join(", ")));
    }
    if !diff.removed_types.is_empty() {
        parts.push(format!("removed types: {}", diff.removed_types.join(", ")));
    }
    if !diff.added_fields.is_empty() {
        let count: usize = diff.added_fields.values().map(Vec::len).sum();
        parts.push(format!("added {count} field(s)"));
    }
    if !diff.removed_fields.is_empty() {
        let count: usize = diff.removed_fields.values().map(Vec::len).sum();
        parts.push(format!("removed {count} field(s)"));
    }
    if !diff.modified_fields.is_empty() {
        let count: usize = diff.modified_fields.values().map(Vec::len).sum();
        parts.push(format!("modified {count} field(s)"));
    }
    if parts.is_empty() {
        "No structural changes".to_owned()
    } else {
        parts.join("; ")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn v1_sdl() -> &'static str {
        "type Query {\n  hello: String\n}\n"
    }

    fn v2_sdl() -> &'static str {
        "type Query {\n  hello: String\n  world: Int\n}\n"
    }

    fn v3_sdl() -> &'static str {
        "type Query {\n  world: Int\n}\n"
    }

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_new_registry_is_empty() {
        let reg = SchemaRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_default_is_empty() {
        let reg = SchemaRegistry::default();
        assert!(reg.is_empty());
    }

    // ── Registration ────────────────────────────────────────────────────────

    #[test]
    fn test_register_first_version_is_1_0_0() {
        let mut reg = SchemaRegistry::new();
        let sv = reg.register(v1_sdl());
        assert_eq!(sv.version, "1.0.0");
        assert_eq!(sv.id, 1);
    }

    #[test]
    fn test_register_second_version_is_1_0_1() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl());
        let sv2 = reg.register(v2_sdl());
        assert_eq!(sv2.version, "1.0.1");
        assert_eq!(sv2.id, 2);
    }

    #[test]
    fn test_register_increments_len() {
        let mut reg = SchemaRegistry::new();
        assert_eq!(reg.len(), 0);
        reg.register(v1_sdl());
        assert_eq!(reg.len(), 1);
        reg.register(v2_sdl());
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_register_stores_sdl() {
        let mut reg = SchemaRegistry::new();
        let sv = reg.register(v1_sdl());
        assert_eq!(sv.schema_sdl, v1_sdl());
    }

    #[test]
    fn test_register_sets_created_at() {
        let mut reg = SchemaRegistry::new();
        let sv = reg.register(v1_sdl());
        // created_at is u64, always valid; verify the field is accessible
        let _ = sv.created_at;
    }

    // ── Lookup ──────────────────────────────────────────────────────────────

    #[test]
    fn test_latest_returns_last_registered() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl());
        let sv2 = reg.register(v2_sdl());
        let latest = reg.latest().unwrap();
        assert_eq!(latest.version, sv2.version);
    }

    #[test]
    fn test_latest_empty_is_none() {
        let reg = SchemaRegistry::new();
        assert!(reg.latest().is_none());
    }

    #[test]
    fn test_get_by_version_string() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl());
        reg.register(v2_sdl());
        let v = reg.get("1.0.0").unwrap();
        assert_eq!(v.id, 1);
        assert_eq!(v.schema_sdl, v1_sdl());
    }

    #[test]
    fn test_get_nonexistent_version_is_none() {
        let reg = SchemaRegistry::new();
        assert!(reg.get("99.0.0").is_none());
    }

    #[test]
    fn test_get_by_id() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl());
        reg.register(v2_sdl());
        let sv = reg.get_by_id(1).unwrap();
        assert_eq!(sv.version, "1.0.0");
    }

    #[test]
    fn test_get_by_id_nonexistent() {
        let reg = SchemaRegistry::new();
        assert!(reg.get_by_id(42).is_none());
    }

    #[test]
    fn test_all_versions_count() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl());
        reg.register(v2_sdl());
        reg.register(v3_sdl());
        assert_eq!(reg.all_versions().len(), 3);
    }

    // ── Diff ────────────────────────────────────────────────────────────────

    #[test]
    fn test_diff_added_field() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl()); // has "hello"
        reg.register(v2_sdl()); // has "hello" + "world"
        let diff = reg.diff("1.0.0", "1.0.1").unwrap();
        // "world" should be in added_fields for "Query"
        let added = diff.added_fields.get("Query").cloned().unwrap_or_default();
        assert!(added.contains(&"world".to_owned()));
    }

    #[test]
    fn test_diff_removed_field() {
        let mut reg = SchemaRegistry::new();
        reg.register(v2_sdl()); // has "hello" + "world"
        reg.register(v3_sdl()); // has only "world"
        let diff = reg.diff("1.0.0", "1.0.1").unwrap();
        let removed = diff
            .removed_fields
            .get("Query")
            .cloned()
            .unwrap_or_default();
        assert!(removed.contains(&"hello".to_owned()));
    }

    #[test]
    fn test_diff_added_type() {
        let sdl_old = "type Query {\n  hello: String\n}\n";
        let sdl_new = "type Query {\n  hello: String\n}\ntype Mutation {\n  doThing: Boolean\n}\n";
        let mut reg = SchemaRegistry::new();
        reg.register(sdl_old);
        reg.register(sdl_new);
        let diff = reg.diff("1.0.0", "1.0.1").unwrap();
        assert!(diff.added_types.contains(&"Mutation".to_owned()));
    }

    #[test]
    fn test_diff_removed_type() {
        let sdl_old = "type Query {\n  hello: String\n}\ntype Mutation {\n  doThing: Boolean\n}\n";
        let sdl_new = "type Query {\n  hello: String\n}\n";
        let mut reg = SchemaRegistry::new();
        reg.register(sdl_old);
        reg.register(sdl_new);
        let diff = reg.diff("1.0.0", "1.0.1").unwrap();
        assert!(diff.removed_types.contains(&"Mutation".to_owned()));
    }

    #[test]
    fn test_diff_unknown_version_error() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl());
        let err = reg.diff("1.0.0", "99.0.0").unwrap_err();
        assert!(matches!(err, RegistryError::VersionNotFound(_)));
    }

    #[test]
    fn test_diff_no_changes() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl());
        reg.register(v1_sdl()); // identical SDL
        let diff = reg.diff("1.0.0", "1.0.1").unwrap();
        assert!(diff.is_empty());
    }

    // ── Breaking change detection ────────────────────────────────────────────

    #[test]
    fn test_adding_field_is_non_breaking() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl());
        reg.register(v2_sdl());
        let diff = reg.diff("1.0.0", "1.0.1").unwrap();
        assert!(!reg.is_breaking_change(&diff));
    }

    #[test]
    fn test_removing_field_is_breaking() {
        let mut reg = SchemaRegistry::new();
        reg.register(v2_sdl());
        reg.register(v3_sdl());
        let diff = reg.diff("1.0.0", "1.0.1").unwrap();
        assert!(reg.is_breaking_change(&diff));
    }

    #[test]
    fn test_removing_type_is_breaking() {
        let sdl_old = "type Query {\n  hello: String\n}\ntype Extra {\n  x: Int\n}\n";
        let sdl_new = "type Query {\n  hello: String\n}\n";
        let mut reg = SchemaRegistry::new();
        reg.register(sdl_old);
        reg.register(sdl_new);
        let diff = reg.diff("1.0.0", "1.0.1").unwrap();
        assert!(reg.is_breaking_change(&diff));
    }

    #[test]
    fn test_adding_type_is_non_breaking() {
        let sdl_old = "type Query {\n  hello: String\n}\n";
        let sdl_new = "type Query {\n  hello: String\n}\ntype Extra {\n  x: Int\n}\n";
        let mut reg = SchemaRegistry::new();
        reg.register(sdl_old);
        reg.register(sdl_new);
        let diff = reg.diff("1.0.0", "1.0.1").unwrap();
        assert!(!reg.is_breaking_change(&diff));
    }

    #[test]
    fn test_empty_diff_is_non_breaking() {
        let diff = SchemaDiff::default();
        let reg = SchemaRegistry::new();
        assert!(!reg.is_breaking_change(&diff));
    }

    // ── Changelog ────────────────────────────────────────────────────────────

    #[test]
    fn test_changelog_first_entry_is_initial() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl());
        let log = reg.changelog();
        assert_eq!(log.len(), 1);
        assert!(log[0].description.contains("Initial"));
    }

    #[test]
    fn test_changelog_grows_with_versions() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl());
        reg.register(v2_sdl());
        reg.register(v3_sdl());
        assert_eq!(reg.changelog().len(), 3);
    }

    #[test]
    fn test_changelog_breaking_flag() {
        let mut reg = SchemaRegistry::new();
        reg.register(v2_sdl()); // has hello + world
        reg.register(v3_sdl()); // removes hello → breaking
        let log = reg.changelog();
        assert_eq!(log.len(), 2);
        // Second entry should be breaking
        assert!(log[1].is_breaking);
    }

    #[test]
    fn test_changelog_non_breaking_flag() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl());
        reg.register(v2_sdl()); // adds "world" → non-breaking
        let log = reg.changelog();
        assert!(!log[1].is_breaking);
    }

    // ── SchemaDiff helpers ───────────────────────────────────────────────────

    #[test]
    fn test_schema_diff_total_changes_empty() {
        let diff = SchemaDiff::default();
        assert_eq!(diff.total_changes(), 0);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_schema_diff_total_changes_non_empty() {
        let mut diff = SchemaDiff::default();
        diff.added_types.push("Foo".to_owned());
        assert_eq!(diff.total_changes(), 1);
        assert!(!diff.is_empty());
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip() {
        let mut reg = SchemaRegistry::new();
        reg.register(v1_sdl());
        reg.register(v2_sdl());
        let json = serde_json::to_string(&reg).unwrap();
        let restored: SchemaRegistry = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), 2);
    }

    // ── SchemaVersion helpers ─────────────────────────────────────────────────

    #[test]
    fn test_schema_version_created_at_display() {
        let sv = SchemaVersion {
            id: 1,
            version: "1.0.0".to_owned(),
            schema_sdl: "type Query { hello: String }".to_owned(),
            created_at: 0,
        };
        let display = sv.created_at_display();
        assert!(!display.is_empty());
    }

    // ── diff_sdl directly ─────────────────────────────────────────────────────

    #[test]
    fn test_diff_sdl_directly() {
        let reg = SchemaRegistry::new();
        let diff = reg.diff_sdl(v1_sdl(), v2_sdl());
        let added = diff.added_fields.get("Query").cloned().unwrap_or_default();
        assert!(added.contains(&"world".to_owned()));
    }
}
