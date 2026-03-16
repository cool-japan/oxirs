//! SAMM Aspect Versioning
//!
//! Provides semver-style versioning for SAMM Aspect Models, including:
//!
//! - [`AspectVersion`] — parsed semver triple
//! - [`VersionedAspect`] — wraps an aspect with version metadata
//! - [`MigrationStep`] — a single versioned transition
//! - [`AspectMigrationRegistry`] — registry of migration paths with BFS search
//!
//! # Example
//!
//! ```rust
//! use oxirs_samm::versioning::{AspectVersion, AspectMigrationRegistry, MigrationStep};
//!
//! let v1 = AspectVersion::parse("1.0.0").expect("should succeed");
//! let v2 = AspectVersion::parse("2.0.0").expect("should succeed");
//!
//! assert!(v2.is_breaking_change(&v1));
//! assert!(v1.is_breaking_change(&v2));
//!
//! let mut registry = AspectMigrationRegistry::new();
//! registry.register_migration(MigrationStep {
//!     from: v1.clone(),
//!     to: v2.clone(),
//!     description: "Major API overhaul".to_string(),
//!     breaking: true,
//! });
//! assert!(registry.can_migrate(&v1, &v2));
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// AspectVersion
// ─────────────────────────────────────────────────────────────────────────────

/// A semver-style version triple for a SAMM Aspect Model.
///
/// Follows the `major.minor.patch` convention where:
/// - `major` bumps indicate breaking changes
/// - `minor` bumps indicate backwards-compatible feature additions
/// - `patch` bumps indicate backwards-compatible bug fixes
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AspectVersion {
    /// Major version component.
    pub major: u32,
    /// Minor version component.
    pub minor: u32,
    /// Patch version component.
    pub patch: u32,
}

impl AspectVersion {
    /// Create a new `AspectVersion` from components.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse a version string in `"major.minor.patch"` format.
    ///
    /// Returns `Err(String)` with a descriptive message if parsing fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_samm::versioning::AspectVersion;
    /// let v = AspectVersion::parse("1.2.3").expect("should succeed");
    /// assert_eq!(v.major, 1);
    /// assert_eq!(v.minor, 2);
    /// assert_eq!(v.patch, 3);
    /// ```
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(format!(
                "invalid version format '{}': expected 'major.minor.patch'",
                s
            ));
        }
        let major = parts[0]
            .parse::<u32>()
            .map_err(|_| format!("invalid major component '{}' in '{}'", parts[0], s))?;
        let minor = parts[1]
            .parse::<u32>()
            .map_err(|_| format!("invalid minor component '{}' in '{}'", parts[1], s))?;
        let patch = parts[2]
            .parse::<u32>()
            .map_err(|_| format!("invalid patch component '{}' in '{}'", parts[2], s))?;
        Ok(Self {
            major,
            minor,
            patch,
        })
    }

    /// Returns `true` if this version is backwards-compatible with `other`.
    ///
    /// Two versions are compatible when they share the same `major` component
    /// and `self >= other` (i.e., self is at least as recent as `other`).
    pub fn is_compatible_with(&self, other: &AspectVersion) -> bool {
        if self.major != other.major {
            return false;
        }
        // self must be >= other in terms of minor/patch
        if self.minor > other.minor {
            return true;
        }
        if self.minor == other.minor && self.patch >= other.patch {
            return true;
        }
        false
    }

    /// Returns `true` if upgrading from `other` to `self` is a breaking change.
    ///
    /// A breaking change occurs when the major version numbers differ.
    pub fn is_breaking_change(&self, other: &AspectVersion) -> bool {
        self.major != other.major
    }
}

impl fmt::Display for AspectVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MigrationStep
// ─────────────────────────────────────────────────────────────────────────────

/// A single migration step from one [`AspectVersion`] to another.
#[derive(Debug, Clone, PartialEq)]
pub struct MigrationStep {
    /// The source version for this migration.
    pub from: AspectVersion,
    /// The target version for this migration.
    pub to: AspectVersion,
    /// Human-readable description of what this migration does.
    pub description: String,
    /// Whether this migration is a breaking change.
    pub breaking: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// VersionedAspect
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps an aspect model with version metadata.
///
/// Type parameter `T` is the aspect type (e.g., `oxirs_samm::Aspect`).
#[derive(Debug, Clone)]
pub struct VersionedAspect<T> {
    /// The wrapped aspect model.
    pub aspect: T,
    /// The version of this aspect model.
    pub version: AspectVersion,
    /// Unix timestamp (seconds) when this version was created.
    pub created_at: u64,
    /// Whether this version has been deprecated.
    pub deprecated: bool,
}

impl<T> VersionedAspect<T> {
    /// Create a new `VersionedAspect`.
    pub fn new(aspect: T, version: AspectVersion, created_at: u64) -> Self {
        Self {
            aspect,
            version,
            created_at,
            deprecated: false,
        }
    }

    /// Mark this version as deprecated, consuming and returning `self`.
    pub fn deprecate(mut self) -> Self {
        self.deprecated = true;
        self
    }

    /// Returns `true` if this aspect's version matches `current` and is not deprecated.
    pub fn is_current(&self, current: &AspectVersion) -> bool {
        !self.deprecated && self.version == *current
    }

    /// Find the migration path from `from` to `to` using the given registry.
    ///
    /// Returns an empty `Vec` if `from == to` (no migration needed) or if the
    /// registry returns no path.
    pub fn migration_path(
        from: &AspectVersion,
        to: &AspectVersion,
        registry: &AspectMigrationRegistry,
    ) -> Vec<MigrationStep> {
        registry.find_path(from, to).unwrap_or_default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AspectMigrationRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Registry of migration steps between [`AspectVersion`]s.
///
/// Supports finding shortest migration paths via BFS.
#[derive(Debug, Default)]
pub struct AspectMigrationRegistry {
    /// All registered migration steps.
    steps: Vec<MigrationStep>,
}

impl AspectMigrationRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Register a migration step.
    pub fn register_migration(&mut self, step: MigrationStep) {
        self.steps.push(step);
    }

    /// Find the shortest migration path from `from` to `to` using BFS.
    ///
    /// Returns:
    /// - `Some(vec![])` if `from == to` (trivial no-op)
    /// - `Some(path)` with the migration steps if reachable
    /// - `None` if no path exists
    pub fn find_path(
        &self,
        from: &AspectVersion,
        to: &AspectVersion,
    ) -> Option<Vec<MigrationStep>> {
        if from == to {
            return Some(Vec::new());
        }

        // BFS: queue of (current_version, path_so_far)
        let mut visited: HashMap<AspectVersion, bool> = HashMap::new();
        let mut queue: VecDeque<(AspectVersion, Vec<MigrationStep>)> = VecDeque::new();

        queue.push_back((from.clone(), Vec::new()));
        visited.insert(from.clone(), true);

        while let Some((current, path)) = queue.pop_front() {
            // Collect next steps from current version
            for step in &self.steps {
                if step.from == current {
                    if visited.contains_key(&step.to) {
                        continue;
                    }
                    let mut new_path = path.clone();
                    new_path.push(step.clone());

                    if step.to == *to {
                        return Some(new_path);
                    }

                    visited.insert(step.to.clone(), true);
                    queue.push_back((step.to.clone(), new_path));
                }
            }
        }

        None
    }

    /// Returns `true` if there is any migration path from `from` to `to`.
    pub fn can_migrate(&self, from: &AspectVersion, to: &AspectVersion) -> bool {
        self.find_path(from, to).is_some()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> AspectVersion {
        AspectVersion::parse(s).expect("valid version string")
    }

    // ── AspectVersion ──────────────────────────────────────────────────────

    #[test]
    fn test_aspect_version_new() {
        let ver = AspectVersion::new(1, 2, 3);
        assert_eq!(ver.major, 1);
        assert_eq!(ver.minor, 2);
        assert_eq!(ver.patch, 3);
    }

    #[test]
    fn test_aspect_version_parse_valid() {
        let ver = v("2.5.11");
        assert_eq!(ver.major, 2);
        assert_eq!(ver.minor, 5);
        assert_eq!(ver.patch, 11);
    }

    #[test]
    fn test_aspect_version_parse_invalid_format() {
        let result = AspectVersion::parse("1.2");
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("invalid version format"));
    }

    #[test]
    fn test_aspect_version_parse_non_numeric() {
        let result = AspectVersion::parse("1.x.3");
        assert!(result.is_err());
    }

    #[test]
    fn test_aspect_version_parse_too_many_parts() {
        let result = AspectVersion::parse("1.2.3.4");
        assert!(result.is_err());
    }

    #[test]
    fn test_aspect_version_is_compatible_same_version() {
        let ver = v("1.0.0");
        assert!(ver.is_compatible_with(&ver));
    }

    #[test]
    fn test_aspect_version_is_compatible_minor_higher() {
        let newer = v("1.2.0");
        let older = v("1.1.0");
        assert!(newer.is_compatible_with(&older));
    }

    #[test]
    fn test_aspect_version_is_compatible_patch_higher() {
        let newer = v("1.0.5");
        let older = v("1.0.3");
        assert!(newer.is_compatible_with(&older));
    }

    #[test]
    fn test_aspect_version_not_compatible_different_major() {
        let v2 = v("2.0.0");
        let v1 = v("1.0.0");
        assert!(!v2.is_compatible_with(&v1));
        assert!(!v1.is_compatible_with(&v2));
    }

    #[test]
    fn test_aspect_version_not_compatible_older_minor() {
        let older = v("1.0.0");
        let newer = v("1.1.0");
        // older is NOT compatible with newer (older < newer)
        assert!(!older.is_compatible_with(&newer));
    }

    #[test]
    fn test_aspect_version_is_breaking_change_major_bump() {
        let v2 = v("2.0.0");
        let v1 = v("1.0.0");
        assert!(v2.is_breaking_change(&v1));
    }

    #[test]
    fn test_aspect_version_not_breaking_minor_bump() {
        let v1_1 = v("1.1.0");
        let v1_0 = v("1.0.0");
        assert!(!v1_1.is_breaking_change(&v1_0));
    }

    #[test]
    fn test_aspect_version_not_breaking_patch_bump() {
        let v1_0_1 = v("1.0.1");
        let v1_0_0 = v("1.0.0");
        assert!(!v1_0_1.is_breaking_change(&v1_0_0));
    }

    #[test]
    fn test_aspect_version_display() {
        let ver = AspectVersion::new(3, 4, 5);
        assert_eq!(ver.to_string(), "3.4.5");
    }

    #[test]
    fn test_aspect_version_ordering() {
        let mut versions = [v("1.2.0"), v("1.0.0"), v("2.0.0"), v("1.1.0")];
        versions.sort();
        assert_eq!(versions[0], v("1.0.0"));
        assert_eq!(versions[1], v("1.1.0"));
        assert_eq!(versions[2], v("1.2.0"));
        assert_eq!(versions[3], v("2.0.0"));
    }

    // ── MigrationStep ─────────────────────────────────────────────────────

    #[test]
    fn test_migration_step_fields() {
        let step = MigrationStep {
            from: v("1.0.0"),
            to: v("2.0.0"),
            description: "Breaking update".to_string(),
            breaking: true,
        };
        assert_eq!(step.from, v("1.0.0"));
        assert_eq!(step.to, v("2.0.0"));
        assert!(step.breaking);
        assert_eq!(step.description, "Breaking update");
    }

    // ── VersionedAspect ───────────────────────────────────────────────────

    #[test]
    fn test_versioned_aspect_new() {
        let va: VersionedAspect<String> =
            VersionedAspect::new("MyAspect".to_string(), v("1.0.0"), 1_700_000_000);
        assert_eq!(va.aspect, "MyAspect");
        assert_eq!(va.version, v("1.0.0"));
        assert_eq!(va.created_at, 1_700_000_000);
        assert!(!va.deprecated);
    }

    #[test]
    fn test_versioned_aspect_deprecate() {
        let va: VersionedAspect<String> =
            VersionedAspect::new("A".to_string(), v("1.0.0"), 0).deprecate();
        assert!(va.deprecated);
    }

    #[test]
    fn test_versioned_aspect_is_current_true() {
        let va: VersionedAspect<i32> = VersionedAspect::new(42, v("1.2.3"), 0);
        assert!(va.is_current(&v("1.2.3")));
    }

    #[test]
    fn test_versioned_aspect_is_current_false_deprecated() {
        let va: VersionedAspect<i32> = VersionedAspect::new(42, v("1.0.0"), 0).deprecate();
        assert!(!va.is_current(&v("1.0.0")));
    }

    #[test]
    fn test_versioned_aspect_is_current_false_different_version() {
        let va: VersionedAspect<i32> = VersionedAspect::new(42, v("1.0.0"), 0);
        assert!(!va.is_current(&v("2.0.0")));
    }

    // ── AspectMigrationRegistry ────────────────────────────────────────────

    #[test]
    fn test_migration_registry_empty_path_same_version() {
        let registry = AspectMigrationRegistry::new();
        let path = registry.find_path(&v("1.0.0"), &v("1.0.0"));
        assert_eq!(path, Some(vec![]));
    }

    #[test]
    fn test_migration_registry_direct_path() {
        let mut registry = AspectMigrationRegistry::new();
        registry.register_migration(MigrationStep {
            from: v("1.0.0"),
            to: v("2.0.0"),
            description: "Major upgrade".to_string(),
            breaking: true,
        });
        let path = registry
            .find_path(&v("1.0.0"), &v("2.0.0"))
            .expect("path should exist");
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].from, v("1.0.0"));
        assert_eq!(path[0].to, v("2.0.0"));
    }

    #[test]
    fn test_migration_registry_multi_step_path() {
        let mut registry = AspectMigrationRegistry::new();
        registry.register_migration(MigrationStep {
            from: v("1.0.0"),
            to: v("1.1.0"),
            description: "Feature add".to_string(),
            breaking: false,
        });
        registry.register_migration(MigrationStep {
            from: v("1.1.0"),
            to: v("2.0.0"),
            description: "Major upgrade".to_string(),
            breaking: true,
        });
        let path = registry
            .find_path(&v("1.0.0"), &v("2.0.0"))
            .expect("path should exist");
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].from, v("1.0.0"));
        assert_eq!(path[1].to, v("2.0.0"));
    }

    #[test]
    fn test_migration_registry_no_path() {
        let registry = AspectMigrationRegistry::new();
        let path = registry.find_path(&v("1.0.0"), &v("3.0.0"));
        assert!(path.is_none());
    }

    #[test]
    fn test_can_migrate_true() {
        let mut registry = AspectMigrationRegistry::new();
        registry.register_migration(MigrationStep {
            from: v("1.0.0"),
            to: v("2.0.0"),
            description: "".to_string(),
            breaking: true,
        });
        assert!(registry.can_migrate(&v("1.0.0"), &v("2.0.0")));
    }

    #[test]
    fn test_can_migrate_false() {
        let registry = AspectMigrationRegistry::new();
        assert!(!registry.can_migrate(&v("1.0.0"), &v("9.9.9")));
    }

    #[test]
    fn test_migration_path_via_versioned_aspect() {
        let mut registry = AspectMigrationRegistry::new();
        registry.register_migration(MigrationStep {
            from: v("1.0.0"),
            to: v("2.0.0"),
            description: "bump".to_string(),
            breaking: true,
        });
        let path = VersionedAspect::<()>::migration_path(&v("1.0.0"), &v("2.0.0"), &registry);
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn test_migration_path_no_path_returns_empty() {
        let registry = AspectMigrationRegistry::new();
        let path = VersionedAspect::<()>::migration_path(&v("1.0.0"), &v("5.0.0"), &registry);
        assert!(path.is_empty());
    }

    #[test]
    fn test_migration_registry_three_hop_path() {
        let mut registry = AspectMigrationRegistry::new();
        registry.register_migration(MigrationStep {
            from: v("1.0.0"),
            to: v("1.1.0"),
            description: "step 1".to_string(),
            breaking: false,
        });
        registry.register_migration(MigrationStep {
            from: v("1.1.0"),
            to: v("1.2.0"),
            description: "step 2".to_string(),
            breaking: false,
        });
        registry.register_migration(MigrationStep {
            from: v("1.2.0"),
            to: v("2.0.0"),
            description: "step 3".to_string(),
            breaking: true,
        });
        let path = registry
            .find_path(&v("1.0.0"), &v("2.0.0"))
            .expect("path should exist");
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_aspect_version_equality() {
        assert_eq!(v("1.2.3"), v("1.2.3"));
        assert_ne!(v("1.2.3"), v("1.2.4"));
    }

    #[test]
    fn test_migration_step_non_breaking() {
        let step = MigrationStep {
            from: v("1.0.0"),
            to: v("1.1.0"),
            description: "minor bump".to_string(),
            breaking: false,
        };
        assert!(!step.breaking);
    }
}
