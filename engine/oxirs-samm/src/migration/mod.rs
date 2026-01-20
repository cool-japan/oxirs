//! Model versioning and migration tools
//!
//! This module provides tools for migrating SAMM models between versions.
//! It handles namespace changes, deprecated features, and structural updates
//! required when upgrading from older SAMM/BAMM versions.
//!
//! # Supported Migrations
//!
//! - BAMM (any version) → SAMM 2.0.0
//! - SAMM 2.0.0 → SAMM 2.1.0
//! - SAMM 2.1.0 → SAMM 2.3.0
//! - Direct migration from any version to target version
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_samm::migration::{ModelMigrator, MigrationOptions, SammVersion};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create migrator with options
//! let options = MigrationOptions {
//!     target_version: SammVersion::V2_3_0,
//!     preserve_comments: true,
//!     dry_run: false,
//!     ..Default::default()
//! };
//!
//! let migrator = ModelMigrator::new(options);
//!
//! // Migrate a model file
//! let ttl_content = std::fs::read_to_string("old_model.ttl")?;
//! let result = migrator.migrate(&ttl_content)?;
//!
//! println!("Migration report:");
//! for change in &result.changes {
//!     println!("  - {}", change);
//! }
//!
//! // Write migrated content
//! std::fs::write("new_model.ttl", result.content)?;
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, SammError};
use regex::Regex;
use std::collections::HashMap;

/// SAMM version identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SammVersion {
    /// BAMM (legacy, any version)
    Bamm,
    /// SAMM 1.0.0
    V1_0_0,
    /// SAMM 2.0.0
    V2_0_0,
    /// SAMM 2.1.0
    V2_1_0,
    /// SAMM 2.3.0 (current)
    V2_3_0,
    /// Unknown version
    Unknown,
}

impl SammVersion {
    /// Parse version from string
    pub fn parse(s: &str) -> Self {
        if s.contains("bamm") {
            Self::Bamm
        } else if s.contains("1.0.0") {
            Self::V1_0_0
        } else if s.contains("2.0.0") {
            Self::V2_0_0
        } else if s.contains("2.1.0") {
            Self::V2_1_0
        } else if s.contains("2.3.0") {
            Self::V2_3_0
        } else {
            Self::Unknown
        }
    }

    /// Get version string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bamm => "bamm",
            Self::V1_0_0 => "1.0.0",
            Self::V2_0_0 => "2.0.0",
            Self::V2_1_0 => "2.1.0",
            Self::V2_3_0 => "2.3.0",
            Self::Unknown => "unknown",
        }
    }

    /// Check if this version is older than another
    pub fn is_older_than(&self, other: &Self) -> bool {
        self < other
    }
}

/// Migration options
#[derive(Debug, Clone)]
pub struct MigrationOptions {
    /// Target SAMM version
    pub target_version: SammVersion,
    /// Preserve comments during migration
    pub preserve_comments: bool,
    /// Dry run mode (don't apply changes)
    pub dry_run: bool,
    /// Generate migration report
    pub generate_report: bool,
    /// Automatically fix common issues
    pub auto_fix: bool,
    /// Backup original content
    pub create_backup: bool,
}

impl Default for MigrationOptions {
    fn default() -> Self {
        Self {
            target_version: SammVersion::V2_3_0,
            preserve_comments: true,
            dry_run: false,
            generate_report: true,
            auto_fix: true,
            create_backup: true,
        }
    }
}

/// Migration result
#[derive(Debug, Clone)]
pub struct MigrationResult {
    /// Migrated content
    pub content: String,
    /// Original version detected
    pub from_version: SammVersion,
    /// Target version
    pub to_version: SammVersion,
    /// List of changes made
    pub changes: Vec<String>,
    /// Warnings encountered
    pub warnings: Vec<String>,
    /// Original content (if backup was requested)
    pub backup: Option<String>,
}

/// Model migrator
pub struct ModelMigrator {
    options: MigrationOptions,
    rules: HashMap<(SammVersion, SammVersion), Vec<MigrationRule>>,
}

/// Migration rule
#[derive(Debug, Clone)]
struct MigrationRule {
    name: String,
    pattern: Regex,
    replacement: String,
    description: String,
}

impl ModelMigrator {
    /// Create a new migrator with options
    pub fn new(options: MigrationOptions) -> Self {
        let mut migrator = Self {
            options,
            rules: HashMap::new(),
        };
        migrator.initialize_rules();
        migrator
    }

    /// Detect the SAMM version used in a model
    pub fn detect_version(&self, content: &str) -> SammVersion {
        // Check for BAMM namespace
        if content.contains("urn:bamm:") || content.contains("bamm:") {
            return SammVersion::Bamm;
        }

        // Check for SAMM version in namespace URIs
        if content.contains("samm:meta-model:2.3.0") {
            return SammVersion::V2_3_0;
        }
        if content.contains("samm:meta-model:2.1.0") {
            return SammVersion::V2_1_0;
        }
        if content.contains("samm:meta-model:2.0.0") {
            return SammVersion::V2_0_0;
        }
        if content.contains("samm:meta-model:1.0.0") {
            return SammVersion::V1_0_0;
        }

        SammVersion::Unknown
    }

    /// Migrate a model to the target version
    pub fn migrate(&self, content: &str) -> Result<MigrationResult> {
        let from_version = self.detect_version(content);

        if from_version == SammVersion::Unknown {
            return Err(SammError::Other(
                "Could not detect SAMM/BAMM version in model".to_string(),
            ));
        }

        let to_version = self.options.target_version;

        // No migration needed if already at target version
        if from_version == to_version {
            return Ok(MigrationResult {
                content: content.to_string(),
                from_version,
                to_version,
                changes: vec!["Model is already at target version".to_string()],
                warnings: vec![],
                backup: if self.options.create_backup {
                    Some(content.to_string())
                } else {
                    None
                },
            });
        }

        if from_version > to_version {
            return Err(SammError::Other(format!(
                "Cannot downgrade from {} to {}",
                from_version.as_str(),
                to_version.as_str()
            )));
        }

        // Apply migration path
        let mut current_content = content.to_string();
        let mut all_changes = Vec::new();
        let mut all_warnings = Vec::new();

        let migration_path = self.get_migration_path(from_version, to_version);

        for (from, to) in migration_path {
            let step_result = self.apply_migration_step(&current_content, from, to)?;
            current_content = step_result.content;
            all_changes.extend(step_result.changes);
            all_warnings.extend(step_result.warnings);
        }

        Ok(MigrationResult {
            content: if self.options.dry_run {
                content.to_string()
            } else {
                current_content
            },
            from_version,
            to_version,
            changes: all_changes,
            warnings: all_warnings,
            backup: if self.options.create_backup {
                Some(content.to_string())
            } else {
                None
            },
        })
    }

    fn get_migration_path(
        &self,
        from: SammVersion,
        to: SammVersion,
    ) -> Vec<(SammVersion, SammVersion)> {
        let mut path = Vec::new();
        let current = from;

        // Define version progression (skipping V1_0_0 as it wasn't widely used)
        const VERSIONS: &[SammVersion] = &[
            SammVersion::Bamm,
            SammVersion::V2_0_0,
            SammVersion::V2_1_0,
            SammVersion::V2_3_0,
        ];

        let start_idx = VERSIONS.iter().position(|v| *v == current).unwrap_or(0);
        let end_idx = VERSIONS
            .iter()
            .position(|v| *v == to)
            .unwrap_or(VERSIONS.len() - 1);

        for i in start_idx..end_idx {
            path.push((VERSIONS[i], VERSIONS[i + 1]));
        }

        path
    }

    fn apply_migration_step(
        &self,
        content: &str,
        from: SammVersion,
        to: SammVersion,
    ) -> Result<MigrationResult> {
        let mut migrated = content.to_string();
        let mut changes = Vec::new();
        let warnings = Vec::new();

        // Get rules for this migration step
        if let Some(rules) = self.rules.get(&(from, to)) {
            for rule in rules {
                let before = migrated.clone();
                migrated = rule
                    .pattern
                    .replace_all(&migrated, &rule.replacement)
                    .to_string();

                if before != migrated {
                    changes.push(format!("{}: {}", rule.name, rule.description));
                }
            }
        }

        // Apply auto-fixes if enabled
        if self.options.auto_fix {
            let (fixed, auto_changes) = self.apply_auto_fixes(&migrated);
            migrated = fixed;
            changes.extend(auto_changes);
        }

        Ok(MigrationResult {
            content: migrated,
            from_version: from,
            to_version: to,
            changes,
            warnings,
            backup: None,
        })
    }

    fn apply_auto_fixes(&self, content: &str) -> (String, Vec<String>) {
        let mut fixed = content.to_string();
        let mut changes = Vec::new();

        // Fix common issues
        // 1. Ensure proper prefix declarations
        if !content.contains("@prefix samm:") {
            let prefix = "@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .\n";
            fixed = format!("{}{}", prefix, fixed);
            changes.push("Added missing samm prefix declaration".to_string());
        }

        // 2. Fix trailing whitespace
        let lines: Vec<_> = fixed.lines().map(|l| l.trim_end()).collect();
        fixed = lines.join("\n");

        (fixed, changes)
    }

    fn initialize_rules(&mut self) {
        // BAMM → SAMM 2.0.0
        self.add_rule(
            SammVersion::Bamm,
            SammVersion::V2_0_0,
            MigrationRule {
                name: "BAMM to SAMM namespace".to_string(),
                pattern: Regex::new(r"urn:bamm:").unwrap(),
                replacement: "urn:samm:".to_string(),
                description: "Replaced BAMM namespace with SAMM namespace".to_string(),
            },
        );

        self.add_rule(
            SammVersion::Bamm,
            SammVersion::V2_0_0,
            MigrationRule {
                name: "BAMM prefix to SAMM".to_string(),
                pattern: Regex::new(r"@prefix\s+bamm:").unwrap(),
                replacement: "@prefix samm:".to_string(),
                description: "Replaced bamm: prefix with samm: prefix".to_string(),
            },
        );

        self.add_rule(
            SammVersion::Bamm,
            SammVersion::V2_0_0,
            MigrationRule {
                name: "BAMM-C to SAMM-C".to_string(),
                pattern: Regex::new(r"@prefix\s+bamm-c:").unwrap(),
                replacement: "@prefix samm-c:".to_string(),
                description: "Replaced bamm-c: prefix with samm-c: prefix".to_string(),
            },
        );

        self.add_rule(
            SammVersion::Bamm,
            SammVersion::V2_0_0,
            MigrationRule {
                name: "BAMM-E to SAMM-E".to_string(),
                pattern: Regex::new(r"@prefix\s+bamm-e:").unwrap(),
                replacement: "@prefix samm-e:".to_string(),
                description: "Replaced bamm-e: prefix with samm-e: prefix".to_string(),
            },
        );

        // SAMM 2.0.0 → 2.1.0
        self.add_rule(
            SammVersion::V2_0_0,
            SammVersion::V2_1_0,
            MigrationRule {
                name: "Update meta-model version to 2.1.0".to_string(),
                pattern: Regex::new(r"meta-model:2\.0\.0").unwrap(),
                replacement: "meta-model:2.1.0".to_string(),
                description: "Updated meta-model version from 2.0.0 to 2.1.0".to_string(),
            },
        );

        self.add_rule(
            SammVersion::V2_0_0,
            SammVersion::V2_1_0,
            MigrationRule {
                name: "Update characteristic version to 2.1.0".to_string(),
                pattern: Regex::new(r"characteristic:2\.0\.0").unwrap(),
                replacement: "characteristic:2.1.0".to_string(),
                description: "Updated characteristic version from 2.0.0 to 2.1.0".to_string(),
            },
        );

        // SAMM 2.1.0 → 2.3.0
        self.add_rule(
            SammVersion::V2_1_0,
            SammVersion::V2_3_0,
            MigrationRule {
                name: "Update meta-model version to 2.3.0".to_string(),
                pattern: Regex::new(r"meta-model:2\.1\.0").unwrap(),
                replacement: "meta-model:2.3.0".to_string(),
                description: "Updated meta-model version from 2.1.0 to 2.3.0".to_string(),
            },
        );

        self.add_rule(
            SammVersion::V2_1_0,
            SammVersion::V2_3_0,
            MigrationRule {
                name: "Update characteristic version to 2.3.0".to_string(),
                pattern: Regex::new(r"characteristic:2\.1\.0").unwrap(),
                replacement: "characteristic:2.3.0".to_string(),
                description: "Updated characteristic version from 2.1.0 to 2.3.0".to_string(),
            },
        );

        self.add_rule(
            SammVersion::V2_1_0,
            SammVersion::V2_3_0,
            MigrationRule {
                name: "Update entity version to 2.3.0".to_string(),
                pattern: Regex::new(r"entity:2\.1\.0").unwrap(),
                replacement: "entity:2.3.0".to_string(),
                description: "Updated entity version from 2.1.0 to 2.3.0".to_string(),
            },
        );

        self.add_rule(
            SammVersion::V2_1_0,
            SammVersion::V2_3_0,
            MigrationRule {
                name: "Update unit version to 2.3.0".to_string(),
                pattern: Regex::new(r"unit:2\.1\.0").unwrap(),
                replacement: "unit:2.3.0".to_string(),
                description: "Updated unit version from 2.1.0 to 2.3.0".to_string(),
            },
        );
    }

    fn add_rule(&mut self, from: SammVersion, to: SammVersion, rule: MigrationRule) {
        self.rules.entry((from, to)).or_default().push(rule);
    }

    /// Get available migration paths from a given version
    pub fn get_available_migrations(&self, from: SammVersion) -> Vec<SammVersion> {
        let versions = vec![
            SammVersion::Bamm,
            SammVersion::V2_0_0,
            SammVersion::V2_1_0,
            SammVersion::V2_3_0,
        ];

        versions.into_iter().filter(|v| *v > from).collect()
    }

    /// Check if a migration is required
    pub fn needs_migration(&self, content: &str) -> bool {
        let current = self.detect_version(content);
        current != self.options.target_version && current < self.options.target_version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        assert_eq!(SammVersion::parse("urn:bamm:test"), SammVersion::Bamm);
        assert_eq!(SammVersion::parse("meta-model:2.3.0"), SammVersion::V2_3_0);
        assert_eq!(SammVersion::parse("meta-model:2.1.0"), SammVersion::V2_1_0);
        assert_eq!(SammVersion::parse("meta-model:2.0.0"), SammVersion::V2_0_0);
    }

    #[test]
    fn test_version_comparison() {
        assert!(SammVersion::Bamm.is_older_than(&SammVersion::V2_0_0));
        assert!(SammVersion::V2_0_0.is_older_than(&SammVersion::V2_3_0));
        assert!(!SammVersion::V2_3_0.is_older_than(&SammVersion::V2_0_0));
    }

    #[test]
    fn test_detect_bamm_version() {
        let content = r#"
            @prefix bamm: <urn:bamm:io.openmanufacturing:meta-model:1.0.0#> .
            @prefix : <urn:bamm:com.example:1.0.0#> .
        "#;

        let migrator = ModelMigrator::new(MigrationOptions::default());
        assert_eq!(migrator.detect_version(content), SammVersion::Bamm);
    }

    #[test]
    fn test_detect_samm_version() {
        let content = r#"
            @prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
            @prefix : <urn:samm:com.example:1.0.0#> .
        "#;

        let migrator = ModelMigrator::new(MigrationOptions::default());
        assert_eq!(migrator.detect_version(content), SammVersion::V2_3_0);
    }

    #[test]
    fn test_migrate_bamm_to_samm() {
        let content = r#"@prefix bamm: <urn:bamm:io.openmanufacturing:meta-model:1.0.0#> .
@prefix bamm-c: <urn:bamm:io.openmanufacturing:characteristic:1.0.0#> .
@prefix : <urn:bamm:com.example:1.0.0#> ."#;

        let options = MigrationOptions {
            target_version: SammVersion::V2_0_0,
            dry_run: false,
            ..Default::default()
        };

        let migrator = ModelMigrator::new(options);
        let result = migrator.migrate(content).unwrap();

        assert_eq!(result.from_version, SammVersion::Bamm);
        assert_eq!(result.to_version, SammVersion::V2_0_0);
        assert!(result.content.contains("urn:samm:"));
        assert!(!result.content.contains("urn:bamm:"));
        assert!(!result.changes.is_empty());
    }

    #[test]
    fn test_migrate_version_upgrade() {
        let content = r#"@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.0.0#> ."#;

        let options = MigrationOptions {
            target_version: SammVersion::V2_3_0,
            ..Default::default()
        };

        let migrator = ModelMigrator::new(options);
        let result = migrator.migrate(content).unwrap();

        assert_eq!(result.from_version, SammVersion::V2_0_0);
        assert_eq!(result.to_version, SammVersion::V2_3_0);
        assert!(result.content.contains("meta-model:2.3.0"));
    }

    #[test]
    fn test_no_migration_needed() {
        let content = r#"@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> ."#;

        let options = MigrationOptions {
            target_version: SammVersion::V2_3_0,
            ..Default::default()
        };

        let migrator = ModelMigrator::new(options);
        let result = migrator.migrate(content).unwrap();

        assert_eq!(result.from_version, SammVersion::V2_3_0);
        assert_eq!(result.to_version, SammVersion::V2_3_0);
        assert_eq!(result.changes.len(), 1);
        assert!(result.changes[0].contains("already at target version"));
    }

    #[test]
    fn test_dry_run_mode() {
        let content = r#"@prefix bamm: <urn:bamm:io.openmanufacturing:meta-model:1.0.0#> ."#;

        let options = MigrationOptions {
            target_version: SammVersion::V2_3_0,
            dry_run: true,
            ..Default::default()
        };

        let migrator = ModelMigrator::new(options);
        let result = migrator.migrate(content).unwrap();

        // Content should be unchanged in dry run mode
        assert_eq!(result.content, content);
        // But changes should be reported
        assert!(!result.changes.is_empty());
    }

    #[test]
    fn test_needs_migration() {
        let bamm_content = r#"@prefix bamm: <urn:bamm:io.openmanufacturing:meta-model:1.0.0#> ."#;
        let samm_content = r#"@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> ."#;

        let migrator = ModelMigrator::new(MigrationOptions::default());

        assert!(migrator.needs_migration(bamm_content));
        assert!(!migrator.needs_migration(samm_content));
    }

    #[test]
    fn test_get_available_migrations() {
        let migrator = ModelMigrator::new(MigrationOptions::default());
        let migrations = migrator.get_available_migrations(SammVersion::Bamm);

        assert!(migrations.contains(&SammVersion::V2_0_0));
        assert!(migrations.contains(&SammVersion::V2_3_0));
        assert!(!migrations.contains(&SammVersion::Bamm));
    }

    #[test]
    fn test_migration_path() {
        let migrator = ModelMigrator::new(MigrationOptions::default());
        let path = migrator.get_migration_path(SammVersion::Bamm, SammVersion::V2_3_0);

        // Should have multiple steps
        assert!(path.len() > 1);
        // First step should start from BAMM
        assert_eq!(path[0].0, SammVersion::Bamm);
        // Last step should end at 2.3.0
        assert_eq!(path[path.len() - 1].1, SammVersion::V2_3_0);
    }
}
