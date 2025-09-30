//! Shape Version Control System
//!
//! This module provides comprehensive version control capabilities for SHACL shapes,
//! including automated versioning, change tracking, impact analysis, and migration planning.

use crate::shape::Shape;
use crate::ShaclAiError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Version identifier for shapes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShapeVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ShapeVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub fn initial() -> Self {
        Self::new(1, 0, 0)
    }

    pub fn increment_major(&mut self) {
        self.major += 1;
        self.minor = 0;
        self.patch = 0;
    }

    pub fn increment_minor(&mut self) {
        self.minor += 1;
        self.patch = 0;
    }

    pub fn increment_patch(&mut self) {
        self.patch += 1;
    }

    pub fn is_compatible_with(&self, other: &ShapeVersion) -> bool {
        self.major == other.major
    }

    pub fn is_backward_compatible_with(&self, other: &ShapeVersion) -> bool {
        if self.major != other.major {
            return false;
        }
        if self.minor < other.minor {
            return false;
        }
        true
    }
}

impl std::fmt::Display for ShapeVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Types of changes that can occur to a shape
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    /// Adding new constraints (backward compatible)
    Addition,
    /// Removing constraints (potentially breaking)
    Removal,
    /// Modifying existing constraints (potentially breaking)
    Modification,
    /// Renaming elements (potentially breaking)
    Rename,
    /// Structural changes (potentially breaking)
    Structure,
}

impl ChangeType {
    pub fn is_breaking(&self) -> bool {
        matches!(
            self,
            ChangeType::Removal
                | ChangeType::Modification
                | ChangeType::Rename
                | ChangeType::Structure
        )
    }
}

/// A single change record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeChange {
    pub id: Uuid,
    pub change_type: ChangeType,
    pub path: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub author: String,
    pub description: String,
}

/// Version metadata and history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeVersionMetadata {
    pub version: ShapeVersion,
    pub timestamp: DateTime<Utc>,
    pub author: String,
    pub commit_message: String,
    pub changes: Vec<ShapeChange>,
    pub parent_version: Option<ShapeVersion>,
    pub tags: HashSet<String>,
    pub is_released: bool,
}

/// Compatibility assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityLevel {
    /// Fully backward compatible
    BackwardCompatible,
    /// Some deprecations but still compatible
    CompatibleWithWarnings,
    /// Breaking changes require major version bump
    BreakingChanges,
    /// Incompatible changes
    Incompatible,
}

/// Impact analysis for a set of changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    pub compatibility_level: CompatibilityLevel,
    pub affected_consumers: Vec<String>,
    pub breaking_changes: Vec<ShapeChange>,
    pub deprecations: Vec<String>,
    pub migration_complexity: MigrationComplexity,
    pub estimated_migration_time: chrono::Duration,
}

/// Migration complexity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationComplexity {
    Trivial,
    Simple,
    Moderate,
    Complex,
    Critical,
}

/// Migration strategy for handling version changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStrategy {
    pub from_version: ShapeVersion,
    pub to_version: ShapeVersion,
    pub steps: Vec<MigrationStep>,
    pub validation_rules: Vec<String>,
    pub rollback_plan: RollbackPlan,
}

/// Individual migration step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStep {
    pub step_number: u32,
    pub description: String,
    pub operation: MigrationOperation,
    pub validation: Option<String>,
    pub rollback_operation: Option<MigrationOperation>,
}

/// Migration operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationOperation {
    AddConstraint {
        path: String,
        constraint: String,
    },
    RemoveConstraint {
        path: String,
    },
    ModifyConstraint {
        path: String,
        old_constraint: String,
        new_constraint: String,
    },
    RenameProperty {
        old_name: String,
        new_name: String,
    },
    AddProperty {
        name: String,
        definition: String,
    },
    RemoveProperty {
        name: String,
    },
}

/// Rollback plan for version changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackPlan {
    pub rollback_steps: Vec<MigrationStep>,
    pub safety_checks: Vec<String>,
    pub data_backup_required: bool,
    pub estimated_rollback_time: chrono::Duration,
}

/// Main version control system
pub struct ShapeVersionControl {
    versions: HashMap<ShapeVersion, ShapeVersionMetadata>,
    current_version: Option<ShapeVersion>,
    shape_registry: HashMap<String, Shape>,
    change_history: Vec<ShapeChange>,
}

impl ShapeVersionControl {
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
            current_version: None,
            shape_registry: HashMap::new(),
            change_history: Vec::new(),
        }
    }

    /// Register a new shape version
    pub fn register_version(
        &mut self,
        version: ShapeVersion,
        shape: Shape,
        author: String,
        commit_message: String,
        changes: Vec<ShapeChange>,
    ) -> Result<(), ShaclAiError> {
        let metadata = ShapeVersionMetadata {
            version: version.clone(),
            timestamp: Utc::now(),
            author,
            commit_message,
            changes: changes.clone(),
            parent_version: self.current_version.clone(),
            tags: HashSet::new(),
            is_released: false,
        };

        self.versions.insert(version.clone(), metadata);
        self.shape_registry.insert(version.to_string(), shape);
        self.change_history.extend(changes);
        self.current_version = Some(version);

        Ok(())
    }

    /// Automatically determine the next version based on changes
    pub fn suggest_next_version(&self, changes: &[ShapeChange]) -> ShapeVersion {
        let initial_version = ShapeVersion::initial();
        let current = self.current_version.as_ref().unwrap_or(&initial_version);

        let has_breaking_changes = changes.iter().any(|c| c.change_type.is_breaking());
        let has_additions = changes
            .iter()
            .any(|c| matches!(c.change_type, ChangeType::Addition));

        let mut next_version = current.clone();

        if has_breaking_changes {
            next_version.increment_major();
        } else if has_additions {
            next_version.increment_minor();
        } else {
            next_version.increment_patch();
        }

        next_version
    }

    /// Analyze the impact of proposed changes
    pub fn analyze_impact(&self, changes: &[ShapeChange]) -> ImpactAnalysis {
        let breaking_changes: Vec<ShapeChange> = changes
            .iter()
            .filter(|c| c.change_type.is_breaking())
            .cloned()
            .collect();

        let compatibility_level = if breaking_changes.is_empty() {
            CompatibilityLevel::BackwardCompatible
        } else if breaking_changes.len() <= 2 {
            CompatibilityLevel::CompatibleWithWarnings
        } else if breaking_changes.len() <= 5 {
            CompatibilityLevel::BreakingChanges
        } else {
            CompatibilityLevel::Incompatible
        };

        let migration_complexity = match breaking_changes.len() {
            0 => MigrationComplexity::Trivial,
            1..=2 => MigrationComplexity::Simple,
            3..=5 => MigrationComplexity::Moderate,
            6..=10 => MigrationComplexity::Complex,
            _ => MigrationComplexity::Critical,
        };

        let estimated_migration_time = match migration_complexity {
            MigrationComplexity::Trivial => chrono::Duration::minutes(15),
            MigrationComplexity::Simple => chrono::Duration::hours(1),
            MigrationComplexity::Moderate => chrono::Duration::hours(4),
            MigrationComplexity::Complex => chrono::Duration::days(1),
            MigrationComplexity::Critical => chrono::Duration::days(3),
        };

        ImpactAnalysis {
            compatibility_level,
            affected_consumers: Vec::new(), // Would be populated from actual usage data
            breaking_changes,
            deprecations: Vec::new(), // Would be extracted from changes
            migration_complexity,
            estimated_migration_time,
        }
    }

    /// Generate a migration strategy between versions
    pub fn generate_migration_strategy(
        &self,
        from_version: &ShapeVersion,
        to_version: &ShapeVersion,
    ) -> Result<MigrationStrategy, ShaclAiError> {
        let changes = self.get_changes_between_versions(from_version, to_version)?;
        let mut steps = Vec::new();

        for (index, change) in changes.iter().enumerate() {
            let operation = match &change.change_type {
                ChangeType::Addition => {
                    if let Some(new_value) = &change.new_value {
                        MigrationOperation::AddConstraint {
                            path: change.path.clone(),
                            constraint: new_value.clone(),
                        }
                    } else {
                        continue;
                    }
                }
                ChangeType::Removal => MigrationOperation::RemoveConstraint {
                    path: change.path.clone(),
                },
                ChangeType::Modification => {
                    if let (Some(old_value), Some(new_value)) =
                        (&change.old_value, &change.new_value)
                    {
                        MigrationOperation::ModifyConstraint {
                            path: change.path.clone(),
                            old_constraint: old_value.clone(),
                            new_constraint: new_value.clone(),
                        }
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };

            let rollback_operation = self.generate_rollback_operation(&operation);

            steps.push(MigrationStep {
                step_number: index as u32 + 1,
                description: change.description.clone(),
                operation,
                validation: Some(format!("Validate constraint at path: {}", change.path)),
                rollback_operation,
            });
        }

        let rollback_plan = RollbackPlan {
            rollback_steps: steps
                .iter()
                .rev()
                .map(|step| MigrationStep {
                    step_number: step.step_number,
                    description: format!("Rollback: {}", step.description),
                    operation: step.rollback_operation.clone().unwrap_or_else(|| {
                        MigrationOperation::RemoveConstraint {
                            path: "unknown".to_string(),
                        }
                    }),
                    validation: step.validation.clone(),
                    rollback_operation: Some(step.operation.clone()),
                })
                .collect(),
            safety_checks: vec!["Validate data integrity".to_string()],
            data_backup_required: steps.len() > 3,
            estimated_rollback_time: chrono::Duration::minutes((steps.len() * 5) as i64),
        };

        Ok(MigrationStrategy {
            from_version: from_version.clone(),
            to_version: to_version.clone(),
            steps,
            validation_rules: vec![
                "Check constraint syntax".to_string(),
                "Validate against sample data".to_string(),
                "Performance impact assessment".to_string(),
            ],
            rollback_plan,
        })
    }

    /// Get all changes between two versions
    fn get_changes_between_versions(
        &self,
        from_version: &ShapeVersion,
        to_version: &ShapeVersion,
    ) -> Result<Vec<ShapeChange>, ShaclAiError> {
        // In a real implementation, this would traverse the version history
        // For now, return all changes in the history
        Ok(self.change_history.clone())
    }

    /// Generate a rollback operation for a given migration operation
    fn generate_rollback_operation(
        &self,
        operation: &MigrationOperation,
    ) -> Option<MigrationOperation> {
        match operation {
            MigrationOperation::AddConstraint { path, .. } => {
                Some(MigrationOperation::RemoveConstraint { path: path.clone() })
            }
            MigrationOperation::RemoveConstraint { path } => {
                // In a real implementation, we'd look up the original constraint
                Some(MigrationOperation::AddConstraint {
                    path: path.clone(),
                    constraint: "original_constraint".to_string(),
                })
            }
            MigrationOperation::ModifyConstraint {
                path,
                old_constraint,
                new_constraint,
            } => Some(MigrationOperation::ModifyConstraint {
                path: path.clone(),
                old_constraint: new_constraint.clone(),
                new_constraint: old_constraint.clone(),
            }),
            MigrationOperation::RenameProperty { old_name, new_name } => {
                Some(MigrationOperation::RenameProperty {
                    old_name: new_name.clone(),
                    new_name: old_name.clone(),
                })
            }
            MigrationOperation::AddProperty { name, .. } => {
                Some(MigrationOperation::RemoveProperty { name: name.clone() })
            }
            MigrationOperation::RemoveProperty { name } => Some(MigrationOperation::AddProperty {
                name: name.clone(),
                definition: "original_definition".to_string(),
            }),
        }
    }

    /// Tag a version
    pub fn tag_version(&mut self, version: &ShapeVersion, tag: String) -> Result<(), ShaclAiError> {
        if let Some(metadata) = self.versions.get_mut(version) {
            metadata.tags.insert(tag);
            Ok(())
        } else {
            Err(ShaclAiError::VersionNotFound(version.to_string()))
        }
    }

    /// Mark a version as released
    pub fn release_version(&mut self, version: &ShapeVersion) -> Result<(), ShaclAiError> {
        if let Some(metadata) = self.versions.get_mut(version) {
            metadata.is_released = true;
            Ok(())
        } else {
            Err(ShaclAiError::VersionNotFound(version.to_string()))
        }
    }

    /// Get version metadata
    pub fn get_version_metadata(&self, version: &ShapeVersion) -> Option<&ShapeVersionMetadata> {
        self.versions.get(version)
    }

    /// Get current version
    pub fn current_version(&self) -> Option<&ShapeVersion> {
        self.current_version.as_ref()
    }

    /// List all versions
    pub fn list_versions(&self) -> Vec<&ShapeVersion> {
        self.versions.keys().collect()
    }

    /// Get version history
    pub fn get_version_history(&self) -> Vec<&ShapeVersionMetadata> {
        let mut history: Vec<&ShapeVersionMetadata> = self.versions.values().collect();
        history.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        history
    }

    /// Check compatibility between versions
    pub fn check_compatibility(&self, v1: &ShapeVersion, v2: &ShapeVersion) -> CompatibilityLevel {
        if v1.is_compatible_with(v2) {
            CompatibilityLevel::BackwardCompatible
        } else if v1.major < v2.major {
            CompatibilityLevel::BreakingChanges
        } else {
            CompatibilityLevel::Incompatible
        }
    }
}

impl Default for ShapeVersionControl {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_creation() {
        let version = ShapeVersion::new(1, 2, 3);
        assert_eq!(version.to_string(), "1.2.3");
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = ShapeVersion::new(1, 0, 0);
        let v2 = ShapeVersion::new(1, 1, 0);
        let v3 = ShapeVersion::new(2, 0, 0);

        assert!(v1.is_compatible_with(&v2));
        assert!(v2.is_backward_compatible_with(&v1));
        assert!(!v1.is_compatible_with(&v3));
    }

    #[test]
    fn test_version_increment() {
        let mut version = ShapeVersion::new(1, 2, 3);

        version.increment_patch();
        assert_eq!(version, ShapeVersion::new(1, 2, 4));

        version.increment_minor();
        assert_eq!(version, ShapeVersion::new(1, 3, 0));

        version.increment_major();
        assert_eq!(version, ShapeVersion::new(2, 0, 0));
    }

    #[test]
    fn test_change_type_breaking() {
        assert!(!ChangeType::Addition.is_breaking());
        assert!(ChangeType::Removal.is_breaking());
        assert!(ChangeType::Modification.is_breaking());
    }

    #[test]
    fn test_version_control_creation() {
        let vc = ShapeVersionControl::new();
        assert!(vc.current_version().is_none());
        assert!(vc.list_versions().is_empty());
    }

    #[test]
    fn test_next_version_suggestion() {
        let mut vc = ShapeVersionControl::new();
        vc.current_version = Some(ShapeVersion::new(1, 0, 0));

        let breaking_change = vec![ShapeChange {
            id: Uuid::new_v4(),
            change_type: ChangeType::Removal,
            path: "test".to_string(),
            old_value: Some("old".to_string()),
            new_value: None,
            timestamp: Utc::now(),
            author: "test".to_string(),
            description: "test change".to_string(),
        }];

        let next_version = vc.suggest_next_version(&breaking_change);
        assert_eq!(next_version, ShapeVersion::new(2, 0, 0));
    }
}
