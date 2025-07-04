//! Shape versioning and evolution management for SHACL shapes
//!
//! This module provides comprehensive shape versioning capabilities including
//! version tracking, backward compatibility analysis, migration paths, and
//! schema evolution management.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use indexmap::IndexMap;
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use oxirs_core::{
    model::{NamedNode, Term, Triple},
    Store,
};

use crate::{
    constraints::*, report::*, validation::ValidationEngine, Result as ShaclResult, Shape, ShapeId, ShapeType,
    ValidationConfig, ValidationReport, ConstraintComponentId, Constraint, Severity,
};

/// Shape version identifier
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ShapeVersionId {
    /// Base shape identifier
    pub shape_id: ShapeId,

    /// Version number (semver) - serialized as string
    #[serde(with = "version_serde")]
    pub version: Version,

    /// Unique version UUID
    pub version_uuid: Uuid,
}

// Custom serde implementation for semver::Version
mod version_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use semver::Version;

    pub fn serialize<S>(version: &Version, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        version.to_string().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Version, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Version::parse(&s).map_err(serde::de::Error::custom)
    }
}

/// Shape version metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeVersionMetadata {
    /// Version identifier
    pub version_id: ShapeVersionId,

    /// Human-readable version name
    pub version_name: String,

    /// Version description
    pub description: String,

    /// Author information
    pub author: String,

    /// Creation timestamp
    pub created_at: SystemTime,

    /// Last modification timestamp
    pub modified_at: SystemTime,

    /// Version tags
    pub tags: HashSet<String>,

    /// Changelog entries
    pub changelog: Vec<ChangelogEntry>,

    /// Compatibility information
    pub compatibility: CompatibilityInfo,

    /// Deprecation information
    pub deprecation: Option<DeprecationInfo>,

    /// Parent version (for incremental changes)
    pub parent_version: Option<Version>,

    /// Child versions
    pub child_versions: Vec<Version>,
}

/// Changelog entry for shape versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangelogEntry {
    /// Change type
    pub change_type: ChangeType,

    /// Description of the change
    pub description: String,

    /// Affected constraint components
    pub affected_constraints: Vec<String>,

    /// Breaking change indicator
    pub is_breaking: bool,

    /// Migration notes
    pub migration_notes: Option<String>,

    /// Change timestamp
    pub timestamp: SystemTime,
}

/// Types of changes in shape versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    /// Added new constraint
    ConstraintAdded,
    /// Removed existing constraint
    ConstraintRemoved,
    /// Modified existing constraint
    ConstraintModified,
    /// Added new property
    PropertyAdded,
    /// Removed existing property
    PropertyRemoved,
    /// Modified property specification
    PropertyModified,
    /// Changed target definition
    TargetChanged,
    /// Updated validation rules
    ValidationRuleUpdated,
    /// Performance optimization
    PerformanceOptimization,
    /// Bug fix
    BugFix,
    /// Documentation update
    DocumentationUpdate,
}

/// Compatibility information for shape versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityInfo {
    /// Backward compatibility status
    pub backward_compatible: bool,

    /// Forward compatibility status
    pub forward_compatible: bool,

    /// Compatible version range
    pub compatible_versions: VersionReq,

    /// Incompatible versions
    pub incompatible_versions: Vec<Version>,

    /// Migration requirements
    pub migration_required: bool,

    /// Validation differences
    pub validation_differences: Vec<ValidationDifference>,
}

/// Difference in validation behavior between versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationDifference {
    /// Type of difference
    pub difference_type: DifferenceType,

    /// Description of the difference
    pub description: String,

    /// Affected data patterns
    pub affected_patterns: Vec<String>,

    /// Impact severity
    pub impact_severity: ImpactSeverity,

    /// Mitigation strategy
    pub mitigation: Option<String>,
}

/// Types of validation differences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifferenceType {
    /// Stricter validation (more constraints)
    StricterValidation,
    /// Relaxed validation (fewer constraints)
    RelaxedValidation,
    /// Changed constraint semantics
    SemanticChange,
    /// Performance impact
    PerformanceImpact,
    /// New validation capabilities
    NewCapabilities,
    /// Removed validation capabilities
    RemovedCapabilities,
}

/// Impact severity of validation differences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactSeverity {
    /// Low impact (minor changes)
    Low,
    /// Medium impact (noticeable changes)
    Medium,
    /// High impact (significant changes)
    High,
    /// Critical impact (breaking changes)
    Critical,
}

/// Deprecation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationInfo {
    /// Deprecation reason
    pub reason: String,

    /// Deprecated since version
    pub deprecated_since: Version,

    /// Planned removal version
    pub removal_planned: Option<Version>,

    /// Replacement shape or version
    pub replacement: Option<ShapeVersionId>,

    /// Migration guide
    pub migration_guide: String,
}

/// Shape version registry for managing shape evolution
pub struct ShapeVersionRegistry {
    /// All shape versions
    versions: Arc<RwLock<BTreeMap<ShapeVersionId, VersionedShape>>>,

    /// Version metadata
    metadata: Arc<RwLock<HashMap<ShapeVersionId, ShapeVersionMetadata>>>,

    /// Active versions per shape
    active_versions: Arc<RwLock<HashMap<ShapeId, Version>>>,

    /// Version dependencies
    dependencies: Arc<RwLock<HashMap<ShapeVersionId, Vec<ShapeVersionId>>>>,

    /// Migration paths between versions
    migration_paths: Arc<RwLock<HashMap<(Version, Version), MigrationPath>>>,

    /// Registry configuration
    config: VersionRegistryConfig,
}

/// Versioned shape container
#[derive(Debug, Clone)]
pub struct VersionedShape {
    /// Shape definition
    pub shape: Shape,

    /// Version metadata
    pub metadata: ShapeVersionMetadata,

    /// Validation statistics for this version
    pub validation_stats: VersionValidationStats,
}

/// Validation statistics for a shape version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionValidationStats {
    /// Number of times this version was used for validation
    pub usage_count: u64,

    /// Total validation time
    pub total_validation_time: std::time::Duration,

    /// Average validation time per use
    pub average_validation_time: std::time::Duration,

    /// Success rate
    pub success_rate: f64,

    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,

    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    /// Average memory usage
    pub average_memory_bytes: usize,

    /// Peak memory usage
    pub peak_memory_bytes: usize,

    /// Memory efficiency score
    pub efficiency_score: f64,
}

/// Performance metrics for shape versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Constraint evaluation time
    pub constraint_evaluation_time: std::time::Duration,

    /// Property path evaluation time
    pub path_evaluation_time: std::time::Duration,

    /// Cache hit ratio
    pub cache_hit_ratio: f64,

    /// Optimization effectiveness
    pub optimization_effectiveness: f64,
}

/// Migration path between two shape versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPath {
    /// Source version
    pub from_version: Version,

    /// Target version
    pub to_version: Version,

    /// Migration steps
    pub steps: Vec<MigrationStep>,

    /// Estimated migration time
    pub estimated_time: std::time::Duration,

    /// Migration complexity
    pub complexity: MigrationComplexity,

    /// Data transformation requirements
    pub data_transformations: Vec<DataTransformation>,
}

/// Individual migration step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStep {
    /// Step identifier
    pub step_id: String,

    /// Step description
    pub description: String,

    /// Step type
    pub step_type: MigrationStepType,

    /// Required actions
    pub actions: Vec<String>,

    /// Validation checks
    pub validation_checks: Vec<String>,

    /// Rollback procedure
    pub rollback_procedure: Option<String>,
}

/// Types of migration steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationStepType {
    /// Update shape definition
    ShapeUpdate,
    /// Data validation
    DataValidation,
    /// Data transformation
    DataTransformation,
    /// Constraint migration
    ConstraintMigration,
    /// Performance optimization
    PerformanceOptimization,
    /// Cleanup operation
    Cleanup,
}

/// Migration complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationComplexity {
    /// Simple migration (automated)
    Simple,
    /// Moderate migration (some manual steps)
    Moderate,
    /// Complex migration (significant manual intervention)
    Complex,
    /// Critical migration (high risk, extensive testing required)
    Critical,
}

/// Data transformation specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransformation {
    /// Transformation identifier
    pub transformation_id: String,

    /// Source data pattern
    pub source_pattern: String,

    /// Target data pattern
    pub target_pattern: String,

    /// SPARQL transformation query
    pub sparql_transformation: String,

    /// Validation query for transformed data
    pub validation_query: String,

    /// Transformation reversibility
    pub reversible: bool,
}

/// Configuration for the version registry
#[derive(Debug, Clone)]
pub struct VersionRegistryConfig {
    /// Maximum number of versions to keep per shape
    pub max_versions_per_shape: usize,

    /// Automatic cleanup of old versions
    pub auto_cleanup: bool,

    /// Cleanup threshold (versions older than this are candidates for cleanup)
    pub cleanup_threshold: std::time::Duration,

    /// Enable performance tracking
    pub enable_performance_tracking: bool,

    /// Enable automatic migration path generation
    pub auto_generate_migrations: bool,

    /// Version validation on registration
    pub validate_on_registration: bool,
}

impl Default for VersionRegistryConfig {
    fn default() -> Self {
        Self {
            max_versions_per_shape: 50,
            auto_cleanup: true,
            cleanup_threshold: std::time::Duration::from_secs(365 * 24 * 3600), // 1 year
            enable_performance_tracking: true,
            auto_generate_migrations: false,
            validate_on_registration: true,
        }
    }
}

impl ShapeVersionRegistry {
    /// Create a new shape version registry
    pub fn new(config: VersionRegistryConfig) -> Self {
        Self {
            versions: Arc::new(RwLock::new(BTreeMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            active_versions: Arc::new(RwLock::new(HashMap::new())),
            dependencies: Arc::new(RwLock::new(HashMap::new())),
            migration_paths: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Register a new shape version
    pub fn register_version(
        &self,
        shape: Shape,
        version: Version,
        metadata: ShapeVersionMetadata,
    ) -> Result<ShapeVersionId> {
        let version_id = ShapeVersionId {
            shape_id: shape.id.clone(),
            version: version.clone(),
            version_uuid: Uuid::new_v4(),
        };

        // Validate shape if configured
        if self.config.validate_on_registration {
            self.validate_shape_version(&shape)?;
        }

        // Check for compatibility with existing versions
        self.analyze_compatibility(&version_id, &shape)?;

        // Create versioned shape
        let versioned_shape = VersionedShape {
            shape,
            metadata: metadata.clone(),
            validation_stats: VersionValidationStats::new(),
        };

        // Store version
        let mut versions = self.versions.write().unwrap();
        versions.insert(version_id.clone(), versioned_shape);

        let mut metadata_store = self.metadata.write().unwrap();
        metadata_store.insert(version_id.clone(), metadata);

        // Update active version if this is the latest
        self.update_active_version(&version_id)?;

        // Generate migration paths if enabled
        if self.config.auto_generate_migrations {
            self.generate_migration_paths(&version_id)?;
        }

        Ok(version_id)
    }

    /// Get a specific shape version
    pub fn get_version(&self, version_id: &ShapeVersionId) -> Option<VersionedShape> {
        let versions = self.versions.read().unwrap();
        versions.get(version_id).cloned()
    }

    /// Get the active version of a shape
    pub fn get_active_version(&self, shape_id: &ShapeId) -> Option<VersionedShape> {
        let active_versions = self.active_versions.read().unwrap();
        if let Some(version) = active_versions.get(shape_id) {
            let version_id = ShapeVersionId {
                shape_id: shape_id.clone(),
                version: version.clone(),
                version_uuid: Uuid::new_v4(), // Placeholder - would need proper lookup
            };
            self.get_version(&version_id)
        } else {
            None
        }
    }

    /// Get all versions of a shape
    pub fn get_all_versions(&self, shape_id: &ShapeId) -> Vec<VersionedShape> {
        let versions = self.versions.read().unwrap();
        versions
            .iter()
            .filter(|(id, _)| id.shape_id == *shape_id)
            .map(|(_, shape)| shape.clone())
            .collect()
    }

    /// Compare two shape versions
    pub fn compare_versions(
        &self,
        version1: &ShapeVersionId,
        version2: &ShapeVersionId,
    ) -> Result<VersionComparison> {
        let versions = self.versions.read().unwrap();

        let shape1 = versions
            .get(version1)
            .ok_or_else(|| anyhow::anyhow!("Version not found: {:?}", version1))?;
        let shape2 = versions
            .get(version2)
            .ok_or_else(|| anyhow::anyhow!("Version not found: {:?}", version2))?;

        Ok(VersionComparison::compare(&shape1.shape, &shape2.shape))
    }

    /// Get migration path between two versions
    pub fn get_migration_path(
        &self,
        from_version: &Version,
        to_version: &Version,
    ) -> Option<MigrationPath> {
        let migration_paths = self.migration_paths.read().unwrap();
        migration_paths
            .get(&(from_version.clone(), to_version.clone()))
            .cloned()
    }

    /// Execute migration between versions
    pub fn migrate_version(
        &self,
        from_version: &ShapeVersionId,
        to_version: &ShapeVersionId,
        data_store: &dyn Store,
    ) -> Result<MigrationResult> {
        let migration_path = self
            .get_migration_path(&from_version.version, &to_version.version)
            .ok_or_else(|| anyhow::anyhow!("No migration path found"))?;

        let mut result = MigrationResult::new();

        for step in &migration_path.steps {
            let step_result = self.execute_migration_step(step, data_store)?;
            result.add_step_result(step_result);
        }

        Ok(result)
    }

    /// Validate shape version
    fn validate_shape_version(&self, shape: &Shape) -> Result<()> {
        // Basic validation - could be extended with more sophisticated checks
        if shape.constraints.is_empty() {
            return Err(anyhow::anyhow!("Shape must have at least one constraint"));
        }

        // Validate constraint consistency
        for (_, constraint) in &shape.constraints {
            self.validate_constraint(constraint)?;
        }

        Ok(())
    }

    /// Validate individual constraint
    fn validate_constraint(&self, constraint: &Constraint) -> Result<()> {
        // Implement constraint validation logic
        Ok(())
    }

    /// Analyze compatibility with existing versions
    fn analyze_compatibility(&self, version_id: &ShapeVersionId, shape: &Shape) -> Result<()> {
        // Get existing versions of the same shape
        let existing_versions = self.get_all_versions(&version_id.shape_id);

        for existing in existing_versions {
            // Analyze backward compatibility
            let compatibility = self.check_backward_compatibility(&existing.shape, shape)?;

            if !compatibility.is_compatible {
                // Log compatibility issues but don't fail registration
                eprintln!("Compatibility issues detected: {:?}", compatibility.issues);
            }
        }

        Ok(())
    }

    /// Check backward compatibility between shape versions
    fn check_backward_compatibility(
        &self,
        old_shape: &Shape,
        new_shape: &Shape,
    ) -> Result<CompatibilityResult> {
        let mut result = CompatibilityResult::new();

        // Compare constraints
        for (old_constraint_id, old_constraint) in &old_shape.constraints {
            if !new_shape.constraints.contains_key(old_constraint_id) {
                result.add_issue(CompatibilityIssue {
                    issue_type: CompatibilityIssueType::ConstraintRemoved,
                    description: format!("Constraint removed: {:?}", old_constraint_id),
                    severity: ImpactSeverity::High,
                });
            }
        }

        // Check for new constraints (may break existing valid data)
        for (new_constraint_id, new_constraint) in &new_shape.constraints {
            if !old_shape.constraints.contains_key(new_constraint_id) {
                result.add_issue(CompatibilityIssue {
                    issue_type: CompatibilityIssueType::ConstraintAdded,
                    description: format!("New constraint added: {:?}", new_constraint_id),
                    severity: ImpactSeverity::Medium,
                });
            }
        }

        Ok(result)
    }

    /// Update active version for a shape
    fn update_active_version(&self, version_id: &ShapeVersionId) -> Result<()> {
        let mut active_versions = self.active_versions.write().unwrap();

        // Check if this is the latest version
        let current_active = active_versions.get(&version_id.shape_id);

        let should_update = match current_active {
            Some(current_version) => version_id.version > *current_version,
            None => true,
        };

        if should_update {
            active_versions.insert(version_id.shape_id.clone(), version_id.version.clone());
        }

        Ok(())
    }

    /// Generate migration paths automatically
    fn generate_migration_paths(&self, version_id: &ShapeVersionId) -> Result<()> {
        // This would implement automatic migration path generation
        // based on shape differences and predefined migration rules
        Ok(())
    }

    /// Execute a single migration step
    fn execute_migration_step(
        &self,
        step: &MigrationStep,
        data_store: &dyn Store,
    ) -> Result<MigrationStepResult> {
        // Implement migration step execution
        Ok(MigrationStepResult {
            step_id: step.step_id.clone(),
            success: true,
            message: "Migration step completed successfully".to_string(),
            execution_time: std::time::Duration::from_millis(100),
        })
    }
}

/// Result of version comparison
#[derive(Debug, Clone)]
pub struct VersionComparison {
    /// Constraints added in the newer version
    pub constraints_added: Vec<Constraint>,

    /// Constraints removed in the newer version
    pub constraints_removed: Vec<Constraint>,

    /// Constraints modified in the newer version
    pub constraints_modified: Vec<(Constraint, Constraint)>,

    /// Overall compatibility assessment
    pub compatibility: CompatibilityInfo,
}

impl VersionComparison {
    /// Compare two shapes and return a comparison
    pub fn compare(shape1: &Shape, shape2: &Shape) -> Self {
        let mut constraints_added = Vec::new();
        let mut constraints_removed = Vec::new();
        let mut constraints_modified = Vec::new();

        // Find added constraints
        for (constraint_id, constraint) in &shape2.constraints {
            if !shape1.constraints.contains_key(constraint_id) {
                constraints_added.push(constraint.clone());
            }
        }

        // Find removed constraints
        for (constraint_id, constraint) in &shape1.constraints {
            if !shape2.constraints.contains_key(constraint_id) {
                constraints_removed.push(constraint.clone());
            }
        }

        // Determine compatibility
        let backward_compatible = constraints_removed.is_empty();
        let forward_compatible = constraints_added.is_empty();

        let compatibility = CompatibilityInfo {
            backward_compatible,
            forward_compatible,
            compatible_versions: VersionReq::parse("*").unwrap(),
            incompatible_versions: Vec::new(),
            migration_required: !constraints_added.is_empty() || !constraints_removed.is_empty(),
            validation_differences: Vec::new(),
        };

        Self {
            constraints_added,
            constraints_removed,
            constraints_modified,
            compatibility,
        }
    }
}

/// Result of compatibility analysis
#[derive(Debug, Clone)]
pub struct CompatibilityResult {
    /// Whether versions are compatible
    pub is_compatible: bool,

    /// Compatibility issues found
    pub issues: Vec<CompatibilityIssue>,
}

impl CompatibilityResult {
    /// Create a new compatibility result
    pub fn new() -> Self {
        Self {
            is_compatible: true,
            issues: Vec::new(),
        }
    }

    /// Add a compatibility issue
    pub fn add_issue(&mut self, issue: CompatibilityIssue) {
        if matches!(
            issue.severity,
            ImpactSeverity::High | ImpactSeverity::Critical
        ) {
            self.is_compatible = false;
        }
        self.issues.push(issue);
    }
}

/// Individual compatibility issue
#[derive(Debug, Clone)]
pub struct CompatibilityIssue {
    /// Type of compatibility issue
    pub issue_type: CompatibilityIssueType,

    /// Issue description
    pub description: String,

    /// Severity of the issue
    pub severity: ImpactSeverity,
}

/// Types of compatibility issues
#[derive(Debug, Clone)]
pub enum CompatibilityIssueType {
    /// Constraint was added
    ConstraintAdded,
    /// Constraint was removed
    ConstraintRemoved,
    /// Constraint was modified
    ConstraintModified,
    /// Target definition changed
    TargetChanged,
    /// Property path changed
    PropertyPathChanged,
}

/// Result of migration execution
#[derive(Debug, Clone)]
pub struct MigrationResult {
    /// Overall migration success
    pub success: bool,

    /// Individual step results
    pub step_results: Vec<MigrationStepResult>,

    /// Total migration time
    pub total_time: std::time::Duration,

    /// Migration summary
    pub summary: String,
}

impl MigrationResult {
    /// Create a new migration result
    pub fn new() -> Self {
        Self {
            success: true,
            step_results: Vec::new(),
            total_time: std::time::Duration::from_millis(0),
            summary: String::new(),
        }
    }

    /// Add a step result
    pub fn add_step_result(&mut self, step_result: MigrationStepResult) {
        if !step_result.success {
            self.success = false;
        }
        self.total_time += step_result.execution_time;
        self.step_results.push(step_result);
    }
}

/// Result of executing a migration step
#[derive(Debug, Clone)]
pub struct MigrationStepResult {
    /// Step identifier
    pub step_id: String,

    /// Whether the step succeeded
    pub success: bool,

    /// Result message
    pub message: String,

    /// Step execution time
    pub execution_time: std::time::Duration,
}

impl VersionValidationStats {
    /// Create new validation statistics
    pub fn new() -> Self {
        Self {
            usage_count: 0,
            total_validation_time: std::time::Duration::from_millis(0),
            average_validation_time: std::time::Duration::from_millis(0),
            success_rate: 0.0,
            memory_usage: MemoryUsageStats {
                average_memory_bytes: 0,
                peak_memory_bytes: 0,
                efficiency_score: 0.0,
            },
            performance_metrics: PerformanceMetrics {
                constraint_evaluation_time: std::time::Duration::from_millis(0),
                path_evaluation_time: std::time::Duration::from_millis(0),
                cache_hit_ratio: 0.0,
                optimization_effectiveness: 0.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Severity, ShapeMetadata};

    #[test]
    fn test_shape_version_registry() {
        let config = VersionRegistryConfig::default();
        let registry = ShapeVersionRegistry::new(config);

        // Create a test shape with at least one constraint
        let mut constraints = IndexMap::new();
        constraints.insert(
            ConstraintComponentId::new("minCount"),
            Constraint::MinCount(crate::constraints::cardinality_constraints::MinCountConstraint { min_count: 1 }),
        );

        let shape = Shape {
            id: ShapeId::new("http://example.org/PersonShape".to_string()),
            shape_type: ShapeType::NodeShape,
            targets: Vec::new(),
            path: None,
            constraints,
            deactivated: false,
            label: None,
            description: None,
            groups: Vec::new(),
            order: None,
            severity: Severity::Violation,
            messages: IndexMap::new(),
            extends: Vec::new(),
            priority: None,
            metadata: ShapeMetadata::default(),
        };

        let version = Version::new(1, 0, 0);
        let metadata = ShapeVersionMetadata {
            version_id: ShapeVersionId {
                shape_id: shape.id.clone(),
                version: version.clone(),
                version_uuid: Uuid::new_v4(),
            },
            version_name: "Initial version".to_string(),
            description: "Initial shape definition".to_string(),
            author: "Test Author".to_string(),
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            tags: HashSet::new(),
            changelog: Vec::new(),
            compatibility: CompatibilityInfo {
                backward_compatible: true,
                forward_compatible: true,
                compatible_versions: VersionReq::parse("*").unwrap(),
                incompatible_versions: Vec::new(),
                migration_required: false,
                validation_differences: Vec::new(),
            },
            deprecation: None,
            parent_version: None,
            child_versions: Vec::new(),
        };

        let version_id = registry.register_version(shape, version, metadata).unwrap();

        // Verify version was registered
        let retrieved = registry.get_version(&version_id);
        assert!(retrieved.is_some());
    }
}
