//! Shape Comparison and Diff Utilities
//!
//! Provides comprehensive comparison and diff capabilities for SHACL shapes:
//! - Shape-to-shape comparison
//! - Constraint diff generation
//! - Breaking change detection
//! - Compatibility analysis
//! - Migration path suggestions

use crate::constraints::Constraint;
use crate::{Shape, Target};
use oxirs_core::RdfTerm;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Type of difference detected between shapes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffType {
    /// Element was added
    Added,
    /// Element was removed
    Removed,
    /// Element was modified
    Modified,
    /// Element unchanged
    Unchanged,
}

/// Severity of a shape change
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ChangeSeverity {
    /// No impact
    None,
    /// Informational - minor change
    Info,
    /// Warning - potentially breaking
    Warning,
    /// Breaking - backward incompatible
    Breaking,
}

/// Represents a single difference item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffItem {
    /// Type of change
    pub diff_type: DiffType,
    /// Path to the changed element
    pub path: String,
    /// Old value (if applicable)
    pub old_value: Option<String>,
    /// New value (if applicable)
    pub new_value: Option<String>,
    /// Severity of the change
    pub severity: ChangeSeverity,
    /// Human-readable description
    pub description: String,
}

/// Result of comparing two shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeDiff {
    /// Shape ID being compared
    pub shape_id: String,
    /// Overall change severity
    pub overall_severity: ChangeSeverity,
    /// List of differences
    pub differences: Vec<DiffItem>,
    /// Summary statistics
    pub stats: DiffStats,
    /// Compatibility assessment
    pub compatibility: CompatibilityAssessment,
}

/// Statistics about differences
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiffStats {
    /// Number of additions
    pub additions: usize,
    /// Number of removals
    pub removals: usize,
    /// Number of modifications
    pub modifications: usize,
    /// Total constraints in old shape
    pub old_constraint_count: usize,
    /// Total constraints in new shape
    pub new_constraint_count: usize,
}

/// Compatibility assessment between shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityAssessment {
    /// Is the new shape backward compatible?
    pub is_backward_compatible: bool,
    /// Is the new shape forward compatible?
    pub is_forward_compatible: bool,
    /// Breaking changes detected
    pub breaking_changes: Vec<String>,
    /// Suggested migration steps
    pub migration_steps: Vec<String>,
}

/// Shape comparator for generating diffs
#[derive(Debug, Default)]
pub struct ShapeComparator {
    /// Configuration options
    config: ComparatorConfig,
}

/// Configuration for shape comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparatorConfig {
    /// Include unchanged items in diff
    pub include_unchanged: bool,
    /// Strict comparison mode
    pub strict_mode: bool,
    /// Compare constraint ordering
    pub compare_ordering: bool,
    /// Compare metadata (labels, descriptions)
    pub compare_metadata: bool,
}

impl Default for ComparatorConfig {
    fn default() -> Self {
        Self {
            include_unchanged: false,
            strict_mode: false,
            compare_ordering: false,
            compare_metadata: true,
        }
    }
}

impl ShapeComparator {
    /// Create a new shape comparator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a comparator with custom configuration
    pub fn with_config(config: ComparatorConfig) -> Self {
        Self { config }
    }

    /// Compare two shapes and generate a diff
    pub fn compare(&self, old_shape: &Shape, new_shape: &Shape) -> ShapeDiff {
        let mut differences = Vec::new();
        let mut stats = DiffStats::default();

        // Compare basic properties
        self.compare_basic_properties(old_shape, new_shape, &mut differences);

        // Compare targets
        self.compare_targets(old_shape, new_shape, &mut differences);

        // Compare constraints
        self.compare_constraints(old_shape, new_shape, &mut differences, &mut stats);

        // Compare metadata if enabled
        if self.config.compare_metadata {
            self.compare_metadata(old_shape, new_shape, &mut differences);
        }

        // Calculate statistics
        stats.additions = differences
            .iter()
            .filter(|d| d.diff_type == DiffType::Added)
            .count();
        stats.removals = differences
            .iter()
            .filter(|d| d.diff_type == DiffType::Removed)
            .count();
        stats.modifications = differences
            .iter()
            .filter(|d| d.diff_type == DiffType::Modified)
            .count();

        // Assess overall severity
        let overall_severity = differences
            .iter()
            .map(|d| d.severity)
            .max()
            .unwrap_or(ChangeSeverity::None);

        // Generate compatibility assessment
        let compatibility = self.assess_compatibility(&differences);

        ShapeDiff {
            shape_id: new_shape.id.as_str().to_string(),
            overall_severity,
            differences,
            stats,
            compatibility,
        }
    }

    /// Compare multiple shapes between two sets
    pub fn compare_shape_sets(&self, old_shapes: &[Shape], new_shapes: &[Shape]) -> Vec<ShapeDiff> {
        let mut diffs = Vec::new();

        // Build maps by shape ID
        let old_map: HashMap<&str, &Shape> =
            old_shapes.iter().map(|s| (s.id.as_str(), s)).collect();
        let new_map: HashMap<&str, &Shape> =
            new_shapes.iter().map(|s| (s.id.as_str(), s)).collect();

        // Find all shape IDs
        let all_ids: HashSet<&str> = old_map
            .keys()
            .copied()
            .chain(new_map.keys().copied())
            .collect();

        for id in all_ids {
            match (old_map.get(id), new_map.get(id)) {
                (Some(old), Some(new)) => {
                    // Shape exists in both - compare
                    diffs.push(self.compare(old, new));
                }
                (Some(old), None) => {
                    // Shape removed
                    diffs.push(self.create_removed_shape_diff(old));
                }
                (None, Some(new)) => {
                    // Shape added
                    diffs.push(self.create_added_shape_diff(new));
                }
                (None, None) => unreachable!(),
            }
        }

        diffs
    }

    /// Compare basic shape properties
    fn compare_basic_properties(
        &self,
        old_shape: &Shape,
        new_shape: &Shape,
        differences: &mut Vec<DiffItem>,
    ) {
        // Compare shape type
        if old_shape.shape_type != new_shape.shape_type {
            differences.push(DiffItem {
                diff_type: DiffType::Modified,
                path: "shapeType".to_string(),
                old_value: Some(format!("{:?}", old_shape.shape_type)),
                new_value: Some(format!("{:?}", new_shape.shape_type)),
                severity: ChangeSeverity::Breaking,
                description: "Shape type changed".to_string(),
            });
        }

        // Compare deactivated status
        if old_shape.deactivated != new_shape.deactivated {
            let severity = if new_shape.deactivated {
                ChangeSeverity::Warning
            } else {
                ChangeSeverity::Info
            };
            differences.push(DiffItem {
                diff_type: DiffType::Modified,
                path: "deactivated".to_string(),
                old_value: Some(old_shape.deactivated.to_string()),
                new_value: Some(new_shape.deactivated.to_string()),
                severity,
                description: format!(
                    "Shape {} deactivated",
                    if new_shape.deactivated {
                        "is now"
                    } else {
                        "is no longer"
                    }
                ),
            });
        }

        // Compare severity
        if old_shape.severity != new_shape.severity {
            differences.push(DiffItem {
                diff_type: DiffType::Modified,
                path: "severity".to_string(),
                old_value: Some(format!("{:?}", old_shape.severity)),
                new_value: Some(format!("{:?}", new_shape.severity)),
                severity: ChangeSeverity::Info,
                description: "Severity level changed".to_string(),
            });
        }
    }

    /// Compare shape targets
    fn compare_targets(
        &self,
        old_shape: &Shape,
        new_shape: &Shape,
        differences: &mut Vec<DiffItem>,
    ) {
        let old_targets: HashSet<String> = old_shape
            .targets
            .iter()
            .map(|t| self.target_to_string(t))
            .collect();
        let new_targets: HashSet<String> = new_shape
            .targets
            .iter()
            .map(|t| self.target_to_string(t))
            .collect();

        // Find added targets
        for target in new_targets.difference(&old_targets) {
            differences.push(DiffItem {
                diff_type: DiffType::Added,
                path: format!("targets[{}]", target),
                old_value: None,
                new_value: Some(target.clone()),
                severity: ChangeSeverity::Warning,
                description: format!("Target added: {}", target),
            });
        }

        // Find removed targets
        for target in old_targets.difference(&new_targets) {
            differences.push(DiffItem {
                diff_type: DiffType::Removed,
                path: format!("targets[{}]", target),
                old_value: Some(target.clone()),
                new_value: None,
                severity: ChangeSeverity::Breaking,
                description: format!("Target removed: {}", target),
            });
        }
    }

    /// Compare constraints
    fn compare_constraints(
        &self,
        old_shape: &Shape,
        new_shape: &Shape,
        differences: &mut Vec<DiffItem>,
        stats: &mut DiffStats,
    ) {
        stats.old_constraint_count = old_shape.constraints.len();
        stats.new_constraint_count = new_shape.constraints.len();

        // Build constraint maps by component ID
        let old_constraints: HashMap<&str, &Constraint> = old_shape
            .constraints
            .iter()
            .map(|(id, c)| (id.as_str(), c))
            .collect();
        let new_constraints: HashMap<&str, &Constraint> = new_shape
            .constraints
            .iter()
            .map(|(id, c)| (id.as_str(), c))
            .collect();

        let all_ids: HashSet<&str> = old_constraints
            .keys()
            .copied()
            .chain(new_constraints.keys().copied())
            .collect();

        for id in all_ids {
            match (old_constraints.get(id), new_constraints.get(id)) {
                (Some(old), Some(new)) => {
                    // Constraint exists in both - compare
                    if !self.constraints_equal(old, new) {
                        let severity = self.assess_constraint_change_severity(id, old, new);
                        differences.push(DiffItem {
                            diff_type: DiffType::Modified,
                            path: format!("constraints[{}]", id),
                            old_value: Some(format!("{:?}", old)),
                            new_value: Some(format!("{:?}", new)),
                            severity,
                            description: format!("Constraint {} modified", id),
                        });
                    } else if self.config.include_unchanged {
                        differences.push(DiffItem {
                            diff_type: DiffType::Unchanged,
                            path: format!("constraints[{}]", id),
                            old_value: Some(format!("{:?}", old)),
                            new_value: Some(format!("{:?}", new)),
                            severity: ChangeSeverity::None,
                            description: format!("Constraint {} unchanged", id),
                        });
                    }
                }
                (Some(old), None) => {
                    // Constraint removed
                    let severity = self.assess_constraint_removal_severity(id);
                    differences.push(DiffItem {
                        diff_type: DiffType::Removed,
                        path: format!("constraints[{}]", id),
                        old_value: Some(format!("{:?}", old)),
                        new_value: None,
                        severity,
                        description: format!("Constraint {} removed", id),
                    });
                }
                (None, Some(new)) => {
                    // Constraint added
                    let severity = self.assess_constraint_addition_severity(id);
                    differences.push(DiffItem {
                        diff_type: DiffType::Added,
                        path: format!("constraints[{}]", id),
                        old_value: None,
                        new_value: Some(format!("{:?}", new)),
                        severity,
                        description: format!("Constraint {} added", id),
                    });
                }
                (None, None) => unreachable!(),
            }
        }
    }

    /// Compare metadata (labels, descriptions)
    fn compare_metadata(
        &self,
        old_shape: &Shape,
        new_shape: &Shape,
        differences: &mut Vec<DiffItem>,
    ) {
        // Compare label
        if old_shape.label != new_shape.label {
            differences.push(DiffItem {
                diff_type: DiffType::Modified,
                path: "label".to_string(),
                old_value: old_shape.label.clone(),
                new_value: new_shape.label.clone(),
                severity: ChangeSeverity::None,
                description: "Label changed".to_string(),
            });
        }

        // Compare description
        if old_shape.description != new_shape.description {
            differences.push(DiffItem {
                diff_type: DiffType::Modified,
                path: "description".to_string(),
                old_value: old_shape.description.clone(),
                new_value: new_shape.description.clone(),
                severity: ChangeSeverity::None,
                description: "Description changed".to_string(),
            });
        }

        // Compare messages
        if old_shape.messages != new_shape.messages {
            differences.push(DiffItem {
                diff_type: DiffType::Modified,
                path: "messages".to_string(),
                old_value: Some(format!("{:?}", old_shape.messages)),
                new_value: Some(format!("{:?}", new_shape.messages)),
                severity: ChangeSeverity::Info,
                description: "Validation messages changed".to_string(),
            });
        }
    }

    /// Check if two constraints are equal
    fn constraints_equal(&self, c1: &Constraint, c2: &Constraint) -> bool {
        // Use Debug representation for comparison
        format!("{:?}", c1) == format!("{:?}", c2)
    }

    /// Convert target to string for comparison
    fn target_to_string(&self, target: &Target) -> String {
        match target {
            Target::Class(node) => format!("class:{}", node.as_str()),
            Target::Node(node) => format!("node:{}", node.as_str()),
            Target::SubjectsOf(node) => format!("subjectsOf:{}", node.as_str()),
            Target::ObjectsOf(node) => format!("objectsOf:{}", node.as_str()),
            Target::Sparql(query) => format!("sparql:{:?}", query),
            Target::Implicit(node) => format!("implicit:{}", node.as_str()),
            Target::Union(union) => format!("union:{}", union.targets.len()),
            Target::Intersection(inter) => format!("intersection:{}", inter.targets.len()),
            Target::Difference(_) => "difference".to_string(),
            Target::Conditional(_) => "conditional".to_string(),
            Target::Hierarchical(h) => format!("hierarchical:{:?}", h.relationship),
            Target::PathBased(p) => format!("pathBased:{:?}", p.direction),
        }
    }

    /// Assess severity of constraint change
    fn assess_constraint_change_severity(
        &self,
        id: &str,
        _old: &Constraint,
        _new: &Constraint,
    ) -> ChangeSeverity {
        // Stricter constraints are breaking changes
        // Looser constraints are warnings
        match id {
            "sh:minCount" | "sh:minLength" | "sh:minInclusive" | "sh:minExclusive" => {
                ChangeSeverity::Warning // Could be breaking if increased
            }
            "sh:maxCount" | "sh:maxLength" | "sh:maxInclusive" | "sh:maxExclusive" => {
                ChangeSeverity::Warning // Could be breaking if decreased
            }
            "sh:datatype" | "sh:class" | "sh:nodeKind" => ChangeSeverity::Breaking,
            "sh:pattern" => ChangeSeverity::Breaking,
            _ => ChangeSeverity::Warning,
        }
    }

    /// Assess severity of constraint removal
    fn assess_constraint_removal_severity(&self, id: &str) -> ChangeSeverity {
        match id {
            // Removing min constraints makes schema looser - generally safe
            "sh:minCount" | "sh:minLength" | "sh:minInclusive" | "sh:minExclusive" => {
                ChangeSeverity::Info
            }
            // Removing max constraints makes schema looser - generally safe
            "sh:maxCount" | "sh:maxLength" | "sh:maxInclusive" | "sh:maxExclusive" => {
                ChangeSeverity::Info
            }
            // Removing type constraints is a warning
            _ => ChangeSeverity::Warning,
        }
    }

    /// Assess severity of constraint addition
    fn assess_constraint_addition_severity(&self, id: &str) -> ChangeSeverity {
        match id {
            // Adding min constraints makes schema stricter - potentially breaking
            "sh:minCount" | "sh:minLength" | "sh:minInclusive" | "sh:minExclusive" => {
                ChangeSeverity::Breaking
            }
            // Adding max constraints makes schema stricter
            "sh:maxCount" | "sh:maxLength" | "sh:maxInclusive" | "sh:maxExclusive" => {
                ChangeSeverity::Breaking
            }
            // Adding type constraints is breaking
            "sh:datatype" | "sh:class" | "sh:nodeKind" | "sh:pattern" => ChangeSeverity::Breaking,
            _ => ChangeSeverity::Warning,
        }
    }

    /// Create diff for a removed shape
    fn create_removed_shape_diff(&self, shape: &Shape) -> ShapeDiff {
        let mut differences = vec![DiffItem {
            diff_type: DiffType::Removed,
            path: "shape".to_string(),
            old_value: Some(shape.id.as_str().to_string()),
            new_value: None,
            severity: ChangeSeverity::Breaking,
            description: format!("Shape {} removed", shape.id.as_str()),
        }];

        // Add all constraints as removed
        for (id, constraint) in &shape.constraints {
            differences.push(DiffItem {
                diff_type: DiffType::Removed,
                path: format!("constraints[{}]", id.as_str()),
                old_value: Some(format!("{:?}", constraint)),
                new_value: None,
                severity: ChangeSeverity::Breaking,
                description: format!("Constraint {} removed with shape", id.as_str()),
            });
        }

        ShapeDiff {
            shape_id: shape.id.as_str().to_string(),
            overall_severity: ChangeSeverity::Breaking,
            differences,
            stats: DiffStats {
                additions: 0,
                removals: shape.constraints.len() + 1,
                modifications: 0,
                old_constraint_count: shape.constraints.len(),
                new_constraint_count: 0,
            },
            compatibility: CompatibilityAssessment {
                is_backward_compatible: false,
                is_forward_compatible: true,
                breaking_changes: vec![format!("Shape {} removed", shape.id.as_str())],
                migration_steps: vec![format!(
                    "Remove all data that was validated by shape {}",
                    shape.id.as_str()
                )],
            },
        }
    }

    /// Create diff for an added shape
    fn create_added_shape_diff(&self, shape: &Shape) -> ShapeDiff {
        let mut differences = vec![DiffItem {
            diff_type: DiffType::Added,
            path: "shape".to_string(),
            old_value: None,
            new_value: Some(shape.id.as_str().to_string()),
            severity: ChangeSeverity::Warning,
            description: format!("New shape {} added", shape.id.as_str()),
        }];

        // Add all constraints as added
        for (id, constraint) in &shape.constraints {
            differences.push(DiffItem {
                diff_type: DiffType::Added,
                path: format!("constraints[{}]", id.as_str()),
                old_value: None,
                new_value: Some(format!("{:?}", constraint)),
                severity: ChangeSeverity::Warning,
                description: format!("Constraint {} added with shape", id.as_str()),
            });
        }

        ShapeDiff {
            shape_id: shape.id.as_str().to_string(),
            overall_severity: ChangeSeverity::Warning,
            differences,
            stats: DiffStats {
                additions: shape.constraints.len() + 1,
                removals: 0,
                modifications: 0,
                old_constraint_count: 0,
                new_constraint_count: shape.constraints.len(),
            },
            compatibility: CompatibilityAssessment {
                is_backward_compatible: true,
                is_forward_compatible: false,
                breaking_changes: vec![],
                migration_steps: vec![format!(
                    "Ensure data conforms to new shape {}",
                    shape.id.as_str()
                )],
            },
        }
    }

    /// Assess compatibility between shapes
    fn assess_compatibility(&self, differences: &[DiffItem]) -> CompatibilityAssessment {
        let breaking_changes: Vec<String> = differences
            .iter()
            .filter(|d| d.severity == ChangeSeverity::Breaking)
            .map(|d| d.description.clone())
            .collect();

        let is_backward_compatible = breaking_changes.is_empty();

        // Forward compatible if no constraints removed
        let is_forward_compatible = !differences
            .iter()
            .any(|d| d.diff_type == DiffType::Removed && d.severity >= ChangeSeverity::Warning);

        let migration_steps = self.generate_migration_steps(differences);

        CompatibilityAssessment {
            is_backward_compatible,
            is_forward_compatible,
            breaking_changes,
            migration_steps,
        }
    }

    /// Generate suggested migration steps
    fn generate_migration_steps(&self, differences: &[DiffItem]) -> Vec<String> {
        let mut steps = Vec::new();

        for diff in differences {
            match diff.diff_type {
                DiffType::Added if diff.severity >= ChangeSeverity::Warning => {
                    if diff.path.contains("minCount") {
                        steps.push(format!(
                            "Ensure all data has required values for {}",
                            diff.path
                        ));
                    } else if diff.path.contains("datatype") || diff.path.contains("class") {
                        steps.push(format!("Validate data types for {}", diff.path));
                    } else if diff.path.contains("pattern") {
                        steps.push(format!("Validate string patterns for {}", diff.path));
                    }
                }
                DiffType::Removed if diff.severity >= ChangeSeverity::Warning => {
                    steps.push(format!(
                        "Review removal of {} - may need data cleanup",
                        diff.path
                    ));
                }
                DiffType::Modified if diff.severity >= ChangeSeverity::Warning => {
                    steps.push(format!(
                        "Review modification of {} and update data accordingly",
                        diff.path
                    ));
                }
                _ => {}
            }
        }

        if steps.is_empty() && !differences.is_empty() {
            steps.push("No migration required - changes are backward compatible".to_string());
        }

        steps
    }
}

/// Generate a human-readable diff report
pub fn generate_diff_report(diff: &ShapeDiff) -> String {
    let mut report = String::new();

    report.push_str(&format!("# Shape Diff Report: {}\n\n", diff.shape_id));
    report.push_str(&format!(
        "Overall Severity: **{:?}**\n\n",
        diff.overall_severity
    ));

    // Statistics
    report.push_str("## Statistics\n\n");
    report.push_str(&format!("- Additions: {}\n", diff.stats.additions));
    report.push_str(&format!("- Removals: {}\n", diff.stats.removals));
    report.push_str(&format!("- Modifications: {}\n", diff.stats.modifications));
    report.push_str(&format!(
        "- Constraint count: {} → {}\n\n",
        diff.stats.old_constraint_count, diff.stats.new_constraint_count
    ));

    // Compatibility
    report.push_str("## Compatibility\n\n");
    report.push_str(&format!(
        "- Backward Compatible: {}\n",
        if diff.compatibility.is_backward_compatible {
            "✅ Yes"
        } else {
            "❌ No"
        }
    ));
    report.push_str(&format!(
        "- Forward Compatible: {}\n\n",
        if diff.compatibility.is_forward_compatible {
            "✅ Yes"
        } else {
            "❌ No"
        }
    ));

    // Breaking changes
    if !diff.compatibility.breaking_changes.is_empty() {
        report.push_str("### Breaking Changes\n\n");
        for change in &diff.compatibility.breaking_changes {
            report.push_str(&format!("- ⚠️ {}\n", change));
        }
        report.push('\n');
    }

    // Differences
    report.push_str("## Changes\n\n");
    for item in &diff.differences {
        let icon = match item.diff_type {
            DiffType::Added => "➕",
            DiffType::Removed => "➖",
            DiffType::Modified => "✏️",
            DiffType::Unchanged => "⏸️",
        };
        let severity_icon = match item.severity {
            ChangeSeverity::None => "",
            ChangeSeverity::Info => "ℹ️",
            ChangeSeverity::Warning => "⚠️",
            ChangeSeverity::Breaking => "🔴",
        };
        report.push_str(&format!(
            "- {} {} `{}`: {}\n",
            icon, severity_icon, item.path, item.description
        ));
    }
    report.push('\n');

    // Migration steps
    if !diff.compatibility.migration_steps.is_empty() {
        report.push_str("## Migration Steps\n\n");
        for (i, step) in diff.compatibility.migration_steps.iter().enumerate() {
            report.push_str(&format!("{}. {}\n", i + 1, step));
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::cardinality_constraints::MinCountConstraint;
    use crate::{ConstraintComponentId, ShapeId, ShapeType};

    fn create_test_shape(id: &str) -> Shape {
        Shape::new(ShapeId::new(id), ShapeType::NodeShape)
    }

    #[test]
    fn test_compare_identical_shapes() {
        let comparator = ShapeComparator::new();
        let shape1 = create_test_shape("http://example.org/Shape1");
        let shape2 = create_test_shape("http://example.org/Shape1");

        let diff = comparator.compare(&shape1, &shape2);

        assert_eq!(diff.overall_severity, ChangeSeverity::None);
        assert!(diff.compatibility.is_backward_compatible);
        assert!(diff.compatibility.is_forward_compatible);
    }

    #[test]
    fn test_compare_modified_shape_type() {
        let comparator = ShapeComparator::new();
        let shape1 = Shape::new(
            ShapeId::new("http://example.org/Shape1"),
            ShapeType::NodeShape,
        );
        let shape2 = Shape::new(
            ShapeId::new("http://example.org/Shape1"),
            ShapeType::PropertyShape,
        );

        let diff = comparator.compare(&shape1, &shape2);

        assert_eq!(diff.overall_severity, ChangeSeverity::Breaking);
        assert!(!diff.compatibility.is_backward_compatible);
    }

    #[test]
    fn test_compare_added_constraint() {
        let comparator = ShapeComparator::new();
        let shape1 = create_test_shape("http://example.org/Shape1");
        let mut shape2 = create_test_shape("http://example.org/Shape1");

        shape2.add_constraint(
            ConstraintComponentId::new("sh:minCount"),
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
        );

        let diff = comparator.compare(&shape1, &shape2);

        assert!(diff.stats.additions > 0);
        assert_eq!(diff.stats.new_constraint_count, 1);
    }

    #[test]
    fn test_compare_removed_constraint() {
        let comparator = ShapeComparator::new();
        let mut shape1 = create_test_shape("http://example.org/Shape1");
        let shape2 = create_test_shape("http://example.org/Shape1");

        shape1.add_constraint(
            ConstraintComponentId::new("sh:minCount"),
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
        );

        let diff = comparator.compare(&shape1, &shape2);

        assert!(diff.stats.removals > 0);
    }

    #[test]
    fn test_compare_shape_sets() {
        let comparator = ShapeComparator::new();

        let old_shapes = vec![
            create_test_shape("http://example.org/Shape1"),
            create_test_shape("http://example.org/Shape2"),
        ];
        let new_shapes = vec![
            create_test_shape("http://example.org/Shape1"),
            create_test_shape("http://example.org/Shape3"),
        ];

        let diffs = comparator.compare_shape_sets(&old_shapes, &new_shapes);

        // Should have 3 diffs: Shape1 unchanged, Shape2 removed, Shape3 added
        assert_eq!(diffs.len(), 3);
    }

    #[test]
    fn test_generate_diff_report() {
        let comparator = ShapeComparator::new();
        let shape1 = create_test_shape("http://example.org/Shape1");
        let mut shape2 = create_test_shape("http://example.org/Shape1");
        shape2.label = Some("New Label".to_string());

        let diff = comparator.compare(&shape1, &shape2);
        let report = generate_diff_report(&diff);

        assert!(report.contains("Shape Diff Report"));
        assert!(report.contains("Statistics"));
    }

    #[test]
    fn test_comparator_config() {
        let config = ComparatorConfig {
            include_unchanged: true,
            strict_mode: true,
            compare_ordering: true,
            compare_metadata: false,
        };
        let comparator = ShapeComparator::with_config(config);

        let mut shape1 = create_test_shape("http://example.org/Shape1");
        let mut shape2 = create_test_shape("http://example.org/Shape1");

        // Add same constraint to both
        shape1.add_constraint(
            ConstraintComponentId::new("sh:minCount"),
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
        );
        shape2.add_constraint(
            ConstraintComponentId::new("sh:minCount"),
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
        );

        let diff = comparator.compare(&shape1, &shape2);

        // Should include unchanged items
        assert!(diff
            .differences
            .iter()
            .any(|d| d.diff_type == DiffType::Unchanged));
    }

    #[test]
    fn test_migration_steps_generation() {
        let comparator = ShapeComparator::new();
        let shape1 = create_test_shape("http://example.org/Shape1");
        let mut shape2 = create_test_shape("http://example.org/Shape1");

        shape2.add_constraint(
            ConstraintComponentId::new("sh:minCount"),
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
        );

        let diff = comparator.compare(&shape1, &shape2);

        assert!(!diff.compatibility.migration_steps.is_empty());
    }
}
