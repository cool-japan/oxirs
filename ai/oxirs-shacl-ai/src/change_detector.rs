//! Schema change detection between two SHACL shape sets.
//!
//! [`ChangeDetector`] compares an "old" set of shapes with a "new" set and
//! classifies every difference into one of the following categories:
//!
//! * **Added shape** — a shape IRI present in `new` but absent in `old`.
//! * **Removed shape** — a shape IRI present in `old` but absent in `new`.
//! * **Modified shape** — same IRI, but at least one property or constraint
//!   has changed.
//! * **Property added** — a new property path appeared inside an existing shape.
//! * **Property removed** — a property path was dropped from an existing shape.
//! * **Constraint changed** — the value of `minCount`, `maxCount`, `datatype`,
//!   or `pattern` was modified.
//!
//! Each change is annotated with a [`ChangeSeverity`] (breaking / non-breaking /
//! informational) and all changes for a run are packaged as a [`ChangeReport`].
//! The detector also maintains a per-IRI [`ChangeHistory`] so successive runs
//! can be compared over time.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_shacl_ai::change_detector::{
//!     ChangeDetector, ShapeSnapshot, PropertyShape, ConstraintValues,
//! };
//!
//! let old = ShapeSnapshot {
//!     iri: "ex:PersonShape".into(),
//!     properties: vec![
//!         PropertyShape {
//!             path: "ex:name".into(),
//!             constraints: ConstraintValues {
//!                 min_count: Some(1), max_count: Some(1),
//!                 datatype: Some("xsd:string".into()), pattern: None,
//!             },
//!         },
//!     ],
//! };
//!
//! let new = ShapeSnapshot {
//!     iri: "ex:PersonShape".into(),
//!     properties: vec![
//!         PropertyShape {
//!             path: "ex:name".into(),
//!             constraints: ConstraintValues {
//!                 min_count: Some(1), max_count: Some(2),   // changed
//!                 datatype: Some("xsd:string".into()), pattern: None,
//!             },
//!         },
//!     ],
//! };
//!
//! let mut detector = ChangeDetector::new();
//! let report = detector.detect(&[old], &[new]);
//! assert!(!report.changes.is_empty());
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types — shape snapshots
// ─────────────────────────────────────────────────────────────────────────────

/// Constraint values for a property shape.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintValues {
    /// `sh:minCount` — minimum cardinality.
    pub min_count: Option<u32>,
    /// `sh:maxCount` — maximum cardinality.
    pub max_count: Option<u32>,
    /// `sh:datatype` — expected XSD datatype IRI.
    pub datatype: Option<String>,
    /// `sh:pattern` — regex pattern value.
    pub pattern: Option<String>,
}

impl ConstraintValues {
    /// Returns `true` when any field differs between `self` and `other`.
    pub fn differs_from(&self, other: &ConstraintValues) -> bool {
        self.min_count != other.min_count
            || self.max_count != other.max_count
            || self.datatype != other.datatype
            || self.pattern != other.pattern
    }
}

/// A single property shape within a node shape.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyShape {
    /// Property path IRI (e.g. `"ex:name"`).
    pub path: String,
    /// Constraint values for this property.
    pub constraints: ConstraintValues,
}

/// A snapshot of a single node shape at a point in time.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeSnapshot {
    /// Node shape IRI.
    pub iri: String,
    /// List of property shapes.
    pub properties: Vec<PropertyShape>,
}

impl ShapeSnapshot {
    /// Build a fast lookup map from property path to [`PropertyShape`].
    fn property_map(&self) -> HashMap<&str, &PropertyShape> {
        self.properties
            .iter()
            .map(|p| (p.path.as_str(), p))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types — change representation
// ─────────────────────────────────────────────────────────────────────────────

/// Severity classification of a detected schema change.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChangeSeverity {
    /// The change may break existing data or queries (e.g. tightening a
    /// constraint or removing a shape).
    Breaking,
    /// The change is backward-compatible (e.g. relaxing a constraint or adding
    /// an optional property).
    NonBreaking,
    /// The change is purely informational (e.g. a comment / metadata tweak
    /// with no constraint effect).
    Informational,
}

/// A single detected schema change.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeChange {
    /// Shape IRI that was affected.
    pub shape_iri: String,
    /// Property path if the change concerns a specific property (`None` for
    /// shape-level changes).
    pub property_path: Option<String>,
    /// Human-readable description of the change.
    pub description: String,
    /// Severity of the change.
    pub severity: ChangeSeverity,
    /// The kind of change.
    pub kind: ChangeKind,
}

/// Discriminant for the type of schema change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeKind {
    /// A shape was added to the schema.
    ShapeAdded,
    /// A shape was removed from the schema.
    ShapeRemoved,
    /// A property path was added to an existing shape.
    PropertyAdded,
    /// A property path was removed from an existing shape.
    PropertyRemoved,
    /// A constraint value changed (min/max count, datatype, pattern).
    ConstraintChanged,
}

/// Full change report produced by a single detection run.
#[derive(Debug, Clone)]
pub struct ChangeReport {
    /// All detected changes in this run.
    pub changes: Vec<ShapeChange>,
    /// Number of shapes in the old snapshot set.
    pub old_shape_count: usize,
    /// Number of shapes in the new snapshot set.
    pub new_shape_count: usize,
}

impl ChangeReport {
    /// Count of breaking changes.
    pub fn breaking_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| c.severity == ChangeSeverity::Breaking)
            .count()
    }

    /// Count of non-breaking changes.
    pub fn non_breaking_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| c.severity == ChangeSeverity::NonBreaking)
            .count()
    }

    /// Count of informational changes.
    pub fn informational_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| c.severity == ChangeSeverity::Informational)
            .count()
    }

    /// `true` when there are no detected changes.
    pub fn is_clean(&self) -> bool {
        self.changes.is_empty()
    }

    /// Filter changes by severity.
    pub fn by_severity(&self, severity: &ChangeSeverity) -> Vec<&ShapeChange> {
        self.changes
            .iter()
            .filter(|c| &c.severity == severity)
            .collect()
    }

    /// Filter changes by shape IRI.
    pub fn for_shape<'a>(&'a self, iri: &str) -> Vec<&'a ShapeChange> {
        self.changes.iter().filter(|c| c.shape_iri == iri).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Change history
// ─────────────────────────────────────────────────────────────────────────────

/// Recorded entry in a change history.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    /// Sequential run index (1-based).
    pub run: usize,
    /// Changes observed in that run.
    pub changes: Vec<ShapeChange>,
}

/// Per-shape change history accumulated across multiple [`ChangeDetector::detect`]
/// calls.
#[derive(Debug, Default)]
pub struct ChangeHistory {
    entries: Vec<HistoryEntry>,
}

impl ChangeHistory {
    /// All history entries.
    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    /// Total number of changes ever recorded.
    pub fn total_change_count(&self) -> usize {
        self.entries.iter().map(|e| e.changes.len()).sum()
    }

    /// Number of detection runs recorded.
    pub fn run_count(&self) -> usize {
        self.entries.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChangeDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Schema change detector that compares old and new shape snapshots.
#[derive(Debug, Default)]
pub struct ChangeDetector {
    history: ChangeHistory,
}

impl ChangeDetector {
    /// Create a new detector with an empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Detect changes between `old` and `new` shape sets.
    ///
    /// The returned [`ChangeReport`] is also stored in the internal history so
    /// successive calls accumulate over time.
    pub fn detect(&mut self, old: &[ShapeSnapshot], new: &[ShapeSnapshot]) -> ChangeReport {
        let mut changes = Vec::new();

        let old_map: HashMap<&str, &ShapeSnapshot> =
            old.iter().map(|s| (s.iri.as_str(), s)).collect();
        let new_map: HashMap<&str, &ShapeSnapshot> =
            new.iter().map(|s| (s.iri.as_str(), s)).collect();

        // Detect added shapes
        for iri in new_map.keys() {
            if !old_map.contains_key(iri) {
                changes.push(ShapeChange {
                    shape_iri: iri.to_string(),
                    property_path: None,
                    description: format!("Shape <{iri}> was added"),
                    severity: ChangeSeverity::NonBreaking,
                    kind: ChangeKind::ShapeAdded,
                });
            }
        }

        // Detect removed shapes
        for iri in old_map.keys() {
            if !new_map.contains_key(iri) {
                changes.push(ShapeChange {
                    shape_iri: iri.to_string(),
                    property_path: None,
                    description: format!("Shape <{iri}> was removed"),
                    severity: ChangeSeverity::Breaking,
                    kind: ChangeKind::ShapeRemoved,
                });
            }
        }

        // Detect property and constraint changes in shapes present in both
        for (iri, new_shape) in &new_map {
            if let Some(old_shape) = old_map.get(iri) {
                changes.extend(self.compare_shapes(old_shape, new_shape));
            }
        }

        let run = self.history.entries.len() + 1;
        let report = ChangeReport {
            changes: changes.clone(),
            old_shape_count: old.len(),
            new_shape_count: new.len(),
        };

        self.history.entries.push(HistoryEntry { run, changes });

        report
    }

    /// Access the accumulated change history.
    pub fn history(&self) -> &ChangeHistory {
        &self.history
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    fn compare_shapes(&self, old: &ShapeSnapshot, new: &ShapeSnapshot) -> Vec<ShapeChange> {
        let mut changes = Vec::new();
        let old_props = old.property_map();
        let new_props = new.property_map();

        // Added properties
        for path in new_props.keys() {
            if !old_props.contains_key(path) {
                changes.push(ShapeChange {
                    shape_iri: new.iri.clone(),
                    property_path: Some(path.to_string()),
                    description: format!("Property <{path}> added to shape <{}>", new.iri),
                    severity: ChangeSeverity::NonBreaking,
                    kind: ChangeKind::PropertyAdded,
                });
            }
        }

        // Removed properties
        for path in old_props.keys() {
            if !new_props.contains_key(path) {
                changes.push(ShapeChange {
                    shape_iri: new.iri.clone(),
                    property_path: Some(path.to_string()),
                    description: format!("Property <{path}> removed from shape <{}>", new.iri),
                    severity: ChangeSeverity::Breaking,
                    kind: ChangeKind::PropertyRemoved,
                });
            }
        }

        // Changed constraints
        for (path, new_prop) in &new_props {
            if let Some(old_prop) = old_props.get(path) {
                let c_old = &old_prop.constraints;
                let c_new = &new_prop.constraints;
                if c_old.differs_from(c_new) {
                    let severity = Self::constraint_severity(c_old, c_new);
                    let description = Self::constraint_description(&new.iri, path, c_old, c_new);
                    changes.push(ShapeChange {
                        shape_iri: new.iri.clone(),
                        property_path: Some(path.to_string()),
                        description,
                        severity,
                        kind: ChangeKind::ConstraintChanged,
                    });
                }
            }
        }

        changes
    }

    /// Classify the severity of a constraint change.
    ///
    /// Heuristic:
    /// - Increasing `minCount` → breaking (tighter requirement).
    /// - Decreasing `maxCount` → breaking.
    /// - Changing `datatype` → breaking.
    /// - Adding a `pattern` → breaking.
    /// - Relaxing `minCount` or increasing `maxCount` → non-breaking.
    /// - Removing a `pattern` → non-breaking.
    fn constraint_severity(old: &ConstraintValues, new: &ConstraintValues) -> ChangeSeverity {
        // Datatype change is always breaking
        if old.datatype != new.datatype {
            return ChangeSeverity::Breaking;
        }
        // Added pattern is breaking; removed pattern is non-breaking
        match (&old.pattern, &new.pattern) {
            (None, Some(_)) => return ChangeSeverity::Breaking,
            (Some(_), None) => return ChangeSeverity::NonBreaking,
            (Some(a), Some(b)) if a != b => return ChangeSeverity::Breaking,
            _ => {}
        }
        // minCount increase is breaking
        if let (Some(o), Some(n)) = (old.min_count, new.min_count) {
            if n > o {
                return ChangeSeverity::Breaking;
            }
        }
        if old.min_count.is_none() && new.min_count.is_some() {
            return ChangeSeverity::Breaking;
        }
        // maxCount decrease is breaking
        if let (Some(o), Some(n)) = (old.max_count, new.max_count) {
            if n < o {
                return ChangeSeverity::Breaking;
            }
        }
        if old.max_count.is_some() && new.max_count.is_none() {
            // Removing maxCount is relaxing
            return ChangeSeverity::NonBreaking;
        }
        ChangeSeverity::NonBreaking
    }

    fn constraint_description(
        shape_iri: &str,
        path: &str,
        old: &ConstraintValues,
        new: &ConstraintValues,
    ) -> String {
        let mut parts = Vec::new();
        if old.min_count != new.min_count {
            parts.push(format!("minCount {:?}→{:?}", old.min_count, new.min_count));
        }
        if old.max_count != new.max_count {
            parts.push(format!("maxCount {:?}→{:?}", old.max_count, new.max_count));
        }
        if old.datatype != new.datatype {
            parts.push(format!("datatype {:?}→{:?}", old.datatype, new.datatype));
        }
        if old.pattern != new.pattern {
            parts.push(format!("pattern {:?}→{:?}", old.pattern, new.pattern));
        }
        format!(
            "Constraint change on <{}> / <{}>: {}",
            shape_iri,
            path,
            parts.join(", ")
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn person_shape(min: Option<u32>, max: Option<u32>) -> ShapeSnapshot {
        ShapeSnapshot {
            iri: "ex:PersonShape".into(),
            properties: vec![PropertyShape {
                path: "ex:name".into(),
                constraints: ConstraintValues {
                    min_count: min,
                    max_count: max,
                    datatype: Some("xsd:string".into()),
                    pattern: None,
                },
            }],
        }
    }

    fn empty_shapes() -> Vec<ShapeSnapshot> {
        vec![]
    }

    // ── Added / removed shapes ───────────────────────────────────────────────

    #[test]
    fn test_detect_added_shape() {
        let mut det = ChangeDetector::new();
        let old: Vec<ShapeSnapshot> = vec![];
        let new = vec![person_shape(Some(1), None)];
        let report = det.detect(&old, &new);
        assert_eq!(report.changes.len(), 1);
        assert_eq!(report.changes[0].kind, ChangeKind::ShapeAdded);
        assert_eq!(report.changes[0].severity, ChangeSeverity::NonBreaking);
    }

    #[test]
    fn test_detect_removed_shape() {
        let mut det = ChangeDetector::new();
        let old = vec![person_shape(Some(1), None)];
        let new: Vec<ShapeSnapshot> = vec![];
        let report = det.detect(&old, &new);
        assert_eq!(report.changes.len(), 1);
        assert_eq!(report.changes[0].kind, ChangeKind::ShapeRemoved);
        assert_eq!(report.changes[0].severity, ChangeSeverity::Breaking);
    }

    #[test]
    fn test_no_change_identical_shapes() {
        let mut det = ChangeDetector::new();
        let shapes = vec![person_shape(Some(1), Some(1))];
        let report = det.detect(&shapes, &shapes.clone());
        assert!(report.is_clean());
    }

    #[test]
    fn test_multiple_shapes_one_added() {
        let mut det = ChangeDetector::new();
        let shape_a = person_shape(Some(1), None);
        let shape_b = ShapeSnapshot {
            iri: "ex:OrganizationShape".into(),
            properties: vec![],
        };
        let old = vec![shape_a.clone()];
        let new = vec![shape_a, shape_b];
        let report = det.detect(&old, &new);
        assert_eq!(
            report
                .changes
                .iter()
                .filter(|c| c.kind == ChangeKind::ShapeAdded)
                .count(),
            1
        );
    }

    // ── Property added / removed ─────────────────────────────────────────────

    #[test]
    fn test_detect_property_added() {
        let mut det = ChangeDetector::new();
        let old = vec![ShapeSnapshot {
            iri: "ex:PersonShape".into(),
            properties: vec![],
        }];
        let new = vec![ShapeSnapshot {
            iri: "ex:PersonShape".into(),
            properties: vec![PropertyShape {
                path: "ex:name".into(),
                constraints: ConstraintValues {
                    min_count: None,
                    max_count: None,
                    datatype: None,
                    pattern: None,
                },
            }],
        }];
        let report = det.detect(&old, &new);
        let prop_changes: Vec<_> = report
            .changes
            .iter()
            .filter(|c| c.kind == ChangeKind::PropertyAdded)
            .collect();
        assert_eq!(prop_changes.len(), 1);
        assert_eq!(prop_changes[0].property_path, Some("ex:name".into()));
        assert_eq!(prop_changes[0].severity, ChangeSeverity::NonBreaking);
    }

    #[test]
    fn test_detect_property_removed() {
        let mut det = ChangeDetector::new();
        let old = vec![ShapeSnapshot {
            iri: "ex:PersonShape".into(),
            properties: vec![PropertyShape {
                path: "ex:name".into(),
                constraints: ConstraintValues {
                    min_count: Some(1),
                    max_count: None,
                    datatype: None,
                    pattern: None,
                },
            }],
        }];
        let new = vec![ShapeSnapshot {
            iri: "ex:PersonShape".into(),
            properties: vec![],
        }];
        let report = det.detect(&old, &new);
        let prop_changes: Vec<_> = report
            .changes
            .iter()
            .filter(|c| c.kind == ChangeKind::PropertyRemoved)
            .collect();
        assert_eq!(prop_changes.len(), 1);
        assert_eq!(prop_changes[0].severity, ChangeSeverity::Breaking);
    }

    // ── Constraint changes ───────────────────────────────────────────────────

    #[test]
    fn test_constraint_min_count_increase_breaking() {
        let mut det = ChangeDetector::new();
        let old = vec![person_shape(Some(0), None)];
        let new = vec![person_shape(Some(1), None)];
        let report = det.detect(&old, &new);
        let cc: Vec<_> = report
            .changes
            .iter()
            .filter(|c| c.kind == ChangeKind::ConstraintChanged)
            .collect();
        assert!(!cc.is_empty());
        assert_eq!(cc[0].severity, ChangeSeverity::Breaking);
    }

    #[test]
    fn test_constraint_min_count_decrease_non_breaking() {
        let mut det = ChangeDetector::new();
        let old = vec![person_shape(Some(2), None)];
        let new = vec![person_shape(Some(1), None)];
        let report = det.detect(&old, &new);
        let cc: Vec<_> = report
            .changes
            .iter()
            .filter(|c| c.kind == ChangeKind::ConstraintChanged)
            .collect();
        assert!(!cc.is_empty());
        assert_eq!(cc[0].severity, ChangeSeverity::NonBreaking);
    }

    #[test]
    fn test_constraint_max_count_decrease_breaking() {
        let mut det = ChangeDetector::new();
        let old = vec![person_shape(None, Some(5))];
        let new = vec![person_shape(None, Some(2))];
        let report = det.detect(&old, &new);
        let cc: Vec<_> = report
            .changes
            .iter()
            .filter(|c| c.kind == ChangeKind::ConstraintChanged)
            .collect();
        assert!(!cc.is_empty());
        assert_eq!(cc[0].severity, ChangeSeverity::Breaking);
    }

    #[test]
    fn test_constraint_datatype_change_breaking() {
        let old = vec![ShapeSnapshot {
            iri: "ex:S".into(),
            properties: vec![PropertyShape {
                path: "ex:p".into(),
                constraints: ConstraintValues {
                    min_count: None,
                    max_count: None,
                    datatype: Some("xsd:string".into()),
                    pattern: None,
                },
            }],
        }];
        let new = vec![ShapeSnapshot {
            iri: "ex:S".into(),
            properties: vec![PropertyShape {
                path: "ex:p".into(),
                constraints: ConstraintValues {
                    min_count: None,
                    max_count: None,
                    datatype: Some("xsd:integer".into()),
                    pattern: None,
                },
            }],
        }];
        let mut det = ChangeDetector::new();
        let report = det.detect(&old, &new);
        let cc: Vec<_> = report
            .changes
            .iter()
            .filter(|c| c.kind == ChangeKind::ConstraintChanged)
            .collect();
        assert!(!cc.is_empty());
        assert_eq!(cc[0].severity, ChangeSeverity::Breaking);
    }

    #[test]
    fn test_constraint_pattern_added_breaking() {
        let old = vec![ShapeSnapshot {
            iri: "ex:S".into(),
            properties: vec![PropertyShape {
                path: "ex:p".into(),
                constraints: ConstraintValues {
                    min_count: None,
                    max_count: None,
                    datatype: None,
                    pattern: None,
                },
            }],
        }];
        let new = vec![ShapeSnapshot {
            iri: "ex:S".into(),
            properties: vec![PropertyShape {
                path: "ex:p".into(),
                constraints: ConstraintValues {
                    min_count: None,
                    max_count: None,
                    datatype: None,
                    pattern: Some("^[A-Z]".into()),
                },
            }],
        }];
        let mut det = ChangeDetector::new();
        let report = det.detect(&old, &new);
        let cc: Vec<_> = report
            .changes
            .iter()
            .filter(|c| c.kind == ChangeKind::ConstraintChanged)
            .collect();
        assert_eq!(cc[0].severity, ChangeSeverity::Breaking);
    }

    #[test]
    fn test_constraint_pattern_removed_non_breaking() {
        let old = vec![ShapeSnapshot {
            iri: "ex:S".into(),
            properties: vec![PropertyShape {
                path: "ex:p".into(),
                constraints: ConstraintValues {
                    min_count: None,
                    max_count: None,
                    datatype: None,
                    pattern: Some("^[A-Z]".into()),
                },
            }],
        }];
        let new = vec![ShapeSnapshot {
            iri: "ex:S".into(),
            properties: vec![PropertyShape {
                path: "ex:p".into(),
                constraints: ConstraintValues {
                    min_count: None,
                    max_count: None,
                    datatype: None,
                    pattern: None,
                },
            }],
        }];
        let mut det = ChangeDetector::new();
        let report = det.detect(&old, &new);
        let cc: Vec<_> = report
            .changes
            .iter()
            .filter(|c| c.kind == ChangeKind::ConstraintChanged)
            .collect();
        assert_eq!(cc[0].severity, ChangeSeverity::NonBreaking);
    }

    // ── ChangeReport helpers ─────────────────────────────────────────────────

    #[test]
    fn test_report_breaking_count() {
        let mut det = ChangeDetector::new();
        let old = vec![person_shape(Some(1), None)];
        let new: Vec<ShapeSnapshot> = vec![];
        let report = det.detect(&old, &new);
        assert!(report.breaking_count() > 0);
    }

    #[test]
    fn test_report_non_breaking_count() {
        let mut det = ChangeDetector::new();
        let old: Vec<ShapeSnapshot> = vec![];
        let new = vec![person_shape(Some(1), None)];
        let report = det.detect(&old, &new);
        assert_eq!(report.non_breaking_count(), 1);
        assert_eq!(report.breaking_count(), 0);
    }

    #[test]
    fn test_report_is_clean() {
        let mut det = ChangeDetector::new();
        let shapes = vec![person_shape(Some(1), Some(1))];
        let report = det.detect(&shapes, &shapes.clone());
        assert!(report.is_clean());
        assert_eq!(report.informational_count(), 0);
    }

    #[test]
    fn test_report_for_shape() {
        let mut det = ChangeDetector::new();
        let old: Vec<ShapeSnapshot> = vec![];
        let new = vec![
            person_shape(Some(1), None),
            ShapeSnapshot {
                iri: "ex:Other".into(),
                properties: vec![],
            },
        ];
        let report = det.detect(&old, &new);
        let person_changes = report.for_shape("ex:PersonShape");
        assert_eq!(person_changes.len(), 1);
    }

    #[test]
    fn test_report_by_severity() {
        let mut det = ChangeDetector::new();
        let old = vec![person_shape(Some(1), None)];
        let new: Vec<ShapeSnapshot> = vec![];
        let report = det.detect(&old, &new);
        let breaking = report.by_severity(&ChangeSeverity::Breaking);
        assert!(!breaking.is_empty());
    }

    #[test]
    fn test_report_shape_counts() {
        let mut det = ChangeDetector::new();
        let old = vec![person_shape(None, None)];
        let new = vec![
            person_shape(None, None),
            ShapeSnapshot {
                iri: "ex:New".into(),
                properties: vec![],
            },
        ];
        let report = det.detect(&old, &new);
        assert_eq!(report.old_shape_count, 1);
        assert_eq!(report.new_shape_count, 2);
    }

    // ── History tracking ─────────────────────────────────────────────────────

    #[test]
    fn test_history_accumulates_runs() {
        let mut det = ChangeDetector::new();
        let old: Vec<ShapeSnapshot> = vec![];
        let new = vec![person_shape(None, None)];
        det.detect(&old, &new);
        det.detect(&new, &new.clone());
        assert_eq!(det.history().run_count(), 2);
    }

    #[test]
    fn test_history_total_change_count() {
        let mut det = ChangeDetector::new();
        // Run 1: 1 added shape
        det.detect(&empty_shapes(), &[person_shape(None, None)]);
        // Run 2: clean
        det.detect(&[person_shape(None, None)], &[person_shape(None, None)]);
        assert_eq!(det.history().total_change_count(), 1);
    }

    #[test]
    fn test_history_entry_run_numbers() {
        let mut det = ChangeDetector::new();
        det.detect(&empty_shapes(), &[person_shape(None, None)]);
        det.detect(&empty_shapes(), &empty_shapes());
        let entries = det.history().entries();
        assert_eq!(entries[0].run, 1);
        assert_eq!(entries[1].run, 2);
    }

    // ── ConstraintValues helpers ─────────────────────────────────────────────

    #[test]
    fn test_constraint_values_differs_from_equal() {
        let cv = ConstraintValues {
            min_count: Some(1),
            max_count: None,
            datatype: None,
            pattern: None,
        };
        assert!(!cv.differs_from(&cv.clone()));
    }

    #[test]
    fn test_constraint_values_differs_from_unequal() {
        let a = ConstraintValues {
            min_count: Some(1),
            max_count: None,
            datatype: None,
            pattern: None,
        };
        let b = ConstraintValues {
            min_count: Some(2),
            max_count: None,
            datatype: None,
            pattern: None,
        };
        assert!(a.differs_from(&b));
    }

    #[test]
    fn test_shape_snapshot_property_map() {
        let shape = person_shape(Some(1), Some(1));
        let map = shape.property_map();
        assert!(map.contains_key("ex:name"));
    }

    #[test]
    fn test_change_kind_debug() {
        let kind = ChangeKind::ShapeAdded;
        assert!(format!("{kind:?}").contains("ShapeAdded"));
    }

    #[test]
    fn test_change_severity_ordering() {
        assert!(ChangeSeverity::Breaking < ChangeSeverity::NonBreaking);
        assert!(ChangeSeverity::NonBreaking < ChangeSeverity::Informational);
    }

    #[test]
    fn test_detect_empty_to_empty() {
        let mut det = ChangeDetector::new();
        let report = det.detect(&empty_shapes(), &empty_shapes());
        assert!(report.is_clean());
        assert_eq!(report.old_shape_count, 0);
        assert_eq!(report.new_shape_count, 0);
    }

    #[test]
    fn test_detect_max_count_removal_non_breaking() {
        let old = vec![person_shape(None, Some(3))];
        // new has no maxCount — relaxing
        let new = vec![ShapeSnapshot {
            iri: "ex:PersonShape".into(),
            properties: vec![PropertyShape {
                path: "ex:name".into(),
                constraints: ConstraintValues {
                    min_count: None,
                    max_count: None,
                    datatype: Some("xsd:string".into()),
                    pattern: None,
                },
            }],
        }];
        let mut det = ChangeDetector::new();
        let report = det.detect(&old, &new);
        let cc: Vec<_> = report
            .changes
            .iter()
            .filter(|c| c.kind == ChangeKind::ConstraintChanged)
            .collect();
        assert!(!cc.is_empty());
        assert_eq!(cc[0].severity, ChangeSeverity::NonBreaking);
    }

    #[test]
    fn test_constraint_description_contains_path() {
        let shape_iri = "ex:S";
        let path = "ex:p";
        let old = ConstraintValues {
            min_count: Some(1),
            max_count: None,
            datatype: None,
            pattern: None,
        };
        let new_cv = ConstraintValues {
            min_count: Some(2),
            max_count: None,
            datatype: None,
            pattern: None,
        };
        let desc = ChangeDetector::constraint_description(shape_iri, path, &old, &new_cv);
        assert!(desc.contains("ex:p"));
        assert!(desc.contains("minCount"));
    }

    #[test]
    fn test_multiple_constraint_changes_same_property() {
        let old = vec![ShapeSnapshot {
            iri: "ex:S".into(),
            properties: vec![PropertyShape {
                path: "ex:p".into(),
                constraints: ConstraintValues {
                    min_count: Some(1),
                    max_count: Some(3),
                    datatype: Some("xsd:string".into()),
                    pattern: None,
                },
            }],
        }];
        let new = vec![ShapeSnapshot {
            iri: "ex:S".into(),
            properties: vec![PropertyShape {
                path: "ex:p".into(),
                constraints: ConstraintValues {
                    min_count: Some(2), // increased — breaking
                    max_count: Some(3),
                    datatype: Some("xsd:integer".into()), // changed — breaking
                    pattern: None,
                },
            }],
        }];
        let mut det = ChangeDetector::new();
        let report = det.detect(&old, &new);
        // There should be one ConstraintChanged entry that covers both differences
        let cc: Vec<_> = report
            .changes
            .iter()
            .filter(|c| c.kind == ChangeKind::ConstraintChanged)
            .collect();
        assert_eq!(cc.len(), 1);
        assert!(cc[0].description.contains("minCount") || cc[0].description.contains("datatype"));
    }
}
