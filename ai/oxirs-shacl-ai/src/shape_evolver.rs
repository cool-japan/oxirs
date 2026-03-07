//! Incremental SHACL shape evolution as RDF data changes.
//!
//! Tracks successive versions of a SHACL node-shape definition.  Each call to
//! [`ShapeEvolver::evolve`] computes a diff from the current version, records
//! the changes, and produces a new immutable version entry.  Rollback creates a
//! new version equal to a historical snapshot.

use std::collections::HashMap;
use std::fmt;

// ── Property constraint ────────────────────────────────────────────────────────

/// Constraints on a single property within a SHACL shape.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyConstraint {
    /// Minimum occurrence count (0 = optional).
    pub min_count: u32,
    /// Maximum occurrence count (`None` = unbounded).
    pub max_count: Option<u32>,
    /// Expected XSD datatype IRI, e.g. `"xsd:string"`.
    pub datatype: Option<String>,
    /// Expected node kind: `"IRI"`, `"Literal"`, or `"BlankNode"`.
    pub node_kind: Option<String>,
}

// ── Shape change events ───────────────────────────────────────────────────────

/// Describes a single semantic change between two shape versions.
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeChange {
    /// A new property was added (with its minimum occurrence count).
    AddedProperty { property: String, min_count: u32 },
    /// A property was removed.
    RemovedProperty { property: String },
    /// The minimum occurrence count for a property was increased.
    TightenedConstraint {
        property: String,
        old_min: u32,
        new_min: u32,
    },
    /// The maximum occurrence count was relaxed (increased or made unbounded).
    RelaxedConstraint {
        property: String,
        old_max: Option<u32>,
        new_max: Option<u32>,
    },
    /// The expected datatype changed.
    TypeChanged {
        property: String,
        old_type: String,
        new_type: String,
    },
    /// A target class was added to the shape.
    AddedClass { class: String },
    /// The target class was removed from the shape.
    RemovedClass { class: String },
}

// ── Shape version ─────────────────────────────────────────────────────────────

/// A single immutable version of a SHACL shape.
#[derive(Debug, Clone)]
pub struct ShapeVersion {
    /// Monotonically increasing version number (starting at 1).
    pub version: u32,
    /// Creation timestamp in milliseconds since the Unix epoch (0 in tests).
    pub timestamp_ms: u64,
    /// Current property constraints, keyed by property IRI.
    pub properties: HashMap<String, PropertyConstraint>,
    /// Optional target class IRI.
    pub target_class: Option<String>,
    /// Changes from the previous version (empty for the initial version).
    pub changes_from_prev: Vec<ShapeChange>,
}

// ── Error ──────────────────────────────────────────────────────────────────────

/// Errors produced by the shape evolver.
#[derive(Debug)]
pub enum EvolverError {
    /// No base shape has been initialised yet.
    NoBaseShape,
    /// The requested version number was not found.
    VersionNotFound(u32),
    /// The proposed evolution is logically invalid.
    InvalidEvolution(String),
    /// The shape definition is empty (no properties and no target class).
    EmptyShape,
}

impl fmt::Display for EvolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvolverError::NoBaseShape => write!(f, "no base shape initialised"),
            EvolverError::VersionNotFound(v) => write!(f, "version {v} not found"),
            EvolverError::InvalidEvolution(msg) => write!(f, "invalid evolution: {msg}"),
            EvolverError::EmptyShape => write!(f, "shape is empty"),
        }
    }
}

impl std::error::Error for EvolverError {}

// ── Evolver ───────────────────────────────────────────────────────────────────

/// Maintains the versioned history of an incrementally evolving SHACL shape.
pub struct ShapeEvolver {
    versions: Vec<ShapeVersion>,
    current_version: u32,
}

impl ShapeEvolver {
    /// Create a new evolver with no versions.
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
            current_version: 0,
        }
    }

    /// Initialise the evolver with a base shape (version 1).
    ///
    /// Can be called only once; subsequent calls on an initialised evolver are
    /// silently ignored (call `evolve` instead).
    pub fn init(
        &mut self,
        properties: HashMap<String, PropertyConstraint>,
        target_class: Option<String>,
    ) {
        if !self.versions.is_empty() {
            return; // already initialised
        }
        self.current_version = 1;
        self.versions.push(ShapeVersion {
            version: 1,
            timestamp_ms: 0,
            properties,
            target_class,
            changes_from_prev: Vec::new(),
        });
    }

    /// Compute the diff from the current version to `new_properties` / `new_target_class`,
    /// record the new version, and return the new version number.
    pub fn evolve(
        &mut self,
        new_properties: HashMap<String, PropertyConstraint>,
        new_target_class: Option<String>,
    ) -> Result<u32, EvolverError> {
        if self.versions.is_empty() {
            return Err(EvolverError::NoBaseShape);
        }

        let current = self.versions.last().expect("checked non-empty above");
        let mut changes = Self::diff(current, &new_properties);

        // Target class changes.
        match (&current.target_class, &new_target_class) {
            (None, Some(cls)) => changes.push(ShapeChange::AddedClass { class: cls.clone() }),
            (Some(cls), None) => changes.push(ShapeChange::RemovedClass { class: cls.clone() }),
            _ => {}
        }

        self.current_version += 1;
        let new_version = self.current_version;
        self.versions.push(ShapeVersion {
            version: new_version,
            timestamp_ms: 0,
            properties: new_properties,
            target_class: new_target_class,
            changes_from_prev: changes,
        });

        Ok(new_version)
    }

    /// Return a reference to the latest version.
    pub fn current(&self) -> Result<&ShapeVersion, EvolverError> {
        self.versions.last().ok_or(EvolverError::NoBaseShape)
    }

    /// Return a reference to a specific version.
    pub fn get_version(&self, v: u32) -> Result<&ShapeVersion, EvolverError> {
        self.versions
            .iter()
            .find(|sv| sv.version == v)
            .ok_or(EvolverError::VersionNotFound(v))
    }

    /// Immutable view of the full version history.
    pub fn history(&self) -> &[ShapeVersion] {
        &self.versions
    }

    /// Number of versions recorded (including the initial version).
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }

    /// Compute the list of changes when transitioning from `old` to `new_props`.
    pub fn diff(
        old: &ShapeVersion,
        new_props: &HashMap<String, PropertyConstraint>,
    ) -> Vec<ShapeChange> {
        let mut changes = Vec::new();

        // Properties removed.
        for prop in old.properties.keys() {
            if !new_props.contains_key(prop) {
                changes.push(ShapeChange::RemovedProperty {
                    property: prop.clone(),
                });
            }
        }

        // Properties added.
        for (prop, constraint) in new_props {
            if !old.properties.contains_key(prop) {
                changes.push(ShapeChange::AddedProperty {
                    property: prop.clone(),
                    min_count: constraint.min_count,
                });
            }
        }

        // Properties with modified constraints.
        for (prop, old_c) in &old.properties {
            if let Some(new_c) = new_props.get(prop) {
                // min_count increased → tightened.
                if new_c.min_count > old_c.min_count {
                    changes.push(ShapeChange::TightenedConstraint {
                        property: prop.clone(),
                        old_min: old_c.min_count,
                        new_min: new_c.min_count,
                    });
                }

                // max_count relaxed (increased or removed).
                let max_relaxed = match (old_c.max_count, new_c.max_count) {
                    (Some(o), Some(n)) if n > o => true,
                    (Some(_), None) => true,
                    _ => false,
                };
                if max_relaxed {
                    changes.push(ShapeChange::RelaxedConstraint {
                        property: prop.clone(),
                        old_max: old_c.max_count,
                        new_max: new_c.max_count,
                    });
                }

                // Datatype changed.
                if old_c.datatype != new_c.datatype {
                    match (&old_c.datatype, &new_c.datatype) {
                        (Some(old_t), Some(new_t)) => {
                            changes.push(ShapeChange::TypeChanged {
                                property: prop.clone(),
                                old_type: old_t.clone(),
                                new_type: new_t.clone(),
                            });
                        }
                        _ => {
                            // Datatype added or removed; treated as a type change.
                            let old_type = old_c
                                .datatype
                                .clone()
                                .unwrap_or_else(|| "<none>".to_string());
                            let new_type = new_c
                                .datatype
                                .clone()
                                .unwrap_or_else(|| "<none>".to_string());
                            changes.push(ShapeChange::TypeChanged {
                                property: prop.clone(),
                                old_type,
                                new_type,
                            });
                        }
                    }
                }
            }
        }

        changes
    }

    /// Create a new version that mirrors `to_version` (rollback).
    ///
    /// Returns the new version number.
    pub fn rollback(&mut self, to_version: u32) -> Result<u32, EvolverError> {
        if self.versions.is_empty() {
            return Err(EvolverError::NoBaseShape);
        }

        let target = self
            .versions
            .iter()
            .find(|sv| sv.version == to_version)
            .ok_or(EvolverError::VersionNotFound(to_version))?;

        let rolled_props = target.properties.clone();
        let rolled_class = target.target_class.clone();

        // evolve() will compute the diff from the current version.
        self.evolve(rolled_props, rolled_class)
    }

    /// Returns `true` when adding / modifying properties in `new_props` relative
    /// to the `current` version is considered backward-compatible.
    ///
    /// Backward-compatible changes:
    /// - Adding a property whose `min_count == 0` (optional).
    /// - Decreasing `min_count` of an existing property.
    /// - Relaxing `max_count` (increasing or removing the bound).
    ///
    /// Backward-*incompatible* changes:
    /// - Adding a required property (`min_count > 0`).
    /// - Increasing `min_count` of an existing property.
    /// - Removing an existing property.
    /// - Tightening `max_count`.
    pub fn is_backward_compatible(
        current: &ShapeVersion,
        new_props: &HashMap<String, PropertyConstraint>,
    ) -> bool {
        // Removed properties — check that every existing property is still present.
        for prop in current.properties.keys() {
            if !new_props.contains_key(prop) {
                return false;
            }
        }

        // Added or modified properties.
        for (prop, new_c) in new_props {
            match current.properties.get(prop) {
                None => {
                    // New property — compatible only if optional.
                    if new_c.min_count > 0 {
                        return false;
                    }
                }
                Some(old_c) => {
                    // Existing property — compatible only if min_count did not increase.
                    if new_c.min_count > old_c.min_count {
                        return false;
                    }
                    // max_count tightened → incompatible.
                    let max_tightened = match (old_c.max_count, new_c.max_count) {
                        (None, Some(_)) => true,
                        (Some(o), Some(n)) if n < o => true,
                        _ => false,
                    };
                    if max_tightened {
                        return false;
                    }
                }
            }
        }

        true
    }
}

impl Default for ShapeEvolver {
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

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn prop(min: u32, max: Option<u32>) -> PropertyConstraint {
        PropertyConstraint {
            min_count: min,
            max_count: max,
            datatype: None,
            node_kind: None,
        }
    }

    fn typed_prop(min: u32, datatype: &str) -> PropertyConstraint {
        PropertyConstraint {
            min_count: min,
            max_count: None,
            datatype: Some(datatype.to_string()),
            node_kind: None,
        }
    }

    fn single(name: &str, min: u32) -> HashMap<String, PropertyConstraint> {
        let mut m = HashMap::new();
        m.insert(name.to_string(), prop(min, None));
        m
    }

    // ── init ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_init_creates_version_1() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:name", 1), Some("ex:Person".into()));
        assert_eq!(ev.version_count(), 1);
        let v = ev.current().unwrap();
        assert_eq!(v.version, 1);
    }

    #[test]
    fn test_init_sets_properties() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:name", 1), None);
        let v = ev.current().unwrap();
        assert!(v.properties.contains_key("ex:name"));
    }

    #[test]
    fn test_init_no_changes_from_prev() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:age", 0), None);
        let v = ev.current().unwrap();
        assert!(v.changes_from_prev.is_empty());
    }

    #[test]
    fn test_init_sets_target_class() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), Some("ex:Entity".into()));
        assert_eq!(
            ev.current().unwrap().target_class.as_deref(),
            Some("ex:Entity")
        );
    }

    #[test]
    fn test_init_second_call_ignored() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:a", 1), None);
        ev.init(single("ex:b", 1), None); // should be ignored
        assert_eq!(ev.version_count(), 1);
        let v = ev.current().unwrap();
        assert!(v.properties.contains_key("ex:a"));
        assert!(!v.properties.contains_key("ex:b"));
    }

    // ── evolve ────────────────────────────────────────────────────────────────

    #[test]
    fn test_evolve_increments_version() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:name", 1), None);
        let v = ev.evolve(single("ex:name", 1), None).unwrap();
        assert_eq!(v, 2);
        assert_eq!(ev.version_count(), 2);
    }

    #[test]
    fn test_evolve_adds_new_property() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:name", 1), None);
        let mut new_props = single("ex:name", 1);
        new_props.insert("ex:age".to_string(), prop(0, None));
        ev.evolve(new_props, None).unwrap();
        let v2 = ev.current().unwrap();
        let added: Vec<_> = v2
            .changes_from_prev
            .iter()
            .filter(|c| matches!(c, ShapeChange::AddedProperty { .. }))
            .collect();
        assert_eq!(added.len(), 1);
        if let ShapeChange::AddedProperty {
            property,
            min_count,
        } = &added[0]
        {
            assert_eq!(property, "ex:age");
            assert_eq!(*min_count, 0);
        }
    }

    #[test]
    fn test_evolve_removes_property() {
        let mut ev = ShapeEvolver::new();
        let mut init_props = single("ex:name", 1);
        init_props.insert("ex:age".to_string(), prop(0, None));
        ev.init(init_props, None);
        ev.evolve(single("ex:name", 1), None).unwrap();
        let v2 = ev.current().unwrap();
        let removed: Vec<_> = v2
            .changes_from_prev
            .iter()
            .filter(|c| matches!(c, ShapeChange::RemovedProperty { .. }))
            .collect();
        assert_eq!(removed.len(), 1);
    }

    #[test]
    fn test_evolve_without_base_returns_error() {
        let mut ev = ShapeEvolver::new();
        let result = ev.evolve(single("ex:p", 0), None);
        assert!(matches!(result, Err(EvolverError::NoBaseShape)));
    }

    #[test]
    fn test_evolve_multiple_versions() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        ev.evolve(single("ex:p", 1), None).unwrap();
        ev.evolve(single("ex:p", 1), None).unwrap();
        assert_eq!(ev.version_count(), 3);
    }

    // ── diff ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_diff_detects_tighten() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:name", 0), None);
        ev.evolve(single("ex:name", 1), None).unwrap();
        let v2 = ev.current().unwrap();
        let tightened: Vec<_> = v2
            .changes_from_prev
            .iter()
            .filter(|c| matches!(c, ShapeChange::TightenedConstraint { .. }))
            .collect();
        assert_eq!(tightened.len(), 1);
        if let ShapeChange::TightenedConstraint {
            property,
            old_min,
            new_min,
        } = &tightened[0]
        {
            assert_eq!(property, "ex:name");
            assert_eq!(*old_min, 0);
            assert_eq!(*new_min, 1);
        }
    }

    #[test]
    fn test_diff_detects_relax_max_count() {
        let mut ev = ShapeEvolver::new();
        let mut init_props = HashMap::new();
        init_props.insert("ex:p".to_string(), prop(0, Some(1)));
        ev.init(init_props, None);
        let mut new_props = HashMap::new();
        new_props.insert("ex:p".to_string(), prop(0, None));
        ev.evolve(new_props, None).unwrap();
        let v2 = ev.current().unwrap();
        assert!(v2
            .changes_from_prev
            .iter()
            .any(|c| matches!(c, ShapeChange::RelaxedConstraint { .. })));
    }

    #[test]
    fn test_diff_detects_type_change() {
        let mut ev = ShapeEvolver::new();
        let mut init_props = HashMap::new();
        init_props.insert("ex:p".to_string(), typed_prop(0, "xsd:string"));
        ev.init(init_props, None);
        let mut new_props = HashMap::new();
        new_props.insert("ex:p".to_string(), typed_prop(0, "xsd:integer"));
        ev.evolve(new_props, None).unwrap();
        let v2 = ev.current().unwrap();
        let type_changes: Vec<_> = v2
            .changes_from_prev
            .iter()
            .filter(|c| matches!(c, ShapeChange::TypeChanged { .. }))
            .collect();
        assert_eq!(type_changes.len(), 1);
        if let ShapeChange::TypeChanged {
            old_type, new_type, ..
        } = &type_changes[0]
        {
            assert_eq!(old_type, "xsd:string");
            assert_eq!(new_type, "xsd:integer");
        }
    }

    #[test]
    fn test_diff_class_added() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        ev.evolve(single("ex:p", 0), Some("ex:Foo".into())).unwrap();
        let v2 = ev.current().unwrap();
        assert!(v2
            .changes_from_prev
            .iter()
            .any(|c| matches!(c, ShapeChange::AddedClass { .. })));
    }

    #[test]
    fn test_diff_class_removed() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), Some("ex:Bar".into()));
        ev.evolve(single("ex:p", 0), None).unwrap();
        let v2 = ev.current().unwrap();
        assert!(v2
            .changes_from_prev
            .iter()
            .any(|c| matches!(c, ShapeChange::RemovedClass { .. })));
    }

    #[test]
    fn test_diff_no_changes_when_identical() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 1), Some("ex:C".into()));
        ev.evolve(single("ex:p", 1), Some("ex:C".into())).unwrap();
        let v2 = ev.current().unwrap();
        assert!(v2.changes_from_prev.is_empty());
    }

    // ── get_version ───────────────────────────────────────────────────────────

    #[test]
    fn test_get_version_returns_correct_version() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        ev.evolve(single("ex:p", 1), None).unwrap();
        let v1 = ev.get_version(1).unwrap();
        assert_eq!(v1.version, 1);
        assert_eq!(v1.properties["ex:p"].min_count, 0);
    }

    #[test]
    fn test_get_version_not_found_error() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        assert!(matches!(
            ev.get_version(99),
            Err(EvolverError::VersionNotFound(99))
        ));
    }

    // ── history ───────────────────────────────────────────────────────────────

    #[test]
    fn test_history_tracks_all_versions() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        ev.evolve(single("ex:p", 1), None).unwrap();
        ev.evolve(single("ex:p", 2), None).unwrap();
        assert_eq!(ev.history().len(), 3);
    }

    #[test]
    fn test_history_versions_are_sequential() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        for _ in 0..4 {
            ev.evolve(single("ex:p", 0), None).unwrap();
        }
        for (i, sv) in ev.history().iter().enumerate() {
            assert_eq!(sv.version, (i + 1) as u32);
        }
    }

    // ── rollback ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rollback_creates_new_version() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        ev.evolve(single("ex:p", 1), None).unwrap();
        let v = ev.rollback(1).unwrap();
        assert_eq!(v, 3);
        assert_eq!(ev.version_count(), 3);
    }

    #[test]
    fn test_rollback_restores_properties() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        ev.evolve(single("ex:p", 1), None).unwrap();
        ev.rollback(1).unwrap();
        let current = ev.current().unwrap();
        assert_eq!(current.properties["ex:p"].min_count, 0);
    }

    #[test]
    fn test_rollback_version_not_found() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        assert!(matches!(
            ev.rollback(99),
            Err(EvolverError::VersionNotFound(99))
        ));
    }

    #[test]
    fn test_rollback_no_base_shape_error() {
        let mut ev = ShapeEvolver::new();
        assert!(matches!(ev.rollback(1), Err(EvolverError::NoBaseShape)));
    }

    // ── is_backward_compatible ────────────────────────────────────────────────

    #[test]
    fn test_backward_compatible_add_optional_property() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 1), None);
        let current = ev.current().unwrap();
        let mut new_props = single("ex:p", 1);
        new_props.insert("ex:optional".to_string(), prop(0, None)); // min=0 → optional
        assert!(ShapeEvolver::is_backward_compatible(current, &new_props));
    }

    #[test]
    fn test_not_backward_compatible_add_required_property() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 1), None);
        let current = ev.current().unwrap();
        let mut new_props = single("ex:p", 1);
        new_props.insert("ex:required".to_string(), prop(1, None)); // min=1 → required
        assert!(!ShapeEvolver::is_backward_compatible(current, &new_props));
    }

    #[test]
    fn test_not_backward_compatible_increase_min_count() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        let current = ev.current().unwrap();
        assert!(!ShapeEvolver::is_backward_compatible(
            current,
            &single("ex:p", 1)
        ));
    }

    #[test]
    fn test_backward_compatible_decrease_min_count() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 2), None);
        let current = ev.current().unwrap();
        assert!(ShapeEvolver::is_backward_compatible(
            current,
            &single("ex:p", 1)
        ));
    }

    #[test]
    fn test_not_backward_compatible_remove_property() {
        let mut ev = ShapeEvolver::new();
        let mut init_props = single("ex:a", 1);
        init_props.insert("ex:b".to_string(), prop(0, None));
        ev.init(init_props, None);
        let current = ev.current().unwrap();
        assert!(!ShapeEvolver::is_backward_compatible(
            current,
            &single("ex:a", 1)
        ));
    }

    #[test]
    fn test_backward_compatible_relax_max_count() {
        let mut ev = ShapeEvolver::new();
        let mut init_props = HashMap::new();
        init_props.insert("ex:p".to_string(), prop(0, Some(3)));
        ev.init(init_props, None);
        let current = ev.current().unwrap();
        let mut new_props = HashMap::new();
        new_props.insert("ex:p".to_string(), prop(0, None));
        assert!(ShapeEvolver::is_backward_compatible(current, &new_props));
    }

    #[test]
    fn test_not_backward_compatible_tighten_max_count() {
        let mut ev = ShapeEvolver::new();
        let mut init_props = HashMap::new();
        init_props.insert("ex:p".to_string(), prop(0, Some(5)));
        ev.init(init_props, None);
        let current = ev.current().unwrap();
        let mut new_props = HashMap::new();
        new_props.insert("ex:p".to_string(), prop(0, Some(2)));
        assert!(!ShapeEvolver::is_backward_compatible(current, &new_props));
    }

    // ── error display ─────────────────────────────────────────────────────────

    #[test]
    fn test_error_display_no_base_shape() {
        let e = EvolverError::NoBaseShape;
        assert!(e.to_string().contains("no base shape"));
    }

    #[test]
    fn test_error_display_version_not_found() {
        let e = EvolverError::VersionNotFound(42);
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn test_error_display_invalid_evolution() {
        let e = EvolverError::InvalidEvolution("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn test_error_display_empty_shape() {
        let e = EvolverError::EmptyShape;
        assert!(!e.to_string().is_empty());
    }

    // ── version numbering ─────────────────────────────────────────────────────

    #[test]
    fn test_version_numbers_increment_correctly() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        let v2 = ev.evolve(single("ex:p", 0), None).unwrap();
        let v3 = ev.evolve(single("ex:p", 0), None).unwrap();
        assert_eq!(v2, 2);
        assert_eq!(v3, 3);
    }

    // ── changes_from_prev correctly populated ──────────────────────────────────

    #[test]
    fn test_changes_from_prev_populated_on_evolve() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:a", 0), None);
        let mut new_props = single("ex:a", 0);
        new_props.insert("ex:b".to_string(), prop(0, None));
        ev.evolve(new_props, None).unwrap();
        let v2 = ev.current().unwrap();
        assert!(!v2.changes_from_prev.is_empty());
    }

    // ── default ───────────────────────────────────────────────────────────────

    #[test]
    fn test_default_evolver_is_empty() {
        let ev = ShapeEvolver::default();
        assert_eq!(ev.version_count(), 0);
    }

    // ── additional coverage ────────────────────────────────────────────────────

    #[test]
    fn test_evolve_with_datatype_added() {
        let mut ev = ShapeEvolver::new();
        let mut init_props = HashMap::new();
        init_props.insert("ex:p".to_string(), prop(0, None)); // no datatype
        ev.init(init_props, None);
        let mut new_props = HashMap::new();
        new_props.insert("ex:p".to_string(), typed_prop(0, "xsd:string"));
        ev.evolve(new_props, None).unwrap();
        let v2 = ev.current().unwrap();
        assert!(v2
            .changes_from_prev
            .iter()
            .any(|c| matches!(c, ShapeChange::TypeChanged { .. })));
    }

    #[test]
    fn test_diff_empty_to_empty_no_changes() {
        let old_version = ShapeVersion {
            version: 1,
            timestamp_ms: 0,
            properties: HashMap::new(),
            target_class: None,
            changes_from_prev: Vec::new(),
        };
        let new_props = HashMap::new();
        let changes = ShapeEvolver::diff(&old_version, &new_props);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_current_after_evolve_reflects_new_properties() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:a", 0), None);
        let mut new_props = HashMap::new();
        new_props.insert("ex:b".to_string(), prop(1, Some(5)));
        ev.evolve(new_props, None).unwrap();
        let current = ev.current().unwrap();
        assert!(current.properties.contains_key("ex:b"));
        assert!(!current.properties.contains_key("ex:a"));
    }

    #[test]
    fn test_version_numbers_survive_rollback() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        ev.evolve(single("ex:p", 1), None).unwrap(); // v2
        ev.evolve(single("ex:p", 2), None).unwrap(); // v3
        let rolled = ev.rollback(1).unwrap(); // v4
        assert_eq!(rolled, 4);
        assert_eq!(ev.version_count(), 4);
    }

    #[test]
    fn test_backward_compatible_identical_shape() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 1), Some("ex:C".into()));
        let current = ev.current().unwrap();
        // Identical props → always compatible.
        assert!(ShapeEvolver::is_backward_compatible(
            current,
            &single("ex:p", 1)
        ));
    }

    #[test]
    fn test_init_without_target_class() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        assert!(ev.current().unwrap().target_class.is_none());
    }

    #[test]
    fn test_history_first_entry_has_no_changes() {
        let mut ev = ShapeEvolver::new();
        ev.init(single("ex:p", 0), None);
        assert!(ev.history()[0].changes_from_prev.is_empty());
    }

    #[test]
    fn test_relax_constraint_detected_when_max_increased() {
        let mut ev = ShapeEvolver::new();
        let mut init_props = HashMap::new();
        init_props.insert("ex:p".to_string(), prop(0, Some(2)));
        ev.init(init_props, None);
        let mut new_props = HashMap::new();
        new_props.insert("ex:p".to_string(), prop(0, Some(5)));
        ev.evolve(new_props, None).unwrap();
        let v2 = ev.current().unwrap();
        assert!(v2
            .changes_from_prev
            .iter()
            .any(|c| matches!(c, ShapeChange::RelaxedConstraint { .. })));
    }
}
